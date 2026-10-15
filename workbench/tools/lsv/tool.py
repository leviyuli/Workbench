from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.figure import Figure

from workbench import theme
from workbench.core.data_io import read_lsv_data
from workbench.core.potential import (
    FARADAY,
    REFERENCE_OPTIONS,
    REFERENCE_TEMPERATURE_K,
    R_GAS,
    PotentialCorrection,
    convert_reference_to_rhe,
    potential_parameter_schema,
)
from workbench.core.tool_base import ParameterSpec, Tool, ToolAction, ToolResult
from workbench.tools.lsv.logic_tafel import (
    calculate_RCL_utilization,
    get_overpotential_at_current,
    iterative_tafel_fit,
)

_KEYS = ["E_rev", "eta_kin", "eta_ohm", "eta_rcl", "eta_res"]


def _as_current_density(current, info: dict, area_cm2: float):
    """Normalize an explicitly absolute-current LSV series to A/cm²."""
    values = np.asarray(current, dtype=float)
    if info.get("is_density", True):
        return values, None
    if not np.isfinite(area_cm2) or area_cm2 <= 0:
        column = info.get("i_col", "current")
        raise ValueError(
            f"LSV column '{column}' is absolute current. Enter a positive "
            "electrode area so it can be normalized to A/cm²."
        )
    return values / area_cm2, (
        f"Absolute current was divided by the {area_cm2:g} cm² electrode area."
    )


class LsvTool(Tool):
    id = "lsv"
    name = "LSV Decoupling"
    subtitle = "Decompose polarization into kinetic, ohmic, catalyst-layer and residual overpotentials."
    needs_files = True
    multiple_files = True
    accepts_directory = False
    accepted_file_types = (".csv", ".txt", ".xlsx", ".xls")
    run_button_label = "Fit & Plot"

    def parameter_schema(self) -> list[ParameterSpec]:
        return [
            ParameterSpec("reaction", "Reaction", kind="choice",
                          choices=["Auto", "OER", "HER"], default="Auto",
                          tooltip="OER expects positive current, HER negative. "
                                  "Auto infers it from the current sign.",
                          group="Electrolyte & reference",
                          show_if=("electrode_setup", "Three-electrode")),
            *potential_parameter_schema(
                allow_raw=False, default_target="RHE", include_temperature=False,
            ),
            ParameterSpec("temperature", "Temperature", kind="float", default=378.0,
                          minimum=0.01, step=1.0, decimals=2, suffix="K",
                          tooltip="Measurement temperature used for the reversible "
                                  "water-splitting voltage and, in three-electrode "
                                  "mode, the RHE Nernst term 2.303RT/F.",
                          group="Electrolyte & reference"),
            ParameterSpec(
                "electrode_area", "Electrode area", kind="float", default=0.0,
                minimum=0.0, step=0.01, decimals=4, suffix="cm²",
                tooltip="Required only when the imported column is absolute current; "
                        "leave at 0 for data already reported in A/cm².",
                group="Electrolyte & reference",
            ),
            ParameterSpec("hfr", "HFR", kind="float", default=0.01, minimum=0.0,
                          step=0.001, decimals=5, suffix="Ω·cm²", group="Resistances"),
            ParameterSpec(
                "include_rcl", "Include RCL analysis", kind="bool", default=True,
                tooltip="Include catalyst-layer overpotential and utilization. "
                        "Turn off to consider only kinetic, ohmic and residual "
                        "overpotentials.",
                group="Resistances",
            ),
            ParameterSpec("rcl", "R_CL", kind="float", default=0.0, minimum=0.0,
                          step=0.001, decimals=5, suffix="Ω·cm²", group="Resistances",
                          show_if=("include_rcl", True)),
            ParameterSpec("i_lower", "Tafel lower", kind="float", default=0.02,
                          minimum=0.0, step=0.01, decimals=4, suffix="A/cm²",
                          group="Tafel fit range"),
            ParameterSpec("i_upper", "Tafel upper", kind="float", default=0.8,
                          minimum=0.0, step=0.01, decimals=4, suffix="A/cm²",
                          group="Tafel fit range"),
            ParameterSpec("target_i", "Breakdown target", kind="float", default=1.0,
                          minimum=1e-12, step=0.1, decimals=4, suffix="A/cm²",
                          tooltip="Current density at which to show the overpotential breakdown; "
                                  "must lie inside the measured current range.",
                          group="Breakdown"),
            ParameterSpec("x_log", "Log current axis", kind="bool", default=False,
                          tooltip="Use a logarithmic current axis for the decoupled "
                                  "curves and catalyst-utilization plots.",
                          group="Breakdown"),
            ParameterSpec(
                "swap_axes", "Exchange X/Y axes", kind="bool", default=False,
                tooltip="Show potential on the X axis and current density on the "
                        "Y axis in the decoupled-curves plot.",
                group="Breakdown",
            ),
        ]

    def actions(self) -> list[ToolAction]:
        return [ToolAction(label="Show plot", method="show_plot")]

    def show_plot(self, files: list[Path], params: dict, extra=None) -> ToolResult:
        """Plot each LSV curve and its average on the selected potential scale."""
        if not files:
            raise ValueError("Add at least one LSV file to plot.")
        correction = PotentialCorrection.from_params(
            {**params, "target_scale": "RHE"}, default_target="RHE",
        )
        is_cell = correction.electrode_setup == "Two-electrode"
        potential_label = correction.axis_label()
        reaction_param = (
            "auto" if is_cell else str(params.get("reaction", "Auto")).strip().lower()
        )
        paths = [str(f) for f in files]

        # Resolve the reaction and the averaged curve exactly as the fit does.
        combined, err = read_lsv_data(paths, reaction=reaction_param)
        if err:
            raise ValueError(err)
        V_avg, i_avg, info = combined
        V_avg = correction.apply(V_avg)
        area_cm2 = float(params.get("electrode_area", 0.0))
        i_avg, area_warning = _as_current_density(i_avg, info, area_cm2)
        resolved = info.get("reaction", "oer")
        is_her = resolved == "her"
        averaged = bool(info.get("averaged", False))

        warnings = list(correction.warnings)
        if info.get("reaction_auto") and not is_cell:
            warnings.append(f"Auto-detected {resolved.upper()} from the current sign.")
        if area_warning:
            warnings.append(area_warning)

        # Per-file curves, read with the resolved reaction so signs are consistent.
        per_file: list[tuple[str, np.ndarray, np.ndarray]] = []
        raw_frames: list[pd.DataFrame] = []
        for p in paths:
            single, serr = read_lsv_data([p], reaction=resolved)
            if serr:
                warnings.append(f"{Path(p).name}: {serr}")
                continue
            Vf, iff, single_info = single
            Vf = correction.apply(Vf)
            iff, _ = _as_current_density(iff, single_info, area_cm2)
            per_file.append((Path(p).name, Vf, iff))
            raw_frames.append(pd.DataFrame({
                "File": Path(p).name,
                potential_label: Vf,
                "Current Density (A/cm²)": iff,
            }))
        if not per_file:
            raise ValueError("No LSV files could be read for plotting.")

        figures = {"LSV originals": _plot_originals(per_file, V_avg, i_avg,
                                                    is_her and not is_cell, averaged,
                                                    potential_label=potential_label)}
        tables = {"LSV data (per file)": pd.concat(raw_frames, ignore_index=True)}
        if averaged:
            tables["Averaged LSV"] = pd.DataFrame({
                potential_label: V_avg,
                "Current Density (A/cm²)": i_avg,
            })
        tables["Potential settings"] = pd.DataFrame(
            correction.metadata().items(), columns=["Parameter", "Value"],
        )
        scalars = {
            "Files plotted": len(per_file),
            "Reaction": "Full cell" if is_cell else resolved.upper(),
            "Averaged": "yes" if averaged else "no (single file)",
            **correction.metadata(),
        }
        return ToolResult(figures=figures, tables=tables, scalars=scalars,
                          text_blocks={"Potential correction": correction.describe()},
                          warnings=warnings)

    def run(self, files: list[Path], params: dict, extra=None) -> ToolResult:
        correction = PotentialCorrection.from_params(
            {**params, "target_scale": "RHE"}, default_target="RHE",
        )
        is_cell = correction.electrode_setup == "Two-electrode"
        i_lower = float(params["i_lower"])
        i_upper = float(params["i_upper"])
        if i_lower >= i_upper:
            raise ValueError("Tafel lower limit must be less than the upper limit.")

        reaction_param = (
            "auto" if is_cell else str(params.get("reaction", "Auto")).strip().lower()
        )
        data, err = read_lsv_data([str(f) for f in files], reaction=reaction_param)
        if err:
            raise ValueError(err)
        V_raw, i_raw, info = data

        T = float(params["temperature"])
        if not np.isfinite(T) or T <= 0:
            raise ValueError("Temperature must be a finite value greater than zero kelvin.")
        ref = correction.reference
        hfr = float(params["hfr"])
        include_rcl = bool(params.get("include_rcl", True))
        rcl = float(params["rcl"]) if include_rcl else 0.0
        target_i = float(params["target_i"])
        area_cm2 = float(params.get("electrode_area", 0.0))
        i_raw, area_warning = _as_current_density(i_raw, info, area_cm2)

        reaction = info.get("reaction", "oer")
        is_her = reaction == "her"
        # The full water-splitting cell has the same reversible voltage magnitude
        # as OER vs RHE; the HER half-cell baseline remains 0 V vs RHE.
        water_reversible = 1.2291 - 0.0008456 * (T - 298.15)
        if is_cell:
            # Cell-voltage polarity is independent of the current sign.
            sign = -1.0 if np.median(V_raw) < 0 else 1.0
            E_rev = sign * water_reversible
        else:
            sign = -1.0 if is_her else 1.0
            E_rev = 0.0 if is_her else water_reversible
        V = correction.apply(V_raw)
        rhe_shift = correction.shift_v
        e_ref_vs_she = correction.e_ref_vs_she

        # Fit current magnitudes and positive losses, then restore voltage polarity.
        i_mag = np.abs(i_raw)
        eta_total_mag = sign * (V - E_rev)
        if not np.isfinite(target_i) or target_i <= 0:
            raise ValueError("Breakdown target current must be greater than zero.")
        measured_min = float(np.min(i_mag))
        measured_max = float(np.max(i_mag))
        if target_i < measured_min or target_i > measured_max:
            raise ValueError(
                f"Breakdown target {target_i:g} A/cm² is outside the measured "
                f"range {measured_min:g}–{measured_max:g} A/cm²."
            )

        result, warn = iterative_tafel_fit(
            eta_total_mag, i_mag, 0.0, hfr, rcl, i_lower, i_upper,
            include_rcl=include_rcl,
        )
        if result is None:
            raise ValueError(warn)
        fit, eta_ohm_all, eta_rcl_all, _ = result
        b_kin, i0 = fit["b_kin"], fit["i0"]

        eta_kin = b_kin * np.log10(i_mag / i0)
        eta_res = np.maximum(eta_total_mag - (eta_kin + eta_ohm_all + eta_rcl_all), 0)

        # Preserve native voltage polarity and the imported current sign in exports.
        y1 = np.full_like(i_raw, E_rev)
        y2 = y1 + sign * eta_kin
        y3 = y2 + sign * eta_ohm_all
        y4 = y3 + sign * eta_rcl_all
        y5 = y4 + sign * eta_res
        y6 = V

        curves = {
            "Current Density (A/cm²)": i_raw,
            "E_rev": y1, "E_rev + η_kin": y2, "+ η_ohm": y3,
        }
        components = {
            "Current Density (A/cm²)": i_raw,
            "η_kin (V)": sign * eta_kin, "η_ohm (V)": sign * eta_ohm_all,
        }
        curve_values = [y1, y2, y3]
        if include_rcl:
            curves["+ η_RCL"] = y4
            components["η_RCL (V)"] = sign * eta_rcl_all
            curve_values.append(y4)
        original_label = "Original LSV (cell voltage)" if is_cell else "Original LSV (RHE)"
        curves.update({"+ η_res": y5, original_label: y6})
        components["η_res (V)"] = sign * eta_res
        curve_values.extend([y5, y6])
        warnings = list(correction.warnings)
        if info.get("reaction_auto") and is_her and not is_cell:
            warnings.append("Auto-detected HER from the negative current sign.")
        if area_warning:
            warnings.append(area_warning)
        if warn:
            warnings.append(warn)
        if info.get("warning") and not area_warning:
            warnings.append(info["warning"])
        d = fit["candidate_details"]
        fit_i_min, fit_i_max = d["current_range"]
        fitting_info = {
            "Tafel slope (V/dec)": b_kin, "Exchange current density (A/cm²)": i0,
            "Intercept (V)": fit["intercept"], "R²": fit["r_squared"],
            "N points": fit["n_points"],
            "Fit current lower (A/cm²)": fit_i_min,
            "Fit current upper (A/cm²)": fit_i_max, "E_rev (V)": E_rev,
        }
        reference_info = {} if is_cell else {
            "E_ref vs SHE (V)": e_ref_vs_she, "RHE conversion shift (V)": rhe_shift,
        }
        fit_scope = "Effective cell" if is_cell else "Working electrode"
        if is_cell:
            fitting_info["Fit scope"] = fit_scope
        fitting_info.update(reference_info)
        info_df = pd.DataFrame(fitting_info.items(), columns=["Parameter", "Value"])
        potential_diag = (
            "Configuration: Two-electrode (cell voltage)\n"
            f"Reversible cell voltage: {E_rev:.5f} V\n"
            "Fit scope: Effective cell contributions\n"
            if is_cell else
            f"Reaction: {reaction.upper()}"
            f"{' (auto-detected)' if info.get('reaction_auto') else ''}\n"
            f"Reference: {ref}\n"
            f"  E_ref vs SHE: {e_ref_vs_she:.5f} V\n"
            f"  RHE conversion shift: {rhe_shift:.5f} V\n"
        )
        diag = potential_diag + (
            f"RCL analysis: {'enabled' if include_rcl else 'disabled'}\n"
            f"Tafel fit ({'iterative' if include_rcl else 'ohmic-corrected'})\n"
            f"  b_kin: {b_kin:.4f} V/dec\n"
            f"  i0:    {i0:.4e} A/cm²\n"
            f"  Intercept: {fit['intercept']:.4f} V\n"
            f"  R²:    {fit['r_squared']:.4f}\n"
            f"  N in [{i_lower:.4g}, {i_upper:.4g}] A/cm²: {fit['n_points']}\n"
            f"Best 60 mV window: {d['window'][0]:.4f}–{d['window'][1]:.4f} V "
            f"(center {d['center']:.4f} V)\n"
            f"  Current range: {fit_i_min:.4g}–{fit_i_max:.4g} A/cm²\n"
            f"  Rel. SE(b): {d['rel_err_b'] * 100:.2f}%   "
            f"Rel. SE(i0): {d['rel_err_i0'] * 100:.2f}%"
        )

        x_log = bool(params.get("x_log", False))
        figures = {
            "Decoupled curves": _plot_curves(
                i_raw, curve_values, x_log=x_log, is_her=is_her,
                swap_axes=bool(params.get("swap_axes", False)),
                include_rcl=include_rcl, is_cell=is_cell),
            "Overpotential breakdown": _plot_breakdown(
                target_i, i_mag, sign * E_rev + eta_total_mag, sign * E_rev,
                hfr, rcl, fit, is_her=is_her,
                include_rcl=include_rcl, is_cell=is_cell),
        }
        tables = {
            "Curves (stacked)": pd.DataFrame(curves),
            "Overpotential components": pd.DataFrame(components),
        }
        if include_rcl:
            utilization = calculate_RCL_utilization(i_mag, rcl, b_kin)
            utilization_order = np.argsort(i_mag)
            utilization_current = i_mag[utilization_order]
            utilization_sorted = utilization[utilization_order]
            figures["Catalyst utilization"] = _plot_utilization(
                utilization_current, utilization_sorted, x_log=x_log,
                is_her=is_her)
            tables["Catalyst utilization"] = pd.DataFrame({
                "Current Density Magnitude (A/cm²)": utilization_current,
                "Catalyst Utilization U_J (-)": utilization_sorted,
            })
        tables["Fitting info"] = info_df
        potential_settings = correction.metadata()
        if is_cell:
            potential_settings["Temperature (K)"] = T
        tables["Potential settings"] = pd.DataFrame(
            potential_settings.items(), columns=["Parameter", "Value"],
        )
        scalars = {
            "b_kin (V/dec)": b_kin, "i0 (A/cm²)": i0,
            "Intercept (V)": fit["intercept"], "R²": fit["r_squared"],
            "N points": fit["n_points"], "E_rev (V)": E_rev,
            **reference_info,
            "Fit scope": fit_scope,
            "Fit current range (A/cm²)": f"{fit_i_min:.4g}–{fit_i_max:.4g}",
            "RCL analysis": "enabled" if include_rcl else "disabled",
            **potential_settings,
        }
        return ToolResult(
            figures=figures,
            tables=tables,
            scalars=scalars,
            text_blocks={"Diagnostics": diag},
            warnings=warnings,
        )


def _plot_originals(per_file, V_avg, i_avg, is_her=False, averaged=False,
                    potential_label="Potential (V)") -> Figure:
    """Overlay each file's LSV curve; bold the averaged curve when present."""
    fig = Figure(figsize=(7.5, 5.5))
    ax = fig.add_subplot(111)
    colors = cm.viridis(np.linspace(0, 0.85, max(len(per_file), 1)))
    for color, (name, Vf, iff) in zip(colors, per_file):
        ax.plot(Vf, iff * 1e3, linewidth=1.1, alpha=0.65, color=color, label=name)
    if averaged:
        ax.plot(V_avg, i_avg * 1e3, linewidth=2.4, color=theme.TEXT,
                label="Average", zorder=5)
    ax.set_xlabel(potential_label)
    ax.set_ylabel(r"Current density  $i$  (mA cm$^{-2}$)")
    title = "LSV originals + average" if averaged else "LSV original"
    ax.set_title(f"{title}  (HER)" if is_her else title)
    ax.legend(fontsize=8, ncol=2, loc="best")
    fig.tight_layout()
    return fig


def _plot_curves(i_data, ys, x_log=False, is_her=False, swap_axes=False,
                 include_rcl=True, is_cell=False) -> Figure:
    is_her = is_her and not is_cell
    original = ys[-1]
    stacked = ys[:-1]
    keys = [key for key in _KEYS if include_rcl or key != "eta_rcl"]
    colors = [theme.OVERPOT_COLORS[key] for key in keys]
    L = theme.OVERPOT_LABELS
    labels = [L[key] if key == "E_rev" else f"+ {L[key]}" for key in keys]

    fig = Figure(figsize=(7.5, 5.5))
    ax = fig.add_subplot(111)
    # A log axis cannot show negative (HER) current, so use |i| in log mode.
    current = np.abs(i_data) if x_log else i_data
    current_label = (
        r"Current density  $|i|$  (A cm$^{-2}$)"
        if x_log else r"Current density  $i$  (A cm$^{-2}$)"
    )
    potential_label = (r"Cell voltage  $V_{\mathrm{cell}}$  (V)" if is_cell
                       else r"Potential  $E$  (V vs RHE)")

    if x_log:
        if swap_axes:
            ax.set_yscale("log")
        else:
            ax.set_xscale("log")

    fill = ax.fill_betweenx if swap_axes else ax.fill_between
    for lower, upper, color in zip(stacked, stacked[1:], colors[1:]):
        fill(current, lower, upper, color=color, alpha=0.18, linewidth=0)

    def coordinates(potential):
        return (potential, current) if swap_axes else (current, potential)

    for index, (potential, color, label) in enumerate(zip(stacked, colors, labels)):
        style = {"linestyle": (0, (4, 3)), "linewidth": 1.4} if index == 0 else {
            "linewidth": 1.6
        }
        ax.plot(*coordinates(potential), color=color, label=label, **style)
    ax.plot(*coordinates(original), color=theme.TEXT, marker="o", linestyle="none",
            markersize=3.2, markerfacecolor="white", markeredgewidth=1.0,
            label="LSV")

    if swap_axes:
        ax.set_xlabel(potential_label)
        ax.set_ylabel(current_label)
    else:
        ax.set_xlabel(current_label)
        ax.set_ylabel(potential_label)
    title = ("Cell LSV decoupled into cumulative contributions" if is_cell
             else "LSV decoupled into cumulative contributions")
    ax.set_title(f"{title}  (HER)" if is_her else title)
    ax.legend(loc="upper left" if not is_her else "lower left", ncol=2, fontsize=9)
    fig.tight_layout()
    return fig


def _plot_utilization(i_data, utilization, x_log=False, is_her=False) -> Figure:
    """Plot the catalyst utilization used in the RCL overpotential model."""
    fig = Figure(figsize=(7.5, 5.5))
    ax = fig.add_subplot(111)
    ax.plot(i_data, utilization, color=theme.OVERPOT_COLORS["eta_rcl"],
            linewidth=2.2)
    if x_log:
        ax.set_xscale("log")
    current_symbol = r"$|i|$" if is_her else r"$i$"
    ax.set_xlabel(rf"Current density  {current_symbol}  (A cm$^{{-2}}$)")
    ax.set_ylabel(r"Catalyst utilization  $U_J$")
    ax.set_title("Catalyst-layer utilization vs current density")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return fig


def _plot_breakdown(target_i, i_data, V, E_rev, hfr, rcl, fit, is_her=False,
                    include_rcl=True, is_cell=False) -> Figure:
    is_her = is_her and not is_cell
    comps = get_overpotential_at_current(
        target_i, i_data, V, E_rev, hfr, rcl, fit, include_rcl=include_rcl,
    )
    keys = [key for key in _KEYS if include_rcl or key != "eta_rcl"]
    values = [max(float(comps[k]), 0.0) for k in keys]
    total = max(sum(values), 1e-9)

    fig = Figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    cumulative = 0.0
    for key, val in zip(keys, values):
        ax.barh(0, val, left=cumulative, height=0.55,
                color=theme.OVERPOT_COLORS[key], edgecolor="white", linewidth=1.2)
        if val > 0.06 * total:
            ax.text(cumulative + val / 2, 0,
                    f"{theme.OVERPOT_LABELS[key]}\n{val * 1000:.0f} mV",
                    ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        cumulative += val
    total_label = "|η| total" if is_her else "Total"
    ax.text(total, 0.42, f"{total_label}\n{total:.3f} V", ha="center", va="bottom",
            fontsize=9, color=theme.TEXT, fontweight="bold")
    ax.set_xlabel(r"Cell voltage  $|V_{\mathrm{cell}}|$  (V)" if is_cell
                  else r"Overpotential  $|\eta|$  (V)" if is_her
                  else r"Potential  $E$  (V vs RHE)")
    title = "Cell voltage breakdown" if is_cell else "Overpotential breakdown"
    ax.set_title(f"{title}  ·  i = {target_i:.4g} A cm$^{{-2}}$"
                 + ("  (HER)" if is_her else ""),
                 loc="left", fontweight="bold")
    ax.set_xlim(0, total * 1.15)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.4)
    fig.tight_layout()
    return fig
