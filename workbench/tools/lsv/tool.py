from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.figure import Figure

from workbench import theme
from workbench.core.data_io import read_lsv_data
from workbench.core.tool_base import ParameterSpec, Tool, ToolAction, ToolResult
from workbench.tools.lsv.logic_tafel import (
    calculate_RCL_utilization,
    get_overpotential_at_current,
    iterative_tafel_fit,
)

R_GAS = 8.31446261815324  # J mol^-1 K^-1
FARADAY = 96485.33212  # C mol^-1
REFERENCE_TEMPERATURE_K = 298.15

# Practical reference-electrode potentials versus SHE at 25 °C.  The filling
# solution is part of the electrode definition and must not be omitted.
REFERENCE_OPTIONS = {
    "RHE": 0.0,
    "Ag/AgCl (sat. KCl, 25 °C)": 0.197,
    "SCE (sat. KCl, 25 °C)": 0.2412,
    "Hg/HgO (1 M KOH, 25 °C)": 0.1053,
    "Custom E_ref vs SHE": None,
}
_KEYS = ["E_rev", "eta_kin", "eta_ohm", "eta_rcl", "eta_res"]


def convert_reference_to_rhe(
    potential,
    reference: str,
    ph: float,
    temperature_k: float,
    custom_e_ref_vs_she: float = 0.0,
):
    """Convert a potential measured versus ``reference`` to the RHE scale.

    E_RHE = E_measured + E_ref(vs SHE) + (2.303 R T / F) pH

    Preset E_ref values are practical 25 °C values.  For another reference
    temperature or filling solution, callers should select ``Custom E_ref vs
    SHE`` and supply a calibrated value at the measurement temperature.
    """
    if reference == "RHE":
        e_ref_vs_she = shift = 0.0
    else:
        preset = REFERENCE_OPTIONS[reference]
        e_ref_vs_she = custom_e_ref_vs_she if preset is None else preset
        shift = e_ref_vs_she + (2.303 * R_GAS * temperature_k / FARADAY) * ph
    return np.asarray(potential, dtype=float) + shift, shift, e_ref_vs_she


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
                          group="Electrolyte & reference"),
            ParameterSpec("reference", "Reference", kind="choice",
                          choices=list(REFERENCE_OPTIONS.keys()), default="RHE",
                          group="Electrolyte & reference"),
            ParameterSpec(
                "custom_e_ref_vs_she", "Custom E_ref vs SHE", kind="float",
                default=0.0, step=0.001, decimals=5, suffix="V",
                tooltip="Calibrated reference-electrode potential versus SHE at "
                        "the measurement temperature.",
                group="Electrolyte & reference",
                show_if=("reference", "Custom E_ref vs SHE"),
            ),
            ParameterSpec("ph", "pH", kind="float", default=0.0, minimum=0.0,
                          maximum=14.0, step=0.1, decimals=2,
                          tooltip="pH of the electrolyte at the measurement temperature.",
                          group="Electrolyte & reference"),
            ParameterSpec("temperature", "Temperature", kind="float", default=378.0,
                          minimum=0.0, step=1.0, decimals=2, suffix="K",
                          tooltip="Measurement temperature used for E_rev and the "
                                  "RHE Nernst term 2.303RT/F.",
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
            ParameterSpec("rcl", "R_CL", kind="float", default=0.0, minimum=0.0,
                          step=0.001, decimals=5, suffix="Ω·cm²", group="Resistances"),
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
        """Plot each LSV file's original curve and, for multiple files, the
        averaged curve the fit uses — a quick visual data check before fitting
        (mirrors the EIS averaged-spectrum overlay)."""
        if not files:
            raise ValueError("Add at least one LSV file to plot.")
        reaction_param = str(params.get("reaction", "Auto")).strip().lower()
        paths = [str(f) for f in files]

        # Resolve the reaction and the averaged curve exactly as the fit does.
        combined, err = read_lsv_data(paths, reaction=reaction_param)
        if err:
            raise ValueError(err)
        V_avg, i_avg, info = combined
        area_cm2 = float(params.get("electrode_area", 0.0))
        i_avg, area_warning = _as_current_density(i_avg, info, area_cm2)
        resolved = info.get("reaction", "oer")
        is_her = resolved == "her"
        averaged = bool(info.get("averaged", False))

        warnings: list[str] = []
        if info.get("reaction_auto"):
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
            iff, _ = _as_current_density(iff, single_info, area_cm2)
            per_file.append((Path(p).name, Vf, iff))
            raw_frames.append(pd.DataFrame({
                "File": Path(p).name,
                "Potential (V)": Vf,
                "Current Density (A/cm²)": iff,
            }))
        if not per_file:
            raise ValueError("No LSV files could be read for plotting.")

        figures = {"LSV originals": _plot_originals(per_file, V_avg, i_avg,
                                                    is_her, averaged)}
        tables = {"LSV data (per file)": pd.concat(raw_frames, ignore_index=True)}
        if averaged:
            tables["Averaged LSV"] = pd.DataFrame({
                "Potential (V)": V_avg,
                "Current Density (A/cm²)": i_avg,
            })
        scalars = {
            "Files plotted": len(per_file),
            "Reaction": resolved.upper(),
            "Averaged": "yes" if averaged else "no (single file)",
        }
        return ToolResult(figures=figures, tables=tables, scalars=scalars,
                          warnings=warnings)

    def run(self, files: list[Path], params: dict, extra=None) -> ToolResult:
        i_lower = float(params["i_lower"])
        i_upper = float(params["i_upper"])
        if i_lower >= i_upper:
            raise ValueError("Tafel lower limit must be less than the upper limit.")

        reaction_param = str(params.get("reaction", "Auto")).strip().lower()
        data, err = read_lsv_data([str(f) for f in files], reaction=reaction_param)
        if err:
            raise ValueError(err)
        V_raw, i_raw, info = data

        T = float(params["temperature"])
        ph = float(params["ph"])
        ref = params["reference"]
        hfr = float(params["hfr"])
        rcl = float(params["rcl"])
        target_i = float(params["target_i"])
        area_cm2 = float(params.get("electrode_area", 0.0))
        i_raw, area_warning = _as_current_density(i_raw, info, area_cm2)

        reaction = info.get("reaction", "oer")
        is_her = reaction == "her"
        sign = -1.0 if is_her else 1.0

        # HER reversible potential is 0 V vs RHE; OER uses the Nernst form.
        E_rev = 0.0 if is_her else (1.2291 - 0.0008456 * (T - 298.15))
        V, rhe_shift, e_ref_vs_she = convert_reference_to_rhe(
            V_raw,
            ref,
            ph,
            T,
            custom_e_ref_vs_she=float(params.get("custom_e_ref_vs_she", 0.0)),
        )

        # Decouple in OER-equivalent magnitudes: |i| and the positive overpotential.
        # For OER (sign=+1) this is identical to passing (V, i, E_rev) directly.
        i_mag = sign * i_raw
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

        result, warn = iterative_tafel_fit(eta_total_mag, i_mag, 0.0, hfr, rcl, i_lower, i_upper)
        if result is None:
            raise ValueError(warn)
        fit, eta_ohm_all, eta_rcl_all, _ = result
        b_kin, i0 = fit["b_kin"], fit["i0"]

        eta_kin = b_kin * np.log10(i_mag / i0)
        eta_res = np.maximum(eta_total_mag - (eta_kin + eta_ohm_all + eta_rcl_all), 0)
        utilization = calculate_RCL_utilization(i_mag, rcl, b_kin)
        utilization_order = np.argsort(i_mag)
        utilization_current = i_mag[utilization_order]
        utilization_sorted = utilization[utilization_order]

        # Re-apply the sign for native display: HER curves descend below E_rev = 0.
        y1 = np.full_like(i_raw, E_rev)
        y2 = y1 + sign * eta_kin
        y3 = y2 + sign * eta_ohm_all
        y4 = y3 + sign * eta_rcl_all
        y5 = y4 + sign * eta_res
        y6 = V

        curves_df = pd.DataFrame({
            "Current Density (A/cm²)": i_raw,
            "E_rev": y1, "E_rev + η_kin": y2, "+ η_ohm": y3,
            "+ η_RCL": y4, "+ η_res": y5, "Original LSV (RHE)": y6,
        })
        comp_df = pd.DataFrame({
            "Current Density (A/cm²)": i_raw,
            "η_kin (V)": sign * eta_kin, "η_ohm (V)": sign * eta_ohm_all,
            "η_RCL (V)": sign * eta_rcl_all, "η_res (V)": sign * eta_res,
        })
        utilization_df = pd.DataFrame({
            "Current Density Magnitude (A/cm²)": utilization_current,
            "Catalyst Utilization U_J (-)": utilization_sorted,
        })
        warnings: list[str] = []
        if info.get("reaction_auto") and is_her:
            warnings.append("Auto-detected HER from the negative current sign.")
        if area_warning:
            warnings.append(area_warning)
        if warn:
            warnings.append(warn)
        if info.get("warning") and not area_warning:
            warnings.append(info["warning"])
        if (
            ref not in {"RHE", "Custom E_ref vs SHE"}
            and abs(T - REFERENCE_TEMPERATURE_K) > 0.5
        ):
            warnings.append(
                f"{ref} has a preset E_ref vs SHE specified at 298.15 K, "
                f"but the measurement temperature is {T:g} K. The RHE pH term "
                "uses the selected temperature; for rigorous conversion, select "
                "Custom E_ref vs SHE and enter a calibrated reference potential "
                "at the measurement temperature."
            )

        d = fit["candidate_details"]
        fit_i_min, fit_i_max = d["current_range"]
        info_df = pd.DataFrame({
            "Parameter": ["Tafel slope (V/dec)", "Exchange current density (A/cm²)",
                          "Intercept (V)", "R²", "N points",
                          "Fit current lower (A/cm²)", "Fit current upper (A/cm²)",
                          "E_rev (V)", "E_ref vs SHE (V)",
                          "RHE conversion shift (V)"],
            "Value": [b_kin, i0, fit["intercept"], fit["r_squared"],
                      fit["n_points"], fit_i_min, fit_i_max, E_rev,
                      e_ref_vs_she, rhe_shift],
        })
        diag = (
            f"Reaction: {reaction.upper()}"
            f"{' (auto-detected)' if info.get('reaction_auto') else ''}\n"
            f"Reference: {ref}\n"
            f"  E_ref vs SHE: {e_ref_vs_she:.5f} V\n"
            f"  RHE conversion shift: {rhe_shift:.5f} V\n"
            f"Tafel fit (iterative)\n"
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
            "Decoupled curves": _plot_curves(i_raw, [y1, y2, y3, y4, y5, y6],
                                             x_log=x_log,
                                             is_her=is_her,
                                             swap_axes=bool(params.get("swap_axes", False))),
            "Overpotential breakdown": _plot_breakdown(
                target_i, i_mag, E_rev + eta_total_mag, E_rev, hfr, rcl, fit,
                is_her=is_her),
            "Catalyst utilization": _plot_utilization(
                utilization_current, utilization_sorted, x_log=x_log,
                is_her=is_her),
        }
        scalars = {
            "b_kin (V/dec)": b_kin, "i0 (A/cm²)": i0,
            "Intercept (V)": fit["intercept"], "R²": fit["r_squared"],
            "N points": fit["n_points"], "E_rev (V)": E_rev,
            "E_ref vs SHE (V)": e_ref_vs_she,
            "RHE conversion shift (V)": rhe_shift,
            "Fit current range (A/cm²)": f"{fit_i_min:.4g}–{fit_i_max:.4g}",
        }
        return ToolResult(
            figures=figures,
            tables={"Curves (stacked)": curves_df,
                    "Overpotential components": comp_df,
                    "Catalyst utilization": utilization_df,
                    "Fitting info": info_df},
            scalars=scalars,
            text_blocks={"Diagnostics": diag},
            warnings=warnings,
        )


def _plot_originals(per_file, V_avg, i_avg, is_her=False, averaged=False) -> Figure:
    """Overlay each file's LSV curve; bold the averaged curve when present."""
    fig = Figure(figsize=(7.5, 5.5))
    ax = fig.add_subplot(111)
    colors = cm.viridis(np.linspace(0, 0.85, max(len(per_file), 1)))
    for color, (name, Vf, iff) in zip(colors, per_file):
        ax.plot(Vf, iff * 1e3, linewidth=1.1, alpha=0.65, color=color, label=name)
    if averaged:
        ax.plot(V_avg, i_avg * 1e3, linewidth=2.4, color=theme.TEXT,
                label="Average", zorder=5)
    ax.set_xlabel(r"Potential  $E$  (V)")
    ax.set_ylabel(r"Current density  $i$  (mA cm$^{-2}$)")
    title = "LSV originals + average" if averaged else "LSV original"
    ax.set_title(f"{title}  (HER)" if is_her else title)
    ax.legend(fontsize=8, ncol=2, loc="best")
    fig.tight_layout()
    return fig


def _plot_curves(i_data, ys, x_log=False, is_her=False, swap_axes=False) -> Figure:
    y6 = ys[-1]
    colors = [theme.OVERPOT_COLORS[key] for key in _KEYS]
    L = theme.OVERPOT_LABELS
    labels = [L["E_rev"], f"+ {L['eta_kin']}", f"+ {L['eta_ohm']}",
              f"+ {L['eta_rcl']}", f"+ {L['eta_res']}"]

    fig = Figure(figsize=(7.5, 5.5))
    ax = fig.add_subplot(111)
    # A log axis cannot show negative (HER) current, so use |i| in log mode.
    current = np.abs(i_data) if x_log else i_data
    current_label = (
        r"Current density  $|i|$  (A cm$^{-2}$)"
        if x_log else r"Current density  $i$  (A cm$^{-2}$)"
    )
    potential_label = r"Potential  $E$  (V vs RHE)"

    if x_log:
        if swap_axes:
            ax.set_yscale("log")
        else:
            ax.set_xscale("log")

    fill = ax.fill_betweenx if swap_axes else ax.fill_between
    for lower, upper, color in zip(ys, ys[1:5], colors[1:]):
        fill(current, lower, upper, color=color, alpha=0.18, linewidth=0)

    def coordinates(potential):
        return (potential, current) if swap_axes else (current, potential)

    for index, (potential, color, label) in enumerate(zip(ys[:5], colors, labels)):
        style = {"linestyle": (0, (4, 3)), "linewidth": 1.4} if index == 0 else {
            "linewidth": 1.6
        }
        ax.plot(*coordinates(potential), color=color, label=label, **style)
    ax.plot(*coordinates(y6), color=theme.TEXT, marker="o", linestyle="none",
            markersize=3.2, markerfacecolor="white", markeredgewidth=1.0,
            label="LSV")

    if swap_axes:
        ax.set_xlabel(potential_label)
        ax.set_ylabel(current_label)
    else:
        ax.set_xlabel(current_label)
        ax.set_ylabel(potential_label)
    title = "LSV decoupled into cumulative contributions"
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


def _plot_breakdown(target_i, i_data, V, E_rev, hfr, rcl, fit, is_her=False) -> Figure:
    comps = get_overpotential_at_current(target_i, i_data, V, E_rev, hfr, rcl, fit)
    values = [max(float(comps[k]), 0.0) for k in _KEYS]
    total = max(sum(values), 1e-9)

    fig = Figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    cumulative = 0.0
    for key, val in zip(_KEYS, values):
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
    ax.set_xlabel(r"Overpotential  $|\eta|$  (V)" if is_her
                  else r"Potential  $E$  (V vs RHE)")
    ax.set_title(f"Overpotential breakdown  ·  i = {target_i:.4g} A cm$^{{-2}}$"
                 + ("  (HER)" if is_her else ""),
                 loc="left", fontweight="bold")
    ax.set_xlim(0, total * 1.15)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.4)
    fig.tight_layout()
    return fig
