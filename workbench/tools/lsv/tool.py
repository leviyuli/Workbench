from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.figure import Figure

from workbench import theme
from workbench.core.data_io import read_lsv_data
from workbench.core.tool_base import ParameterSpec, Tool, ToolAction, ToolResult
from workbench.tools.lsv.logic_tafel import get_overpotential_at_current, iterative_tafel_fit

REFERENCE_OPTIONS = {
    "RHE": 0.0,
    "Ag/AgCl (sat)": 0.197,
    "SCE (sat)": 0.242,
    "HgO (sat)": 0.098,
}
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
                          group="Electrolyte & reference"),
            ParameterSpec("reference", "Reference", kind="choice",
                          choices=list(REFERENCE_OPTIONS.keys()), default="RHE",
                          group="Electrolyte & reference"),
            ParameterSpec("ph", "pH", kind="float", default=0.0, minimum=0.0,
                          maximum=14.0, step=0.1, decimals=2, group="Electrolyte & reference"),
            ParameterSpec("temperature", "Temperature", kind="float", default=378.0,
                          minimum=0.0, step=1.0, decimals=2, suffix="K",
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
                          tooltip="Plot the decoupled curves with a logarithmic current axis.",
                          group="Breakdown"),
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
        offset = REFERENCE_OPTIONS[ref]
        V = V_raw if ref == "RHE" else (V_raw + 0.0592 * ph + offset)

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
        info_df = pd.DataFrame({
            "Parameter": ["Tafel slope (V/dec)", "Exchange current density (A/cm²)",
                          "Intercept (V)", "R²", "N points", "E_rev (V)"],
            "Value": [b_kin, i0, fit["intercept"], fit["r_squared"], fit["n_points"], E_rev],
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

        d = fit["candidate_details"]
        diag = (
            f"Reaction: {reaction.upper()}"
            f"{' (auto-detected)' if info.get('reaction_auto') else ''}\n"
            f"Tafel fit (iterative)\n"
            f"  b_kin: {b_kin:.4f} V/dec\n"
            f"  i0:    {i0:.4e} A/cm²\n"
            f"  Intercept: {fit['intercept']:.4f} V\n"
            f"  R²:    {fit['r_squared']:.4f}\n"
            f"  N in [{i_lower:.4g}, {i_upper:.4g}] A/cm²: {fit['n_points']}\n"
            f"Best 60 mV window: {d['window'][0]:.4f}–{d['window'][1]:.4f} V "
            f"(center {d['center']:.4f} V)\n"
            f"  Rel. SE(b): {d['rel_err_b'] * 100:.2f}%   "
            f"Rel. SE(i0): {d['rel_err_i0'] * 100:.2f}%"
        )

        figures = {
            "Decoupled curves": _plot_curves(i_raw, [y1, y2, y3, y4, y5, y6],
                                             x_log=bool(params.get("x_log", False)),
                                             is_her=is_her),
            "Overpotential breakdown": _plot_breakdown(
                target_i, i_mag, E_rev + eta_total_mag, E_rev, hfr, rcl, fit,
                is_her=is_her),
        }
        scalars = {
            "b_kin (V/dec)": b_kin, "i0 (A/cm²)": i0,
            "Intercept (V)": fit["intercept"], "R²": fit["r_squared"],
            "N points": fit["n_points"], "E_rev (V)": E_rev,
        }
        return ToolResult(
            figures=figures,
            tables={"Curves (stacked)": curves_df,
                    "Overpotential components": comp_df,
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


def _plot_curves(i_data, ys, x_log=False, is_her=False) -> Figure:
    y1, y2, y3, y4, y5, y6 = ys
    c0 = theme.OVERPOT_COLORS["E_rev"]
    c1 = theme.OVERPOT_COLORS["eta_kin"]
    c2 = theme.OVERPOT_COLORS["eta_ohm"]
    c3 = theme.OVERPOT_COLORS["eta_rcl"]
    c4 = theme.OVERPOT_COLORS["eta_res"]
    L = theme.OVERPOT_LABELS

    fig = Figure(figsize=(7.5, 5.5))
    ax = fig.add_subplot(111)
    # A log axis cannot show negative (HER) current, so fall back to |i| there.
    if x_log:
        x = np.abs(i_data)
        ax.set_xscale("log")
        xlabel = r"Current density  $|i|$  (A cm$^{-2}$)"
    else:
        x = i_data
        xlabel = r"Current density  $i$  (A cm$^{-2}$)"
    ax.fill_between(x, y1, y2, color=c1, alpha=0.18, linewidth=0)
    ax.fill_between(x, y2, y3, color=c2, alpha=0.18, linewidth=0)
    ax.fill_between(x, y3, y4, color=c3, alpha=0.18, linewidth=0)
    ax.fill_between(x, y4, y5, color=c4, alpha=0.18, linewidth=0)
    ax.plot(x, y1, color=c0, linestyle=(0, (4, 3)), linewidth=1.4, label=L["E_rev"])
    ax.plot(x, y2, color=c1, linewidth=1.6, label=f"+ {L['eta_kin']}")
    ax.plot(x, y3, color=c2, linewidth=1.6, label=f"+ {L['eta_ohm']}")
    ax.plot(x, y4, color=c3, linewidth=1.6, label=f"+ {L['eta_rcl']}")
    ax.plot(x, y5, color=c4, linewidth=1.6, label=f"+ {L['eta_res']}")
    ax.plot(x, y6, color=theme.TEXT, marker="o", linestyle="none",
            markersize=3.2, markerfacecolor="white", markeredgewidth=1.0, label="LSV")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Potential  $E$  (V vs RHE)")
    title = "LSV decoupled into cumulative contributions"
    ax.set_title(f"{title}  (HER)" if is_her else title)
    ax.legend(loc="upper left" if not is_her else "lower left", ncol=2, fontsize=9)
    fig.tight_layout()
    return fig


def _plot_breakdown(target_i, i_data, V, E_rev, hfr, rcl, fit, is_her=False) -> Figure:
    comps = get_overpotential_at_current(target_i, i_data, V, E_rev, hfr, rcl, fit)
    values = [max(float(comps[k]), 0.0) for k in _KEYS]
    total = sum(values) if sum(values) > 0 else 1e-9

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
