from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from workbench import theme
from workbench.core.data_io import read_lsv_data
from workbench.core.tool_base import ParameterSpec, Tool, ToolResult
from workbench.tools.lsv.logic_tafel import get_overpotential_at_current, iterative_tafel_fit

REFERENCE_OPTIONS = {
    "RHE": 0.0,
    "Ag/AgCl (sat)": 0.197,
    "SCE (sat)": 0.242,
    "HgO (sat)": 0.098,
}
_KEYS = ["E_rev", "eta_kin", "eta_ohm", "eta_rcl", "eta_res"]


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
            ParameterSpec("reference", "Reference", kind="choice",
                          choices=list(REFERENCE_OPTIONS.keys()), default="RHE",
                          group="Electrolyte & reference"),
            ParameterSpec("ph", "pH", kind="float", default=0.0, minimum=0.0,
                          maximum=14.0, step=0.1, decimals=2, group="Electrolyte & reference"),
            ParameterSpec("temperature", "Temperature", kind="float", default=378.0,
                          minimum=0.0, step=1.0, decimals=2, suffix="K",
                          group="Electrolyte & reference"),
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
                          minimum=0.0, step=0.1, decimals=4, suffix="A/cm²",
                          tooltip="Current density at which to show the overpotential breakdown.",
                          group="Breakdown"),
            ParameterSpec("x_log", "Log current axis", kind="bool", default=False,
                          tooltip="Plot the decoupled curves with a logarithmic current axis.",
                          group="Breakdown"),
        ]

    def run(self, files: list[Path], params: dict, extra=None) -> ToolResult:
        i_lower = float(params["i_lower"])
        i_upper = float(params["i_upper"])
        if i_lower >= i_upper:
            raise ValueError("Tafel lower limit must be less than the upper limit.")

        data, err = read_lsv_data([str(f) for f in files])
        if err:
            raise ValueError(err)
        V_raw, i_raw, info = data

        T = float(params["temperature"])
        ph = float(params["ph"])
        ref = params["reference"]
        hfr = float(params["hfr"])
        rcl = float(params["rcl"])
        target_i = float(params["target_i"])

        E_rev = 1.2291 - 0.0008456 * (T - 298.15)
        offset = REFERENCE_OPTIONS[ref]
        V = V_raw if ref == "RHE" else (V_raw + 0.0592 * ph + offset)

        result, warn = iterative_tafel_fit(V, i_raw, E_rev, hfr, rcl, i_lower, i_upper)
        if result is None:
            raise ValueError(warn)
        fit, eta_ohm_all, eta_rcl_all, _ = result
        b_kin, i0 = fit["b_kin"], fit["i0"]

        eta_kin = b_kin * np.log10(i_raw / i0)
        eta_res = np.maximum((V - E_rev) - (eta_kin + eta_ohm_all + eta_rcl_all), 0)

        y1 = np.full_like(i_raw, E_rev)
        y2 = y1 + eta_kin
        y3 = y2 + eta_ohm_all
        y4 = y3 + eta_rcl_all
        y5 = y4 + eta_res
        y6 = V

        curves_df = pd.DataFrame({
            "Current Density (A/cm²)": i_raw,
            "E_rev": y1, "E_rev + η_kin": y2, "+ η_ohm": y3,
            "+ η_RCL": y4, "+ η_res": y5, "Original LSV (RHE)": y6,
        })
        comp_df = pd.DataFrame({
            "Current Density (A/cm²)": i_raw,
            "η_kin (V)": eta_kin, "η_ohm (V)": eta_ohm_all,
            "η_RCL (V)": eta_rcl_all, "η_res (V)": eta_res,
        })
        info_df = pd.DataFrame({
            "Parameter": ["Tafel slope (V/dec)", "Exchange current density (A/cm²)",
                          "Intercept (V)", "R²", "N points", "E_rev (V)"],
            "Value": [b_kin, i0, fit["intercept"], fit["r_squared"], fit["n_points"], E_rev],
        })

        warnings: list[str] = []
        if warn:
            warnings.append(warn)
        if info.get("warning"):
            warnings.append(info["warning"])

        d = fit["candidate_details"]
        diag = (
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
                                             x_log=bool(params.get("x_log", False))),
            "Overpotential breakdown": _plot_breakdown(
                target_i, i_raw, V, E_rev, hfr, rcl, fit),
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


def _plot_curves(i_data, ys, x_log=False) -> Figure:
    y1, y2, y3, y4, y5, y6 = ys
    c0 = theme.OVERPOT_COLORS["E_rev"]
    c1 = theme.OVERPOT_COLORS["eta_kin"]
    c2 = theme.OVERPOT_COLORS["eta_ohm"]
    c3 = theme.OVERPOT_COLORS["eta_rcl"]
    c4 = theme.OVERPOT_COLORS["eta_res"]
    L = theme.OVERPOT_LABELS

    fig = Figure(figsize=(7.5, 5.5))
    ax = fig.add_subplot(111)
    if x_log:
        ax.set_xscale("log")
    ax.fill_between(i_data, y1, y2, color=c1, alpha=0.18, linewidth=0)
    ax.fill_between(i_data, y2, y3, color=c2, alpha=0.18, linewidth=0)
    ax.fill_between(i_data, y3, y4, color=c3, alpha=0.18, linewidth=0)
    ax.fill_between(i_data, y4, y5, color=c4, alpha=0.18, linewidth=0)
    ax.plot(i_data, y1, color=c0, linestyle=(0, (4, 3)), linewidth=1.4, label=L["E_rev"])
    ax.plot(i_data, y2, color=c1, linewidth=1.6, label=f"+ {L['eta_kin']}")
    ax.plot(i_data, y3, color=c2, linewidth=1.6, label=f"+ {L['eta_ohm']}")
    ax.plot(i_data, y4, color=c3, linewidth=1.6, label=f"+ {L['eta_rcl']}")
    ax.plot(i_data, y5, color=c4, linewidth=1.6, label=f"+ {L['eta_res']}")
    ax.plot(i_data, y6, color=theme.TEXT, marker="o", linestyle="none",
            markersize=3.2, markerfacecolor="white", markeredgewidth=1.0, label="LSV")
    ax.set_xlabel(r"Current density  $i$  (A cm$^{-2}$)")
    ax.set_ylabel(r"Potential  $E$  (V)")
    ax.set_title("LSV decoupled into cumulative contributions")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    fig.tight_layout()
    return fig


def _plot_breakdown(target_i, i_data, V, E_rev, hfr, rcl, fit) -> Figure:
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
    ax.text(total, 0.42, f"Total\n{total:.3f} V", ha="center", va="bottom",
            fontsize=9, color=theme.TEXT, fontweight="bold")
    ax.set_xlabel(r"Potential  $E$  (V vs RHE)")
    ax.set_title(f"Overpotential breakdown  ·  i = {target_i:.4g} A cm$^{{-2}}$",
                 loc="left", fontweight="bold")
    ax.set_xlim(0, total * 1.15)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.4)
    fig.tight_layout()
    return fig
