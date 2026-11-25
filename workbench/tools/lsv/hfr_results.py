"""One HFR fit figure and its complete, exportable scientific record."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from workbench import theme
from workbench.core.tool_base import ToolResult
from workbench.tools.lsv.hfr import HfrProfile
from workbench.tools.lsv.hfr_data import parse_hfr_pairs


def hfr_fit_result(profile: HfrProfile, currents, extra=None) -> ToolResult:
    inputs = extra or {}
    current = np.abs(np.asarray(currents, dtype=float).ravel())
    current = current[np.isfinite(current)]
    points = np.asarray(profile.points)
    measured_lower, measured_upper = profile.current_range
    lsv_lower = float(np.min(current)) if len(current) else measured_lower
    lsv_upper = float(np.max(current)) if len(current) else measured_upper
    lower, upper = min(lsv_lower, measured_lower), max(lsv_upper, measured_upper)
    grid = np.unique(np.r_[np.linspace(lower, upper, 501), points[:, 0], lsv_lower, lsv_upper])
    fitted = profile(grid)
    extended = (grid < measured_lower) | (grid > measured_upper)
    tables = {"HFR points": pd.DataFrame(points, columns=["|j| (A/cm²)", "HFR (Ω·cm²)"])}
    if "hfr_text" in inputs:
        data = parse_hfr_pairs(inputs["hfr_text"])
        tables["HFR source data"] = pd.DataFrame(
            data.rows, columns=["|j| (A/cm²)", *data.column_names],
        )
    metadata = {
        **profile.metadata(),
        "HFR source": inputs.get("hfr_source") or "Pasted measurements",
        "HFR measurements": len(points),
        "LSV current lower (A/cm²)": lsv_lower, "LSV current upper (A/cm²)": lsv_upper,
        "HFR extended below measurements": bool(lsv_lower < measured_lower),
        "HFR extended above measurements": bool(lsv_upper > measured_upper),
    }
    tables["HFR fit data"] = pd.DataFrame({
        "|j| (A/cm²)": grid, "Fitted HFR (Ω·cm²)": fitted,
        "Extended": extended,
    })
    tables["HFR settings"] = pd.DataFrame(metadata.items(), columns=["Parameter", "Value"])
    scalars = {key: metadata[key] for key in (
        "HFR fit method", "HFR smoothing strength", "HFR fit RMS (Ω·cm²)",
        "HFR extended below measurements", "HFR extended above measurements",
    )}
    return ToolResult(
        figures={"HFR fit": _plot_hfr(points, grid, fitted, measured_lower, measured_upper)},
        tables=tables, scalars=scalars, display_tables=False,
        text_blocks={"HFR fit":
            "Cubic smoothing spline fitted to the uploaded or pasted HFR measurements. "
            "Larger smoothing strength gives a smoother curve. Dashed portions use the "
            "fitted endpoint slopes to cover the LSV current range. Measurements, fitted "
            "curve and settings are included in the exported workbook."},
    )


def _plot_hfr(points, current, resistance, measured_lower, measured_upper) -> Figure:
    # A 6 x 5 cm plotting frame; space for labels preserves the frame on export.
    width, height = 7.8, 6.5
    fig = Figure(figsize=(width / 2.54, height / 2.54), facecolor="white")
    ax = fig.add_axes([1.4 / width, 1.2 / height, 6 / width, 5 / height])
    interior = (current >= measured_lower) & (current <= measured_upper)
    ax.plot(current[interior], resistance[interior], color=theme.OVERPOT_COLORS["eta_ohm"],
            lw=1, label="Cubic smoothing spline")
    for index, select in enumerate((current <= measured_lower, current >= measured_upper)):
        if np.count_nonzero(select) > 1:
            ax.plot(current[select], resistance[select], color=theme.OVERPOT_COLORS["eta_ohm"],
                    ls="--", lw=1, label="Extended to LSV range" if index == 0
                    or not np.any(current < measured_lower) else "_nolegend_")
    ax.plot(points[:, 0], points[:, 1], ls="none", marker="o", ms=3, mew=.5,
            mec=theme.TEXT, mfc="white", label="Measured HFR", zorder=4)
    ax.set_xlabel(r"$|j|$ / A cm$^{-2}$", fontsize=7)
    ax.set_ylabel(r"HFR / $\Omega$ cm$^2$", fontsize=7)
    ax.tick_params(axis="both", which="major", labelsize=7, length=3, width=.5, direction="out")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    for name, spine in ax.spines.items():
        spine.set_visible(name in {"left", "bottom"})
        spine.set_linewidth(.5)
    ax.grid(False)
    ax.legend(frameon=False, fontsize=7, loc="best", handlelength=1.5)
    return fig
