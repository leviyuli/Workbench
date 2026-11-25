"""CCD numeric data in one integrated analysis workbook, with separate plots."""
from __future__ import annotations

import pandas as pd

from workbench.core.tool_base import Artifact, ToolResult, _slug


_SETTINGS = {
    "baseline_mode": ("Baseline selection", "", 1),
    "baseline_lower": ("Baseline start", "V", 1),
    "baseline_upper": ("Baseline end", "V", 1),
    "threshold_v": ("CCD excess voltage", "mV", 1000),
    "reference_hfr": ("Reference HFR", "Ω·cm²", 1),
    "bin_width_v": ("Voltage averaging interval", "mV", 1000),
    "smoothing_window": ("Smoothing window", "bins", 1),
    "persistence_v": ("CCD confirmation span", "mV", 1000),
    "plateau_window_v": ("Minimum flat-region span", "mV", 1000),
    "plateau_slope_fraction": ("Maximum flat-region slope", "% of baseline", 100),
    "plateau_spread_fraction": ("Maximum flat-region variation", "%", 100),
    "current_branch": ("Current sign", "", 1),
    "electrode_area_cm2": ("Electrode area", "cm²", 1),
}

_FORWARD_COLUMNS = {
    "File": "File",
    "Cycle": "Cycle",
    "Curve": "Curve",
    "Scan direction": "Scan direction",
    "Measured potential (V)": "Potential (V)",
    "HFR-corrected potential (V)": "iR-corrected potential (V)",
    "Smoothed current (A/cm²)": "Current density (A/cm²)",
    "Corrected baseline potential (V)": "iR-corrected baseline at scan current (V)",
    "Residual increase (mV)": "Excess voltage (mV)",
    "Baseline fit": "Baseline region",
    "CCD point": "CCD point",
    "Plateau": "Limiting-current region",
}

_PLOTS = {
    "LSV processing": ("Forward scans", "ccd_forward_scans.png"),
    "Current statistics": ("Current statistics", "ccd_currents.png"),
    "Cycle trends": ("Cycle trends", "ccd_cycle_trends.png"),
    "Residual onset": ("CCD onset check", "ccd_onset_check.png"),
}


def _settings_table(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, value in table[["Parameter", "Value"]].itertuples(index=False, name=None):
        label, unit, scale = _SETTINGS.get(
            key, (str(key).replace("_", " ").capitalize(), "", 1))
        rows.append({"Parameter": label,
                     "Value": value * scale if scale != 1 else value,
                     "Unit": unit})
    return pd.DataFrame(rows, columns=["Parameter", "Value", "Unit"])


def ccd_artifacts(result: ToolResult) -> list[Artifact]:
    """Keep all numeric analysis data in one workbook without duplicate files."""
    items = []
    sheets = {}
    if "Current results" in result.tables:
        sheets["Current results"] = result.tables["Current results"].copy()
    if "Batch statistics" in result.tables:
        sheets["Batch statistics"] = result.tables["Batch statistics"].copy()
    if "Cycle statistics" in result.tables:
        sheets["Cycle statistics"] = result.tables["Cycle statistics"].copy()
    if "Processed curves" in result.tables:
        sheets["Forward scans"] = result.tables["Processed curves"].rename(
            columns=_FORWARD_COLUMNS).copy()
    if "Input curves" in result.tables:
        sheets["Input curves"] = result.tables["Input curves"].copy()
    for name in ["Benchmark lines", "Baseline fits"]:
        if name in result.tables:
            sheets[name] = result.tables[name].copy()
    if "Analysis settings" in result.tables:
        sheets["Settings"] = _settings_table(result.tables["Analysis settings"])
    if sheets:
        items.append(Artifact("Analysis workbook (Excel)", "xlsx", "ccd_analysis.xlsx", sheets))
    for name in dict.fromkeys([*result.figures, *result.figure_images]):
        title, filename = _PLOTS.get(name, (name, f"ccd_{_slug(name)}.png"))
        if name in result.figures:
            items.append(Artifact(f"{title} (PNG)", "png", filename, result.figures[name]))
        else:
            items.append(Artifact(f"{title} (PNG)", "png", filename, None,
                                  serialized_payload=result.figure_images[name]))
    return items
