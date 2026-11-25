"""Readable scientific results without chart-layout coordinates or duplicate data."""
import numpy as np
import pandas as pd

from workbench.core.export_tables import summary_table
from workbench.core.tool_base import Artifact, _slug


def _curve_data(tables, include_rcl, potential_shift):
    raw = tables.get("Raw selected scans")
    complete = tables.get("Complete decomposition")
    if raw is None and complete is None:
        return None
    keys = ["Curve", "Source row"]
    if raw is not None:
        data = raw[keys].assign(**{
            "Potential (V)": raw["Measured potential (V)"] + potential_shift,
            "Signed current (A/cm²)": raw["Current density (A/cm²)"],
        })
    else:
        columns = {
            "Reference-converted potential (V)": "Potential (V)",
            "Signed current density (A/cm²)": "Signed current (A/cm²)",
        }
        data = complete[keys + list(columns)].rename(columns=columns)
    mapping = {
        "Kinetic overpotential (V)": "Kinetic (V)",
        "Ohmic overpotential (V)": "Ohmic (V)",
    }
    if include_rcl:
        mapping["RCL overpotential (V)"] = "RCL (V)"
    mapping["Signed residual (V)"] = "Residual (V)"
    if complete is not None:
        model = complete[keys + list(mapping)].rename(columns=mapping)
        data = data.merge(model, on=keys, how="left", sort=False, validate="one_to_one")
    else:
        for column in mapping.values():
            data[column] = np.nan
    return data


def _run_settings(table, settings, include_rcl):
    fields = {
        "input_type", "current_branch", "Electrode setup", "Potential scale", "Potential shift (V)",
        "electrode_area", "hfr_mode", "include_rcl", "i_lower", "i_upper", "target_i",
        "ccd_fit_mode", "ccd_threshold_mv", "persistence_v", "method_version", "Tafel method",
        "Temperature (K)" if "Temperature (K)" in settings else "temperature",
    }
    if settings.get("input_type") == "CV":
        fields.update(("cv_scan", "cv_cycle_mode"))
        if settings.get("cv_cycle_mode") == "Specific":
            fields.add("cv_cycle")
    if settings.get("Electrode setup") == "Three-electrode":
        fields.update(("Reference electrode", "Reference E vs SHE (V)", "pH", "reaction"))
    if settings.get("hfr_mode") == "Constant HFR":
        fields.add("hfr")
    if include_rcl:
        fields.add("rcl")
    if settings.get("ccd_fit_mode") == "Manual":
        fields.update(("ccd_j_lower", "ccd_j_upper"))
    return table.loc[table["Parameter"].isin(fields)]


def readable_batch_tables(result):
    """One row per scientific result, followed by one copy of each curve record.

    File/cycle identities belong in Current results. Dense curve sheets refer to
    that short Curve ID rather than repeating long filenames on every row.
    """
    tables = result.tables
    if "Current results" not in tables:
        return {name: frame.copy() for name, frame in tables.items()
                if not name.startswith("Plot ")}
    settings_table = tables.get("Settings")
    settings = {} if settings_table is None else dict(
        settings_table[["Parameter", "Value"]].itertuples(index=False, name=None),
    )
    include_rcl = bool(settings.get("include_rcl", result.scalars.get("RCL analysis") == "enabled"))
    summary = tables["Current results"].copy()
    processed = tables.get("Processed scans")
    if processed is not None and len(processed):
        detected = processed.loc[processed["CCD point"].eq(True),
                                 ["Curve", "Signed residual (V)", "Potential (V)"]]
        detected = detected.rename(columns={"Signed residual (V)": "CCD residual (V)",
                                            "Potential (V)": "CCD potential (V)"})
        summary = summary.merge(detected, on="Curve", how="left", sort=False, validate="one_to_one")
    else:
        summary["CCD residual (V)"] = np.nan
        summary["CCD potential (V)"] = np.nan
    columns = ["File", "Cycle", "Scan", "Curve", "CCD (A/cm²)", "CCD residual (V)",
               "CCD potential (V)", "Limiting current (A/cm²)", "Decoupling status",
               "Tafel status",
               "CCD status", "Limiting-current status", "Breakdown status", "Notes"]
    sections = {}
    if settings_table is not None:
        sections["Settings"] = _run_settings(settings_table, settings, include_rcl)
    sections.update({name: tables[name] for name in ("HFR points", "HFR settings") if name in tables})
    sheets = {"Summary": summary_table(sections)} if sections else {}
    sheets["Current results"] = summary[[key for key in columns if key in summary]]
    if "Breakdown results" in tables:
        data = tables["Breakdown results"]
        columns = ["File", "Cycle", "Scan", "Curve", "Target |j| (A/cm²)", "Kinetic (V)", "Ohmic (V)"]
        if include_rcl:
            columns.append("RCL (V)")
        columns += ["Residual (V)", "Measured potential (V)", "Status"]
        sheets["Breakdown results"] = data[[key for key in columns if key in data]].rename(
            columns={"Measured potential (V)": "Potential at target (V)"},
        )
    for name, unit in [("Current statistics", "A/cm²"), ("Breakdown statistics", "V")]:
        if name in tables:
            sheets[name] = tables[name].drop(columns="Source", errors="ignore").rename(
                columns={"Mean": f"Mean ({unit})", "Sample SD": f"Sample SD ({unit})"}).copy()
    if "Batch statistics" in tables:
        sheets["Batch statistics"] = tables["Batch statistics"].copy()
    data = _curve_data(tables, include_rcl, settings.get("Potential shift (V)", 0.0))
    if data is not None:
        sheets["Curve data"] = data
    if processed is not None:
        mapping = {
            "Curve": "Curve", "Segment": "Segment",
            "Smoothed current (A/cm²)": "Smoothed current (A/cm²)",
            "Signed residual (V)": "Residual (V)", "Residual baseline (V)": "Baseline (V)",
            "Baseline fit": "Baseline fit",
            "CCD point": "CCD point", "Limiting plateau": "Limiting plateau",
        }
        sheets["CCD data"] = processed[[key for key in mapping if key in processed]].rename(columns=mapping).copy()
    if "Fit diagnostics" in tables:
        data = tables["Fit diagnostics"]
        columns = ["Curve", "Tafel b (V/dec)", "Exchange current (A/cm²)", "Tafel R²",
                   "Tafel actual lower (A/cm²)", "Tafel actual upper (A/cm²)", "Tafel RMS (V)",
                   "Tafel fitted points", "Residual baseline status", "Baseline lower (A/cm²)",
                   "Baseline upper (A/cm²)", "Baseline slope (Ω·cm²)", "Baseline intercept (V)",
                   "Baseline R²", "Baseline RMS (V)"]
        sheets["Fit diagnostics"] = data[[key for key in columns if key in data]].copy()
    for name in ("HFR source data", "HFR fit data"):
        if name in tables:
            sheets[name] = tables[name].copy()
    return sheets


def batch_artifacts(result):
    items = []
    if result.tables:
        items.append(Artifact("Batch analysis workbook (Excel)", "xlsx", "batch_decoupling_ccd.xlsx",
                              readable_batch_tables(result)))
    for name in dict.fromkeys([*result.figures, *result.figure_images]):
        items.append(Artifact(f"{name} (PNG)", "png", f"batch_{_slug(name)}.png",
                              result.figures.get(name),
                              None if name in result.figures else result.figure_images[name]))
    return items
