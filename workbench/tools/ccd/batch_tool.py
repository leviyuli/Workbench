"""Independent scan decoupling and CCD from the signed residual baseline."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba

from workbench.core.tool_base import Tool, ToolResult
from workbench.tools.ccd.batch_exports import batch_artifacts
from workbench.tools.ccd.batch_inputs import migrate_batch_page_state, read_batch_scans
from workbench.tools.ccd.batch_plots import BatchCurve, build_batch_views
from workbench.tools.ccd.parameters import ccd_parameters
from workbench.tools.ccd.plot_style import source_colors
from workbench.tools.ccd.residual import ResidualCcdSettings, analyze_residual_ccd
from workbench.tools.lsv.analysis import DecouplingSettings, decouple_curve
from workbench.tools.lsv.hfr import HfrProfile, evaluate_hfr
from workbench.tools.lsv.hfr_results import hfr_fit_result
from workbench.tools.lsv.logic_tafel_stable import StableTafelSettings, TafelFitError


CCD_COLUMN = "CCD (A/cm²)"
LIMIT_COLUMN = "Limiting current (A/cm²)"
TAFEL_METHOD = StableTafelSettings().method


def _identity(scan):
    return {"Curve": scan.code, "Source": scan.source, "File": scan.file,
            "Cycle": scan.cycle, "Scan": scan.branch, "Native direction": scan.direction}


def _frame(identity, columns):
    return pd.DataFrame({**identity, **columns})


def _tafel_counts(diagnostics):
    labels = {"range_points": "Tafel search points", "tested_windows": "Tafel tested windows",
              "qualified_windows": "Tafel qualifying windows", "segments": "Tafel acquisition segments",
              "accepted_intervals": "Tafel accepted intervals", "iterations": "Tafel iterations",
              "rcl_seed_used": "Tafel RCL seed used", "rcl_seed_slope": "Tafel RCL seed slope (V/dec)",
              "rcl_seed_count": "Tafel RCL seed count"}
    return {label: diagnostics[key] for key, label in labels.items() if key in diagnostics}


def _tafel_fit_record(curve):
    fit, candidate = curve.fit, curve.fit["candidate_details"]
    diagnostics = fit.get("diagnostics", {})
    details = {"Tafel status": "Resolved", "Tafel b (V/dec)": fit["b_kin"],
               "Exchange current (A/cm²)": fit["i0"], "Tafel R²": fit["r_squared"],
               "Tafel actual lower (A/cm²)": candidate["current_range"][0],
               "Tafel actual upper (A/cm²)": candidate["current_range"][1],
               "Tafel corrected lower (V)": candidate["window"][0],
               "Tafel corrected upper (V)": candidate["window"][1],
               "Tafel log-current span (decades)": candidate.get("log_current_span", np.nan),
               "Tafel RMS (V)": candidate.get("rmse", np.nan),
               "Tafel fitted points": fit["n_points"],
               "Tafel joined windows": candidate.get("window_count", np.nan),
               "Tafel half-slope difference (V/dec)": candidate.get("half_slope_difference", np.nan),
               "Tafel window-slope spread (V/dec)": candidate.get("constituent_slope_spread", np.nan),
               "Tafel source row start": candidate.get("source_start", np.nan),
               "Tafel source row end": candidate.get("source_end", np.nan),
               **_tafel_counts(diagnostics)}
    validation = diagnostics.get("rcl_final_interval_validation", {})
    for key, label in [("model_rmse", "Tafel final RCL model RMS (V)"),
                       ("refit_rmse", "Tafel final RCL refit RMS (V)")]:
        if key in validation:
            details[label] = validation[key]
    return details


def _matching_cycle_statistics(rows):
    records = []
    for cycle, group in rows.groupby("Cycle", dropna=False, sort=True):
        for field in (CCD_COLUMN, LIMIT_COLUMN):
            values = group[field].dropna().to_numpy(dtype=float)
            records.append({"Cycle": cycle, "Metric": field, "Resolved files": len(values),
                            "Mean (A/cm²)": np.mean(values) if len(values) else np.nan,
                            "Sample SD (A/cm²)": np.std(values, ddof=1) if len(values) > 1 else np.nan})
    return pd.DataFrame(records)


class CcdTool(Tool):
    id = "lsv_ccd"
    name = "Batch decoupling & CCD"
    subtitle = "Decouple each LSV or CV cycle and identify CCD above its residual baseline."
    shared_file_group = "lsv_analysis_inputs"
    run_button_label = "Analyze batch"
    extra_after_parameter = "hfr_mode"

    def parameter_schema(self):
        return ccd_parameters()

    def extra_widget(self, parent, on_change):
        from workbench.tools.lsv.inputs import HfrInputs
        return HfrInputs(parent, on_change=on_change)

    def migrate_page_state(self, state):
        return migrate_batch_page_state(state)

    def exportable_artifacts(self, result):
        return batch_artifacts(result)

    def run(self, files: list[Path], params: dict, extra=None) -> ToolResult:
        values = {spec.key: spec.default for spec in self.parameter_schema()}
        values.update(params)
        settings = DecouplingSettings.from_params(values, extra)
        threshold_error = "CCD threshold must be a finite value greater than zero mV."
        try:
            threshold_mv = float(values.get("ccd_threshold_mv", 50.))
        except (TypeError, ValueError, OverflowError):
            raise ValueError(threshold_error) from None
        if (isinstance(values.get("ccd_threshold_mv"), bool)
                or not np.isfinite(threshold_mv) or threshold_mv <= 0):
            raise ValueError(threshold_error)
        values["ccd_threshold_mv"] = threshold_mv
        residual_settings = ResidualCcdSettings(threshold_v=threshold_mv / 1000)
        target = float(values["target_i"])
        if not np.isfinite(target) or target <= 0:
            raise ValueError("Breakdown target current must be a finite value greater than zero.")
        area = float(values["electrode_area"])
        if not np.isfinite(area) or area < 0:
            raise ValueError("Electrode area must be finite and nonnegative.")
        branch = values.get("current_branch", "Auto")
        if branch not in {"Auto", "Positive", "Negative"}:
            raise ValueError("Current sign must be Auto, Positive or Negative.")
        scans = read_batch_scans(files, values)
        sources = list(dict.fromkeys(scan.source for scan in scans))
        palette = source_colors(sources)
        cycle_extents = {}
        for source in sources:
            cycles = [scan.cycle for scan in scans if scan.source == source and scan.cycle is not None]
            cycle_extents[source] = (min(cycles), max(cycles)) if cycles else (1, 1)
        curves, rows, raw, complete, processed, fits, breakdowns = [], [], [], [], [], [], []
        warnings = list(settings.correction.warnings)
        for scan in scans:
            identity = _identity(scan)
            notes = list(scan.warnings)
            row = {**identity, "Decoupling status": "Analysis failed", CCD_COLUMN: np.nan,
                   "Tafel status": "Unavailable", "Tafel method": TAFEL_METHOD,
                   LIMIT_COLUMN: np.nan, "CCD status": "Decoupling failed",
                   "Limiting-current status": "Decoupling failed", "Breakdown status": "Unavailable",
                   "Notes": ""}
            breakdown = {**identity, "Target |j| (A/cm²)": target, "Kinetic (V)": np.nan,
                         "Ohmic (V)": np.nan, "RCL (V)": np.nan, "Residual (V)": np.nan,
                         "Measured potential (V)": np.nan, "Status": "Decoupling failed"}
            fit_row = {**identity, "Tafel method": TAFEL_METHOD, "Tafel status": "Unavailable",
                       "Tafel search lower (A/cm²)": settings.i_lower,
                       "Tafel search upper (A/cm²)": settings.i_upper,
                       **{name: np.nan for name in (
                           "Tafel b (V/dec)", "Exchange current (A/cm²)", "Tafel R²",
                           "Tafel actual lower (A/cm²)", "Tafel actual upper (A/cm²)",
                           "Tafel corrected lower (V)", "Tafel corrected upper (V)",
                           "Tafel log-current span (decades)", "Tafel RMS (V)",
                           "Tafel fitted points", "Tafel joined windows",
                           "Tafel half-slope difference (V/dec)",
                           "Tafel window-slope spread (V/dec)",
                           "Tafel source row start", "Tafel source row end")},
                       "Tafel detail": ""}
            density = scan.current.copy()
            if not scan.is_density and area > 0:
                density /= area
            if len(scan.potential):
                raw.append(_frame(identity, {
                    "Source row": scan.source_indices, "Measured potential (V)": scan.potential,
                    "Imported current (A or A/cm²)": scan.current,
                    "Input is current density": np.full(len(density), scan.is_density),
                    "Current density (A/cm²)": density if scan.is_density or area > 0
                    else np.full(len(density), np.nan),
                }))
            try:
                if scan.error:
                    raise ValueError(scan.error)
                if not scan.is_density and area <= 0:
                    raise ValueError("Enter a positive electrode area to normalize absolute current to A/cm².")
                is_cell = settings.correction.electrode_setup == "Two-electrode"
                reaction = "auto" if is_cell else str(values.get("reaction", "Auto")).lower()
                if branch != "Auto":
                    reaction = "her" if branch == "Negative" else "oer"
                d = decouple_curve(scan.potential, density, settings, reaction,
                                   source_indices=scan.source_indices, tafel_method="contiguous-stable")
                row["Tafel status"] = "Resolved"
                fit_row.update(_tafel_fit_record(d))
                r = analyze_residual_ccd(d, mode=values["ccd_fit_mode"],
                                         current_lower=float(values["ccd_j_lower"]),
                                         current_upper=float(values["ccd_j_upper"]),
                                         settings=residual_settings)
                first, last = cycle_extents[scan.source]
                alpha = (.35 + .65 * (scan.cycle - first) / (last - first)
                         if scan.cycle is not None and last > first else 1.0)
                curves.append(BatchCurve(scan, d, r, to_rgba(palette[scan.source], alpha)))
                notes.extend(d.warnings)
                notes.extend(r.warnings)
                row.update({"Decoupling status": "Resolved", CCD_COLUMN: r.ccd_current,
                            LIMIT_COLUMN: r.limiting_current, "CCD status": r.ccd_status,
                            "Limiting-current status": r.limiting_status})
                try:
                    target_values = d.breakdown_at(target)
                    breakdown.update({"Kinetic (V)": target_values["eta_kin"],
                                      "Ohmic (V)": target_values["eta_ohm"],
                                      "RCL (V)": target_values["eta_rcl"],
                                      "Residual (V)": target_values["residual"],
                                      "Measured potential (V)": target_values["native_potential"],
                                      "Status": "Resolved"})
                    row["Breakdown status"] = "Resolved"
                except ValueError as exc:
                    breakdown["Status"] = str(exc)
                    row["Breakdown status"] = str(exc)
                    notes.append(str(exc))
                complete.append(_frame(identity, {
                    "Source row": d.source_indices, "Measured potential (V)": d.measured_potential,
                    "Reference-converted potential (V)": d.potential,
                    "Signed current density (A/cm²)": d.current,
                    "Current magnitude (A/cm²)": d.current_magnitude,
                    "Measured drive (V)": d.drive, "Total overpotential (V)": d.eta_total,
                    "Kinetic overpotential (V)": d.eta_kin, "Ohmic overpotential (V)": d.eta_ohm,
                    "RCL overpotential (V)": d.eta_rcl, "Signed residual (V)": d.residual,
                }))
                plateau = np.zeros(len(r.drive), dtype=bool)
                if r.plateau_start is not None:
                    plateau[r.plateau_start:r.plateau_end + 1] = True
                ccd = np.zeros(len(r.drive), dtype=bool)
                if r.ccd_index is not None:
                    ccd[r.ccd_index] = True
                processed.append(_frame(identity, {
                    "Source row start": r.source_start, "Source row end": r.source_end,
                    "Segment": r.segment_id, "Measured drive (V)": r.drive,
                    "Potential (V)": r.potential, "Median current (A/cm²)": r.median_current,
                    "Smoothed current (A/cm²)": r.current,
                    "Kinetic overpotential (V)": r.eta_kin, "Ohmic overpotential (V)": r.eta_ohm,
                    "RCL overpotential (V)": r.eta_rcl, "Signed residual (V)": r.residual,
                    "Residual baseline (V)": r.baseline, "Excess above baseline (V)": r.excess,
                    "Baseline fit": r.baseline_mask, "CCD point": ccd, "Limiting plateau": plateau,
                }))
                fit_row.update({"Residual baseline status": r.baseline_status,
                             "Residual fit mode": values["ccd_fit_mode"],
                             "Baseline lower (A/cm²)": r.baseline_lower_current,
                             "Baseline upper (A/cm²)": r.baseline_upper_current,
                             "Baseline slope (Ω·cm²)": r.baseline_slope,
                             "Baseline intercept (V)": r.baseline_intercept,
                             "Baseline R²": r.baseline_r_squared, "Baseline RMS (V)": r.baseline_rmse,
                             "Reversed toward increasing drive": r.reversed_for_analysis})
            except TafelFitError as exc:
                reason = str(exc)
                row.update({"Decoupling status": "Tafel unresolved", "Tafel status": "Unresolved",
                            "CCD status": "Tafel unresolved", "Limiting-current status": "Tafel unresolved",
                            "Breakdown status": "Tafel unresolved"})
                breakdown["Status"] = "Tafel unresolved"
                fit_row.update({"Tafel status": "Unresolved", "Tafel detail": reason})
                fit_row.update(_tafel_counts(exc.diagnostics))
                notes.append(reason)
            except ValueError as exc:
                fit_row["Tafel detail"] = str(exc)
                notes.append(str(exc))
            fits.append(fit_row)
            row["Notes"] = " | ".join(dict.fromkeys(notes))
            warnings.extend(f"{scan.code} ({scan.file}): {note}" for note in dict.fromkeys(notes))
            rows.append(row)
            breakdowns.append(breakdown)
        results = pd.DataFrame(rows)
        results["Cycle"] = results["Cycle"].astype("Int64")
        for column in (CCD_COLUMN, LIMIT_COLUMN):
            results[column] = pd.to_numeric(results[column], errors="coerce")
        breakdown = pd.DataFrame(breakdowns)
        if curves:
            figures, plot_sheets, statistics, breakdown_statistics = build_batch_views(
                curves, results, breakdown, settings.include_rcl)
        else:
            figures, plot_sheets, statistics, breakdown_statistics = {}, {}, pd.DataFrame(), pd.DataFrame()
        tables = {**plot_sheets, "Current results": results, "Current statistics": statistics,
                  "Batch statistics": _matching_cycle_statistics(results),
                  "Breakdown results": breakdown, "Breakdown statistics": breakdown_statistics,
                  "Fit diagnostics": pd.DataFrame(fits)}
        for name, frames in [("Processed scans", processed), ("Complete decomposition", complete),
                             ("Raw selected scans", raw)]:
            if frames:
                tables[name] = pd.concat(frames, ignore_index=True)
        internal = asdict(residual_settings)
        tafel_internal = asdict(StableTafelSettings())
        metadata = {**values, **settings.correction.metadata(), **internal,
                    "Tafel method": TAFEL_METHOD,
                    **{f"Tafel {key}": value for key, value in tafel_internal.items() if key != "method"},
                    "Tafel interval selection": "Widest stable log-current span; earliest corrected voltage breaks ties",
                    "Residual definition": "total - kinetic - j*HFR - RCL; no clipping",
                    "Back scan evaluation": "Toward increasing measured electrochemical drive",
                    "Source row basis": "Zero-based data-table row, excluding source header",
                    "Component plot minimum current (A/cm²)": settings.i_lower}
        tables["Settings"] = pd.DataFrame(metadata.items(), columns=["Parameter", "Value"])
        hfr_result = None
        if isinstance(settings.hfr, HfrProfile):
            analyzed_current = (np.concatenate([item.decomposition.current_magnitude for item in curves])
                                if curves else np.array([]))
            hfr_result = hfr_fit_result(settings.hfr, analyzed_current, extra)
            figures.update(hfr_result.figures)
            tables.update(hfr_result.tables)
        try:
            target_hfr = float(evaluate_hfr(settings.hfr, target))
        except ValueError as exc:
            if not isinstance(settings.hfr, HfrProfile):
                raise
            target_hfr = np.nan
            warnings.append(f"HFR at breakdown target is unresolved: {exc}")
        common_ohmic = target * target_hfr
        scalars = {"Files selected": len(files), "Scans selected": len(scans),
                   "Scans decoupled": len(curves), "CCD resolved": int(results[CCD_COLUMN].notna().sum()),
                   "Limiting currents resolved": int(results[LIMIT_COLUMN].notna().sum()),
                   "Method": internal["method_version"], "HFR mode": values["hfr_mode"],
                   "CCD threshold (mV)": 1000 * residual_settings.threshold_v,
                   "CCD persistence (V)": residual_settings.persistence_v,
                   "Tafel method": TAFEL_METHOD,
                   "Tafel unresolved": int(results["Tafel status"].eq("Unresolved").sum()),
                   "HFR at target (Ω·cm²)": target_hfr,
                   "Breakdown target (A/cm²)": target, "Common target ohmic drop (V)": common_ohmic,
                   "RCL analysis": "enabled" if settings.include_rcl else "disabled"}
        result = ToolResult(
            figures=figures, tables=tables, scalars=scalars, warnings=list(dict.fromkeys(warnings)),
            display_tables=False,
            text_blocks={
                "Tafel fitting": "Each scan uses a contiguous stable-slope interval within the manual "
                    "Tafel current limits. The search uses 60 mV windows stepped by 5 mV, nested "
                    "20/40/60 mV fits, at least three neighboring windows, a 5 mV RMS limit, "
                    "20% slope-stability checks and at least 0.5 current decades. The widest "
                    "qualifying log-current interval is refitted once. Unresolved scans retain "
                    "their raw data and diagnostic status in the workbook.",
                "CCD definition": f"CCD is the first sustained {1000 * residual_settings.threshold_v:g} mV "
                    "excess above a line fitted to the signed decoupling residual, confirmed over "
                    f"{1000 * residual_settings.persistence_v:g} mV of measured drive. "
                    "Auto selects the widest contiguous stable region "
                    "above the configured Tafel current range. Inspect the highlighted fit; the baseline "
                    "can already contain losses and is not an HFR measurement.",
                "Scan direction": "Each selected scan is decoupled independently. Back scans are "
                    "evaluated toward increasing electrochemical drive; their CCD is not a temporal "
                    "onset during the recorded return sweep. Gaps are preserved.",
                "Statistics": "Dots are individual scans; short black bars show file means. Sample SD "
                    "is included in the workbook when at least two scans resolve. Cycle spread is "
                    "descriptive and does not imply "
                    "independent replication. Missing results are excluded separately for each metric.",
                "Plot range": "Component and residual-scan displays start at the Tafel search lower "
                    "current bound, avoiding near-zero-current extrapolation of the kinetic model. "
                    "The original polarization and complete exported decomposition retain every input row.",
                "Export": "The workbook starts with Summary for run settings and HFR measurements, "
                    "followed by Current results and Breakdown results with one row per scan. "
                    "Curve data contains the analysis-scale potential, signed current and loss components. "
                    "CCD data contains the smoothed current, residual, fitted baseline and detection flags. "
                    "Fit diagnostics reports the final Tafel and residual-baseline fit values.",
            },
        )
        if hfr_result is not None:
            result.scalars.update(hfr_result.scalars)
            result.text_blocks.update(hfr_result.text_blocks)
        return result
