from __future__ import annotations

from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.figure import Figure

from workbench import theme
from workbench.core.export_tables import summary_table
from workbench.core.potential import (
    FARADAY,
    REFERENCE_OPTIONS,
    REFERENCE_TEMPERATURE_K,
    R_GAS,
    PotentialCorrection,
    convert_reference_to_rhe,
    potential_parameter_schema,
)
from workbench.core.tool_base import Artifact, ParameterSpec, Tool, ToolAction, ToolResult
from workbench.tools.lsv.analysis import DecouplingSettings, decouple_curve
from workbench.tools.lsv.ccd import analyze_ccd, ccd_parameters
from workbench.tools.lsv.data import ALL_MEASURED_VOLTAGES, read_decoupling_data
from workbench.tools.lsv.hfr import HfrProfile, evaluate_hfr
from workbench.tools.lsv.hfr_results import hfr_fit_result
from workbench.tools.lsv.logic_tafel import (
    calculate_RCL_utilization,
    get_overpotential_at_current,
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
    shared_file_group = "lsv_analysis_inputs"
    accepts_directory = False
    accepted_file_types = (".csv", ".txt", ".xlsx", ".xls")
    run_button_label = "Fit & Plot"
    extra_after_parameter = "hfr_mode"

    def parameter_schema(self) -> list[ParameterSpec]:
        return [
            ParameterSpec(
                "input_type", "Input type", kind="choice", default="LSV",
                choices=["LSV", "CV"], group="Input data",
                tooltip="Use an LSV curve, or select one scan from a CV cycle for decoupling.",
            ),
            ParameterSpec(
                "average_range", "Averaging range", kind="choice", default="Overlap only",
                choices=["Overlap only", ALL_MEASURED_VOLTAGES], group="Input data",
                show_if=("input_type", "LSV"),
                tooltip="For multiple LSVs, All measured voltages averages the scans "
                        "covering each voltage. Fit & Plot requires continuous voltage coverage.",
            ),
            ParameterSpec(
                "cv_scan", "Scan", kind="choice", default="Forward",
                choices=["Forward", "Back"], group="Input data",
                show_if=("input_type", "CV"),
                tooltip="Forward follows the initial sweep direction in each file. "
                        "Back is the return sweep in the selected cycle. "
                        "The chosen scan is used for both plotting and fitting.",
            ),
            ParameterSpec(
                "cv_cycle", "Cycle", kind="int", default=0, minimum=0,
                maximum=1_000_000, special_value_text="Last", group="Input data",
                show_if=("input_type", "CV"),
                tooltip="Last uses the last acquired cycle in each file. "
                        "Increase to a cycle number to select it explicitly. "
                        "Labelled files use their source cycle numbers; "
                        "unlabelled cycles are numbered from 1.",
            ),
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
                "electrode_area", "Electrode area", kind="float", default=1.0,
                minimum=0.0, step=0.01, decimals=4, suffix="cm²",
                tooltip="Default electrode area: 1 cm². Used to convert absolute current (A) "
                        "to current density. Data already in A/cm² are used directly.",
                group="Electrolyte & reference",
            ),
            ParameterSpec(
                "hfr_mode", "HFR mode", kind="choice", default="Constant HFR",
                choices=["Constant HFR", "HFR vs |j|"], group="Resistances",
                tooltip="Use one HFR value, or upload/paste HFR versus current density "
                        "below. The selected test is fitted with a cubic smoothing spline "
                        "and extended using endpoint slopes across the LSV range.",
            ),
            ParameterSpec("hfr", "HFR", kind="float", default=0.01, minimum=0.0,
                          step=0.001, decimals=5, suffix="Ω·cm²", group="Resistances",
                          show_if=("hfr_mode", "Constant HFR")),
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
            *ccd_parameters(),
        ]

    def actions(self) -> list[ToolAction]:
        return [ToolAction(label="Show plot", method="show_plot")]

    def extra_widget(self, parent, on_change):
        from workbench.tools.lsv.inputs import HfrInputs

        return HfrInputs(parent, on_change=on_change)

    def migrate_page_state(self, state):
        if not isinstance(state, dict) or not isinstance(state.get("result"), dict):
            return state
        migrated = deepcopy(state)
        result = migrated["result"]
        result["display_tables"] = False
        count = len(result.get("figures", {})) + int(result.get("display_summary", True))
        if migrated.get("result_tab", 0) >= count:
            migrated["result_tab"] = 0
        return migrated

    def show_plot(self, files: list[Path], params: dict, extra=None) -> ToolResult:
        """Plot each selected polarization scan on the selected potential scale."""
        if not files:
            raise ValueError("Add at least one LSV or CV file to plot.")
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
        combined, err = read_decoupling_data(paths, params, reaction=reaction_param)
        if err:
            raise ValueError(err)
        V_avg, i_avg, info = combined
        V_avg = correction.apply(V_avg)
        area_cm2 = float(params.get("electrode_area", 1.0))
        i_avg, area_warning = _as_current_density(i_avg, info, area_cm2)
        resolved = info.get("reaction", "oer")
        is_her = resolved == "her"
        averaged = bool(info.get("averaged", False))
        is_cv = params.get("input_type", "LSV") == "CV"

        warnings = list(correction.warnings)
        if info.get("reaction_auto") and not is_cell:
            warnings.append(f"Auto-detected {resolved.upper()} from the current sign.")
        if area_warning:
            warnings.append(area_warning)
        if info.get("warning") and not area_warning:
            warnings.append(info["warning"])
        if np.isnan(i_avg).any():
            warnings.append("The averaged LSV has a gap in voltage coverage.")

        # Per-file curves, read with the resolved reaction so signs are consistent.
        per_file: list[tuple[str, np.ndarray, np.ndarray]] = []
        raw_frames: list[pd.DataFrame] = []
        used_labels: set[str] = set()
        for p in paths:
            single, serr = read_decoupling_data([p], params, reaction=resolved)
            if serr:
                warnings.append(f"{Path(p).name}: {serr}")
                continue
            Vf, iff, single_info = single
            Vf = correction.apply(Vf)
            iff, _ = _as_current_density(iff, single_info, area_cm2)
            label = Path(p).name
            suffix = 2
            while label in used_labels:
                label = f"{Path(p).name} ({suffix})"
                suffix += 1
            used_labels.add(label)
            per_file.append((label, Vf, iff))
            raw_frames.append(pd.DataFrame({
                "File": label,
                potential_label: Vf,
                "Current Density (A/cm²)": iff,
            }))
        if not per_file:
            raise ValueError("No input scans could be read for plotting.")

        figure_name = "CV scans" if is_cv else "LSV originals"
        input_table = "CV scans (per file)" if is_cv else "LSV data (per file)"
        figures = {figure_name: _plot_originals(
            per_file, V_avg, i_avg, is_her and not is_cell, averaged,
            potential_label=potential_label, source_label="CV scan" if is_cv else "LSV",
        )}
        tables = {input_table: pd.concat(raw_frames, ignore_index=True)}
        potential_settings = correction.metadata()
        if averaged:
            tables["Averaged CV scan" if is_cv else "Averaged LSV"] = pd.DataFrame({
                potential_label: V_avg,
                "Current Density (A/cm²)": i_avg,
            })
        tables["Potential settings"] = pd.DataFrame(
            potential_settings.items(), columns=["Parameter", "Value"],
        )
        scalars = {
            "Files plotted": len(per_file),
            "Reaction": "Full cell" if is_cell else resolved.upper(),
            "Averaged": "yes" if averaged else "no (single file)",
            **potential_settings,
        }
        if is_cv:
            tables["CV selection"] = pd.DataFrame(info["cv_selections"])
            scalars.update({"Input type": "CV", "CV scan": info["cv_scan"],
                            "CV cycle": info["cv_cycle"]})
        return ToolResult(figures=figures, tables=tables, scalars=scalars,
                          text_blocks={"Potential correction": correction.describe()},
                          warnings=warnings, display_tables=False)

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
        data, err = read_decoupling_data([str(f) for f in files], params, reaction=reaction_param)
        if err:
            raise ValueError(err)
        V_raw, i_raw, info = data
        is_cv = params.get("input_type", "LSV") == "CV"
        if np.isnan(i_raw).any():
            raise ValueError(
                "The averaged LSV has a gap in voltage coverage. "
                "Use Show plot to see the measured segments or Batch decoupling & CCD "
                "to fit each scan independently."
            )

        settings = DecouplingSettings.from_params(params, extra)
        T = settings.temperature
        ref = correction.reference
        hfr_mode = params.get("hfr_mode", "Constant HFR")
        hfr = settings.hfr
        include_rcl, rcl = settings.include_rcl, settings.rcl
        target_i = float(params["target_i"])
        area_cm2 = float(params.get("electrode_area", 1.0))
        i_raw, area_warning = _as_current_density(i_raw, info, area_cm2)

        reaction = info.get("reaction", "oer")
        is_her = reaction == "her"
        i_mag = np.abs(i_raw)
        if not np.isfinite(target_i) or target_i <= 0:
            raise ValueError("Breakdown target current must be greater than zero.")
        measured_min = float(np.min(i_mag))
        measured_max = float(np.max(i_mag))
        if target_i < measured_min or target_i > measured_max:
            raise ValueError(
                f"Breakdown target {target_i:g} A/cm² is outside the measured "
                f"range {measured_min:g}–{measured_max:g} A/cm²."
            )

        # Fit current magnitudes and positive losses, then restore voltage polarity.
        curve = decouple_curve(V_raw, i_raw, settings, reaction=reaction)
        sign, E_rev, V = curve.sign, curve.e_rev, curve.potential
        rhe_shift = correction.shift_v
        e_ref_vs_she = correction.e_ref_vs_she
        eta_total_mag = curve.eta_total
        hfr_values = evaluate_hfr(hfr, i_mag)
        hfr_at_target = evaluate_hfr(hfr, target_i)
        fit = curve.fit
        warn = " ".join(curve.warnings) or None
        eta_ohm_all, eta_rcl_all = curve.eta_ohm, curve.eta_rcl
        b_kin, i0 = fit["b_kin"], fit["i0"]

        eta_kin = curve.eta_kin
        eta_res = np.maximum(curve.residual, 0)

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
        source_label = "CV scan" if is_cv else "LSV"
        original_label = f"Original {source_label} ({'cell voltage' if is_cell else 'RHE'})"
        curves.update({"+ η_res": y5, original_label: y6})
        components["η_res (V)"] = sign * eta_res
        if isinstance(hfr, HfrProfile):
            components["HFR (Ω·cm²)"] = hfr_values
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
        hfr_diag = (
            f"HFR: cubic smoothing spline, {len(hfr.points)} measurements; "
            "endpoint-slope extension to the LSV range\n"
            f"  HFR at {target_i:g} A/cm²: {hfr_at_target:.6g} Ω·cm²\n"
            if isinstance(hfr, HfrProfile) else
            f"HFR: {hfr:g} Ω·cm² (constant)\n"
        )
        diag = potential_diag + hfr_diag + (
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
                include_rcl=include_rcl, is_cell=is_cell, source_label=source_label),
            "Overpotential breakdown": _plot_breakdown(
                target_i, i_mag, sign * E_rev + eta_total_mag, sign * E_rev,
                hfr, rcl, fit, is_her=is_her,
                include_rcl=include_rcl, is_cell=is_cell),
        }
        tables = {
            "Curves (stacked)": pd.DataFrame(curves),
            "Overpotential components": pd.DataFrame(components),
        }
        if isinstance(hfr, HfrProfile):
            tables["HFR points"] = pd.DataFrame(
                hfr.points, columns=["|j| (A/cm²)", "HFR (Ω·cm²)"],
            )
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
            "HFR mode": hfr_mode,
            **potential_settings,
        }
        if isinstance(hfr, HfrProfile):
            scalars["HFR at target (Ω·cm²)"] = hfr_at_target
        breakdown = get_overpotential_at_current(
            target_i, i_mag, sign * E_rev + eta_total_mag, sign * E_rev,
            hfr, rcl, fit, include_rcl=include_rcl,
        )
        breakdown_values = {
            "Target |j| (A/cm²)": target_i,
            "Reversible (V)": max(float(breakdown["E_rev"]), 0.0),
            "Kinetic (V)": max(float(breakdown["eta_kin"]), 0.0),
            "Ohmic (V)": max(float(breakdown["eta_ohm"]), 0.0),
        }
        if include_rcl:
            breakdown_values["Catalyst-layer (V)"] = max(float(breakdown["eta_rcl"]), 0.0)
        breakdown_values["Residual (V)"] = max(float(breakdown["eta_res"]), 0.0)
        conditions = {"Reaction": "Full cell" if is_cell else reaction.upper()}
        if is_cv:
            tables["CV selection"] = pd.DataFrame(info["cv_selections"])
            selection_settings = {"Input type": "CV", "CV scan": info["cv_scan"],
                                  "CV cycle": info["cv_cycle"]}
            scalars.update(selection_settings)
            conditions.update(selection_settings)
            diag = (f"Input: CV, {info['cv_scan'].lower()} scan, cycle {info['cv_cycle']}\n" + diag)
        if isinstance(hfr, HfrProfile):
            conditions.update({
                "HFR mode": hfr_mode,
                **hfr.metadata(),
                "HFR at target (Ω·cm²)": hfr_at_target,
            })
        else:
            conditions["HFR (Ω·cm²)"] = hfr
        if include_rcl:
            conditions["R_CL (Ω·cm²)"] = rcl
        result = ToolResult(
            figures=figures,
            tables=tables,
            scalars=scalars,
            text_blocks={"Diagnostics": diag},
            warnings=warnings,
            payload={"export_breakdown": breakdown_values, "export_conditions": conditions},
            display_tables=False,
        )
        if isinstance(hfr, HfrProfile):
            hfr_result = hfr_fit_result(hfr, i_mag, extra)
            result.figures.update(hfr_result.figures)
            result.tables.update(hfr_result.tables)
            result.scalars.update(hfr_result.scalars)
            result.text_blocks.update(hfr_result.text_blocks)
        if params.get("include_ccd", False):
            ccd = analyze_ccd(curve, params, source_label=source_label,
                              averaged=bool(info.get("averaged", False)))
            result.figures.update(ccd.figures)
            result.tables.update(ccd.tables)
            result.scalars.update(ccd.scalars)
            result.text_blocks.update(ccd.text_blocks)
            result.warnings.extend(ccd.warnings)
        return result

    def exportable_artifacts(self, result: ToolResult) -> list[Artifact]:
        sheets = _export_tables(result)
        artifacts = []
        if sheets:
            artifacts.append(Artifact(
                name="LSV results (Excel)", kind="xlsx",
                default_filename="lsv_results.xlsx", payload=sheets,
            ))
        artifacts.extend(Artifact(
            name=f"{name} (CSV)", kind="csv",
            default_filename=f"lsv_{name.lower().replace(' ', '_')}.csv", payload=table,
        ) for name, table in sheets.items())
        artifacts.extend(item for item in super().exportable_artifacts(result) if item.kind == "png")
        return artifacts


def _export_tables(result: ToolResult) -> dict[str, pd.DataFrame]:
    is_cv = "CV selection" in result.tables
    input_table = "CV scans (per file)" if is_cv else "LSV data (per file)"
    average_table = "Averaged CV scan" if is_cv else "Averaged LSV"
    if input_table in result.tables:
        curves = []
        if average_table in result.tables:
            curves.append(result.tables[average_table].reset_index(drop=True)
                          .add_prefix("Average · "))
        for label, frame in result.tables[input_table].groupby("File", sort=False):
            curves.append(frame.drop(columns="File").reset_index(drop=True)
                          .add_prefix(f"{label} · "))
        data = pd.concat(curves, axis=1)
        sheets = {"CV scans" if is_cv else "LSV curves": data,
                  "Conditions": result.tables["Potential settings"].copy()}
        if is_cv:
            selection_settings = {key: result.scalars[key]
                                  for key in ("Input type", "CV scan", "CV cycle")}
            sheets["Conditions"] = pd.concat([
                sheets["Conditions"],
                pd.DataFrame(selection_settings.items(), columns=["Parameter", "Value"]),
            ], ignore_index=True)
            sheets["CV selection"] = result.tables["CV selection"].copy()
        return sheets

    if "Curves (stacked)" not in result.tables:
        return {}
    data = result.tables["Curves (stacked)"].copy()
    data.rename(columns={
        "E_rev": "Reversible potential (V)",
        "E_rev + η_kin": "After kinetics (V)",
        "+ η_ohm": "After ohmic loss (V)",
        "+ η_RCL": "After catalyst-layer loss (V)",
        "+ η_res": "After residual loss (V)",
        "Original LSV (RHE)": "Measured potential (V vs RHE)",
        "Original LSV (cell voltage)": "Measured cell voltage (V)",
        "Original CV scan (RHE)": "Measured potential (V vs RHE)",
        "Original CV scan (cell voltage)": "Measured cell voltage (V)",
    }, inplace=True)
    if "Catalyst utilization" in result.tables:
        order = np.argsort(np.abs(data.iloc[:, 0].to_numpy()))
        utilization = np.empty(len(order))
        utilization[order] = result.tables["Catalyst utilization"].iloc[:, 1].to_numpy()
        data["Catalyst utilization U_J (-)"] = utilization
    if "HFR points" in result.tables:
        data["HFR (Ω·cm²)"] = result.tables["Overpotential components"]["HFR (Ω·cm²)"].to_numpy()
    fitting = dict(result.tables["Fitting info"].itertuples(index=False, name=None))
    potential = dict(result.tables["Potential settings"].itertuples(index=False, name=None))
    for duplicate in ("E_ref vs SHE (V)", "RHE conversion shift (V)"):
        fitting.pop(duplicate, None)
    fitting.update(potential)
    fitting.update(result.payload.get("export_conditions", {}))
    sections = {"Fit results": pd.DataFrame(fitting.items(), columns=["Parameter", "Value"])}
    if result.payload.get("export_breakdown"):
        sections["Target breakdown"] = pd.DataFrame([result.payload["export_breakdown"]])
    for name in ("CV selection", "HFR points", "HFR settings", "CCD results", "CCD settings"):
        if name in result.tables:
            sections[name] = result.tables[name]
    sheets = {"Summary": summary_table(sections), "Decoupled curves": data}
    for name in ("HFR source data", "HFR fit data", "CCD data"):
        if name in result.tables:
            sheets[name] = result.tables[name].copy()
    return sheets


def _plot_originals(per_file, V_avg, i_avg, is_her=False, averaged=False,
                    potential_label="Potential (V)", source_label="LSV") -> Figure:
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
    title = f"{source_label} originals + average" if averaged else f"{source_label} original"
    ax.set_title(f"{title}  (HER)" if is_her else title)
    ax.legend(fontsize=8, ncol=2, loc="best")
    fig.tight_layout()
    return fig


def _plot_curves(i_data, ys, x_log=False, is_her=False, swap_axes=False,
                 include_rcl=True, is_cell=False, source_label="LSV") -> Figure:
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
            label=source_label)

    if swap_axes:
        ax.set_xlabel(potential_label)
        ax.set_ylabel(current_label)
    else:
        ax.set_xlabel(current_label)
        ax.set_ylabel(potential_label)
    title = (f"Cell {source_label} decoupled into cumulative contributions" if is_cell
             else f"{source_label} decoupled into cumulative contributions")
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
