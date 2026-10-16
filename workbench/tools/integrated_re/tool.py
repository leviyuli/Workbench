from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from workbench import theme
from workbench.core.data_io import read_lsv_curve
from workbench.core.potential import PotentialCorrection, REFERENCE_OPTIONS
from workbench.core.tool_base import Artifact, ParameterSpec, Tool, ToolResult
from workbench.tools.integrated_re.exports import plot_tables, results_workbook
from workbench.tools.integrated_re.hfr import extract_role_hfr
from workbench.tools.integrated_re.inputs import ROLES, IntegratedReInputs
from workbench.tools.integrated_re.logic import find_feature, normalize_current, reconstruct_cell
from workbench.tools.lsv.tool import _as_current_density
from workbench.tools.lsv_generator.logic import oer_reversible_potential


@dataclass
class _Curve:
    path: Path
    source_potential: np.ndarray
    source_current: np.ndarray
    raw_current: np.ndarray
    potential: np.ndarray
    current: np.ndarray
    mask: np.ndarray
    info: dict
    correction: PotentialCorrection | None
    multiplier: float
    area: float


def _finite(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a finite number.") from None
    if not np.isfinite(result):
        raise ValueError(f"{label} must be a finite number.")
    return result


class IntegratedReTool(Tool):
    id = "integrated_re_cell"
    name = "Integrated RE for cell"
    subtitle = "Review electrode LSVs versus RHE and reconstruct the cell voltage at matching current density."
    needs_files = False
    files_required = False
    extra_before_parameters = True
    run_button_label = "Review LSVs"

    def parameter_schema(self) -> list[ParameterSpec]:
        calibration = ("correction_mode", ["3E onset", "3E peak"])
        specs = [
            ParameterSpec("electrode_axis", "Electrode X axis", kind="choice",
                          choices=["E vs RHE", "Overpotential"], default="E vs RHE", group="Display"),
            ParameterSpec("include_ir", "Include iR correction", kind="bool", default=False,
                          group="Display",
                          tooltip="Overlay measured and E − j·HFR curves. Requires EIS for both "
                                  "electrodes and, when a cell LSV is loaded, EIS for the cell. "
                                  "Each role uses the mean HFR extracted from its own files."),
            ParameterSpec("reference", "Reference electrode", kind="choice", choices=list(REFERENCE_OPTIONS),
                          default="RHE", group="Integrated reference",
                          tooltip="Reference used by both integrated electrode LSVs. RHE means already converted."),
            ParameterSpec("custom_e_ref_vs_she", "Custom E_ref vs SHE", default=0.0, step=0.001,
                          decimals=5, suffix="V", group="Integrated reference",
                          show_if=("reference", "Custom E_ref vs SHE")),
            ParameterSpec("ph", "pH", default=0.0, minimum=0.0, maximum=14.0,
                          step=0.1, decimals=2, group="Integrated reference"),
            ParameterSpec("temperature", "Temperature", default=298.15, minimum=0.01,
                          step=1.0, decimals=2, suffix="K", group="Integrated reference"),
            ParameterSpec("correction_mode", "Reference correction", kind="choice",
                          choices=["None", "Manual", "3E onset", "3E peak"], default="None",
                          group="Reference correction",
                          tooltip="3E correction matches iR-corrected electrode curves and requires "
                                  "EIS for the electrode to calibrate. The bundled PtRuC cathode "
                                  "calibration is already iR corrected, in 1 M KOH at 298 K, "
                                  "with an onset current of 0.05 A/cm²."),
            ParameterSpec("manual_shift", "Manual offset / fine adjustment", default=0.0,
                          step=0.01, decimals=5, suffix="V", group="Reference correction",
                          tooltip="Added equally to both integrated electrodes. In 3E modes this is "
                                  "added to the calculated offset; in Manual mode it is the full offset.",
                          show_if=("correction_mode", ["Manual", "3E onset", "3E peak"])),
            ParameterSpec("calculated_shift", "Calculated 3E offset", kind="str",
                          default="Not calculated", suffix="V", read_only=True,
                          group="Reference correction", show_if=calibration,
                          tooltip="Automatic offset from the last successful 3E calculation, "
                                  "before manual fine adjustment. Display only; recomputed on every run."),
            ParameterSpec("applied_shift", "Applied shared offset", kind="str",
                          default="Not calculated", suffix="V", read_only=True,
                          group="Reference correction",
                          show_if=("correction_mode", ["Manual", "3E onset", "3E peak"]),
                          tooltip="Total offset from the last successful run, including manual fine "
                                  "adjustment. Applied equally to the integrated anode and cathode."),
            ParameterSpec("calibration_electrode", "Electrode to calibrate", kind="choice",
                          choices=["Cathode", "Anode"], default="Cathode",
                          group="Reference correction", show_if=calibration),
            ParameterSpec("calibration_reference", "3E reference electrode", kind="choice",
                          choices=list(REFERENCE_OPTIONS), default="Hg/HgO (1 M KOH, 25 °C)",
                          group="Reference correction", show_if=calibration),
            ParameterSpec("calibration_custom_e_ref_vs_she", "3E custom E_ref vs SHE", default=0.0,
                          step=0.001, decimals=5, suffix="V", group="Reference correction",
                          show_if=[calibration, ("calibration_reference", "Custom E_ref vs SHE")]),
            ParameterSpec("calibration_ph", "3E pH", default=14.0, minimum=0.0, maximum=14.0,
                          step=0.1, decimals=2, group="Reference correction", show_if=calibration),
            ParameterSpec("calibration_temperature", "3E temperature", default=298.0,
                          minimum=0.01, step=1.0, decimals=2, suffix="K",
                          group="Reference correction", show_if=calibration),
            ParameterSpec("onset_current", "Onset |j|", default=0.05, minimum=1e-12,
                          step=0.001, decimals=5, suffix="A/cm²", group="Reference correction",
                          tooltip="Match the iR-corrected reaction branches at this current density.",
                          show_if=("correction_mode", "3E onset")),
        ]
        for prefix, label in [("integrated", "Integrated peak"), ("calibration", "3E peak")]:
            for bound, default in [("min", -1.0), ("max", 1.0)]:
                specs.append(ParameterSpec(
                    f"{prefix}_peak_{bound}", f"{label} E {bound}", default=default,
                    step=0.05, decimals=4, suffix="V vs RHE", group="Reference correction",
                    tooltip="Search iR-corrected potentials for an interior reaction-current peak "
                            "before the shared reference correction.",
                    show_if=("correction_mode", "3E peak"),
                ))
        return specs

    def extra_widget(self, parent, on_change):
        return IntegratedReInputs(parent, on_change)

    def exportable_artifacts(self, result: ToolResult) -> list[Artifact]:
        tables = plot_tables(result)
        items = [Artifact("Plot data (Excel)", "workbook", f"{self.id}_results.xlsx",
                          results_workbook(result, tables))]
        for name, frame in tables.items():
            items.append(Artifact(f"{name} (CSV)", "csv",
                                  f"{self.id}_{name.lower().replace(' ', '_')}.csv", frame))
        if "Calibration features" in result.tables:
            features = result.tables["Calibration features"][[
                "Source", "E after correction (V vs RHE)", "Current density (A/cm²)",
            ]]
            items.append(Artifact("Calibration matches (CSV)", "csv",
                                  f"{self.id}_calibration_matches.csv", features))
        if result.warnings:
            items.append(Artifact("Run warnings (TXT)", "txt", f"{self.id}_warnings.txt",
                                  "\n".join(result.warnings) + "\n"))
        items.extend(super().exportable_artifacts(ToolResult(
            figures=result.figures, figure_images=result.figure_images)))
        return items

    def _load_curve(self, role, path, params, *, calibration_role=None):
        label = ROLES[role]
        data, error = read_lsv_curve(str(path))
        if error:
            raise ValueError(f"{label}: {error}")
        potential, raw_current, info = data
        usable = np.isfinite(potential) & np.isfinite(raw_current)
        if np.count_nonzero(usable) < 2:
            raise ValueError(f"{label}: need at least two finite numeric points.")
        area = _finite(params.get(f"{role}_area", 0.0), f"{label} electrode area")
        try:
            source_current, area_warning = _as_current_density(raw_current, info, area)
            current, multiplier = normalize_current(
                source_current, calibration_role if role == "calibration" else role,
                params.get(f"{role}_sign", "Auto"),
            )
            if role == "cell":
                correction = None
                converted = potential.copy()
            else:
                correction_params = params
                if role == "calibration":
                    correction_params = {
                        key: params.get(f"calibration_{key}", default)
                        for key, default in [("reference", "Hg/HgO (1 M KOH, 25 °C)"), ("ph", 14.0),
                                             ("temperature", 298.0), ("custom_e_ref_vs_she", 0.0)]
                    }
                correction = PotentialCorrection.from_params({
                    **correction_params, "electrode_setup": "Three-electrode", "target_scale": "RHE",
                }, default_target="RHE")
                converted = correction.apply(potential)
            mask = usable.copy()
            if params.get(f"{role}_use_window", False):
                low = _finite(params.get(f"{role}_e_min", -2.0), f"{label} window minimum")
                high = _finite(params.get(f"{role}_e_max", 3.0), f"{label} window maximum")
                if low >= high:
                    raise ValueError("window minimum must be less than its maximum.")
                mask &= (converted >= low) & (converted <= high)
                if not np.any(mask):
                    raise ValueError("no points in the selected comparison window.")
        except ValueError as exc:
            raise ValueError(f"{label}: {exc}") from exc
        warnings = []
        if not np.all(usable):
            warnings.append(f"{label}: {int(np.count_nonzero(~usable))} invalid measurement row(s) "
                            "retained as gaps; interpolation does not cross these rows.")
        if info.get("warning") and info.get("is_density", True):
            warnings.append(f"{label}: {info['warning']}")
        if area_warning:
            warnings.append(f"{label}: {area_warning}")
        if multiplier != 1:
            warnings.append(f"{label}: reversed the source current sign for the selected convention.")
        if correction:
            warnings.extend(f"{label}: {notice}" for notice in correction.warnings)
        return _Curve(Path(path), potential, source_current, raw_current, converted, current,
                      mask, info, correction, multiplier, area), warnings

    def run(self, files: list[Path], params: dict, extra=None) -> ToolResult:
        inputs = extra or {}
        params = {**params, **inputs.get("parameters", {})}
        mode = params.get("correction_mode", "None")
        if mode not in {"None", "Manual", "3E onset", "3E peak"}:
            raise ValueError(f"Unknown reference correction: {mode!r}")
        axis = params.get("electrode_axis", "E vs RHE")
        if axis not in {"E vs RHE", "Overpotential"}:
            raise ValueError(f"Unknown electrode plot scale: {axis!r}")
        chosen = params.get("calibration_electrode", "Cathode")
        if mode.startswith("3E") and chosen not in {"Anode", "Cathode"}:
            raise ValueError("Select Anode or Cathode for calibration.")
        chosen_role = chosen.lower()
        curves, warnings = {}, []
        for role in ("anode", "cathode", "cell", "calibration"):
            path = inputs.get(role)
            required = role in {"anode", "cathode"} or (role == "calibration" and mode.startswith("3E"))
            if role == "calibration" and not required:
                continue
            if not path:
                if required:
                    raise ValueError(f"Add the {ROLES[role]} input first.")
                continue
            curves[role], notices = self._load_curve(role, path, params, calibration_role=chosen_role)
            warnings.extend(notices)

        include_ir = bool(params.get("include_ir", False))
        eis_inputs = inputs.get("eis", {}) or {}
        hfrs, hfr_rows, hfr_spectra, hfr_summary = {}, [], [], []
        hfr_annotations = {}
        for role in ("anode", "cathode", "cell"):
            eis_files = eis_inputs.get(role, []) or []
            if isinstance(eis_files, (str, Path)):
                eis_files = [eis_files]
            mean, estimates, spectra, notices = extract_role_hfr(
                eis_files, area_cm2=params.get(f"{role}_area", 0.0), role=role,
            )
            hfrs[role] = mean
            warnings.extend(notices)
            used = int((estimates["Status"] == "OK").sum()) if not estimates.empty else 0
            hfr_summary.append({"Role": role, "Mean HFR (Ω·cm²)": mean if mean is not None else np.nan,
                                "Spectra used": used, "Spectra uploaded": len(eis_files)})
            hfr_annotations[f"{role}_hfr"] = f"{used}/{len(eis_files)} EIS"
            hfr_rows.append(estimates)
            hfr_spectra.append(spectra)
        if mode.startswith("3E") and hfrs[chosen_role] is None:
            raise ValueError(f"{chosen}: add {chosen} EIS data to calculate HFR for 3E reference "
                             "correction. The measured electrode LSV must be iR corrected "
                             "before matching the already iR-corrected calibration LSV.")
        if include_ir:
            for role in ("anode", "cathode", *(["cell"] if "cell" in curves else [])):
                if hfrs[role] is None:
                    raise ValueError(f"{ROLES[role]}: add EIS data to calculate HFR for iR correction, "
                                     "or turn off Include iR correction.")

        temperature = curves["anode"].correction.temperature
        erev = oer_reversible_potential(temperature)
        shift = 0.0 if mode == "None" else _finite(params.get("manual_shift", 0.0), "Shared offset")
        automatic_shift = 0.0
        features = None
        if mode.startswith("3E"):
            selected, calibration = curves[chosen_role], curves["calibration"]
            calibration_integrated = replace(
                selected, potential=selected.potential - selected.current * hfrs[chosen_role],
            )
            select = "min" if chosen_role == "cathode" else "max"
            orientation = -1.0 if chosen_role == "cathode" else 1.0
            method = "onset" if mode == "3E onset" else "peak"
            kwargs = {"select": select}
            if method == "onset":
                kwargs["target_current"] = _finite(params.get("onset_current", 0.05), "Onset current")
                integrated_potential = np.where(selected.mask, calibration_integrated.potential, np.nan)
            else:
                integrated_potential = calibration_integrated.potential
            matches = []
            for prefix, curve, potential in [("integrated", selected, integrated_potential),
                                             ("calibration", calibration, calibration.potential)]:
                if method == "peak":
                    kwargs["window"] = (
                        _finite(params.get(f"{prefix}_peak_min", -1.0), f"{prefix} peak minimum"),
                        _finite(params.get(f"{prefix}_peak_max", 1.0), f"{prefix} peak maximum"),
                    )
                try:
                    feature = find_feature(potential, orientation * curve.current, method, **kwargs)
                except ValueError as exc:
                    label = "Integrated electrode" if prefix == "integrated" else "3E calibration LSV"
                    raise ValueError(f"{label}: {exc}") from exc
                matches.append(feature)
            automatic_shift = matches[1]["potential"] - matches[0]["potential"]
            shift += automatic_shift
            features = pd.DataFrame([
                {"Source": source, "File": curve.path.name, "Method": method,
                 "E before correction (V vs RHE)": feature["potential"],
                 "E after correction (V vs RHE)": feature["potential"] + (shift if source == "Integrated" else 0),
                 "Current density (A/cm²)": orientation * feature["current"],
                 "Prominence (A/cm²)": feature.get("prominence", np.nan)}
                for source, curve, feature in [("Integrated", selected, matches[0]),
                                                ("Three-electrode", calibration, matches[1])]
            ])

        anode, cathode = curves["anode"], curves["cathode"]
        cell = curves.get("cell")
        reconstructed, comparison, notices = reconstruct_cell(
            anode.potential + shift, anode.current, cathode.potential + shift, cathode.current,
            anode_mask=anode.mask, cathode_mask=cathode.mask,
            cell_potential=cell.potential if cell else None,
            cell_current=cell.current if cell else None, cell_mask=cell.mask if cell else None,
            erev=erev,
        )
        warnings.extend(notices)
        corrected = _ir_reconstruction(reconstructed, hfrs) if include_ir else None
        corrected_comparison = None
        if include_ir and cell:
            corrected_comparison = _ir_reconstruction(comparison, hfrs)
            j = corrected_comparison["Current density (A/cm²)"].to_numpy()
            corrected_comparison["Measured cell voltage (V)"] = comparison["Measured cell voltage (V)"] - j * hfrs["cell"]
            corrected_comparison["Voltage difference (V)"] = (
                corrected_comparison["Reconstructed cell voltage (V)"]
                - corrected_comparison["Measured cell voltage (V)"]
            )
            corrected_comparison["Cell HFR (Ω·cm²)"] = hfrs["cell"]
            corrected_comparison["Measured cell iR drop (V)"] = j * hfrs["cell"]
        figures = {"Electrode LSVs": _plot_electrodes(
                        anode, cathode, shift, axis, erev, hfrs=hfrs if include_ir else None),
                   "Cell comparison": _plot_comparison(
                        reconstructed, cell, corrected=corrected, cell_hfr=hfrs["cell"] if include_ir else None)}
        tables = {"Anode LSV": _curve_table(anode, "anode", shift, erev, hfr=hfrs["anode"], include_ir=include_ir),
                  "Cathode LSV": _curve_table(cathode, "cathode", shift, erev, hfr=hfrs["cathode"], include_ir=include_ir),
                  "Reconstructed cell": reconstructed}
        if cell:
            tables["Cell LSV"] = _curve_table(cell, "cell", 0.0, erev, hfr=hfrs["cell"], include_ir=include_ir)
            tables["Cell comparison"] = comparison
        if corrected is not None:
            tables["Reconstructed cell (iR corrected)"] = corrected
        if corrected_comparison is not None:
            tables["Cell comparison (iR corrected)"] = corrected_comparison
        tables["HFR per spectrum"] = pd.concat(hfr_rows, ignore_index=True)
        tables["HFR summary"] = pd.DataFrame(hfr_summary)
        tables["EIS spectra"] = pd.concat(hfr_spectra, ignore_index=True)
        if features is not None:
            tables["Calibration integrated LSV"] = _curve_table(
                calibration_integrated, chosen_role, shift, erev,
            )
            tables["Calibration LSV"] = _curve_table(curves["calibration"], chosen_role, 0.0, erev)
            tables["Calibration features"] = features
            figures["Calibration alignment"] = _plot_calibration(
                calibration_integrated, curves["calibration"], shift, features, chosen,
            )

        metadata = {"Correction method": mode, "Shared correction (V)": shift,
                    "Automatic correction (V)": automatic_shift,
                    "Manual adjustment (V)": 0.0 if mode == "None" else params.get("manual_shift", 0.0),
                    "Electrode plot scale": axis, "Reversible cell voltage (V)": erev,
                    "Include iR correction": include_ir,
                    "HFR extraction": "Per-file high-frequency x-axis intersection; minimum Re(Z) fallback",
                    "HFR averaging": "Arithmetic mean of valid per-spectrum HFR estimates, independently per role",
                    "iR correction formula": "E_corrected = E - signed j * HFR; HFR in Ω·cm², j in A/cm²",
                    "Reconstructed iR drop": "|j| * (anode HFR + cathode HFR)",
                    "Calibration basis": ("iR-corrected measured electrode and already iR-corrected 3E LSV"
                                          if features is not None else "No 3E calibration"),
                    "Comparison voltage definition": "E_anode(RHE) - E_cathode(RHE)",
                    "Voltage difference definition": "Reconstructed minus measured cell voltage"}
        for row in hfr_summary:
            role = row["Role"]
            metadata[f"{role} mean HFR (Ω·cm²)"] = row["Mean HFR (Ω·cm²)"]
            metadata[f"{role} EIS spectra used"] = row["Spectra used"]
            metadata[f"{role} EIS spectra uploaded"] = row["Spectra uploaded"]
        for role, curve in curves.items():
            prefix = ROLES[role]
            metadata.update({f"{prefix} file": str(curve.path),
                             f"{prefix} source potential column": curve.info.get("v_col", "E"),
                             f"{prefix} source current column": curve.info.get("i_col", "i"),
                             f"{prefix} current multiplier": curve.multiplier,
                             f"{prefix} electrode area (cm²)": curve.area,
                             f"{prefix} comparison window enabled": bool(params.get(f"{role}_use_window", False))})
            if curve.correction:
                metadata.update({f"{prefix}: {key}": value for key, value in curve.correction.metadata().items()})
            if params.get(f"{role}_use_window", False):
                metadata[f"{prefix} window min (V)"] = params.get(f"{role}_e_min")
                metadata[f"{prefix} window max (V)"] = params.get(f"{role}_e_max")
        if features is not None:
            metadata["Calibration electrode"] = chosen
            if mode == "3E onset":
                metadata["Onset |j| (A/cm²)"] = params.get("onset_current", 0.05)
            else:
                for prefix in ("integrated", "calibration"):
                    for bound in ("min", "max"):
                        metadata[f"{prefix} peak E {bound} (V vs RHE)"] = params.get(f"{prefix}_peak_{bound}", -1.0 if bound == "min" else 1.0)
        tables["Potential settings"] = pd.DataFrame(metadata.items(), columns=["Parameter", "Value"])
        scalars = {"Shared correction (V)": shift, "Reversible cell voltage (V)": erev,
                   "Reconstructed points": len(reconstructed), "Compared points": len(comparison),
                   "Current range (A/cm²)": f"{reconstructed.iloc[0, 0]:.6g} – {reconstructed.iloc[-1, 0]:.6g}"}
        if not comparison.empty:
            differences = comparison["Voltage difference (V)"].to_numpy()
            scalars["Mean voltage difference (V)"] = float(np.mean(differences))
            scalars["Maximum |voltage difference| (V)"] = float(np.max(np.abs(differences)))
        if corrected_comparison is not None and not corrected_comparison.empty:
            scalars["Mean iR-corrected voltage difference (V)"] = float(
                corrected_comparison["Voltage difference (V)"].mean())
        text_blocks = {"Integrated RE review":
                       "Anode and cathode are compared at matching |j| without extrapolation. "
                       "A shared reference offset shifts both electrodes equally and cancels "
                       "from the reconstructed cell voltage. "
                       + ("Dashed curves remove signed j times each role's mean HFR; solid curves retain iR losses."
                          if include_ir else "Curves retain measured iR losses.")}
        details = [{"Kind": "Result", "Name": key, "Value": value} for key, value in scalars.items()]
        details += [{"Kind": "Note", "Name": key, "Value": value} for key, value in text_blocks.items()]
        details += [{"Kind": "Warning", "Name": f"Warning {number}", "Value": warning}
                    for number, warning in enumerate(warnings, start=1)]
        tables["Run details"] = pd.DataFrame(details)
        updates = {"calculated_shift": f"{automatic_shift:+.6f}", "applied_shift": f"{shift:+.6f}"}
        updates.update({f"{role}_hfr": f"{value:.6g}" if value is not None else "Not available"
                        for role, value in hfrs.items()})
        return ToolResult(figures=figures, tables=tables, scalars=scalars, warnings=warnings,
                          param_updates=updates, param_annotations=hfr_annotations,
                          text_blocks=text_blocks, display_tables=False, display_summary=False)


def _curve_table(curve, role, shift, erev, *, hfr=None, include_ir=False):
    data = {
        "File": curve.path.name,
        "Source potential (original unit)": curve.source_potential / curve.info.get("v_scale", 1.0),
        "Source current (original unit)": curve.raw_current / curve.info.get("i_scale", 1.0),
        "Source potential (V)": curve.source_potential,
        "Source current (A/cm²)": curve.source_current,
        "Current density (A/cm²)": curve.current,
        "In comparison window": curve.mask,
    }
    if role == "cell":
        data["Cell voltage (V)"] = curve.potential
        data["Overpotential (V)"] = curve.potential - erev
    else:
        data["E before correction (V vs RHE)"] = curve.potential
        data["E (V vs RHE)"] = curve.potential + shift
        data["Overpotential (V)"] = curve.potential + shift - (erev if role == "anode" else 0.0)
    if hfr is not None:
        data["HFR (Ω·cm²)"] = hfr
        if include_ir:
            drop = curve.current * hfr
            data["iR drop (V)"] = drop
            data["iR-corrected overpotential (V)"] = data["Overpotential (V)"] - drop
            key = "iR-corrected cell voltage (V)" if role == "cell" else "iR-corrected E (V vs RHE)"
            data[key] = curve.potential + shift - drop
    return pd.DataFrame(data)


def _ir_reconstruction(original, hfrs):
    corrected = original.copy()
    j = original["Current density (A/cm²)"].to_numpy()
    anode_drop, cathode_drop = j * hfrs["anode"], j * hfrs["cathode"]
    corrected["Anode E (V vs RHE)"] -= anode_drop
    corrected["Cathode E (V vs RHE)"] += cathode_drop
    corrected["Anode η (V)"] -= anode_drop
    corrected["Cathode η (V)"] += cathode_drop
    corrected["Reconstructed cell voltage (V)"] -= anode_drop + cathode_drop
    corrected["Cell overpotential (V)"] -= anode_drop + cathode_drop
    corrected["Anode HFR (Ω·cm²)"] = hfrs["anode"]
    corrected["Cathode HFR (Ω·cm²)"] = hfrs["cathode"]
    corrected["Electrode iR drop (V)"] = anode_drop + cathode_drop
    return corrected


def _plot_electrodes(anode, cathode, shift, axis, erev, *, hfrs=None):
    fig = Figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    potentials = []
    for role, curve, color in [("Anode", anode, theme.DANGER), ("Cathode", cathode, theme.SERIES_COLORS[1])]:
        potential = curve.potential + shift - (erev if role == "Anode" and axis == "Overpotential" else 0)
        usable = np.isfinite(potential) & np.isfinite(curve.current)
        potentials.append(potential[usable])
        ax.plot(np.where(usable, potential, np.nan), np.where(usable, curve.current, np.nan),
                color=color, linewidth=1.7,
                label=f"{role} · {curve.path.name}")
        if hfrs is not None:
            resistance = hfrs[role.lower()]
            corrected = potential - curve.current * resistance
            potentials.append(corrected[usable])
            ax.plot(np.where(usable, corrected, np.nan), np.where(usable, curve.current, np.nan),
                    color=color, linewidth=1.6, linestyle="--",
                    label=f"{role} iR corrected · HFR {resistance:.4g} Ω·cm²")
    ax.axhline(0, color=theme.MUTED, linewidth=0.8)
    ax.axvline(0, color=theme.MUTED, linewidth=0.9, linestyle="--",
               label="Equilibrium" if axis == "Overpotential" else "HER equilibrium")
    if axis == "E vs RHE":
        ax.axvline(erev, color=theme.MUTED, linewidth=0.9, linestyle=":", label="OER equilibrium")
    extent = np.concatenate(potentials)
    low = min(float(np.min(extent)), 0.0)
    high = max(float(np.max(extent)), erev if axis == "E vs RHE" else 0.0)
    margin = max(0.04 * (high - low), 0.02)
    ax.set_xlim(low - margin, high + margin)
    ax.set_xlabel("Potential E (V vs RHE)" if axis == "E vs RHE" else "Overpotential η (V)")
    ax.set_ylabel("Current density j (A/cm²)")
    ax.set_title("Integrated RE · electrode LSVs", loc="left")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _plot_comparison(reconstructed, cell, *, corrected=None, cell_hfr=None):
    fig = Figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.plot(reconstructed["Reconstructed cell voltage (V)"], reconstructed["Current density (A/cm²)"],
            color=theme.OK, linewidth=1.8, marker="o", markersize=2.5,
            label="Reconstructed · Integrated RE")
    if cell:
        mask = cell.current >= 0
        ax.plot(np.where(mask, cell.potential, np.nan), np.where(mask, cell.current, np.nan),
                color=theme.TEXT, linewidth=1.5, label=f"Measured cell · {cell.path.name}")
    if corrected is not None:
        ax.plot(corrected["Reconstructed cell voltage (V)"], corrected["Current density (A/cm²)"],
                color=theme.OK, linewidth=1.8, marker="o", markersize=2.5, linestyle="--",
                label="Reconstructed · iR corrected")
        if cell is not None and cell_hfr is not None:
            corrected_cell = cell.potential - cell.current * cell_hfr
            ax.plot(np.where(mask, corrected_cell, np.nan), np.where(mask, cell.current, np.nan),
                    color=theme.TEXT, linewidth=1.5, linestyle="--",
                    label=f"Measured cell · iR corrected (HFR {cell_hfr:.4g} Ω·cm²)")
    ax.set_xlabel("Cell voltage (V)")
    ax.set_ylabel("Current density |j| (A/cm²)")
    ax.set_ylim(bottom=0)
    ax.set_title("Integrated RE · cell comparison", loc="left")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _plot_calibration(integrated, calibration, shift, features, electrode):
    fig = Figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.plot(integrated.potential, integrated.current, color=theme.MUTED, linestyle="--",
            linewidth=1.2, label="Integrated · iR corrected")
    ax.plot(integrated.potential + shift, integrated.current, color=theme.ACCENT,
            linewidth=1.7, label="Integrated · iR + RE corrected")
    ax.plot(calibration.potential, calibration.current, color=theme.SERIES_COLORS[1],
            linewidth=1.7, label=f"3E · iR corrected · {calibration.path.name}")
    ax.scatter(features["E after correction (V vs RHE)"], features["Current density (A/cm²)"],
               color=[theme.ACCENT, theme.SERIES_COLORS[1]], s=45, zorder=3, label="Matched features")
    ax.set_xlabel("Potential E (V vs RHE)")
    ax.set_ylabel("Current density j (A/cm²)")
    ax.set_title(f"{electrode} calibration · iR corrected · shared shift {shift:+.5f} V", loc="left")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
