"""Conductivity-driven CL impedance generation and standard Workbench exports."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from workbench.core.tool_base import Artifact, ParameterSpec, Tool, ToolResult, _slug
from workbench.tools.eis_generator.logic import DEFAULT_PARAMETERS, generate_cl_eis
from workbench.tools.eis_generator.plotting import plot_cl_nyquist


class EisGeneratorTool(Tool):
    id = "eis_generator"
    name = "CL EIS generation"
    subtitle = "Generate blocking CL impedance from effective through-plane conductivities."
    needs_files = False
    files_required = False
    run_button_label = "Generate"

    def parameter_schema(self) -> list[ParameterSpec]:
        def physical(key, label, unit, step, tooltip="", minimum=1e-12):
            return ParameterSpec(key, label, default=DEFAULT_PARAMETERS[key],
                                 minimum=minimum, maximum=1e12, step=step,
                                 suffix=unit, tooltip=tooltip, group="CL properties")
        return [
            physical("electronic_conductivity", "Electronic conductivity (TP)", "S/cm", 0.001,
                     "Effective through-plane electronic conductivity of the hydrated CL."),
            physical("ionic_conductivity", "Ionic conductivity (TP)", "S/cm", 0.001,
                     "Effective through-plane ionic conductivity, not bulk ionomer conductivity."),
            physical("thickness_um", "CL thickness", "µm", 1.0, minimum=1e-9),
            physical("capacitance_mf_cm2", "Areal capacitance", "mF/cm²", 1.0,
                     "Ideal interfacial capacitance per projected electrode area.", minimum=1e-9),
            physical("external_resistance", "External series resistance", "Ω·cm²", 0.01,
                     "Membrane and other series contributions outside the CL. Excludes the CL HFR contribution.", minimum=0.0),
            ParameterSpec("f_min", "Minimum frequency", default=DEFAULT_PARAMETERS["f_min"],
                          minimum=1e-9, maximum=1e12, step=0.1, suffix="Hz", group="Spectrum"),
            ParameterSpec("f_max", "Maximum frequency", default=DEFAULT_PARAMETERS["f_max"],
                          minimum=1e-9, maximum=1e12, step=1000.0, suffix="Hz", group="Spectrum"),
            ParameterSpec("n_points", "Logarithmically spaced points", kind="int", default=DEFAULT_PARAMETERS["n_points"],
                          minimum=2, maximum=20000, step=100, group="Spectrum"),
            ParameterSpec("view", "Nyquist view", kind="choice", default="Transport detail",
                          choices=["Transport detail", "Full frequency range"], group="Display",
                          tooltip="Transport detail magnifies the resistance feature. Full range includes the entire capacitive tail and a transport-detail panel."),
        ]

    def extra_widget(self, parent, on_change):
        from workbench.tools.eis_generator.inputs import ClEisCasesWidget
        return ClEisCasesWidget(parent=parent, on_change=on_change)

    def run(self, files: list[Path], params: dict, extra=None) -> ToolResult:
        view = params.get("view", "Transport detail")
        if view not in {"Transport detail", "Full frequency range"}:
            raise ValueError("Choose Transport detail or Full frequency range.")
        cases = [{"id": "current", "name": "Current", "generated": generate_cl_eis(params)}]
        selected = "current"
        if extra is not None:
            if not isinstance(extra, dict) or not isinstance(extra.get("cases", []), list):
                raise ValueError("Saved CL cases must be a list of parameter snapshots.")
            used_ids = {"current"}
            for case in extra.get("cases", []):
                if not isinstance(case, dict):
                    raise ValueError("Invalid saved CL case.")
                case_id, name = case.get("id"), case.get("name")
                if (not isinstance(case_id, str) or not case_id or case_id in used_ids
                        or not isinstance(name, str) or not name.strip()
                        or not isinstance(case.get("visible"), bool)
                        or not isinstance(case.get("params"), dict)):
                    raise ValueError("Invalid saved CL case identity, visibility or parameters.")
                used_ids.add(case_id)
                if case["visible"]:
                    cases.append({"id": case_id, "name": name.strip(),
                                  "generated": generate_cl_eis(case["params"])})
            selected = extra.get("annotated_case", "current")
            if selected not in {case["id"] for case in cases}:
                selected = "current"

        rows = []
        tables = {}
        for index, case in enumerate(cases, 1):
            generated = case["generated"]
            p, r = generated["parameters"], generated["resistances"]
            spectrum_name = f"Spectrum {index}"
            rows.append({
                "Case": case["name"], "Case ID": case["id"], "Spectrum": spectrum_name,
                "Electronic conductivity TP (S/cm)": p["electronic_conductivity"],
                "Ionic conductivity TP (S/cm)": p["ionic_conductivity"],
                "Thickness (µm)": p["thickness_um"],
                "Areal capacitance (mF/cm²)": p["capacitance_mf_cm2"],
                "External resistance (Ω·cm²)": p["external_resistance"],
                "R_i (Ω·cm²)": r["r_i"], "R_e (Ω·cm²)": r["r_e"],
                "CL HFR contribution (Ω·cm²)": r["hfr_cl"],
                "Total HFR (Ω·cm²)": r["hfr_total"],
                "Width ΔZ′ (Ω·cm²)": r["width"],
                "R_CL,app (Ω·cm²)": r["rcl_app"],
                "LF asymptote intercept (Ω·cm²)": r["lf_intercept"],
                "f_min (Hz)": p["f_min"], "f_max (Hz)": p["f_max"],
                "Points": p["n_points"],
            })
            z = generated["z_total"]
            tables[spectrum_name] = pd.DataFrame({
                "Freq(Hz)": generated["frequency"],
                "Z'(Ohm.cm²)": z.real,
                "Z''(Ohm.cm²)": z.imag,
            })
        tables = {"Resistance summary": pd.DataFrame(rows), **tables}
        annotated = next(case for case in cases if case["id"] == selected)
        r = annotated["generated"]["resistances"]
        return ToolResult(
            figures={"CL Nyquist": plot_cl_nyquist(cases, selected, view)},
            tables=tables,
            scalars={
                "Annotated case": annotated["name"], "Generated cases": len(cases),
                "R_i (Ω·cm²)": r["r_i"], "R_e (Ω·cm²)": r["r_e"],
                "CL HFR contribution (Ω·cm²)": r["hfr_cl"],
                "Total HFR (Ω·cm²)": r["hfr_total"],
                "Width ΔZ′ (Ω·cm²)": r["width"],
                "R_CL,app = 3ΔZ′ (Ω·cm²)": r["rcl_app"],
                "LF asymptote intercept (Ω·cm²)": r["lf_intercept"],
            },
            text_blocks={
                "Model": "Padgett et al. (2023), Eq. 14; homogeneous through-plane, blocking CL with an ideal capacitor.\n"
                         "https://doi.org/10.1149/1945-7111/acee25\n"
                         "Defaults are illustrative, not material-property estimates.\n"
                         "HFR_total = R_external + R_HF,CL.\n"
                         "R_HF,CL + R_CL,app/3 = (R_i + R_e)/3.\n"
                         "Guides are exact model asymptotes; sampled endpoints are finite-frequency values.\n"
                         "The LF vertical-line intercept is not the DC resistance of this blocking electrode.",
                "Display": f"{view}. Generated spectra and exports retain every frequency point.\n"
                           "Exported Z'' is signed; the Nyquist ordinate is −Z''.\n"
                           "No in-plane spreading, faradaic kinetics, wire inductance or operating-overpotential calculation is included.",
            },
        )

    def exportable_artifacts(self, result: ToolResult) -> list[Artifact]:
        summary = result.tables.get("Resistance summary")
        artifacts = []
        if summary is not None:
            artifacts.append(Artifact(
                name="CL EIS results (Excel)", kind="xlsx",
                default_filename="eis_generator_results.xlsx", payload=dict(result.tables),
            ))
            used_names = set()
            used_filenames = set()
            for _, row in summary.iterrows():
                spectrum = result.tables[str(row["Spectrum"])]
                label = row["Case"].strip() if isinstance(row["Case"], str) else ""
                name_base = f"{label} spectrum" if label else "Spectrum"
                display_name = name_base
                suffix = 2
                while display_name.casefold() in used_names:
                    display_name = f"{name_base} ({suffix})"
                    suffix += 1
                used_names.add(display_name.casefold())

                slug = _slug(label) if any(char.isalnum() for char in label) else "spectrum"
                file_base = f"eis_generator_{slug}"
                file_stem = file_base
                suffix = 2
                while file_stem.casefold() in used_filenames:
                    file_stem = f"{file_base}_{suffix}"
                    suffix += 1
                used_filenames.add(file_stem.casefold())
                artifacts.append(Artifact(
                    name=f"{display_name} (CSV)", kind="csv",
                    default_filename=f"{file_stem}.csv",
                    payload=spectrum,
                ))
        artifacts.extend(item for item in super().exportable_artifacts(result) if item.kind == "png")
        return artifacts
