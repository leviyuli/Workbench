from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from workbench.core.tool_base import Artifact, ParameterSpec, Tool, ToolResult
from workbench.tools.ftir.logic import (
    apply_vertical_offset,
    convert_t_to_a,
    normalize_spectrum,
    plot_ftir_spectra,
    read_ftir_file,
)


class FtirTool(Tool):
    id = "ftir"
    name = "FTIR Display"
    subtitle = "Display processed FTIR spectra as transmittance or absorbance with normalization and offsets."
    needs_files = True
    multiple_files = True
    accepts_directory = False
    accepted_file_types = (".csv", ".txt", ".dat")
    run_button_label = "Plot"

    def parameter_schema(self) -> list[ParameterSpec]:
        return [
            ParameterSpec(
                "display_mode", "Display mode", kind="choice",
                choices=["T (%)", "A"], default="T (%)", group="Display",
            ),
            ParameterSpec(
                "normalization", "Normalization", kind="choice",
                choices=["None", "Maximum normalization", "Min-max normalization"],
                default="None", group="Display",
            ),
            ParameterSpec(
                "curve_names", "Curve names", kind="str", default="",
                tooltip="Optional comma-separated names. Missing names use file names.",
                group="Display",
            ),
            ParameterSpec(
                "line_width", "Line width", kind="float", default=1.8,
                minimum=0.1, maximum=12.0, step=0.1, decimals=2, group="Style",
            ),
            ParameterSpec(
                "vertical_offset", "Vertical offset", kind="float", default=0.0,
                step=0.05, decimals=4, group="Style",
            ),
            ParameterSpec(
                "legend_position", "Legend", kind="choice",
                choices=[
                    "Best", "Upper right", "Upper left", "Lower right",
                    "Lower left", "Outside right", "Hidden",
                ],
                default="Best", group="Style",
            ),
            ParameterSpec(
                "hide_y_ticks", "Hide y tick labels", kind="bool",
                default=False, group="Style",
            ),
        ]

    def run(self, files: list[Path], params: dict, extra=None) -> ToolResult:
        if not files:
            raise ValueError("Add at least one FTIR file.")

        display_mode = str(params.get("display_mode", "T (%)"))
        normalization = str(params.get("normalization", "None"))
        offset = float(params.get("vertical_offset", 0.0))
        line_width = float(params.get("line_width", 1.8))
        legend_position = str(params.get("legend_position", "Best"))
        hide_y_ticks = bool(params.get("hide_y_ticks", False))
        name_overrides = _curve_names(str(params.get("curve_names", "")))

        warnings: list[str] = []
        curves: list[dict] = []
        plot_rows: list[pd.DataFrame] = []
        meta_rows: list[dict] = []

        for file_index, path in enumerate(files):
            spec = read_ftir_file(path)
            warnings.extend(spec.warnings)
            curve_name = (
                name_overrides[file_index]
                if file_index < len(name_overrides) and name_overrides[file_index]
                else spec.path.stem
            )

            y_raw = spec.intensity.copy()
            source_for_display = spec.source_type
            if display_mode.startswith("T"):
                if spec.source_type != "T":
                    warnings.append(f"{spec.path.name}: skipped because T (%) mode requires T% data.")
                    continue
                y_display_base = y_raw
            else:
                if spec.source_type == "A":
                    y_display_base = y_raw
                elif spec.source_type == "T":
                    invalid = int(np.sum(~np.isfinite(y_raw) | (y_raw <= 0)))
                    if invalid:
                        warnings.append(
                            f"{spec.path.name}: {invalid} nonpositive/invalid T% value(s) converted to NaN."
                        )
                    y_display_base = convert_t_to_a(y_raw)
                    source_for_display = "T -> A"
                else:
                    y_display_base = y_raw
                    warnings.append(f"{spec.path.name}: unknown intensity type; plotted directly as A.")

            y_norm = normalize_spectrum(y_display_base, normalization)
            if normalization != "None" and np.array_equal(y_norm, y_display_base, equal_nan=True):
                finite = y_display_base[np.isfinite(y_display_base)]
                if len(finite) == 0 or float(np.nanmax(finite)) == float(np.nanmin(finite)):
                    warnings.append(f"{spec.path.name}: normalization left a flat/empty spectrum unchanged.")

            curve_number = len(curves)
            y_offset = apply_vertical_offset(y_norm, curve_number, offset)
            curve = {
                "curve": curve_name,
                "source_file": spec.path.name,
                "wavenumber": spec.wavenumber,
                "y_before_offset": y_norm,
                "y_display": y_offset,
                "source_type": source_for_display,
            }
            curves.append(curve)

            plot_rows.append(pd.DataFrame({
                "Wavenumber (cm-1)": spec.wavenumber,
                "Curve": curve_name,
                "Y display": y_offset,
                "Y before offset": y_norm,
                "Source file": spec.path.name,
                "Detected source type": source_for_display,
            }))
            meta_rows.append({
                "Source file": spec.path.name,
                "Curve": curve_name,
                "Wavenumber column": spec.x_column,
                "Intensity column": spec.y_column,
                "Detected source type": spec.source_type,
                "Rows plotted": int(np.isfinite(y_offset).sum()),
            })

        if not curves:
            raise ValueError("No FTIR spectra could be plotted with the selected display mode.")

        plot_df = pd.concat(plot_rows, ignore_index=True)
        meta_df = pd.DataFrame(meta_rows)
        fig = plot_ftir_spectra(
            curves, display_mode, normalization, line_width, legend_position, hide_y_ticks,
        )

        scalars = {
            "Spectra plotted": len(curves),
            "Display mode": display_mode,
            "Normalization": normalization,
            "Vertical offset": offset,
        }
        return ToolResult(
            figures={"FTIR spectra": fig},
            tables={"FTIR plot data": plot_df, "FTIR metadata": meta_df},
            scalars=scalars,
            warnings=warnings,
            payload={"plot_data": plot_df, "metadata": meta_df},
        )

    def exportable_artifacts(self, result: ToolResult) -> list[Artifact]:
        plot_df = result.payload.get("plot_data")
        if plot_df is None:
            plot_df = result.tables.get("FTIR plot data")
        meta_df = result.payload.get("metadata")
        if meta_df is None:
            meta_df = result.tables.get("FTIR metadata")
        if plot_df is None:
            return []
        sheets = {"FTIR plot data": plot_df}
        if meta_df is not None:
            sheets["FTIR metadata"] = meta_df
        return [
            Artifact(
                name="FTIR plot data (CSV)", kind="csv",
                default_filename="ftir_plot_data.csv", payload=plot_df,
            ),
            Artifact(
                name="FTIR plot data (Excel)", kind="xlsx",
                default_filename="ftir_plot_data.xlsx", payload=sheets,
            ),
        ]


def _curve_names(text: str) -> list[str]:
    if not text.strip():
        return []
    return [part.strip() for part in text.split(",")]
