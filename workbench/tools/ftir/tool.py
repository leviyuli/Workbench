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
    read_spectrum_file,
)


class FtirTool(Tool):
    id = "ftir"
    name = "FTIR / XPS Display"
    subtitle = "Stack processed FTIR or XPS spectra with technique-specific axes and a display-only x range."
    needs_files = True
    multiple_files = True
    accepts_directory = False
    accepted_file_types = (".csv", ".txt", ".dat")
    run_button_label = "Plot"

    def parameter_schema(self) -> list[ParameterSpec]:
        return [
            ParameterSpec(
                "technique", "Technique", kind="choice",
                choices=["FTIR", "XPS"], default="FTIR", group="Display",
            ),
            ParameterSpec(
                "display_mode", "Display mode", kind="choice",
                choices=["T", "A"], default="T", group="Display",
                show_if=("technique", "FTIR"),
            ),
            ParameterSpec(
                "normalization", "Normalization", kind="choice",
                choices=["None", "Maximum normalization", "Min-max normalization"],
                default="Min-max normalization", group="Display",
            ),
            ParameterSpec(
                "vertical_offset", "Vertical offset", kind="float", default=1.0,
                step=0.05, decimals=4, group="Display",
            ),
            ParameterSpec(
                "hide_y_ticks", "Hide y tick labels", kind="bool",
                default=False, group="Display",
            ),
            ParameterSpec(
                "custom_x_range", "Custom x-axis range", kind="bool",
                default=False, group="X axis",
                tooltip="Changes the plot view only; exported data retains the full input range.",
            ),
            ParameterSpec(
                "x_display_min", "Minimum displayed x", kind="float",
                default=0.0, step=10.0, decimals=4, group="X axis",
                show_if=("custom_x_range", True),
            ),
            ParameterSpec(
                "x_display_max", "Maximum displayed x", kind="float",
                default=4000.0, step=10.0, decimals=4, group="X axis",
                show_if=("custom_x_range", True),
            ),
        ]

    def run(self, files: list[Path], params: dict, extra=None) -> ToolResult:
        if not files:
            raise ValueError("Add at least one FTIR or XPS file.")

        technique = str(params.get("technique", "FTIR")).upper()
        if technique not in {"FTIR", "XPS"}:
            raise ValueError(f"Unknown technique: {technique}")
        display_mode = str(params.get("display_mode", "T"))
        normalization = str(params.get("normalization", "Min-max normalization"))
        offset = float(params.get("vertical_offset", 1.0))
        hide_y_ticks = bool(params.get("hide_y_ticks", False))
        custom_x_range = bool(params.get("custom_x_range", False))
        x_display_min = float(params.get("x_display_min", 0.0)) if custom_x_range else None
        x_display_max = float(params.get("x_display_max", 4000.0)) if custom_x_range else None
        if custom_x_range and x_display_min >= x_display_max:
            raise ValueError("Minimum displayed x must be smaller than maximum displayed x.")

        warnings: list[str] = []
        curves: list[dict] = []
        plot_rows: list[pd.DataFrame] = []
        meta_rows: list[dict] = []

        for path in files:
            spec = read_spectrum_file(path, technique=technique)
            warnings.extend(spec.warnings)
            curve_name = spec.path.stem

            y_raw = spec.intensity.copy()
            source_for_display = spec.source_type
            if technique == "XPS":
                y_display_base = y_raw
            elif display_mode.startswith("T"):
                if spec.source_type != "T":
                    warnings.append(f"{spec.path.name}: skipped because T mode requires T data.")
                    continue
                y_display_base = y_raw
            else:
                if spec.source_type == "A":
                    y_display_base = y_raw
                elif spec.source_type == "T":
                    invalid = int(np.sum(~np.isfinite(y_raw) | (y_raw <= 0)))
                    if invalid:
                        warnings.append(
                            f"{spec.path.name}: {invalid} nonpositive/invalid T value(s) converted to NaN."
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
                "wavenumber": spec.x,
                "y_before_offset": y_norm,
                "y_display": y_offset,
                "source_type": source_for_display,
            }
            curves.append(curve)

            x_column = "Binding energy (eV)" if technique == "XPS" else "Wavenumber (cm-1)"
            plot_rows.append(pd.DataFrame({
                x_column: spec.x,
                "Curve": curve_name,
                "Y display": y_offset,
                "Y before offset": y_norm,
                "Source file": spec.path.name,
                "Detected source type": source_for_display,
            }))
            x_meta_column = "Binding energy column" if technique == "XPS" else "Wavenumber column"
            meta_rows.append({
                "Source file": spec.path.name,
                "Curve": curve_name,
                x_meta_column: spec.x_column,
                "Intensity column": spec.y_column,
                "Detected source type": spec.source_type,
                "Rows plotted": int(np.isfinite(y_offset).sum()),
            })

        if not curves:
            raise ValueError(f"No {technique} spectra could be plotted with the selected display settings.")

        plot_df = pd.concat(plot_rows, ignore_index=True)
        meta_df = pd.DataFrame(meta_rows)
        fig = plot_ftir_spectra(
            curves,
            display_mode=display_mode,
            normalization=normalization,
            line_width=1.0,
            legend_position="Best",
            hide_y_ticks=hide_y_ticks,
            technique=technique,
            x_display_min=x_display_min,
            x_display_max=x_display_max,
        )

        scalars = {
            "Spectra plotted": len(curves),
            "Technique": technique,
            "Display mode": display_mode if technique == "FTIR" else "XPS intensity",
            "Normalization": normalization,
            "Vertical offset": offset,
            "Displayed x range": (
                f"{x_display_min:g} to {x_display_max:g}" if custom_x_range else "Full range"
            ),
        }
        result_name = f"{technique} spectra"
        table_name = f"{technique} plot data"
        return ToolResult(
            figures={result_name: fig},
            tables={table_name: plot_df, f"{technique} metadata": meta_df},
            scalars=scalars,
            warnings=warnings,
            payload={"plot_data": plot_df, "metadata": meta_df, "technique": technique},
        )

    def exportable_artifacts(self, result: ToolResult) -> list[Artifact]:
        technique = str(result.payload.get("technique", "FTIR")).upper()
        plot_df = result.payload.get("plot_data")
        if plot_df is None:
            plot_df = result.tables.get(f"{technique} plot data")
        meta_df = result.payload.get("metadata")
        if meta_df is None:
            meta_df = result.tables.get(f"{technique} metadata")
        if plot_df is None:
            return []
        sheets = {f"{technique} plot data": plot_df}
        if meta_df is not None:
            sheets[f"{technique} metadata"] = meta_df
        prefix = technique.lower()
        return [
            Artifact(
                name=f"{technique} plot data (CSV)", kind="csv",
                default_filename=f"{prefix}_plot_data.csv", payload=plot_df,
            ),
            Artifact(
                name=f"{technique} plot data (Excel)", kind="xlsx",
                default_filename=f"{prefix}_plot_data.xlsx", payload=sheets,
            ),
        ]
