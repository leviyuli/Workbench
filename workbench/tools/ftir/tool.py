from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from workbench.core.tool_base import Artifact, ParameterSpec, Tool, ToolAction, ToolResult
from workbench.tools.ftir.logic import (
    apply_vertical_offset,
    convert_t_to_a,
    find_spectrum_peaks,
    normalize_spectrum,
    plot_detected_peaks,
    plot_ftir_spectra,
    read_spectrum_file,
)


def _paired_plot_data(curves: list[dict], x_axis_label: str) -> pd.DataFrame:
    """Return one adjacent X/Y-display pair for every plotted spectrum."""
    columns: list[pd.Series] = []
    column_names: list[str] = []
    for curve in curves:
        sample_name = str(curve["curve"])
        columns.extend([
            pd.Series(curve["wavenumber"]),
            pd.Series(curve["y_display"]),
        ])
        column_names.extend([
            f"{sample_name} {x_axis_label}",
            f"{sample_name} Y display",
        ])

    paired = pd.concat(columns, axis=1)
    paired.columns = column_names
    return paired


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
                "peak_file", "Peak detection file", kind="input_file",
                default="", group="Peak detection",
                tooltip="Choose one file from the input list, then click Find peaks.",
            ),
            ParameterSpec(
                "peak_prominence_percent", "Minimum prominence", kind="float",
                default=5.0, minimum=0.0, maximum=100.0, step=1.0, suffix="%",
                group="Peak detection",
                tooltip="Minimum peak prominence as a percentage of the spectrum's y range.",
            ),
            ParameterSpec(
                "peak_noise_multiplier", "Noise rejection", kind="float",
                default=4.0, minimum=0.0, maximum=20.0, step=0.5, suffix="× noise",
                group="Peak detection",
                tooltip="A peak must also exceed this multiple of the robust noise estimate.",
            ),
            ParameterSpec(
                "peak_min_width", "Minimum peak width", kind="float",
                default=3.0, minimum=1.0, maximum=1000.0, step=1.0, suffix="points",
                group="Peak detection",
                tooltip="Reject isolated spikes narrower than this many sampled points.",
            ),
            ParameterSpec(
                "peak_smoothing_window", "Smoothing window", kind="int",
                default=0, minimum=0, maximum=1001, step=1, suffix="points",
                group="Peak detection",
                tooltip="0 chooses automatically. Custom values are adjusted to a valid odd window.",
            ),
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
                "show_reference_line", "Show vertical reference line", kind="bool",
                default=False, group="Reference line",
            ),
            ParameterSpec(
                "reference_x", "Reference x position", kind="float",
                default=1300.0, step=1.0, decimals=4, group="Reference line",
                show_if=("show_reference_line", True),
                tooltip="Uses cm⁻¹ in FTIR mode and eV in XPS mode.",
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

    def actions(self) -> list[ToolAction]:
        return [ToolAction(label="Find peaks", method="find_peaks")]

    def find_peaks(self, files: list[Path], params: dict, extra=None) -> ToolResult:
        if not files:
            raise ValueError("Add at least one FTIR or XPS file.")
        selected = str(params.get("peak_file", "") or "")
        if not selected:
            raise ValueError("Choose a peak detection file in the parameter panel.")
        selected_path = Path(selected)
        available = {str(Path(path)): Path(path) for path in files}
        path = available.get(str(selected_path))
        if path is None:
            raise ValueError("The selected peak detection file is no longer in the input list.")

        technique = str(params.get("technique", "FTIR")).upper()
        if technique not in {"FTIR", "XPS"}:
            raise ValueError(f"Unknown technique: {technique}")
        spec = read_spectrum_file(path, technique=technique)
        minima = technique == "FTIR" and spec.source_type == "T"
        prominence_percent = float(params.get("peak_prominence_percent", 5.0))
        noise_multiplier = float(params.get("peak_noise_multiplier", 4.0))
        minimum_width = float(params.get("peak_min_width", 3.0))
        smoothing_window = int(params.get("peak_smoothing_window", 0))
        detection = find_spectrum_peaks(
            spec.x,
            spec.intensity,
            minima=minima,
            prominence_percent=prominence_percent,
            noise_multiplier=noise_multiplier,
            minimum_width=minimum_width,
            smoothing_window=smoothing_window,
        )
        indices = np.asarray(detection["indices"], dtype=int)
        clean_x = np.asarray(detection["x"], dtype=float)
        clean_y = np.asarray(detection["y"], dtype=float)
        prominences = np.asarray(detection["prominences"], dtype=float)
        widths = np.asarray(detection["widths"], dtype=float)
        median_x_step = float(np.median(np.abs(np.diff(clean_x))))

        x_label = "Peak binding energy (eV)" if technique == "XPS" else "Peak wavenumber (cm-1)"
        width_label = "Peak width (eV)" if technique == "XPS" else "Peak width (cm-1)"
        peaks = pd.DataFrame({
            x_label: clean_x[indices],
            "Intensity": clean_y[indices],
            "Prominence": prominences,
            "Peak width (points)": widths,
            width_label: widths * median_x_step,
        }).sort_values(x_label, ascending=False, ignore_index=True)
        peaks.insert(0, "Peak", np.arange(1, len(peaks) + 1))
        title = f"{technique} peaks — {path.name}"
        fig = plot_detected_peaks(
            clean_x, clean_y, indices, technique=technique,
            source_type=spec.source_type, title=title,
        )
        warnings = list(spec.warnings)
        if peaks.empty:
            warnings.append(
                f"{path.name}: no peaks passed the selected prominence, noise, and width filters."
            )
        return ToolResult(
            figures={"Peak detection": fig},
            tables={"Peaks": peaks},
            scalars={
                "Source file": path.name,
                "Technique": technique,
                "Detected peaks": len(peaks),
                "Peak direction": "Minima" if minima else "Maxima",
                "Prominence setting": f"{prominence_percent:g}% of y range",
                "Noise rejection": f"{noise_multiplier:g}× estimated noise",
                "Minimum peak width": f"{minimum_width:g} points",
                "Smoothing window used": f"{int(detection['smoothing_window'])} points",
                "Effective prominence threshold": float(detection["prominence_floor"]),
            },
            warnings=warnings,
            payload={"peak_data": peaks, "technique": technique},
        )

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
        show_reference_line = bool(params.get("show_reference_line", False))
        reference_x = float(params.get("reference_x", 1300.0)) if show_reference_line else None
        custom_x_range = bool(params.get("custom_x_range", False))
        x_display_min = float(params.get("x_display_min", 0.0)) if custom_x_range else None
        x_display_max = float(params.get("x_display_max", 4000.0)) if custom_x_range else None
        if custom_x_range and x_display_min >= x_display_max:
            raise ValueError("Minimum displayed x must be smaller than maximum displayed x.")

        warnings: list[str] = []
        curves: list[dict] = []
        meta_rows: list[dict] = []
        x_column = "Binding energy (eV)" if technique == "XPS" else "Wavenumber (cm-1)"

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

        plot_df = _paired_plot_data(curves, x_column)
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
            reference_x=reference_x,
        )

        scalars = {
            "Spectra plotted": len(curves),
            "Technique": technique,
            "Display mode": display_mode if technique == "FTIR" else "XPS intensity",
            "Normalization": normalization,
            "Vertical offset": offset,
            "Vertical reference line": (
                f"{reference_x:g} {'eV' if technique == 'XPS' else 'cm⁻¹'}"
                if reference_x is not None else "Hidden"
            ),
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
        peak_df = result.payload.get("peak_data")
        if peak_df is not None:
            prefix = technique.lower()
            return [
                Artifact(
                    name=f"{technique} peaks (CSV)", kind="csv",
                    default_filename=f"{prefix}_peaks.csv", payload=peak_df,
                ),
                Artifact(
                    name=f"{technique} peaks (Excel)", kind="xlsx",
                    default_filename=f"{prefix}_peaks.xlsx", payload={"Peaks": peak_df},
                ),
            ]
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
