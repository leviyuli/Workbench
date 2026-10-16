from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from workbench.core.potential import PotentialCorrection, potential_parameter_schema
from workbench.core.tool_base import Artifact, ParameterSpec, Tool, ToolResult
from workbench.tools.it_integrate.logic import (
    INTERVAL_OPTIONS,
    TRACE_MODE_E_T,
    TRACE_TYPE_OPTIONS,
    integrate_potentiostatic_tests,
    normalize_trace_type,
)


def _unique_source_names(sources: list[dict]) -> list[str]:
    """Return stable display names, disambiguating repeated basenames."""
    counts: dict[str, int] = {}
    names: list[str] = []
    for source in sources:
        base = str(source["name"])
        counts[base] = counts.get(base, 0) + 1
        occurrence = counts[base]
        names.append(base if occurrence == 1 else f"{base} ({occurrence})")
    return names


def _source_wide_dataframe(sources: list[dict]) -> pd.DataFrame:
    """Place each source's columns side by side, padding shorter files."""
    columns: list[pd.Series] = []
    column_names: list[str] = []
    for label, source in zip(_unique_source_names(sources), sources):
        data = source["data"]
        for idx, header in enumerate(source["headers"]):
            columns.append(pd.Series(data[:, idx]))
            column_names.append(f"{label} · {header}")
    if not columns:
        return pd.DataFrame()
    frame = pd.concat(columns, axis=1)
    frame.columns = column_names
    return frame


def _source_wide_text(df: pd.DataFrame, sources: list[dict]) -> str:
    """Render the source-designated table with the legacy numeric formats."""
    if df.empty:
        return ""
    time_columns = {
        f"{label} · {source['time_header']}"
        for label, source in zip(_unique_source_names(sources), sources)
    }
    rows = ["\t".join(str(column) for column in df.columns)]
    for _, row in df.iterrows():
        cells: list[str] = []
        for column, value in row.items():
            if pd.isna(value):
                cells.append("")
            elif column in time_columns:
                cells.append(f"{value:.5f}")
            else:
                cells.append(f"{value:.8g}")
        rows.append("\t".join(cells))
    return "\n".join(rows) + "\n"


class ItIntegrateTool(Tool):
    id = "it_integrate"
    name = "i–t Integrate"
    subtitle = "Stitch repeated current-time or potential-time files into one continuous trace; optional resampling."
    needs_files = True
    multiple_files = True
    accepts_directory = True
    accepted_file_types = (".txt",)
    run_button_label = "Integrate"

    def parameter_schema(self) -> list[ParameterSpec]:
        return [
            ParameterSpec(
                "trace_type", "Trace type", kind="choice",
                choices=list(TRACE_TYPE_OPTIONS.keys()), default="i_t (current vs time)",
                tooltip="Choose i_t for steady voltammetry/chronoamperometry data or E_t for steady-current/chronopotentiometry data.",
                group="Options",
            ),
            ParameterSpec(
                "interval", "Resample interval", kind="choice",
                choices=list(INTERVAL_OPTIONS.keys()) + ["custom"], default="1 min",
                tooltip="Linearly resample the stitched data onto an even time grid.",
                group="Options",
            ),
            ParameterSpec(
                "custom_seconds", "Custom interval", kind="float", default=60.0,
                minimum=0.0, step=1.0, suffix="s",
                show_if=("interval", "custom"), group="Options",
            ),
            ParameterSpec(
                "designate_sources", "Designate data source", kind="bool",
                default=False, group="Options",
                tooltip=(
                    "Color each input file separately and export one adjacent "
                    "column group per source file."
                ),
            ),
        ] + potential_parameter_schema(include_input_unit=True)

    def run(self, files: list[Path], params: dict, extra=None) -> ToolResult:
        choice = params.get("interval", "1 min")
        if choice == "custom":
            interval_s = float(params.get("custom_seconds", 0.0)) or None
        else:
            interval_s = INTERVAL_OPTIONS.get(choice)
        trace_type = normalize_trace_type(params.get("trace_type", "i_t (current vs time)"))
        designate_sources = bool(params.get("designate_sources", False))
        correction = PotentialCorrection.from_params(params)

        # Preserve the order shown in the input window. This also controls the
        # stitching sequence, plot colors/legend, and source-grouped exports.
        out = integrate_potentiostatic_tests(
            files, interval_s=interval_s, trace_type=trace_type,
            potential_correction=correction,
            potential_unit=params.get("potential_unit", "V"),
        )
        potential_metadata = correction.metadata()
        potential_description = correction.describe()
        corrected_columns = len(out.get("potential_labels", {})) if correction.is_converted else 0
        potential_metadata["Potential columns corrected"] = corrected_columns
        if correction.is_converted and not corrected_columns:
            potential_description = (
                "No reference potential columns found; no potential correction applied."
            )
            potential_metadata.update({
                "Requested potential scale": correction.target_scale,
                "Potential scale": "Original",
                "Potential shift (V)": 0.0,
                "Correction status": potential_description,
            })

        figures: dict[str, Figure] = {}
        tables: dict[str, pd.DataFrame] = {}
        df = None
        if out["data"].size:
            if designate_sources:
                df = _source_wide_dataframe(out["sources"])
            else:
                df = pd.DataFrame(out["data"], columns=out["headers"])
            tables["Integrated"] = df
            tab_name = "E-t" if trace_type == TRACE_MODE_E_T else "i-t"
            figures[tab_name] = self._plot(
                pd.DataFrame(out["data"], columns=out["headers"]),
                trace_type,
                out.get("time_header"),
                sources=out["sources"] if designate_sources else None,
                potential_labels=out.get("potential_labels"),
            )

        result = ToolResult(
            figures=figures, tables=tables,
            scalars={
                "Trace type": "E_t" if trace_type == TRACE_MODE_E_T else "i_t",
                "Input points": out["n_in"],
                "Output points": out["n_out"],
                "Files": len(files),
                "Data source designation": "On" if designate_sources else "Off",
            },
            warnings=out["warnings"] + correction.warnings,
            text_blocks={
                "Output": (
                    "Use Export to save the source-grouped .txt or .xlsx."
                    if designate_sources
                    else "Use Export to save the stitched .txt (original format) or .xlsx."
                ),
                "Potential scale": potential_description,
            },
        )
        result.payload["text"] = (
            _source_wide_text(df, out["sources"])
            if designate_sources and df is not None
            else out["text"]
        )
        result.payload["df"] = df
        result.payload["trace_type"] = trace_type
        result.payload["designate_sources"] = designate_sources
        result.payload["potential_correction"] = potential_metadata
        return result

    @staticmethod
    def _pick_primary(y_cols: list[str], trace_type: str) -> str | None:
        """Choose the trace to preview according to the selected experiment."""
        def looks_like(name: str, *, current: bool) -> bool:
            n = name.strip().lower()
            if current:
                return (
                    n.startswith(("i(", "i/", "j(", "j/", "<i>"))
                    or "a/cm" in n
                    or "current" in n
                    or n in ("i", "j")
                )
            return (
                n.startswith(("e(", "e/", "v(", "v/"))
                or "potential" in n
                or "voltage" in n
                or "ewe" in n
                or n in ("e", "v")
            )

        prefer_current = trace_type != TRACE_MODE_E_T
        for c in y_cols:
            if looks_like(c, current=prefer_current):
                return c
        for c in y_cols:
            if looks_like(c, current=not prefer_current):
                return c
        return y_cols[0] if y_cols else None

    @staticmethod
    def _pick_time_col(df: pd.DataFrame, time_header: str | None) -> str:
        if time_header in df.columns:
            return str(time_header)
        if "T(s)" in df.columns:
            return "T(s)"
        for c in df.columns:
            n = c.strip().lower()
            if n == "t(s)" or n.startswith(("t(", "t/")) or "time" in n:
                return c
        return df.columns[0]

    def _plot(
        self,
        df: pd.DataFrame,
        trace_type: str,
        time_header: str | None = None,
        sources: list[dict] | None = None,
        potential_labels: dict[str, str] | None = None,
    ) -> Figure:
        fig = Figure(figsize=(7.5, 4.5))
        ax = fig.add_subplot(111)
        t_col = self._pick_time_col(df, time_header)
        # Plot every numeric column except time vs time.
        y_cols = [c for c in df.columns if c != t_col]
        primary = self._pick_primary(y_cols, trace_type)
        if primary is not None:
            if sources:
                primary_idx = list(df.columns).index(primary)
                time_idx = list(df.columns).index(t_col)
                for label, source in zip(_unique_source_names(sources), sources):
                    data = source["data"]
                    ax.plot(
                        data[:, time_idx], data[:, primary_idx],
                        linewidth=1.0, label=label,
                    )
                ax.legend()
            else:
                ax.plot(df[t_col], df[primary], linewidth=1.0)
            ax.set_ylabel((potential_labels or {}).get(primary, primary))
        ax.set_xlabel(t_col)
        title = "Stitched potential-time record" if trace_type == TRACE_MODE_E_T else "Stitched current-time record"
        ax.set_title(title)
        fig.tight_layout()
        return fig

    def exportable_artifacts(self, result: ToolResult) -> list[Artifact]:
        items: list[Artifact] = []
        text = result.payload.get("text")
        if text:
            items.append(Artifact(
                name="Integrated (TXT)", kind="txt",
                default_filename="txt_integrate_interpolated.txt", payload=text,
            ))
        df = result.payload.get("df")
        if df is not None:
            items.append(Artifact(
                name="Integrated (XLSX)", kind="xlsx",
                default_filename="txt_integrate.xlsx", payload={"Integrated": df},
            ))
        return items
