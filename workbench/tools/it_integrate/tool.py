from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from workbench.core.tool_base import Artifact, ParameterSpec, Tool, ToolResult
from workbench.tools.it_integrate.logic import (
    INTERVAL_OPTIONS,
    TRACE_MODE_E_T,
    TRACE_TYPE_OPTIONS,
    integrate_potentiostatic_tests,
    normalize_trace_type,
)


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
        ]

    def run(self, files: list[Path], params: dict, extra=None) -> ToolResult:
        choice = params.get("interval", "1 min")
        if choice == "custom":
            interval_s = float(params.get("custom_seconds", 0.0)) or None
        else:
            interval_s = INTERVAL_OPTIONS.get(choice)
        trace_type = normalize_trace_type(params.get("trace_type", "i_t (current vs time)"))

        # Sort files by name so stitching order is deterministic.
        files = sorted(files, key=lambda p: p.name.lower())
        out = integrate_potentiostatic_tests(files, interval_s=interval_s, trace_type=trace_type)

        figures: dict[str, Figure] = {}
        tables: dict[str, pd.DataFrame] = {}
        df = None
        if out["data"].size:
            df = pd.DataFrame(out["data"], columns=out["headers"])
            tables["Integrated"] = df
            tab_name = "E-t" if trace_type == TRACE_MODE_E_T else "i-t"
            figures[tab_name] = self._plot(df, trace_type, out.get("time_header"))

        result = ToolResult(
            figures=figures, tables=tables,
            scalars={
                "Trace type": "E_t" if trace_type == TRACE_MODE_E_T else "i_t",
                "Input points": out["n_in"],
                "Output points": out["n_out"],
                "Files": len(files),
            },
            warnings=out["warnings"],
            text_blocks={"Output": "Use Export to save the stitched .txt (original format) or .xlsx."},
        )
        result.payload["text"] = out["text"]
        result.payload["df"] = df
        result.payload["trace_type"] = trace_type
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

    def _plot(self, df: pd.DataFrame, trace_type: str, time_header: str | None = None) -> Figure:
        fig = Figure(figsize=(7.5, 4.5))
        ax = fig.add_subplot(111)
        t_col = self._pick_time_col(df, time_header)
        # Plot every numeric column except time vs time.
        y_cols = [c for c in df.columns if c != t_col]
        primary = self._pick_primary(y_cols, trace_type)
        if primary is not None:
            ax.plot(df[t_col], df[primary], linewidth=1.0)
            ax.set_ylabel(primary)
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
