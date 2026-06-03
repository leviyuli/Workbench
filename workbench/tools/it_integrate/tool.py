from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from workbench.core.tool_base import Artifact, ParameterSpec, Tool, ToolResult
from workbench.tools.it_integrate.logic import INTERVAL_OPTIONS, integrate_potentiostatic_tests


class ItIntegrateTool(Tool):
    id = "it_integrate"
    name = "i–t Integrate"
    subtitle = "Stitch repeated potentiostatic test files into one continuous trace; optional resampling."
    needs_files = True
    multiple_files = True
    accepts_directory = True
    accepted_file_types = (".txt",)
    run_button_label = "Integrate"

    def parameter_schema(self) -> list[ParameterSpec]:
        return [
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

        # Sort files by name so stitching order is deterministic.
        files = sorted(files, key=lambda p: p.name.lower())
        out = integrate_potentiostatic_tests(files, interval_s=interval_s)

        figures: dict[str, Figure] = {}
        tables: dict[str, pd.DataFrame] = {}
        df = None
        if out["data"].size:
            df = pd.DataFrame(out["data"], columns=out["headers"])
            tables["Integrated"] = df
            figures["i–t"] = self._plot(df)

        result = ToolResult(
            figures=figures, tables=tables,
            scalars={
                "Input points": out["n_in"],
                "Output points": out["n_out"],
                "Files": len(files),
            },
            warnings=out["warnings"],
            text_blocks={"Output": "Use Export to save the stitched .txt (original format) or .xlsx."},
        )
        result.payload["text"] = out["text"]
        result.payload["df"] = df
        return result

    def _plot(self, df: pd.DataFrame) -> Figure:
        fig = Figure(figsize=(7.5, 4.5))
        ax = fig.add_subplot(111)
        t_col = "T(s)" if "T(s)" in df.columns else df.columns[0]
        # Plot every numeric column except time vs time.
        y_cols = [c for c in df.columns if c != t_col]
        primary = "I(A)" if "I(A)" in y_cols else ("E(V)" if "E(V)" in y_cols else (y_cols[0] if y_cols else None))
        if primary is not None:
            ax.plot(df[t_col], df[primary], linewidth=1.0)
            ax.set_ylabel(primary)
        ax.set_xlabel(t_col)
        ax.set_title("Stitched potentiostatic record")
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
