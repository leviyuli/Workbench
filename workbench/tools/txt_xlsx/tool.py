from __future__ import annotations

from pathlib import Path

import pandas as pd

from workbench.core.tool_base import Artifact, ParameterSpec, Tool, ToolResult
from workbench.tools.txt_xlsx.logic import convert_to_xlsx


class TxtXlsxTool(Tool):
    id = "txt_xlsx"
    name = "TXT → XLSX"
    subtitle = "Bundle many .txt / .csv / .xlsx files into one Excel workbook, one sheet per file (xlsx sheets expanded)."
    needs_files = True
    multiple_files = True
    accepts_directory = True
    accepted_file_types = (".txt", ".csv", ".xlsx")
    run_button_label = "Convert"

    def parameter_schema(self) -> list[ParameterSpec]:
        return [
            ParameterSpec(
                "sort_order", "Sheet order", kind="choice",
                choices=["name", "modified"], default="name",
                tooltip="Order the sheets by file name or by file modification time.",
                group="Options",
            ),
            ParameterSpec(
                "number_format", "Number format", kind="str", default="0.####",
                tooltip="Excel number format applied to numeric cells.",
                group="Options",
            ),
        ]

    def run(self, files: list[Path], params: dict, extra=None) -> ToolResult:
        out = convert_to_xlsx(
            files,
            sort_order=params.get("sort_order", "name"),
            number_format=params.get("number_format", "0.####"),
        )
        summary = pd.DataFrame({
            "Sheet": out["sheets"],
            "Rows": [out["rows_per_sheet"][s] for s in out["sheets"]],
        })
        result = ToolResult(
            tables={"Sheets": summary} if out["sheets"] else {},
            scalars={"Files converted": len(out["sheets"])},
            text_blocks={"Conversion": (
                f"{len(out['sheets'])} file(s) written to one workbook.\n"
                "Use Export to save the .xlsx."
            )},
        )
        result.payload["workbook"] = out["workbook"]
        return result

    def exportable_artifacts(self, result: ToolResult) -> list[Artifact]:
        wb = result.payload.get("workbook")
        items: list[Artifact] = []
        if wb is not None:
            items.append(Artifact(
                name="Workbook (XLSX)", kind="workbook",
                default_filename="txt_xlsx.xlsx", payload=wb,
            ))
        return items
