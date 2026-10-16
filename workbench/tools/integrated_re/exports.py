"""Final plotted series and their formatted Excel export."""
from __future__ import annotations

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

from workbench.core.tool_base import ToolResult


CURRENT = "Current density (A/cm²)"
_BODY_FONT = Font(name="Aptos", size=11, color="202B33")
_HEADER_FONT = Font(name="Aptos", size=11, bold=True, color="FFFFFF")
_HEADER_FILL = PatternFill("solid", fgColor="243746")
_HEADER_ALIGNMENT = Alignment(horizontal="center", vertical="center", wrap_text=True)
_NUMBER_ALIGNMENT = Alignment(horizontal="right", vertical="center")
_TEXT_ALIGNMENT = Alignment(horizontal="left", vertical="center")


def _column(table: pd.DataFrame, key: str, mask=None) -> pd.Series:
    values = table[key]
    return (values if mask is None else values.where(mask)).reset_index(drop=True)


def plot_tables(result: ToolResult) -> dict[str, pd.DataFrame]:
    """Preserve each plotted trace's sampling order, current grid and gaps."""
    tables = result.tables
    reconstructed = tables["Reconstructed cell"]
    cell = {
        "Reconstructed V (V)": _column(reconstructed, "Reconstructed cell voltage (V)"),
        "Reconstructed |j| (A/cm²)": _column(reconstructed, CURRENT),
    }
    corrected = tables.get("Reconstructed cell (iR corrected)")
    if corrected is not None:
        cell["Reconstructed V iR corrected (V)"] = _column(corrected, "Reconstructed cell voltage (V)")
    measured = tables.get("Cell LSV")
    if measured is not None:
        branch = measured[CURRENT] >= 0
        cell["Measured (2E) V (V)"] = _column(measured, "Cell voltage (V)", branch)
        cell["Measured (2E) |j| (A/cm²)"] = _column(measured, CURRENT, branch)
        if corrected is not None:
            cell["Measured (2E) V iR corrected (V)"] = _column(measured, "iR-corrected cell voltage (V)", branch)

    settings = dict(tables["Potential settings"].itertuples(index=False, name=None))
    overpotential = settings["Electrode plot scale"] == "Overpotential"
    value_key = "Overpotential (V)" if overpotential else "E (V vs RHE)"
    corrected_key = "iR-corrected overpotential (V)" if overpotential else "iR-corrected E (V vs RHE)"
    label = "η (V)" if overpotential else "E (V vs RHE)"
    corrected_label = "η iR corrected (V)" if overpotential else "E iR corrected (V vs RHE)"
    electrodes = {}
    for role in ("Anode", "Cathode"):
        curve = tables[f"{role} LSV"]
        usable = np.isfinite(curve[value_key]) & np.isfinite(curve[CURRENT])
        electrodes[f"{role} {label}"] = _column(curve, value_key, usable)
        electrodes[f"{role} j (A/cm²)"] = _column(curve, CURRENT, usable)
        if corrected is not None:
            electrodes[f"{role} {corrected_label}"] = _column(curve, corrected_key, usable)

    output = {"Cell comparison": pd.DataFrame(cell), "Electrode LSVs": pd.DataFrame(electrodes)}
    if "Calibration LSV" in tables:
        integrated = tables["Calibration integrated LSV"]
        calibration = tables["Calibration LSV"]
        output["Calibration alignment"] = pd.DataFrame({
            "Integrated E iR corrected (V vs RHE)": _column(integrated, "E before correction (V vs RHE)"),
            "Integrated E iR + RE corrected (V vs RHE)": _column(integrated, "E (V vs RHE)"),
            "Integrated j (A/cm²)": _column(integrated, CURRENT),
            "3E E iR corrected (V vs RHE)": _column(calibration, "E (V vs RHE)"),
            "3E j (A/cm²)": _column(calibration, CURRENT),
        })
    return output


def _write_table(sheet, frame: pd.DataFrame, header_row: int) -> None:
    for column, heading in enumerate(frame.columns, start=1):
        cell = sheet.cell(header_row, column, str(heading))
        cell.font = _HEADER_FONT
        cell.fill = _HEADER_FILL
        cell.alignment = _HEADER_ALIGNMENT
    sheet.row_dimensions[header_row].height = 42
    for row_number, values in enumerate(frame.itertuples(index=False, name=None), start=header_row + 1):
        for column, value in enumerate(values, start=1):
            if pd.isna(value) or isinstance(value, (float, np.floating)) and not np.isfinite(value):
                continue
            if isinstance(value, np.generic):
                value = value.item()
            cell = sheet.cell(row_number, column, value)
            cell.font = _BODY_FONT
            numeric = isinstance(value, (float, int))
            cell.alignment = _NUMBER_ALIGNMENT if numeric else _TEXT_ALIGNMENT
            if numeric:
                cell.number_format = "0.000E+00" if value != 0 and abs(value) < 1e-6 else "0.######"


def results_workbook(result: ToolResult, tables: dict[str, pd.DataFrame]) -> Workbook:
    """One sheet per final plot, with calibration markers above their curves."""
    workbook = Workbook()
    workbook.remove(workbook.active)
    for name, frame in tables.items():
        sheet = workbook.create_sheet(name)
        sheet.sheet_view.showGridLines = False
        sheet.sheet_properties.tabColor = "243746" if name == "Cell comparison" else "54758C"
        sheet["A2"] = name
        sheet["A2"].font = Font(name="Aptos", size=15, bold=True, color="243746")
        sheet.row_dimensions[2].height = 24
        header_row = 4
        if name == "Calibration alignment":
            sheet["A4"] = "Matched features"
            sheet["A4"].font = Font(name="Aptos", size=11, bold=True, color="243746")
            features = result.tables["Calibration features"][[
                "Source", "E after correction (V vs RHE)", CURRENT,
            ]].rename(columns={"E after correction (V vs RHE)": "E (V vs RHE)", CURRENT: "j (A/cm²)"})
            _write_table(sheet, features, 5)
            header_row = 10
        _write_table(sheet, frame, header_row)
        for column, heading in enumerate(frame.columns, start=1):
            letter = sheet.cell(header_row, column).column_letter
            sheet.column_dimensions[letter].width = max(22, min(30, len(heading) * 0.8))
        sheet.freeze_panes = f"A{header_row + 1}"
        sheet.sheet_properties.pageSetUpPr.fitToPage = True
        sheet.page_setup.orientation = "landscape"
        sheet.page_setup.paperSize = sheet.PAPERSIZE_A4
        sheet.page_setup.fitToWidth = 1
        sheet.page_setup.fitToHeight = 0
        sheet.print_title_rows = f"1:{header_row}"
        sheet.print_area = f"A1:{sheet.cell(header_row + len(frame), len(frame.columns)).coordinate}"
    return workbook
