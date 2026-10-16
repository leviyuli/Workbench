"""Readable worksheet formatting for analysis and conversion exports."""
from __future__ import annotations

import math
from numbers import Real
import unicodedata

from openpyxl.styles import Alignment, Font, PatternFill


_BODY = Font(name="Aptos", size=11, color="202B33")
_HEADER = Font(name="Aptos", size=11, bold=True, color="FFFFFF")
_FILL = PatternFill("solid", fgColor="243746")
_HEADER_ALIGNMENT = Alignment(horizontal="center", vertical="center", wrap_text=True)
_NUMBER_ALIGNMENT = Alignment(horizontal="right", vertical="center")
_TEXT_ALIGNMENT = Alignment(horizontal="left", vertical="center")
_WRAPPED_ALIGNMENT = Alignment(horizontal="left", vertical="center", wrap_text=True)


def _width(text: str) -> int:
    return max((sum(2 if unicodedata.east_asian_width(char) in "WF" else 1
                    for char in line) for line in text.splitlines()), default=0)


def _number_format(value: Real) -> str:
    if value != 0 and (abs(value) < 1e-4 or abs(value) >= 1e7):
        return "0.000E+00"
    return "#,##0" if isinstance(value, int) else "0.0#####"


def format_sheet(sheet, *, header_row: int | None = 1,
                 number_format: str | None = None,
                 preserve_number_formats: bool = False) -> None:
    """Fit content and style headers while keeping cell values and types intact."""
    sheet.sheet_view.showGridLines = False
    widths = {}
    for column in range(1, sheet.max_column + 1):
        samples = [sheet.cell(row, column) for row in range(1, min(sheet.max_row, 300) + 1)]
        text_width = 0
        for cell in samples:
            if cell.value is None or cell.data_type == "f":
                continue
            content_width = _width(str(cell.value))
            if cell.row == header_row:
                content_width = min(26, content_width)
            elif isinstance(cell.value, str):
                content_width = math.ceil(content_width * 1.12)
            text_width = max(text_width, content_width)
        width = min(52, max(16, text_width + 2))
        widths[column] = width
        sheet.column_dimensions[sheet.cell(1, column).column_letter].width = width
    if header_row is not None:
        height = 30
        for cell in sheet[header_row]:
            if cell.value is None:
                continue
            cell.font = _HEADER
            cell.fill = _FILL
            cell.alignment = _HEADER_ALIGNMENT
            lines = sum(max(1, math.ceil(_width(line) / (widths[cell.column] - 2)))
                        for line in str(cell.value).splitlines())
            height = max(height, 15 * lines + 6)
        sheet.row_dimensions[header_row].height = height
        sheet.freeze_panes = f"A{header_row + 1}"
    for row in sheet.iter_rows(min_row=(header_row + 1) if header_row is not None else 1):
        row_height = 15
        for cell in row:
            value = cell.value
            if value is None:
                continue
            cell.font = _BODY
            numeric = isinstance(value, Real) and not isinstance(value, bool)
            if numeric:
                cell.alignment = _NUMBER_ALIGNMENT
                if not preserve_number_formats and math.isfinite(value):
                    cell.number_format = number_format or _number_format(value)
            else:
                wrapped = cell.data_type != "f" and (
                    "\n" in str(value) or _width(str(value)) > widths[cell.column] - 2
                )
                cell.alignment = _WRAPPED_ALIGNMENT if wrapped else _TEXT_ALIGNMENT
                if wrapped:
                    lines = sum(max(1, math.ceil(_width(line) / (widths[cell.column] - 2)))
                                for line in str(value).splitlines())
                    row_height = max(row_height, 15 * lines + 3)
        if row_height > 15:
            sheet.row_dimensions[row[0].row].height = row_height
