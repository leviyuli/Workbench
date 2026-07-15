"""Mixed TXT / CSV / XLSX -> XLSX conversion. Refactored from txt_xlsx.py into a
pure function that returns an in-memory workbook (so the unified Export button
saves it) and a summary, instead of writing to a hard-coded path.

One sheet per input .txt/.csv file; each sheet of an input .xlsx is expanded
into its own output sheet. Numeric cells get the original ``0.####`` format,
text cells are preserved as text.
"""
from __future__ import annotations

import os
import math
import re
from pathlib import Path

from openpyxl import Workbook

from workbench.core.io import (
    MAX_TABLE_CELLS,
    MAX_TABLE_ROWS,
    XlsxCellValue,
    neutralize_spreadsheet_formula,
    read_text_grid,
    read_xlsx_grids,
    safe_sheet_title,
)

_TEXT_EXTS = (".txt", ".csv", ".tsv", ".dat")
_XLSX_EXTS = (".xlsx",)
_NUMERIC_RE = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)


def _coerce_numeric_text(value):
    """Convert ordinary numeric text without corrupting identifier-like text.

    Excel stores only 15 significant decimal digits.  Leading-zero integers
    and higher-precision values therefore stay strings instead of being
    rounded (or displayed in scientific notation) as the old ``float()`` path
    did.
    """
    if not isinstance(value, str):
        return value, False
    text = value.strip()
    if not text or _NUMERIC_RE.fullmatch(text) is None:
        return value, False

    unsigned = text.lstrip("+-")
    mantissa = re.split(r"[eE]", unsigned, maxsplit=1)[0]
    integer_part = mantissa.split(".", maxsplit=1)[0]
    is_integer_literal = "." not in mantissa and "e" not in text.lower()

    if len(integer_part) > 1 and integer_part.startswith("0"):
        return value, False

    digits = mantissa.replace(".", "").lstrip("0") or "0"
    if (is_integer_literal and len(integer_part) > 15) or len(digits) > 15:
        return value, False

    if is_integer_literal:
        return int(text), True
    number = float(text)
    if not math.isfinite(number):
        return value, False
    return number, True


def convert_to_xlsx(paths, sort_order: str = "name", number_format: str = "0.####"):
    """Build a workbook from a mix of .txt/.csv/.xlsx files.

    Text files contribute one sheet each (named after the file); each sheet of
    an .xlsx is expanded into its own output sheet (keeping the source sheet
    name). Returns dict: {workbook, sheets: [name], rows_per_sheet: {name: n},
    used: [paths]}.
    """
    files = [Path(p) for p in paths]
    if sort_order == "modified":
        files = sorted(files, key=lambda p: p.stat().st_mtime if p.exists() else 0)
    else:  # name
        files = sorted(files, key=lambda p: p.name.lower())

    # Flatten inputs into (base_name, grid, typed) sources, preserving order.
    # typed=False -> raw string grid (coerce numbers via float()).
    # typed=True  -> already-typed grid from openpyxl (write values directly).
    sources: list[tuple[str, list[list], bool]] = []
    total_rows = 0
    total_cells = 0
    for fp in files:
        ext = fp.suffix.lower()
        if ext in _XLSX_EXTS:
            for title, grid in read_xlsx_grids(fp):
                sources.append((title or "Sheet", grid, True))
                total_rows += len(grid)
                total_cells += sum(len(row) for row in grid)
        elif ext in _TEXT_EXTS:
            base = os.path.splitext(fp.name)[0] or "Sheet"
            grid = read_text_grid(fp)
            sources.append((base, grid, False))
            total_rows += len(grid)
            total_cells += sum(len(row) for row in grid)
        else:
            raise ValueError(
                f"Unsupported input '{fp.name}'. Use .txt, .csv, .tsv, .dat, or .xlsx."
            )
        if total_rows > MAX_TABLE_ROWS or total_cells > MAX_TABLE_CELLS:
            raise ValueError(
                "Combined input exceeds the conversion limit "
                f"({MAX_TABLE_ROWS:,} rows or {MAX_TABLE_CELLS:,} cells)."
            )

    wb = Workbook()
    wb.remove(wb.active)

    sheets: list[str] = []
    rows_per_sheet: dict[str, int] = {}
    used_names: set[str] = set()

    def write_cell(ws, r, c, value, typed):
        if isinstance(value, XlsxCellValue):
            cell = ws.cell(row=r, column=c, value=value.value)
            if value.data_type in {"s", "inlineStr"}:
                # Assignment of a string beginning '=' normally changes it to
                # a formula; restore the source workbook's literal-text type.
                cell.data_type = "s"
            elif value.data_type == "f":
                cell.data_type = "f"
            elif value.data_type == "e":
                cell.data_type = "e"
            if isinstance(value.value, (int, float)) and not isinstance(value.value, bool):
                cell.number_format = number_format
            elif value.number_format:
                cell.number_format = value.number_format
        elif typed:
            if isinstance(value, bool):
                ws.cell(row=r, column=c, value=value)
            elif isinstance(value, (int, float)):
                cell = ws.cell(row=r, column=c, value=value)
                cell.number_format = number_format
            else:
                ws.cell(row=r, column=c, value=value)
        else:
            converted, is_numeric = _coerce_numeric_text(value)
            if is_numeric:
                cell = ws.cell(row=r, column=c, value=converted)
                cell.number_format = number_format
            else:
                ws.cell(
                    row=r,
                    column=c,
                    value=neutralize_spreadsheet_formula(value),
                )

    for base, grid, typed in sources:
        name = safe_sheet_title(base, used_names)

        ws = wb.create_sheet(title=name)
        for r, row in enumerate(grid, start=1):
            for c, value in enumerate(row, start=1):
                write_cell(ws, r, c, value, typed)
        sheets.append(name)
        rows_per_sheet[name] = len(grid)

    if not sheets:                          # openpyxl can't save an empty workbook
        wb.create_sheet(title="Empty")
    return {
        "workbook": wb,
        "sheets": sheets,
        "rows_per_sheet": rows_per_sheet,
        "used": [str(f) for f in files],
    }
