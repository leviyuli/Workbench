"""Mixed TXT / CSV / XLSX -> XLSX conversion. Refactored from txt_xlsx.py into a
pure function that returns an in-memory workbook (so the unified Export button
saves it) and a summary, instead of writing to a hard-coded path.

One sheet per input .txt/.csv file; each sheet of an input .xlsx is expanded
into its own output sheet. Numeric cells get the original ``0.####`` format,
text cells are preserved as text.
"""
from __future__ import annotations

import os
from pathlib import Path

from openpyxl import Workbook

from workbench.core.io import read_text_grid, read_xlsx_grids

_TEXT_EXTS = (".txt", ".csv", ".tsv", ".dat")
_XLSX_EXTS = (".xlsx",)


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
    for fp in files:
        ext = fp.suffix.lower()
        if ext in _XLSX_EXTS:
            for title, grid in read_xlsx_grids(fp):
                sources.append((title or "Sheet", grid, True))
        else:  # text-like (.txt/.csv/.tsv/.dat or anything else)
            base = os.path.splitext(fp.name)[0] or "Sheet"
            sources.append((base, read_text_grid(fp), False))

    wb = Workbook()
    wb.remove(wb.active)

    sheets: list[str] = []
    rows_per_sheet: dict[str, int] = {}
    used_names: set[str] = set()

    def write_cell(ws, r, c, value, typed):
        if typed:
            if isinstance(value, bool):
                ws.cell(row=r, column=c, value=value)
            elif isinstance(value, (int, float)):
                cell = ws.cell(row=r, column=c, value=value)
                cell.number_format = number_format
            else:
                ws.cell(row=r, column=c, value=value)
        else:
            try:
                cell = ws.cell(row=r, column=c, value=float(value))
                cell.number_format = number_format
            except (ValueError, TypeError):
                ws.cell(row=r, column=c, value=value)

    for base, grid, typed in sources:
        base = str(base)[:31] or "Sheet"
        name = base
        n = 1
        while name in used_names:           # openpyxl requires unique sheet names
            suffix = f"_{n}"
            name = base[:31 - len(suffix)] + suffix
            n += 1
        used_names.add(name)

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
