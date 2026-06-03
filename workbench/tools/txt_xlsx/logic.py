"""TXT -> XLSX conversion. Refactored from txt_xlsx.py into a pure function
that returns an in-memory workbook (so the unified Export button saves it) and
a summary, instead of writing to a hard-coded path.

One sheet per input file; numeric cells get the original ``0.####`` format,
text cells are preserved as text.
"""
from __future__ import annotations

import os
from pathlib import Path

from openpyxl import Workbook

from workbench.core.io import read_text_grid


def convert_txt_to_xlsx(paths, sort_order: str = "name", number_format: str = "0.####"):
    """Build a workbook with one sheet per .txt file.

    Returns dict: {workbook, sheets: [name], rows_per_sheet: {name: n}, used: [paths]}.
    """
    files = [Path(p) for p in paths]
    if sort_order == "modified":
        files = sorted(files, key=lambda p: p.stat().st_mtime if p.exists() else 0)
    else:  # name
        files = sorted(files, key=lambda p: p.name.lower())

    wb = Workbook()
    wb.remove(wb.active)

    sheets: list[str] = []
    rows_per_sheet: dict[str, int] = {}
    used_names: set[str] = set()

    for fp in files:
        grid = read_text_grid(fp)
        base = os.path.splitext(fp.name)[0][:31] or "Sheet"
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
                try:
                    numeric = float(value)
                    cell = ws.cell(row=r, column=c, value=numeric)
                    cell.number_format = number_format
                except ValueError:
                    ws.cell(row=r, column=c, value=value)
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
