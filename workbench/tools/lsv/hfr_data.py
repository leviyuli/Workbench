"""Read HFR test columns from uploaded tables or pasted text."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import re

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HfrData:
    """One current column followed by independent HFR test columns."""

    column_names: tuple[str, ...]
    rows: tuple[tuple[float | None, ...], ...]

    def points(self, column_index: int = 0) -> list[tuple[float, float]]:
        if isinstance(column_index, bool) or not isinstance(column_index, (int, np.integer)):
            raise ValueError("Select an HFR test column.")
        if not 0 <= column_index < len(self.column_names):
            raise ValueError("The selected HFR test column is unavailable.")
        return [(float(row[0]), float(row[column_index + 1])) for row in self.rows
                if row[column_index + 1] is not None]


def _is_number(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def parse_hfr_text(text: str) -> HfrData:
    """Parse a first-current-column table, preserving empty test cells in TSV/CSV."""
    if not isinstance(text, str):
        raise ValueError("Paste or upload an HFR table containing numeric current and HFR columns.")
    lines = []
    for line_number, raw in enumerate(text.lstrip("\ufeff").splitlines(), 1):
        line = raw.rstrip()
        # Pasted scientific examples sometimes use a trailing backslash per row.
        if line.endswith("\\"):
            line = line[:-1].rstrip()
        if line.strip():
            lines.append((line_number, line))
    if not lines:
        raise ValueError("Paste or upload HFR data: current density first, then one or more HFR tests.")
    delimiter = next((candidate for candidate in ("\t", ",", ";")
                      if candidate in lines[0][1]), None)
    parsed = []
    for line_number, line in lines:
        cells = (next(csv.reader([line], delimiter=delimiter)) if delimiter is not None
                 else re.split(r"\s+", line.strip()))
        parsed.append((line_number, [cell.strip() for cell in cells]))
    width = max(len(cells) for _, cells in parsed)
    if width < 2:
        raise ValueError("HFR data needs at least two columns: current density and HFR.")
    first = parsed[0][1]
    header = (len(first) >= 2 and bool(first[0]) and not _is_number(first[0])
              and not any(_is_number(cell) for cell in first[1:] if cell))
    if header:
        names = first[1:] + [""] * (width - len(first))
        parsed = parsed[1:]
    else:
        names = [""] * (width - 1)
    if not parsed:
        raise ValueError("The HFR table has a header but no measurements.")
    labels, used = [], set()
    for index, name in enumerate(names, 1):
        label = name or f"Test {index}"
        count = 2
        original = label
        while label in used:
            label = f"{original} ({count})"
            count += 1
        used.add(label)
        labels.append(label)
    rows = []
    for line_number, cells in parsed:
        cells += [""] * (width - len(cells))
        if not cells[0]:
            raise ValueError(f"HFR row {line_number} is missing its current density.")
        row = []
        for column, cell in enumerate(cells):
            if not cell:
                row.append(None)
                continue
            try:
                value = float(cell)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(f"HFR row {line_number}, column {column + 1} must be numeric.") from None
            if not np.isfinite(value) or value < 0:
                quantity = "Current density" if column == 0 else "HFR"
                raise ValueError(f"{quantity} in HFR row {line_number} must be finite and nonnegative.")
            row.append(value)
        if not any(value is not None for value in row[1:]):
            raise ValueError(f"HFR row {line_number} has no HFR measurements.")
        rows.append(tuple(row))
    return HfrData(tuple(labels), tuple(rows))


def read_hfr_file(path: str | Path) -> str:
    """Load table contents as portable editable text, without keeping a file dependency."""
    source = Path(path)
    try:
        if source.suffix.lower() in {".xlsx", ".xls"}:
            frame = pd.read_excel(source, header=None, sheet_name=0)
            buffer = StringIO()
            frame.to_csv(buffer, sep="\t", index=False, header=False, na_rep="")
            text = buffer.getvalue()
        else:
            raw = source.read_bytes()
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = raw.decode("gb18030")
    except (OSError, ValueError, UnicodeError, ImportError) as exc:
        raise ValueError(f"Could not read HFR file '{source.name}': {exc}") from exc
    parse_hfr_text(text)
    return text


def parse_hfr_pairs(text: str) -> HfrData:
    """Require one current column and one HFR column for the current input form."""
    data = parse_hfr_text(text)
    if len(data.column_names) != 1:
        raise ValueError("HFR data needs exactly two columns: current density and HFR.")
    return data
