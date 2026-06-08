"""Unified, tolerant file I/O shared by every tool.

The point of this module is the thing the user asked for: one way to *read*
tabular data regardless of where it came from, and one way to *write* each
export format so the buttons behave identically across tools.

``read_table`` accepts .csv / .txt / .xlsx / .xls, auto-detects the delimiter
(tab, comma, semicolon or whitespace), recovers from encoding problems, and
finds the header row even when metadata precedes it. Tools that need
domain-specific extraction (unit scaling, cycle detection) build on top of
this or on the specialised readers in ``core.data_io``.
"""
from __future__ import annotations

import csv
import io as _io
from pathlib import Path

import pandas as pd

_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
TABLE_EXTS = (".csv", ".txt", ".tsv", ".xlsx", ".xls", ".dat")


def _read_text(path: Path) -> str:
    """Read a text file, trying a sequence of encodings and never raising."""
    for enc in _ENCODINGS:
        try:
            with open(path, "r", encoding=enc) as fh:
                return fh.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _sniff_delimiter(sample: str) -> str:
    """Best-effort delimiter detection for a text sample."""
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,; ")
        return dialect.delimiter
    except csv.Error:
        pass
    # Manual fallback: pick the candidate that appears most on the first
    # data-looking line.
    for line in sample.splitlines():
        if line.strip():
            counts = {d: line.count(d) for d in ("\t", ",", ";")}
            best = max(counts, key=counts.get)
            if counts[best] > 0:
                return best
            break
    return r"\s+"


def _looks_numeric(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _find_header_row(lines: list[str], delim: str) -> int:
    """Return the index of the first row that looks like a header (mostly
    non-numeric tokens, at least two columns), else 0."""
    splitter = (lambda s: s.split(delim)) if delim != r"\s+" else (lambda s: s.split())
    for idx, line in enumerate(lines[:60]):
        toks = [t.strip() for t in splitter(line) if t.strip() != ""]
        if len(toks) < 2:
            continue
        non_numeric = sum(0 if _looks_numeric(t) else 1 for t in toks)
        if non_numeric >= max(1, len(toks) // 2):
            return idx
    return 0


def read_table(path, header: str = "auto") -> pd.DataFrame:
    """Read one tabular file into a DataFrame as tolerantly as possible.

    header: "auto" (detect a header row), "yes" (row 0 is header) or
    "no" (no header — columns become 0,1,2,...).
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in (".xlsx", ".xls"):
        hdr = 0 if header != "no" else None
        return pd.read_excel(path, header=hdr)

    text = _read_text(path)
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    if not lines:
        return pd.DataFrame()

    delim = "," if ext == ".csv" else _sniff_delimiter("\n".join(lines[:40]))

    if header == "no":
        skip, hdr = 0, None
    elif header == "yes":
        skip, hdr = 0, 0
    else:
        skip, hdr = _find_header_row(lines, delim), 0

    sep = delim if delim != r"\s+" else r"\s+"
    try:
        df = pd.read_csv(
            _io.StringIO(text), sep=sep, skiprows=skip, header=hdr,
            engine="python", on_bad_lines="skip",
        )
    except Exception:
        df = pd.read_csv(
            _io.StringIO(text), sep=r"\s+", header=None,
            engine="python", on_bad_lines="skip",
        )
    return df


def read_text_grid(path) -> list[list[str]]:
    """Read a tab/whitespace-delimited text file into a raw grid of string
    cells, preserving every row (including headers and non-numeric text).

    Used by the TXT->XLSX tool, which must keep the file's literal layout
    rather than coercing it into a typed DataFrame.
    """
    path = Path(path)
    text = _read_text(path)
    grid: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.rstrip("\n")
        if stripped.strip() == "":
            continue
        if "\t" in stripped:
            cells = stripped.split("\t")
        elif "," in stripped:
            cells = stripped.split(",")
        else:
            cells = stripped.split()
        grid.append([c.strip() for c in cells])
    return grid


def read_xlsx_grids(path) -> list[tuple[str, list[list]]]:
    """Read an .xlsx file into one typed grid per worksheet.

    Returns ``[(sheet_title, grid), ...]`` in workbook order, where each grid is
    a list of rows of cell values. Formula cells yield their cached computed
    value (``data_only=True``); ``None`` cells become ``""``. Values stay typed
    (int/float/str) so the writer can format numbers without re-parsing text.

    Used by the TXT->XLSX tool to expand each input workbook's sheets. Tolerant:
    an unreadable file yields an empty list rather than raising.
    """
    from openpyxl import load_workbook

    path = Path(path)
    try:
        wb = load_workbook(path, read_only=True, data_only=True)
    except Exception:
        return []

    grids: list[tuple[str, list[list]]] = []
    try:
        for ws in wb.worksheets:
            rows: list[list] = []
            for row in ws.iter_rows(values_only=True):
                rows.append(["" if v is None else v for v in row])
            # Drop fully-empty trailing rows.
            while rows and all(c == "" for c in rows[-1]):
                rows.pop()
            grids.append((ws.title, rows))
    finally:
        wb.close()
    return grids


# ---------------------------------------------------------------------------
# Writers — one per export kind, used by widgets.export_button
# ---------------------------------------------------------------------------
def write_csv(df: pd.DataFrame, path) -> None:
    df.to_csv(path, index=False)


def write_xlsx(sheets: dict, path, number_format: str | None = None) -> None:
    """Write a dict of {sheet_name: DataFrame} to one .xlsx."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe = str(name)[:31] or "Sheet1"
            df.to_excel(writer, sheet_name=safe, index=False)
            if number_format:
                ws = writer.sheets[safe]
                for row in ws.iter_rows():
                    for cell in row:
                        if isinstance(cell.value, (int, float)):
                            cell.number_format = number_format


def write_txt(text: str, path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def save_figure(fig, path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight")


def expand_inputs(paths, accepted_exts) -> list[Path]:
    """Expand a mixed list of files and directories into a flat list of files
    whose extension is in ``accepted_exts``. Directories are searched one level
    deep (then recursively if nothing matched at the top)."""
    accepted = tuple(e.lower() for e in accepted_exts)
    out: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            matched = [f for f in sorted(p.iterdir())
                       if f.is_file() and f.suffix.lower() in accepted]
            if not matched:
                matched = [f for f in sorted(p.rglob("*"))
                           if f.is_file() and f.suffix.lower() in accepted]
            out.extend(matched)
        elif p.is_file() and (not accepted or p.suffix.lower() in accepted):
            out.append(p)
    return out
