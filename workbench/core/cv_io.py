"""Read voltage/current files and retain either branch of each acquired CV cycle."""
from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import re

import numpy as np
import pandas as pd

from workbench.core.data_io import (
    _classify_column, _find_cols, _find_header_row, _split_lsv_header,
)
from workbench.core.io import _guard_shape, read_excel, read_text


@dataclass(frozen=True)
class CvScan:
    """One CV branch in acquisition order, in V and A or A/cm²."""

    potential: np.ndarray
    current: np.ndarray
    cycle: int
    direction: str
    is_density: bool
    potential_column: str
    current_column: str
    error: str = ""
    # Zero-based rows in the source data table, before branch/cycle selection.
    source_indices: np.ndarray | None = None


_CYCLE = re.compile(r"\bcycle\s*[_#-]?\s*(\d+)\b", re.IGNORECASE)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        raw = read_excel(path, header=None)
        header = next((index for index in range(min(80, len(raw)))
                       if None not in _find_cols(raw.iloc[index].tolist())), None)
        if header is None:
            return raw
        table = raw.iloc[header + 1:].reset_index(drop=True)
        table.columns = [str(value) for value in raw.iloc[header].tolist()]
        return table
    if path.suffix.lower() not in {".csv", ".txt", ".tsv", ".dat"}:
        raise ValueError("Use a .txt, .csv, .xlsx or .xls voltage/current file.")
    source = read_text(path)
    lines = source.splitlines(keepends=True)
    header = _find_header_row(lines)
    if header >= 0:
        columns, delimiter = _split_lsv_header(lines[header])
        table = pd.read_csv(StringIO("".join(lines[header + 1:])), sep=delimiter,
                            header=None, engine="python", skip_blank_lines=False)
        if len(table.columns) != len(columns):
            raise ValueError("The voltage/current header and data have different column counts.")
        table.columns = columns
    else:
        delimiter = "," if path.suffix.lower() == ".csv" else r"\s+"
        table = pd.read_csv(StringIO(source), sep=delimiter, header=None,
                            engine="python", skip_blank_lines=False)
    _guard_shape(len(table), len(table.columns), f"Table '{path.name}'")
    return table


def _column_kind(column: object):
    # Cycle prefixes are metadata, rather than part of the physical quantity.
    return _classify_column(_CYCLE.sub("", str(column)).strip(" _:-"))


def _arrays(table: pd.DataFrame, voltage_index: int, current_index: int):
    voltage_column = str(table.columns[voltage_index])
    current_column = str(table.columns[current_index])
    _, voltage_scale, _ = _column_kind(voltage_column)
    _, current_scale, density = _column_kind(current_column)
    voltage = (pd.to_numeric(table.iloc[:, voltage_index], errors="coerce").to_numpy(dtype=float)
               * voltage_scale)
    current = (pd.to_numeric(table.iloc[:, current_index], errors="coerce").to_numpy(dtype=float)
               * current_scale)
    source_indices = table.index.to_numpy(dtype=int)
    # Shorter wide-format cycles are padded with blank rows. Only trim the
    # outside padding; interior invalid rows remain acquisition gaps.
    occupied = np.flatnonzero(np.isfinite(voltage) | np.isfinite(current))
    if len(occupied):
        voltage = voltage[occupied[0]:occupied[-1] + 1]
        current = current[occupied[0]:occupied[-1] + 1]
        source_indices = source_indices[occupied[0]:occupied[-1] + 1]
    return voltage, current, True if density is None else density, source_indices


def _sweep_intervals(voltage: np.ndarray, forward_sign: int | None = None):
    """Locate turning extrema with hysteresis so voltage jitter is not a cycle.

    Reversal confirmation uses at most 1% of the measured span and normally
    four acquisition steps. The returned boundaries are the actual extrema,
    not the later rows that confirm a reversal.
    """
    finite = np.flatnonzero(np.isfinite(voltage))
    if len(finite) < 2:
        raise ValueError("At least two finite potential points are needed to identify a forward scan.")
    values = voltage[finite]
    span = float(np.ptp(values))
    steps = np.abs(np.diff(values))
    steps = steps[steps > 1e-10]
    if not len(steps) or span <= 1e-10:
        raise ValueError("The measurement does not span a potential interval.")
    tolerance = max(1e-10, min(.01 * span, 4 * float(np.median(steps))))
    initial = np.flatnonzero(np.abs(values - values[0]) > tolerance)
    if not len(initial):
        raise ValueError("The measurement does not contain a resolvable voltage sweep.")
    state = 1 if values[initial[0]] > values[0] else -1
    if forward_sign is None:
        forward_sign = state
    start = extreme = int(finite[0])
    intervals = []
    for index in finite[1:]:
        index = int(index)
        difference = state * (voltage[index] - voltage[extreme])
        if difference >= 0:
            extreme = index
        elif difference < -tolerance:
            if extreme > start:
                intervals.append((start, extreme, state))
            state = -state
            start, extreme = extreme, index
    if extreme > start:
        intervals.append((start, extreme, state))
    return intervals, forward_sign


def read_cv_scans(filepath, branch="Forward"):
    """Return ``((scans, info), error)`` for the requested branch of every cycle.

    Supports acquisition-order curves, a long table with a Cycle column, and
    paired ``Cycle N voltage/current`` columns. The first acquired sweep sets
    the forward direction for the file, including negative-going sweeps. Unit
    conversion happens here; absolute current still requires electrode area
    in the caller. Cycle identification precedes any current-sign filtering.
    A numerically invalid explicitly labelled cycle remains a scan with an
    ``error`` so it cannot discard valid neighbouring cycles.
    """
    path = Path(filepath)
    try:
        if branch not in {"Forward", "Back"}:
            raise ValueError("CV scan must be Forward or Back.")
        table = _read_table(path)
        if table.shape[1] < 2:
            raise ValueError("The file needs voltage and current columns.")
        wide = {}
        for index, column in enumerate(table.columns):
            match = _CYCLE.search(str(column))
            kind, _, _ = _column_kind(column)
            if match and kind in {"V", "i"}:
                cycle = int(match.group(1))
                if kind in wide.setdefault(cycle, {}):
                    raise ValueError(f"Cycle {cycle} has more than one {kind} column.")
                wide[cycle][kind] = index

        sources = []
        source_format = "acquisition"
        if wide:
            source_format = "wide cycles"
            for cycle, columns in wide.items():
                if set(columns) != {"V", "i"}:
                    raise ValueError(f"Cycle {cycle} needs both a voltage and a current column.")
                sources.append((cycle, table, columns["V"], columns["i"]))
        else:
            voltage_index, current_index = _find_cols(table.columns)
            if voltage_index is None or current_index is None:
                voltage_index, current_index = 0, 1
            cycle_index = next((index for index, column in enumerate(table.columns)
                                if re.fullmatch(r"cycle(?:[ _-]*(?:number|index|no\.?))?",
                                                str(column).strip(), re.IGNORECASE)), None)
            if cycle_index is None:
                sources.append((None, table, voltage_index, current_index))
            else:
                source_format = "long cycles"
                labels = pd.to_numeric(table.iloc[:, cycle_index], errors="coerce").to_numpy(dtype=float)
                finite_labels = np.isfinite(labels)
                measurement = (pd.to_numeric(table.iloc[:, voltage_index], errors="coerce").notna()
                               | pd.to_numeric(table.iloc[:, current_index], errors="coerce").notna())
                if np.any(~finite_labels & measurement.to_numpy()):
                    raise ValueError("Every voltage/current row needs a numeric cycle number.")
                for cycle in dict.fromkeys(labels[finite_labels]):
                    if cycle != int(cycle):
                        raise ValueError("Cycle numbers must be integers.")
                    indices = np.flatnonzero(labels == cycle)
                    if np.any(finite_labels[indices[0]:indices[-1] + 1]
                              & (labels[indices[0]:indices[-1] + 1] != cycle)):
                        raise ValueError("Each cycle must occupy one contiguous block of rows.")
                    sources.append((int(cycle), table.iloc[indices[0]:indices[-1] + 1],
                                    voltage_index, current_index))

        scans = []
        warnings = []
        forward_sign = None
        for cycle, frame, voltage_index, current_index in sources:
            voltage, current, is_density, source_indices = _arrays(frame, voltage_index, current_index)
            try:
                if np.count_nonzero(np.isfinite(voltage) & np.isfinite(current)) < 2:
                    raise ValueError("At least two finite potential/current points are needed to identify a forward scan.")
                intervals, candidate_sign = _sweep_intervals(voltage, forward_sign)
                forward_indices = [index for index, (_, _, sign) in enumerate(intervals)
                                   if sign == candidate_sign]
                if cycle is not None and len(forward_indices) > 1:
                    raise ValueError(f"Cycle {cycle} contains multiple forward scans; check its cycle labels.")
                if not forward_indices:
                    raise ValueError(f"Cycle {cycle} does not contain a forward scan.")
                forward_sign = candidate_sign
            except ValueError as exc:
                if cycle is None:
                    raise
                try:
                    _, error_sign = _sweep_intervals(voltage)
                    direction = "Increasing" if error_sign > 0 else "Decreasing"
                except ValueError:
                    direction = "Unknown"
                scans.append(CvScan(
                    potential=voltage, current=current, cycle=cycle, direction=direction,
                    is_density=bool(is_density), potential_column=str(frame.columns[voltage_index]),
                    current_column=str(frame.columns[current_index]), error=str(exc),
                    source_indices=source_indices,
                ))
                continue
            for interval_index in forward_indices:
                number = len(scans) + 1 if cycle is None else cycle
                start, end, scan_sign = intervals[interval_index]
                error = ""
                if branch == "Back":
                    next_index = interval_index + 1
                    if next_index < len(intervals) and intervals[next_index][2] != forward_sign:
                        start, end, scan_sign = intervals[next_index]
                    else:
                        error = f"Cycle {number} does not contain a back scan."
                        start = end = 0
                        scan_sign = -forward_sign
                scans.append(CvScan(
                    potential=voltage[start:end + 1] if not error else np.array([], dtype=float),
                    current=current[start:end + 1] if not error else np.array([], dtype=float),
                    cycle=number, direction="Increasing" if scan_sign > 0 else "Decreasing",
                    is_density=bool(is_density), potential_column=str(frame.columns[voltage_index]),
                    current_column=str(frame.columns[current_index]), error=error,
                    source_indices=(source_indices[start:end + 1] if not error
                                    else np.array([], dtype=int)),
                ))
        spans = [(scan, float(np.ptp(scan.potential[np.isfinite(scan.potential)])))
                 for scan in scans if not scan.error and np.any(np.isfinite(scan.potential))]
        if spans:
            longest = max(span for _, span in spans)
            for scan, span in spans:
                if span < .9 * longest:
                    warnings.append(f"Cycle {scan.cycle} has an incomplete {branch.lower()} voltage span; inspect its results.")
        info = {"file": path.name, "source_format": source_format,
                "scan_count": len(scans), "cycle_count": len(scans), "branch": branch, "warnings": warnings}
        return (scans, info), None
    except (OSError, ValueError, TypeError) as exc:
        return None, f"Could not read {path.name}: {exc}"
