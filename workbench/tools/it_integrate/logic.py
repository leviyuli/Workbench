"""Concatenate repeated current-time or potential-time test files into one
continuous record, shifting the time column so time runs unbroken across files,
with an optional resample to a fixed interval.

Refactored from txt_integrate.py: instead of writing a hard-coded output file,
``integrate_potentiostatic_tests`` returns the headers, the numeric matrix, a
ready-to-save text rendering, and a list of warnings.
"""
from __future__ import annotations

import os

import numpy as np


TRACE_MODE_I_T = "i_t"
TRACE_MODE_E_T = "E_t"

TRACE_TYPE_OPTIONS = {
    "i_t (current vs time)": TRACE_MODE_I_T,
    "E_t (potential vs time)": TRACE_MODE_E_T,
}


def normalize_trace_type(trace_type) -> str:
    """Map UI labels and API-friendly aliases onto the internal trace mode."""
    if trace_type in (TRACE_MODE_I_T, TRACE_MODE_E_T):
        return trace_type
    if trace_type in TRACE_TYPE_OPTIONS:
        return TRACE_TYPE_OPTIONS[trace_type]

    text = str(trace_type or "").strip().lower().replace("-", "_")
    if text in {"e", "et", "e_t", "potential", "potential_time", "voltage", "voltage_time"}:
        return TRACE_MODE_E_T
    return TRACE_MODE_I_T


def _empty_result(headers, warnings, trace_type):
    return {
        "headers": headers,
        "data": np.empty((0, 0)),
        "n_in": 0,
        "n_out": 0,
        "text": "",
        "warnings": warnings,
        "trace_type": trace_type,
    }


def _clean_header(name: str) -> str:
    return (
        name.strip().lower()
        .replace("µ", "u")
        .replace("²", "2")
        .replace(" ", "")
    )


def _looks_like_time(name: str) -> bool:
    n = _clean_header(name)
    return (
        n == "t(s)"
        or n.startswith(("t(", "t/"))
        or n in {"t", "time", "time(s)"}
        or "time" in n
    )


def _looks_like_current(name: str) -> bool:
    n = _clean_header(name)
    return (
        n.startswith(("i(", "i/", "j(", "j/", "<i>"))
        or n in {"i", "j", "current"}
        or "current" in n
        or "a/cm" in n
    )


def _looks_like_potential(name: str) -> bool:
    n = _clean_header(name)
    return (
        n.startswith(("e(", "e/", "v(", "v/"))
        or n in {"e", "v", "potential", "voltage"}
        or "potential" in n
        or "voltage" in n
        or "ewe" in n
    )


def _find_column(headers: list[str], predicate) -> int:
    for idx, header in enumerate(headers):
        if predicate(header):
            return idx
    return -1


def _signal_predicate(trace_type: str):
    return _looks_like_potential if trace_type == TRACE_MODE_E_T else _looks_like_current


def _signal_name(trace_type: str) -> str:
    return "potential" if trace_type == TRACE_MODE_E_T else "current"


def _find_header(lines: list[str], trace_type: str):
    signal_predicate = _signal_predicate(trace_type)
    for i, line in enumerate(lines):
        current_headers = line.strip().split()
        if len(current_headers) < 2:
            continue
        t_idx = _find_column(current_headers, _looks_like_time)
        signal_idx = _find_column(current_headers, signal_predicate)
        if t_idx >= 0 and signal_idx >= 0:
            return i, current_headers, t_idx
    return -1, [], -1


def integrate_potentiostatic_tests(input_files, interval_s=None, trace_type=TRACE_MODE_I_T):
    """Integrate i-t or E-t files. Returns a dict:
        headers, data (ndarray rows×cols), n_in, n_out, text, warnings.
    """
    trace_type = normalize_trace_type(trace_type)
    warnings: list[str] = []
    last_t = 0.0
    last_dt = 0.0
    headers: list[str] = []
    t_idx = -1
    data_dict: dict[str, list] = {}

    for file_idx, file_path in enumerate(input_files):
        file_path = str(file_path)
        if not os.path.exists(file_path):
            warnings.append(f"File not found, skipped: {os.path.basename(file_path)}")
            continue

        with open(file_path, "r", encoding="utf-8", errors="replace") as infile:
            lines = infile.readlines()

        header_idx, current_headers, current_t_idx = _find_header(lines, trace_type)
        if header_idx == -1:
            warnings.append(
                f"No time/{_signal_name(trace_type)} header found, skipped: "
                f"{os.path.basename(file_path)}"
            )
            continue

        if not headers:
            headers = current_headers
            t_idx = current_t_idx
            if t_idx < 0:
                return _empty_result(
                    [], warnings + [f"No time column in {os.path.basename(file_path)}; aborted."],
                    trace_type,
                )
            for h in headers:
                data_dict[h] = []
        else:
            if len(current_headers) != len(headers):
                warnings.append(f"Column count mismatch, skipped: {os.path.basename(file_path)}")
                continue

        data_lines = [ln for ln in lines[header_idx + 1:] if ln.strip()]
        if not data_lines:
            continue

        file_t_values, valid_lines = [], []
        for line in data_lines:
            cols = line.strip().split()
            if len(cols) != len(headers):
                continue
            try:
                float_cols = [float(c) for c in cols]
            except ValueError:
                continue
            file_t_values.append(float_cols[t_idx])
            valid_lines.append(float_cols)

        if not file_t_values:
            continue

        shift = 0.0
        if file_idx > 0:
            shift = last_t + last_dt - file_t_values[0]

        for float_cols in valid_lines:
            float_cols[t_idx] += shift
            for col_name, val in zip(headers, float_cols):
                data_dict[col_name].append(val)

        shifted = [t + shift for t in file_t_values]
        last_t = shifted[-1]
        if len(shifted) >= 2:
            last_dt = shifted[-1] - shifted[-2]
        elif last_dt == 0.0:
            last_dt = 1.0

    time_header = headers[t_idx] if headers and t_idx >= 0 else "T(s)"
    if not data_dict or len(data_dict.get(time_header, [])) == 0:
        return _empty_result(headers, warnings + ["No valid data extracted."], trace_type)

    for h in headers:
        data_dict[h] = np.array(data_dict[h])
    t_array = data_dict[time_header]

    if interval_s is not None and interval_s > 0:
        new_t = np.arange(t_array[0], t_array[-1], interval_s)
        out = {h: (new_t if h == time_header else np.interp(new_t, t_array, data_dict[h]))
               for h in headers}
        warnings.append(f"Resampled to a {interval_s:g} s interval.")
    else:
        out = data_dict
        new_t = t_array

    # Render text in the original format: time as %.5f, others as %.8g.
    rows = [("\t".join(headers))]
    for i in range(len(new_t)):
        cells = []
        for h in headers:
            val = out[h][i]
            cells.append(f"{val:.5f}" if h == time_header else f"{val:.8g}")
        rows.append("\t".join(cells))
    text = "\n".join(rows) + "\n"

    data_matrix = np.column_stack([out[h] for h in headers])
    return {
        "headers": headers, "data": data_matrix,
        "n_in": int(len(t_array)), "n_out": int(len(new_t)),
        "text": text, "warnings": warnings,
        "trace_type": trace_type,
        "time_header": time_header,
    }


# Interval presets shared with the tool UI.
INTERVAL_OPTIONS = {
    "none": None,
    "1 s": 1,
    "10 s": 10,
    "1 min": 60,
    "1 hour": 3600,
    "1 day": 86400,
}
