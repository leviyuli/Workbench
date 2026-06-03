"""Concatenate repeated potentiostatic (i–t) test files into one continuous
record, shifting the T(s) column so time runs unbroken across files, with an
optional resample to a fixed interval.

Refactored from txt_integrate.py: instead of writing a hard-coded output file,
``integrate_potentiostatic_tests`` returns the headers, the numeric matrix, a
ready-to-save text rendering, and a list of warnings.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def integrate_potentiostatic_tests(input_files, interval_s=None):
    """Integrate i–t files. Returns a dict:
        headers, data (ndarray rows×cols), n_in, n_out, text, warnings.
    """
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

        header_idx = -1
        for i, line in enumerate(lines):
            if "E(V)" in line and "T(s)" in line:
                header_idx = i
                break
        if header_idx == -1:
            warnings.append(f"No 'E(V)'/'T(s)' header found, skipped: {os.path.basename(file_path)}")
            continue

        current_headers = lines[header_idx].strip().split()
        if not headers:
            headers = current_headers
            try:
                t_idx = headers.index("T(s)")
            except ValueError:
                return {"headers": [], "data": np.empty((0, 0)), "n_in": 0, "n_out": 0,
                        "text": "", "warnings": warnings + [f"No 'T(s)' column in {os.path.basename(file_path)}; aborted."]}
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

    if not data_dict or len(data_dict.get("T(s)", [])) == 0:
        return {"headers": headers, "data": np.empty((0, 0)), "n_in": 0, "n_out": 0,
                "text": "", "warnings": warnings + ["No valid data extracted."]}

    for h in headers:
        data_dict[h] = np.array(data_dict[h])
    t_array = data_dict["T(s)"]

    if interval_s is not None and interval_s > 0:
        new_t = np.arange(t_array[0], t_array[-1], interval_s)
        out = {h: (new_t if h == "T(s)" else np.interp(new_t, t_array, data_dict[h]))
               for h in headers}
        warnings.append(f"Resampled to a {interval_s:g} s interval.")
    else:
        out = data_dict
        new_t = t_array

    # Render text in the original format: T(s) as %.5f, others as %.8g.
    rows = [("\t".join(headers))]
    for i in range(len(new_t)):
        cells = []
        for h in headers:
            val = out[h][i]
            cells.append(f"{val:.5f}" if h == "T(s)" else f"{val:.8g}")
        rows.append("\t".join(cells))
    text = "\n".join(rows) + "\n"

    data_matrix = np.column_stack([out[h] for h in headers])
    return {
        "headers": headers, "data": data_matrix,
        "n_in": int(len(t_array)), "n_out": int(len(new_t)),
        "text": text, "warnings": warnings,
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
