"""Concatenate current-time or potential-time traces with optional resampling."""
from __future__ import annotations

import os
import re

import numpy as np

from workbench.core.io import read_text
from workbench.core.potential import PotentialCorrection


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
        "sources": [],
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


def _is_reference_potential(name: str) -> bool:
    """Cell voltages and electrode differences do not have a reference offset."""
    n = _clean_header(name).replace("_", "").replace("−", "-")
    return _looks_like_potential(name) and not any(
        marker in n for marker in ("cellvoltage", "voltagecell", "ecell", "ewe-ece")
    )


def _potential_unit(header: str) -> str | None:
    """Prefer explicit units over the input-unit fallback selected in the UI."""
    if _clean_header(header) in {"e", "v", "potential", "voltage"}:
        return None
    for unit in ("mV", "V"):
        if re.search(rf"(?<![a-z]){unit}(?![a-z])", header, flags=re.IGNORECASE):
            return unit
    return None


def _correct_potentials(
    headers: list[str],
    data: np.ndarray,
    sources: list[dict],
    correction: PotentialCorrection | None,
    fallback_unit: str,
    warnings: list[str],
) -> tuple[list[str], dict[str, str]]:
    """Update every representation together so plots and exports stay aligned."""
    if correction is None or (
        not correction.is_converted and correction.electrode_setup != "Two-electrode"
    ):
        return headers, {}

    corrected_headers = list(headers)
    labels: dict[str, str] = {}
    for idx, header in enumerate(headers):
        if not _is_reference_potential(header):
            continue
        explicit_unit = _potential_unit(header)
        unit = explicit_unit or fallback_unit
        if explicit_unit is None and correction.is_converted:
            warnings.append(
                f"Potential column '{header}' has no recognized unit; "
                f"using the selected input-unit fallback ({unit})."
            )
        data[:, idx] = correction.apply(data[:, idx], unit=unit)
        for source in sources:
            source["data"][:, idx] = correction.apply(source["data"][:, idx], unit=unit)
        suffix = (
            "cell_voltage" if correction.electrode_setup == "Two-electrode"
            else f"vs_{correction.target_scale}"
        )
        base_header = re.sub(r"\[(?:vs_[^\]]+|cell_voltage)\]$", "", header)
        if explicit_unit is None and correction.is_converted:
            base_header = f"{base_header}({unit})"
        corrected_headers[idx] = f"{base_header}[{suffix}]"
        labels[corrected_headers[idx]] = (
            correction.axis_label(unit=unit)
            if explicit_unit is not None or correction.is_converted
            else f"Cell voltage ({header})"
        )

    for source in sources:
        source["headers"] = list(corrected_headers)
    if correction.is_converted and not labels:
        warnings.append("No reference potential columns found; no potential correction applied.")
    return corrected_headers, labels


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


def _column_order(source_headers: list[str], target_headers: list[str]) -> list[int] | None:
    """Map a file's columns onto the first valid file's canonical order.

    Units and punctuation must match; case and whitespace are ignored.
    Duplicate headers are ambiguous and rejected.
    """
    source_keys = [_clean_header(h) for h in source_headers]
    target_keys = [_clean_header(h) for h in target_headers]
    if len(set(source_keys)) != len(source_keys):
        return None
    positions = {key: idx for idx, key in enumerate(source_keys)}
    if set(positions) != set(target_keys):
        return None
    return [positions[key] for key in target_keys]


def _resample_matrix(data: np.ndarray, t_idx: int, interval_s: float) -> np.ndarray:
    """Resample one shifted source while retaining its measured endpoint."""
    source_t = data[:, t_idx]
    start, end = float(source_t[0]), float(source_t[-1])
    new_t = np.arange(start, end, interval_s)
    if not len(new_t):
        new_t = np.array([start], dtype=float)
    if np.isclose(new_t[-1], end, rtol=1e-12, atol=1e-12):
        new_t[-1] = end
    elif end > new_t[-1]:
        new_t = np.append(new_t, end)

    columns = [
        new_t if idx == t_idx else np.interp(new_t, source_t, data[:, idx])
        for idx in range(data.shape[1])
    ]
    return np.column_stack(columns)


def integrate_potentiostatic_tests(
    input_files, interval_s=None, trace_type=TRACE_MODE_I_T,
    *, potential_correction: PotentialCorrection | None = None,
    potential_unit: str = "V",
):
    """Return stacked traces, per-file sources, metadata and export text.

    Reference correction affects potential columns after resampling in both
    the stacked data and per-file sources.
    """
    trace_type = normalize_trace_type(trace_type)
    warnings: list[str] = []
    last_t = 0.0
    last_dt = 0.0
    headers: list[str] = []
    t_idx = -1
    data_dict: dict[str, list] = {}
    sources: list[dict] = []

    for file_path in input_files:
        file_path = str(file_path)
        if not os.path.exists(file_path):
            warnings.append(f"File not found, skipped: {os.path.basename(file_path)}")
            continue

        lines = read_text(file_path).splitlines(keepends=True)

        header_idx, current_headers, current_t_idx = _find_header(lines, trace_type)
        if header_idx == -1:
            warnings.append(
                f"No time/{_signal_name(trace_type)} header found, skipped: "
                f"{os.path.basename(file_path)}"
            )
            continue

        if not headers:
            if len({_clean_header(h) for h in current_headers}) != len(current_headers):
                warnings.append(
                    f"Duplicate column headers are ambiguous, skipped: "
                    f"{os.path.basename(file_path)}"
                )
                continue
            order = list(range(len(current_headers)))
        else:
            order = _column_order(current_headers, headers)
            if order is None:
                warnings.append(
                    f"Column headers differ from the first valid file, skipped: "
                    f"{os.path.basename(file_path)}"
                )
                continue

        data_lines = [ln for ln in lines[header_idx + 1:] if ln.strip()]
        if not data_lines:
            continue

        file_t_values, valid_lines = [], []
        for line in data_lines:
            cols = line.strip().split()
            if len(cols) != len(current_headers):
                continue
            try:
                float_cols = [float(c) for c in cols]
            except ValueError:
                continue
            file_t_values.append(float_cols[current_t_idx])
            valid_lines.append([float_cols[idx] for idx in order])

        if not file_t_values:
            continue

        if not np.all(np.isfinite(file_t_values)):
            warnings.append(
                f"Time values are nonfinite, skipped: {os.path.basename(file_path)}"
            )
            continue

        if len(file_t_values) >= 2 and np.any(np.diff(file_t_values) < 0):
            warnings.append(
                f"Time values decrease within the file, skipped: "
                f"{os.path.basename(file_path)}"
            )
            continue

        if not headers:
            headers = current_headers
            t_idx = current_t_idx
            data_dict = {header: [] for header in headers}

        shift = 0.0
        already_have_data = bool(data_dict[headers[t_idx]])
        if already_have_data:
            shift = last_t + last_dt - file_t_values[0]

        for float_cols in valid_lines:
            float_cols[t_idx] += shift
            for col_name, val in zip(headers, float_cols):
                data_dict[col_name].append(val)

        shifted = [t + shift for t in file_t_values]
        source_data = np.asarray(valid_lines, dtype=float)
        sources.append({
            "name": os.path.basename(file_path),
            "path": file_path,
            "headers": list(headers),
            "data": source_data,
            "time_header": headers[t_idx],
        })
        last_t = shifted[-1]
        if len(shifted) >= 2:
            positive_steps = np.diff(shifted)
            positive_steps = positive_steps[positive_steps > 0]
            if len(positive_steps):
                last_dt = float(positive_steps[-1])
            elif last_dt == 0.0:
                last_dt = 1.0
        elif last_dt == 0.0:
            last_dt = 1.0

    time_header = headers[t_idx] if headers and t_idx >= 0 else "T(s)"
    if not data_dict or len(data_dict.get(time_header, [])) == 0:
        return _empty_result(headers, warnings + ["No valid data extracted."], trace_type)

    for h in headers:
        data_dict[h] = np.array(data_dict[h])
    t_array = data_dict[time_header]

    if interval_s is not None and interval_s > 0:
        start, end = float(t_array[0]), float(t_array[-1])
        new_t = np.arange(start, end, interval_s)
        # Preserve the endpoint with a shorter final interval when needed.
        if not len(new_t):
            new_t = np.array([start], dtype=float)
        if np.isclose(new_t[-1], end, rtol=1e-12, atol=1e-12):
            new_t[-1] = end
        elif end > new_t[-1]:
            new_t = np.append(new_t, end)
        out = {h: (new_t if h == time_header else np.interp(new_t, t_array, data_dict[h]))
               for h in headers}
        warnings.append(
            f"Resampled to a {interval_s:g} s interval while retaining the measured endpoint."
        )
        for source in sources:
            source["data"] = _resample_matrix(source["data"], t_idx, interval_s)
    else:
        out = data_dict
        new_t = t_array

    data_matrix = np.column_stack([out[h] for h in headers])
    headers, potential_labels = _correct_potentials(
        headers, data_matrix, sources, potential_correction, potential_unit, warnings,
    )

    # Render text in the original format: time as %.5f, others as %.8g.
    rows = [("\t".join(headers))]
    for row in data_matrix:
        cells = []
        for h, val in zip(headers, row):
            cells.append(f"{val:.5f}" if h == time_header else f"{val:.8g}")
        rows.append("\t".join(cells))
    text = "\n".join(rows) + "\n"

    return {
        "headers": headers, "data": data_matrix,
        "n_in": int(len(t_array)), "n_out": int(len(new_t)),
        "text": text, "warnings": warnings,
        "trace_type": trace_type,
        "time_header": time_header,
        "sources": sources,
        "potential_labels": potential_labels,
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
