"""CV import, raw cycle splitting, and non-faradaic ECSA pipeline helpers.

Raw splitting preserves source values and units.  The ECSA helpers retain their
potential/current conversions and cycle-selection behavior.
"""
import csv
import os
import re
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

from workbench.core.io import read_text

# ==========================================
# CONFIGURATION (module defaults; the GUI overrides per run)
# ==========================================
RE_TYPE = 'Hg/HgO'
TARGET_SCALE = 'RHE'
PH = 14

E_RE_DICT = {
    'Hg/HgO': 0.098,
    'Ag/AgCl': 0.210,
}

FILENAME_PATTERN = r"(?i)cv[_\s\-]*(\d+(?:\.\d+)?)\s*(?:m\s*v(?:[/_\s]*s)?)?(?:[_\s\-]+(\d+))?\.txt"
REPLICATE_MODE = 'last'
# ==========================================


def convert_potential(e_raw, re_type, target_scale, ph):
    """Converts the raw potential to SHE or RHE."""
    e_she = e_raw + E_RE_DICT.get(re_type, 0.0)
    if target_scale == 'RHE':
        return e_she + (0.0591 * ph)
    return e_she


_DELIM_RE = re.compile(r'[\s,;\t]+')


def _normalize_cv_header(value):
    return (str(value).strip().lower().replace('μ', 'µ')
            .replace('²', '2').replace(' ', ''))


def _classify_cv_column(value):
    """Return ``(kind, scale, is_density)`` for a CV table header."""
    s = _normalize_cv_header(value)
    current = (
        'current' in s or 'we.current' in s or s.startswith('<i>')
        or s in {'i', 'j'} or s.startswith(('i(', 'i[', 'i/', 'j(', 'j[', 'j/'))
    )
    if current:
        if 'µa' in s or 'ua' in s:
            scale = 1e-6
        elif 'ma' in s:
            scale = 1e-3
        else:
            scale = 1.0
        is_density = (
            '/cm' in s or 'cm2' in s or 'cm^2' in s or 'cm-2' in s
            or 'cm^-2' in s or 'cm⁻2' in s or 'cm**2' in s
            or 'density' in s or s.startswith('j')
        )
        return 'current', scale, is_density

    potential = (
        'potential' in s or 'voltage' in s or 'ewe' in s or 'we.potential' in s
        or s in {'e', 'v', 'u'} or s.startswith(('e(', 'e[', 'e/'))
    )
    if potential:
        return 'potential', (1e-3 if 'mv' in s else 1.0), None
    return None, 1.0, None


def _split_cv_line(line, delimiter=None):
    """Split one CV line while preserving header cells containing spaces."""
    if delimiter is None:
        if '\t' in line:
            delimiter = '\t'
        elif ',' in line:
            delimiter = ','
        elif ';' in line:
            delimiter = ';'
    if delimiter is not None:
        return [cell.strip() for cell in next(csv.reader([line], delimiter=delimiter))]

    raw = line.split()
    # Whitespace tables often spell headers as ``Potential (V)``.  Reattach a
    # parenthesized/bracketed unit so header indices still match numeric rows.
    cells = []
    continuing_unit = False
    for token in raw:
        if continuing_unit and cells:
            cells[-1] += token
            if token.endswith((')', ']')):
                continuing_unit = False
        elif cells and token[:1] in {'(', '['}:
            cells[-1] += token
            continuing_unit = not token.endswith((')', ']'))
        elif cells and token.lower() == 'density' and 'current' in cells[-1].lower():
            cells[-1] += token
        else:
            cells.append(token)
            continuing_unit = (
                token.count('(') > token.count(')')
                or token.count('[') > token.count(']')
            )
    return cells


def _find_cv_columns(headers):
    potential_idx = current_idx = None
    potential_scale = current_scale = 1.0
    is_density = None
    for j, header in enumerate(headers):
        kind, scale, density = _classify_cv_column(header)
        if kind == 'potential' and potential_idx is None:
            potential_idx, potential_scale = j, scale
        elif kind == 'current' and current_idx is None:
            current_idx, current_scale, is_density = j, scale, density
    if potential_idx is None or current_idx is None or potential_idx == current_idx:
        return None
    return potential_idx, current_idx, potential_scale, current_scale, is_density


def _source_cv_unit(header, kind):
    """Read a display unit without changing the source numerical values."""
    if not header:
        return None
    header = str(header).replace('μ', 'µ')
    if kind == 'potential':
        match = re.search(r'(?<![a-z])m?v(?![a-z])', header, re.IGNORECASE)
        return ('mV' if match.group().lower() == 'mv' else 'V') if match else None

    match = re.search(r'(?<![a-z])([munpkµ]?a)(?![a-z])', header, re.IGNORECASE)
    if match is None:
        return None
    prefix = match.group(1)[:-1].lower().replace('u', 'µ')
    unit = prefix + 'A'
    # Normalize equivalent area notation for a concise, unambiguous CSV label.
    suffix = header[match.end():].strip().lower()
    area = re.match(
        r'(?:/\s*(mm|cm|m)\s*(?:\^?2|²|\*\*2)'
        r'|\s*(mm|cm|m)\s*(?:\^?-2|⁻²|⁻2))', suffix,
    )
    if area:
        unit += '/' + (area.group(1) or area.group(2)) + '²'
    return unit


def robust_read_cv_data(filepath, *, allow_absolute_current=False, preserve_units=False):
    """Read potential/current-density columns and convert them to V and A/cm².

    A recognizable header takes precedence over numeric position, so exports
    such as ``time,Ewe,<I>`` no longer treat time as potential.  Potential mV
    and current-density mA/µA prefixes are converted before analysis.  A
    header that explicitly denotes absolute current is rejected because no
    electrode area is available for normalization.  Headerless two-column
    files retain the original potential-first/current-second fallback.

    ``allow_absolute_current`` permits ordinary current for raw processing.
    ``preserve_units`` disables numerical scaling and reports the source units;
    neither option changes the default ECSA import behavior.
    """
    lines = read_text(filepath).splitlines(keepends=True)

    header_idx = -1
    delimiter = None
    column_info = None
    headers = None
    for idx, line in enumerate(lines[:100]):
        if not line.strip():
            continue
        if '\t' in line:
            candidate_delimiter = '\t'
        elif ',' in line:
            candidate_delimiter = ','
        elif ';' in line:
            candidate_delimiter = ';'
        else:
            candidate_delimiter = None
        candidate = _split_cv_line(line.strip(), candidate_delimiter)
        match = _find_cv_columns(candidate)
        if match is not None:
            header_idx = idx
            delimiter = candidate_delimiter
            column_info = match
            headers = candidate
            break

    E_raw_list, i_list = [], []
    if column_info is not None:
        e_idx, i_idx, e_scale, i_scale, is_density = column_info
        if is_density is False and not allow_absolute_current:
            raise ValueError(
                f"Current column '{headers[i_idx]}' is absolute current, but "
                "ECSA analysis requires current density. Divide the current "
                "by the electrode area and label the column A/cm² before importing."
            )
        for line in lines[header_idx + 1:]:
            if not line.strip():
                continue
            parts = _split_cv_line(line.strip(), delimiter)
            if len(parts) <= max(e_idx, i_idx):
                continue
            try:
                e_val = float(parts[e_idx]) * (1.0 if preserve_units else e_scale)
                i_val = float(parts[i_idx]) * (1.0 if preserve_units else i_scale)
            except ValueError:
                continue
            E_raw_list.append(e_val)
            i_list.append(i_val)
    else:
        is_density = None
        headers = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = _DELIM_RE.split(line)
            if len(parts) < 2:
                continue
            try:
                e_val = float(parts[0])
                i_val = float(parts[1])
            except ValueError:
                continue
            E_raw_list.append(e_val)
            i_list.append(i_val)

    result = pd.DataFrame({'E_raw': E_raw_list, 'i': i_list})
    result.attrs.update({
        'potential_column': headers[column_info[0]] if headers else None,
        'current_column': headers[column_info[1]] if headers else None,
        'current_is_density': is_density,
        'potential_unit': (
            _source_cv_unit(headers[column_info[0]], 'potential') if headers else None
        ) if preserve_units else 'V',
        'current_unit': (
            _source_cv_unit(headers[column_info[1]], 'current') if headers else None
        ) if preserve_units else ('A/cm²' if is_density is True else None),
    })
    return result


def _raw_cycle_bounds(e_array):
    """Find repeated vertex-to-vertex loops without imposing a cycle limit."""
    amplitude = float(np.ptp(e_array))
    if len(e_array) < 3 or amplitude == 0:
        return []
    peaks, _ = find_peaks(e_array, prominence=amplitude * 0.5)
    valleys, _ = find_peaks(-e_array, prominence=amplitude * 0.5)
    tolerance = max(amplitude * 0.02, np.finfo(float).eps * 16)

    def candidates(vertices, opposite_vertices, target):
        anchors = [int(index) for index in vertices]
        if abs(e_array[0] - target) <= tolerance:
            anchors.insert(0, 0)
        if abs(e_array[-1] - target) <= tolerance:
            anchors.append(len(e_array) - 1)
        anchors = sorted(set(anchors))
        return [
            (start, end)
            for start, end in zip(anchors, anchors[1:])
            if end - start >= 2
            and np.any((opposite_vertices > start) & (opposite_vertices < end))
        ]

    choices = [
        candidates(valleys, peaks, float(np.min(e_array))),
        candidates(peaks, valleys, float(np.max(e_array))),
    ]
    # Prefer the most complete loops, then the earliest starting vertex.  This
    # also favors endpoint-bounded cycles over the opposite, interior vertices.
    bounds = max(choices, key=lambda bounds: (len(bounds), -bounds[0][0] if bounds else 0))
    if not bounds and abs(e_array[0] - e_array[-1]) <= tolerance:
        # One complete loop can start between the vertices.  In that case no
        # vertex repeats, but returning to the starting potential after visiting
        # both an interior minimum and maximum closes the loop.
        min_index = int(np.argmin(e_array))
        max_index = int(np.argmax(e_array))
        if 0 < min_index < len(e_array) - 1 and 0 < max_index < len(e_array) - 1:
            return [(0, len(e_array) - 1)]
    return bounds


def split_cv_cycles(filepath):
    """Return a wide raw CV table with one adjacent E/i pair per cycle.

    Files may have arbitrary names, absolute current or current density, and
    extra columns such as time.  No potential conversion or unit scaling is
    applied.  Unequal cycle lengths are padded with NaN.  Shared turnaround
    samples belong to the following segment, and the final source sample is
    retained, so concatenating the exported pairs reconstructs the source E/i
    rows exactly once.

    Partial scans are exported as explicitly labeled pairs and described in
    ``attrs['warnings']``.  ``cycle_count`` counts all pairs;
    ``full_cycle_count`` counts complete loops.  ``cycle_bounds`` contains
    half-open source-row intervals.  A trace with no full loop raises ValueError.
    """
    data = robust_read_cv_data(
        filepath, allow_absolute_current=True, preserve_units=True,
    )
    if data.empty:
        raise ValueError('No numeric potential/current data found in the CV file.')
    values = data[['E_raw', 'i']].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('CV potential and current values must all be finite.')
    bounds = _raw_cycle_bounds(values[:, 0])
    if not bounds:
        raise ValueError('No complete CV cycles detected; the potential must sweep out and back.')

    warnings = []
    potential_unit = data.attrs['potential_unit']
    current_unit = data.attrs['current_unit']
    if potential_unit is None or current_unit is None:
        warnings.append('Some source units are unspecified; their columns are labeled raw.')

    segments = []
    if bounds[0][0] > 0:
        segments.append(('Leading partial', 0, bounds[0][0], False))
        warnings.append('A leading partial scan is retained in a separate E/i pair.')
    cursor = bounds[0][0]
    interior_partial_count = 0
    for index, (start, end) in enumerate(bounds, 1):
        if start > cursor:
            # Smaller excursions between full-range loops may not pass the
            # prominence threshold.  Keep their source rows too.
            interior_partial_count += 1
            segments.append((f'Partial scan {interior_partial_count}', cursor, start, False))
            warnings.append('An intermediate partial scan is retained in a separate E/i pair.')
        # The final endpoint belongs to the final full cycle when no trailing
        # scan follows it.  Otherwise it starts the next exported segment.
        stop = end + 1 if end == len(data) - 1 else end
        segments.append((f'Cycle {index}', start, stop, True))
        cursor = stop
    if bounds[-1][1] < len(data) - 1:
        segments.append(('Trailing partial', bounds[-1][1], len(data), False))
        warnings.append('A trailing partial scan is retained in a separate E/i pair.')

    parts = []
    for label, start, stop, _complete in segments:
        parts.append(pd.DataFrame({
            f'{label} E ({potential_unit or "raw"})': values[start:stop, 0],
            f'{label} i ({current_unit or "raw"})': values[start:stop, 1],
        }))
    result = pd.concat(parts, axis=1)
    result.attrs.update(data.attrs)
    result.attrs.update({
        'cycle_count': len(segments),
        'full_cycle_count': len(bounds),
        'partial_cycle_count': len(segments) - len(bounds),
        'source_row_count': len(data),
        'cycle_bounds': [(start, stop) for _, start, stop, _ in segments],
        'segments': [
            {'label': label, 'start': start, 'stop': stop, 'complete': complete}
            for label, start, stop, complete in segments
        ],
        'warnings': warnings,
    })
    return result


def parse_scan_rate(filename):
    """Extract (scan_rate_mV_per_s, replicate_index) from a CV filename."""
    m = re.match(FILENAME_PATTERN, os.path.basename(filename))
    if m is None:
        return None, None
    rate = float(m.group(1))
    rep = int(m.group(2)) if m.group(2) is not None else 1
    return rate, rep


def detect_cycle_bounds(e_array):
    """Identify full CV cycles, including cycles bounded by endpoints.

    ``scipy.find_peaks`` intentionally excludes endpoints.  Since ordinary CV
    files often start and finish at their lower (or upper) vertex, relying on
    interior valleys alone discarded valid one- and two-cycle traces.
    """
    e_array = np.asarray(e_array, dtype=float)
    if len(e_array) < 2:
        return []
    if not np.isfinite(e_array).all():
        return []
    amplitude = np.max(e_array) - np.min(e_array)
    if amplitude == 0:
        return []
    min_separation = max(1, len(e_array) // 100)
    valleys, _ = find_peaks(-e_array, prominence=amplitude * 0.5, distance=min_separation)
    peaks, _ = find_peaks(e_array, prominence=amplitude * 0.5, distance=min_separation)

    edge_tolerance = max(amplitude * 0.05, np.finfo(float).eps * 16)

    def with_matching_endpoints(interior, target):
        bounds = [int(x) for x in interior]
        if abs(e_array[0] - target) <= edge_tolerance:
            bounds.insert(0, 0)
        if abs(e_array[-1] - target) <= edge_tolerance:
            bounds.append(len(e_array) - 1)
        bounds = sorted(set(bounds))
        return [(a, b) for a, b in zip(bounds, bounds[1:]) if b - a >= 2]

    valley_cycles = with_matching_endpoints(valleys, float(np.min(e_array)))
    peak_cycles = with_matching_endpoints(peaks, float(np.max(e_array)))
    return peak_cycles if len(peak_cycles) > len(valley_cycles) else valley_cycles


def process_one_file(filepath, re_type=None, target_scale=None, ph=None):
    """Read raw CV, convert potential, detect cycles. Returns {cycle_idx: DataFrame}."""
    re_type = RE_TYPE if re_type is None else re_type
    target_scale = TARGET_SCALE if target_scale is None else target_scale
    ph = PH if ph is None else ph

    df = robust_read_cv_data(filepath)
    if df.empty:
        return None
    e_converted = convert_potential(df['E_raw'], re_type, target_scale, ph).values
    i_array = df['i'].values
    bounds = detect_cycle_bounds(e_converted)
    if not bounds:
        return None
    cycles = {}
    for j, (start, end) in enumerate(bounds):
        cycles[j] = pd.DataFrame({
            'E': e_converted[start:end + 1],
            'i': i_array[start:end + 1],
        })
    return cycles


def select_replicates(files_by_rate, mode):
    """For each scan rate, choose which replicate file(s) feed the ECSA output."""
    chosen = {}
    for rate, group in files_by_rate.items():
        sorted_group = sorted(group, key=lambda x: x[0])
        if mode == 'last':
            chosen[rate] = [sorted_group[-1][1]]
        elif mode == 'all':
            chosen[rate] = [fp for _, fp in sorted_group]
        elif mode == 'mean':
            print(f"  Notice: REPLICATE_MODE='mean' is not yet implemented "
                  f"(needs grid alignment); using last replicate for {rate} mV/s.")
            chosen[rate] = [sorted_group[-1][1]]
        else:
            raise ValueError(f"Unknown REPLICATE_MODE: {mode!r}")
    return chosen


def _resolve_cycle_index(cycles, cycle_index, filepath=None):
    available = sorted(cycles.keys())
    if cycle_index < 0:
        try:
            return available[cycle_index]
        except IndexError:
            if filepath is not None:
                print(f"  Warning: cycle index {cycle_index} out of range for "
                      f"{os.path.basename(filepath)}; using last available.")
            return available[-1]
    if cycle_index in cycles:
        return cycle_index
    if filepath is not None:
        print(f"  Warning: cycle {cycle_index} not in "
              f"{os.path.basename(filepath)} (has {len(cycles)} cycles); "
              f"using last available.")
    return available[-1]


def build_ecsa_dataset(chosen_files, per_file_cycles, cycle_index):
    """Build long-format DataFrame [scan_rate, potential, current] for ECSA."""
    parts = []
    for rate, filepaths in chosen_files.items():
        for fp in filepaths:
            cycles = per_file_cycles.get(fp)
            if not cycles:
                continue
            idx = _resolve_cycle_index(cycles, cycle_index, fp)
            cyc_df = cycles[idx]
            parts.append(pd.DataFrame({
                'scan_rate': rate,
                'potential': cyc_df['E'].values,
                'current': cyc_df['i'].values,
            }))
    if not parts:
        return pd.DataFrame(columns=['scan_rate', 'potential', 'current'])
    return pd.concat(parts, ignore_index=True)[['scan_rate', 'potential', 'current']]


def build_time_series_wide(file_order, per_file_cycles, cycle_index, target_scale=TARGET_SCALE):
    """Wide-format DataFrame: two columns per file (E, i), MultiIndex header."""
    parts = []
    for fp in file_order:
        cycles = per_file_cycles.get(fp)
        if not cycles:
            continue
        idx = _resolve_cycle_index(cycles, cycle_index, fp)
        cyc_df = cycles[idx]
        label = os.path.splitext(os.path.basename(fp))[0]
        cols = pd.MultiIndex.from_tuples([
            (label, f"E (V vs {target_scale})"),
            (label, "i (A/cm²)"),
        ])
        parts.append(pd.DataFrame(
            np.column_stack([cyc_df['E'].values, cyc_df['i'].values]),
            columns=cols,
        ))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, axis=1)
