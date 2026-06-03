"""CV pipeline helpers, copied verbatim from ECSA_nonfaradic/CV_process.py.

Reader, filename parsing, potential conversion, cycle detection, replicate
selection, and long/wide dataset builders. Kept as-is so ECSA results match the
standalone program exactly.
"""
import os
import re
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

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


def robust_read_cv_data(filepath):
    """Tolerant reader: keep lines whose first two tokens parse as floats."""
    E_raw_list, i_list = [], []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
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
    return pd.DataFrame({'E_raw': E_raw_list, 'i': i_list})


def parse_scan_rate(filename):
    """Extract (scan_rate_mV_per_s, replicate_index) from a CV filename."""
    m = re.match(FILENAME_PATTERN, os.path.basename(filename))
    if m is None:
        return None, None
    rate = float(m.group(1))
    rep = int(m.group(2)) if m.group(2) is not None else 1
    return rate, rep


def detect_cycle_bounds(e_array):
    """Identify cycle boundaries as valleys in the potential trace."""
    e_array = np.asarray(e_array)
    if len(e_array) < 2:
        return []
    amplitude = np.max(e_array) - np.min(e_array)
    if amplitude == 0:
        return []
    min_separation = max(1, len(e_array) // 100)
    valleys, _ = find_peaks(-e_array, prominence=amplitude * 0.5, distance=min_separation)
    if len(valleys) < 2:
        return []
    return [(int(valleys[j]), int(valleys[j + 1])) for j in range(len(valleys) - 1)]


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
