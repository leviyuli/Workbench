"""LSV and EIS readers with column detection, unit scaling and sweep handling."""
import csv
import os
import re
import numpy as np
import pandas as pd

from workbench.core.io import read_excel as _read_excel, read_text as _read_text


# ---------------------------------------------------------------------------
# LSV column / unit detection
# ---------------------------------------------------------------------------
_V_PATTERNS = [
    ('e(mv', 1e-3),
    ('e/mv', 1e-3),
    ('potential(mv)', 1e-3),
    ('voltage(mv)', 1e-3),
    ('ewe(mv)', 1e-3),
    ('e(v', 1.0),
    ('e/v', 1.0),
    ('potential', 1.0),
    ('voltage', 1.0),
    ('we.potential', 1.0),
    ('wepot', 1.0),
    ('ewe', 1.0),
]

_I_PATTERNS = [
    ('current(µa', 1e-6),
    ('current[µa', 1e-6),
    ('current/µa', 1e-6),
    ('current(ua', 1e-6),
    ('current[ua', 1e-6),
    ('current/ua', 1e-6),
    ('i(µa', 1e-6),
    ('j(µa', 1e-6),
    ('i(ua', 1e-6),
    ('j(ua', 1e-6),
    ('µa/cm', 1e-6),
    ('ua/cm', 1e-6),
    ('current(ma', 1e-3),
    ('current[ma', 1e-3),
    ('current/ma', 1e-3),
    ('i(ma', 1e-3),
    ('j(ma', 1e-3),
    ('i/ma', 1e-3),
    ('j/ma', 1e-3),
    ('ma/cm', 1e-3),
    ('current(a', 1.0),
    ('current[a', 1.0),
    ('current/a', 1.0),
    ('i(a', 1.0),
    ('j(a', 1.0),
    ('i/a', 1.0),
    ('j/a', 1.0),
    ('a/cm', 1.0),
    ('we.current', 1.0),
    ('current', 1.0),
]

_BARE_V = {'e', 'v', 'u'}
_BARE_I = {'i', 'j'}


def _classify_column(col_name):
    s = (str(col_name).strip().lower().replace('μ', 'µ')
         .replace(' ', ''))
    if s in _BARE_V:
        return ('V', 1.0, None)
    voltage_like = (
        s.startswith(('e(', 'e[', 'e/', 'ewe(', 'ewe[', 'ewe/'))
        or 'potential' in s or 'voltage' in s or 'wepot' in s
    )
    if voltage_like:
        return ('V', 1e-3 if 'mv' in s else 1.0, None)

    if s in _BARE_I:
        return ('i', 1.0, None)
    for pat, scale in _V_PATTERNS:
        if pat in s:
            return ('V', scale, None)
    is_density = ('/cm' in s) or ('cm²' in s) or ('cm^2' in s) \
        or ('cm-2' in s) or ('cm^-2' in s) or ('cm⁻²' in s)
    current_like = (
        'current' in s or s.startswith(('<i>', 'i(', 'i[', 'i/',
                                            'j(', 'j[', 'j/'))
        or 'a/cm' in s
    )
    if current_like:
        if 'µa' in s or 'ua' in s:
            scale = 1e-6
        elif 'ma' in s:
            scale = 1e-3
        else:
            scale = 1.0
        if is_density or s.startswith('j'):
            return ('i', scale, True)
        return ('i', scale, False)

    for pat, scale in _I_PATTERNS:
        if pat in s:
            # i denotes absolute current; j denotes current density.
            if is_density:
                return ('i', scale, True)
            if pat.startswith('j'):
                return ('i', scale, True)
            if pat.startswith('i') or 'current' in pat:
                return ('i', scale, False)
            return ('i', scale, None)
    return (None, 1.0, None)


def _find_cols(columns):
    strong_v = strong_i = bare_v = bare_i = None
    for j, c in enumerate(columns):
        s = str(c).strip().lower()
        is_bare = s in _BARE_V or s in _BARE_I
        kind, _, _ = _classify_column(c)
        if kind == 'V':
            if is_bare and bare_v is None:
                bare_v = j
            elif not is_bare and strong_v is None:
                strong_v = j
        elif kind == 'i':
            if is_bare and bare_i is None:
                bare_i = j
            elif not is_bare and strong_i is None:
                strong_i = j
    v_idx = strong_v if strong_v is not None else bare_v
    i_idx = strong_i if strong_i is not None else bare_i
    return v_idx, i_idx


def _split_lsv_header(line):
    """Split an LSV header without separating names from spaced units."""
    for delimiter in ('\t', ',', ';'):
        if delimiter in line:
            try:
                return (
                    [part.strip() for part in next(csv.reader([line], delimiter=delimiter))],
                    delimiter,
                )
            except csv.Error:
                return [part.strip() for part in line.split(delimiter)], delimiter

    cells = []
    continuing_unit = False
    for token in line.split():
        if continuing_unit and cells:
            cells[-1] += token
            if token.endswith((')', ']')):
                continuing_unit = False
        elif cells and token.startswith(('(', '[')):
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
    return cells, r'\s+'


def _find_header_row(lines, max_scan=80):
    for idx, line in enumerate(lines[:max_scan]):
        tokens, _delimiter = _split_lsv_header(line)
        if len(tokens) < 2:
            continue
        v_idx, i_idx = _find_cols(tokens)
        if v_idx is not None and i_idx is not None and v_idx != i_idx:
            return idx
    return -1


def _trim_to_forward_sweep(V, i):
    if len(V) < 3:
        return V, i, False
    dV = np.diff(V)
    pos = int(np.sum(dV > 0))
    neg = int(np.sum(dV < 0))
    if pos > 0 and neg > 0:
        apex = int(np.argmax(V)) if pos >= neg else int(np.argmin(V))
        if 0 < apex < len(V) - 1:
            return V[:apex + 1], i[:apex + 1], True
    return V, i, False


def _read_single_lsv(filepath, trim_sweep=True, preserve_invalid=False):
    info = {'file': os.path.basename(filepath)}
    try:
        ext = os.path.splitext(filepath)[1].lower()
        if ext in ('.xlsx', '.xls'):
            df_raw = _read_excel(filepath, header=None)
            header_idx = -1
            for r in range(min(len(df_raw), 20)):
                row = [str(x) for x in df_raw.iloc[r].tolist()]
                v_idx, i_idx = _find_cols(row)
                if v_idx is not None and i_idx is not None and v_idx != i_idx:
                    header_idx = r
                    break
            if header_idx >= 0:
                df = df_raw.iloc[header_idx + 1:].reset_index(drop=True)
                df.columns = [str(x) for x in df_raw.iloc[header_idx].tolist()]
            else:
                df = _read_excel(filepath)
        elif ext in ('.csv', '.txt'):
            lines = _read_text(filepath).splitlines(keepends=True)
            header_idx = _find_header_row(lines)
            default_delim = ',' if ext == '.csv' else r'\s+'
            if header_idx >= 0:
                headers, delim = _split_lsv_header(lines[header_idx])
                df = pd.read_csv(
                    filepath, sep=delim, skiprows=header_idx + 1, header=None,
                    engine='python', on_bad_lines='skip',
                )
                if df.shape[1] != len(headers):
                    return None, None, info, (
                        f"Header has {len(headers)} column(s), but data has "
                        f"{df.shape[1]}; check the delimiter and header units."
                    )
                df.columns = headers
            else:
                df = pd.read_csv(filepath, sep=default_delim, engine='python',
                                 header=None, on_bad_lines='skip')
        else:
            return None, None, info, f"Unsupported file extension: {ext}. Allowed: .txt, .csv, .xlsx"

        if df.shape[1] < 2:
            return None, None, info, "File must have at least two columns (Voltage, Current)."

        v_idx, i_idx = _find_cols(df.columns)
        if v_idx is None or i_idx is None:
            v_idx, i_idx = 0, 1
            v_name, i_name = 'col0', 'col1'
            v_scale, i_scale, is_density = 1.0, 1.0, True
        else:
            v_name = str(df.columns[v_idx])
            i_name = str(df.columns[i_idx])
            _, v_scale, _ = _classify_column(v_name)
            _, i_scale, is_density = _classify_column(i_name)
            if is_density is None:
                is_density = True

        V = pd.to_numeric(df.iloc[:, v_idx], errors='coerce') * v_scale
        i = pd.to_numeric(df.iloc[:, i_idx], errors='coerce') * i_scale
        invalid_rows = 0
        if preserve_invalid:
            # Invalid rows separate acquired segments during interpolation.
            V = V.to_numpy(dtype=float)
            i = i.to_numpy(dtype=float)
            finite = np.isfinite(V) & np.isfinite(i)
            invalid_rows = int(np.count_nonzero(~finite))
            no_numeric_rows = not np.any(finite)
        else:
            mask = V.notna() & i.notna()
            V = V[mask].to_numpy(dtype=float)
            i = i[mask].to_numpy(dtype=float)
            no_numeric_rows = len(V) == 0
        if no_numeric_rows:
            return None, None, info, "No numeric rows found in the data table."

        trimmed = False
        if trim_sweep:
            V, i, trimmed = _trim_to_forward_sweep(V, i)
        info.update({
            'v_col': v_name, 'i_col': i_name,
            'v_scale': v_scale, 'i_scale': i_scale,
            'is_density': bool(is_density), 'sweep_trimmed': trimmed,
            'n_rows': len(V),
        })
        if preserve_invalid:
            info['invalid_rows'] = invalid_rows
        if is_density is False:
            info['warning'] = (
                f"Column '{i_name}' looks like absolute current (no /cm² in header); "
                "an electrode area is required before current-density analysis."
            )
        return V, i, info, None
    except Exception as e:
        return None, None, info, f"Error reading {info['file']}: {e}"


def read_lsv_curve(filepath):
    """Return ``((potential, current, info), error)`` for a complete LSV curve.

    Shared column and unit detection retains acquisition order,
    reverse sweeps, zero currents and both current signs. Invalid measurement
    rows retain their positions as nonfinite values to separate curve segments.
    """
    if filepath is None or not str(filepath).strip():
        return None, "No file selected."
    potential, current, info, error = _read_single_lsv(
        str(filepath), trim_sweep=False, preserve_invalid=True,
    )
    if error is not None:
        return None, error
    return (potential, current, info), None


def read_lsv_data(filepaths, reaction="auto"):
    """Load LSV data from one or more files. Returns ((V, i, info), err).

    ``reaction`` selects the expected current sign:
      - "auto" → infer from the dominant sign of the current (OER if mostly
        positive, HER if mostly negative);
      - "oer"  → keep only positive current (anodic);
      - "her"  → keep only negative current (cathodic).
    The current sign is preserved in the returned data; the resolved reaction is
    reported in ``info['reaction']`` (and ``info['reaction_auto']``).
    """
    if filepaths is None:
        return None, "No file selected."
    reaction = str(reaction).strip().lower()
    paths = [filepaths] if isinstance(filepaths, str) else [str(p) for p in filepaths if p]
    if not paths:
        return None, "No file selected."

    series, file_infos, warns = [], [], []
    for fp in paths:
        V, i, info, err = _read_single_lsv(fp)
        file_infos.append(info)
        if err is not None:
            warns.append(err)
            continue
        if 'warning' in info:
            warns.append(f"{info['file']}: {info['warning']}")
        order = np.argsort(V)
        series.append((V[order], i[order], info))

    if not series:
        return None, "Could not read any file:\n" + "\n".join(warns)

    density_states = {
        bool(item[2].get('is_density', True))
        for item in series
    }
    if len(density_states) > 1:
        return None, (
            "Cannot average a mixture of absolute-current and current-density "
            "LSV files. Normalize every file by electrode area first, or use "
            "files with consistent units."
        )

    if len(series) == 1:
        V_out, i_out, base_info = series[0]
        averaged = False
    else:
        v_lo = max(s[0].min() for s in series)
        v_hi = min(s[0].max() for s in series)
        if v_hi <= v_lo:
            return None, "Selected files have no overlapping voltage range."
        n_med = max(int(np.median([len(s[0]) for s in series])), 50)
        V_out = np.linspace(v_lo, v_hi, n_med)
        stack = np.stack([np.interp(V_out, s[0], s[1]) for s in series])
        i_out = stack.mean(axis=0)
        base_info = series[0][2]
        averaged = True

    finite = np.isfinite(V_out) & np.isfinite(i_out)
    auto = reaction not in ('oer', 'her')
    if auto:
        i_fin = i_out[finite]
        n_pos = int(np.sum(i_fin > 0))
        n_neg = int(np.sum(i_fin < 0))
        resolved = 'her' if n_neg > n_pos else 'oer'
    else:
        resolved = reaction

    if resolved == 'her':
        mask = finite & (i_out < 0)
    else:
        mask = finite & (i_out > 0)
    V_clean = V_out[mask]
    i_clean = i_out[mask]
    if len(V_clean) == 0:
        return None, (
            "No current data of the expected sign after filtering "
            "(Reaction = OER expects i > 0, HER expects i < 0). "
            "Check the Reaction selector."
        )

    info = {
        'files': file_infos,
        'v_col': base_info.get('v_col', 'E'),
        'i_col': base_info.get('i_col', 'i'),
        'is_density': base_info.get('is_density', True),
        'sweep_trimmed': any(s[2].get('sweep_trimmed', False) for s in series),
        'averaged': averaged,
        'reaction': resolved,
        'reaction_auto': auto,
        'n_rows': len(V_clean),
    }
    if warns:
        info['warning'] = ' | '.join(warns)
    return (V_clean, i_clean, info), None


def _normalize_eis_header(value):
    """Return a compact, ASCII-like form for EIS header matching."""
    return (str(value).strip().lower()
            .replace('−', '-').replace('–', '-').replace('—', '-')
            .replace('″', "''").replace('”', '"')
            .replace('′', "'").replace('’', "'")
            .replace(' ', ''))


def _classify_eis_header(value):
    """Classify a header as frequency, real Z, or imaginary Z.

    The second tuple item is the multiplier needed to turn a displayed
    imaginary column into canonical Im(Z).  Instruments commonly export a
    positive ``-Z''``/``-Im(Z)`` column; those values therefore need -1.
    """
    s = _normalize_eis_header(value)
    if 'freq' in s or 'frequency' in s or s in {'f', 'f(hz)', 'f/hz'}:
        return 'frequency', 1.0

    imag = ("z''" in s or 'z"' in s or 'zimag' in s or 'zimaginary' in s
            or 'im(z)' in s or 'imag(z)' in s or 'imaginary(z)' in s
            or s.startswith('imz') or s.startswith('-imz')
            or s.startswith('zim') or s.startswith('-zim')
            or s in {'im', '-im'} or s.startswith('im(') or s.startswith('-im('))
    if imag:
        return 'z_imag', (-1.0 if s.startswith('-') else 1.0)

    real = ("z'" in s or 'zreal' in s or 'zrealpart' in s
            or 're(z)' in s or 'real(z)' in s or s.startswith('rez')
            or s.startswith('zre') or s in {'zr', 're'})
    if real:
        return 'z_real', 1.0
    return None, 1.0


def _find_eis_columns(headers):
    found = {}
    imag_multiplier = 1.0
    for j, header in enumerate(headers):
        kind, multiplier = _classify_eis_header(header)
        if kind is not None and kind not in found:
            found[kind] = j
            if kind == 'z_imag':
                imag_multiplier = multiplier
    if {'frequency', 'z_real', 'z_imag'} <= found.keys():
        return (found['frequency'], found['z_real'], found['z_imag'],
                imag_multiplier)
    return None


def _eis_unit_kind(headers):
    """Classify impedance units as area-normalized, raw ohms, or unknown."""
    kinds = set()
    for header in headers:
        s = _normalize_eis_header(header)
        has_ohm = 'ohm' in s or 'ω' in s
        has_area = any(token in s for token in (
            'cm2', 'cm^2', 'cm²', 'cm-2', 'cm⁻²', 'cm**2',
        ))
        if has_ohm and has_area:
            kinds.add('ohm_cm2')
        elif has_ohm:
            kinds.add('ohm')
    if len(kinds) > 1:
        return 'inconsistent'
    return next(iter(kinds), 'unknown')


def _eis_impedance_scale(header):
    """Convert an explicitly prefixed impedance unit to base Ω or Ω·cm².

    Prefix matching is case-sensitive: mΩ and MΩ differ by nine orders of magnitude.
    """
    match = re.search(
        r'([A-Za-zµμ]*)(?:(?i:ohms?)|Ω|ω)',
        str(header).replace('Ω', 'Ω'),
    )
    if match is None:
        return 1.0
    prefix = match.group(1)
    scales = {'': 1.0, 'm': 1e-3, 'u': 1e-6, 'µ': 1e-6, 'μ': 1e-6,
              'k': 1e3, 'K': 1e3, 'M': 1e6}
    if prefix not in scales:
        raise ValueError(
            f"Unsupported impedance unit prefix '{prefix}' in header '{header}'. "
            "Use Ω, mΩ, µΩ, kΩ or MΩ."
        )
    return scales[prefix]


def _split_eis_header(line):
    if '\t' in line:
        return [part.strip() for part in line.rstrip('\r\n').split('\t')], '\t'
    if ',' in line:
        return [part.strip() for part in next(csv.reader([line]))], ','

    # Preserve parenthesized units in whitespace-delimited headers.  Numeric
    # rows still have one token per column, so this keeps the indices aligned.
    cells = []
    unit_parts = []
    for token in line.split():
        if unit_parts:
            unit_parts.append(token)
            if token.endswith((')', ']')):
                cells[-1] += ''.join(unit_parts)
                unit_parts = []
        elif token.startswith(('(', '[')) and cells:
            if token.endswith((')', ']')):
                cells[-1] += token
            else:
                unit_parts = [token]
        else:
            cells.append(token)
    if unit_parts and cells:
        cells[-1] += ''.join(unit_parts)
    return cells, r'\s+'


def read_eis_data(filepath, *, area_cm2=None):
    """Read EIS data and return canonical complex impedance.

    ``z_imag`` is Im(Z), including for ``-Z''`` and ``-Im(Z)`` source columns.
    Raw ohms require a positive ``area_cm2`` for conversion to Ω·cm²; values
    labelled in Ω·cm² retain their area normalization. SI prefixes are scaled
    before area conversion. Headerless files assume Ω·cm² and report a warning.
    """
    try:
        ext = os.path.splitext(filepath)[1].lower()
        headers = None
        cols = None
        zr_scale = zi_scale = 1.0

        if ext in ('.xlsx', '.xls'):
            df = _read_excel(filepath, header=None)
            for i, row in df.iterrows():
                candidate = [str(x) for x in row.values]
                match = _find_eis_columns(candidate)
                if match is not None:
                    cols = match
                    headers = candidate
                    df = df.iloc[i + 1:].copy()
                    break
        elif ext in ('.csv', '.txt'):
            lines = _read_text(filepath).splitlines(keepends=True)
            header_idx = -1
            delimiter = ',' if ext == '.csv' else r'\s+'
            matched_headers = None
            matched_cols = None
            for i, line in enumerate(lines[:100]):
                candidate, candidate_delimiter = _split_eis_header(line)
                match = _find_eis_columns(candidate)
                if match is not None:
                    header_idx = i
                    delimiter = candidate_delimiter
                    matched_headers = candidate
                    matched_cols = match
                    break
            if header_idx >= 0:
                df = pd.read_csv(filepath, sep=delimiter, skiprows=header_idx + 1,
                                 header=None, engine='python', on_bad_lines='skip')
                headers = matched_headers
                cols = matched_cols
            else:
                df = pd.read_csv(filepath, sep=delimiter, engine='python',
                                 header=None, on_bad_lines='skip')
        else:
            return None, (
                f"Unsupported file extension: {ext}. "
                "Allowed: .txt, .csv, .xlsx, .xls"
            )

        if cols is not None:
            freq_col, zr_col, zi_col, imag_multiplier = cols
            unit_kind = _eis_unit_kind([headers[zr_col], headers[zi_col]])
            if unit_kind == 'inconsistent':
                return None, "Real and imaginary impedance columns use inconsistent units."
            if unit_kind == 'ohm' and area_cm2 is None:
                return None, (
                    "Impedance is labelled in raw Ω, but this EIS analysis reports "
                    "area-normalized Ω·cm². Multiply Z by the electrode area "
                    "and label the columns Ω·cm² before importing."
                )
            zr_scale = _eis_impedance_scale(headers[zr_col])
            zi_scale = _eis_impedance_scale(headers[zi_col])
            freq = pd.to_numeric(df.iloc[:, freq_col], errors='coerce')
            z_real = pd.to_numeric(df.iloc[:, zr_col], errors='coerce') * zr_scale
            z_imag = (pd.to_numeric(df.iloc[:, zi_col], errors='coerce')
                      * imag_multiplier * zi_scale)
        else:
            df_numeric = df.apply(pd.to_numeric, errors='coerce').dropna(thresh=3)
            if df_numeric.empty:
                return None, "No valid numeric data or recognizable headers found."
            freq = df_numeric.iloc[:, 0]
            z_real = df_numeric.iloc[:, 1]
            z_imag = df_numeric.iloc[:, 2]
            imag_multiplier = 1.0
            unit_kind = 'unknown'

        mask = freq.notna() & z_real.notna() & z_imag.notna()
        freq_clean = freq[mask].to_numpy(dtype=float)
        z_real_clean = z_real[mask].to_numpy(dtype=float)
        z_imag_clean = z_imag[mask].to_numpy(dtype=float)
        if len(freq_clean) == 0:
            return None, "No valid EIS data points found after parsing."

        source_unit = unit_kind
        applied_area = None
        if unit_kind == 'ohm':
            try:
                applied_area = float(area_cm2)
            except (TypeError, ValueError):
                return None, "Raw Ω impedance requires a finite electrode area greater than zero (cm²)."
            if not np.isfinite(applied_area) or applied_area <= 0:
                return None, "Raw Ω impedance requires a finite electrode area greater than zero (cm²)."
            z_real_clean = z_real_clean * applied_area
            z_imag_clean = z_imag_clean * applied_area
            unit_kind = 'ohm_cm2'

        z_complex = z_real_clean + 1j * z_imag_clean
        result = {
            'frequency': freq_clean,
            'z_real': z_real_clean,
            'z_imag': z_imag_clean,
            'z_mod': np.abs(z_complex),
            'phase': np.degrees(np.arctan2(z_imag_clean, z_real_clean)),
            'z_complex': z_complex,
            'impedance_unit': unit_kind,
            'source_impedance_unit': source_unit,
            'source_impedance_scale_real': zr_scale,
            'source_impedance_scale_imag': zi_scale,
            'area_cm2': applied_area,
            'imaginary_convention': '-Im(Z)' if imag_multiplier < 0 else 'Im(Z)',
        }
        if unit_kind == 'unknown':
            result['warning'] = (
                "Impedance units were not identified from the headers; values "
                "are being assumed to be area-normalized Ω·cm²."
            )
        return result, None
    except Exception as e:
        return None, f"Error reading the EIS file:\n{e}"
