"""C_dl / ECSA computation, extracted verbatim from ECSA_nonfaradic/CV_viewer.py
(lines 43-112). Self-contained: pandas/numpy only.
"""
import numpy as np
import pandas as pd


def compute_cdl_ecsa(long_df, Cs_uF_per_cm2=None, n_nearest=5):
    """Compute C_dl from a long-format CV DataFrame.

    long_df : columns [scan_rate (mV/s), potential (V), current (A/cm²)].
    Returns dict with records, Cdl, intercept, ECSA, r2, warnings.
    """
    warnings = []
    grouped = long_df.groupby('scan_rate')
    if grouped.ngroups < 2:
        raise ValueError("Need at least two distinct scan rates for a fit.")

    records = []
    for rate, grp in grouped:
        e_min, e_max = grp['potential'].min(), grp['potential'].max()
        mid_pt = (e_min + e_max) / 2
        grp2 = grp.assign(diff=(grp['potential'] - mid_pt).abs())
        nearest = grp2.nsmallest(n_nearest, 'diff')
        pos = nearest[nearest['current'] > 0]
        neg = nearest[nearest['current'] < 0]
        if not pos.empty and not neg.empty:
            ja = pos['current'].max()
            jc = neg['current'].min()
        else:
            ja = nearest['current'].max()
            jc = nearest['current'].min()
            warnings.append(
                f"{rate:g} mV/s: no opposite-sign currents at E={mid_pt:.4f} V → j_dl unreliable"
            )
        jdl = (ja - jc) / 2
        records.append({'scan_rate_mV_per_s': float(rate), 'jdl_A_per_cm2': jdl})

    res = (pd.DataFrame(records)
           .sort_values('scan_rate_mV_per_s')
           .replace([np.inf, -np.inf], np.nan)
           .dropna(subset=['jdl_A_per_cm2'])
           .reset_index(drop=True))
    if len(res) < 2:
        raise ValueError("Fewer than two usable j_dl points after processing.")

    vs_Vs = res['scan_rate_mV_per_s'].values * 1e-3
    jdl_vals = res['jdl_A_per_cm2'].values
    slope, intercept = np.polyfit(vs_Vs, jdl_vals, 1)
    fitted = np.poly1d((slope, intercept))(vs_Vs)
    ss_res = np.sum((jdl_vals - fitted) ** 2)
    ss_tot = np.sum((jdl_vals - jdl_vals.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    Cdl = float(slope)
    ECSA = (Cdl / (Cs_uF_per_cm2 * 1e-6)) if Cs_uF_per_cm2 else None

    res['jdl_fit_A_per_cm2'] = fitted
    res['Cdl_F_per_cm2'] = Cdl
    res['ECSA_cm2'] = ECSA if ECSA is not None else np.nan

    return {'records': res, 'Cdl': Cdl, 'intercept': float(intercept),
            'ECSA': ECSA, 'r2': float(r2), 'warnings': warnings}
