"""Iterative Tafel decoupling, copied verbatim from
LSV_decoupling/lsv_decoupling/logic.py. Pure numpy/scipy.
"""
import numpy as np
from scipy.stats import linregress

# ---------------------------
# Constants
# ---------------------------
WINDOW_WIDTH = 0.06  # 60 mV candidate window for Tafel fitting
GRID_STEP = 0.005    # Voltage step used in grid search


def fit_tafel_window(eta_corr, i, i_lower=0.005, i_upper=0.1, window_width=WINDOW_WIDTH, step=GRID_STEP):
    """Grid-search a 60-mV candidate window; test 20/40/60 mV subwindows.
    Returns: (best_result_dict, error_string)."""
    mask_range = (i >= i_lower) & (i <= i_upper)
    if np.sum(mask_range) < 5:
        return None, f"Not enough points in the chosen range: {i_lower:.4g}–{i_upper:.4g} A/cm²."

    eta_sel = eta_corr[mask_range]
    i_sel = i[mask_range]
    vmin, vmax = np.min(eta_sel), np.max(eta_sel)

    if vmax - vmin < window_width:
        return None, "Voltage span too narrow for a 60 mV window."

    candidate_results = []
    for start in np.arange(vmin, vmax - window_width, step):
        end = start + window_width
        cand_mask = (eta_sel >= start) & (eta_sel <= end)
        if np.sum(cand_mask) < 3:
            continue

        center = 0.5 * (start + end)
        subwindows = {
            "20mV": (center - 0.01, center + 0.01),
            "40mV": (center - 0.02, center + 0.02),
            "60mV": (start, end),
        }

        slopes, intercepts, r2s = [], [], []
        counts = {}
        for key, (low, high) in subwindows.items():
            sub_mask = (eta_sel >= low) & (eta_sel <= high)
            n = int(np.sum(sub_mask))
            counts[key] = n
            if n < 3:
                slopes.append(np.nan)
                intercepts.append(np.nan)
                r2s.append(np.nan)
                continue
            i_sub = i_sel[sub_mask]
            eta_sub = eta_sel[sub_mask]
            res = linregress(np.log10(i_sub), eta_sub)
            slopes.append(res.slope)
            intercepts.append(res.intercept)
            r2s.append(res.rvalue ** 2)

        slopes = np.array(slopes)
        if np.any(np.isnan(slopes)):
            continue

        i0s = np.array([10 ** (-intercepts[j] / slopes[j]) for j in range(len(slopes))])
        avg_slope = float(np.mean(slopes))
        std_slope = float(np.std(slopes))
        avg_intercept = float(np.mean(intercepts))
        avg_i0 = float(np.mean(i0s))
        std_i0 = float(np.std(i0s))

        rel_err_b = std_slope / abs(avg_slope) if avg_slope != 0 else np.inf
        rel_err_i0 = std_i0 / avg_i0 if avg_i0 != 0 else np.inf
        metric = 0.5 * (rel_err_b + rel_err_i0)
        i_window = i_sel[cand_mask]

        candidate_results.append({
            "window": (start, end),
            "current_range": (float(np.min(i_window)), float(np.max(i_window))),
            "center": center,
            "subwindows": subwindows,
            "slopes": slopes,
            "avg_slope": avg_slope,
            "std_slope": std_slope,
            "intercepts": intercepts,
            "avg_intercept": avg_intercept,
            "i0s": i0s,
            "avg_i0": avg_i0,
            "std_i0": std_i0,
            "rel_err_b": rel_err_b,
            "rel_err_i0": rel_err_i0,
            "candidate_metric": metric,
            "avg_r2": float(np.mean(r2s)),
            "counts": counts,
            "n_points": int(np.sum(mask_range)),
        })

    if not candidate_results:
        return None, "No valid fits in any candidate window."

    best = min(candidate_results, key=lambda x: x["candidate_metric"])
    return {
        "b_kin": best["avg_slope"],
        "intercept": best["avg_intercept"],
        "i0": 10 ** (-best["avg_intercept"] / best["avg_slope"]),
        "r_squared": best["avg_r2"],
        "n_points": best["n_points"],
        "candidate_details": best,
    }, None


def calculate_RCL_utilization(i, R_CL, b_kin):
    """Return the catalyst utilization ``U_J`` used by the RCL model."""
    term = (i * np.log(10) * R_CL) / (2 * b_kin)
    term = np.clip(term, 0, None)
    return (1 + term ** 1.1982) ** (-1 / 1.1982)


def calculate_RCL_overpotential(i, R_CL, b_kin):
    """Catalyst-layer overpotential from Tafel slope and R_CL via utilization U_J."""
    utilization = calculate_RCL_utilization(i, R_CL, b_kin)
    return -b_kin * np.log10(utilization)


def iterative_tafel_fit(V, i, E_rev, HFR, R_CL, i_lower=0.005, i_upper=0.1, tol=1e-3, max_iter=10):
    """Iteratively fit Tafel slope while subtracting ohmic and R_CL drops.
    Returns: ((fit_result, eta_ohm_all, eta_rcl_all, acc_all), error_or_warning_msg)."""
    b_prev = None
    fit_result = None

    for _ in range(max_iter):
        eta_ohm = i * HFR
        eta_rcl = np.zeros_like(i) if b_prev is None else calculate_RCL_overpotential(i, R_CL, b_prev)
        eta_corr = (V - E_rev) - eta_ohm - eta_rcl

        fit_result, err = fit_tafel_window(eta_corr, i, i_lower, i_upper)
        if fit_result is None:
            return None, err

        b_new = fit_result["b_kin"]
        i0 = fit_result["i0"]

        eta_kin_all = b_new * np.log10(i / i0)
        eta_ohm_all = i * HFR
        eta_rcl_all = calculate_RCL_overpotential(i, R_CL, b_new)
        acc_all = eta_kin_all + eta_ohm_all + eta_rcl_all

        idxs = np.where(i >= i_upper)[0]
        condition_ok = True
        if len(idxs) > 0:
            idx = idxs[0]
            allowed = V[idx] - E_rev
            condition_ok = (acc_all[idx] <= allowed)

        if (b_prev is not None) and (abs(b_new - b_prev) < tol) and condition_ok:
            return (fit_result, eta_ohm_all, eta_rcl_all, acc_all), None

        b_prev = b_new

    return (fit_result, eta_ohm_all, eta_rcl_all,
            acc_all), "Warning: Tafel fitting reached max iterations without fully converging."


def get_overpotential_at_current(target_i, i_array, V_array, E_rev, HFR, R_CL, fit_result):
    """Calculate/interpolate the breakdown values at a specific target current density."""
    target_i = float(target_i)
    i_array = np.asarray(i_array, dtype=float)
    V_array = np.asarray(V_array, dtype=float)
    finite = np.isfinite(i_array) & np.isfinite(V_array)
    i_array = i_array[finite]
    V_array = V_array[finite]
    if not np.isfinite(target_i) or target_i <= 0:
        raise ValueError("Breakdown target current must be a finite value greater than zero.")
    if len(i_array) == 0:
        raise ValueError("Cannot compute a breakdown without finite current/potential data.")
    i_min = float(np.min(i_array))
    i_max = float(np.max(i_array))
    if target_i < i_min or target_i > i_max:
        raise ValueError(
            f"Breakdown target {target_i:g} A/cm² is outside the measured "
            f"range {i_min:g}–{i_max:g} A/cm²."
        )

    b_kin = fit_result["b_kin"]
    i0 = fit_result["i0"]

    eta_kin = b_kin * np.log10(target_i / i0)
    eta_ohm = target_i * HFR
    eta_rcl = calculate_RCL_overpotential(target_i, R_CL, b_kin)

    sort_idx = np.argsort(i_array)
    i_sorted = i_array[sort_idx]
    V_sorted = V_array[sort_idx]
    V_exp_target = np.interp(target_i, i_sorted, V_sorted)

    eta_res = (V_exp_target - E_rev) - (eta_kin + eta_ohm + eta_rcl)
    eta_res = max(eta_res, 0)

    return {
        "E_rev": E_rev, "eta_kin": eta_kin, "eta_ohm": eta_ohm,
        "eta_rcl": eta_rcl, "eta_res": eta_res, "V_exp": V_exp_target,
    }
