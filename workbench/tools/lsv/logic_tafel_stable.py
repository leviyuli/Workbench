"""Contiguous stable-slope Tafel fitting for independent batch scans.

Acquisition runs define barriers. An internal voltage index supports interval
statistics without changing the native point order or bridging those barriers.
The numerical criteria match the exploratory contiguous-window method.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from workbench.tools.lsv.hfr import evaluate_hfr
from workbench.tools.lsv.logic_tafel import calculate_RCL_overpotential


@dataclass(frozen=True)
class StableTafelSettings:
    method: str = "contiguous-stable-tafel-v1"
    window_width: float = .060
    step: float = .005
    minimum_decades: float = .5
    slope_fraction: float = .20
    rmse_v: float = .005
    minimum_windows: int = 3


class TafelFitError(ValueError):
    """A scan has no accepted kinetic fit; diagnostics remain exportable."""

    def __init__(self, message: str, diagnostics: dict):
        super().__init__(message)
        self.diagnostics = diagnostics
        self.method_version = diagnostics.get("settings", {}).get(
            "method", StableTafelSettings().method)


class _Extrema:
    """Static range minima and maxima in constant time after sparse indexing."""

    def __init__(self, values):
        values = np.asarray(values, dtype=float)
        self.minimum, self.maximum = [values], [values]
        width = 1
        while 2 * width <= len(values):
            self.minimum.append(np.minimum(self.minimum[-1][:-width], self.minimum[-1][width:]))
            self.maximum.append(np.maximum(self.maximum[-1][:-width], self.maximum[-1][width:]))
            width *= 2

    def bounds(self, start, end):
        power = (int(end - start)).bit_length() - 1
        width = 1 << power
        return (float(min(self.minimum[power][start], self.minimum[power][end - width])),
                float(max(self.maximum[power][start], self.maximum[power][end - width])))


class _VoltageIndex:
    def __init__(self, voltage, current, native):
        self.native = native[np.argsort(voltage[native], kind="stable")]
        self.y = voltage[self.native]
        self.current = current[self.native]
        self.x = np.log10(self.current)
        self.origin_x, self.origin_y = float(self.x[0]), float(self.y[0])
        x, y = self.x - self.origin_x, self.y - self.origin_y
        self.sums = np.vstack([np.r_[0., np.cumsum(value)]
                               for value in (x, y, x * x, y * y, x * y)])
        self.extrema = _Extrema(self.current)

    def bounds(self, lower, upper, epsilon):
        return (int(np.searchsorted(self.y, lower - epsilon, side="left")),
                int(np.searchsorted(self.y, upper + epsilon, side="right")))

    def fit(self, start, end):
        count = end - start
        if count < 3:
            return None
        sx, sy, sxx, syy, sxy = self.sums[:, end] - self.sums[:, start]
        variance = sxx - sx * sx / count
        if variance <= np.finfo(float).eps * max(abs(sxx), 1e-24):
            return None
        covariance = sxy - sx * sy / count
        slope = float(covariance / variance)
        intercept = float(self.origin_y + sy / count - slope * (self.origin_x + sx / count))
        if not np.isfinite(slope) or slope <= 0 or not np.isfinite(intercept):
            return None
        total = max(float(syy - sy * sy / count), 0.)
        error = max(float(total - covariance * covariance / variance), 0.)
        return {"slope": slope, "intercept": intercept,
                "r_squared": 1 - error / total if total > 1e-24 else 0.,
                "rmse": float(np.sqrt(error / count)), "n_points": count}


def _native_runs(mask, source):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    gaps = (np.diff(indices) != 1) | (np.abs(np.diff(source[indices])) != 1)
    return np.split(indices, np.flatnonzero(gaps) + 1)


def _direct_fit(current, voltage):
    if len(current) < 3:
        return None
    x, y = np.log10(current), np.asarray(voltage, dtype=float)
    dx, dy = x - np.mean(x), y - np.mean(y)
    variance = float(dx @ dx)
    if np.ptp(x) <= 1e-12 or variance <= 0:
        return None
    slope = float((dx @ dy) / variance)
    intercept = float(np.mean(y) - slope * np.mean(x))
    if slope <= 0 or not np.isfinite(slope) or not np.isfinite(intercept):
        return None
    errors = y - (intercept + slope * x)
    error, total = float(errors @ errors), float(dy @ dy)
    return {"slope": slope, "intercept": intercept,
            "rmse": float(np.sqrt(error / len(x))),
            "r_squared": 1 - error / total if total > 0 else 0., "n_points": len(x)}


def fit_stable_tafel(eta_corr, current, i_lower=.02, i_upper=.8, *,
                     minimum_decades=.5, slope_fraction=.20, rmse_v=.005,
                     window_width=.060, step=.005, minimum_windows=3,
                     source_indices=None):
    """Return ``(fit_or_None, compact_diagnostics)`` on the native raw points.

    Local centered 20/40/60 mV regressions qualify neighboring 60 mV windows.
    Merged regions must pass half-slope and constituent-slope checks; the widest
    log-current span wins, with earliest corrected drive breaking ties. No
    candidate is ranked by maximum slope, exchange current or R².
    """
    settings = StableTafelSettings(window_width=window_width, step=step,
        minimum_decades=minimum_decades, slope_fraction=slope_fraction,
        rmse_v=rmse_v, minimum_windows=minimum_windows)
    for name in ("window_width", "step", "minimum_decades", "slope_fraction", "rmse_v"):
        value = getattr(settings, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if minimum_windows < 3 or int(minimum_windows) != minimum_windows:
        raise ValueError("minimum_windows must be an integer of at least three.")
    if not np.isfinite(i_lower) or not np.isfinite(i_upper) or i_lower < 0 or i_upper <= i_lower:
        raise ValueError("Current search bounds must be finite, nonnegative and increasing.")
    voltage, density = np.asarray(eta_corr, dtype=float), np.asarray(current, dtype=float)
    if voltage.ndim != 1 or density.shape != voltage.shape:
        raise ValueError("Corrected overpotential and current must be matching 1D arrays.")
    source = np.arange(len(voltage)) if source_indices is None else np.asarray(source_indices)
    if source.shape != voltage.shape:
        raise ValueError("Source rows must match the input arrays.")
    if not np.isfinite(source).all():
        raise ValueError("Source rows must be finite.")
    mask = (np.isfinite(voltage) & np.isfinite(density) & (density > 0)
            & (density >= i_lower) & (density <= i_upper))
    diagnostics = {"settings": asdict(settings), "search_lower": float(i_lower),
        "search_upper": float(i_upper), "range_points": int(mask.sum()),
        "tested_windows": 0, "qualified_windows": 0, "segments": 0,
        "accepted_intervals": 0, "status": "Unresolved", "reason": ""}
    best, best_index = None, None
    for segment, native in enumerate(_native_runs(mask, source)):
        diagnostics["segments"] += 1
        if len(native) < 5:
            continue
        index = _VoltageIndex(voltage, density, native)
        vmin, vmax = float(index.y[0]), float(index.y[-1])
        epsilon = 8 * np.finfo(float).eps * max(abs(vmin), abs(vmax), abs(window_width))
        if vmax - vmin < window_width - epsilon:
            continue
        windows = []
        for start in np.arange(vmin, vmax - window_width + epsilon, step):
            diagnostics["tested_windows"] += 1
            end, center = float(start + window_width), float(start + window_width / 2)
            lo, hi = index.bounds(start, end, epsilon)
            full = None
            if hi - lo >= 5:
                fits = [index.fit(*index.bounds(center - width / 2, center + width / 2, epsilon))
                        for width in (window_width / 3, window_width * 2 / 3, window_width)]
                if all(fit is not None for fit in fits):
                    candidate = fits[-1]
                    if (candidate["rmse"] <= rmse_v
                            and np.ptp([fit["slope"] for fit in fits])
                            <= slope_fraction * candidate["slope"]):
                        full = candidate
                        diagnostics["qualified_windows"] += 1
            windows.append({"start": float(start), "end": end,
                            "slope": full["slope"] if full else np.nan})
        qualified = np.flatnonzero(np.isfinite([window["slope"] for window in windows]))
        if not len(qualified):
            continue
        for run in np.split(qualified, np.flatnonzero(np.diff(qualified) != 1) + 1):
            if len(run) < minimum_windows:
                continue
            slopes = _Extrema([windows[int(number)]["slope"] for number in run])
            for first in range(len(run) - minimum_windows + 1):
                for last in range(first + minimum_windows - 1, len(run)):
                    start, end = windows[int(run[first])]["start"], windows[int(run[last])]["end"]
                    lo, hi = index.bounds(start, end, epsilon)
                    lower, upper = index.extrema.bounds(lo, hi)
                    decades = float(np.log10(upper / lower))
                    if decades < minimum_decades - 1e-12:
                        continue
                    fit = index.fit(lo, hi)
                    if fit is None or fit["rmse"] > rmse_v:
                        continue
                    middle = int(np.searchsorted(index.y, (start + end) / 2, side="right"))
                    left, right = index.fit(lo, middle), index.fit(middle, hi)
                    if left is None or right is None:
                        continue
                    tolerance = slope_fraction * fit["slope"]
                    half_difference = abs(left["slope"] - right["slope"])
                    minimum, maximum = slopes.bounds(first, last + 1)
                    spread = maximum - minimum
                    if half_difference > tolerance or spread > tolerance:
                        continue
                    diagnostics["accepted_intervals"] += 1
                    candidate = {"segment": segment, "start": start, "end": end,
                        "current_lower": lower, "current_upper": upper,
                        "log_current_span": decades, **fit, "window_count": last - first + 1,
                        "half_slope_difference": half_difference,
                        "constituent_slope_spread": spread, "bounds": (lo, hi)}
                    if (best is None or decades > best["log_current_span"] + 1e-12
                            or abs(decades - best["log_current_span"]) <= 1e-12
                            and (start, segment) < (best["start"], best["segment"])):
                        best, best_index = candidate, index
    if best is None:
        diagnostics["reason"] = (
            "No contiguous stable Tafel interval meets the neighboring-window, RMS and "
            f"minimum {minimum_decades:g}-decade requirements within the selected current range."
        )
        return None, diagnostics
    lo, hi = best.pop("bounds")
    selected = np.sort(best_index.native[lo:hi])
    best.update(_direct_fit(density[selected], voltage[selected]))
    best.update(indices=selected.tolist(), source_start=int(source[selected].min()),
                source_end=int(source[selected].max()))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        exchange = float(10. ** (-best["intercept"] / best["slope"]))
    if not np.isfinite(exchange) or exchange <= 0:
        diagnostics["reason"] = "The stable Tafel interval produced an unrepresentable exchange current."
        return None, diagnostics
    diagnostics["status"] = "Resolved"
    diagnostics["selected_interval"] = {key: value for key, value in best.items() if key != "indices"}
    details = {**best, "current_range": (best["current_lower"], best["current_upper"]),
        "window": (best["start"], best["end"]), "method": settings.method,
        "avg_slope": best["slope"], "avg_intercept": best["intercept"],
        "avg_r2": best["r_squared"]}
    return {"b_kin": best["slope"], "intercept": best["intercept"], "i0": exchange,
        "r_squared": best["r_squared"], "n_points": best["n_points"],
        "candidate_details": details, "method": settings.method,
        "diagnostics": diagnostics}, diagnostics


def _bootstrap_slope(voltage, current, source, i_lower, i_upper):
    """Lowest-current positive local slope, used only to initialize RCL."""
    mask = (np.isfinite(voltage) & np.isfinite(current) & (current > 0)
            & (current >= i_lower) & (current <= i_upper))
    best = None
    for native in _native_runs(mask, source):
        if len(native) < 5:
            continue
        index = _VoltageIndex(voltage, current, native)
        epsilon = 8 * np.finfo(float).eps * max(abs(index.y[0]), abs(index.y[-1]), .06)
        for start in np.arange(index.y[0], index.y[-1] - .06 + epsilon, .005):
            lo, hi = index.bounds(start, start + .06, epsilon)
            fit = index.fit(lo, hi)
            if hi - lo < 5 or fit is None:
                continue
            lower, upper = index.extrema.bounds(lo, hi)
            key = (float(np.sqrt(lower * upper)), float(start))
            if best is None or key < best[0]:
                best = key, fit["slope"]
    return None if best is None else best[1]


def _rcl_consistent(voltage, current, fit, previous_corrected):
    """Validate the reported interval after applying its own final RCL slope."""
    settings = StableTafelSettings()
    details = fit["candidate_details"]
    selected = np.asarray(details["indices"], dtype=int)
    j, y = current[selected], voltage[selected]
    full = _direct_fit(j, y)
    if full is None or full["rmse"] > settings.rmse_v:
        return False
    errors = y - fit["intercept"] - fit["b_kin"] * np.log10(j)
    model_rms = float(np.sqrt(np.mean(errors * errors)))
    if model_rms > settings.rmse_v:
        return False
    # Follow the selected physical points as the iterative correction shifts
    # their voltage coordinates; the raw current/source interval is unchanged.
    shift = float(np.median(y - previous_corrected[selected]))
    start, end = np.asarray(details["window"]) + shift
    middle = (start + end) / 2
    left, right = _direct_fit(j[y <= middle], y[y <= middle]), _direct_fit(j[y > middle], y[y > middle])
    if left is None or right is None:
        return False
    tolerance = settings.slope_fraction * full["slope"]
    half_difference = abs(left["slope"] - right["slope"])
    if half_difference > tolerance:
        return False
    local_slopes = []
    epsilon = 8 * np.finfo(float).eps * max(abs(start), abs(end), settings.window_width)
    for offset in range(details["window_count"]):
        center = start + offset * settings.step + settings.window_width / 2
        fits = []
        for width in (settings.window_width / 3, settings.window_width * 2 / 3, settings.window_width):
            keep = (y >= center - width / 2 - epsilon) & (y <= center + width / 2 + epsilon)
            fits.append(_direct_fit(j[keep], y[keep]))
        if any(local is None for local in fits) or fits[-1]["n_points"] < 5:
            return False
        local = fits[-1]
        if (local["rmse"] > settings.rmse_v
                or np.ptp([part["slope"] for part in fits]) > settings.slope_fraction * local["slope"]):
            return False
        local_slopes.append(local["slope"])
    spread = float(np.ptp(local_slopes))
    if spread > tolerance:
        return False
    fit["diagnostics"]["rcl_final_interval_validation"] = {
        "model_rmse": model_rms, "refit_rmse": full["rmse"],
        "half_slope_difference": half_difference, "constituent_slope_spread": spread}
    return True


def iterative_stable_tafel_fit(V, i, E_rev, HFR, R_CL, i_lower=.02, i_upper=.8,
                               tol=1e-3, max_iter=10, include_rcl=True, *, source_indices=None):
    """Fit native batch coordinates with known ohmic and optional iterative RCL drops."""
    if not np.isfinite(tol) or tol <= 0 or max_iter < 1 or int(max_iter) != max_iter:
        raise ValueError("Tafel iteration tolerance must be positive and max_iter a positive integer.")
    voltage, current = np.asarray(V, dtype=float), np.asarray(i, dtype=float)
    valid = np.isfinite(voltage) & np.isfinite(current) & (current > 0)
    ohmic = np.full(current.shape, np.nan)
    ohmic[valid] = current[valid] * evaluate_hfr(HFR, current[valid])
    previous = None
    fit = None
    source = np.arange(len(current)) if source_indices is None else np.asarray(source_indices)
    bootstrap = {}
    for iteration in range(1, (max_iter if include_rcl and R_CL else 1) + 1):
        catalyst = np.zeros_like(current)
        if include_rcl and R_CL and previous is not None:
            catalyst[valid] = calculate_RCL_overpotential(current[valid], R_CL, previous)
        corrected = voltage - E_rev - ohmic - catalyst
        fit, diagnostics = fit_stable_tafel(corrected,
            current, i_lower, i_upper, source_indices=source_indices)
        diagnostics["iterations"] = iteration
        diagnostics.update(bootstrap)
        if fit is None:
            if include_rcl and R_CL and iteration < max_iter:
                seed = _bootstrap_slope(corrected, current, source, i_lower, i_upper)
                if seed is not None:
                    previous = seed
                    bootstrap = {"rcl_seed_used": True, "rcl_seed_slope": seed,
                                 "rcl_seed_count": bootstrap.get("rcl_seed_count", 0) + 1}
                    continue
            raise TafelFitError(diagnostics["reason"], diagnostics)
        kinetic = np.full(current.shape, np.nan)
        kinetic[valid] = fit["b_kin"] * np.log10(current[valid] / fit["i0"])
        catalyst[valid] = (calculate_RCL_overpotential(current[valid], R_CL, fit["b_kin"])
                           if include_rcl else 0.)
        accumulated = kinetic + ohmic + catalyst
        if not include_rcl or not R_CL:
            return (fit, ohmic, catalyst, accumulated), None
        above = np.flatnonzero(valid & (current >= i_upper))
        condition = True
        if len(above):
            closest = above[np.argmin(current[above] - i_upper)]
            condition = accumulated[closest] <= voltage[closest] - E_rev + 1e-12
        consistent = _rcl_consistent(voltage - E_rev - ohmic - catalyst, current, fit, corrected)
        if previous is not None and abs(fit["b_kin"] - previous) < tol and condition and consistent:
            return (fit, ohmic, catalyst, accumulated), None
        previous = fit["b_kin"]
    if not consistent:
        diagnostics["status"] = "Unresolved"
        diagnostics["reason"] = "The final RCL-corrected Tafel interval did not satisfy the stable-slope checks."
        raise TafelFitError(diagnostics["reason"], diagnostics)
    return (fit, ohmic, catalyst, accumulated), "Warning: Tafel fitting reached max iterations without fully converging."
