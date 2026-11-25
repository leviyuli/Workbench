"""CCD from a stable affine baseline in the signed decoupling residual.

Measured-drive order defines adjacency and persistence. Residual voltage and
current are never sorted: neither must be monotonic on an acquired branch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import savgol_filter

if TYPE_CHECKING:
    from workbench.tools.lsv.analysis import DecoupledCurve


@dataclass(frozen=True)
class ResidualCcdSettings:
    method_version: str = "residual-baseline-v1"
    bin_width_v: float = 0.01
    smoothing_window: int = 15
    baseline_window_v: float = 0.35
    baseline_min_windows: int = 3
    baseline_min_current_fraction: float = 0.01
    baseline_rmse_v: float = 0.005
    baseline_slope_fraction: float = 0.20
    threshold_v: float = 0.050
    persistence_v: float = 0.10
    plateau_window_v: float = 0.20
    plateau_slope_fraction: float = 0.15
    plateau_spread_fraction: float = 0.05


@dataclass
class ResidualCcdResult:
    drive: np.ndarray
    potential: np.ndarray
    current: np.ndarray
    median_current: np.ndarray
    residual: np.ndarray
    eta_kin: np.ndarray
    eta_ohm: np.ndarray
    eta_rcl: np.ndarray
    segment_id: np.ndarray
    source_start: np.ndarray
    source_end: np.ndarray
    baseline_mask: np.ndarray
    baseline: np.ndarray
    excess: np.ndarray
    settings: ResidualCcdSettings
    baseline_slope: float | None = None
    baseline_intercept: float | None = None
    baseline_lower_current: float | None = None
    baseline_upper_current: float | None = None
    baseline_r_squared: float | None = None
    baseline_rmse: float | None = None
    baseline_start: int | None = None
    baseline_end: int | None = None
    baseline_segment: int | None = None
    measured_baseline_slope: float | None = None
    ccd_index: int | None = None
    ccd_current: float | None = None
    ccd_residual: float | None = None
    ccd_potential: float | None = None
    plateau_start: int | None = None
    plateau_end: int | None = None
    limiting_current: float | None = None
    baseline_status: str = "Unresolved"
    ccd_status: str = "Baseline unresolved"
    limiting_status: str = "Plateau not resolved"
    reversed_for_analysis: bool = False
    warnings: list[str] = field(default_factory=list)


def _runs(mask: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    return ([] if not len(indices) else
            list(np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)))


@dataclass(frozen=True)
class _Fit:
    slope: float
    intercept: float
    r_squared: float
    rmse: float


class _Regression:
    """Constant-time interval regressions for the merged-window search."""

    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        # Center once to reduce cancellation in prefix differences.
        self.x_origin = float(self.x[0])
        self.y_origin = float(self.y[0])
        x = self.x - self.x_origin
        y = self.y - self.y_origin
        self.sums = np.vstack([np.r_[0.0, np.cumsum(values)]
                               for values in (x, y, x * x, y * y, x * y)])

    def fit(self, start: int, end: int) -> _Fit | None:
        n = end - start
        if n < 5:
            return None
        sx, sy, sxx, syy, sxy = self.sums[:, end] - self.sums[:, start]
        variance = sxx - sx * sx / n
        if variance <= np.finfo(float).eps * max(abs(sxx), 1e-12):
            return None
        covariance = sxy - sx * sy / n
        total = max(float(syy - sy * sy / n), 0.0)
        slope = float(covariance / variance)
        intercept = float(self.y_origin + sy / n
                          - slope * (self.x_origin + sx / n))
        error = max(float(total - covariance * covariance / variance), 0.0)
        rmse = float(np.sqrt(error / n))
        r_squared = 1 - error / total if total > 1e-24 else float("nan")
        return _Fit(slope, intercept, float(r_squared), rmse)


def _qualified(regression: _Regression, start: int, end: int,
               settings: ResidualCcdSettings, minimum_span: float = 0.0):
    current = regression.x[start:end]
    span = float(np.ptp(current))
    if span <= max(minimum_span, 1e-12):
        return None
    fit = regression.fit(start, end)
    split = (start + end) // 2
    left, right = regression.fit(start, split), regression.fit(split, end)
    if fit is None or left is None or right is None:
        return None
    tolerance = max(settings.baseline_slope_fraction * abs(fit.slope),
                    settings.baseline_rmse_v / span)
    if (fit.rmse <= settings.baseline_rmse_v
            and fit.slope >= -settings.baseline_rmse_v / span
            and abs(left.slope - right.slope) <= tolerance):
        return fit, span
    return None


def _validate_settings(settings: ResidualCcdSettings):
    for key in ("bin_width_v", "baseline_window_v", "baseline_rmse_v", "threshold_v",
                "persistence_v", "plateau_window_v", "baseline_min_current_fraction",
                "baseline_slope_fraction", "plateau_slope_fraction", "plateau_spread_fraction"):
        value = getattr(settings, key)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Residual CCD {key} must be finite and positive.")
    if settings.smoothing_window < 5 or settings.smoothing_window % 2 != 1:
        raise ValueError("Residual CCD smoothing window must be odd and at least five.")
    if settings.baseline_min_windows < 3:
        raise ValueError("Residual CCD needs at least three neighboring windows.")


def _processed(curve: DecoupledCurve, settings: ResidualCcdSettings):
    drive = np.asarray(curve.drive, dtype=float)
    current = np.asarray(curve.current_magnitude, dtype=float)
    indices = np.asarray(curve.source_indices, dtype=int)
    if drive.ndim != 1 or drive.shape != current.shape or indices.shape != drive.shape:
        raise ValueError("Residual CCD needs matching one-dimensional drive, current and source rows.")
    finite = np.isfinite(drive) & np.isfinite(current) & (current > 0)
    changes = np.diff(drive[np.isfinite(drive)])
    changes = changes[np.abs(changes) > 1e-10]
    reverse = bool(len(changes) and np.median(changes) < 0)
    if reverse:
        drive, current, indices, finite = (values[::-1]
                                           for values in (drive, current, indices, finite))
    chunks = []
    segment = 0
    for acquired in _runs(finite):
        v, j, rows = drive[acquired], current[acquired], indices[acquired]
        origin = np.floor(float(np.min(v)) / settings.bin_width_v) * settings.bin_width_v
        codes = np.floor((v - origin) / settings.bin_width_v + 1e-9).astype(int)
        occupied = np.unique(codes)
        for positions in np.split(occupied, np.flatnonzero(np.diff(occupied) > 1) + 1):
            centers = origin + (positions + .5) * settings.bin_width_v
            median = np.array([np.median(j[codes == code]) for code in positions])
            first = np.array([np.min(rows[codes == code]) for code in positions], dtype=int)
            last = np.array([np.max(rows[codes == code]) for code in positions], dtype=int)
            window = min(settings.smoothing_window,
                         len(median) if len(median) % 2 else len(median) - 1)
            smooth = savgol_filter(median, window, 3) if window >= 5 else median.copy()
            potential = curve.sign * centers
            components = curve.evaluate_at(potential, smooth)
            chunks.append({"drive": centers, "potential": potential,
                           "current": smooth, "median_current": median,
                           "residual": np.asarray(components["residual"], dtype=float),
                           "eta_kin": np.asarray(components["eta_kin"], dtype=float),
                           "eta_ohm": np.asarray(components["eta_ohm"], dtype=float),
                           "eta_rcl": np.asarray(components["eta_rcl"], dtype=float),
                           "segment_id": np.full(len(positions), segment, dtype=int),
                           "source_start": first, "source_end": last})
            segment += 1
    names = ("drive", "potential", "current", "median_current", "residual", "eta_kin",
             "eta_ohm", "eta_rcl", "segment_id", "source_start", "source_end")
    values = {key: np.concatenate([part[key] for part in chunks]) if chunks else np.array(
        [], dtype=int if key in {"segment_id", "source_start", "source_end"} else float)
              for key in names}
    return values, reverse


def _automatic(result: ResidualCcdResult, tafel_upper: float):
    settings = result.settings
    count = max(12, int(np.ceil(settings.baseline_window_v / settings.bin_width_v)))
    finite = np.isfinite(result.current) & np.isfinite(result.residual)
    if not np.any(finite):
        return None
    minimum_span = settings.baseline_min_current_fraction * float(
        np.quantile(result.current[finite], .95))
    best = None
    for segment in np.unique(result.segment_id):
        segment_rows = np.flatnonzero((result.segment_id == segment) & finite)
        # Nonfinite evaluated components are barriers too.
        for rows in np.split(segment_rows, np.flatnonzero(np.diff(segment_rows) > 1) + 1):
            if len(rows) < count + settings.baseline_min_windows - 1:
                continue
            offset = int(rows[0])
            regression = _Regression(result.current[rows], result.residual[rows])
            candidates = {}
            for start in range(len(rows) - count + 1):
                if np.min(regression.x[start:start + count]) <= tafel_upper:
                    continue
                qualified = _qualified(regression, start, start + count, settings, minimum_span)
                if qualified is not None:
                    candidates[start] = qualified
            if not candidates:
                continue
            starts = np.array(list(candidates), dtype=int)
            candidate_runs = np.split(starts, np.flatnonzero(np.diff(starts) > 1) + 1)
            for run in candidate_runs:
                if len(run) < settings.baseline_min_windows:
                    continue
                local_slopes = np.array([candidates[int(start)][0].slope for start in run])
                local_spans = np.array([candidates[int(start)][1] for start in run])
                found = None
                # Longest first, earliest first; no maximum-slope or R² ranking.
                for size in range(len(run), settings.baseline_min_windows - 1, -1):
                    span_v = (size + count - 2) * settings.bin_width_v
                    if best is not None and span_v < best[0] - 1e-10:
                        break
                    for left in range(len(run) - size + 1):
                        start = int(run[left])
                        end = int(run[left + size - 1]) + count
                        qualified = _qualified(regression, start, end, settings, minimum_span)
                        if qualified is None:
                            continue
                        fit, _span = qualified
                        tolerance = max(settings.baseline_slope_fraction * abs(fit.slope),
                                        settings.baseline_rmse_v / float(
                                            np.median(local_spans[left:left + size])))
                        if np.ptp(local_slopes[left:left + size]) > tolerance:
                            continue
                        found = (span_v, offset + start, offset + end - 1, fit, int(segment))
                        break
                    if found is not None:
                        break
                if found is not None and (best is None or found[0] > best[0] + 1e-10
                                           or (abs(found[0] - best[0]) <= 1e-10
                                               and found[1] < best[1])):
                    best = found
    return None if best is None else best[1:]


def _manual(result: ResidualCcdResult, lower: float, upper: float):
    if not np.isfinite(lower) or not np.isfinite(upper) or lower < 0 or lower >= upper:
        raise ValueError("Residual-baseline current limits must be finite, nonnegative and increasing.")
    choices = []
    for segment in np.unique(result.segment_id):
        in_segment = result.segment_id == segment
        finite = in_segment & np.isfinite(result.current) & np.isfinite(result.residual)
        if not np.any(finite):
            continue
        values = result.current[finite]
        if lower < np.min(values) - 1e-10 or upper > np.max(values) + 1e-10:
            continue
        select = finite & (result.current >= lower) & (result.current <= upper)
        for rows in _runs(select):
            if len(rows) < 5:
                continue
            fit = _Regression(result.current[rows], result.residual[rows]).fit(0, len(rows))
            if fit is not None:
                choices.append((float(result.drive[rows[-1]] - result.drive[rows[0]]),
                                int(rows[0]), int(rows[-1]), fit, int(segment)))
    if not choices:
        return None
    best = max(choices, key=lambda choice: (choice[0], -choice[1]))
    return best[1:]


def _plateau(result: ResidualCcdResult):
    settings = result.settings
    if result.ccd_index is None or result.measured_baseline_slope is None:
        return
    if result.measured_baseline_slope <= 0:
        result.warnings.append("The baseline current does not increase with measured drive; plateau slope is unresolved.")
        return
    segment_rows = np.flatnonzero(result.segment_id == result.baseline_segment)
    segment_end = int(segment_rows[-1]) + 1
    count = max(6, int(np.ceil(settings.plateau_window_v / settings.bin_width_v)) + 1)
    tail = result.current[result.ccd_index:segment_end]
    tail = tail[np.isfinite(tail)]
    if not len(tail):
        return
    high = float(np.quantile(tail, .95))
    candidates = []
    for start in range(result.ccd_index, segment_end - count + 1):
        end = start + count
        current = result.current[start:end]
        if not np.all(np.isfinite(current)):
            continue
        level = float(np.median(current))
        fit = _Regression(result.drive[start:end], current).fit(0, count)
        if fit is None or level <= 0 or level < .95 * high:
            continue
        spread = float(np.quantile(current, .9) - np.quantile(current, .1))
        if (abs(fit.slope) <= settings.plateau_slope_fraction * result.measured_baseline_slope
                and spread <= settings.plateau_spread_fraction * level):
            candidates.append((start, end - 1, level))
    if not candidates:
        return
    peak = max(candidates, key=lambda candidate: candidate[2])
    by_start = {start: end for start, end, _level in candidates}
    first = last = peak[0]
    while first - 1 in by_start:
        first -= 1
    while last + 1 in by_start:
        last += 1
    result.plateau_start, result.plateau_end = first, by_start[last]
    result.limiting_current = float(np.median(
        result.median_current[first:result.plateau_end + 1]))
    result.limiting_status = "Resolved plateau"
    tail_count = max(3, int(np.ceil(.10 / settings.bin_width_v)))
    if np.median(result.current[max(segment_end - tail_count, first):segment_end]) < .95 * max(
            result.limiting_current, high):
        result.limiting_status = "Resolved plateau; subsequent current decline"
        result.warnings.append("Current declines after the plateau; limiting current uses the observed plateau.")


def analyze_residual_ccd(curve: DecoupledCurve, mode: str = "Auto",
                         current_lower: float = 1.0, current_upper: float = 3.0,
                         *, settings: ResidualCcdSettings | None = None) -> ResidualCcdResult:
    """Analyze one fitted branch without changing its kinetic decomposition.

    An unresolved baseline is a result, not an exception, so batch callers can
    retain the decomposition and report the reason. Invalid settings still raise.
    """
    settings = settings or ResidualCcdSettings()
    _validate_settings(settings)
    mode = str(mode).strip().lower()
    if mode not in {"auto", "manual"}:
        raise ValueError("Residual-baseline mode must be Auto or Manual.")
    processed, reverse = _processed(curve, settings)
    size = len(processed["drive"])
    result = ResidualCcdResult(**processed, settings=settings,
                               baseline_mask=np.zeros(size, dtype=bool),
                               baseline=np.full(size, np.nan), excess=np.full(size, np.nan),
                               reversed_for_analysis=reverse)
    if reverse:
        result.warnings.append("CCD is evaluated toward increasing drive; the selected return branch is preserved.")
    if len(np.unique(result.segment_id)) > 1:
        result.warnings.append("Acquisition or empty-bin gaps separate analysis segments; CCD cannot cross them.")
    selected = (_automatic(result, float(curve.settings.i_upper)) if mode == "auto" else
                _manual(result, current_lower, current_upper))
    if selected is None:
        reason = ("No stable residual-baseline interval found; enter a manual current range."
                  if mode == "auto" else
                  "The manual current range needs at least five points inside one continuously sampled segment.")
        result.warnings.append(reason)
        return result
    start, end, fit, segment = selected
    result.baseline_start, result.baseline_end, result.baseline_segment = start, end, segment
    result.baseline_mask[start:end + 1] = True
    in_segment = result.segment_id == segment
    result.baseline[in_segment] = fit.intercept + fit.slope * result.current[in_segment]
    result.excess[in_segment] = result.residual[in_segment] - result.baseline[in_segment]
    result.baseline_slope, result.baseline_intercept = fit.slope, fit.intercept
    result.baseline_lower_current = float(np.min(result.current[start:end + 1]))
    result.baseline_upper_current = float(np.max(result.current[start:end + 1]))
    result.baseline_r_squared, result.baseline_rmse = fit.r_squared, fit.rmse
    measured = _Regression(result.drive[start:end + 1], result.current[start:end + 1]).fit(0, end - start + 1)
    result.measured_baseline_slope = measured.slope if measured is not None else None
    result.baseline_status = "Resolved"
    result.ccd_status = "Not reached within measured range"
    if mode == "manual" and fit.rmse > settings.baseline_rmse_v:
        result.warnings.append("Manual residual-baseline scatter exceeds 5 mV; inspect the selected range.")
    count = max(1, int(np.ceil(settings.persistence_v / settings.bin_width_v - 1e-10)))
    segment_end = int(np.flatnonzero(in_segment)[-1]) + 1
    for index in range(end + 1, segment_end - count + 1):
        if np.all(result.excess[index:index + count] >= settings.threshold_v - 1e-12):
            result.ccd_index = index
            result.ccd_current = float(result.current[index])
            result.ccd_residual = float(result.residual[index])
            result.ccd_potential = float(result.potential[index])
            result.ccd_status = "Detected"
            break
    _plateau(result)
    return result
