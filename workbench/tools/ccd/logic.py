"""Current-density onsets and resolved plateaus from acquired LSV sweeps.

The CCD is an operational excess-voltage criterion, not a mechanistic
identification of mass transport. All current densities supplied here must
already use A/cm². Voltage order is preserved; current is never sorted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import savgol_filter


@dataclass(frozen=True)
class CcdSettings:
    baseline_mode: str = "Manual"
    baseline_lower: float = 1.9
    baseline_upper: float = 2.45
    threshold_v: float = 0.02
    reference_hfr: float = 0.12
    bin_width_v: float = 0.01
    smoothing_window: int = 15
    persistence_v: float = 0.10
    plateau_window_v: float = 0.20
    plateau_slope_fraction: float = 0.15
    plateau_spread_fraction: float = 0.05


@dataclass
class CcdResult:
    potential: np.ndarray
    current: np.ndarray
    potential_sign: float
    binned_potential: np.ndarray
    binned_current: np.ndarray
    smoothed_current: np.ndarray
    baseline_current: np.ndarray
    excess_voltage: np.ndarray
    corrected_potential: np.ndarray
    binned_corrected_potential: np.ndarray
    corrected_baseline_potential: np.ndarray
    corrected_baseline_slope: float
    corrected_baseline_intercept: float
    baseline_lower: float
    baseline_upper: float
    baseline_slope: float
    baseline_intercept: float
    baseline_r_squared: float
    ccd_index: int | None = None
    ccd_current: float | None = None
    ccd_voltage: float | None = None
    plateau_start: int | None = None
    plateau_end: int | None = None
    limiting_current: float | None = None
    ccd_status: str = "Not reached within measured range"
    limiting_status: str = "Plateau not resolved"
    warnings: list[str] = field(default_factory=list)


def _validate_settings(settings: CcdSettings) -> None:
    if settings.baseline_mode.lower() not in {"manual", "auto"}:
        raise ValueError("Baseline mode must be Manual or Auto.")
    positive = {
        "CCD threshold": settings.threshold_v,
        "Voltage bin width": settings.bin_width_v,
        "CCD persistence span": settings.persistence_v,
        "Plateau window": settings.plateau_window_v,
        "Plateau slope fraction": settings.plateau_slope_fraction,
        "Plateau spread fraction": settings.plateau_spread_fraction,
    }
    for name, value in positive.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and greater than zero.")
    if not np.isfinite(settings.reference_hfr) or settings.reference_hfr < 0:
        raise ValueError("Reference HFR must be finite and nonnegative.")
    if not (np.isfinite(settings.baseline_lower)
            and np.isfinite(settings.baseline_upper)):
        raise ValueError("Baseline limits must be finite.")
    if settings.baseline_lower >= settings.baseline_upper:
        raise ValueError("Baseline lower limit must be smaller than its upper limit.")
    window = settings.smoothing_window
    if (not isinstance(window, (int, np.integer)) or isinstance(window, bool)
            or window < 5 or window % 2 != 1):
        raise ValueError("Smoothing window must be an odd integer of at least five bins.")
    if settings.plateau_slope_fraction >= 1:
        raise ValueError("Plateau slope fraction must be smaller than one.")
    if settings.plateau_spread_fraction >= 1:
        raise ValueError("Plateau spread fraction must be smaller than one.")


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    centered = x - np.mean(x)
    denominator = float(centered @ centered)
    if len(x) < 5 or denominator <= 0:
        raise ValueError("The baseline needs at least five points spanning a voltage interval.")
    slope = float(centered @ (y - np.mean(y)) / denominator)
    intercept = float(np.mean(y) - slope * np.mean(x))
    residual = y - (intercept + slope * x)
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1 - float(residual @ residual) / total if total > 0 else 0.0
    return slope, intercept, r_squared


def _contiguous_runs(mask: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return []
    return list(np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1))


def _forward_sweep(potential, current, bin_width: float):
    try:
        voltage = np.asarray(potential, dtype=float)
        density = np.asarray(current, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("LSV potential and current density must be numeric arrays.") from exc
    if voltage.ndim != 1 or density.ndim != 1 or voltage.shape != density.shape:
        raise ValueError("LSV potential and current density must be matching one-dimensional arrays.")
    finite = np.isfinite(voltage) & np.isfinite(density)
    runs = _contiguous_runs(finite)
    if not runs:
        raise ValueError("The LSV does not contain finite potential/current-density points.")
    warnings = []
    if not np.all(finite):
        warnings.append("Nonfinite rows separate sweeps; used the longest contiguous finite segment.")
    chosen = max(runs, key=lambda run: (float(np.ptp(voltage[run])), len(run)))
    voltage, density = voltage[chosen], density[chosen]
    if len(voltage) < 10:
        raise ValueError("At least ten finite LSV points are required.")
    differences = np.diff(voltage)
    meaningful = differences[np.abs(differences) > 1e-10]
    if not len(meaningful):
        raise ValueError("The LSV must span a nonzero potential interval.")
    # The initial sweep determines direction, including HER toward negative voltage.
    direction = 1.0 if float(np.median(meaningful[:100])) > 0 else -1.0
    oriented = direction * voltage
    reversed_rows = np.flatnonzero(np.maximum.accumulate(oriented) - oriented > bin_width)
    if len(reversed_rows):
        turn = int(np.argmax(oriented[:reversed_rows[0] + 1]))
        oriented, density = oriented[:turn + 1], density[:turn + 1]
        warnings.append("Used the initial voltage sweep before its reversal.")
    if len(oriented) < 10 or np.ptp(oriented) < 4 * bin_width:
        raise ValueError("The initial LSV sweep is too short for CCD analysis.")
    return oriented, np.abs(density), direction, warnings


def _median_bins(voltage: np.ndarray, current: np.ndarray, width: float):
    origin = np.floor(float(np.min(voltage)) / width) * width
    codes = np.floor((voltage - origin) / width + 1e-9).astype(int)
    last = int(np.max(codes))
    centers = origin + (np.arange(last + 1) + .5) * width
    median = np.full(last + 1, np.nan)
    for index in np.unique(codes):
        median[index] = float(np.median(current[codes == index]))
    # A sparsely sampled voltage gap is not filled or smoothed across.
    runs = _contiguous_runs(np.isfinite(median))
    longest = max(runs, key=len)
    gap = len(longest) != len(median)
    if len(longest) < 5:
        raise ValueError("The LSV has too few consecutive occupied voltage bins; increase bin width.")
    return centers[longest], median[longest], gap


def _automatic_baseline(voltage: np.ndarray, current: np.ndarray, width: float):
    window = max(12, int(np.ceil(.35 / width)))
    if len(voltage) < window + 10:
        raise ValueError("Not enough data for an automatic baseline; select a manual range.")
    high = float(np.quantile(current, .95))
    candidates = []
    for start in range(len(voltage) - window + 1):
        x = voltage[start:start + window]
        y = current[start:start + window]
        if not .15 * high <= np.median(y) <= .65 * high:
            continue
        slope, _intercept, r_squared = _linear_fit(x, y)
        split = len(x) // 2
        left = _linear_fit(x[:split], y[:split])[0]
        right = _linear_fit(x[split:], y[split:])[0]
        if slope > 0 and r_squared >= .995 and abs(left - right) <= .20 * slope:
            candidates.append((slope, float(x[0] - width / 2), float(x[-1] + width / 2)))
    if not candidates:
        raise ValueError("No stable middle region found; select a manual baseline range.")
    _slope, lower, upper = max(candidates)
    return lower, upper


def _find_plateau(result: CcdResult, settings: CcdSettings) -> None:
    if result.ccd_index is None:
        return
    voltage = result.binned_potential
    current = result.smoothed_current
    count = max(6, int(np.ceil(settings.plateau_window_v / settings.bin_width_v)) + 1)
    high = float(np.quantile(current[result.ccd_index:], .95))
    candidates = []
    for start in range(result.ccd_index, len(voltage) - count + 1):
        end = start + count
        x, y = voltage[start:end], current[start:end]
        level = float(np.median(y))
        if level <= 0 or level < .95 * high:
            continue
        slope = _linear_fit(x, y)[0]
        spread = float(np.quantile(y, .9) - np.quantile(y, .1))
        if (abs(slope) <= settings.plateau_slope_fraction * result.baseline_slope
                and spread <= settings.plateau_spread_fraction * level):
            candidates.append((start, end - 1, level))
    if not candidates:
        return
    # Select the highest sustained level, then include its contiguous stable run.
    peak = max(candidates, key=lambda item: item[2])
    by_start = {start: (end, level) for start, end, level in candidates}
    first = last = peak[0]
    while first - 1 in by_start:
        first -= 1
    while last + 1 in by_start:
        last += 1
    end = by_start[last][0]
    result.plateau_start = first
    result.plateau_end = end
    # Use median binned measurements, not the highest observation or SG overshoot.
    result.limiting_current = float(np.median(result.binned_current[first:end + 1]))
    result.limiting_status = "Resolved plateau"
    tail_count = max(3, int(np.ceil(.10 / settings.bin_width_v)))
    tail = float(np.median(current[-tail_count:]))
    if tail < .95 * max(result.limiting_current, high):
        result.limiting_status = "Resolved plateau; subsequent current decline"
        result.warnings.append(
            "Current declines after the plateau; limiting current uses the resolved plateau, not the final current.")


def analyze_curve(potential, current, settings: CcdSettings | None = None) -> CcdResult:
    """Analyze one acquired curve and return non-extrapolated current estimates.

    Negative-potential sweeps are internally reversed in sign; manual baseline
    limits therefore refer to the resulting increasing-drive coordinate.
    ``potential_sign`` restores the original potential axis. Current is a
    magnitude in every returned array and scalar. Reference HFR uses Ω·cm².
    The same baseline points are fit in corrected-voltage/current space, so
    subtracting constant HFR preserves excess voltage and CCD current.
    """
    settings = settings or CcdSettings()
    _validate_settings(settings)
    voltage, density, direction, warnings = _forward_sweep(
        potential, current, settings.bin_width_v)
    centers, median, gaps = _median_bins(voltage, density, settings.bin_width_v)
    if gaps:
        warnings.append("Empty voltage bins separate segments; used the longest continuously sampled interval.")
    window = min(settings.smoothing_window, len(median) if len(median) % 2 else len(median) - 1)
    smooth = savgol_filter(median, window, 3)
    lower, upper = settings.baseline_lower, settings.baseline_upper
    if settings.baseline_mode.lower() == "auto":
        lower, upper = _automatic_baseline(centers, smooth, settings.bin_width_v)
    if lower < np.min(voltage) - settings.bin_width_v or upper > np.max(voltage) + settings.bin_width_v:
        raise ValueError("The baseline range must lie inside the measured initial sweep.")
    half_bin = settings.bin_width_v / 2
    tolerance = settings.bin_width_v * 1e-8
    if (lower < centers[0] - half_bin - tolerance
            or upper > centers[-1] + half_bin + tolerance):
        raise ValueError(
            "The baseline range must lie inside the retained continuously sampled voltage interval; "
            "increase bin width or select a baseline within that interval.")
    select = (voltage >= lower) & (voltage <= upper)
    # The same measured-voltage rows define the baseline at every HFR value.
    # Fit corrected voltage on current so subtracting a constant jR changes
    # the baseline slope, but cannot change the excess above that baseline.
    reference_hfr = settings.reference_hfr
    corrected = voltage - reference_hfr * density
    corrected_slope, corrected_intercept, _ = _linear_fit(density[select], corrected[select])
    _raw_slope, _raw_intercept, r_squared = _linear_fit(density[select], voltage[select])
    total_resistance = corrected_slope + reference_hfr
    if total_resistance <= 0:
        raise ValueError("The selected middle region must have increasing current with voltage.")
    slope = 1 / total_resistance
    intercept = -corrected_intercept / total_resistance
    baseline = (centers - corrected_intercept) / total_resistance
    binned_corrected = centers - reference_hfr * smooth
    corrected_baseline = corrected_intercept + corrected_slope * smooth
    extra = binned_corrected - corrected_baseline
    if corrected_slope < -1e-12:
        warnings.append(
            "Reference HFR exceeds the fitted middle-region slope; review the HFR and baseline range.")
    if r_squared < .995:
        warnings.append("The selected baseline is weakly linear; review its range.")
    baseline_error_v = np.std(
        corrected[select] - (corrected_intercept + corrected_slope * density[select]))
    if baseline_error_v > settings.threshold_v:
        warnings.append("Baseline scatter exceeds the CCD voltage threshold; review the fitted range.")
    result = CcdResult(
        potential=voltage, current=density, potential_sign=direction,
        binned_potential=centers, binned_current=median, smoothed_current=smooth,
        baseline_current=baseline, excess_voltage=extra,
        corrected_potential=corrected, binned_corrected_potential=binned_corrected,
        corrected_baseline_potential=corrected_baseline,
        corrected_baseline_slope=corrected_slope,
        corrected_baseline_intercept=corrected_intercept,
        baseline_lower=lower, baseline_upper=upper, baseline_slope=slope,
        baseline_intercept=intercept, baseline_r_squared=r_squared,
        warnings=warnings,
    )
    count = max(1, int(np.ceil(settings.persistence_v / settings.bin_width_v - 1e-10)))
    for index in range(len(centers) - count + 1):
        if centers[index] < upper or smooth[index] <= 0:
            continue
        if np.all(extra[index:index + count] >= settings.threshold_v):
            result.ccd_index = index
            result.ccd_current = float(smooth[index])
            result.ccd_voltage = float(centers[index])
            result.ccd_status = "Detected"
            break
    _find_plateau(result, settings)
    return result
