"""Area-specific HFR evaluated at current-density magnitudes."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PPoly, make_smoothing_spline


class HfrProfile:
    """Natural cubic smoothing spline, continued by endpoint tangents in |j|."""

    def __init__(self, points, smoothing=1e-4):
        try:
            values = np.asarray(list(points), dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("HFR points must contain numeric current-density/HFR pairs.") from exc
        if values.size == 0:
            raise ValueError("Add at least five HFR measurements for cubic smoothing spline fitting.")
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("Each HFR point must contain a current density and an HFR value.")
        if not np.all(np.isfinite(values)):
            raise ValueError("HFR points must contain finite current densities and HFR values.")
        if np.any(values < 0):
            raise ValueError("HFR point current densities and HFR values must be nonnegative.")
        values = values[np.argsort(values[:, 0])]
        if np.any(np.diff(values[:, 0]) == 0):
            raise ValueError("HFR point current densities must be unique.")
        if len(values) < 5:
            raise ValueError("Cubic smoothing spline fitting needs at least five distinct HFR current densities.")
        try:
            strength = float(smoothing)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("HFR smoothing strength must be finite and nonnegative.") from None
        if isinstance(smoothing, bool) or not np.isfinite(strength) or strength < 0:
            raise ValueError("HFR smoothing strength must be finite and nonnegative.")
        self.points = tuple((float(current), float(hfr)) for current, hfr in values)
        self.smoothing = strength
        self.current_range = (float(values[0, 0]), float(values[-1, 0]))
        self._span = self.current_range[1] - self.current_range[0]
        normalized = (values[:, 0] - self.current_range[0]) / self._span
        self._spline = make_smoothing_spline(normalized, values[:, 1], lam=strength)
        self.endpoint_values = tuple(float(value) for value in self._spline([0.0, 1.0]))
        self.endpoint_slopes = tuple(float(value) / self._span
                                     for value in self._spline.derivative()([0.0, 1.0]))
        stationary = PPoly.from_spline(self._spline).derivative().roots(extrapolate=False)
        stationary = stationary[np.isfinite(stationary) & (stationary > 0) & (stationary < 1)]
        extrema = self._spline(np.r_[0.0, stationary, 1.0])
        if not np.all(np.isfinite(extrema)) or np.any(extrema < 0):
            raise ValueError("The fitted HFR spline must remain finite and nonnegative; "
                             "check the measurements or adjust smoothing strength.")
        self.fit_rmse = float(np.sqrt(np.mean((self._spline(normalized) - values[:, 1]) ** 2)))

    def metadata(self) -> dict:
        """Scientific settings shared by diagnostics and portable exports."""
        return {
            "HFR fit method": "Cubic smoothing spline",
            "HFR smoothing strength": self.smoothing,
            "HFR current normalization": "Measured current interval scaled to [0, 1]",
            "HFR measured lower (A/cm²)": self.current_range[0],
            "HFR measured upper (A/cm²)": self.current_range[1],
            "HFR extension": "Endpoint tangent; negative or nonfinite predictions rejected",
            "HFR lower endpoint slope (Ω·cm² per A/cm²)": self.endpoint_slopes[0],
            "HFR upper endpoint slope (Ω·cm² per A/cm²)": self.endpoint_slopes[1],
            "HFR fit RMS (Ω·cm²)": self.fit_rmse,
        }

    def __call__(self, current):
        current = np.asarray(current, dtype=float)
        if not np.all(np.isfinite(current)):
            raise ValueError("Current density must be finite when evaluating HFR.")
        magnitude = np.abs(current)
        lower, upper = self.current_range
        normalized = (np.clip(magnitude, lower, upper) - lower) / self._span
        result = self._spline(normalized)
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.where(magnitude < lower,
                              self.endpoint_values[0] + self.endpoint_slopes[0] * (magnitude - lower), result)
            result = np.where(magnitude > upper,
                              self.endpoint_values[1] + self.endpoint_slopes[1] * (magnitude - upper), result)
        if not np.all(np.isfinite(result)) or np.any(result < 0):
            raise ValueError("HFR endpoint-tangent extension predicts a negative or nonfinite resistance. "
                             "Upload HFR measurements covering the LSV current range or adjust smoothing strength.")
        return float(result) if current.ndim == 0 else result


def evaluate_hfr(hfr, current):
    """Evaluate a scalar or callable HFR, preserving the current input's shape."""
    current = np.asarray(current, dtype=float)
    if not np.all(np.isfinite(current)):
        raise ValueError("Current density must be finite when evaluating HFR.")
    raw_values = hfr(np.abs(current)) if callable(hfr) else hfr
    try:
        values = np.asarray(raw_values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("HFR must contain numeric values.") from exc
    if not callable(hfr) and values.ndim != 0:
        raise ValueError("HFR must be a scalar value or a current-dependent profile.")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("HFR must be finite and nonnegative.")
    try:
        values = np.broadcast_to(values, current.shape)
    except ValueError as exc:
        raise ValueError("HFR values must match the current-density shape.") from exc
    return float(values) if current.ndim == 0 else np.array(values, copy=True)
