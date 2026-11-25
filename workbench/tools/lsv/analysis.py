"""Shared, acquisition-aligned polarization decomposition for single and batch use."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from workbench.core.potential import PotentialCorrection
from workbench.tools.lsv.hfr import HfrProfile, evaluate_hfr
from workbench.tools.lsv.hfr_data import parse_hfr_pairs
from workbench.tools.lsv.logic_tafel import calculate_RCL_overpotential, iterative_tafel_fit
from workbench.tools.lsv.logic_tafel_stable import iterative_stable_tafel_fit


@dataclass(frozen=True)
class DecouplingSettings:
    correction: PotentialCorrection
    temperature: float
    hfr: float | HfrProfile
    include_rcl: bool
    rcl: float
    i_lower: float
    i_upper: float

    @classmethod
    def from_params(cls, params: dict, extra=None) -> "DecouplingSettings":
        correction = PotentialCorrection.from_params(
            {**params, "target_scale": "RHE"}, default_target="RHE",
        )
        temperature = float(params.get("temperature", 378.0))
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be a finite value greater than zero kelvin.")
        lower = float(params.get("i_lower", .02))
        upper = float(params.get("i_upper", .8))
        if not np.isfinite(lower) or not np.isfinite(upper) or lower < 0 or upper <= lower:
            raise ValueError("Tafel lower limit must be less than the upper limit and both must be finite and nonnegative.")
        mode = params.get("hfr_mode", "Constant HFR")
        if mode == "HFR vs |j|":
            if extra is not None and not isinstance(extra, dict):
                raise ValueError("HFR points must be supplied as a table of |j| and HFR values.")
            inputs = extra or {}
            if "hfr_text" in inputs:
                data = parse_hfr_pairs(inputs["hfr_text"])
                points = data.points()
            else:
                points = inputs.get("hfr_points", [])
            hfr = HfrProfile(points, smoothing=inputs.get("hfr_smoothing", 1e-4))
        elif mode == "Constant HFR":
            hfr = float(params.get("hfr", .01))
            evaluate_hfr(hfr, 0.0)
        else:
            raise ValueError(f"Unknown HFR mode: {mode}.")
        include_rcl = bool(params.get("include_rcl", True))
        rcl = float(params.get("rcl", 0.0)) if include_rcl else 0.0
        if not np.isfinite(rcl) or rcl < 0:
            raise ValueError("R_CL must be finite and nonnegative.")
        return cls(correction, temperature, hfr, include_rcl, rcl, lower, upper)


@dataclass
class DecoupledCurve:
    """Aligned arrays retain native scan order; component voltages use loss polarity.

    ``residual`` is signed model discrepancy, with no zero clipping. Invalid and
    opposite-current rows remain NaNs in analysis arrays, separating acquired
    segments. ``measured_*`` and ``source_indices`` retain the input record.
    """

    measured_potential: np.ndarray
    measured_current: np.ndarray
    potential: np.ndarray
    current: np.ndarray
    current_magnitude: np.ndarray
    drive: np.ndarray
    sign: float
    e_rev: float
    eta_total: np.ndarray
    eta_kin: np.ndarray
    eta_ohm: np.ndarray
    eta_rcl: np.ndarray
    residual: np.ndarray
    fit: dict
    warnings: list[str]
    settings: DecouplingSettings
    source_indices: np.ndarray
    reaction: str

    def evaluate_at(self, potential, current_magnitude) -> dict[str, np.ndarray]:
        """Evaluate the fitted model at native, reference-converted potentials."""
        voltage, density = np.broadcast_arrays(
            np.asarray(potential, dtype=float), np.asarray(current_magnitude, dtype=float),
        )
        values = {key: np.full(voltage.shape, np.nan) for key in
                  ("eta_total", "eta_kin", "eta_ohm", "eta_rcl", "residual")}
        valid = np.isfinite(voltage) & np.isfinite(density) & (density > 0)
        if not np.any(valid):
            return values
        j = density[valid]
        total = self.sign * (voltage[valid] - self.e_rev)
        kinetic = self.fit["b_kin"] * np.log10(j / self.fit["i0"])
        ohmic = j * evaluate_hfr(self.settings.hfr, j)
        catalyst = (calculate_RCL_overpotential(j, self.settings.rcl, self.fit["b_kin"])
                    if self.settings.include_rcl else np.zeros_like(j))
        for key, component in zip(values, (total, kinetic, ohmic, catalyst,
                                          total - (kinetic + ohmic + catalyst))):
            values[key][valid] = component
        return values

    def breakdown_at(self, target_j: float) -> dict[str, float]:
        """Evaluate a target within a measured contiguous segment, never extrapolating."""
        target = float(target_j)
        if not np.isfinite(target) or target <= 0:
            raise ValueError("Breakdown target current must be a finite value greater than zero.")
        valid = np.isfinite(self.current_magnitude) & np.isfinite(self.potential)
        indices = np.flatnonzero(valid)
        if not len(indices):
            raise ValueError("Cannot compute a breakdown without finite current/potential data.")
        low = float(np.min(self.current_magnitude[indices]))
        high = float(np.max(self.current_magnitude[indices]))
        if not low <= target <= high:
            raise ValueError(
                f"Breakdown target {target:g} A/cm² is outside the measured "
                f"range {low:g}–{high:g} A/cm²."
            )
        runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        for run in runs:
            current = self.current_magnitude[run]
            if float(np.min(current)) <= target <= float(np.max(current)):
                order = np.argsort(current, kind="stable")
                potential = float(np.interp(target, current[order], self.potential[run][order]))
                values = {key: float(value) for key, value in
                          self.evaluate_at(potential, target).items()}
                return {**values, "native_potential": potential, "e_rev": self.e_rev}
        raise ValueError(f"Breakdown target {target:g} A/cm² falls in an acquisition gap.")


def decouple_curve(potential, current, settings: DecouplingSettings, reaction="auto",
                   source_indices=None, *, tafel_method="nested-window") -> DecoupledCurve:
    """Fit one measured scan, retaining its adjacency, gaps and native polarity.

    Current inputs must already be A/cm². Reference conversion occurs exactly
    once here. The default nested-window method retains the established sorted
    single-analysis calculation. Contiguous-stable fitting instead receives the
    native points and gap mappings. Returned arrays remain acquisition aligned.
    """
    voltage = np.asarray(potential, dtype=float)
    density = np.asarray(current, dtype=float)
    if voltage.ndim != 1 or density.shape != voltage.shape:
        raise ValueError("Potential and current must be matching one-dimensional arrays.")
    if source_indices is None:
        source_indices = np.arange(len(voltage))
    source_indices = np.asarray(source_indices)
    if source_indices.ndim != 1 or source_indices.shape != voltage.shape:
        raise ValueError("Source indices must match the potential/current arrays.")
    finite = np.isfinite(voltage) & np.isfinite(density)
    reaction = str(reaction).strip().lower()
    if reaction not in {"auto", "oer", "her"}:
        raise ValueError("Reaction must be Auto, OER or HER.")
    if reaction == "auto":
        reaction = "her" if np.count_nonzero(density[finite] < 0) > np.count_nonzero(density[finite] > 0) else "oer"
    keep = finite & ((density < 0) if reaction == "her" else (density > 0))
    if not np.any(keep):
        raise ValueError(
            "No current data of the expected sign after filtering "
            "(Reaction = OER expects i > 0, HER expects i < 0). Check the Reaction selector."
        )
    converted = settings.correction.apply(voltage)
    is_cell = settings.correction.electrode_setup == "Two-electrode"
    reversible = 1.2291 - .0008456 * (settings.temperature - 298.15)
    sign = (-1.0 if float(np.median(voltage[keep])) < 0 else 1.0) if is_cell else (
        -1.0 if reaction == "her" else 1.0)
    e_rev = sign * reversible if is_cell else (0.0 if reaction == "her" else reversible)
    analyzed = np.where(keep, density, np.nan)
    magnitude = np.abs(analyzed)
    total = sign * (converted - e_rev)
    if tafel_method == "nested-window":
        order = np.flatnonzero(keep)
        order = order[np.argsort(voltage[order], kind="stable")]
        result, warning = iterative_tafel_fit(
            total[order], magnitude[order], 0.0, settings.hfr, settings.rcl,
            settings.i_lower, settings.i_upper, include_rcl=settings.include_rcl,
        )
    elif tafel_method == "contiguous-stable":
        result, warning = iterative_stable_tafel_fit(
            total, magnitude, 0.0, settings.hfr, settings.rcl,
            settings.i_lower, settings.i_upper, include_rcl=settings.include_rcl,
            source_indices=source_indices,
        )
    else:
        raise ValueError(f"Unknown Tafel fitting method: {tafel_method}.")
    if result is None:
        raise ValueError(warning)
    fit = result[0]
    if not np.isfinite(fit["b_kin"]) or fit["b_kin"] <= 0 or not np.isfinite(fit["i0"]) or fit["i0"] <= 0:
        raise ValueError("The selected Tafel range did not produce a positive finite kinetic fit.")
    empty = np.full(voltage.shape, np.nan)
    curve = DecoupledCurve(
        measured_potential=voltage.copy(), measured_current=density.copy(),
        potential=converted, current=analyzed, current_magnitude=magnitude,
        drive=sign * converted, sign=sign, e_rev=e_rev,
        eta_total=empty.copy(), eta_kin=empty.copy(), eta_ohm=empty.copy(),
        eta_rcl=empty.copy(), residual=empty.copy(), fit=fit,
        warnings=[warning] if warning else [], settings=settings,
        source_indices=source_indices.copy(), reaction=reaction,
    )
    for key, values in curve.evaluate_at(converted, magnitude).items():
        setattr(curve, key, values)
    return curve
