"""Pure forward model for synthetic HER, OER, and electrolyzer LSV curves."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.optimize import brentq

from workbench.tools.lsv.logic_tafel import (
    calculate_RCL_overpotential,
    calculate_RCL_utilization,
)


NOMINAL_MIN_CURRENT = 1e-6  # A cm^-2
N_POINTS = 400
MAX_VOLTAGE_SOLVE_CURRENT = 1e6  # A cm^-2; numerical safety bound


@dataclass(frozen=True)
class ElectrodeParameters:
    """Parameters for one Tafel/R_CL electrode branch."""

    b: float
    i0: float
    rcl: float = 0.0


@dataclass(frozen=True)
class CatalystPreset:
    """An editable, condition-specific example rather than a material constant."""

    b: float
    i0: float
    reference_temperature_k: float
    conditions: str
    source: str


OER_PRESETS: dict[str, CatalystPreset] = {
    "NiFe-LDH (1 M KOH)": CatalystPreset(
        b=0.066,
        i0=8.18e-7,
        reference_temperature_k=298.15,
        conditions=(
            "NiFe-LDH pyramid electrode; 1 M KOH; 25 °C; geometric area; "
            "85% iR-corrected LSV."
        ),
        source="https://www.nature.com/articles/s41598-021-04347-9",
    ),
    "Commercial IrO₂ (PEM MEA)": CatalystPreset(
        b=0.04152,
        i0=10 ** -5.80,
        reference_temperature_k=353.15,
        conditions=(
            "Commercial IrO₂, 2 mg cm⁻² anode loading; PEM electrolyzer MEA; "
            "80 °C."
        ),
        source="https://doi.org/10.1021/acs.energyfuels.3c03238",
    ),
}

HER_PRESETS: dict[str, CatalystPreset] = {
    "Pt/C (1 M NaOH)": CatalystPreset(
        b=0.033,
        i0=3.2e-4,
        reference_temperature_k=298.15,
        conditions=(
            "Pt/C/Nafion; 1 M NaOH; 25 °C room-temperature reference; "
            "1 atm H₂; 1 mV s⁻¹; iR-corrected."
        ),
        source="https://www.rsc.org/suppdata/ee/c4/c4ee02940b/c4ee02940b1.pdf",
    ),
    "Pt/C (1 M H₂SO₄)": CatalystPreset(
        b=0.027,
        i0=7.1e-6,
        reference_temperature_k=298.15,
        conditions=(
            "Pt/C/Nafion; 1 M H₂SO₄; 25 °C room-temperature reference; "
            "1 atm H₂; 1 mV s⁻¹; iR-corrected."
        ),
        source="https://www.rsc.org/suppdata/ee/c4/c4ee02940b/c4ee02940b1.pdf",
    ),
}


def oer_reversible_potential(temperature_k: float) -> float:
    """Return the OER equilibrium potential versus RHE used by LSV Decoupling."""
    return 1.2291 - 0.0008456 * (temperature_k - 298.15)


def _finite_float(name: str, value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number.") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _coerce_electrode(name: str, value) -> ElectrodeParameters:
    if isinstance(value, ElectrodeParameters):
        electrode = value
    elif isinstance(value, Mapping):
        try:
            electrode = ElectrodeParameters(
                b=value["b"], i0=value["i0"], rcl=value.get("rcl", 0.0)
            )
        except KeyError as exc:
            raise ValueError(f"{name} parameters require b and i0.") from exc
    elif value is None:
        raise ValueError(f"{name} parameters are required for this mode.")
    else:
        raise ValueError(f"{name} parameters must be a mapping or ElectrodeParameters.")

    b = _finite_float(f"{name} Tafel slope", electrode.b)
    i0 = _finite_float(f"{name} exchange current density", electrode.i0)
    rcl = _finite_float(f"{name} R_CL", electrode.rcl)
    if b <= 0:
        raise ValueError(f"{name} Tafel slope must be greater than zero.")
    if i0 <= 0:
        raise ValueError(f"{name} exchange current density must be greater than zero.")
    if rcl < 0:
        raise ValueError(f"{name} R_CL must be nonnegative.")
    return ElectrodeParameters(b=b, i0=i0, rcl=rcl)


def _normalize_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("_", "-")
    aliases = {
        "oer": "oer",
        "her": "her",
        "two-electrode": "two-electrode",
        "two electrode": "two-electrode",
        "full-cell": "two-electrode",
        "full cell": "two-electrode",
    }
    if normalized not in aliases:
        raise ValueError("Mode must be OER, HER, or Two-electrode.")
    return aliases[normalized]


def _normalize_limit_mode(limit_mode: str) -> str:
    normalized = str(limit_mode).strip().lower().replace("_", " ")
    aliases = {
        "current": "current",
        "current density": "current",
        "voltage": "voltage",
        "potential": "voltage",
    }
    if normalized not in aliases:
        raise ValueError("Endpoint mode must be Current density or Voltage.")
    return aliases[normalized]


def _electrode_components(current, electrode: ElectrodeParameters):
    current = np.asarray(current, dtype=float)
    eta_kin = electrode.b * np.log10(current / electrode.i0)
    utilization = calculate_RCL_utilization(current, electrode.rcl, electrode.b)
    eta_rcl = calculate_RCL_overpotential(current, electrode.rcl, electrode.b)
    return eta_kin, eta_rcl, utilization


def _evaluate(
    current,
    mode: str,
    e_rev: float,
    hfr: float,
    oer: ElectrodeParameters | None,
    her: ElectrodeParameters | None,
) -> dict[str, np.ndarray]:
    current = np.asarray(current, dtype=float)
    zeros = np.zeros_like(current)
    nans = np.full_like(current, np.nan)

    if oer is None:
        eta_kin_oer, eta_rcl_oer, utilization_oer = zeros, zeros, nans
    else:
        eta_kin_oer, eta_rcl_oer, utilization_oer = _electrode_components(
            current, oer
        )
    if her is None:
        eta_kin_her, eta_rcl_her, utilization_her = zeros, zeros, nans
    else:
        eta_kin_her, eta_rcl_her, utilization_her = _electrode_components(
            current, her
        )

    eta_ohm = current * hfr
    if mode == "oer":
        eta_total = eta_kin_oer + eta_rcl_oer + eta_ohm
        potential = e_rev + eta_total
        signed_current = current
    elif mode == "her":
        eta_total = eta_kin_her + eta_rcl_her + eta_ohm
        potential = -eta_total
        signed_current = -current
    else:
        eta_total = (
            eta_kin_oer + eta_rcl_oer + eta_kin_her + eta_rcl_her + eta_ohm
        )
        potential = e_rev + eta_total
        signed_current = current

    return {
        "current_magnitude": current,
        "signed_current": signed_current,
        "potential": potential,
        "e_rev": np.full_like(current, e_rev),
        "eta_total": eta_total,
        "eta_ohm": eta_ohm,
        "eta_kin_oer": eta_kin_oer,
        "eta_rcl_oer": eta_rcl_oer,
        "utilization_oer": utilization_oer,
        "eta_kin_her": eta_kin_her,
        "eta_rcl_her": eta_rcl_her,
        "utilization_her": utilization_her,
    }


def generate_lsv(
    mode: str,
    temperature_k: float,
    hfr: float,
    limit_mode: str,
    *,
    current_limit: float | None = None,
    voltage_limit: float | None = None,
    oer_params: ElectrodeParameters | Mapping | None = None,
    her_params: ElectrodeParameters | Mapping | None = None,
    n_points: int = N_POINTS,
) -> dict:
    """Generate a polarization curve and all modeled loss components.

    Current calculations use positive magnitudes. HER returns negative signed current
    and negative potential versus RHE; OER and full-cell modes use positive current.
    """
    resolved_mode = _normalize_mode(mode)
    resolved_limit_mode = _normalize_limit_mode(limit_mode)
    temperature_k = _finite_float("Temperature", temperature_k)
    hfr = _finite_float("HFR", hfr)
    if temperature_k <= 0:
        raise ValueError("Temperature must be greater than zero kelvin.")
    if hfr < 0:
        raise ValueError("HFR must be nonnegative.")
    if isinstance(n_points, bool) or not isinstance(n_points, (int, np.integer)):
        raise ValueError("Number of points must be an integer.")
    if n_points < 2:
        raise ValueError("Number of points must be at least two.")

    oer = (
        _coerce_electrode("OER", oer_params)
        if resolved_mode in {"oer", "two-electrode"}
        else None
    )
    her = (
        _coerce_electrode("HER", her_params)
        if resolved_mode in {"her", "two-electrode"}
        else None
    )
    active_i0 = [p.i0 for p in (oer, her) if p is not None]
    current_start = max(NOMINAL_MIN_CURRENT, *active_i0)
    e_rev = 0.0 if resolved_mode == "her" else oer_reversible_potential(temperature_k)

    def potential_at(current_value: float) -> float:
        evaluated = _evaluate(
            np.array([current_value]), resolved_mode, e_rev, hfr, oer, her
        )
        return float(evaluated["potential"][0])

    potential_start = potential_at(current_start)
    if resolved_limit_mode == "current":
        current_end = _finite_float("Current-density limit", current_limit)
        if current_end <= current_start:
            raise ValueError(
                "Current-density limit must be greater than the effective starting "
                f"current ({current_start:.6g} A/cm²)."
            )
        requested_endpoint = current_end
    else:
        target_voltage = _finite_float("Voltage limit", voltage_limit)
        increasing = resolved_mode != "her"
        if increasing and target_voltage <= potential_start:
            raise ValueError(
                f"Voltage limit must exceed the starting potential "
                f"({potential_start:.6g} V)."
            )
        if not increasing and target_voltage >= potential_start:
            raise ValueError(
                f"HER voltage limit must be below the starting potential "
                f"({potential_start:.6g} V vs RHE)."
            )

        def objective(current_value: float) -> float:
            return potential_at(current_value) - target_voltage

        if current_start >= MAX_VOLTAGE_SOLVE_CURRENT:
            raise ValueError(
                "The effective starting current is at or above the voltage solver's "
                f"{MAX_VOLTAGE_SOLVE_CURRENT:g} A/cm² safety bound."
            )
        upper = min(current_start * 10.0, MAX_VOLTAGE_SOLVE_CURRENT)
        while True:
            upper_value = potential_at(upper)
            crossed = (
                upper_value >= target_voltage
                if increasing
                else upper_value <= target_voltage
            )
            if crossed:
                break
            if upper >= MAX_VOLTAGE_SOLVE_CURRENT:
                raise ValueError(
                    "Voltage limit is not reachable below the numerical safety "
                    f"bound of {MAX_VOLTAGE_SOLVE_CURRENT:g} A/cm²."
                )
            upper = min(upper * 10.0, MAX_VOLTAGE_SOLVE_CURRENT)
        current_end = float(
            brentq(
                objective,
                current_start,
                upper,
                xtol=max(1e-14, current_start * 1e-12),
                rtol=1e-12,
            )
        )
        requested_endpoint = target_voltage

    currents = np.geomspace(current_start, current_end, int(n_points))
    result = _evaluate(currents, resolved_mode, e_rev, hfr, oer, her)
    result["metadata"] = {
        "mode": resolved_mode,
        "temperature_k": temperature_k,
        "hfr": hfr,
        "limit_mode": resolved_limit_mode,
        "requested_endpoint": requested_endpoint,
        "current_start": current_start,
        "current_end": float(currents[-1]),
        "potential_start": float(result["potential"][0]),
        "potential_end": float(result["potential"][-1]),
        "e_rev": e_rev,
        "n_points": int(n_points),
    }
    return result
