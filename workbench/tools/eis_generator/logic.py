"""Nonfaradaic, homogeneous through-plane two-rail CL impedance.

The model is Eq. 14 of Padgett et al., J. Electrochem. Soc. 170,
084512 (2023), DOI: 10.1149/1945-7111/acee25, with an ideal distributed
capacitance. Impedance and resistances are area-specific (ohm cm^2),
conductivities are S/cm, and capacitance is per geometric area.
"""

from collections.abc import Mapping
import math

import numpy as np


DEFAULT_PARAMETERS = {
    "electronic_conductivity": 0.010,
    "ionic_conductivity": 0.002,
    "thickness_um": 5.0,
    "capacitance_mf_cm2": 20.0,
    "external_resistance": 0.050,
    "f_min": 0.1,
    "f_max": 1e6,
    "n_points": 400,
}

_LABELS = {
    "electronic_conductivity": "Electronic conductivity",
    "ionic_conductivity": "Ionic conductivity",
    "thickness_um": "CL thickness",
    "capacitance_mf_cm2": "Areal capacitance",
    "external_resistance": "External series resistance",
    "f_min": "Minimum frequency",
    "f_max": "Maximum frequency",
    "n_points": "Number of frequency points",
}
_LOG_TWO_PI = math.log(2.0 * math.pi)
_LOG_FLOAT_MAX = math.log(np.finfo(float).max)
_LOG_SMALL_LAMBDA_SQUARED = math.log(1e-4)


def _finite_number(value, label, *, allow_zero=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    if number < 0 or (number == 0 and not allow_zero):
        condition = "nonnegative" if allow_zero else "greater than zero"
        raise ValueError(f"{label} must be {condition}.")
    return number


def _rail_summary(r_i, r_e):
    """Return model asymptotes without overflowing a resistance product."""
    total = r_i + r_e
    if not math.isfinite(total):
        raise ValueError("The sum of the CL resistances is not representable.")
    if total == 0:
        return total, 0.0, 0.0
    low, high = min(r_i, r_e), max(r_i, r_e)
    hfr = low / (1.0 + low / high)
    apparent = total - 3.0 * hfr
    if not all(math.isfinite(value) for value in (hfr, apparent)):
        raise ValueError("The derived CL resistances are not representable.")
    return total, hfr, apparent


def _derived_values(parameters):
    thickness_cm = parameters["thickness_um"] * 1e-4
    capacitance = parameters["capacitance_mf_cm2"] * 1e-3
    r_i = thickness_cm / parameters["ionic_conductivity"]
    r_e = thickness_cm / parameters["electronic_conductivity"]
    if not all(
        math.isfinite(value) and value > 0
        for value in (thickness_cm, capacitance, r_i, r_e)
    ):
        raise ValueError("The derived thickness, capacitance, or CL resistance is not representable.")
    total, hfr, apparent = _rail_summary(r_i, r_e)
    external = parameters["external_resistance"]
    resistances = {
        "r_i": r_i,
        "r_e": r_e,
        "hfr_cl": hfr,
        "hfr_total": external + hfr,
        "rcl_app": apparent,
        "width": apparent / 3.0,
        "lf_intercept": external + total / 3.0,
    }
    if not all(math.isfinite(value) and value > 0 for value in resistances.values()):
        raise ValueError("The derived CL intercepts or width are not representable.")
    return capacitance, resistances


def validate_parameters(parameters):
    """Merge defaults and return normalized core parameters only.

    Unknown keys are ignored so UI/export metadata can accompany a parameter
    mapping. Frequencies must satisfy f_min < f_max. Invalid or unrepresentable
    physical values raise ValueError.
    """
    if not isinstance(parameters, Mapping):
        raise ValueError("Generator parameters must be a mapping.")
    normalized = {}
    for key, default in DEFAULT_PARAMETERS.items():
        normalized[key] = _finite_number(
            parameters.get(key, default), _LABELS[key],
            allow_zero=(key == "external_resistance"),
        )
    count = normalized["n_points"]
    if not count.is_integer() or not 2 <= count <= 20000:
        raise ValueError("Number of frequency points must be an integer from 2 to 20000.")
    normalized["n_points"] = int(count)
    if normalized["f_min"] >= normalized["f_max"]:
        raise ValueError("Minimum frequency must be less than maximum frequency.")
    capacitance, resistances = _derived_values(normalized)
    # Endpoints bound the magnitude extremes of this passive capacitive model.
    endpoint_z = two_rail_impedance(
        [normalized["f_max"], normalized["f_min"]],
        resistances["r_i"], resistances["r_e"], capacitance,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        endpoint_z = endpoint_z + normalized["external_resistance"]
    if not np.all(np.isfinite(endpoint_z)):
        raise ValueError("The total impedance is not representable for these parameters.")
    return normalized


def two_rail_impedance(frequency, r_i, r_e, capacitance_f_cm2):
    """Return CL ASR impedance for positive frequencies (Hz).

    Either rail resistance may be zero. With both rails zero, the result is
    the ideal capacitor. Scalar or one-dimensional frequency inputs preserve
    their NumPy shape. This helper includes no external series resistance.
    """
    r_i = _finite_number(r_i, "Ionic resistance", allow_zero=True)
    r_e = _finite_number(r_e, "Electronic resistance", allow_zero=True)
    capacitance = _finite_number(capacitance_f_cm2, "Areal capacitance")
    try:
        raw_frequency = np.asarray(frequency)
        if raw_frequency.dtype.kind == "b" or np.iscomplexobj(raw_frequency):
            raise ValueError("Frequency must contain positive finite real numbers.")
        frequencies = np.asarray(frequency, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Frequency must contain positive finite real numbers.") from exc
    if frequencies.ndim > 1 or frequencies.size == 0:
        raise ValueError("Frequency must be a nonempty scalar or one-dimensional array.")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
        raise ValueError("Frequency must contain positive finite real numbers.")

    total, hfr, _ = _rail_summary(r_i, r_e)
    # Logs avoid overflowing intermediate products such as 2*pi*f*C.
    log_omega_c = np.log(frequencies) + math.log(capacitance) + _LOG_TWO_PI
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        if total == 0:
            result = -1j * np.exp(-log_omega_c)
        else:
            log_lambda_squared = log_omega_c + math.log(total)
            small = log_lambda_squared < _LOG_SMALL_LAMBDA_SQUARED
            result = np.empty(frequencies.shape, dtype=complex)
            if np.any(small):
                u = np.exp(log_lambda_squared[small])
                cap_magnitude = np.exp(-log_omega_c[small])
                # Lambda^2 = j*u. Explicit real/imaginary terms retain the
                # low-frequency real intercept beside a large capacitive tail.
                result[small] = (
                    total / 3.0 - u * u * (2.0 * total / 945.0 - hfr / 120.0)
                    + 1j * (-cap_magnitude + u * (hfr / 12.0 - total / 45.0))
                )
            if np.any(~small):
                log_lambda = 0.5 * log_lambda_squared[~small]
                if np.any(log_lambda > _LOG_FLOAT_MAX):
                    raise ValueError("The model frequency scale is not representable.")
                magnitude = np.exp(log_lambda)
                lam = (magnitude / math.sqrt(2.0)) * (1.0 + 1.0j)
                inverse_lam = (np.exp(-log_lambda) / math.sqrt(2.0)) * (1.0 - 1.0j)
                t = np.exp(-lam)
                denominator = -np.expm1(-2.0 * lam)
                coth = (1.0 + t * t) / denominator
                csch = 2.0 * t / denominator
                result[~small] = (
                    hfr + (total - 2.0 * hfr) * inverse_lam * coth
                    + 2.0 * hfr * inverse_lam * csch
                )
    if not np.all(np.isfinite(result)):
        raise ValueError("The impedance is not representable at the requested frequencies.")
    if np.any(result.imag >= 0):
        raise ValueError("The capacitive impedance is too small to represent at the requested frequencies.")
    return result


def generate_cl_eis(parameters):
    """Generate descending frequencies, CL/total spectra, and exact markers."""
    normalized = validate_parameters(parameters)
    capacitance, resistances = _derived_values(normalized)
    frequency = np.geomspace(
        normalized["f_max"], normalized["f_min"], normalized["n_points"]
    )
    if not np.all(np.isfinite(frequency)) or np.any(np.diff(frequency) >= 0):
        raise ValueError("The frequency range is too narrow to represent distinct descending points.")
    z_cl = two_rail_impedance(
        frequency, resistances["r_i"], resistances["r_e"], capacitance
    )
    with np.errstate(over="ignore", invalid="ignore"):
        z_total = z_cl + normalized["external_resistance"]
    if not np.all(np.isfinite(z_total)):
        raise ValueError("The total impedance is not representable for these parameters.")
    return {
        "parameters": normalized,
        "frequency": frequency,
        "z_cl": z_cl,
        "z_total": z_total,
        "resistances": resistances,
    }
