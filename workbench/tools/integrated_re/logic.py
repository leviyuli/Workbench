"""Curve matching for integrated reference-electrode LSV measurements.

All curve operations retain acquisition order. A point removed by a potential
window (or an invalid measurement) also removes its adjoining line segments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


RECONSTRUCTED_COLUMNS = [
    "Current density (A/cm²)",
    "Anode E (V vs RHE)",
    "Cathode E (V vs RHE)",
    "Anode η (V)",
    "Cathode η (V)",
    "Reconstructed cell voltage (V)",
    "Cell overpotential (V)",
]
COMPARISON_COLUMNS = RECONSTRUCTED_COLUMNS + [
    "Measured cell voltage (V)",
    "Voltage difference (V)",
]


def _vector(values, label):
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain numeric values.") from exc
    if vector.ndim != 1:
        raise ValueError(f"{label} must be a one-dimensional array.")
    if not vector.size:
        raise ValueError(f"{label} must contain at least one point.")
    return vector


def _curve(potential, current, mask=None):
    potential = _vector(potential, "Potential")
    current = _vector(current, "Current density")
    if potential.size != current.size:
        raise ValueError("Potential and current density must have equal lengths.")
    valid = np.isfinite(potential) & np.isfinite(current)
    if mask is not None:
        selected = np.asarray(mask)
        if selected.ndim != 1 or selected.size != potential.size:
            raise ValueError("The branch mask must match the curve length.")
        valid &= selected.astype(bool)
    return potential, current, valid


def _selection(select):
    if select not in {"min", "max"}:
        raise ValueError("Crossing selection must be 'min' or 'max'.")
    return select


def _window(window, *, required=False):
    if window is None:
        if required:
            raise ValueError("Peak calibration requires a potential search window.")
        return None
    try:
        bounds = np.asarray(window, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Potential window must contain two finite bounds.") from exc
    if (
        bounds.shape != (2,)
        or not np.all(np.isfinite(bounds))
        or bounds[0] >= bounds[1]
    ):
        raise ValueError("Potential window requires finite lower < upper bounds.")
    return float(bounds[0]), float(bounds[1])


def normalize_current(current, role, convention="Auto"):
    """Orient a complete curve once, retaining zero and minority-sign currents.

    ``As recorded`` and ``Invert`` explicitly use multipliers +1 and -1.
    Automatic orientation uses the largest absolute finite current; ties use
    the first point in acquisition order.
    """
    values = _vector(current, "Current density")
    if role not in {"anode", "cathode", "cell"}:
        raise ValueError("Current role must be anode, cathode, or cell.")
    if convention not in {"Auto", "As recorded", "Invert"}:
        raise ValueError("Current convention must be Auto, As recorded, or Invert.")
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("Current density contains no finite measurements.")
    multiplier = -1.0 if convention == "Invert" else 1.0
    if convention == "Auto":
        finite_values = values[finite]
        dominant = finite_values[np.argmax(np.abs(finite_values))]
        expected = -1.0 if role == "cathode" else 1.0
        if dominant != 0.0 and np.sign(dominant) != expected:
            multiplier = -1.0
    return values * multiplier, multiplier


def potential_at_current(potential, current, targets, *, select="max", mask=None):
    """Return extreme potential crossings without sorting or extrapolating.

    ``current`` is oriented in the reaction direction (positive for the
    requested reaction). Only adjacent valid measured points form segments.
    Exact vertices remain usable even if they are isolated by a branch mask.
    """
    potential, current, valid = _curve(potential, current, mask)
    select = _selection(select)
    try:
        targets = np.asarray(targets, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Target currents must contain finite numeric values.") from exc
    if targets.ndim == 0:
        targets = targets.reshape(1)
    if targets.ndim != 1 or not np.all(np.isfinite(targets)):
        raise ValueError("Target currents must be a finite one-dimensional array.")
    result = np.full(targets.shape, np.nan, dtype=float)
    combine = np.fmax if select == "max" else np.fmin
    for index in np.flatnonzero(valid):
        exact = targets == current[index]
        result[exact] = combine(result[exact], potential[index])
    for index in np.flatnonzero(valid[:-1] & valid[1:]):
        left_current, right_current = current[index : index + 2]
        if left_current == right_current:
            # Exact vertices already account for both extremes of this segment.
            continue
        crossing = (
            (targets >= min(left_current, right_current))
            & (targets <= max(left_current, right_current))
        )
        fraction = (
            (targets[crossing] - left_current) / (right_current - left_current)
        )
        candidates = potential[index] + fraction * (
            potential[index + 1] - potential[index]
        )
        result[crossing] = combine(result[crossing], candidates)
    return result


def reconstruct_cell(
    anode_potential,
    anode_current,
    cathode_potential,
    cathode_current,
    *,
    anode_mask=None,
    cathode_mask=None,
    cell_potential=None,
    cell_current=None,
    cell_mask=None,
    erev=1.2291,
):
    """Reconstruct cell voltage at common positive current magnitudes.

    Electrode inputs use signed currents, with negative HER current. The
    returned residual is reconstructed voltage minus measured cell voltage.
    Optional measured-cell sampling points are included in the common grid.
    """
    try:
        erev = float(erev)
    except (TypeError, ValueError) as exc:
        raise ValueError("Reversible cell voltage must be finite.") from exc
    if not np.isfinite(erev):
        raise ValueError("Reversible cell voltage must be finite.")
    ea, ja, va = _curve(anode_potential, anode_current, anode_mask)
    ec, jc, vc = _curve(cathode_potential, cathode_current, cathode_mask)
    qc = -jc
    for label, q, valid in (("Anode", ja, va), ("Cathode", qc, vc)):
        if not np.any(valid & (q > 0)):
            raise ValueError(f"{label} LSV has no positive reaction-current branch.")
    lower = max(0.0, float(np.min(ja[va])), float(np.min(qc[vc])))
    upper = min(float(np.max(ja[va])), float(np.max(qc[vc])))
    if upper <= 0 or upper < lower:
        raise ValueError("Anode and cathode LSVs have no overlapping positive current range.")
    grids = [ja[va], qc[vc]]
    cell = None
    if (cell_potential is None) != (cell_current is None):
        raise ValueError("Measured cell potential and current must be supplied together.")
    if cell_potential is not None:
        cell = _curve(cell_potential, cell_current, cell_mask)
        grids.append(cell[1][cell[2]])
    candidates = np.unique(np.concatenate(grids))
    targets = candidates[(candidates > 0) & (candidates >= lower) & (candidates <= upper)]
    anode = potential_at_current(ea, ja, targets, select="max", mask=va)
    cathode = potential_at_current(ec, qc, targets, select="min", mask=vc)
    common = np.isfinite(anode) & np.isfinite(cathode)
    if not np.any(common):
        raise ValueError("Anode and cathode LSVs have no overlapping positive current crossings.")
    targets, anode, cathode = targets[common], anode[common], cathode[common]
    voltage = anode - cathode
    reconstructed = pd.DataFrame(
        {
            RECONSTRUCTED_COLUMNS[0]: targets,
            RECONSTRUCTED_COLUMNS[1]: anode,
            RECONSTRUCTED_COLUMNS[2]: cathode,
            RECONSTRUCTED_COLUMNS[3]: anode - erev,
            RECONSTRUCTED_COLUMNS[4]: cathode,
            RECONSTRUCTED_COLUMNS[5]: voltage,
            RECONSTRUCTED_COLUMNS[6]: voltage - erev,
        },
        columns=RECONSTRUCTED_COLUMNS,
    )
    comparison = pd.DataFrame(columns=COMPARISON_COLUMNS, dtype=float)
    warnings = []
    if cell is not None:
        measured = potential_at_current(
            cell[0], cell[1], targets, select="max", mask=cell[2],
        )
        available = np.isfinite(measured)
        if np.any(available):
            comparison = reconstructed.loc[available].copy().reset_index(drop=True)
            comparison[COMPARISON_COLUMNS[-2]] = measured[available]
            comparison[COMPARISON_COLUMNS[-1]] = voltage[available] - measured[available]
        else:
            warnings.append(
                "Measured cell LSV has no current overlap with the reconstructed curve; "
                "numerical comparison is unavailable."
            )
    return reconstructed, comparison, warnings


def find_feature(
    potential,
    current,
    method,
    *,
    target_current=0.01,
    window=None,
    select="max",
):
    """Locate a threshold crossing or a genuine interior reaction-current peak.

    ``current`` must already be oriented toward the selected reaction. Windows
    operate on the unshifted potential input, including for onset calibration.
    """
    potential, current, valid = _curve(potential, current)
    select = _selection(select)
    if method not in {"onset", "peak"}:
        raise ValueError("Feature method must be onset or peak.")
    bounds = _window(window, required=method == "peak")
    if bounds is not None:
        valid &= (potential >= bounds[0]) & (potential <= bounds[1])
    if method == "onset":
        try:
            target = float(target_current)
        except (TypeError, ValueError) as exc:
            raise ValueError("Onset current density must be finite and positive.") from exc
        if not np.isfinite(target) or target <= 0:
            raise ValueError("Onset current density must be finite and positive.")
        feature = potential_at_current(
            potential, current, [target], select=select, mask=valid,
        )[0]
        if not np.isfinite(feature):
            raise ValueError("Onset current density is not crossed in the selected reaction branch.")
        return {"potential": float(feature), "current": target}

    indices = np.flatnonzero(valid)
    runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    features = []
    for run in runs:
        if run.size < 3:
            continue
        peaks, properties = find_peaks(current[run], prominence=0)
        for position, prominence in zip(peaks, properties["prominences"]):
            index = run[position]
            if current[index] > 0 and prominence > 0:
                features.append(
                    {
                        "potential": float(potential[index]),
                        "current": float(current[index]),
                        "prominence": float(prominence),
                    }
                )
    if not features:
        raise ValueError("No interior reaction-current peak exists in the selected potential window.")
    extreme = 1.0 if select == "max" else -1.0
    return max(features, key=lambda feature: (
        feature["prominence"], extreme * feature["potential"],
    ))
