"""Validated AEM catalyst-ink recipe calculations.

The calculator intentionally rejects impossible recipes instead of replacing
undefined quantities with zero.  This module is pure calculation code: valid
parameters produce a complete theoretical and spray-loss-adjusted recipe, and
invalid parameters raise :class:`ValueError` with actionable field messages.
"""
from __future__ import annotations

import math
from typing import Any


def _number(value: Any, label: str, errors: list[str]) -> float:
    """Return a finite float, recording a validation error when necessary."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label} must be a number.")
        return 0.0
    if not math.isfinite(number):
        errors.append(f"{label} must be finite.")
        return 0.0
    return number


def _validated_params(params: dict) -> dict:
    """Normalize and validate recipe parameters in one pass."""
    errors: list[str] = []
    mode = params.get("mode", "area")
    if mode not in {"area", "catalyst"}:
        errors.append("Mode must be 'area' or 'catalyst'.")

    loading = _number(params.get("loading", 0.0), "Loading", errors)
    ink_conc = _number(params.get("ink_density", 0.0), "Ink concentration", errors)
    length = _number(params.get("length", 0.0), "Active length", errors)
    width = _number(params.get("width", 0.0), "Active width", errors)
    avail_cat = _number(params.get("avail_cat", 0.0), "Available catalyst", errors)

    if mode == "area":
        if length <= 0:
            errors.append("Active length must be above zero in area mode.")
        if width <= 0:
            errors.append("Active width must be above zero in area mode.")
        if loading <= 0:
            errors.append("Loading must be above zero in area mode.")
    elif mode == "catalyst" and avail_cat <= 0:
        errors.append("Available catalyst must be above zero in catalyst mode.")
    if ink_conc <= 0:
        errors.append("Ink concentration must be above zero.")

    raw_ionomers = params.get("ionomers", []) or []
    if not isinstance(raw_ionomers, (list, tuple)):
        errors.append("Ionomers must be a list of recipe rows.")
        raw_ionomers = []
    ionomers = []
    for index, raw in enumerate(raw_ionomers, start=1):
        if not isinstance(raw, dict):
            errors.append(f"Ionomer {index} must be a recipe row.")
            continue
        ratio = _number(raw.get("ratio", 0.0), f"Ionomer {index} ratio", errors)
        conc = _number(raw.get("conc", 0.0), f"Ionomer {index} concentration", errors)
        density = _number(raw.get("density", 0.0), f"Ionomer {index} density", errors)
        if not 0 <= ratio < 100:
            errors.append(f"Ionomer {index} ratio must be at least 0% and below 100%.")
        if conc < 0 or conc > 100:
            errors.append(f"Ionomer {index} concentration must be between 0% and 100%.")
        if density < 0:
            errors.append(f"Ionomer {index} density cannot be negative.")
        if ratio > 0 and conc <= 0:
            errors.append(f"Ionomer {index} concentration must be above zero when used.")
        if ratio > 0 and density <= 0:
            errors.append(f"Ionomer {index} density must be above zero when used.")
        ionomers.append({"ratio": ratio, "conc": conc, "density": density})

    raw_extras = params.get("extras", []) or []
    if not isinstance(raw_extras, (list, tuple)):
        errors.append("Other solids must be a list of recipe rows.")
        raw_extras = []
    extras = []
    for index, raw in enumerate(raw_extras, start=1):
        if not isinstance(raw, dict):
            errors.append(f"Other solid {index} must be a recipe row.")
            continue
        ratio = _number(raw.get("ratio", 0.0), f"Other solid {index} ratio", errors)
        if not 0 <= ratio < 100:
            errors.append(f"Other solid {index} ratio must be at least 0% and below 100%.")
        extras.append({"name": str(raw.get("name") or "Material"), "ratio": ratio})

    if sum(row["ratio"] for row in extras) >= 100:
        errors.append("Other-solid ratios must total less than 100%.")

    if errors:
        raise ValueError("Invalid ink recipe:\n- " + "\n- ".join(errors))

    return {
        "mode": mode,
        "loading": loading,
        "ink_density": ink_conc,
        "length": length,
        "width": width,
        "avail_cat": avail_cat,
        "ionomers": ionomers,
        "extras": extras,
    }


def calculate_ink_recipe(params: dict) -> dict:
    """Calculate a complete ink recipe from validated parameters.

    ``ink_density`` is retained as the input key for backward compatibility,
    but its unit is mg catalyst/mL and it is therefore reported to users as ink
    concentration.
    """
    values = _validated_params(params)
    mode = values["mode"]
    loading = values["loading"]
    ink_conc = values["ink_density"]
    length = values["length"]
    width = values["width"]

    catalyst_mg = (
        loading * length * width if mode == "area" else values["avail_cat"]
    )

    total_ionomer_mg = 0.0
    total_dispersion_mass_mg = 0.0
    total_dispersion_volume_ul = 0.0
    ionomer_results = []
    for index, ionomer in enumerate(values["ionomers"], start=1):
        ratio = ionomer["ratio"] / 100.0
        active_mg = catalyst_mg * ratio / (1.0 - ratio)
        dispersion_fraction = ionomer["conc"] / 100.0
        dispersion_mg = active_mg / dispersion_fraction if ratio > 0 else 0.0
        volume_ul = dispersion_mg / ionomer["density"] if ratio > 0 else 0.0
        total_ionomer_mg += active_mg
        total_dispersion_mass_mg += dispersion_mg
        total_dispersion_volume_ul += volume_ul
        ionomer_results.append({
            "type": index,
            "active_mg": active_mg,
            "dispersion_mg": dispersion_mg,
            "volume_ul": volume_ul,
        })

    extra_fraction = sum(row["ratio"] for row in values["extras"]) / 100.0
    base_solids_mg = catalyst_mg + total_ionomer_mg
    total_solids_mg = base_solids_mg / (1.0 - extra_fraction)
    extra_results = [
        {
            "name": row["name"],
            "mass_mg": total_solids_mg * row["ratio"] / 100.0,
        }
        for row in values["extras"]
    ]
    total_extra_mg = sum(row["mass_mg"] for row in extra_results)
    ink_volume_ml = catalyst_mg / ink_conc

    adjusted = None
    if mode == "area":
        loss_factor = min(0.212 * ((1.0 / length) + (1.0 / width)), 0.99)
        scale_factor = 1.0 / (1.0 - loss_factor)
        adjusted = {
            "loss_factor": loss_factor,
            "scale_factor": scale_factor,
            "required_catalyst_mg": catalyst_mg * scale_factor,
            "ink_volume_ml": ink_volume_ml * scale_factor,
            "total_ionomer_mg": total_ionomer_mg * scale_factor,
            "total_dispersion_mass_mg": total_dispersion_mass_mg * scale_factor,
            "total_dispersion_volume_ul": total_dispersion_volume_ul * scale_factor,
            "total_extra_mg": total_extra_mg * scale_factor,
            "total_solids_mg": total_solids_mg * scale_factor,
            "ionomers": [
                {
                    "type": row["type"],
                    "active_mg": row["active_mg"] * scale_factor,
                    "dispersion_mg": row["dispersion_mg"] * scale_factor,
                    "volume_ul": row["volume_ul"] * scale_factor,
                    # Keep the old key for callers that consumed it.
                    "required_volume_ul": row["volume_ul"] * scale_factor,
                }
                for row in ionomer_results
            ],
            "extras": [
                {"name": row["name"], "mass_mg": row["mass_mg"] * scale_factor}
                for row in extra_results
            ],
        }

    return {
        "catalyst_mg": catalyst_mg,
        "ink_volume_ml": ink_volume_ml,
        "total_ionomer_mg": total_ionomer_mg,
        "total_dispersion_mass_mg": total_dispersion_mass_mg,
        "total_dispersion_volume_ul": total_dispersion_volume_ul,
        "total_extra_mg": total_extra_mg,
        "total_solids_mg": total_solids_mg,
        "ionomers": ionomer_results,
        "extras": extra_results,
        "adjusted": adjusted,
    }
