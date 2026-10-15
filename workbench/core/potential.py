"""Shared electrode setup, reference presets and potential-scale conversion.

Preset filling solutions and temperature are part of the reference definition.
See https://pineresearch.com/tools/reference-electrode-conversion-calculator/
and Kawashima et al., ACS Catal. 2023, 13, 1893–1898 (Hg/HgO),
https://doi.org/10.1021/acscatal.2c05655.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from workbench.core.tool_base import ParameterSpec

R_GAS = 8.31446261815324
FARADAY = 96485.33212
REFERENCE_TEMPERATURE_K = 298.15
DEFAULT_ELECTRODE_SETUP = "Two-electrode"
REFERENCE_OPTIONS = {
    "RHE": 0.0,
    "SHE": 0.0,
    "Ag/AgCl (sat. KCl, 25 °C)": 0.197,
    "Ag/AgCl (3 M KCl, 25 °C)": 0.210,
    "SCE (sat. KCl, 25 °C)": 0.2412,
    "Hg/HgO (1 M KOH, 25 °C)": 0.1053,
    "Custom E_ref vs SHE": None,
}
# Resolve saved legacy names to presets with explicit filling solutions.
_LEGACY_REFERENCES = {
    "Hg/HgO": "Hg/HgO (1 M KOH, 25 °C)",
    "Ag/AgCl": "Ag/AgCl (3 M KCl, 25 °C)",
    "SCE": "SCE (sat. KCl, 25 °C)",
}


def _finite(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a finite number.") from None
    if not np.isfinite(result):
        raise ValueError(f"{label} must be a finite number.")
    return result


@dataclass(frozen=True)
class PotentialCorrection:
    electrode_setup: str
    reference: str
    target_scale: str
    ph: float = 0.0
    temperature: float = REFERENCE_TEMPERATURE_K
    shift_v: float = 0.0
    e_ref_vs_she: float = 0.0
    notices: tuple[str, ...] = ()

    @classmethod
    def from_params(cls, params, *, default_reference="RHE", default_ph=0.0,
                    default_target="Original"):
        setup = params.get("electrode_setup", DEFAULT_ELECTRODE_SETUP)
        if setup not in {"Three-electrode", "Two-electrode"}:
            raise ValueError(f"Unknown electrode setup: {setup!r}")
        # Cell voltage and original-scale output ignore inactive reference settings.
        if setup == "Two-electrode":
            return cls(setup, "", "Original")
        target = params.get("target_scale", default_target)
        if target not in {"Original", "RHE", "SHE"}:
            raise ValueError(f"Unknown target potential scale: {target!r}")
        if target == "Original":
            return cls(setup, "", target)
        original_reference = params.get("reference", params.get("re_type", default_reference))
        reference = _LEGACY_REFERENCES.get(original_reference, original_reference)
        if reference not in REFERENCE_OPTIONS:
            raise ValueError(f"Unknown reference electrode: {reference!r}")
        ph = _finite(params.get("ph", default_ph), "pH")
        temperature = _finite(params.get("temperature", REFERENCE_TEMPERATURE_K), "Temperature")
        if temperature <= 0:
            raise ValueError("Temperature must be greater than zero kelvin.")
        nernst = 2.303 * R_GAS * temperature / FARADAY * ph
        if reference == "RHE":
            e_ref = -nernst
        else:
            preset = REFERENCE_OPTIONS[reference]
            e_ref = (_finite(params.get("custom_e_ref_vs_she", params.get("custom_e_ref", 0.0)),
                             "Custom reference potential") if preset is None else preset)
        shift = e_ref + (nernst if target == "RHE" else 0.0)
        notices = []
        if original_reference in _LEGACY_REFERENCES:
            notices.append(f"Legacy reference {original_reference!r} uses {reference}. "
                           "Use Custom E_ref vs SHE for a separately calibrated potential.")
        if (reference not in {"RHE", "SHE", "Custom E_ref vs SHE"}
                and not np.isclose(temperature, REFERENCE_TEMPERATURE_K, rtol=0, atol=0.01)):
            notices.append(f"{reference} uses a 25 °C reference potential at {temperature:g} K. "
                           "Only the RHE pH term changes with temperature; select Custom E_ref "
                           "vs SHE for a calibrated reference at the measurement temperature.")
        return cls(setup, reference, target, ph, temperature, shift, e_ref, tuple(notices))

    @property
    def is_converted(self):
        return self.electrode_setup == "Three-electrode" and self.target_scale != "Original"

    @property
    def warnings(self):
        return list(self.notices)

    def apply(self, values, unit="V"):
        result = np.asarray(values, dtype=float)
        if not self.is_converted:
            return result.copy()
        if unit not in {"V", "mV"}:
            raise ValueError("Potential unit must be V or mV for reference correction.")
        return result + self.shift_v * (1000.0 if unit == "mV" else 1.0)

    def axis_label(self, unit="V", symbol="Potential"):
        if self.electrode_setup == "Two-electrode":
            return f"Cell voltage ({unit})"
        scale = f" vs {self.target_scale}" if self.is_converted else ""
        return f"{symbol} ({unit}{scale})"

    def metadata(self):
        metadata = {"Electrode setup": self.electrode_setup,
                    "Potential scale": ("Cell voltage" if self.electrode_setup == "Two-electrode"
                                        else self.target_scale),
                    "Potential shift (V)": self.shift_v}
        if self.is_converted:
            metadata.update({"Reference electrode": self.reference,
                             "Reference E vs SHE (V)": self.e_ref_vs_she,
                             "pH": self.ph, "Temperature (K)": self.temperature})
        return metadata

    def describe(self):
        if self.electrode_setup == "Two-electrode":
            return "Two-electrode cell voltage; no reference-electrode shift is applied."
        if not self.is_converted:
            return "Original potential scale; no reference-electrode shift is applied."
        return (f"{self.reference} → {self.target_scale}; pH {self.ph:g}, "
                f"{self.temperature:g} K; potential shift {self.shift_v:+.6f} V.")


def convert_reference_to_rhe(potential, reference, ph, temperature_k,
                             custom_e_ref_vs_she=0.0):
    """Compatibility API for callers converting volt-valued arrays to RHE."""
    correction = PotentialCorrection.from_params({
        "electrode_setup": "Three-electrode",
        "reference": reference, "ph": ph, "temperature": temperature_k,
        "custom_e_ref_vs_she": custom_e_ref_vs_she, "target_scale": "RHE",
    })
    # Retain the historical third return value for already-RHE inputs.
    e_ref = 0.0 if reference == "RHE" else correction.e_ref_vs_she
    return correction.apply(potential), correction.shift_v, e_ref


def potential_parameter_schema(*, default_reference="RHE", default_ph=0.0,
                               allow_raw=True, default_target="Original",
                               include_temperature=True, include_input_unit=False,
                               show_if=None):
    """The same potential controls for each measured-potential workflow.

    ``allow_raw=False`` fixes three-electrode output to RHE (half-cell LSV fitting);
    two-electrode output always retains cell voltage.
    Input-unit fallback controls are only useful for parsers accepting unitless
    headers; recognized source units always take precedence.
    """
    group = "Potential"
    three = [("electrode_setup", "Three-electrode")]
    converted = three + ([("target_scale", ["RHE", "SHE"])] if allow_raw else [])
    specs = [ParameterSpec(
        "electrode_setup", "Electrode setup", kind="choice",
        choices=["Three-electrode", "Two-electrode"], default=DEFAULT_ELECTRODE_SETUP, group=group,
        tooltip="Three-electrode: potential versus a reference. Two-electrode: cell voltage.",
    )]
    if allow_raw:
        specs.append(ParameterSpec(
            "target_scale", "Output potential scale", kind="choice",
            choices=["Original", "RHE", "SHE"], default=default_target, group=group,
            tooltip="Original preserves measured potentials. RHE/SHE applies the reference correction.",
            show_if=three,
        ))
    specs.extend([
        ParameterSpec("reference", "Reference electrode", kind="choice",
                      choices=list(REFERENCE_OPTIONS), default=default_reference,
                      group=group, show_if=converted),
        ParameterSpec("custom_e_ref_vs_she", "Custom E_ref vs SHE", default=0.0,
                      step=0.001, decimals=5, suffix="V", group=group,
                      tooltip="Calibrated reference potential versus SHE at the measurement temperature.",
                      show_if=converted + [("reference", "Custom E_ref vs SHE")]),
        ParameterSpec("ph", "pH", default=default_ph, minimum=0.0, maximum=14.0,
                      step=0.1, decimals=2, group=group, show_if=converted,
                      tooltip="Electrolyte pH at the measurement temperature."),
    ])
    if include_temperature:
        specs.append(ParameterSpec(
            "temperature", "Temperature", default=REFERENCE_TEMPERATURE_K,
            minimum=0.01, step=1.0, decimals=2, suffix="K", group=group, show_if=converted,
            tooltip="RHE conversion uses 2.303RT/F × pH. Preset reference offsets are at 25 °C.",
        ))
    if include_input_unit:
        specs.append(ParameterSpec(
            "potential_unit", "Unspecified potential unit", kind="choice",
            choices=["V", "mV"], default="V", group=group, show_if=converted,
            tooltip="Used only when a potential column has no unit. Recognized V/mV headers take precedence.",
        ))
    if show_if is not None:
        outer = [show_if] if isinstance(show_if, tuple) else list(show_if)
        for spec in specs:
            spec.show_if = outer + (spec.show_if or [])
    return specs
