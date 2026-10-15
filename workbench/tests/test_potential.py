"""Shared reference conversion and the controls used by measured-potential tools."""

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

from workbench.core.potential import (
    PotentialCorrection,
    convert_reference_to_rhe,
    potential_parameter_schema,
)
from workbench.core.tool_base import ParameterSpec
from workbench.tools.ecsa.tool import EcsaTool
from workbench.tools.it_integrate.tool import ItIntegrateTool
from workbench.tools.lsv.tool import LsvTool
from workbench.widgets.parameter_panel import ParameterPanel


def _correction(**updates):
    return PotentialCorrection.from_params({
        "electrode_setup": "Three-electrode",
        "target_scale": "RHE", "reference": "SHE", "ph": 7.0,
        "temperature": 298.15, **updates,
    })


@pytest.mark.parametrize("ph,temperature", [(0, 298.15), (7, 273.15), (14, 353.15)])
def test_already_rhe_data_are_not_corrected_twice(ph, temperature):
    values = np.array([-0.2, 0.0, 1.5])
    correction = _correction(reference="RHE", ph=ph, temperature=temperature)

    np.testing.assert_array_equal(correction.apply(values), values)
    assert correction.shift_v == 0.0
    assert correction.axis_label() == "Potential (V vs RHE)"
    assert not correction.warnings


@pytest.mark.parametrize("ph,temperature", [(0, 298.15), (7, 298.15), (14, 333.15)])
def test_she_and_rhe_conversions_are_inverse(ph, temperature):
    original = np.array([-0.3, 0.2, 1.0])
    to_rhe = _correction(ph=ph, temperature=temperature)
    to_she = _correction(reference="RHE", target_scale="SHE", ph=ph,
                         temperature=temperature)

    assert to_rhe.shift_v == pytest.approx(0.05917 * ph * temperature / 298.15,
                                         abs=1e-6)
    assert to_she.shift_v == pytest.approx(-to_rhe.shift_v)
    np.testing.assert_allclose(to_she.apply(to_rhe.apply(original)), original)


@pytest.mark.parametrize("reference,offset", [
    ("Ag/AgCl (sat. KCl, 25 °C)", 0.197),
    ("Ag/AgCl (3 M KCl, 25 °C)", 0.210),
    ("SCE (sat. KCl, 25 °C)", 0.2412),
    ("Hg/HgO (1 M KOH, 25 °C)", 0.1053),
    ("Custom E_ref vs SHE", 0.1432),
])
def test_reference_offsets_and_temperature_dependent_rhe_term(reference, offset):
    at_25 = _correction(reference=reference, custom_e_ref_vs_she=offset)
    at_60 = _correction(reference=reference, custom_e_ref_vs_she=offset,
                         temperature=333.15)
    to_she = _correction(reference=reference, custom_e_ref_vs_she=offset,
                        target_scale="SHE", temperature=333.15)

    assert to_she.shift_v == pytest.approx(offset)
    assert at_25.shift_v == pytest.approx(offset + 7 * 0.05917, abs=1e-6)
    assert at_60.shift_v - offset == pytest.approx(
        (at_25.shift_v - offset) * 333.15 / 298.15,
    )


@pytest.mark.parametrize("setup,target", [
    (None, "RHE"),
    ("Two-electrode", "not a scale"),
    ("Three-electrode", "Original"),
])
def test_inactive_correction_ignores_stale_reference_ph_temperature_and_units(setup, target):
    params = dict(target_scale=target, reference="not a reference",
                  ph="invalid", temperature=float("nan"), custom_e_ref_vs_she=float("inf"))
    if setup is not None:
        params["electrode_setup"] = setup
    correction = PotentialCorrection.from_params(params)
    original = np.array([0.2, 1.5, 2.0])
    output = correction.apply(original, unit="raw")

    np.testing.assert_array_equal(output, original)
    assert not np.shares_memory(output, original)
    assert correction.shift_v == 0.0
    assert not correction.is_converted
    assert not correction.warnings
    assert "pH" not in correction.metadata()
    if setup in {None, "Two-electrode"}:
        assert correction.axis_label("mV") == "Cell voltage (mV)"
        assert correction.metadata()["Potential scale"] == "Cell voltage"
    else:
        assert correction.axis_label("mV", symbol="E") == "E (mV)"


def test_mv_correction_matches_volt_correction_without_mutating_input():
    correction = _correction(reference="Custom E_ref vs SHE", ph=0,
                             custom_e_ref_vs_she=0.2)
    volts = np.array([-0.1, 0.0, 0.3])

    np.testing.assert_allclose(correction.apply(volts), [0.1, 0.2, 0.5])
    np.testing.assert_allclose(correction.apply(volts * 1000, "mV"), [100, 200, 500])
    np.testing.assert_array_equal(volts, [-0.1, 0.0, 0.3])


@pytest.mark.parametrize("unit", ["raw", "kV", "", None])
def test_active_correction_rejects_unknown_units(unit):
    with pytest.raises(ValueError, match="unit must be V or mV"):
        _correction().apply([0.0], unit=unit)


@pytest.mark.parametrize("updates,message", [
    ({"reference": "Ag/AgCl unknown filling solution"}, "Unknown reference"),
    ({"target_scale": "NHE"}, "Unknown target"),
    ({"electrode_setup": "Four-electrode"}, "Unknown electrode setup"),
    ({"temperature": 0}, "greater than zero kelvin"),
    ({"temperature": -10}, "greater than zero kelvin"),
])
def test_active_correction_rejects_invalid_options(updates, message):
    with pytest.raises(ValueError, match=message):
        _correction(**updates)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), None, "bad"])
@pytest.mark.parametrize("key,label", [
    ("ph", "pH"),
    ("temperature", "Temperature"),
    ("custom_e_ref_vs_she", "Custom reference potential"),
])
def test_active_numeric_settings_must_be_finite(key, label, value):
    with pytest.raises(ValueError, match=f"{label} must be a finite number"):
        _correction(reference="Custom E_ref vs SHE", **{key: value})


@pytest.mark.parametrize("reference", ["RHE", "SHE", "Custom E_ref vs SHE"])
def test_temperature_notice_is_not_emitted_for_nonpreset_references(reference):
    assert not _correction(reference=reference, temperature=353.15).warnings


@pytest.mark.parametrize("reference", [
    "Ag/AgCl (sat. KCl, 25 °C)", "Ag/AgCl (3 M KCl, 25 °C)",
    "SCE (sat. KCl, 25 °C)", "Hg/HgO (1 M KOH, 25 °C)",
])
def test_preset_temperature_notice_identifies_the_calibration_limit(reference):
    assert not _correction(reference=reference).warnings
    warnings = _correction(reference=reference, temperature=353.15).warnings
    assert len(warnings) == 1
    assert "25 °C" in warnings[0]
    assert "353.15 K" in warnings[0]
    assert "Custom E_ref vs SHE" in warnings[0]


@pytest.mark.parametrize("legacy,explicit,offset", [
    ("Ag/AgCl", "Ag/AgCl (3 M KCl, 25 °C)", 0.210),
    ("SCE", "SCE (sat. KCl, 25 °C)", 0.2412),
    ("Hg/HgO", "Hg/HgO (1 M KOH, 25 °C)", 0.1053),
])
def test_legacy_reference_names_resolve_to_documented_filling_solutions(legacy, explicit, offset):
    correction = PotentialCorrection.from_params({
        "electrode_setup": "Three-electrode",
        "re_type": legacy, "target_scale": "SHE", "ph": 0,
    })
    assert correction.reference == explicit
    assert correction.shift_v == pytest.approx(offset)
    assert len(correction.warnings) == 1
    assert explicit in correction.warnings[0]


def test_compatibility_api_preserves_custom_offset_and_rhe_return_contract():
    output, shift, reference_offset = convert_reference_to_rhe(
        [0.1], "Custom E_ref vs SHE", 0, 298.15, 0.145,
    )
    np.testing.assert_allclose(output, [0.245])
    assert shift == reference_offset == 0.145
    output, shift, reference_offset = convert_reference_to_rhe([0.1], "RHE", 14, 333.15)
    np.testing.assert_array_equal(output, [0.1])
    assert shift == reference_offset == 0
    assert _correction(reference="Custom E_ref vs SHE", custom_e_ref=0.145,
                       ph=0).shift_v == 0.145


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance()
    if app is None:
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
            app = QApplication([])
    yield app


def _assert_rows_visible(panel, keys, visible):
    for key in keys:
        assert panel._rows[key].isHidden() is not visible, key
        assert panel._row_labels[key].isHidden() is not visible, key


def test_reference_controls_follow_setup_target_and_custom_reference(qt_app):
    panel = ParameterPanel(potential_parameter_schema(include_input_unit=True))
    correction_fields = ["reference", "ph", "temperature", "potential_unit"]
    try:
        assert panel.values()["electrode_setup"] == "Two-electrode"
        _assert_rows_visible(panel, ["electrode_setup"], True)
        _assert_rows_visible(panel, correction_fields + ["target_scale", "custom_e_ref_vs_she"], False)

        panel.set_values({"electrode_setup": "Three-electrode", "target_scale": "RHE"})
        _assert_rows_visible(panel, correction_fields, True)
        _assert_rows_visible(panel, ["custom_e_ref_vs_she"], False)

        panel.set_values({"reference": "Custom E_ref vs SHE"})
        _assert_rows_visible(panel, ["custom_e_ref_vs_she"], True)

        panel.set_values({"electrode_setup": "Two-electrode"})
        _assert_rows_visible(panel, ["electrode_setup"], True)
        _assert_rows_visible(panel, correction_fields + ["target_scale", "custom_e_ref_vs_she"], False)

        panel.set_values({"electrode_setup": "Three-electrode", "target_scale": "Original"})
        _assert_rows_visible(panel, ["target_scale"], True)
        _assert_rows_visible(panel, correction_fields + ["custom_e_ref_vs_she"], False)
    finally:
        panel.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def test_outer_visibility_gate_also_hides_all_reference_controls(qt_app):
    specs = [ParameterSpec("mode", "Mode", kind="choice", choices=["Current", "Potential"],
                           default="Current")]
    specs += potential_parameter_schema(default_target="RHE", show_if=("mode", "Potential"))
    panel = ParameterPanel(specs)
    try:
        _assert_rows_visible(panel, [spec.key for spec in specs[1:]], False)
        panel.set_values({"mode": "Potential", "electrode_setup": "Three-electrode",
                          "reference": "Custom E_ref vs SHE"})
        _assert_rows_visible(panel, [spec.key for spec in specs[1:]], True)
        panel.set_values({"electrode_setup": "Two-electrode"})
        _assert_rows_visible(panel, ["electrode_setup"], True)
        _assert_rows_visible(panel, [spec.key for spec in specs[2:]], False)
    finally:
        panel.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


@pytest.mark.parametrize("tool", [EcsaTool, LsvTool, ItIntegrateTool])
def test_measured_potential_tools_share_reference_choices_and_visibility(qt_app, tool):
    specs = tool().parameter_schema()
    by_key = {spec.key: spec for spec in specs}
    assert by_key["electrode_setup"].default == "Two-electrode"
    shared = {spec.key: spec for spec in potential_parameter_schema()}
    assert len(by_key) == len(specs), "A shared control must not shadow another parameter"
    for key in ["electrode_setup", "reference", "custom_e_ref_vs_she", "ph"]:
        for attribute in ["label", "choices", "suffix"]:
            assert getattr(by_key[key], attribute) == getattr(shared[key], attribute)
    panel = ParameterPanel(specs)
    try:
        assert panel.values()["electrode_setup"] == "Two-electrode"
        _assert_rows_visible(panel, ["reference", "custom_e_ref_vs_she", "ph"], False)
        panel.set_values({"electrode_setup": "Three-electrode", "target_scale": "RHE",
                          "reference": "Custom E_ref vs SHE"})
        _assert_rows_visible(panel, ["electrode_setup", "reference", "custom_e_ref_vs_she", "ph"], True)
        panel.set_values({"electrode_setup": "Two-electrode"})
        _assert_rows_visible(panel, ["electrode_setup"], True)
        hidden = ["reference", "custom_e_ref_vs_she", "ph"]
        if "target_scale" in by_key:
            hidden.append("target_scale")
        _assert_rows_visible(panel, hidden, False)
    finally:
        panel.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def test_ecsa_original_scale_still_exposes_input_unit_for_volt_normalization(qt_app):
    panel = ParameterPanel(EcsaTool().parameter_schema())
    try:
        panel.set_values({"mode": "ECSA calculation", "electrode_setup": "Three-electrode",
                          "target_scale": "Original"})
        _assert_rows_visible(panel, ["potential_unit"], True)
        _assert_rows_visible(panel, ["reference", "custom_e_ref_vs_she", "ph"], False)
        panel.set_values({"potential_unit": "mV"})
        assert panel.values()["potential_unit"] == "mV"
    finally:
        panel.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
