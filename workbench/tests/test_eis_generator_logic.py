"""Physical identities and numerical limits for the CL impedance generator."""

import numpy as np
import pytest

from workbench.tools.eis_generator.logic import (
    DEFAULT_PARAMETERS,
    generate_cl_eis,
    two_rail_impedance,
    validate_parameters,
)


def test_default_units_markers_and_descending_frequency_grid():
    result = generate_cl_eis({})
    r = result["resistances"]
    assert result["parameters"] == DEFAULT_PARAMETERS
    assert r == pytest.approx({
        "r_i": 0.25,
        "r_e": 0.05,
        "hfr_cl": 1.0 / 24.0,
        "hfr_total": 0.05 + 1.0 / 24.0,
        "rcl_app": 0.175,
        "width": 0.175 / 3.0,
        "lf_intercept": 0.15,
    })
    assert r["hfr_cl"] + r["rcl_app"] / 3 == pytest.approx(
        (r["r_i"] + r["r_e"]) / 3
    )
    assert len(result["frequency"]) == 400
    assert result["frequency"][0] == 1e6
    assert result["frequency"][-1] == 0.1
    assert np.all(np.diff(result["frequency"]) < 0)
    assert np.iscomplexobj(result["z_cl"])
    assert np.all(result["z_cl"].real > 0)
    assert np.all(result["z_cl"].imag < 0)


def test_exchanging_ionic_and_electronic_rails_preserves_the_spectrum():
    frequencies = np.geomspace(1e-6, 1e9, 500)
    np.testing.assert_allclose(
        two_rail_impedance(frequencies, 0.23, 0.017, 0.04),
        two_rail_impedance(frequencies, 0.017, 0.23, 0.04),
        rtol=2e-14,
    )


@pytest.mark.parametrize("rails", [(0.4, 0.0), (0.0, 0.4)])
def test_single_rail_reduces_to_classic_transmission_line(rails):
    frequencies = np.geomspace(0.01, 1e5, 200)
    resistance, capacitance = 0.4, 0.02
    lam = np.sqrt(1j * 2 * np.pi * frequencies * capacitance * resistance)
    expected = resistance / lam / np.tanh(lam)
    np.testing.assert_allclose(
        two_rail_impedance(frequencies, *rails, capacitance),
        expected, rtol=1e-12, atol=1e-12,
    )


def test_equal_rails_are_exactly_a_series_term_plus_a_shorter_single_rail():
    frequencies = np.geomspace(1e-9, 1e12, 400)
    rail, capacitance = 0.2, 0.02
    expected = rail / 2 + two_rail_impedance(
        frequencies, rail / 2, 0, capacitance
    )
    actual = two_rail_impedance(frequencies, rail, rail, capacitance)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_zero_rails_are_an_ideal_capacitor_and_scalar_input_is_supported():
    frequencies = np.geomspace(1e-10, 1e10, 40)
    np.testing.assert_allclose(
        two_rail_impedance(frequencies, 0, 0, 0.03),
        1 / (1j * 2 * np.pi * frequencies * 0.03), rtol=1e-13,
    )
    result = two_rail_impedance(10.0, 0.1, 0.2, 0.03)
    assert result.shape == ()
    assert result.item().imag < 0


def test_capacitance_only_translates_the_frequency_scale():
    frequencies = np.geomspace(0.001, 1e9, 250)
    np.testing.assert_allclose(
        two_rail_impedance(frequencies, 0.13, 0.07, 0.2),
        two_rail_impedance(10 * frequencies, 0.13, 0.07, 0.02),
        rtol=3e-14, atol=1e-13,
    )
    assert generate_cl_eis({"capacitance_mf_cm2": 200})["resistances"] == (
        generate_cl_eis({})["resistances"]
    )


def test_external_resistance_translates_real_impedance_and_intercepts_only():
    first = generate_cl_eis({"external_resistance": 0})
    second = generate_cl_eis({"external_resistance": 0.7})
    np.testing.assert_array_equal(first["frequency"], second["frequency"])
    np.testing.assert_array_equal(first["z_cl"], second["z_cl"])
    np.testing.assert_allclose(second["z_total"] - first["z_total"], 0.7, atol=1e-14)
    for key in ("r_i", "r_e", "hfr_cl", "rcl_app", "width"):
        assert first["resistances"][key] == second["resistances"][key]
    for key in ("hfr_total", "lf_intercept"):
        assert second["resistances"][key] - first["resistances"][key] == pytest.approx(0.7)


def test_extreme_frequencies_remain_finite_and_approach_exact_asymptotes():
    frequencies = np.geomspace(1e-200, 1e200, 801)
    z = two_rail_impedance(frequencies, 0.25, 0.05, 0.02)
    assert np.all(np.isfinite(z))
    assert np.all(z.real > 0)
    assert np.all(z.imag < 0)
    assert z[0].real == pytest.approx(0.3 / 3, rel=1e-14)
    assert z[-1].real == pytest.approx(1 / 24, rel=1e-14)
    assert z[0].imag == pytest.approx(-1 / (2 * np.pi * frequencies[0] * 0.02), rel=1e-13)


def test_series_and_exponential_evaluation_join_continuously():
    # |Lambda|=0.01 is the implementation's series boundary.
    transition = 1e-4 / (2 * np.pi * 0.02 * 0.3)
    frequencies = transition * np.array([1 - 1e-8, 1 + 1e-8])
    actual = two_rail_impedance(frequencies, 0.25, 0.05, 0.02)
    lam = np.sqrt(1j * 2 * np.pi * frequencies * 0.02 * 0.3)
    hfr, d = 1 / 24, 0.3 - 2 / 24
    expected = hfr + (d / np.tanh(lam) + 2 * hfr / np.sinh(lam)) / lam
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_parameter_normalization_ignores_ui_metadata_without_mutating_inputs():
    supplied = {"thickness_um": "7", "n_points": 25.0, "ui_title": "A spectrum"}
    saved = dict(supplied)
    normalized = validate_parameters(supplied)
    assert normalized["thickness_um"] == 7.0
    assert normalized["n_points"] == 25
    assert type(normalized["n_points"]) is int
    assert set(normalized) == set(DEFAULT_PARAMETERS)
    assert supplied == saved
    assert DEFAULT_PARAMETERS["thickness_um"] == 5.0


@pytest.mark.parametrize("key", [
    "electronic_conductivity", "ionic_conductivity", "thickness_um",
    "capacitance_mf_cm2", "f_min", "f_max",
])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, None])
def test_invalid_positive_parameters_are_rejected(key, value):
    with pytest.raises(ValueError):
        validate_parameters({key: value})


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, None])
def test_invalid_external_resistance_is_rejected(value):
    with pytest.raises(ValueError):
        validate_parameters({"external_resistance": value})


@pytest.mark.parametrize("value", [1, 20001, 2.5, float("nan"), True, "invalid"])
def test_invalid_frequency_point_count_is_rejected(value):
    with pytest.raises(ValueError):
        validate_parameters({"n_points": value})


@pytest.mark.parametrize("parameters", [
    {"f_min": 10, "f_max": 10},
    {"f_min": 11, "f_max": 10},
    {"thickness_um": 1e308, "ionic_conductivity": 1e-300},
    {"thickness_um": 5e-324},
    {"capacitance_mf_cm2": 5e-324},
    {"f_min": 5e-324, "f_max": 1e-300},
])
def test_invalid_or_unrepresentable_derived_parameters_are_rejected(parameters):
    with pytest.raises(ValueError):
        validate_parameters(parameters)


@pytest.mark.parametrize("frequency,ri,re,capacitance", [
    ([0, 1], 0.1, 0.2, 0.02),
    ([1, float("inf")], 0.1, 0.2, 0.02),
    ([], 0.1, 0.2, 0.02),
    ([[1, 2]], 0.1, 0.2, 0.02),
    ([True], 0.1, 0.2, 0.02),
    ([1 + 1j], 0.1, 0.2, 0.02),
    ([1], -0.1, 0.2, 0.02),
    ([1], 0.1, float("inf"), 0.02),
    ([1], 0.1, 0.2, 0),
    ([1], 1e308, 1e308, 0.02),
])
def test_helper_rejects_invalid_arguments(frequency, ri, re, capacitance):
    with pytest.raises(ValueError):
        two_rail_impedance(frequency, ri, re, capacitance)


def test_non_mapping_parameters_are_rejected():
    with pytest.raises(ValueError, match="mapping"):
        validate_parameters(None)


def test_unresolvable_frequency_grid_is_rejected():
    with pytest.raises(ValueError, match="distinct descending"):
        generate_cl_eis({"f_min": 1.0, "f_max": np.nextafter(1.0, 2.0)})
