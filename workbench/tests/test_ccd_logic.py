"""Operational CCD and observed limiting plateaus on acquired LSV sweeps."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import brentq

from workbench.tools.ccd.logic import CcdSettings, analyze_curve


def _curve(end=3.6):
    voltage = np.arange(1.0, end + .0001, .0005)
    current = np.maximum(4 * (voltage - 1.6), 0)
    high = voltage > 2.6
    current[high] = 5 - np.exp(-4 * (voltage[high] - 2.6))
    return voltage, current


def test_sustained_onset_recovers_analytic_threshold_and_observed_plateau():
    voltage, current = _curve()
    result = analyze_curve(voltage, current)

    onset = brentq(lambda v: v - 1.6 - (5 - np.exp(-4 * (v - 2.6))) / 4 - .02,
                   2.6, 3.0)
    expected_current = 5 - np.exp(-4 * (onset - 2.6))
    assert result.ccd_voltage == pytest.approx(onset, abs=.011)
    assert result.ccd_current == pytest.approx(expected_current, abs=.04)
    assert result.ccd_status == "Detected"
    assert result.limiting_current == pytest.approx(5, abs=.1)
    assert result.plateau_start > result.ccd_index
    assert result.plateau_end < len(result.binned_current)
    assert result.limiting_current == pytest.approx(np.median(
        result.binned_current[result.plateau_start:result.plateau_end + 1]))


def test_line_has_neither_ccd_nor_limiting_plateau():
    voltage = np.linspace(1, 3.6, 5201)
    result = analyze_curve(voltage, 4 * voltage - 3)

    assert result.ccd_current is None
    assert result.limiting_current is None
    assert "Not reached" in result.ccd_status
    assert result.baseline_slope == pytest.approx(4)


def test_single_spike_cannot_define_limiting_current_or_trigger_onset():
    voltage, current = _curve()
    current[np.argmin(abs(voltage - 2.50))] -= 2
    current[np.argmin(abs(voltage - 3.30))] = 100
    clean = analyze_curve(*_curve())
    result = analyze_curve(voltage, current)

    assert result.ccd_current == pytest.approx(clean.ccd_current, abs=.02)
    assert result.limiting_current == pytest.approx(clean.limiting_current, abs=.02)
    assert result.limiting_current < 6


def test_truncated_transport_region_does_not_extrapolate_limiting_current():
    result = analyze_curve(*_curve(end=3.0))

    assert result.ccd_current is not None
    assert result.limiting_current is None
    assert result.limiting_status == "Plateau not resolved"


def test_plateau_preceding_rollover_is_kept_and_decline_is_flagged():
    voltage, current = _curve(end=4.1)
    current[voltage > 3.7] -= 2 * (voltage[voltage > 3.7] - 3.7)
    result = analyze_curve(voltage, current)

    assert result.limiting_current == pytest.approx(5, abs=.1)
    assert result.limiting_current > np.median(current[-100:]) + .3
    assert "subsequent current decline" in result.limiting_status


def test_late_higher_current_prevents_a_low_shoulder_being_called_a_plateau():
    voltage, current = _curve(end=4.1)
    current[voltage > 3.4] += 3 * (voltage[voltage > 3.4] - 3.4)
    result = analyze_curve(voltage, current)

    assert result.ccd_current is not None
    assert result.limiting_current is None


def test_her_reverses_potential_direction_and_returns_current_magnitudes():
    voltage, current = _curve()
    reference = analyze_curve(voltage, current)
    result = analyze_curve(-voltage, -current)

    assert result.potential_sign == -1
    assert result.ccd_current == pytest.approx(reference.ccd_current)
    assert result.limiting_current == pytest.approx(reference.limiting_current)
    np.testing.assert_array_equal(result.potential, voltage)
    np.testing.assert_allclose(result.corrected_potential, reference.corrected_potential)
    np.testing.assert_allclose(result.excess_voltage, reference.excess_voltage)


def test_return_sweep_is_not_merged_with_forward_branch():
    voltage, current = _curve()
    reference = analyze_curve(voltage, current)
    result = analyze_curve(np.r_[voltage, voltage[-2::-1]],
                           np.r_[current, current[-2::-1] * 2])

    assert result.ccd_current == pytest.approx(reference.ccd_current)
    assert result.limiting_current == pytest.approx(reference.limiting_current)
    assert any("reversal" in warning for warning in result.warnings)


def test_nonfinite_gap_is_not_interpolated_into_an_artificial_onset():
    voltage, current = _curve()
    current[(voltage > 2.6) & (voltage < 2.8)] = np.nan
    result = analyze_curve(voltage, current)

    assert result.ccd_current is None
    assert result.limiting_current is None
    assert any("Nonfinite rows" in warning for warning in result.warnings)


def test_finite_voltage_gap_cannot_connect_an_earlier_baseline_to_a_later_knee():
    voltage, current = _curve(end=4.8)
    # All rows are finite, but absent voltage bins split the measured sweep.
    # The longest retained bin run is after the gap and excludes the baseline.
    measured = (voltage <= 2.5) | (voltage >= 2.8)

    with pytest.raises(ValueError, match="continuously sampled voltage interval"):
        analyze_curve(voltage[measured], current[measured])


def test_auto_baseline_works_on_a_reference_shifted_curve():
    voltage, current = _curve()
    result = analyze_curve(voltage - 1.5, current,
                           replace(CcdSettings(), baseline_mode="Auto"))

    assert result.baseline_slope == pytest.approx(4, rel=.01)
    assert result.ccd_current == pytest.approx(4.35, abs=.1)
    assert result.limiting_current == pytest.approx(5, abs=.1)


@pytest.mark.parametrize("reference_hfr", [0, .12, .30])
def test_constant_hfr_transforms_voltage_and_baseline_without_changing_ccd(reference_hfr):
    voltage, current = _curve()
    reference = analyze_curve(voltage, current, replace(CcdSettings(), reference_hfr=0))
    result = analyze_curve(voltage, current,
                           replace(CcdSettings(), reference_hfr=reference_hfr))

    np.testing.assert_allclose(result.corrected_potential, voltage - reference_hfr * current)
    np.testing.assert_allclose(result.binned_corrected_potential,
                               result.binned_potential - reference_hfr * result.smoothed_current)
    np.testing.assert_allclose(result.corrected_baseline_potential,
                               result.corrected_baseline_intercept
                               + result.corrected_baseline_slope * result.smoothed_current)
    np.testing.assert_allclose(result.excess_voltage,
                               result.binned_corrected_potential - result.corrected_baseline_potential)
    np.testing.assert_allclose(result.excess_voltage, reference.excess_voltage, atol=2e-14)
    np.testing.assert_allclose(result.baseline_current, reference.baseline_current, atol=2e-14)
    assert result.corrected_baseline_slope == pytest.approx(.25 - reference_hfr)
    assert result.baseline_slope == pytest.approx(reference.baseline_slope)
    assert result.baseline_r_squared == pytest.approx(reference.baseline_r_squared)
    assert result.ccd_index == reference.ccd_index
    assert result.ccd_current == pytest.approx(reference.ccd_current)
    assert result.limiting_current == pytest.approx(reference.limiting_current)
    if reference_hfr > .25:
        assert any("HFR exceeds" in warning for warning in result.warnings)


@pytest.mark.parametrize("reference_hfr", [0, .12, .30])
def test_a_purely_linear_ohmic_curve_has_no_ccd_after_any_constant_hfr(reference_hfr):
    current = np.linspace(.01, 7, 7001)
    voltage = 1.5 + .25 * current
    result = analyze_curve(voltage, current,
                           replace(CcdSettings(), reference_hfr=reference_hfr))

    assert result.ccd_current is None
    assert result.limiting_current is None
    assert result.corrected_baseline_slope == pytest.approx(.25 - reference_hfr)
    # Median-bin representatives need not fall at bin centers, especially at
    # partially occupied endpoints; their sub-bin offsets cannot trigger CCD.
    assert np.max(np.abs(result.excess_voltage)) < CcdSettings().bin_width_v / 2


def test_late_current_dependent_resistance_remains_in_the_corrected_excess():
    current = np.linspace(.01, 6, 12001)
    late_resistance = .04 * np.maximum(current - 3, 0)
    voltage = 1.5 + (.13 + .12) * current + current * late_resistance
    settings = replace(CcdSettings(), baseline_lower=1.9, baseline_upper=2.2,
                       reference_hfr=.12)
    result = analyze_curve(voltage, current, settings)

    # .04*j*(j-3) = .020 V has a known positive root.
    expected = (3 + np.sqrt(11)) / 2
    assert result.ccd_current == pytest.approx(expected, abs=.04)
    assert result.corrected_baseline_slope == pytest.approx(.13)
    assert result.excess_voltage[-1] > .6
    assert result.limiting_current is None


@pytest.mark.parametrize("changes, message", [
    ({"baseline_lower": 2.5, "baseline_upper": 2.0}, "Baseline lower"),
    ({"baseline_lower": np.nan}, "Baseline limits"),
    ({"threshold_v": 0}, "CCD threshold"),
    ({"bin_width_v": np.inf}, "Voltage bin"),
    ({"smoothing_window": 14}, "odd integer"),
    ({"plateau_slope_fraction": 1}, "smaller than one"),
    ({"reference_hfr": -.1}, "Reference HFR"),
    ({"reference_hfr": np.nan}, "Reference HFR"),
    ({"reference_hfr": np.inf}, "Reference HFR"),
])
def test_settings_reject_invalid_analysis_intervals_and_thresholds(changes, message):
    with pytest.raises(ValueError, match=message):
        analyze_curve(*_curve(), replace(CcdSettings(), **changes))


def test_baseline_outside_measurement_is_not_extrapolated():
    with pytest.raises(ValueError, match="inside the measured"):
        analyze_curve(*_curve(end=2.3))


def test_shape_and_empty_data_errors_are_actionable():
    with pytest.raises(ValueError, match="matching one-dimensional"):
        analyze_curve([1, 2, 3], [1, 2])
    with pytest.raises(ValueError, match="finite"):
        analyze_curve([np.nan] * 20, [np.nan] * 20)
