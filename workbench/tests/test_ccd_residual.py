"""Residual CCD uses stable trends while preserving scan adjacency and signs."""
from types import SimpleNamespace

import numpy as np
import pytest

from workbench.tools.ccd.residual import ResidualCcdSettings, analyze_residual_ccd


class _KnownCurve:
    """An exact decomposition with independently specified residual losses."""

    def __init__(self, current, residual, *, reverse=False, sign=1.0, gap=None,
                 tafel_upper=.4, kinetic=.06):
        self.sign = sign
        self.e_rev = sign * 1.2291
        self.settings = SimpleNamespace(i_upper=tafel_upper)
        self.kinetic = kinetic
        self.hfr = .12
        density = np.asarray(current, dtype=float)
        voltage = self.e_rev + sign * (
            kinetic * np.log10(density / 1e-6) + self.hfr * density + residual)
        self.measured_potential = voltage.copy()
        self.potential = voltage.copy()
        self.current = sign * density
        self.current_magnitude = density.copy()
        self.drive = sign * voltage
        self.source_indices = np.arange(len(density))
        if gap is not None:
            for array in (self.measured_potential, self.potential, self.current,
                          self.current_magnitude, self.drive):
                array[gap] = np.nan
        if reverse:
            for key in ("measured_potential", "potential", "current", "current_magnitude",
                        "drive", "source_indices"):
                setattr(self, key, getattr(self, key)[::-1])

    def evaluate_at(self, potential, current_magnitude):
        with np.errstate(invalid="ignore", divide="ignore"):
            kinetic = self.kinetic * np.log10(current_magnitude / 1e-6)
        ohmic = self.hfr * current_magnitude
        return {"eta_kin": kinetic, "eta_ohm": ohmic,
                "eta_rcl": np.zeros_like(kinetic),
                "residual": self.sign * (potential - self.e_rev) - kinetic - ohmic}


def _onset_curve(*, reverse=False, sign=1.0, gap=None):
    current = np.linspace(.01, 6, 20001)
    residual = .08 * current + .06 * np.maximum(current - 3, 0) ** 2
    return _KnownCurve(current, residual, reverse=reverse, sign=sign, gap=gap)


def test_manual_residual_baseline_recovers_known_sustained_onset():
    result = analyze_residual_ccd(_onset_curve(), "Manual", .8, 2.5)
    expected = 3 + np.sqrt(.050 / .06)
    assert result.baseline_slope == pytest.approx(.08, abs=.002)
    assert result.baseline_intercept == pytest.approx(0, abs=.004)
    assert result.ccd_current == pytest.approx(expected, abs=.04)
    assert result.ccd_status == "Detected"
    assert result.excess[result.ccd_index] >= .050
    assert result.limiting_current is None
    assert result.baseline_end < result.ccd_index
    np.testing.assert_allclose(result.excess,
                               result.residual - result.baseline, atol=1e-14)


def test_fifty_mv_threshold_ignores_sustained_lower_fluctuations_but_detects_later_rise():
    current = np.linspace(.01, 8, 24001)
    fluctuation = (.035 * np.clip((current - 3) / .3, 0, 1)
                   + .004 * np.sin(6 * np.maximum(current - 3.3, 0)))
    fluctuating = _KnownCurve(current, .08 * current + fluctuation)
    default = analyze_residual_ccd(fluctuating, "Manual", .8, 2.5)
    earlier = analyze_residual_ccd(
        fluctuating, "Manual", .8, 2.5,
        settings=ResidualCcdSettings(threshold_v=.020),
    )

    assert earlier.ccd_status == "Detected"
    assert earlier.ccd_current < 3.5
    assert default.ccd_status == "Not reached within measured range"
    assert default.ccd_current is None
    assert np.max(default.excess) < .050

    rise = .07 * np.maximum(current - 6, 0) ** 2
    result = analyze_residual_ccd(
        _KnownCurve(current, .08 * current + fluctuation + rise),
        "Manual", .8, 2.5,
    )
    assert result.ccd_status == "Detected"
    assert result.ccd_current > 6
    assert result.settings.persistence_v == .10
    count = int(np.ceil(result.settings.persistence_v / result.settings.bin_width_v))
    assert np.all(result.excess[result.ccd_index:result.ccd_index + count] >= .050)


def test_auto_uses_a_merged_stable_region_after_the_configured_tafel_range():
    curve = _onset_curve()
    curve.settings.i_upper = .8
    result = analyze_residual_ccd(curve)
    assert result.baseline_status == "Resolved"
    assert result.baseline_lower_current > .8
    assert result.drive[result.baseline_end] - result.drive[result.baseline_start] > .35
    assert result.baseline_rmse <= .005
    # Stable describes the fitted interval, not an independently established
    # transport-free region; Auto need not recover an assumed physical onset.
    assert np.all(np.diff(np.flatnonzero(result.baseline_mask)) == 1)


@pytest.mark.parametrize("offset,slope", [(0, 0), (-.03, 0), (0, .12)])
def test_large_linear_flat_or_negative_residual_does_not_itself_trigger_ccd(offset, slope):
    current = np.linspace(.01, 8, 24001)
    curve = _KnownCurve(current, offset + slope * current)
    result = analyze_residual_ccd(curve)
    assert result.baseline_status == "Resolved"
    assert result.baseline_slope == pytest.approx(slope, abs=.002)
    assert result.ccd_current is None
    assert result.limiting_current is None
    if offset < 0:
        assert np.median(result.residual[result.baseline_mask]) < -.02


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_whole_branch_reversal_retains_the_selected_hysteresis_and_source_mapping(sign):
    reference = analyze_residual_ccd(_onset_curve(sign=sign), "Manual", .8, 2.5)
    back = analyze_residual_ccd(_onset_curve(reverse=True, sign=sign), "Manual", .8, 2.5)
    assert back.reversed_for_analysis
    assert back.ccd_current == pytest.approx(reference.ccd_current)
    np.testing.assert_allclose(back.drive, reference.drive)
    np.testing.assert_allclose(back.residual, reference.residual)
    np.testing.assert_array_equal(back.source_start, reference.source_start)
    np.testing.assert_array_equal(back.source_end, reference.source_end)
    assert np.all(np.diff(back.drive) > 0)


def test_nonmonotonic_residual_is_neither_sorted_nor_used_to_trim_the_scan():
    current = np.linspace(.01, 7, 20001)
    residual = .05 * current + .012 * np.sin(14 * current)
    curve = _KnownCurve(current, residual)
    result = analyze_residual_ccd(curve, "Manual", 1, 2)
    assert np.any(np.diff(result.residual) < 0)
    assert result.drive[-1] > curve.drive[-1] - .02
    assert result.source_end[-1] == len(current) - 1
    assert np.all(np.diff(result.source_start) > 0)


def test_a_gap_cannot_connect_an_earlier_baseline_to_a_later_onset():
    curve = _onset_curve(gap=slice(8000, 12000))
    result = analyze_residual_ccd(curve, "Manual", .8, 2.0)
    assert len(np.unique(result.segment_id)) == 2
    assert result.baseline_status == "Resolved"
    assert result.ccd_current is None
    assert result.limiting_current is None
    other = result.segment_id != result.baseline_segment
    assert np.isnan(result.baseline[other]).all()
    assert np.isnan(result.excess[other]).all()
    assert result.source_end[result.segment_id == 0][-1] < 8000
    assert result.source_start[result.segment_id == 1][0] >= 12000


def test_manual_range_spanning_a_gap_is_unresolved_without_discarding_decomposition():
    curve = _onset_curve(gap=slice(8000, 12000))
    result = analyze_residual_ccd(curve, "Manual", 2.0, 4.0)
    assert result.baseline_status == "Unresolved"
    assert result.ccd_status == "Baseline unresolved"
    assert len(result.current) > 50
    assert any("continuously sampled" in warning for warning in result.warnings)


def test_missing_voltage_bins_are_a_barrier_even_when_all_source_rows_are_finite():
    curve = _onset_curve()
    keep = np.r_[np.arange(8000), np.arange(12000, len(curve.current))]
    for name in ("measured_potential", "potential", "current", "current_magnitude",
                 "drive", "source_indices"):
        setattr(curve, name, getattr(curve, name)[keep])
    result = analyze_residual_ccd(curve, "Manual", .8, 2.0)
    assert len(np.unique(result.segment_id)) == 2
    assert result.ccd_current is None
    assert result.source_start[result.segment_id == 1][0] >= 12000


def test_auto_failure_retains_processed_arrays_and_recommends_manual_range():
    current = np.linspace(.01, .3, 1001)
    result = analyze_residual_ccd(_KnownCurve(current, .1 * current, tafel_upper=.4))
    assert result.baseline_status == "Unresolved"
    assert len(result.current) > 10
    assert any("manual current range" in warning for warning in result.warnings)


def test_observed_plateau_is_resolved_instead_of_returning_the_highest_observation():
    drive = np.arange(1.3, 4.1001, .0005)
    current = np.maximum(4 * (drive - 1.6), .0001)
    high = drive > 2.6
    current[high] = 5 - np.exp(-4 * (drive[high] - 2.6))
    # Specify this exact E(j) path without needing a monotonic inverse j->E.
    curve = _KnownCurve(current, np.zeros_like(current), kinetic=0)
    curve.measured_potential = drive.copy()
    curve.potential = drive.copy()
    curve.drive = drive.copy()
    result = analyze_residual_ccd(curve, "Manual", 1.2, 3.3)
    assert result.ccd_current is not None
    assert result.limiting_current == pytest.approx(5, abs=.05)
    assert result.plateau_start > result.ccd_index
    assert result.limiting_current == pytest.approx(np.median(
        result.median_current[result.plateau_start:result.plateau_end + 1]))


def test_plateau_detection_supports_a_flat_residual_baseline_without_inverting_its_slope():
    drive = np.arange(1.25, 2.8001, .0005)
    current = np.minimum((drive - 1.2291) / .12, 5.0)
    curve = _KnownCurve(current, np.zeros_like(current), kinetic=0)
    curve.measured_potential = drive.copy()
    curve.potential = drive.copy()
    curve.drive = drive.copy()
    result = analyze_residual_ccd(curve, "Manual", 1.2, 3.3)
    assert result.baseline_slope == pytest.approx(0, abs=.002)
    assert result.measured_baseline_slope == pytest.approx(1 / .12, rel=.01)
    assert result.ccd_current == pytest.approx(5, abs=.02)
    assert result.limiting_current == pytest.approx(5, abs=.01)


@pytest.mark.parametrize("mode,lower,upper", [("Other", 1, 2), ("Manual", 2, 1),
                                             ("Manual", np.nan, 2)])
def test_invalid_range_controls_raise_clear_errors(mode, lower, upper):
    with pytest.raises(ValueError, match="mode|limits"):
        analyze_residual_ccd(_onset_curve(), mode, lower, upper)
