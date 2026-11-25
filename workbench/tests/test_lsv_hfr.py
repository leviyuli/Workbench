"""Current-dependent HFR is shared by Tafel fitting and target breakdowns."""
import numpy as np
import pytest

from workbench.tools.lsv.hfr import HfrProfile, evaluate_hfr
from workbench.tools.lsv.logic_tafel import (
    get_overpotential_at_current,
    iterative_tafel_fit,
)


def test_profile_sorts_points_fits_linear_trend_and_extends_endpoint_tangents():
    profile = HfrProfile([("0.5", "0.12"), ("0.1", "0.1"), (1, 0.145),
                          (.3, .11), (.75, .1325)])

    assert profile.points == ((0.1, 0.1), (.3, .11), (.5, .12), (.75, .1325), (1., .145))
    current = np.array([0, 0.1, 0.3, 0.5, 0.75, 1, 2])
    expected = .095 + .05 * current
    np.testing.assert_allclose(profile(current), expected)
    np.testing.assert_allclose(profile(-current), expected)
    assert isinstance(profile(0.3), float)
    assert profile(-0.3) == pytest.approx(0.11)


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_cubic_smoothing_spline_requires_five_distinct_measurements(count):
    with pytest.raises(ValueError, match="at least five"):
        HfrProfile([(float(index), .12) for index in range(count)])


def test_constant_data_stays_constant_and_vector_shape_is_preserved():
    profile = HfrProfile([(current, .12) for current in (.1, .5, 1., 2., 3.)])
    np.testing.assert_allclose(profile(np.array([[-2, 0], [0.5, 4]])), .12)
    assert profile(.5) == pytest.approx(.12)


@pytest.mark.parametrize("points, message", [
    ([], "at least five"),
    (None, "numeric"),
    ([(0.1,)], "Each HFR point"),
    ([(0.1, 0.1), (0.1, 0.2)], "unique"),
    ([("0.1", 0.1), (0.1, 0.2)], "unique"),
    ([(-0.1, 0.1)], "nonnegative"),
    ([(0.1, -0.1)], "nonnegative"),
    ([(np.nan, 0.1)], "finite"),
    ([(0.1, np.inf)], "finite"),
    ([("unfinished", 0.1)], "numeric"),
])
def test_invalid_profile_points_have_actionable_errors(points, message):
    with pytest.raises(ValueError, match=message):
        HfrProfile(points)


@pytest.mark.parametrize("smoothing", [-1, np.nan, np.inf, "unfinished", None, True])
def test_invalid_smoothing_strength_is_rejected(smoothing):
    with pytest.raises(ValueError, match="smoothing strength"):
        HfrProfile([(current, .1) for current in range(5)], smoothing=smoothing)


def test_profile_extends_smoothly_and_exports_reproducible_fit_metadata():
    profile = HfrProfile([(current, .08 + .002 * current ** 2) for current in range(1, 7)])
    for endpoint, slope in zip(profile.current_range, profile.endpoint_slopes):
        step = 1e-5
        left = (profile(endpoint) - profile(endpoint - step)) / step
        right = (profile(endpoint + step) - profile(endpoint)) / step
        assert left == pytest.approx(slope, rel=1e-4, abs=1e-8)
        assert right == pytest.approx(slope, rel=1e-4, abs=1e-8)
    assert profile.fit_rmse >= 0
    assert profile.metadata()["HFR fit method"] == "Cubic smoothing spline"
    assert profile.metadata()["HFR smoothing strength"] == 1e-4
    assert profile.metadata()["HFR measured lower (A/cm²)"] == 1.


def test_negative_endpoint_extension_is_rejected_without_clipping():
    profile = HfrProfile([(current, .2 - .05 * current) for current in (0, 1, 2, 3, 3.5)])
    with pytest.raises(ValueError, match="covering the LSV current range"):
        profile(5.)
    with pytest.raises(ValueError, match="covering the LSV current range"):
        evaluate_hfr(profile, 5.)


def test_spline_undershoot_inside_measured_interval_is_rejected():
    with pytest.raises(ValueError, match="fitted HFR spline"):
        HfrProfile([(0, .01), (1, .01), (2, 10), (3, .01), (4, .01)], smoothing=0)


def test_evaluate_hfr_broadcasts_constant_and_evaluates_callable_at_magnitudes():
    current = np.array([[-0.1, 0.5], [-1, 2]])

    np.testing.assert_array_equal(evaluate_hfr(0.12, current), np.full((2, 2), 0.12))
    np.testing.assert_allclose(evaluate_hfr(lambda j: 0.1 + 0.02 * j, current),
                               [[0.102, 0.11], [0.12, 0.14]])
    assert isinstance(evaluate_hfr(0.12, 0.1), float)
    assert evaluate_hfr(lambda j: 0.1 + 0.02 * j, -0.5) == pytest.approx(0.11)


@pytest.mark.parametrize("hfr", [-0.1, np.nan, np.inf, "unfinished"])
def test_evaluate_hfr_rejects_invalid_constant_resistance(hfr):
    with pytest.raises(ValueError, match="HFR"):
        evaluate_hfr(hfr, np.array([0.1, 1]))


def test_evaluate_hfr_rejects_invalid_callable_results_and_shape():
    with pytest.raises(ValueError, match="finite and nonnegative"):
        evaluate_hfr(lambda j: -j, np.array([0.1, 1]))
    with pytest.raises(ValueError, match="shape"):
        evaluate_hfr(lambda j: np.ones(3), np.array([0.1, 1]))
    with pytest.raises(ValueError, match="scalar"):
        evaluate_hfr(np.array([0.1, 0.2]), np.array([0.1, 1]))


@pytest.mark.parametrize("include_rcl", [False, True])
def test_fit_recovers_kinetics_with_current_dependent_hfr(include_rcl):
    current = np.logspace(-3, 0, 350)
    profile = HfrProfile([(0.005, 0.08), (0.03, 0.15), (0.1, 0.22), (0.5, 0.25), (1, .27)])
    kinetic = 0.06 * np.log10(current / 1e-6)
    ohmic = current * profile(current)
    potential = 1.2291 + kinetic + ohmic

    result, warning = iterative_tafel_fit(
        potential, current, 1.2291, profile, 0.0, include_rcl=include_rcl,
    )

    assert warning is None
    fit, actual_ohmic, catalyst, accumulated = result
    assert fit["b_kin"] == pytest.approx(0.06)
    assert fit["i0"] == pytest.approx(1e-6)
    np.testing.assert_allclose(actual_ohmic, ohmic)
    np.testing.assert_allclose(catalyst, 0)
    np.testing.assert_allclose(accumulated, potential - 1.2291)


def test_target_evaluates_fitted_hfr_before_multiplying_by_current():
    profile = HfrProfile([(.1, .1), (.3, .11), (.5, .12), (.75, .1325), (1, .145)])
    current = np.array([0.1, 0.5, 1.0])
    potential = np.array([1.6, 1.8, 2.0])

    breakdown = get_overpotential_at_current(
        0.3, current, potential, 1.2291, profile, 0,
        {"b_kin": 0.06, "i0": 1e-6}, include_rcl=False,
    )

    assert breakdown["eta_ohm"] == pytest.approx(0.3 * 0.11)
    assert breakdown["V_exp"] == pytest.approx(1.7)
    # Interpolating measured jR instead would give 0.035 V.
    assert breakdown["eta_ohm"] != pytest.approx(0.035)
