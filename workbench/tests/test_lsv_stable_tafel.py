"""Stable Tafel selection keeps physical inputs and acquisition barriers intact."""
import numpy as np
import pytest

from workbench.tools.lsv.analysis import DecouplingSettings, decouple_curve
from workbench.tools.lsv.logic_tafel import calculate_RCL_overpotential, fit_tafel_window
from workbench.tools.lsv.logic_tafel_stable import (
    TafelFitError, fit_stable_tafel, iterative_stable_tafel_fit,
)


def test_exact_tafel_recovers_parameters_from_a_broad_interval():
    current = np.geomspace(.01, 1., 1001)
    eta = .45 + .12 * np.log10(current)
    fit, diagnostics = fit_stable_tafel(eta, current, .01, 1., minimum_decades=1.)
    assert diagnostics["status"] == "Resolved"
    assert fit["b_kin"] == pytest.approx(.12, abs=1e-12)
    assert fit["intercept"] == pytest.approx(.45, abs=1e-12)
    details = fit["candidate_details"]
    assert details["log_current_span"] > 1.9
    # Prefix-sum residual diagnostics may retain a few nanovolts of roundoff.
    assert details["rmse"] < 1e-7
    selected = np.asarray(details["indices"], dtype=int)
    assert len(selected) > 950
    assert np.all(np.diff(selected) == 1)
    assert details["current_range"] == pytest.approx((current[selected].min(), current[selected].max()))
    assert details["source_start"] == selected.min()
    assert details["source_end"] == selected.max()


def test_seeded_noise_does_not_break_adjacency_or_change_the_known_slope():
    random = np.random.default_rng(20261106)
    current = np.geomspace(.01, 1., 1001)
    eta = .45 + .12 * np.log10(current) + random.normal(0, .0006, len(current))
    assert np.count_nonzero(np.diff(eta) < 0) > 100
    fit, diagnostics = fit_stable_tafel(eta, current, .01, 1., minimum_decades=1.)
    assert diagnostics["status"] == "Resolved"
    assert diagnostics["segments"] == 1
    assert fit["b_kin"] == pytest.approx(.12, abs=.002)
    assert fit["candidate_details"]["rmse"] < .001


@pytest.mark.parametrize("barrier", ["nan", "source_gap", "negative_current"])
def test_acquisition_barriers_prevent_joining_two_short_regions(barrier):
    current = np.geomspace(.01, .1, 501)
    eta = .5 + .2 * np.log10(current)
    source = np.arange(len(current))
    if barrier == "nan":
        eta[250] = np.nan
    elif barrier == "negative_current":
        current[250] *= -1
    else:
        source[250:] += 20
    fit, diagnostics = fit_stable_tafel(eta, current, .01, .1,
                                        minimum_decades=.8, source_indices=source)
    assert fit is None
    assert diagnostics["status"] == "Unresolved"
    assert diagnostics["segments"] == 2
    assert diagnostics["qualified_windows"] > 0
    assert diagnostics["reason"]


def test_gradual_slope_drift_cannot_qualify_as_a_wide_tafel_region():
    current = np.geomspace(.01, 1., 1001)
    x = np.log10(current) + 2
    eta = .1 + .04 * x + .12 * x ** 2
    fit, diagnostics = fit_stable_tafel(eta, current, .01, 1., minimum_decades=1.)
    assert fit is None
    assert diagnostics["qualified_windows"] > 3


@pytest.mark.parametrize("kind", ["flat", "negative", "narrow", "noisy"])
def test_unusable_tafel_data_remain_unresolved(kind):
    current = np.geomspace(.01, 1., 501)
    x = np.log10(current)
    if kind == "flat":
        eta = np.full(len(current), .3)
    elif kind == "negative":
        eta = .3 - .1 * x
    elif kind == "narrow":
        eta = .3 + .01 * x
    else:
        eta = .3 + .1 * x + np.random.default_rng(1).normal(0, .03, len(current))
    fit, diagnostics = fit_stable_tafel(eta, current, .01, 1., minimum_decades=1.)
    assert fit is None
    assert diagnostics["status"] == "Unresolved"
    assert diagnostics["reason"]


def test_whole_back_scan_reversal_keeps_the_fit_and_original_source_rows():
    current = np.geomspace(.01, 1., 501)
    eta = .3 + .12 * np.log10(current)
    forward, _ = fit_stable_tafel(eta, current, .01, 1.)
    back, _ = fit_stable_tafel(eta[::-1], current[::-1], .01, 1.,
                              source_indices=np.arange(len(current))[::-1])
    assert back["b_kin"] == pytest.approx(forward["b_kin"])
    assert back["intercept"] == pytest.approx(forward["intercept"])
    assert back["candidate_details"]["source_start"] == forward["candidate_details"]["source_start"]
    assert back["candidate_details"]["source_end"] == forward["candidate_details"]["source_end"]
    assert np.all(np.diff(back["candidate_details"]["indices"]) > 0)


def test_invalid_settings_fail_explicitly():
    with pytest.raises(ValueError, match="minimum_decades"):
        fit_stable_tafel([.1, .2], [.1, .2], minimum_decades=0)
    with pytest.raises(ValueError, match="Source"):
        fit_stable_tafel([.1, .2], [.1, .2], source_indices=[1])


def test_both_selectors_agree_when_the_data_are_exactly_tafel():
    current = np.geomspace(.01, 1., 1001)
    eta = .45 + .12 * np.log10(current)
    old, error = fit_tafel_window(eta, current, .01, 1.)
    new, _ = fit_stable_tafel(eta, current, .01, 1.)
    assert error is None
    assert new["b_kin"] == pytest.approx(old["b_kin"], abs=1e-12)
    assert new["intercept"] == pytest.approx(old["intercept"], abs=1e-12)


@pytest.mark.parametrize("mode", ["OER", "HER", "Cell positive", "Cell negative"])
@pytest.mark.parametrize("profile,include_rcl", [(False, False), (True, True)])
@pytest.mark.parametrize("back", [False, True])
def test_engine_recovers_known_components_for_polarity_back_and_hfr_rcl(mode, profile, include_rcl, back):
    current = np.unique(np.r_[np.geomspace(.001, 1., 1501), np.linspace(.8, 5., 1501)])
    kinetic = .12 * np.log10(current / 2e-6)
    resistance = .1 + .02 * current if profile else .12
    catalyst = calculate_RCL_overpotential(current, .15, .12) if include_rcl else np.zeros_like(current)
    residual = .08 * np.maximum(current - .8, 0)
    cell = mode.startswith("Cell")
    sign_current = -1 if mode in {"HER", "Cell positive"} else 1
    sign_voltage = -1 if mode in {"HER", "Cell negative"} else 1
    voltage = sign_voltage * ((0. if mode == "HER" else 1.2291)
                              + kinetic + current * resistance + catalyst + residual)
    density = sign_current * current
    source = np.arange(len(current)) + 100
    params = dict(electrode_setup="Two-electrode" if cell else "Three-electrode",
                  reference="RHE", reaction="HER" if sign_current < 0 else "OER",
                  temperature=298.15, ph=0., hfr=.12,
                  hfr_mode="HFR vs |j|" if profile else "Constant HFR",
                  include_rcl=include_rcl, rcl=.15, i_lower=.02, i_upper=.8)
    extra = {"hfr_points": [[.01, .1002], [.1, .102], [1., .12], [3., .16], [5., .2]]}
    direction = slice(None, None, -1) if back else slice(None)
    voltage, density, source = voltage[direction], density[direction], source[direction]
    result = decouple_curve(voltage, density, DecouplingSettings.from_params(params, extra),
                            source_indices=source, tafel_method="contiguous-stable")
    np.testing.assert_array_equal(result.measured_potential, voltage)
    np.testing.assert_array_equal(result.measured_current, density)
    np.testing.assert_array_equal(result.source_indices, source)
    assert result.fit["b_kin"] == pytest.approx(.12, rel=.002)
    assert result.fit["i0"] == pytest.approx(2e-6, rel=.015)
    for actual, known in [(result.eta_kin, kinetic[direction]),
                          (result.eta_ohm, (current * resistance)[direction]),
                          (result.eta_rcl, catalyst[direction]), (result.residual, residual[direction])]:
        np.testing.assert_allclose(actual, known, atol=.002, rtol=.002)
    np.testing.assert_allclose(result.eta_total,
        result.eta_kin + result.eta_ohm + result.eta_rcl + result.residual, atol=1e-14)
    details = result.fit["candidate_details"]
    chosen = np.asarray(details["indices"], dtype=int)
    assert details["source_start"] == source[chosen].min()
    assert details["source_end"] == source[chosen].max()


def test_stable_engine_does_not_fall_back_to_legacy_across_opposite_current_barrier():
    current = np.geomspace(.02, .16, 401)
    voltage = 1.2291 + .45 + .2 * np.log10(current) + .12 * current
    current[200] *= -1
    settings = DecouplingSettings.from_params(dict(electrode_setup="Two-electrode",
        temperature=298.15, hfr=.12, include_rcl=False, i_lower=.02, i_upper=.16))
    legacy = decouple_curve(voltage, current, settings)
    assert legacy.fit["b_kin"] == pytest.approx(.2)
    with pytest.raises(TafelFitError) as caught:
        decouple_curve(voltage, current, settings, tafel_method="contiguous-stable")
    assert caught.value.diagnostics["status"] == "Unresolved"
    assert caught.value.diagnostics["segments"] == 2


def test_stable_engine_preserves_signed_residual_and_missing_acquisition_rows():
    current = np.unique(np.r_[np.geomspace(.001, .8, 1801), np.linspace(.8, 5., 1501)])
    residual = -.04 * np.maximum(current - .8, 0)
    voltage = 1.2291 + .45 + .12 * np.log10(current) + .12 * current + residual
    gap = (current > 1.5) & (current < 1.7)
    voltage[gap] = np.nan
    settings = DecouplingSettings.from_params(dict(electrode_setup="Two-electrode",
        temperature=298.15, hfr=.12, include_rcl=False, i_lower=.02, i_upper=.8))
    result = decouple_curve(voltage, current, settings, tafel_method="contiguous-stable")
    assert np.isnan(result.residual[gap]).all()
    np.testing.assert_allclose(result.residual[~gap], residual[~gap], atol=1e-12)
    assert result.breakdown_at(3.)["residual"] < 0
    with pytest.raises(ValueError, match="acquisition gap"):
        result.breakdown_at(1.6)


def test_default_engine_still_matches_explicit_nested_selector():
    current = np.geomspace(.005, 2., 1001)
    x = np.log10(current / .005)
    voltage = 1.2291 + .2 + .06 * x + .035 * x ** 2 + .12 * current
    settings = DecouplingSettings.from_params(dict(electrode_setup="Two-electrode",
        temperature=298.15, hfr=.12, include_rcl=False, i_lower=.005, i_upper=.8))
    implicit = decouple_curve(voltage, current, settings)
    explicit = decouple_curve(voltage, current, settings, tafel_method="nested-window")
    assert implicit.fit["b_kin"] == explicit.fit["b_kin"]
    assert implicit.fit["i0"] == explicit.fit["i0"]
    np.testing.assert_array_equal(implicit.eta_kin, explicit.eta_kin)


def test_rcl_iteration_recovers_known_kinetics_after_local_slope_initialization():
    current = np.geomspace(.001, 2., 2001)
    catalyst = calculate_RCL_overpotential(current, .5, .06)
    total = .45 + .06 * np.log10(current) + .12 * current + catalyst
    settings = DecouplingSettings.from_params(dict(electrode_setup="Two-electrode",
        temperature=298.15, hfr=.12, include_rcl=True, rcl=.5, i_lower=.02, i_upper=.8))
    curve = decouple_curve(1.2291 + total, current, settings, tafel_method="contiguous-stable")
    assert curve.fit["b_kin"] == pytest.approx(.06, abs=.0001)
    assert not curve.warnings
    diagnostics = curve.fit["diagnostics"]
    assert diagnostics["rcl_seed_used"]
    assert diagnostics["iterations"] <= 10
    validation = diagnostics["rcl_final_interval_validation"]
    assert validation["model_rmse"] <= .005
    assert validation["refit_rmse"] <= .005
    np.testing.assert_allclose(curve.eta_total,
        curve.eta_kin + curve.eta_ohm + curve.eta_rcl + curve.residual, atol=1e-14)


def test_unvalidated_rcl_fit_is_not_reported_at_the_iteration_limit():
    current = np.geomspace(.001, 2., 2001)
    total = (.45 + .06 * np.log10(current) + .12 * current
             + calculate_RCL_overpotential(current, .5, .06))
    with pytest.raises(TafelFitError, match="final RCL-corrected"):
        iterative_stable_tafel_fit(total, current, 0., .12, .5,
                                   .02, .8, max_iter=1, include_rcl=True)
