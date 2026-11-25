"""One numerical decomposition preserves polarity, signed losses and acquisition gaps."""
import numpy as np
import pandas as pd
import pytest

from workbench.tools.lsv.analysis import DecouplingSettings, decouple_curve
from workbench.tools.lsv.logic_tafel import calculate_RCL_overpotential
from workbench.tools.lsv.tool import LsvTool


def _source(mode="OER", *, include_rcl=False, profile=False):
    current = np.unique(np.r_[np.logspace(-3, 0, 500), .01, .1, .3, .5])
    kinetic = .06 * np.log10(current / 2e-6)
    ohmic_resistance = .1 + .04 * current if profile else .15
    ohmic = current * ohmic_resistance
    catalyst = calculate_RCL_overpotential(current, .2, .06) if include_rcl else np.zeros_like(current)
    residual = .1 * np.maximum(current - .3, 0)
    is_cell = mode.startswith("Cell")
    current_sign = -1 if mode in {"HER", "Cell positive"} else 1
    voltage_sign = -1 if mode in {"HER", "Cell negative"} else 1
    reversible = 0.0 if mode == "HER" else 1.2291
    voltage = voltage_sign * (reversible + kinetic + ohmic + catalyst + residual)
    params = {
        "electrode_setup": "Two-electrode" if is_cell else "Three-electrode",
        "reaction": "HER" if current_sign < 0 else "OER", "reference": "RHE",
        "temperature": 298.15, "ph": 0, "hfr": .15,
        "hfr_mode": "HFR vs |j|" if profile else "Constant HFR",
        "include_rcl": include_rcl, "rcl": .2, "i_lower": .005,
        "i_upper": .1, "target_i": .5,
    }
    extra = {"hfr_points": [[.8, .132], [.05, .102], [.2, .108], [.4, .116], [.6, .124]]}
    return voltage, current_sign * current, params, extra, (kinetic, ohmic, catalyst, residual)


@pytest.mark.parametrize("mode", ["OER", "HER", "Cell positive", "Cell negative"])
@pytest.mark.parametrize("include_rcl", [False, True])
@pytest.mark.parametrize("profile", [False, True])
def test_engine_recovers_known_components_without_reordering(mode, include_rcl, profile):
    potential, current, params, extra, expected = _source(mode, include_rcl=include_rcl, profile=profile)
    original_potential, original_current = potential.copy(), current.copy()
    settings = DecouplingSettings.from_params(params, extra)
    result = decouple_curve(potential, current, settings)

    np.testing.assert_array_equal(potential, original_potential)
    np.testing.assert_array_equal(current, original_current)
    np.testing.assert_array_equal(result.measured_potential, potential)
    np.testing.assert_array_equal(result.current, current)
    np.testing.assert_array_equal(result.source_indices, np.arange(len(current)))
    assert result.fit["b_kin"] == pytest.approx(.06, rel=2e-4)
    assert result.fit["i0"] == pytest.approx(2e-6, rel=2e-3, abs=0)
    for actual, known in zip((result.eta_kin, result.eta_ohm, result.eta_rcl, result.residual), expected):
        np.testing.assert_allclose(actual, known, atol=2e-7, rtol=2e-5)
    np.testing.assert_allclose(result.drive, result.sign * result.potential)
    np.testing.assert_allclose(result.eta_total,
                               result.eta_kin + result.eta_ohm + result.eta_rcl + result.residual)
    target = result.breakdown_at(.5)
    assert target["eta_ohm"] == pytest.approx(.06 if profile else .075)
    assert target["residual"] == pytest.approx(.02, abs=2e-7)


@pytest.mark.parametrize("mode", ["OER", "HER", "Cell negative"])
def test_averaged_adapter_matches_the_shared_numerical_result(tmp_path, mode):
    potential, current, params, extra, _ = _source(mode, profile=True)
    source = tmp_path / "curve.csv"
    pd.DataFrame({"E(V)": potential, "j(A/cm2)": current}).to_csv(source, index=False)
    engine = decouple_curve(potential, current, DecouplingSettings.from_params(params, extra))
    averaged = LsvTool().run([source], params, extra)
    order = np.argsort(potential)
    exported = averaged.tables["Overpotential components"]
    for label, values in [("η_kin (V)", engine.eta_kin), ("η_ohm (V)", engine.eta_ohm),
                          ("η_res (V)", np.maximum(engine.residual, 0))]:
        np.testing.assert_allclose(exported[label], engine.sign * values[order], atol=1e-10)
    assert averaged.scalars["b_kin (V/dec)"] == pytest.approx(engine.fit["b_kin"], rel=1e-9)


def test_signed_residual_and_source_rows_survive_invalid_and_opposite_sign_rows():
    potential, current, params, extra, _ = _source()
    potential[current > .3] -= .05
    gap = (current > .15) & (current < .2)
    potential[gap] = np.nan
    current[0] = -current[0]
    indices = np.arange(len(current)) + 100
    curve = decouple_curve(potential, current, DecouplingSettings.from_params(params, extra),
                           source_indices=indices)

    np.testing.assert_array_equal(curve.source_indices, indices)
    np.testing.assert_array_equal(curve.measured_current, current)
    assert np.isnan(curve.current[0])
    assert np.isnan(curve.residual[gap]).all()
    assert np.min(curve.residual[np.isfinite(curve.residual)]) < -.04
    assert curve.breakdown_at(.5)["residual"] == pytest.approx(-.03, abs=2e-7)
    with pytest.raises(ValueError, match="acquisition gap"):
        curve.breakdown_at(.17)
    evaluated = curve.evaluate_at([2., np.nan, 2.], [.5, .5, 0.])
    assert np.isfinite(evaluated["residual"][0])
    assert np.isnan(evaluated["residual"][1:]).all()


def test_reference_conversion_occurs_once_and_breakdown_never_extrapolates():
    potential, current, params, extra, _ = _source()
    params.update(reference="Custom E_ref vs SHE", custom_e_ref_vs_she=.123, ph=0.)
    measured = potential - .123
    curve = decouple_curve(measured, current, DecouplingSettings.from_params(params, extra))
    np.testing.assert_allclose(curve.potential, potential, atol=1e-15)
    np.testing.assert_array_equal(curve.measured_potential, measured)
    assert curve.fit["b_kin"] == pytest.approx(.06)
    for target in [0., np.nan, 2.]:
        with pytest.raises(ValueError, match="Breakdown target"):
            curve.breakdown_at(target)
