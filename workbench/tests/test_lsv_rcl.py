"""Regression coverage for optional catalyst-layer analysis in LSV decoupling."""
import numpy as np
import pandas as pd
import pytest

from workbench import theme
from workbench.tools.lsv import logic_tafel, tool as lsv_tool


def _synthetic_lsv(tmp_path, reaction):
    current = np.sort(np.append(np.logspace(-3, 0, 350), 0.7))
    kinetic = 0.06 * np.log10(current / 1e-6)
    ohmic = 0.15 * current
    residual = 0.5 * np.maximum(current - 0.5, 0)
    sign = -1 if reaction == "HER" else 1
    reversible = 0.0 if reaction == "HER" else 1.2291
    potential = reversible + sign * (kinetic + ohmic + residual)
    path = tmp_path / f"{reaction}.csv"
    pd.DataFrame({"E(V)": potential, "j(A/cm2)": sign * current}).to_csv(
        path, index=False,
    )
    params = {
        "electrode_setup": "Three-electrode",
        "reaction": reaction, "reference": "RHE", "temperature": 298.15,
        "ph": 0.0, "hfr": 0.15, "rcl": 3.0, "i_lower": 0.005,
        "i_upper": 0.1, "target_i": 0.7,
    }
    return path, params, current, kinetic, ohmic, residual, reversible, sign


def _reject_rcl(*args, **kwargs):
    pytest.fail("Disabled catalyst-layer analysis must not evaluate the RCL model.")


@pytest.mark.parametrize("offset", [0.0, 0.01, 0.6, 0.9, -0.6])
def test_tafel_fit_accepts_a_single_complete_60_mv_window(offset):
    overpotential = np.linspace(offset, offset + 0.06, 61)
    current = 0.005 * 10 ** ((overpotential - offset) / 0.06)

    fit, error = logic_tafel.fit_tafel_window(
        overpotential, current, i_lower=0.004, i_upper=0.06,
    )

    assert error is None
    assert fit["b_kin"] == pytest.approx(0.06)
    assert fit["i0"] == pytest.approx(0.005 * 10 ** (-offset / 0.06), rel=1e-9, abs=0)
    assert fit["candidate_details"]["window"] == pytest.approx((offset, offset + 0.06))
    assert fit["candidate_details"]["counts"]["60mV"] == 61
    details = fit["candidate_details"]
    for name, (low, high) in details["subwindows"].items():
        assert np.count_nonzero((overpotential >= low) & (overpotential <= high)) == details["counts"][name]
    low, high = details["window"]
    included_current = current[(overpotential >= low) & (overpotential <= high)]
    assert details["current_range"] == (included_current.min(), included_current.max())


@pytest.mark.parametrize("offset", [0.0, 0.01, 0.6, 0.9, -0.6])
def test_tafel_fit_rejects_a_measurably_short_60_mv_window(offset):
    overpotential = np.linspace(offset, offset + 0.06 - 1e-9, 61)
    current = 0.005 * 10 ** ((overpotential - offset) / 0.06)

    fit, error = logic_tafel.fit_tafel_window(
        overpotential, current, i_lower=0.004, i_upper=0.06,
    )

    assert fit is None
    assert "Voltage span too narrow" in error


def test_disabled_rcl_fits_once_and_skips_convergence(monkeypatch):
    current = np.logspace(-3, 0, 350)
    kinetic = 0.06 * np.log10(current / 1e-6)
    potential = 1.2291 + kinetic + 0.15 * current
    original_fit = logic_tafel.fit_tafel_window
    calls = []

    def record_fit(*args, **kwargs):
        calls.append(args[0])
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(logic_tafel, "fit_tafel_window", record_fit)
    monkeypatch.setattr(logic_tafel, "calculate_RCL_overpotential", _reject_rcl)
    monkeypatch.setattr(logic_tafel, "calculate_RCL_utilization", _reject_rcl)
    result, warning = logic_tafel.iterative_tafel_fit(
        potential, current, 1.2291, 0.15, 3.0, max_iter=1, include_rcl=False,
    )

    assert warning is None
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0], kinetic)
    fit, ohmic, rcl, accumulated = result
    assert fit["b_kin"] == pytest.approx(0.06)
    assert fit["i0"] == pytest.approx(1e-6)
    np.testing.assert_allclose(ohmic, 0.15 * current)
    np.testing.assert_array_equal(rcl, np.zeros_like(current))
    np.testing.assert_allclose(accumulated, potential - 1.2291)


def test_disabled_rcl_preserves_fit_errors():
    result, error = logic_tafel.iterative_tafel_fit(
        np.array([1.4, 1.5]), np.array([0.01, 0.02]), 1.2291, 0.15, 3.0,
        include_rcl=False,
    )
    assert result is None
    assert "Not enough points" in error


def test_disabled_rcl_breakdown_retains_unassigned_overpotential(monkeypatch):
    current = np.array([0.1, 0.5, 1.0])
    kinetic = 0.06 * np.log10(current / 1e-6)
    potential = 1.2291 + kinetic + 0.15 * current + 0.12
    monkeypatch.setattr(logic_tafel, "calculate_RCL_overpotential", _reject_rcl)
    components = logic_tafel.get_overpotential_at_current(
        0.5, current, potential, 1.2291, 0.15, 3.0,
        {"b_kin": 0.06, "i0": 1e-6}, include_rcl=False,
    )
    assert components["eta_kin"] == pytest.approx(kinetic[1])
    assert components["eta_ohm"] == pytest.approx(0.075)
    assert components["eta_rcl"] == 0.0
    assert components["eta_res"] == pytest.approx(0.12)


@pytest.mark.parametrize("reaction", ["OER", "HER"])
def test_disabled_rcl_outputs_only_requested_components(tmp_path, monkeypatch, reaction):
    path, params, current, kinetic, ohmic, residual, reversible, sign = (
        _synthetic_lsv(tmp_path, reaction)
    )
    params.update(include_rcl=False, x_log=True, swap_axes=True)
    monkeypatch.setattr(logic_tafel, "calculate_RCL_overpotential", _reject_rcl)
    monkeypatch.setattr(logic_tafel, "calculate_RCL_utilization", _reject_rcl)
    monkeypatch.setattr(lsv_tool, "calculate_RCL_utilization", _reject_rcl)
    tool = lsv_tool.LsvTool()
    result = tool.run([path], params)

    assert result.scalars["b_kin (V/dec)"] == pytest.approx(0.06)
    assert result.scalars["i0 (A/cm²)"] == pytest.approx(1e-6)
    assert not any("converg" in warning.lower() for warning in result.warnings)
    assert "Catalyst utilization" not in result.tables
    assert "Catalyst utilization" not in result.figures

    components = result.tables["Overpotential components"]
    assert list(components.columns) == [
        "Current Density (A/cm²)", "η_kin (V)", "η_ohm (V)", "η_res (V)",
    ]
    # Imported HER samples can be reordered by the parser; compare by |current|.
    order = np.argsort(np.abs(components.iloc[:, 0].to_numpy()))
    np.testing.assert_allclose(components.iloc[order, 0], sign * current)
    np.testing.assert_allclose(components.iloc[order, 1], sign * kinetic)
    np.testing.assert_allclose(components.iloc[order, 2], sign * ohmic)
    np.testing.assert_allclose(components.iloc[order, 3], sign * residual, atol=1e-12)

    curves = result.tables["Curves (stacked)"]
    assert list(curves.columns) == [
        "Current Density (A/cm²)", "E_rev", "E_rev + η_kin", "+ η_ohm",
        "+ η_res", "Original LSV (RHE)",
    ]
    np.testing.assert_allclose(curves["E_rev"], reversible)
    np.testing.assert_allclose(curves["+ η_res"], curves["Original LSV (RHE)"])

    curves_ax = result.figures["Decoupled curves"].axes[0]
    labels = [line.get_label() for line in curves_ax.lines]
    assert labels == [
        theme.OVERPOT_LABELS["E_rev"],
        f"+ {theme.OVERPOT_LABELS['eta_kin']}",
        f"+ {theme.OVERPOT_LABELS['eta_ohm']}",
        f"+ {theme.OVERPOT_LABELS['eta_res']}", "LSV",
    ]
    assert len(curves_ax.collections) == 3
    assert curves_ax.get_yscale() == "log"
    for line, column in zip(curves_ax.lines, curves.columns[1:]):
        np.testing.assert_allclose(line.get_xdata(), curves[column])
        np.testing.assert_allclose(line.get_ydata(), np.abs(curves.iloc[:, 0]))

    breakdown_ax = result.figures["Overpotential breakdown"].axes[0]
    target_kinetic = 0.06 * np.log10(0.7 / 1e-6)
    np.testing.assert_allclose(
        [patch.get_width() for patch in breakdown_ax.patches],
        [reversible, target_kinetic, 0.15 * 0.7, 0.1],
    )
    assert all("RCL" not in text.get_text() for text in breakdown_ax.texts)

    artifacts = tool.exportable_artifacts(result)
    assert all("Catalyst utilization" not in artifact.name for artifact in artifacts)
    workbook = next(artifact.payload for artifact in artifacts if artifact.kind == "xlsx")
    assert "Catalyst utilization" not in workbook
    assert all("RCL" not in column for table in workbook.values() for column in table.columns)


def test_rcl_remains_enabled_by_default(tmp_path):
    path, params, *_ = _synthetic_lsv(tmp_path, "OER")
    params["rcl"] = 0.2
    tool = lsv_tool.LsvTool()
    default_result = tool.run([path], params)
    enabled_result = tool.run([path], {**params, "include_rcl": True})

    assert default_result.scalars == enabled_result.scalars
    assert default_result.warnings == enabled_result.warnings
    assert set(default_result.tables) == set(enabled_result.tables)
    for name in default_result.tables:
        pd.testing.assert_frame_equal(default_result.tables[name], enabled_result.tables[name])
    assert "η_RCL (V)" in default_result.tables["Overpotential components"]
    assert (default_result.tables["Overpotential components"]["η_RCL (V)"] > 0).all()
    assert "Catalyst utilization" in default_result.figures
    workbook = next(
        artifact.payload for artifact in tool.exportable_artifacts(default_result)
        if artifact.kind == "xlsx"
    )
    assert "Catalyst utilization" in workbook


def test_rcl_setting_defaults_to_enabled_and_gates_resistance_input():
    parameters = {spec.key: spec for spec in lsv_tool.LsvTool().parameter_schema()}
    assert parameters["include_rcl"].kind == "bool"
    assert parameters["include_rcl"].default is True
    assert parameters["rcl"].show_if == ("include_rcl", True)
