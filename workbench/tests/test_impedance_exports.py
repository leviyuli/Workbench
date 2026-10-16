"""Impedance workbooks keep final plotted data and scientific diagnostics."""

import numpy as np
import pandas as pd
import pytest

from workbench.tools.eis.logic_eis import EisLogic
from workbench.tools.eis.tool import EisTool


@pytest.fixture
def spectrum(tmp_path):
    frequency = np.logspace(4, -1, 8)
    impedance = 0.2 + 0.8 / (1 + 2j * np.pi * frequency * 0.02)
    path = tmp_path / "sample.csv"
    pd.DataFrame({"Frequency (Hz)": frequency,
                  "Re(Z) (Ohm.cm2)": impedance.real,
                  "Im(Z) (Ohm.cm2)": impedance.imag}).to_csv(path, index=False)
    return path, frequency, impedance


@pytest.fixture
def controlled_kk(monkeypatch):
    class KkResult:
        def __init__(self, dataset):
            self.dataset = dataset

        def get_residuals_data(self):
            frequency = self.dataset.get_frequencies()
            return frequency, np.arange(1, len(frequency) + 1), np.zeros(len(frequency))

    import pyimpspec
    monkeypatch.setattr(pyimpspec, "perform_kramers_kronig_test",
                        lambda dataset, **_: KkResult(dataset))


def workbook_tables(tool, result):
    return next(artifact.payload for artifact in tool.exportable_artifacts(result)
                if artifact.kind == "xlsx")


def default_params(**changes):
    params = {spec.key: spec.default for spec in EisTool().parameter_schema()}
    params.update(changes)
    return params


@pytest.mark.parametrize("mode", ["KK test", "Batch KK test"])
def test_kk_export_preserves_percent_diagnostics_and_omits_duplicate_spectra(
        spectrum, controlled_kk, mode):
    tool = EisTool()
    result = tool.run([spectrum[0]], {"mode": mode, "kk_threshold_pct": 5.5})
    original = result.tables["KK diagnostics (raw)"].copy(deep=True)

    sheets = workbook_tables(tool, result)

    assert "KK diagnostics (raw)" not in sheets
    assert not any(name.startswith("Spectrum ") for name in sheets)
    diagnostics = sheets["KK diagnostics"]
    np.testing.assert_array_equal(diagnostics["Frequency (Hz)"], original["f"])
    np.testing.assert_allclose(diagnostics["Re(Z) (Ω·cm²)"], original["zr"])
    np.testing.assert_allclose(diagnostics["Im(Z) (Ω·cm²)"], original["zi"])
    np.testing.assert_allclose(diagnostics["KK residual (%)"], original["kk_err"] * 100)
    np.testing.assert_array_equal(diagnostics["Passes KK"], original["kk_err"] <= 0.055)
    assert not {"scan", "in_scope", "range_f_min", "range_f_max"} & set(diagnostics)
    pd.testing.assert_frame_equal(result.tables["KK diagnostics (raw)"], original)
    if mode == "KK test":
        assert set(sheets) == {"KK summary", "KK diagnostics", "Averaged spectrum"}
        assert list(sheets["Averaged spectrum"]) == [
            "Frequency (Hz)", "Re(Z) (Ω·cm²)", "Im(Z) (Ω·cm²)"]
        summary = sheets["KK summary"].set_index("Parameter")["Value"]
        assert summary["KK threshold (%)"] == 5.5
        assert "HFR estimate (Ω·cm²)" in summary
    else:
        assert set(sheets) == {"HFR summary", "KK diagnostics"}
        assert sheets["HFR summary"]["KK threshold (%)"].iloc[0] == 5.5
        assert diagnostics["File"].unique().tolist() == [spectrum[0].name]


def test_multi_scan_kk_keeps_scan_identity_and_nonzero_standard_deviations(
        spectrum, controlled_kk, tmp_path):
    first, frequency, impedance = spectrum
    second = tmp_path / "second.csv"
    pd.DataFrame({"Frequency (Hz)": frequency,
                  "Re(Z) (Ohm.cm2)": impedance.real + 0.1,
                  "Im(Z) (Ohm.cm2)": impedance.imag - 0.02}).to_csv(second, index=False)
    tool = EisTool()
    result = tool.run([first, second], {"mode": "KK test"})

    sheets = workbook_tables(tool, result)

    assert sheets["KK diagnostics"]["Scan"].unique().tolist() == [1, 2]
    assert "Re(Z) SD (Ω·cm²)" in sheets["Averaged spectrum"]
    np.testing.assert_allclose(sheets["Averaged spectrum"]["Re(Z) SD (Ω·cm²)"],
                               result.tables["Averaged spectrum"]["Z'_std"])


def test_fit_export_keeps_parameters_actual_fit_points_and_signed_residuals(
        spectrum, controlled_kk, monkeypatch):
    fitted = np.array([0.0, 0.2, 0.3, 0.1, 0.9, 0.0, np.inf])
    info = {"ssr": 0.0123, "converged": True, "n_restarts": 1, "best_attempt": 1,
            "ranking_basis": "hfr_rcl_se", "score_hfr_rcl_pct": 2.0}
    monkeypatch.setattr(EisLogic, "fit_impedance", lambda *args, **kwargs:
                        ((fitted, np.zeros(7), np.zeros(8), info), None))
    tool = EisTool()
    result = tool.run([spectrum[0]], default_params(mode="Fit", kk_threshold_pct=5.5))
    original = result.tables["Fitted curve"].copy(deep=True)

    sheets = workbook_tables(tool, result)

    assert set(sheets) == {"Fit parameters", "Fitted curve", "Fit quality"}
    pd.testing.assert_frame_equal(sheets["Fit parameters"], result.tables["Fit parameters"])
    curve = sheets["Fitted curve"]
    assert len(curve) == 5  # Three points failed the KK threshold.
    np.testing.assert_allclose(curve["Frequency (Hz)"], original["Frequency (Hz)"])
    np.testing.assert_allclose(curve["Fit − data Re(Z) (Ω·cm²)"],
                               original["Fit Re(Z)"] - original["Data Re(Z)"])
    np.testing.assert_allclose(curve["Fit − data Im(Z) (Ω·cm²)"],
                               original["Fit Im(Z)"] - original["Data Im(Z)"])
    quality = sheets["Fit quality"].set_index("Parameter")["Value"]
    assert quality["SSR"] == info["ssr"]
    assert quality["Residual weighting"] == "modulus"
    pd.testing.assert_frame_equal(result.tables["Fitted curve"], original)
    assert "Averaged spectrum" in result.tables


@pytest.mark.parametrize("with_overlay", [False, True])
def test_simulation_export_matches_plots_on_independent_frequency_grids(spectrum, with_overlay):
    tool = EisTool()
    result = tool.run([spectrum[0]] if with_overlay else [],
                      default_params(mode="Simulate", sim_fmin=1.0, sim_fmax=1e4, sim_points=13))

    sheets = workbook_tables(tool, result)

    curve = sheets["Simulated curve"]
    assert len(curve) == 13
    assert list(curve) == ["Frequency (Hz)", "Re(Z) (Ω·cm²)",
                           "Im(Z) (Ω·cm²)", "|Z| (Ω·cm²)"]
    nyquist = result.figures["Simulation"].axes[0]
    model_line = next(line for line in nyquist.lines if line.get_label() == "Simulation")
    np.testing.assert_allclose(curve["Re(Z) (Ω·cm²)"], model_line.get_xdata())
    np.testing.assert_allclose(curve["Im(Z) (Ω·cm²)"], -model_line.get_ydata())
    parameters = sheets["Simulation parameters"].set_index("Parameter")
    assert parameters.loc["HFR", "Value"] == 0.0
    assert "L_wire" not in parameters.index
    if with_overlay:
        overlay = sheets["Measured overlay"]
        measured = next(line for line in nyquist.lines if line.get_label() == "Data")
        np.testing.assert_allclose(overlay["Re(Z) (Ω·cm²)"], measured.get_xdata())
        np.testing.assert_allclose(overlay["Im(Z) (Ω·cm²)"], -measured.get_ydata())
        assert len(overlay) != len(curve)
    else:
        assert "Measured overlay" not in sheets
