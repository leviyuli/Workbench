"""Batch KK keeps each spectrum, sample identifier, and HFR independent."""

import numpy as np
import pandas as pd
import pytest

import workbench.tools.eis.logic_eis as eis_logic_module
from workbench.tools.eis.logic_eis import EisLogic
from workbench.tools.eis.tool import EisTool


@pytest.fixture(autouse=True)
def kk_backend(monkeypatch):
    """Use controlled percent residuals without invoking a KK optimizer."""
    control = {"datasets": [], "residuals": None, "fail_calls": set()}

    class KkResult:
        def __init__(self, dataset):
            self.dataset = dataset

        def get_residuals_data(self):
            frequency = self.dataset.get_frequencies()
            residuals = control["residuals"]
            error = (np.ones(len(frequency)) if residuals is None
                     else np.asarray(residuals(frequency), dtype=float))
            return frequency, error, np.zeros(len(frequency))

    def perform_kk(dataset, test, num_procs):
        assert test == "complex"
        assert num_procs == 1
        control["datasets"].append(dataset)
        if len(control["datasets"]) in control["fail_calls"]:
            raise RuntimeError("controlled KK failure")
        return KkResult(dataset)

    def forbidden(*args, **kwargs):
        pytest.fail("Batch KK must not fit or average separate spectra.")

    monkeypatch.setattr(
        eis_logic_module.pyimpspec, "perform_kramers_kronig_test", perform_kk)
    monkeypatch.setattr(EisLogic, "fit_impedance", forbidden)
    monkeypatch.setattr(EisLogic, "_average_points", forbidden)
    return control


def write_spectrum(path, rows):
    """Rows contain frequency, real impedance, and signed imaginary impedance."""
    path.write_text(
        "Frequency (Hz),Re(Z) (Ohm.cm2),-Im(Z) (Ohm.cm2)\n"
        + "".join(f"{frequency},{real},{-imag}\n"
                  for frequency, real, imag in rows),
        encoding="utf-8",
    )
    return path


FIRST = [(10000, 0.1, 0.1), (1000, 0.5, -0.1), (100, 0.9, -0.6)]
SECOND = [(10000, 0.5, 0.3), (1000, 0.9, -0.1), (100, 1.3, -0.6)]


def plot_coordinates(axis):
    """Accept either a line or a scatter for the HFR summary plot."""
    if axis.lines:
        return (np.asarray(axis.lines[0].get_xdata(), dtype=float),
                np.asarray(axis.lines[0].get_ydata(), dtype=float))
    offsets = np.asarray(axis.collections[0].get_offsets(), dtype=float)
    return offsets[:, 0], offsets[:, 1]


def test_batch_reports_each_intercept_without_cross_file_averaging(tmp_path, kk_backend):
    sources = [write_spectrum(tmp_path / "first.csv", FIRST),
               write_spectrum(tmp_path / "second.csv", SECOND)]

    result = EisTool().run(sources, {"mode": "Batch KK test"})

    summary = result.tables["HFR summary"]
    assert summary["Sample"].tolist() == [1, 2]
    assert summary["File"].tolist() == ["first.csv", "second.csv"]
    np.testing.assert_allclose(summary["HFR (Ω·cm²)"], [0.3, 0.8])
    assert summary["Status"].tolist() == ["OK", "OK"]
    assert summary["Points"].tolist() == [3, 3]
    assert summary["Points passing KK"].tolist() == [3, 3]
    np.testing.assert_allclose(summary["KK pass (%)"], [100, 100])
    assert summary["HFR method"].tolist() == [
        "High-frequency x-axis intersection",
        "High-frequency x-axis intersection",
    ]
    assert len(kk_backend["datasets"]) == 2
    np.testing.assert_allclose(
        kk_backend["datasets"][0].get_impedances().real, [0.1, 0.5, 0.9])
    np.testing.assert_allclose(
        kk_backend["datasets"][1].get_impedances().real, [0.5, 0.9, 1.3])

    assert list(result.figures)[0] == "HFR by sample"
    assert "KK sample 1" in result.figures and "KK sample 2" in result.figures
    assert "Spectrum 1" in result.tables and "Spectrum 2" in result.tables
    raw = result.tables["KK diagnostics (raw)"]
    assert raw.groupby("Sample")["File"].first().to_dict() == {
        1: "first.csv", 2: "second.csv"}
    assert raw.groupby("Sample")["scan"].first().to_dict() == {1: 1, 2: 2}
    axis = result.figures["HFR by sample"].axes[0]
    x, y = plot_coordinates(axis)
    np.testing.assert_array_equal(x, [1, 2])
    np.testing.assert_allclose(y, [0.3, 0.8])
    assert "HFR" in axis.get_ylabel()
    assert "sample" in axis.get_xlabel().lower()
    np.testing.assert_array_equal(axis.get_xticks(), [1, 2])
    for sample in (1, 2):
        labels = result.figures[f"KK sample {sample}"].axes[0].get_legend_handles_labels()[1]
        assert "spectrum" in labels
        assert "averaged" not in labels


def test_batch_preserves_selection_order_and_independent_fallback(tmp_path):
    fallback = [(10000, 1.2, -0.3), (1000, 0.7, -0.1), (100, 1.8, -0.6)]
    sources = [write_spectrum(tmp_path / "z_crossing.csv", SECOND),
               write_spectrum(tmp_path / "a_no_crossing.csv", fallback)]

    result = EisTool().run(sources, {"mode": "Batch KK test"})

    summary = result.tables["HFR summary"]
    assert summary["File"].tolist() == ["z_crossing.csv", "a_no_crossing.csv"]
    np.testing.assert_allclose(summary["HFR (Ω·cm²)"], [0.8, 0.7])
    assert "no x-axis intersection" in summary.loc[1, "HFR method"]
    x, y = plot_coordinates(result.figures["HFR by sample"].axes[0])
    np.testing.assert_array_equal(x, [1, 2])
    np.testing.assert_allclose(y, [0.8, 0.7])


def test_unreadable_file_retains_sample_number_and_later_file_runs(tmp_path, kk_backend):
    sources = [write_spectrum(tmp_path / "first.csv", FIRST),
               tmp_path / "missing.csv",
               write_spectrum(tmp_path / "last.csv", SECOND)]

    result = EisTool().run(sources, {"mode": "Batch KK test"})

    summary = result.tables["HFR summary"]
    assert summary["Sample"].tolist() == [1, 2, 3]
    assert summary["File"].tolist() == ["first.csv", "missing.csv", "last.csv"]
    assert summary["Status"].tolist() == ["OK", "Failed", "OK"]
    assert np.isnan(summary.loc[1, "HFR (Ω·cm²)"])
    assert "missing.csv" in summary.loc[1, "Error"]
    assert len(kk_backend["datasets"]) == 2
    assert "KK sample 3" in result.figures and "KK sample 2" not in result.figures
    raw = result.tables["KK diagnostics (raw)"]
    assert raw["Sample"].unique().tolist() == [1, 3]
    assert raw["scan"].unique().tolist() == [1, 3]
    axis = result.figures["HFR by sample"].axes[0]
    x, y = plot_coordinates(axis)
    finite = np.isfinite(y)
    np.testing.assert_array_equal(x[finite], [1, 3])
    np.testing.assert_allclose(y[finite], [0.3, 0.8])
    np.testing.assert_array_equal(axis.get_xticks(), [1, 2, 3])
    assert any("missing.csv" in warning for warning in result.warnings)


def test_kk_failure_does_not_stop_later_spectrum(tmp_path, kk_backend):
    kk_backend["fail_calls"] = {1}
    sources = [write_spectrum(tmp_path / "failed.csv", FIRST),
               write_spectrum(tmp_path / "success.csv", SECOND)]

    result = EisTool().run(sources, {"mode": "Batch KK test"})

    summary = result.tables["HFR summary"]
    assert summary["Status"].tolist() == ["Failed", "OK"]
    assert np.isnan(summary.loc[0, "HFR (Ω·cm²)"])
    assert "controlled KK failure" in summary.loc[0, "Error"]
    assert summary.loc[1, "HFR (Ω·cm²)"] == pytest.approx(0.8)
    assert len(kk_backend["datasets"]) == 2
    assert result.tables["KK diagnostics (raw)"]["Sample"].unique().tolist() == [2]
    assert result.tables["KK diagnostics (raw)"]["scan"].unique().tolist() == [2]


@pytest.mark.parametrize("threshold, passing, minimum", [(2.5, 2, 100), (3.5, 3, 10)])
def test_batch_applies_frequency_window_and_kk_threshold(
        tmp_path, kk_backend, threshold, passing, minimum):
    rows = [(10000, 0.1, 0.3), (1000, 0.5, -0.1),
            (100, 0.9, -0.6), (10, 1.3, -0.8), (1, 1.7, -1.0)]
    source = write_spectrum(tmp_path / "window.csv", rows)
    kk_backend["residuals"] = lambda frequency: np.arange(1, len(frequency) + 1)

    result = EisTool().run([source], {
        "mode": "Batch KK test", "f_min": 10.0, "f_max": 1000.0,
        "kk_threshold_pct": threshold,
    })

    np.testing.assert_array_equal(
        kk_backend["datasets"][0].get_frequencies(), [1000, 100, 10])
    summary = result.tables["HFR summary"].iloc[0]
    assert summary["Points"] == 3
    assert summary["Points passing KK"] == passing
    assert summary["KK pass (%)"] == pytest.approx(100 * passing / 3)
    assert summary["Suggested f_min (Hz)"] == minimum
    assert summary["Suggested f_max (Hz)"] == 1000
    assert summary["HFR (Ω·cm²)"] == pytest.approx(0.5)
    assert "no x-axis intersection" in summary["HFR method"]
    raw = result.tables["KK diagnostics (raw)"]
    np.testing.assert_allclose(raw["kk_err"], [0.01, 0.02, 0.03])
    axis = result.figures["KK sample 1"].axes[1]
    np.testing.assert_allclose(axis.lines[0].get_ydata(), [threshold, threshold])


def test_batch_preserves_single_analysis_cache_and_parameters(tmp_path):
    source = write_spectrum(tmp_path / "spectrum.csv", FIRST)
    cache = {"key": ("single-analysis",), "range": None,
             "kk_by_freq": pd.Series([0.01], index=[1000.0])}
    tool = EisTool()
    tool._kk_cache = cache

    result = tool.run([source], {"mode": "Batch KK test"})

    assert tool._kk_cache is cache
    assert result.param_updates == {}
    assert result.param_annotations == {}


def test_batch_requires_input_and_reports_when_every_spectrum_fails(tmp_path, kk_backend):
    tool = EisTool()
    with pytest.raises(ValueError, match="(?i)(needs|add|select|at least).*file"):
        tool.run([], {"mode": "Batch KK test"})

    with pytest.raises(ValueError, match="missing.csv"):
        tool.run([tmp_path / "missing.csv"], {"mode": "Batch KK test"})

    source = write_spectrum(tmp_path / "failed.csv", FIRST)
    kk_backend["fail_calls"] = {1}
    with pytest.raises(ValueError, match="controlled KK failure"):
        tool.run([source], {"mode": "Batch KK test"})


def test_batch_summary_and_diagnostics_are_exportable(tmp_path):
    source = write_spectrum(tmp_path / "spectrum.csv", FIRST)
    tool = EisTool()
    result = tool.run([source], {"mode": "Batch KK test"})

    artifacts = tool.exportable_artifacts(result)

    workbook = next(artifact for artifact in artifacts if artifact.kind == "xlsx")
    pd.testing.assert_frame_equal(workbook.payload["HFR summary"],
                                  result.tables["HFR summary"])
    assert "KK diagnostics (raw)" in workbook.payload
    pngs = {artifact.name: artifact.payload for artifact in artifacts if artifact.kind == "png"}
    assert pngs["HFR by sample (PNG)"] is result.figures["HFR by sample"]
    assert pngs["KK sample 1 (PNG)"] is result.figures["KK sample 1"]
