"""Regression tests for data-integrity defects found during the audit."""

import numpy as np
import pandas as pd
import pytest

import workbench.tools.eis.logic_eis as eis_logic_module
from workbench.tools.eis.logic_eis import EisLogic
from workbench.tools.eis.tool import EisTool
from workbench.tools.it_integrate.logic import integrate_potentiostatic_tests
from workbench.tools.it_integrate.tool import ItIntegrateTool
from workbench.tools.lsv.logic_tafel import get_overpotential_at_current


def test_it_integrate_reorders_later_columns_by_header(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("T(s)\ti(A)\n0\t10\n1\t11\n", encoding="utf-8")
    second.write_text("i(A)\tT(s)\n20\t0\n21\t1\n", encoding="utf-8")

    out = integrate_potentiostatic_tests([first, second], trace_type="i_t")

    assert out["warnings"] == []
    assert out["headers"] == ["T(s)", "i(A)"]
    assert np.allclose(out["data"][:, 0], [0.0, 1.0, 2.0, 3.0])
    assert np.allclose(out["data"][:, 1], [10.0, 11.0, 20.0, 21.0])


def test_it_resample_preserves_measured_endpoint(tmp_path):
    source = tmp_path / "trace.txt"
    source.write_text("T(s)\ti(A)\n0\t10\n1\t20\n", encoding="utf-8")

    out = integrate_potentiostatic_tests([source], interval_s=1.0, trace_type="i_t")

    assert np.allclose(out["data"][:, 0], [0.0, 1.0])
    assert np.allclose(out["data"][:, 1], [10.0, 20.0])


def test_it_source_designation_colors_and_exports_column_groups(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text(
        "E(V)\tj(A/cm2)\tT(s)\n1.0\t10\t0\n1.1\t11\t1\n",
        encoding="utf-8",
    )
    second.write_text(
        "E(V)\tj(A/cm2)\tT(s)\n2.0\t20\t0\n2.1\t21\t1\n",
        encoding="utf-8",
    )

    result = ItIntegrateTool().run(
        [second, first],
        {
            "trace_type": "i_t",
            "interval": "none",
            "designate_sources": True,
        },
    )

    exported = result.tables["Integrated"]
    assert list(exported.columns) == [
        "second.txt · E(V)", "second.txt · j(A/cm2)", "second.txt · T(s)",
        "first.txt · E(V)", "first.txt · j(A/cm2)", "first.txt · T(s)",
    ]
    assert exported.shape == (2, 6)
    assert np.allclose(exported["first.txt · T(s)"], [2.0, 3.0])

    ax = result.figures["i-t"].axes[0]
    assert [line.get_label() for line in ax.lines] == ["second.txt", "first.txt"]
    assert ax.lines[0].get_color() != ax.lines[1].get_color()


def test_it_source_designation_is_off_by_default(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("E(V)\tj(A/cm2)\tT(s)\n1\t10\t0\n", encoding="utf-8")
    second.write_text("E(V)\tj(A/cm2)\tT(s)\n2\t20\t0\n", encoding="utf-8")

    result = ItIntegrateTool().run(
        [first, second],
        {"trace_type": "i_t", "interval": "none"},
    )

    assert list(result.tables["Integrated"].columns) == ["E(V)", "j(A/cm2)", "T(s)"]
    assert result.tables["Integrated"].shape == (2, 3)


def test_it_source_designation_resamples_each_file_group(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text(
        "E(V)\tj(A/cm2)\tT(s)\n1\t10\t0\n3\t30\t2\n",
        encoding="utf-8",
    )
    second.write_text(
        "E(V)\tj(A/cm2)\tT(s)\n4\t40\t0\n6\t60\t2\n",
        encoding="utf-8",
    )

    result = ItIntegrateTool().run(
        [first, second],
        {
            "trace_type": "i_t",
            "interval": "1 s",
            "designate_sources": True,
        },
    )

    exported = result.tables["Integrated"]
    assert exported.shape == (3, 6)
    assert np.allclose(exported["first.txt · T(s)"], [0.0, 1.0, 2.0])
    assert np.allclose(exported["first.txt · j(A/cm2)"], [10.0, 20.0, 30.0])
    assert np.allclose(exported["second.txt · T(s)"], [4.0, 5.0, 6.0])


def test_lsv_breakdown_rejects_nonpositive_and_unmeasured_targets():
    current = np.array([0.1, 0.2])
    voltage = np.array([1.5, 1.6])
    fit = {"b_kin": 0.06, "i0": 1e-6}

    with pytest.raises(ValueError, match="greater than zero"):
        get_overpotential_at_current(0, current, voltage, 1.23, 0, 0, fit)
    with pytest.raises(ValueError, match="outside the measured range"):
        get_overpotential_at_current(1.0, current, voltage, 1.23, 0, 0, fit)


def test_eis_average_aligns_near_equal_frequency_grids():
    points = pd.DataFrame({
        "f": [100.0, 10.0, 100.0001, 10.0001],
        "zr": [1.0, 2.0, 3.0, 4.0],
        "zi": [-1.0, -2.0, -3.0, -4.0],
        "scan": [1, 1, 2, 2],
    })

    averaged, rsd = EisLogic()._average_points(points)

    assert len(averaged) == 2
    assert np.allclose(sorted(averaged["Z'(Ohm.cm²)"]), [2.0, 3.0])
    assert np.isfinite(rsd) and rsd > 0


def test_eis_kk_cache_range_must_cover_requested_fit_range():
    assert EisTool._range_covers((10.0, 1000.0), (20.0, 500.0))
    assert not EisTool._range_covers((10.0, 1000.0), (1.0, 500.0))
    assert EisTool._range_covers(None, (1.0, 1e6))


def test_eis_kk_mode_has_full_spectrum_frequency_reset():
    specs = {spec.key: spec for spec in EisTool().parameter_schema()}

    reset = specs["freq_reset"]
    assert reset.kind == "reset"
    assert reset.reset_keys == ["f_min", "f_max"]
    assert reset.reset_method == "input_frequency_bounds"
    assert reset.show_if == ("mode", "KK test")
    assert specs["f_min"].default == 0.0
    assert specs["f_max"].default == 0.0
    assert specs["kk_threshold_pct"].default == 10.0


def test_eis_frequency_reset_uses_input_file_bounds(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    header = "Frequency (Hz),Re(Z) (Ohm.cm2),-Im(Z) (Ohm.cm2)\n"
    first.write_text(header + "100,1,2\n1,2,3\n", encoding="utf-8")
    second.write_text(header + "2500,1,2\n0.05,2,3\n", encoding="utf-8")

    bounds = EisTool().input_frequency_bounds([first, second])

    assert bounds == {"f_min": 0.05, "f_max": 2500.0}


def test_eis_pyimpspec_percent_residuals_are_normalized_and_aligned(
        tmp_path, monkeypatch):
    source = tmp_path / "ascending.csv"
    source.write_text(
        "Frequency (Hz),Re(Z) (Ohm.cm2),-Im(Z) (Ohm.cm2)\n"
        "1,11,1\n"
        "10,22,2\n"
        "100,33,3\n",
        encoding="utf-8",
    )

    class FakeKkResult:
        def __init__(self, dataset):
            self.dataset = dataset

        def get_residuals_data(self):
            return (self.dataset.get_frequencies(), np.array([1.0, 2.0, 3.0]),
                    np.zeros(3))

    def fake_kk_test(dataset, test, num_procs):
        assert test == "complex"
        assert num_procs == 1
        return FakeKkResult(dataset)

    monkeypatch.setattr(
        eis_logic_module.pyimpspec, "perform_kramers_kronig_test", fake_kk_test)

    result = EisTool().run([source], {
        "mode": "KK test",
        "kk_threshold_pct": 2.5,
        "f_min": 0.0,
        "f_max": 0.0,
    })

    raw = result.tables["KK diagnostics (raw)"]
    assert np.allclose(raw["f"], [100.0, 10.0, 1.0])
    assert np.allclose(raw["zr"], [33.0, 22.0, 11.0])
    assert np.allclose(raw["kk_err"], [0.01, 0.02, 0.03])
    assert result.scalars["Points passing KK"] == "2/3 (67%)"
    assert result.param_updates["f_min"] == 10.0
    assert result.param_updates["f_max"] == 100.0

    residual_axis = result.figures["KK diagnostics"].axes[1]
    plotted = np.asarray(residual_axis.collections[0].get_offsets())
    assert np.allclose(plotted[:, 1], [1.0, 2.0, 3.0])
    assert np.allclose(residual_axis.lines[0].get_ydata(), [2.5, 2.5])


def test_eis_complex_fit_uses_two_observations_per_frequency():
    logic = EisLogic()
    logic.fit_n_jobs = 1
    frequency = np.logspace(5, -2, 6)
    truth = [2e-5, 0.5, 0.6, 0.1, 0.85, 0.9]
    impedance = logic.evaluate_model(
        "Transmission Line", truth + [np.inf], frequency)
    rng = np.random.default_rng(1)
    noisy = impedance * (
        1.0 + 1e-3 * (rng.normal(size=frequency.size)
                      + 1j * rng.normal(size=frequency.size)))

    out, error = logic.fit_impedance(
        "Transmission Line", truth, frequency, noisy, n_restarts=1,
        fit_inductance=True, max_nfev=2000)

    assert error is None
    _params, standard_errors, residuals, info = out
    assert residuals.shape == (12,)
    assert np.all(np.isfinite(standard_errors[:6]))
    assert np.all(standard_errors[:6] > 0)

    underdetermined, error = logic.fit_impedance(
        "Transmission Line", truth, frequency[:3], impedance[:3], n_restarts=1,
        fit_inductance=True, max_nfev=2000)
    assert error is None
    assert np.all(np.isnan(underdetermined[1][:6]))
