"""Repeated EIS frequencies must be consolidated before KK and averaging."""

import numpy as np
import pandas as pd
import pytest

import workbench.tools.eis.logic_eis as eis_logic_module
from workbench.tools.eis.logic_eis import EisLogic


def spectrum(frequencies, impedances):
    impedance = np.asarray(impedances, dtype=complex)
    return {
        "frequency": np.asarray(frequencies, dtype=float),
        "z_real": impedance.real.copy(),
        "z_imag": impedance.imag.copy(),
        "z_complex": impedance.copy(),
    }


@pytest.fixture
def kk_datasets(monkeypatch):
    datasets = []

    class KkResult:
        def __init__(self, dataset):
            self.dataset = dataset

        def get_residuals_data(self):
            frequencies = self.dataset.get_frequencies()
            return frequencies, np.arange(1, len(frequencies) + 1), np.zeros(len(frequencies))

    def kk_test(dataset, test, num_procs):
        assert isinstance(dataset, eis_logic_module.pyimpspec.DataSet)
        assert test == "complex"
        assert num_procs == 1
        datasets.append(dataset)
        return KkResult(dataset)

    monkeypatch.setattr(eis_logic_module.pyimpspec, "perform_kramers_kronig_test", kk_test)
    return datasets


def test_kk_averages_duplicates_and_preserves_residual_alignment(kk_datasets):
    data = spectrum([10, 1, 100, 10], [18 - 2j, 10 - 1j, 30 - 3j, 22 - 6j])
    original = {key: value.copy() for key, value in data.items()}
    logic = EisLogic()
    logic.kk_threshold = 0.025

    raw, averaged, suggested, _rsd, warnings = logic.process_spectra([data])

    assert len(kk_datasets) == 1
    np.testing.assert_array_equal(kk_datasets[0].get_frequencies(), [100, 10, 1])
    np.testing.assert_allclose(kk_datasets[0].get_impedances(), [30 - 3j, 20 - 4j, 10 - 1j])
    np.testing.assert_array_equal(raw["f"], [100, 10, 1])
    np.testing.assert_allclose(raw["zr"], [30, 20, 10])
    np.testing.assert_allclose(raw["zi"], [-3, -4, -1])
    np.testing.assert_allclose(raw["kk_err"], [0.01, 0.02, 0.03])
    assert suggested == (10, 100)
    assert len(averaged) == 3
    assert len(warnings) == 1
    assert "duplicate frequency" in warnings[0].lower()
    assert "averaged" in warnings[0].lower()
    for key in data:
        np.testing.assert_array_equal(data[key], original[key])


@pytest.mark.parametrize("method", ["process_spectra", "average_spectra"])
@pytest.mark.parametrize("window", [None, (10, 100)])
def test_minimum_count_uses_unique_in_window_frequencies(method, window, kk_datasets):
    data = spectrum([100, 100, 10], [1 - 1j, 2 - 2j, 3 - 3j])
    if window is not None:
        data = spectrum([100, 100, 10, 1], [1 - 1j, 2 - 2j, 3 - 3j, 4 - 4j])

    with pytest.raises(ValueError, match="need at least 3") as error:
        getattr(EisLogic(), method)([data], freq_range=window)

    assert "2 in-scope" in str(error.value)
    assert kk_datasets == []


def test_duplicates_outside_frequency_window_do_not_warn(kk_datasets):
    data = spectrum([1000, 100, 10, 1, 1000], [1 - 1j, 3 - 3j, 4 - 4j, 5 - 5j, 2 - 2j])
    logic = EisLogic()

    raw, kk_average, _suggested, _rsd, warnings = logic.process_spectra(
        [data], freq_range=(1, 100)
    )
    fit_average, _rsd, fit_warnings = logic.average_spectra([data], freq_range=(1, 100))

    assert warnings == fit_warnings == []
    np.testing.assert_array_equal(raw["f"], [100, 10, 1])
    pd.testing.assert_frame_equal(kk_average, fit_average)


def test_invalid_duplicate_is_filtered_before_averaging(kk_datasets):
    data = spectrum([100, 10, 1, 10, 10], [30 - 3j, 18 - 2j, 10 - 1j, 22 - 6j, 99 - 99j])
    data["z_real"][-1] = np.nan

    raw, _average, _suggested, _rsd, warnings = EisLogic().process_spectra([data])

    np.testing.assert_array_equal(raw["f"], [100, 10, 1])
    np.testing.assert_allclose(raw["zr"], [30, 20, 10])
    np.testing.assert_allclose(raw["zi"], [-3, -4, -1])
    assert any("dropped 1" in warning for warning in warnings)
    assert any("duplicate frequency" in warning for warning in warnings)


def test_close_distinct_frequencies_remain_separate_for_kk(kk_datasets):
    data = spectrum([10, 1, 10.0001, 100], [20 - 2j, 10 - 1j, 21 - 3j, 30 - 4j])

    raw, _average, _suggested, _rsd, warnings = EisLogic().process_spectra([data])

    assert warnings == []
    np.testing.assert_array_equal(kk_datasets[0].get_frequencies(), [100, 10.0001, 10, 1])
    np.testing.assert_array_equal(raw["f"], [100, 10.0001, 10, 1])
    np.testing.assert_allclose(raw["zr"], [30, 21, 20, 10])


def test_repeated_rows_do_not_overweight_a_scan_in_either_path(kk_datasets):
    first = spectrum([100, 10, 1, 10], [3 - 3j, 1 - 1j, 5 - 5j, 3 - 3j])
    second = spectrum([100, 10, 1], [5 - 5j, 10 - 10j, 7 - 7j])
    logic = EisLogic()

    raw, kk_average, _suggested, kk_rsd, kk_warnings = logic.process_spectra([first, second])
    fit_average, fit_rsd, fit_warnings = logic.average_spectra([first, second])

    assert len(raw) == 6
    assert len(kk_datasets) == 2
    pd.testing.assert_frame_equal(kk_average, fit_average)
    assert kk_rsd == pytest.approx(fit_rsd)
    assert kk_warnings == fit_warnings
    averaged = kk_average.set_index("Freq(Hz)")
    np.testing.assert_allclose(averaged.loc[[100, 10, 1], "Z'(Ohm.cm²)"], [4, 6, 6])
    np.testing.assert_allclose(averaged.loc[[100, 10, 1], "Z''(Ohm.cm²)"], [-4, -6, -6])
    assert averaged.loc[10, "Z'_std"] == pytest.approx(np.sqrt(32))
