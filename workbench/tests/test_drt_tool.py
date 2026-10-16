"""DRT calls preserve solver settings, native distributions and residual units."""

from io import BytesIO

import numpy as np
import pyimpspec
import pytest
from pyimpspec.analysis.drt import BHTResult, LMResult, TRNNLSResult

from workbench.tools.drt.tool import DrtTool


@pytest.fixture
def spectrum():
    frequencies = np.logspace(4, -1, 31)
    impedance = 0.2 + 2.0 / (1.0 + 2j * np.pi * frequencies * 0.01)
    return frequencies, impedance


def _native_result(method, spectrum):
    frequencies, impedance = spectrum
    fields = dict(time_constants=np.logspace(-3, 1, 5), frequencies=frequencies,
                  impedances=impedance, residuals=np.full(len(frequencies), 0.01 - 0.02j),
                  pseudo_chisqr=1e-6)
    if method == "bht":
        return BHTResult(**fields, real_gammas=np.array([0.0, 0.8, 0.0, 0.5, 0.0]),
                         imaginary_gammas=np.array([0.0, 0.3, 0.6, 0.2, 0.0]), scores={})
    if method == "lm":
        return LMResult(**fields, gammas=np.array([1.0, -0.3, 3.0, -0.4, 2.0]),
                        singular_values=np.array([1.0, 0.1]))
    return TRNNLSResult(**fields, gammas=np.array([0.0, 0.3, 0.6, 0.2, 0.0]),
                        lambda_value=0.01)


def _stub_solver(monkeypatch, method, spectrum):
    result = _native_result(method, spectrum)
    calls = []
    monkeypatch.setattr(DrtTool, "_read_and_average", staticmethod(lambda *_: (*spectrum, [])))

    def calculate(data, **kwargs):
        calls.append((data, kwargs))
        return result

    monkeypatch.setattr(pyimpspec, "calculate_drt", calculate)
    return result, calls


@pytest.mark.parametrize("method", ["tr-nnls", "tr-rbf"])
def test_manual_regularization_is_passed_to_solver(monkeypatch, spectrum, method):
    _native, calls = _stub_solver(monkeypatch, method, spectrum)
    DrtTool().run([], {"method": method, "lambda": 0.025})
    expected = {"method": method, "lambda_value": 0.025}
    if method == "tr-rbf":
        expected["cross_validation"] = ""
    assert calls[0][1] == expected


@pytest.mark.parametrize("method", ["tr-nnls", "tr-rbf"])
def test_automatic_regularization_uses_solver_defaults(monkeypatch, spectrum, method):
    _native, calls = _stub_solver(monkeypatch, method, spectrum)
    DrtTool().run([], {"method": method, "lambda": 0.0})
    assert calls[0][1] == {"method": method}


@pytest.mark.parametrize("method", ["tr-nnls", "tr-rbf", "bht", "lm"])
def test_native_distribution_components_and_percent_residuals(monkeypatch, spectrum, method):
    native, _calls = _stub_solver(monkeypatch, method, spectrum)

    def forbid_peak_approximation(*args, **kwargs):
        raise AssertionError("Plot the native result rather than refitting its peaks")

    monkeypatch.setattr(type(native), "analyze_peaks", forbid_peak_approximation)
    result = DrtTool().run([], {"method": method})
    drt_axis, _nyquist, real_axis, imaginary_axis = result.figures["DRT analysis"].axes
    drt_data = native.get_drt_data()
    expected = ([(drt_data[0], drt_data[1]), (drt_data[0], drt_data[2])] if method == "bht"
                else [(drt_data[0], drt_data[1]), (drt_data[2], drt_data[3])] if method == "lm"
                else [(drt_data[0], drt_data[1])])
    assert len(drt_axis.lines) == len(expected)
    for line, (tau, gamma) in zip(drt_axis.lines, expected):
        np.testing.assert_array_equal(line.get_xdata(), tau)
        np.testing.assert_array_equal(line.get_ydata(), gamma)
    assert "Ω·cm²" in drt_axis.get_ylabel()
    assert all("ohm" not in label for label in result.tables["DRT peaks"].columns)
    np.testing.assert_array_equal(real_axis.lines[0].get_ydata(), np.ones(len(spectrum[0])))
    np.testing.assert_array_equal(imaginary_axis.lines[0].get_ydata(), -2 * np.ones(len(spectrum[0])))
    assert real_axis.get_ylabel() == "Real residual (%)"
    assert imaginary_axis.get_ylabel() == "Imaginary residual (%)"
    if method == "bht":
        assert result.scalars["Peaks found"] == 3
        assert "Real:" in result.text_blocks["Identified time constants"]
        assert "Imaginary:" in result.text_blocks["Identified time constants"]
        assert "nan" not in result.text_blocks["Identified time constants"]
    elif method == "lm":
        assert result.scalars["Peaks found"] == 5
        assert "RC:" in result.text_blocks["Identified time constants"]
        assert "RL:" in result.text_blocks["Identified time constants"]


def test_circuit_dependent_method_is_not_offered(monkeypatch, spectrum):
    _native, calls = _stub_solver(monkeypatch, "tr-nnls", spectrum)
    method = next(spec for spec in DrtTool().parameter_schema() if spec.key == "method")
    assert "mrq-fit" not in method.choices
    with pytest.raises(ValueError, match="supported DRT method"):
        DrtTool().run([], {"method": "mrq-fit"})
    assert not calls


@pytest.mark.parametrize("method", ["tr-nnls", "tr-rbf", "bht", "lm"])
def test_workbook_exports_native_distribution_impedance_and_percent_residuals(
        monkeypatch, spectrum, method):
    native, _calls = _stub_solver(monkeypatch, method, spectrum)
    tool = DrtTool()
    result = tool.run([], {"method": method})

    sheets = next(artifact.payload for artifact in tool.exportable_artifacts(result)
                  if artifact.kind == "xlsx")

    assert set(sheets) == {"DRT distribution", "DRT peaks", "DRT statistics",
                            "Measured impedance", "Residuals"}
    assert sheets["DRT statistics"].set_index("Label").loc["DRT method", "Value"] == method
    distribution = sheets["DRT distribution"]
    axis = result.figures["DRT analysis"].axes[0]
    for offset, line in enumerate(axis.lines):
        tau, gamma = distribution.iloc[:, 2 * offset:2 * offset + 2].dropna().to_numpy().T
        np.testing.assert_array_equal(tau, line.get_xdata())
        np.testing.assert_array_equal(gamma, line.get_ydata())
    measured = sheets["Measured impedance"]
    np.testing.assert_array_equal(measured["Frequency (Hz)"], spectrum[0])
    np.testing.assert_array_equal(measured["Re(Z) (Ω·cm²)"], spectrum[1].real)
    np.testing.assert_array_equal(measured["Im(Z) (Ω·cm²)"], spectrum[1].imag)
    frequency, real, imag = native.get_residuals_data()
    np.testing.assert_array_equal(sheets["Residuals"]["Frequency (Hz)"], frequency)
    np.testing.assert_array_equal(sheets["Residuals"]["Real residual (%)"], real)
    np.testing.assert_array_equal(sheets["Residuals"]["Imaginary residual (%)"], imag)


def test_default_analysis_recovers_synthetic_rc_peak_and_renders(tmp_path, spectrum):
    frequencies, impedance = spectrum
    source = tmp_path / "rc.csv"
    np.savetxt(source, np.column_stack((frequencies, impedance.real, impedance.imag)),
               delimiter=",", header="Frequency(Hz),Zre(Ohm.cm2),Zim(Ohm.cm2)", comments="")
    result = DrtTool().run([source], {})
    assert result.scalars["DRT method"] == "tr-nnls"
    assert result.scalars["Data points"] == len(frequencies)
    peaks = result.tables["DRT peaks"]
    highest = peaks.loc[peaks["gamma (Ω·cm²)"].idxmax(), "tau (s)"]
    assert abs(np.log10(highest / 0.01)) < 0.25
    figure = result.figures["DRT analysis"]
    image = BytesIO()
    figure.savefig(image, format="png")
    assert image.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
