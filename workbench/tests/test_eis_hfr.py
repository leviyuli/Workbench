"""KK HFR estimates from measured Nyquist intersections and their fallback."""

import numpy as np
import pytest

import workbench.tools.eis.logic_eis as eis_logic_module
from workbench.tools.eis.tool import EisTool


@pytest.fixture(autouse=True)
def passing_kk(monkeypatch):
    class KkResult:
        def __init__(self, dataset):
            self.dataset = dataset

        def get_residuals_data(self):
            frequency = self.dataset.get_frequencies()
            return frequency, np.zeros(len(frequency)), np.zeros(len(frequency))

    monkeypatch.setattr(
        eis_logic_module.pyimpspec, "perform_kramers_kronig_test",
        lambda dataset, **kwargs: KkResult(dataset),
    )


def write_spectrum(path, rows):
    path.write_text(
        "Frequency (Hz),Re(Z) (Ohm.cm2),-Im(Z) (Ohm.cm2)\n"
        + "".join(f"{f},{zr},{-zi}\n" for f, zr, zi in rows),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    "rows, expected, intersection_found",
    [
        pytest.param(
            [(10000, 0.1, 0.3), (1000, 0.5, -0.1), (100, 0.9, -0.6)],
            0.4, True, id="interpolate-crossing",
        ),
        pytest.param(
            [(10000, 0.1, 0.3), (1000, 0.45, 0.0), (100, 0.9, -0.6)],
            0.45, True, id="exact-on-axis-point",
        ),
        pytest.param(
            [(10000, 0.25, 0.0), (1000, 0.5, 0.0), (100, 0.9, -0.6)],
            0.25, True, id="on-axis-segment-high-frequency-end",
        ),
        pytest.param(
            [(10000, 0.7, 0.2), (1000, 0.9, -0.2),
             (100, 0.1, -0.1), (10, 0.3, 0.1)],
            0.8, True, id="highest-frequency-of-multiple-crossings",
        ),
        pytest.param(
            [(10000, 0.1, -0.3), (1000, 0.5, -0.1), (100, 0.9, -0.6)],
            0.1, False, id="no-crossing-capacitive",
        ),
        pytest.param(
            [(10000, 0.1, 0.3), (1000, 0.5, 0.1), (100, 0.9, 0.6)],
            0.1, False, id="no-crossing-inductive",
        ),
        pytest.param(
            [(10000, 0.3, -1e-12), (1000, 0.1, -2e-12), (100, 0.9, -0.6)],
            0.1, False, id="near-zero-does-not-create-intersection",
        ),
    ],
)
def test_kk_hfr_intersection_and_fallback(tmp_path, rows, expected, intersection_found):
    source = write_spectrum(tmp_path / "spectrum.csv", rows)

    result = EisTool().run([source], {"mode": "KK test"})

    assert result.scalars["HFR estimate (Ω·cm²)"] == pytest.approx(expected)
    assert result.param_updates["hfr"] == pytest.approx(round(expected, 5))
    method = result.scalars["HFR method"]
    if intersection_found:
        assert method == "High-frequency x-axis intersection"
    else:
        assert "no x-axis intersection" in method


@pytest.mark.parametrize("order", [[0, 1, 2], [2, 1, 0], [1, 0, 2]])
def test_kk_hfr_uses_averaged_spectrum_regardless_of_input_order(tmp_path, order):
    # Separate scans cross at 0.3 and 0.8; their averaged curve crosses at 17/30.
    # Averaging the individual intercepts would incorrectly give 0.55.
    first = [(10000, 0.1, 0.1), (1000, 0.5, -0.1), (100, 0.9, -0.6)]
    second = [(10000, 0.5, 0.3), (1000, 0.9, -0.1), (100, 1.3, -0.6)]
    sources = [
        write_spectrum(tmp_path / "first.csv", [first[i] for i in order]),
        write_spectrum(tmp_path / "second.csv", [second[i] for i in order]),
    ]

    result = EisTool().run(sources, {"mode": "KK test"})

    assert result.scalars["HFR estimate (Ω·cm²)"] == pytest.approx(17 / 30)
    assert result.param_updates["hfr"] == pytest.approx(0.56667)


def test_kk_hfr_ignores_crossings_outside_selected_window(tmp_path):
    source = write_spectrum(tmp_path / "spectrum.csv", [
        (10000, 0.1, 0.3), (1000, 0.5, -0.1),
        (100, 0.9, -0.6), (10, 1.3, -0.8),
    ])

    result = EisTool().run([source], {"mode": "KK test", "f_max": 1000.0})

    assert result.scalars["HFR estimate (Ω·cm²)"] == pytest.approx(0.5)
    assert result.param_updates["hfr"] == pytest.approx(0.5)
    assert "no x-axis intersection" in result.scalars["HFR method"]
