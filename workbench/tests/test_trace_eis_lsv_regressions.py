"""Regression tests for data-integrity defects found during the audit."""

import numpy as np
import pandas as pd
import pytest

from workbench.tools.eis.logic_eis import EisLogic
from workbench.tools.eis.tool import EisTool
from workbench.tools.it_integrate.logic import integrate_potentiostatic_tests
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
