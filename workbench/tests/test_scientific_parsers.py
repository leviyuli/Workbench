"""Regression tests for electrochemistry-specific import semantics."""

import numpy as np
import pandas as pd
import pytest

from workbench.core.data_io import read_eis_data, read_lsv_data
from workbench.tools.ecsa.logic_cv import (
    detect_cycle_bounds,
    process_one_file,
    robust_read_cv_data,
)
from workbench.tools.lsv.tool import _as_current_density


@pytest.mark.parametrize(
    ("header", "raw", "expected"),
    [
        ("Current (A)", 2.0, 2.0),
        ("Current (mA)", 2.0, 2e-3),
        ("Current (µA)", 2.0, 2e-6),
        ("Current (μA)", 2.0, 2e-6),
    ],
)
def test_lsv_scales_absolute_current_and_marks_it_non_density(
        tmp_path, header, raw, expected):
    path = tmp_path / "lsv.csv"
    path.write_text(f"E (V),{header}\n0.1,{raw}\n", encoding="utf-8")

    parsed, err = read_lsv_data([str(path)], reaction="oer")

    assert err is None
    voltage, current, info = parsed
    assert np.allclose(voltage, [0.1])
    assert np.allclose(current, [expected])
    assert info["is_density"] is False
    assert "absolute current" in info["warning"]


def test_lsv_scales_explicit_current_density(tmp_path):
    path = tmp_path / "lsv_density.csv"
    path.write_text(
        "Potential (mV),Current Density (µA/cm²)\n100,250\n",
        encoding="utf-8",
    )

    parsed, err = read_lsv_data([str(path)], reaction="oer")

    assert err is None
    voltage, current, info = parsed
    assert np.allclose(voltage, [0.1])
    assert np.allclose(current, [250e-6])
    assert info["is_density"] is True
    assert "warning" not in info


def test_lsv_scales_bracketed_units_and_inverse_square_notation(tmp_path):
    path = tmp_path / "lsv_density.csv"
    path.write_text(
        "Potential [mV],i [mA cm^-2]\n100,2\n",
        encoding="utf-8",
    )

    parsed, err = read_lsv_data([str(path)], reaction="oer")

    assert err is None
    voltage, current, info = parsed
    assert np.allclose(voltage, [0.1])
    assert np.allclose(current, [2e-3])
    assert info["is_density"] is True


def test_lsv_reads_whitespace_headers_with_spaced_units(tmp_path):
    path = tmp_path / "lsv.txt"
    path.write_text(
        "time (s) Potential [mV] Current Density [mA cm^-2]\n"
        "0 100 2\n1 200 3\n",
        encoding="utf-8",
    )

    parsed, err = read_lsv_data([str(path)], reaction="oer")

    assert err is None
    voltage, current, info = parsed
    assert np.allclose(voltage, [0.1, 0.2])
    assert np.allclose(current, [2e-3, 3e-3])
    assert info["is_density"] is True


def test_lsv_absolute_current_requires_area_before_analysis():
    info = {"is_density": False, "i_col": "Current (mA)"}

    with pytest.raises(ValueError, match="positive electrode area"):
        _as_current_density(np.array([2e-3]), info, 0.0)

    normalized, warning = _as_current_density(
        np.array([2e-3, 4e-3]), info, 2.0,
    )
    assert np.allclose(normalized, [1e-3, 2e-3])
    assert "divided by the 2 cm² electrode area" in warning


def test_lsv_rejects_mixed_absolute_and_density_batches(tmp_path):
    absolute = tmp_path / "absolute.csv"
    density = tmp_path / "density.csv"
    absolute.write_text("E (V),Current (mA)\n0.1,2\n", encoding="utf-8")
    density.write_text("E (V),j (mA/cm²)\n0.1,2\n", encoding="utf-8")

    parsed, err = read_lsv_data([str(absolute), str(density)], reaction="oer")

    assert parsed is None
    assert "mixture of absolute-current and current-density" in err


def test_lsv_excel_uses_the_same_header_and_unit_detection(tmp_path):
    path = tmp_path / "lsv.xlsx"
    pd.DataFrame([
        ["metadata", None],
        ["E (V)", "Current (mA)"],
        [0.1, 2.0],
    ]).to_excel(path, header=False, index=False)

    parsed, err = read_lsv_data([str(path)], reaction="oer")

    assert err is None
    assert np.allclose(parsed[1], [2e-3])
    assert parsed[2]["is_density"] is False


@pytest.mark.parametrize(
    "header",
    [
        "Freq (Hz),Z' (Ohm.cm2),-Z'' (Ohm.cm2)",
        "Frequency (Hz),Re(Z) (Ω·cm²),−Im(Z) (Ω·cm²)",
    ],
)
def test_eis_converts_displayed_negative_imaginary_to_im_z(tmp_path, header):
    path = tmp_path / "eis.csv"
    path.write_text(f"{header}\n100,1,2\n10,3,4\n", encoding="utf-8")

    data, err = read_eis_data(str(path))

    assert err is None
    assert data["impedance_unit"] == "ohm_cm2"
    assert data["imaginary_convention"] == "-Im(Z)"
    assert np.allclose(data["z_imag"], [-2.0, -4.0])
    assert np.allclose(data["z_complex"], [1 - 2j, 3 - 4j])


def test_eis_does_not_relabel_raw_ohms_as_area_normalized(tmp_path):
    path = tmp_path / "raw_ohm.csv"
    path.write_text(
        "Frequency (Hz),Re(Z) (Ohm),-Im(Z) (Ohm)\n100,1,2\n",
        encoding="utf-8",
    )

    data, err = read_eis_data(str(path))

    assert data is None
    assert "raw Ω" in err
    assert "electrode area" in err


def test_eis_reads_whitespace_headers_with_spaced_units_and_zre_zim(tmp_path):
    path = tmp_path / "eis.txt"
    path.write_text(
        "Freq (Hz) Zre (Ohm cm2) -Zim (Ohm cm2)\n100 1 2\n10 3 4\n",
        encoding="utf-8",
    )

    data, err = read_eis_data(str(path))

    assert err is None
    assert np.allclose(data["z_complex"], [1 - 2j, 3 - 4j])


def test_eis_excel_applies_negative_imaginary_header_semantics(tmp_path):
    path = tmp_path / "eis.xlsx"
    pd.DataFrame([
        ["Frequency (Hz)", "Re(Z) (Ohm.cm2)", "-Im(Z) (Ohm.cm2)"],
        [100, 1, 2],
    ]).to_excel(path, header=False, index=False)

    data, err = read_eis_data(str(path))

    assert err is None
    assert np.allclose(data["z_complex"], [1 - 2j])


def test_eis_headerless_input_discloses_unit_assumption(tmp_path):
    path = tmp_path / "headerless.txt"
    path.write_text("100 1 -2\n10 3 -4\n", encoding="utf-8")

    data, err = read_eis_data(str(path))

    assert err is None
    assert data["impedance_unit"] == "unknown"
    assert "assumed" in data["warning"]


def test_ecsa_selects_named_columns_and_scales_mv_and_ma(tmp_path):
    path = tmp_path / "CV_50mV_1.txt"
    path.write_text(
        "time (s),Ewe (mV),Current Density (mA/cm2)\n"
        "0,0,1\n1,100,2\n2,0,-1\n",
        encoding="utf-8",
    )

    data = robust_read_cv_data(str(path))

    assert np.allclose(data["E_raw"], [0.0, 0.1, 0.0])
    assert np.allclose(data["i"], [1e-3, 2e-3, -1e-3])
    assert data.attrs["potential_column"] == "Ewe (mV)"
    assert data.attrs["current_is_density"] is True

    cycles = process_one_file(
        str(path), re_type="SHE", target_scale="SHE", ph=0,
    )
    assert list(cycles) == [0]
    assert np.allclose(cycles[0]["E"], [0.0, 0.1, 0.0])


def test_ecsa_scales_microamp_current_density_column_after_time(tmp_path):
    path = tmp_path / "CV_25mV_1.txt"
    path.write_text(
        "time/s\tEwe/V\ti(µA/cm2)\n0\t0.0\t10\n1\t0.1\t20\n2\t0.0\t-10\n",
        encoding="utf-8",
    )

    data = robust_read_cv_data(str(path))

    assert np.allclose(data["E_raw"], [0.0, 0.1, 0.0])
    assert np.allclose(data["i"], [10e-6, 20e-6, -10e-6])
    assert data.attrs["current_column"] == "i(µA/cm2)"
    assert data.attrs["current_is_density"] is True


def test_ecsa_recognizes_inverse_square_density_notation(tmp_path):
    path = tmp_path / "CV_25mV_1.txt"
    path.write_text(
        "Ewe/V\ti(mA cm^-2)\n0.0\t1\n0.1\t2\n0.0\t-1\n",
        encoding="utf-8",
    )

    data = robust_read_cv_data(str(path))

    assert np.allclose(data["i"], [1e-3, 2e-3, -1e-3])
    assert data.attrs["current_is_density"] is True


def test_ecsa_rejects_absolute_current_without_electrode_area(tmp_path):
    path = tmp_path / "CV_25mV_1.txt"
    path.write_text(
        "time/s\tEwe/V\t<I>/µA\n0\t0.0\t10\n1\t0.1\t20\n2\t0.0\t-10\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires current density") as exc_info:
        robust_read_cv_data(str(path))
    assert "electrode area" in str(exc_info.value)


def test_cycle_detection_accepts_endpoint_bounded_one_and_two_cycles():
    one_cycle = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
    two_cycles = np.array([0.0, 0.5, 1.0, 0.5, 0.0,
                           0.5, 1.0, 0.5, 0.0])

    assert detect_cycle_bounds(one_cycle) == [(0, 4)]
    assert detect_cycle_bounds(two_cycles) == [(0, 4), (4, 8)]


def test_cycle_detection_accepts_upper_vertex_endpoints():
    cycle = np.array([1.0, 0.5, 0.0, 0.5, 1.0])
    assert detect_cycle_bounds(cycle) == [(0, 4)]
