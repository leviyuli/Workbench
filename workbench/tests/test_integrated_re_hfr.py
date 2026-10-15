"""Integrated RE HFR uses independent measured intersections, without fitting."""

import numpy as np
import pytest

from workbench.core.data_io import read_eis_data
from workbench.tools.eis.logic_eis import EisLogic
from workbench.tools.integrated_re.hfr import (
    ESTIMATE_COLUMNS, FALLBACK_METHOD, INTERSECTION_METHOD, SPECTRUM_COLUMNS,
    extract_role_hfr,
)


def write_eis(path, rows, *, unit="Ohm.cm2", minus_imag=False):
    imag_header = "-Im(Z)" if minus_imag else "Im(Z)"
    path.write_text(
        f"Frequency (Hz),Re(Z) ({unit}),{imag_header} ({unit})\n"
        + "".join(f"{freq},{zr},{-zi if minus_imag else zi}\n"
                  for freq, zr, zi in rows),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("reverse_files", [False, True])
@pytest.mark.parametrize("order", [[0, 1, 2], [2, 0, 1]])
def test_average_is_unweighted_mean_of_file_intercepts(tmp_path, order, reverse_files):
    # Intercepts are 0.3 and 0.8: their mean is 0.55, whereas averaging
    # impedances on the shared frequency grid first would give 17/30.
    first = [(10000, 0.1, 0.1), (1000, 0.5, -0.1), (100, 0.9, -0.6)]
    second = [(10000, 0.5, 0.3), (1000, 0.9, -0.1), (100, 1.3, -0.6)]
    files = [write_eis(tmp_path / "first.csv", [first[i] for i in order]),
             write_eis(tmp_path / "second.csv", [second[i] for i in order]
                       + [(10, 1.8, -0.7)])]
    if reverse_files:
        files.reverse()
    hfr, estimates, spectra, warnings = extract_role_hfr(files)

    assert hfr == pytest.approx(0.55)
    assert hfr != pytest.approx(17 / 30)
    assert estimates["HFR (Ω·cm²)"].tolist() == pytest.approx(
        [0.8, 0.3] if reverse_files else [0.3, 0.8])
    assert estimates["File"].tolist() == [file.name for file in files]
    assert estimates["Status"].tolist() == ["OK", "OK"]
    assert estimates["HFR method"].tolist() == [INTERSECTION_METHOD] * 2
    assert list(estimates.columns) == ESTIMATE_COLUMNS
    assert list(spectra.columns) == SPECTRUM_COLUMNS
    assert not warnings
    for file in files:
        freq = spectra.loc[spectra["File"] == file.name, "Frequency (Hz)"]
        assert np.all(np.diff(freq) < 0)


@pytest.mark.parametrize("rows,expected", [
    ([(1000, 0.4, 0.0), (100, 0.7, -0.2)], 0.4),
    ([(1000, 0.1, 0.3), (100, 0.5, -0.1)], 0.4),
    ([(1000, 0.7, 0.2), (100, 0.9, -0.2),
      (10, 0.1, -0.1), (1, 0.3, 0.1)], 0.8),
])
def test_highest_frequency_crossing_with_as_few_as_two_points(tmp_path, rows, expected):
    file = write_eis(tmp_path / "spectrum.csv", rows)
    hfr, estimates, _, _ = extract_role_hfr([file])
    assert hfr == pytest.approx(expected)
    assert estimates.iloc[0]["HFR method"] == INTERSECTION_METHOD


def test_fallback_is_explicit_and_warns_with_filename(tmp_path):
    file = write_eis(tmp_path / "no_crossing.csv",
                     [(1000, 0.5, -0.2), (100, 0.3, -0.5)])
    hfr, estimates, _, warnings = extract_role_hfr([file], role="cathode")
    assert hfr == pytest.approx(0.3)
    assert estimates.iloc[0]["HFR method"] == FALLBACK_METHOD
    assert any("no_crossing.csv" in warning and "fallback" in warning
               for warning in warnings)


def test_raw_ohm_area_conversion_and_metadata_preserve_sign(tmp_path):
    file = write_eis(tmp_path / "raw.csv",
                     [(1000, 0.2, 0.2), (100, 0.6, -0.2)],
                     unit="Ohm", minus_imag=True)
    data, error = read_eis_data(file)
    assert data is None
    assert "raw Ω" in error

    data, error = read_eis_data(file, area_cm2=2.0)
    assert error is None
    assert data["source_impedance_unit"] == "ohm"
    assert data["impedance_unit"] == "ohm_cm2"
    assert data["area_cm2"] == 2.0
    assert data["imaginary_convention"] == "-Im(Z)"
    assert data["z_real"] == pytest.approx([0.4, 1.2])
    assert data["z_imag"] == pytest.approx([0.4, -0.4])
    assert data["z_complex"] == pytest.approx([0.4 + 0.4j, 1.2 - 0.4j])
    hfr, _, spectra, _ = extract_role_hfr([file], area_cm2=2.0)
    assert hfr == pytest.approx(0.8)
    assert spectra["Im(Z) (Ω·cm²)"].tolist() == pytest.approx([0.4, -0.4])


@pytest.mark.parametrize("area", [0, -1, np.inf, np.nan, "invalid"])
def test_raw_ohm_requires_valid_area_only_when_needed(tmp_path, area):
    rows = [(1000, 0.2, 0.2), (100, 0.6, -0.2)]
    raw = write_eis(tmp_path / "raw.csv", rows, unit="Ohm")
    data, error = read_eis_data(raw, area_cm2=area)
    assert data is None
    assert "finite electrode area greater than zero" in error

    normalized = write_eis(tmp_path / "normalized.csv", rows)
    data, error = read_eis_data(normalized, area_cm2=area)
    assert error is None
    assert data["z_real"] == pytest.approx([0.2, 0.6])
    assert data["source_impedance_unit"] == "ohm_cm2"
    assert data["area_cm2"] is None


def test_unknown_units_keep_existing_assumption_without_area_multiplication(tmp_path):
    file = tmp_path / "headerless.txt"
    file.write_text("1000 0.2 0.2\n100 0.6 -0.2\n", encoding="utf-8")
    data, error = read_eis_data(file, area_cm2=5)
    assert error is None
    assert data["impedance_unit"] == "unknown"
    assert data["z_real"] == pytest.approx([0.2, 0.6])
    assert "assumed" in data["warning"]
    hfr, _, _, warnings = extract_role_hfr([file], area_cm2=5)
    assert hfr == pytest.approx(0.4)
    assert any("headerless.txt" in warning and "assumed" in warning
               for warning in warnings)


def test_finite_cleanup_and_only_exact_duplicate_frequency_averaging(tmp_path):
    file = write_eis(tmp_path / "duplicates.csv", [
        (100, 0.9, -0.6), (1000, 0.4, -0.1), (10000, 0.1, 0.2),
        (1000, 0.6, 0.1), (1000.00001, 0.45, 0.05),
        (0, 10, -2), (-10, 5, 1), (np.inf, 1, 1),
    ])
    hfr, estimates, spectra, warnings = extract_role_hfr([file])
    assert hfr == pytest.approx(0.5)
    assert estimates.iloc[0]["Points"] == 4
    assert spectra["Frequency (Hz)"].tolist() == [10000, 1000.00001, 1000, 100]
    assert spectra["Re(Z) (Ω·cm²)"].tolist() == pytest.approx([0.1, 0.45, 0.5, 0.9])
    assert any("duplicates.csv" in warning and "dropped 3" in warning
               for warning in warnings)
    assert any("duplicates.csv" in warning and "duplicate frequency" in warning
               for warning in warnings)


def test_partial_failure_keeps_rows_and_averages_only_successes(tmp_path):
    valid = write_eis(tmp_path / "good.csv", [(1000, 0.2, 0.2), (100, 0.6, -0.2)])
    bad = tmp_path / "broken.csv"
    bad.write_text("not,eis\nno,data\n", encoding="utf-8")
    negative = write_eis(tmp_path / "negative.csv",
                         [(1000, -0.6, 0.2), (100, -0.2, -0.2)])
    single = write_eis(tmp_path / "single.csv", [(1000, 0.2, 0)])
    hfr, estimates, _, warnings = extract_role_hfr([bad, valid, negative, single], role="cell")
    assert hfr == pytest.approx(0.4)
    assert estimates["Status"].tolist() == ["Failed", "OK", "Failed", "Failed"]
    assert estimates["Role"].tolist() == ["cell"] * 4
    assert "nonnegative" in estimates.iloc[2]["Error"]
    assert "need at least 2" in estimates.iloc[3]["Error"]
    for failed in (bad, negative, single):
        assert any(failed.name in warning for warning in warnings)


def test_all_failed_is_role_specific_and_no_files_is_optional(tmp_path):
    hfr, estimates, spectra, warnings = extract_role_hfr([])
    assert hfr is None and estimates.empty and spectra.empty and not warnings
    assert list(estimates.columns) == ESTIMATE_COLUMNS
    assert list(spectra.columns) == SPECTRUM_COLUMNS

    failed = tmp_path / "missing.csv"
    with pytest.raises(ValueError, match="cathode EIS") as exc:
        extract_role_hfr([failed], role="cathode")
    assert "missing.csv" in str(exc.value)


def test_hfr_does_not_invoke_kk_fit_or_cross_spectrum_averaging(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Simple Integrated RE HFR must not perform KK, fitting, or averaging spectra")

    for method in ("__init__", "validate_spectra", "process_spectra", "average_spectra"):
        monkeypatch.setattr(EisLogic, method, forbidden)
    file = write_eis(tmp_path / "no_fit.csv", [(1000, 0.2, 0.2), (100, 0.6, -0.2)])
    hfr, _, _, _ = extract_role_hfr([file])
    assert hfr == pytest.approx(0.4)


@pytest.mark.parametrize("unit,scale", [
    ("mOhm", 1e-3), ("mΩ", 1e-3),
    ("uOhm", 1e-6), ("µΩ", 1e-6), ("μΩ", 1e-6),
    ("kOhm", 1e3), ("kΩ", 1e3),
    ("MOhm", 1e6), ("MΩ", 1e6),
])
def test_raw_prefixed_impedance_scaled_before_area_and_exported(tmp_path, unit, scale):
    file = write_eis(tmp_path / "prefixed.csv", [
        (1000, 0.2 / scale, 0.2 / scale),
        (100, 0.6 / scale, -0.2 / scale),
    ], unit=unit)
    data, error = read_eis_data(file, area_cm2=2.0)
    assert error is None
    assert data["z_real"] == pytest.approx([0.4, 1.2])
    assert data["z_imag"] == pytest.approx([0.4, -0.4])
    assert data["source_impedance_scale_real"] == scale
    assert data["source_impedance_scale_imag"] == scale

    hfr, estimates, _, _ = extract_role_hfr([file], area_cm2=2)
    assert hfr == pytest.approx(0.8)
    row = estimates.iloc[0]
    assert row["Source impedance unit"] == "ohm"
    assert row["Area applied (cm²)"] == 2
    assert row["Real impedance scale"] == scale
    assert row["Imaginary impedance scale"] == scale


@pytest.mark.parametrize("unit,scale", [("mOhm.cm2", 1e-3), ("MΩ·cm²", 1e6)])
def test_normalized_prefixed_impedance_scales_once_without_area(tmp_path, unit, scale):
    file = write_eis(tmp_path / "normalized_prefix.csv", [
        (1000, 0.2 / scale, 0.2 / scale),
        (100, 0.6 / scale, -0.2 / scale),
    ], unit=unit)
    hfr, estimates, _, _ = extract_role_hfr([file], area_cm2=5)
    assert hfr == pytest.approx(0.4)
    row = estimates.iloc[0]
    assert row["Source impedance unit"] == "ohm_cm2"
    assert np.isnan(row["Area applied (cm²)"])
    assert row["Real impedance scale"] == scale


def test_distinct_real_imaginary_prefixes_are_scaled_independently(tmp_path):
    file = tmp_path / "mixed_prefixes.csv"
    file.write_text(
        "Frequency (Hz),Re(Z) (mOhm),-Im(Z) (µOhm)\n"
        "1000,200,-200000\n100,600,200000\n", encoding="utf-8",
    )
    hfr, estimates, spectra, _ = extract_role_hfr([file], area_cm2=2)
    assert hfr == pytest.approx(0.8)
    assert spectra["Re(Z) (Ω·cm²)"].tolist() == pytest.approx([0.4, 1.2])
    assert spectra["Im(Z) (Ω·cm²)"].tolist() == pytest.approx([0.4, -0.4])
    assert estimates.iloc[0]["Real impedance scale"] == 1e-3
    assert estimates.iloc[0]["Imaginary impedance scale"] == 1e-6


def test_unsupported_impedance_prefix_is_not_silently_treated_as_ohms(tmp_path):
    file = write_eis(tmp_path / "unsupported_prefix.csv",
                     [(1000, 0.2, 0.2), (100, 0.6, -0.2)], unit="fooOhm.cm2")
    data, error = read_eis_data(file)
    assert data is None
    assert "Unsupported impedance unit prefix 'foo'" in error


@pytest.mark.parametrize("delimiter,suffix", [(",", ".csv"), ("\t", ".txt")])
def test_bare_column_name_before_spaced_ohm_unit_is_not_a_prefix(tmp_path, delimiter, suffix):
    file = tmp_path / f"bare_headers{suffix}"
    rows = [("Frequency (Hz)", "Zre Ohm.cm2", "Zim Ohm.cm2"),
            (1000, 0.2, 0.2), (100, 0.6, -0.2)]
    file.write_text(
        "\n".join(delimiter.join(map(str, row)) for row in rows) + "\n",
        encoding="utf-8",
    )
    data, error = read_eis_data(file)
    assert error is None
    assert data["z_real"] == pytest.approx([0.2, 0.6])
    assert data["z_imag"] == pytest.approx([0.2, -0.2])
    assert data["source_impedance_scale_real"] == 1.0
    assert data["source_impedance_scale_imag"] == 1.0
    hfr, _, _, _ = extract_role_hfr([file])
    assert hfr == pytest.approx(0.4)
