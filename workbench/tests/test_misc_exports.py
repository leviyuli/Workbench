"""Spectrum and ink exports keep final curves and recipes without duplicate data."""

import numpy as np
import pandas as pd
import pytest

from workbench.tools.ftir.tool import FtirTool
from workbench.tools.ink_calc.tool import InkCalcTool
from workbench.widgets.export_button import ExportButton


def _save_and_check(workbook, folder):
    target = folder / workbook.default_filename
    ExportButton._write(workbook, str(target))
    assert pd.ExcelFile(target).sheet_names == list(workbook.payload)
    for name, expected in workbook.payload.items():
        pd.testing.assert_frame_equal(pd.read_excel(target, sheet_name=name), expected,
                                      check_dtype=False, atol=1e-12, rtol=1e-12)


def _ftir_inputs(folder):
    folder.mkdir(parents=True, exist_ok=True)
    first, second = folder / "transmission.csv", folder / "absorbance.txt"
    first.write_text("cm-1,T%\n4000,100\n2500,10\n1600,0\n800,50\n", encoding="utf-8")
    second.write_text("cm-1\tA\n3500\t0.1\n1800\t0.4\n600\t0.2\n", encoding="utf-8")
    return [first, second]


def _xps_inputs(folder):
    folder.mkdir(parents=True, exist_ok=True)
    first, second = folder / "c1s.csv", folder / "survey.csv"
    first.write_text("Binding Energy (eV),Counts,Background Counts\n280,20,0\n285,60,16\n290,100,0\n",
                     encoding="utf-8")
    second.write_text("Binding Energy (eV),Counts,Background Counts\n0,20,10\n300,40,14\n600,60,18\n1200,80,22\n",
                      encoding="utf-8")
    return [first, second]


def _peak_input(folder, technique, *, flat=False):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{technique.lower()}_peaks.csv"
    if technique == "FTIR":
        x = np.linspace(500, 4000, 1001)
        y = 95 - 30 * np.exp(-0.5 * ((x - 1710) / 30) ** 2)
        headers = ["Wavenumber (cm-1)", "T%"]
    else:
        x = np.linspace(280, 300, 1001)
        y = 20 + 30 * np.exp(-0.5 * ((x - 285) / 0.7) ** 2)
        headers = ["Binding Energy (eV)", "Counts"]
    if flat:
        y = np.full_like(x, 20.0)
    pd.DataFrame(dict(zip(headers, [x, y]))).to_csv(path, index=False)
    return path


def _ink_params(mode="area", *, ionomers=True, extras=True):
    return {
        "mode": mode, "length": 5.0, "width": 4.0, "loading": 0.5,
        "avail_cat": 12.5, "ink_density": 2.0,
        "ionomers": ([{"ratio": 20.0, "conc": 5.0, "density": 0.78},
                      {"ratio": 10.0, "conc": 10.0, "density": 0.9}]
                     if ionomers else []),
        "extras": ([{"name": "Carbon", "ratio": 10.0}, {"name": "PTFE", "ratio": 5.0}]
                   if extras else []),
    }


def test_ftir_workbook_exports_native_xy_and_nan_gaps_over_full_range(tmp_path):
    tool = FtirTool()
    result = tool.run(_ftir_inputs(tmp_path), {
        "technique": "FTIR", "display_mode": "A", "normalization": "None",
        "vertical_offset": 0.25, "custom_x_range": True,
        "x_display_min": 1000, "x_display_max": 2000,
    })
    metadata = result.tables["FTIR metadata"].copy(deep=True)
    artifacts = tool.exportable_artifacts(result)
    assert [artifact.kind for artifact in artifacts] == ["csv", "xlsx"]
    workbook = artifacts[1]
    assert list(workbook.payload) == ["FTIR plot data"]
    exported = workbook.payload["FTIR plot data"]
    assert exported.shape == (4, 4)
    axes = result.figures["FTIR spectra"].axes[0]
    for offset, line in zip((0, 2), axes.lines):
        size = len(line.get_xdata())
        np.testing.assert_allclose(exported.iloc[:size, offset], line.get_xdata())
        np.testing.assert_allclose(exported.iloc[:size, offset + 1], line.get_ydata(), equal_nan=True)
        assert exported.iloc[size:, offset:offset + 2].isna().all().all()
    assert exported.iloc[:, 0].min() < 1000 and exported.iloc[:, 0].max() > 2000
    assert exported.iloc[:, 1].isna().sum() == 1
    np.testing.assert_allclose(exported.iloc[:, 0], [800, 1600, 2500, 4000])
    pd.testing.assert_frame_equal(artifacts[0].payload, exported)
    _save_and_check(workbook, tmp_path)
    pd.testing.assert_frame_equal(result.tables["FTIR metadata"], metadata)


def test_xps_workbook_retains_backgrounds_and_independent_native_grids(tmp_path):
    tool = FtirTool()
    result = tool.run(_xps_inputs(tmp_path), {
        "technique": "XPS", "normalization": "Min-max normalization", "vertical_offset": 1.0,
        "custom_x_range": True, "x_display_min": 282, "x_display_max": 288,
    })
    originals = {name: frame.copy(deep=True) for name, frame in result.tables.items()}
    artifacts = tool.exportable_artifacts(result)
    workbook = next(item for item in artifacts if item.kind == "xlsx")
    assert list(workbook.payload) == ["XPS plot data"]
    table = workbook.payload["XPS plot data"]
    assert table.shape == (4, 6)
    lines = result.figures["XPS spectra"].axes[0].lines
    for offset, sample, background in ((0, lines[0], lines[1]), (3, lines[2], lines[3])):
        size = len(sample.get_xdata())
        np.testing.assert_allclose(table.iloc[:size, offset], sample.get_xdata())
        np.testing.assert_allclose(table.iloc[:size, offset + 1], sample.get_ydata())
        np.testing.assert_allclose(table.iloc[:size, offset + 2], background.get_ydata(), equal_nan=True)
        assert table.iloc[size:, offset:offset + 3].isna().all().all()
    assert table["survey Binding energy (eV)"].tolist() == [0, 300, 600, 1200]
    assert table["c1s Background display"].iloc[[0, 2]].isna().all()
    _save_and_check(workbook, tmp_path)
    for name, original in originals.items():
        pd.testing.assert_frame_equal(result.tables[name], original)


@pytest.mark.parametrize("technique,physical_width", [("FTIR", "Peak width (cm-1)"), ("XPS", "Peak width (eV)")])
@pytest.mark.parametrize("flat", [False, True])
def test_peak_exports_keep_physical_width_and_omit_sampling_width(tmp_path, technique, physical_width, flat):
    path = _peak_input(tmp_path, technique, flat=flat)
    tool = FtirTool()
    result = tool.find_peaks([path], {"technique": technique, "peak_file": str(path)})
    original = result.tables["Peaks"].copy(deep=True)
    assert "Peak width (points)" in original
    assert original.empty == flat
    artifacts = tool.exportable_artifacts(result)
    assert [item.kind for item in artifacts] == ["csv", "xlsx"]
    workbook = artifacts[1]
    assert list(workbook.payload) == ["Peaks"]
    exported = workbook.payload["Peaks"]
    assert "Peak width (points)" not in exported and physical_width in exported
    pd.testing.assert_frame_equal(exported, original.drop(columns="Peak width (points)"))
    pd.testing.assert_frame_equal(artifacts[0].payload, exported)
    _save_and_check(workbook, tmp_path)
    pd.testing.assert_frame_equal(result.tables["Peaks"], original)


@pytest.mark.parametrize("mode", ["area", "catalyst"])
@pytest.mark.parametrize("ionomer_case", ["absent", "zero", "multiple"])
@pytest.mark.parametrize("extras", [False, True])
def test_ink_workbook_compares_complete_base_and_batch_recipes(tmp_path, mode, ionomer_case, extras):
    params = _ink_params(mode, ionomers=ionomer_case == "multiple", extras=extras)
    if ionomer_case == "zero":
        params["ionomers"] = [{"ratio": 0.0, "conc": 0.0, "density": 0.0}]
    tool = InkCalcTool()
    result = tool.run([], params)
    originals = {name: frame.copy(deep=True) for name, frame in result.tables.items()}
    workbook = next(item for item in tool.exportable_artifacts(result) if item.kind == "xlsx")
    assert list(workbook.payload) == (["Recipe summary"]
                                    + (["Ionomers"] if ionomer_case != "absent" else [])
                                    + (["Other materials"] if extras else []))
    summary = workbook.payload["Recipe summary"].set_index("Quantity")
    assert list(summary.columns) == (["Base recipe", "Required batch", "Unit"]
                                     if mode == "area" else ["Base recipe", "Unit"])
    assert len(summary) == (9 if mode == "area" else 7)
    assert not any(quantity.startswith("Required") for quantity in summary.index)
    assert summary["Base recipe"]["Total solids"] == pytest.approx(
        summary["Base recipe"][["Catalyst", "Total ionomer", "Total other solids"]].sum())
    for quantity, row in summary.iterrows():
        expected = originals["Recipe summary"].set_index("Quantity").loc[quantity]
        if quantity not in {"Spray loss", "Batch scale factor"}:
            assert row["Base recipe"] == expected["Value"]
        else:
            assert pd.isna(row["Base recipe"])
            assert row["Required batch"] == expected["Value"]
    if mode == "area":
        scale = 1 / (1 - 0.212 * (1 / params["length"] + 1 / params["width"]))
        for quantity in summary.index.difference(["Spray loss", "Batch scale factor"]):
            assert summary.loc[quantity, "Required batch"] == pytest.approx(
                summary.loc[quantity, "Base recipe"] * scale)
    if ionomer_case != "absent":
        ions = workbook.payload["Ionomers"]
        assert ions.columns.tolist() == ["Type", "Base active (mg)", "Base dispersion (mg)", "Base volume (µL)"] + (
            ["Batch active (mg)", "Batch dispersion (mg)", "Batch volume (µL)"] if mode == "area" else [])
        np.testing.assert_allclose(ions.iloc[:, :4], originals["Ionomers"])
        if mode == "area":
            np.testing.assert_allclose(ions.iloc[:, 4:], originals["Adjusted ionomers"].iloc[:, 1:])
        assert ions["Base active (mg)"].sum() == pytest.approx(summary.loc["Total ionomer", "Base recipe"])
    if extras:
        other = workbook.payload["Other materials"]
        pd.testing.assert_frame_equal(other, originals["Other materials"])
        assert other["Mass (mg)"].sum() == pytest.approx(summary.loc["Total other solids", "Base recipe"])
        if mode == "area":
            assert other["Required mass (mg)"].sum() == pytest.approx(summary.loc["Total other solids", "Required batch"])
    _save_and_check(workbook, tmp_path)
    for name, original in originals.items():
        pd.testing.assert_frame_equal(result.tables[name], original)
