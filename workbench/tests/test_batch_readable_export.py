"""Batch workbooks contain physical results, without drawing offsets or duplicate scans."""

import numpy as np
import pandas as pd
import pytest

from workbench.core.io import write_xlsx
from workbench.core.tool_base import ToolResult
from workbench.tools.ccd.batch_exports import readable_batch_tables
from workbench.tools.ccd.tool import CcdTool


@pytest.fixture
def scientific_result():
    """A Back scan with a gap and a second scan whose decomposition failed."""
    first, failed = "S01-C01", "S02-C01"
    identity = pd.DataFrame({
        "Curve": [first, failed], "Source": ["S01", "S02"],
        "File": ["sample.csv", "sample.csv (2)"], "Cycle": [1, 1],
        "Scan": ["Back", "Back"], "Native direction": ["decreasing", "decreasing"],
    })
    current = identity.assign(**{
        "CCD (A/cm²)": [1.23456789012345, np.nan],
        "Limiting current (A/cm²)": [np.nan, np.nan],
        "Decoupling status": ["Resolved", "Analysis failed"],
        "CCD status": ["Resolved", "Decoupling failed"],
        "Limiting-current status": ["Unresolved", "Decoupling failed"],
        "Breakdown status": ["Resolved", "Unavailable"], "Notes": ["", "Fit failed"],
    })
    breakdown = identity.assign(**{
        "Target |j| (A/cm²)": [.5, .5], "Kinetic (V)": [.333333333333333, np.nan],
        "Ohmic (V)": [.06, np.nan], "RCL (V)": [.02, np.nan],
        "Residual (V)": [-.0123456789012345, np.nan],
        "Measured potential (V)": [2.9, np.nan], "Status": ["Resolved", "Decoupling failed"],
    })
    raw = pd.DataFrame({
        "Curve": [first] * 4 + [failed] * 2, "Source row": [5, 6, 8, 9, 21, 22],
        "Measured potential (V)": [3.1, 3., 2.9, 2.8, 2.5, 2.4],
        "Current density (A/cm²)": [-.01, .2, .5, 1.23456789012345, .7, .8],
        "Imported current (A or A/cm²)": [-.025, .5, 1.25, 3.086419725308625, 1.75, 2.],
        "Input is current density": [False] * 6,
    })
    complete = pd.DataFrame({
        "Curve": [first] * 4, "Source row": [5, 6, 8, 9],
        "Measured potential (V)": [3.1, 3., 2.9, 2.8],
        "Reference-converted potential (V)": [3.1, 3., 2.9, 2.8],
        "Signed current density (A/cm²)": [np.nan, .2, .5, 1.23456789012345],
        "Current magnitude (A/cm²)": [np.nan, .2, .5, 1.23456789012345],
        "Kinetic overpotential (V)": [np.nan, .3, np.nan, .4],
        "Ohmic overpotential (V)": [np.nan, .024, np.nan, .148148146814814],
        "RCL overpotential (V)": [np.nan, .02, np.nan, .03],
        "Signed residual (V)": [np.nan, -.0123456789012345, np.nan, .08],
    })
    processed = pd.DataFrame({
        "Curve": [first] * 3, "Source row start": [9, 8, 6], "Source row end": [9, 8, 6],
        "Segment": [0, -1, 1], "Potential (V)": [2.8, np.nan, 3.],
        "Median current (A/cm²)": [1.23456789012345, np.nan, .2],
        "Smoothed current (A/cm²)": [1.23456789012345, np.nan, .201],
        "Signed residual (V)": [.08, np.nan, -.0123456789012345],
        "Residual baseline (V)": [.04, np.nan, -.0123456789012345],
        "Excess above baseline (V)": [.04, np.nan, 0.],
        "Baseline fit": [False, False, True], "CCD point": [True, False, False],
        "Limiting plateau": [False, False, False],
    })
    statistics = pd.DataFrame({
        "Source": ["S01"], "File": ["sample.csv"], "Metric": ["CCD (A/cm²)"],
        "Resolved scans": [1], "Mean": [1.23456789012345], "Sample SD": [np.nan],
    })
    return ToolResult(tables={
        "Plot CCD points": pd.DataFrame({
            "S01-C01 · CCD · File position": [-.16],
            "S01-C01 · CCD · CCD (A/cm²)": [1.23456789012345],
        }),
        "Plot series": pd.DataFrame({"Artist": ["Plot CCD points:1"]}),
        "Current results": current, "Breakdown results": breakdown,
        "Current statistics": statistics,
        "Breakdown statistics": statistics.assign(Metric="Residual (V)", Mean=-.0123456789012345),
        "Batch statistics": pd.DataFrame({"Cycle": [1], "Mean (A/cm²)": [1.23456789012345]}),
        "Raw selected scans": raw, "Complete decomposition": complete,
        "Processed scans": processed, "Fit diagnostics": identity.copy(),
        "Settings": pd.DataFrame({"Parameter": ["include_rcl"], "Value": [False]}),
    })


def test_summaries_are_one_row_per_curve_with_physical_marker_values(scientific_result):
    sheets = readable_batch_tables(scientific_result)
    current = sheets["Current results"]
    assert list(sheets)[:3] == ["Summary", "Current results", "Breakdown results"]
    assert current.Curve.tolist() == ["S01-C01", "S02-C01"]
    assert current.File.tolist() == ["sample.csv", "sample.csv (2)"]
    assert not current.Curve.duplicated().any()
    assert current.iloc[0]["CCD (A/cm²)"] == 1.23456789012345
    assert current.iloc[0]["CCD residual (V)"] == .08
    assert current.iloc[0]["CCD potential (V)"] == 2.8
    assert current.iloc[1][["CCD (A/cm²)", "CCD residual (V)", "CCD potential (V)"]].isna().all()
    assert sheets["Breakdown results"].Curve.tolist() == current.Curve.tolist()
    assert sheets["Breakdown results"]["Target |j| (A/cm²)"].tolist() == [.5, .5]
    assert sheets["Breakdown results"].iloc[0]["Potential at target (V)"] == 2.9
    assert "Measured potential (V)" not in sheets["Breakdown results"]
    assert sheets["Breakdown results"].iloc[0]["Residual (V)"] == -.0123456789012345
    assert sheets["Breakdown results"].iloc[1]["Status"] == "Decoupling failed"
    assert {"Mean (A/cm²)", "Sample SD (A/cm²)"} <= set(sheets["Current statistics"])
    assert {"Mean (V)", "Sample SD (V)"} <= set(sheets["Breakdown statistics"])


@pytest.mark.parametrize("shift", [None, .25])
def test_curve_data_uses_one_potential_scale_and_preserves_unresolved_rows(scientific_result, shift):
    if shift is not None:
        settings = scientific_result.tables["Settings"]
        settings.loc[len(settings)] = ["Potential shift (V)", shift]
        model = scientific_result.tables["Complete decomposition"]
        model["Reference-converted potential (V)"] = model["Measured potential (V)"] + shift
    exported = readable_batch_tables(scientific_result)["Curve data"]
    assert list(exported) == [
        "Curve", "Source row", "Potential (V)", "Signed current (A/cm²)",
        "Kinetic (V)", "Ohmic (V)", "Residual (V)",
    ]
    assert exported["Source row"].tolist() == [5, 6, 8, 9, 21, 22]
    assert exported.Curve.tolist() == ["S01-C01"] * 4 + ["S02-C01"] * 2
    native = np.array([3.1, 3., 2.9, 2.8, 2.5, 2.4])
    np.testing.assert_allclose(exported["Potential (V)"], native + (shift or 0))
    assert exported["Signed current (A/cm²)"].tolist() == [-.01, .2, .5, 1.23456789012345, .7, .8]
    assert exported.iloc[1]["Residual (V)"] == -.0123456789012345
    assert exported.iloc[[0, 2, 4, 5]]["Residual (V)"].isna().all()
    assert exported.iloc[4:][["Kinetic (V)", "Ohmic (V)", "Residual (V)"]].isna().all().all()
    assert not exported[["Curve", "Source row"]].duplicated().any()


def test_processed_ccd_data_keeps_gaps_and_detection_flags(scientific_result):
    exported = readable_batch_tables(scientific_result)["CCD data"]
    assert list(exported) == [
        "Curve", "Segment", "Smoothed current (A/cm²)", "Residual (V)",
        "Baseline (V)", "Baseline fit", "CCD point", "Limiting plateau",
    ]
    assert exported.Segment.tolist() == [0, -1, 1]
    assert exported["CCD point"].tolist() == [True, False, False]
    assert exported["Baseline fit"].tolist() == [False, False, True]
    assert exported.iloc[1][["Smoothed current (A/cm²)", "Residual (V)", "Baseline (V)"]].isna().all()
    assert exported.iloc[2]["Residual (V)"] < 0
    assert exported.iloc[0]["Baseline (V)"] == .04
    assert exported.iloc[2]["Smoothed current (A/cm²)"] == .201


@pytest.mark.parametrize("include_rcl", [False, True])
def test_readable_export_has_optional_rcl_and_no_chart_layout_metadata(scientific_result, include_rcl):
    scientific_result.tables["Settings"].loc[0, "Value"] = include_rcl
    originals = {name: data.copy(deep=True) for name, data in scientific_result.tables.items()}
    sheets = readable_batch_tables(scientific_result)
    assert ("RCL (V)" in sheets["Curve data"]) is include_rcl
    assert ("RCL (V)" in sheets["Breakdown results"]) is include_rcl
    if include_rcl:
        assert sheets["Curve data"].iloc[1]["RCL (V)"] == .02
    assert not {"Complete decomposition", "Raw selected scans", "Processed scans"} & set(sheets)
    assert not any(name.startswith("Plot ") for name in sheets)
    assert not any(any(term in str(column).casefold() for term in
                       ("file position", "artist", "jitter", "x column", "y column"))
                   for table in sheets.values() for column in table)
    for name, original in originals.items():
        pd.testing.assert_frame_equal(scientific_result.tables[name], original)


def test_readable_workbook_retains_scientific_precision_and_missing_values(scientific_result, tmp_path):
    target = tmp_path / "batch_decoupling_ccd.xlsx"
    sheets = readable_batch_tables(scientific_result)
    write_xlsx(sheets, target)
    written = pd.read_excel(target, sheet_name=None)
    assert list(written) == list(sheets)
    for name in ("Current results", "Breakdown results", "Curve data", "CCD data"):
        for column in sheets[name].select_dtypes(include="number"):
            np.testing.assert_allclose(written[name][column], sheets[name][column],
                                       rtol=1e-14, atol=1e-15, equal_nan=True)
    assert written["Curve data"].iloc[0]["Signed current (A/cm²)"] == -.01
    assert written["Curve data"].iloc[1]["Residual (V)"] < 0


@pytest.mark.parametrize("variable_hfr", [False, True])
def test_actual_batch_workbook_removes_duplicate_curve_and_artist_cells(tmp_path, variable_hfr):
    density = np.unique(np.r_[np.logspace(-3, -1, 100), np.linspace(.1, 6., 801)])
    residual = .08 * np.maximum(density - .3, 0.) + .06 * np.maximum(density - 3., 0.) ** 2
    potential = 1.2291 + .06 * np.log10(density / 2e-6) + .12 * density + residual
    source = tmp_path / "two_cycles.csv"
    pd.DataFrame({
        "Cycle 1 E(V)": potential, "Cycle 1 j(A/cm2)": density,
        "Cycle 2 E(V)": potential, "Cycle 2 j(A/cm2)": .9 * density,
    }).to_csv(source, index=False)
    params = {
        "input_type": "CV", "temperature": 298.15, "hfr": .12,
        "electrode_setup": "Three-electrode", "reference": "RHE", "reaction": "OER",
        "i_lower": .005, "i_upper": .3, "target_i": 1.,
        "ccd_fit_mode": "Manual", "ccd_j_lower": .8, "ccd_j_upper": 2.5,
    }
    extra = None
    if variable_hfr:
        params["hfr_mode"] = "HFR vs |j|"
        extra = {"hfr_text": "j,HFR\n.05,.12\n.2,.12\n.5,.12\n1,.12\n3,.12\n5,.12",
                 "hfr_smoothing": .0001, "hfr_source": "hfr.csv"}
    result = CcdTool().run([source], params, extra)
    sheets = readable_batch_tables(result)
    assert len(sheets["Current results"]) == len(sheets["Breakdown results"]) == 2
    assert sheets["Current results"]["Decoupling status"].eq("Resolved").all()
    assert len(sheets["Curve data"]) == len(result.tables["Raw selected scans"])
    assert len(sheets["CCD data"]) == len(result.tables["Processed scans"])
    old_cells = sum(table.size for table in result.tables.values())
    readable_cells = sum(table.size for table in sheets.values())
    assert readable_cells < .5 * old_cells
    model = result.tables["Complete decomposition"]
    np.testing.assert_array_equal(sheets["Curve data"]["Source row"], model["Source row"])
    np.testing.assert_allclose(sheets["Curve data"]["Residual (V)"], model["Signed residual (V)"],
                               rtol=0., atol=0., equal_nan=True)
    summary = sheets["Summary"]
    assert summary.Section.drop_duplicates().tolist() == (
        ["Settings", "HFR points", "HFR settings"] if variable_hfr else ["Settings"]
    )
    assert not {"Settings", "HFR points", "HFR settings"} & set(sheets)
    settings = summary.loc[summary.Section == "Settings", ["Parameter", "Value"]]
    assert {"input_type", "Potential scale", "Potential shift (V)", "Tafel method"} <= set(settings.Parameter)
    assert not {"bin_width_v", "smoothing_window", "Tafel window_width", "Tafel minimum_windows"} & set(settings.Parameter)
    source_settings = result.tables["Settings"]
    pd.testing.assert_frame_equal(
        settings.reset_index(drop=True),
        source_settings.loc[source_settings.Parameter.isin(settings.Parameter)].reset_index(drop=True),
        check_dtype=False,
    )
    fit = sheets["Fit diagnostics"]
    assert {"Curve", "Tafel b (V/dec)", "Baseline slope (Ω·cm²)"} <= set(fit)
    assert not {"Source", "File", "Tafel search lower (A/cm²)", "Tafel joined windows",
                "Tafel window-slope spread (V/dec)"} & set(fit)
    pd.testing.assert_frame_equal(fit, result.tables["Fit diagnostics"][fit.columns])
    if variable_hfr:
        hfr_settings = summary.loc[summary.Section == "HFR settings", ["Parameter", "Value"]]
        pd.testing.assert_frame_equal(hfr_settings.reset_index(drop=True), result.tables["HFR settings"],
                                      check_dtype=False)
        pairs = summary.loc[summary.Section == "HFR points"].pivot(
            index="Record", columns="Parameter", values="Value",
        )
        expected = result.tables["HFR points"]
        assert pairs.index.tolist() == list(range(1, len(expected) + 1))
        np.testing.assert_allclose(pairs[expected.columns].to_numpy(dtype=float), expected.to_numpy(),
                                   rtol=0., atol=0.)
        target = tmp_path / "batch_hfr.xlsx"
        write_xlsx(sheets, target)
        restored = pd.read_excel(target, sheet_name=None, keep_default_na=False)
        pd.testing.assert_frame_equal(restored["Summary"], summary, check_dtype=False,
                                      rtol=1e-12, atol=1e-14)
        for name in ("HFR source data", "HFR fit data"):
            pd.testing.assert_frame_equal(restored[name], result.tables[name], check_dtype=False,
                                          rtol=1e-12, atol=1e-14)
