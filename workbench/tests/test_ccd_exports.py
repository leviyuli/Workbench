"""CCD exports retain cycle identities and numeric data without duplicate files."""
from dataclasses import asdict

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from workbench.core.tool_base import ToolResult
from workbench.tools.ccd.exports import ccd_artifacts
from workbench.tools.ccd.logic import CcdSettings
from workbench.widgets.export_button import ExportButton


def _result():
    return ToolResult(
        tables={
            "Current results": pd.DataFrame({
                "Curve": ["S01-C01", "S01-C02"], "File": ["same.csv"] * 2,
                "Cycle": [1, 2], "CCD (A/cm²)": [4.2, np.nan],
                "Limiting current (A/cm²)": [5.0, np.nan],
                "CCD status": ["Resolved onset", "Unresolved"],
            }),
            "Batch statistics": pd.DataFrame({"Cycle": [1, 2], "Resolved files": [1, 0]}),
            "Processed curves": pd.DataFrame({
                "Curve": ["S01-C01", "S01-C02"], "File": ["same.csv"] * 2,
                "Cycle": [1, 2], "Measured potential (V)": [-2.1, -2.2],
                "HFR-corrected potential (V)": [-1.9, -1.95],
                "Smoothed current (A/cm²)": [4.2, 4.3],
                "Median current (A/cm²)": [4.1, 4.4],
                "HFR drop (V)": [.2, .25], "Residual increase (mV)": [20., 12.],
                "Baseline fit": [False, True], "CCD point": [True, False],
                "Plateau": [False, False],
            }),
            "Input curves": pd.DataFrame({
                "Measured potential (V)": [-2.101, -2.199],
                "Current magnitude (A/cm²)": [4.15, 4.35],
            }),
            "Benchmark lines": pd.DataFrame({
                "Curve": ["S01-C01", "S01-C02"], "File": ["same.csv"] * 2,
                "Cycle": [1, 2], "Potential (V)": [-2.1, -2.2],
                "iR-corrected benchmark potential (V)": [-1.9, -1.95],
                "Benchmark current density (A/cm²)": [4.2, 4.3],
                "Baseline region": [True, False],
            }),
            "Baseline fits": pd.DataFrame({
                "Curve": ["S01-C01", "S01-C02"], "Fit selection": ["Manual"] * 2,
                "Fit start potential (V)": [-1.9] * 2,
                "Fit end potential (V)": [-2.45] * 2,
                "Fit R²": [.999, .998],
            }),
            "Analysis settings": pd.DataFrame(
                asdict(CcdSettings()).items(), columns=["Parameter", "Value"]),
        },
        figures={name: Figure() for name in
                 ("LSV processing", "Current statistics", "Residual onset")},
    )


def test_exports_one_complete_workbook_without_mutating_analysis_tables(tmp_path):
    result = _result()
    originals = {name: frame.copy(deep=True) for name, frame in result.tables.items()}
    artifacts = ccd_artifacts(result)

    assert [item.kind for item in artifacts] == ["xlsx", "png", "png", "png"]
    assert len({item.default_filename for item in artifacts}) == 4
    assert next(item.name for item in artifacts if item.default_filename == "ccd_currents.png") == (
        "Current statistics (PNG)")
    workbook = artifacts[0]
    assert list(workbook.payload) == [
        "Current results", "Batch statistics", "Forward scans", "Input curves",
        "Benchmark lines", "Baseline fits", "Settings"]
    pd.testing.assert_frame_equal(workbook.payload["Current results"], result.tables["Current results"])
    pd.testing.assert_frame_equal(workbook.payload["Batch statistics"], result.tables["Batch statistics"])
    forward = workbook.payload["Forward scans"]
    assert forward["Cycle"].tolist() == [1, 2]
    assert forward["File"].tolist() == ["same.csv", "same.csv"]
    assert forward["Potential (V)"].tolist() == [-2.1, -2.2]
    assert forward["iR-corrected potential (V)"].tolist() == [-1.9, -1.95]
    assert forward["Current density (A/cm²)"].tolist() == [4.2, 4.3]
    assert forward["Excess voltage (mV)"].tolist() == [20., 12.]
    assert forward["CCD point"].tolist() == [True, False]
    assert forward["Median current (A/cm²)"].tolist() == [4.1, 4.4]
    assert forward["HFR drop (V)"].tolist() == [.2, .25]
    assert len(forward.columns) == len(result.tables["Processed curves"].columns)
    pd.testing.assert_frame_equal(workbook.payload["Input curves"], result.tables["Input curves"])
    for name in ["Benchmark lines", "Baseline fits"]:
        pd.testing.assert_frame_equal(workbook.payload[name], result.tables[name])

    output = tmp_path / workbook.default_filename
    ExportButton._write(workbook, str(output))
    actual = pd.read_excel(output, sheet_name=None)
    actual["Settings"]["Unit"] = actual["Settings"]["Unit"].fillna("")
    for name, expected in workbook.payload.items():
        pd.testing.assert_frame_equal(actual[name], expected, check_dtype=False,
                                      check_exact=False, rtol=1e-12, atol=1e-14)
    for name, expected in originals.items():
        pd.testing.assert_frame_equal(result.tables[name], expected)


def test_cycle_statistics_export_preserves_each_sources_sample_spread(tmp_path):
    result = _result()
    cycle_statistics = pd.DataFrame({
        "File": ["same.csv"] * 4,
        "Source": ["S01", "S01", "S02", "S02"],
        "Metric": ["CCD", "Limiting current"] * 2,
        "Resolved cycles": [2, 1, 3, 0],
        "Mean (A/cm²)": [4.2, 5.0, 3.0, np.nan],
        "Sample SD (A/cm²)": [.2, np.nan, .5, np.nan],
    })
    result.tables["Cycle statistics"] = cycle_statistics
    result.figures["Cycle trends"] = Figure()
    artifacts = ccd_artifacts(result)
    workbook = next(item for item in artifacts if item.kind == "xlsx")

    assert [item.kind for item in artifacts] == ["xlsx", "png", "png", "png", "png"]
    assert list(workbook.payload) == [
        "Current results", "Batch statistics", "Cycle statistics", "Forward scans", "Input curves",
        "Benchmark lines", "Baseline fits", "Settings"]
    pd.testing.assert_frame_equal(workbook.payload["Cycle statistics"], cycle_statistics)
    pd.testing.assert_frame_equal(workbook.payload["Batch statistics"], result.tables["Batch statistics"])
    assert next(item.payload for item in artifacts if item.default_filename == "ccd_cycle_trends.png") is (
        result.figures["Cycle trends"])

    output = tmp_path / workbook.default_filename
    ExportButton._write(workbook, str(output))
    written = pd.read_excel(output, sheet_name="Cycle statistics")
    pd.testing.assert_frame_equal(written, cycle_statistics, check_dtype=False)
    pd.testing.assert_frame_equal(result.tables["Cycle statistics"], cycle_statistics)


def test_settings_use_readable_labels_and_explicit_units_without_changing_values():
    artifacts = ccd_artifacts(_result())
    settings = next(item.payload["Settings"] for item in artifacts if item.kind == "xlsx")
    settings = settings.set_index("Parameter")
    assert settings.loc["CCD excess voltage"].tolist() == [20., "mV"]
    assert settings.loc["Reference HFR"].tolist() == [.12, "Ω·cm²"]
    assert settings.loc["Maximum flat-region slope"].tolist() == [15., "% of baseline"]
    assert settings.loc["Maximum flat-region variation"].tolist() == [5., "%"]
    assert settings.loc["Voltage averaging interval"].tolist() == [10., "mV"]


def test_restored_plot_bytes_remain_exportable_and_live_figures_take_precedence(tmp_path):
    result = ToolResult(figures={"LSV processing": Figure()}, figure_images={
        "LSV processing": b"old image", "Residual onset": b"restored image"})
    artifacts = ccd_artifacts(result)
    assert len(artifacts) == 2
    assert artifacts[0].payload is result.figures["LSV processing"]
    assert artifacts[0].serialized_payload is None
    assert artifacts[1].serialized_payload == b"restored image"
    output = tmp_path / artifacts[1].default_filename
    ExportButton._write(artifacts[1], str(output))
    assert output.read_bytes() == b"restored image"


def test_restored_cycle_trend_keeps_the_named_export_and_saved_bytes(tmp_path):
    result = ToolResult(figure_images={"Cycle trends": b"saved cycle trend"})
    artifacts = ccd_artifacts(result)

    assert len(artifacts) == 1
    assert artifacts[0].name == "Cycle trends (PNG)"
    assert artifacts[0].default_filename == "ccd_cycle_trends.png"
    output = tmp_path / artifacts[0].default_filename
    ExportButton._write(artifacts[0], str(output))
    assert output.read_bytes() == b"saved cycle trend"
