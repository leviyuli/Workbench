"""CCD displays plots and exports scientific results while projects retain full data."""
from io import BytesIO

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication, QTableWidget

from workbench.core.project import load_project, restore_result, save_project, snapshot_result
from workbench.tools.ccd.tool import CcdTool
from workbench.widgets.results_panel import ResultsPanel


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance()
    if app is None:
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("QT_QPA_PLATFORM", "offscreen")
            app = QApplication([])
    yield app


@pytest.fixture(scope="module")
def ccd_result(tmp_path_factory):
    path = tmp_path_factory.mktemp("ccd_export_only") / "cycles.csv"
    current = np.logspace(-4, np.log10(5), 2500)
    residual = .2 * np.maximum(current - .8, 0) + .08 * np.maximum(current - 3, 0) ** 2
    potential = 1.2291 + .06 * np.log10(current / 1e-6) + .12 * current + residual
    pd.DataFrame({
        "Cycle 1 E(V)": potential, "Cycle 1 j(A/cm2)": current,
        "Cycle 2 E(V)": potential, "Cycle 2 j(A/cm2)": .9 * current,
    }).to_csv(path, index=False)
    return CcdTool().run([path], {
        "input_type": "CV", "hfr": .12, "temperature": 298.15,
        "i_lower": .02, "i_upper": .8,
    })


def _assert_plot_tabs_only(panel, result):
    panel.set_result(result)
    plot_names = list(dict.fromkeys([*result.figures, *result.figure_images]))
    assert [panel.tabs.tabText(index) for index in range(panel.tabs.count())] == [
        *plot_names, "Summary",
    ]
    assert not panel.findChildren(QTableWidget)
    assert "CCD" in panel.summary.toPlainText()


def test_ccd_hides_tables_and_exports_readable_results(qt_app, ccd_result):
    result = ccd_result
    assert not result.display_tables
    assert result.display_summary
    panel = ResultsPanel()
    try:
        _assert_plot_tabs_only(panel, result)
    finally:
        panel.deleteLater()
        qt_app.processEvents()

    artifacts = CcdTool().exportable_artifacts(result)
    assert [artifact.kind for artifact in artifacts].count("xlsx") == 1
    assert all(artifact.kind in {"xlsx", "png"} for artifact in artifacts)
    workbook = next(artifact.payload for artifact in artifacts if artifact.kind == "xlsx")
    assert {"Current results", "Breakdown results", "Batch statistics", "Current statistics",
            "Curve data", "CCD data", "Fit diagnostics", "Summary"} <= set(workbook)
    assert list(workbook)[:3] == ["Summary", "Current results", "Breakdown results"]
    assert not any(name.startswith("Plot ") for name in workbook)
    assert not {"Complete decomposition", "Processed scans", "Raw selected scans"} & set(workbook)
    pd.testing.assert_frame_equal(workbook["Batch statistics"], result.tables["Batch statistics"])
    fit = workbook["Fit diagnostics"]
    pd.testing.assert_frame_equal(fit, result.tables["Fit diagnostics"][fit.columns])
    settings = workbook["Summary"].loc[workbook["Summary"].Section == "Settings", ["Parameter", "Value"]]
    assert "Potential scale" in settings.Parameter.values
    assert "bin_width_v" not in settings.Parameter.values
    processed = result.tables["Processed scans"]
    exported = workbook["CCD data"]
    assert len(exported) == len(processed)
    for column, source in [("Smoothed current (A/cm²)", "Smoothed current (A/cm²)"),
                           ("Residual (V)", "Signed residual (V)"),
                           ("Baseline (V)", "Residual baseline (V)")]:
        pd.testing.assert_series_equal(exported[column], processed[source], check_names=False)
    raw = result.tables["Raw selected scans"]
    assert len(workbook["Curve data"]) == len(raw) > len(exported)
    pd.testing.assert_series_equal(workbook["Curve data"]["Source row"], raw["Source row"])
    np.testing.assert_allclose(workbook["Curve data"]["Ohmic (V)"],
                               result.tables["Complete decomposition"]["Ohmic overpotential (V)"])
    assert len(result.tables["Current results"]) == 2
    assert result.tables["Current results"]["Decoupling status"].eq("Resolved").all()


def test_saved_ccd_result_retains_hidden_tables_and_integrated_workbook(
    qt_app, ccd_result, tmp_path,
):
    result = ccd_result
    artifacts = CcdTool().exportable_artifacts(result)
    expected_workbook = next(artifact.payload for artifact in artifacts if artifact.kind == "xlsx")
    target = tmp_path / "ccd.ecwb"
    save_project(target, {"result": snapshot_result(result, artifacts)})
    with load_project(target) as loaded:
        restored, saved_artifacts = restore_result(loaded.state["result"])
        assert not restored.display_tables
        assert list(restored.tables) == list(result.tables)
        for name, expected in result.tables.items():
            pd.testing.assert_frame_equal(restored.tables[name], expected)
        assert all(artifact.kind in {"xlsx", "png"} for artifact in saved_artifacts)
        workbook = next(artifact for artifact in saved_artifacts if artifact.kind == "xlsx")
        written = pd.read_excel(BytesIO(workbook.serialized_payload), sheet_name=None)
        assert list(written) == list(expected_workbook)
        for name in ("Curve data", "CCD data"):
            pd.testing.assert_frame_equal(written[name], expected_workbook[name],
                                          check_dtype=False, rtol=1e-12, atol=1e-14)
        pd.testing.assert_frame_equal(written["Summary"].fillna(""), expected_workbook["Summary"].fillna(""),
                                      check_dtype=False, rtol=1e-12, atol=1e-14)

        panel = ResultsPanel()
        try:
            _assert_plot_tabs_only(panel, restored)
        finally:
            panel.deleteLater()
            qt_app.processEvents()
