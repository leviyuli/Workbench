"""Uploaded and pasted HFR pairs feed the fit, plot and portable export together."""
from io import BytesIO
import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication, QTableWidget

from workbench.app import ToolPage
from workbench.core.project import load_project, save_project
from workbench.tools.lsv.hfr_data import read_hfr_file
from workbench.tools.lsv.tool import LsvTool


def _case(tmp_path):
    current = np.geomspace(.001, 1., 501)
    potential = 1.2291 + .06 * np.log10(current / 2e-6) + current * (.1 + .04 * current)
    source = tmp_path / "lsv.csv"
    pd.DataFrame({"E(V)": potential, "j(A/cm2)": current}).to_csv(source, index=False)
    hfr = tmp_path / "hfr.csv"
    measured = np.array([.05, .2, .4, .6, .8])
    pd.DataFrame({"j": measured, "HFR": .1 + .04 * measured}).to_csv(hfr, index=False)
    tool = LsvTool()
    params = {spec.key: spec.default for spec in tool.parameter_schema()}
    params.update(electrode_setup="Three-electrode", reference="RHE", reaction="OER",
                  temperature=298.15, ph=0., hfr_mode="HFR vs |j|", include_rcl=False,
                  i_lower=.005, i_upper=.1, target_i=.3)
    extra = {"hfr_text": read_hfr_file(hfr), "hfr_smoothing": .0001,
             "hfr_source": hfr.name}
    return tool, source, hfr, params, extra


def _assert_hfr_summary(summary, tables):
    points = summary.loc[summary["Section"].eq("HFR points")]
    expected = tables["HFR points"]
    restored = points.pivot(index="Record", columns="Parameter", values="Value")
    assert restored.index.tolist() == list(range(1, len(expected) + 1))
    restored = (restored.reindex(columns=expected.columns).reset_index(drop=True)
                .rename_axis(columns=None))
    pd.testing.assert_frame_equal(restored, expected, check_dtype=False,
                                  rtol=1e-12, atol=1e-14)
    settings = summary.loc[summary["Section"].eq("HFR settings")]
    assert (settings["Record"].isna() | settings["Record"].eq("")).all()
    pd.testing.assert_frame_equal(settings[["Parameter", "Value"]].reset_index(drop=True),
                                  tables["HFR settings"], check_dtype=False,
                                  rtol=1e-12, atol=1e-14)


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance()
    if app is None:
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("QT_QPA_PLATFORM", "offscreen")
            app = QApplication([])
    yield app


def test_uploaded_hfr_pairs_drive_full_lsv_correction_and_export_only_data(qt_app, tmp_path):
    tool, source, _, params, extra = _case(tmp_path)
    result = tool.run([source], params, extra)
    assert result.scalars["b_kin (V/dec)"] == pytest.approx(.06, rel=2e-4)
    assert result.payload["export_breakdown"]["Ohmic (V)"] == pytest.approx(.0336)
    components = result.tables["Overpotential components"]
    current = np.abs(components["Current Density (A/cm²)"].to_numpy())
    np.testing.assert_allclose(components["HFR (Ω·cm²)"], .1 + .04 * current, atol=1e-12)
    assert "HFR fit" in result.figures
    assert not result.display_tables
    assert {"HFR points", "HFR fit data", "HFR settings", "HFR source data"} <= result.tables.keys()
    fitted = result.tables["HFR fit data"]
    grid = fitted["|j| (A/cm²)"].to_numpy()
    assert grid[0] == pytest.approx(current.min())
    assert grid[-1] == pytest.approx(current.max())
    assert np.all(np.diff(grid) > 0)
    np.testing.assert_allclose(fitted["Fitted HFR (Ω·cm²)"], .1 + .04 * grid, atol=1e-12)
    np.testing.assert_array_equal(fitted["Extended"], (grid < .05) | (grid > .8))
    assert fitted["Extended"].any()
    assert list(result.tables["HFR source data"]) == ["|j| (A/cm²)", "HFR"]
    artifacts = tool.exportable_artifacts(result)
    workbook = next(artifact.payload for artifact in artifacts if artifact.kind == "xlsx")
    assert list(workbook) == ["Summary", "Decoupled curves", "HFR source data", "HFR fit data"]
    _assert_hfr_summary(workbook["Summary"], result.tables)
    for name in ("HFR fit data", "HFR source data"):
        pd.testing.assert_frame_equal(workbook[name], result.tables[name])
    page = ToolPage(tool, lambda _: None)
    try:
        page.results.set_result(result)
        names = [page.results.tabs.tabText(index) for index in range(page.results.tabs.count())]
        assert names[-1] == "Summary"
        assert "HFR fit" in names
        assert not page.results.findChildren(QTableWidget)
        preview = tool.show_plot([source], params, extra)
        assert not preview.display_tables
        assert preview.tables
    finally:
        page.deleteLater()
        qt_app.processEvents()


def test_multiple_hfr_columns_are_rejected_instead_of_silently_selected(tmp_path):
    tool, source, _, params, extra = _case(tmp_path)
    extra["hfr_text"] = "j\tHFR1\tHFR2\n.05\t.102\t.2\n.2\t.108\t.21\n.4\t.116\t.22\n.6\t.124\t.23\n.8\t.132\t.24"
    with pytest.raises(ValueError, match="(?i)(two columns|one HFR|single HFR)"):
        tool.run([source], params, extra)


def test_hfr_text_fit_and_exports_roundtrip_without_external_hfr_file(qt_app, tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    tool, source, hfr_source, params, extra = _case(tmp_path)
    original = ToolPage(tool, lambda _: None)
    restored = ToolPage(LsvTool(), lambda _: None)
    try:
        original.file_input.set_files([source])
        original.panel.set_values(params)
        original.extra.restore_state(extra)
        saved_inputs = original.extra.values()
        original.last_result = tool.run([source], params, saved_inputs)
        original.last_artifacts = tool.exportable_artifacts(original.last_result)
        original.results.set_result(original.last_result)
        expected = original.last_result
        project = tmp_path / "hfr_spline.ecwb"
        save_project(project, {"page": original.save_state()})
        hfr_source.unlink()
        source.unlink()
        with load_project(project) as loaded:
            restored.restore_state(loaded.state["page"], tmp_path)
            assert restored.extra.values() == saved_inputs
            assert restored.file_input.files()[0].is_file()
            assert "HFR fit" in restored.last_result.figure_images
            assert not restored.last_result.display_tables
            for name in ("HFR points", "HFR fit data", "HFR settings", "HFR source data"):
                pd.testing.assert_frame_equal(restored.last_result.tables[name], expected.tables[name])
            workbook_artifact = next(item for item in restored.last_artifacts if item.kind == "xlsx")
            workbook = pd.read_excel(BytesIO(workbook_artifact.serialized_payload), sheet_name=None)
            assert list(workbook) == ["Summary", "Decoupled curves", "HFR source data", "HFR fit data"]
            _assert_hfr_summary(workbook["Summary"], expected.tables)
            for name in ("HFR fit data", "HFR source data"):
                pd.testing.assert_frame_equal(workbook[name], expected.tables[name], check_dtype=False,
                                              rtol=1e-12, atol=1e-14)
            rerun = restored.tool.run(restored.file_input.files(), restored.panel.values(),
                                     restored.extra.values())
            assert rerun.payload["export_breakdown"]["Ohmic (V)"] == pytest.approx(.0336)
    finally:
        original.deleteLater()
        restored.deleteLater()
        qt_app.processEvents()


def test_historical_selected_table_restores_to_plot_with_all_export_data(qt_app, tmp_path):
    tool, source, _, params, extra = _case(tmp_path)
    original = ToolPage(tool, lambda _: None)
    restored = ToolPage(LsvTool(), lambda _: None)
    try:
        original.file_input.set_files([source])
        original.panel.set_values(params)
        original.extra.restore_state(extra)
        original.last_result = tool.run([source], params, extra)
        original.last_artifacts = tool.exportable_artifacts(original.last_result)
        state = original.save_state()
        state["result"]["display_tables"] = True
        first_table = len(state["result"]["figures"]) + 1
        state["result_tab"] = first_table

        restored.restore_state(state, tmp_path)

        names = [restored.results.tabs.tabText(index)
                 for index in range(restored.results.tabs.count())]
        assert names == [*original.last_result.figures, "Summary"]
        assert restored.results.tabs.currentIndex() == 0
        assert not restored.last_result.display_tables
        assert not restored.results.findChildren(QTableWidget)
        for name, frame in original.last_result.tables.items():
            pd.testing.assert_frame_equal(restored.last_result.tables[name], frame)
        assert any(artifact.kind == "xlsx" for artifact in restored.last_artifacts)
        assert state["result"]["display_tables"] is True
        assert state["result_tab"] == first_table
    finally:
        original.deleteLater()
        restored.deleteLater()
        qt_app.processEvents()
