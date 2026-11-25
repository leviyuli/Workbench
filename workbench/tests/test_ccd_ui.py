"""CCD navigation shares selected curves and preserves portable LSV projects."""
from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QMessageBox

from workbench.app import GroupedToolPage, MainWindow
from workbench.core.project import load_project, save_project
from workbench.core.tool_base import ToolResult
from workbench.tools.lsv_analysis.tool import LsvAnalysisTool


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance()
    if app is None:
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("QT_QPA_PLATFORM", "offscreen")
            app = QApplication([])
    yield app


@pytest.fixture
def windows(qt_app, monkeypatch):
    created = []
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(args))

    def create():
        window = MainWindow([LsvAnalysisTool()])
        created.append(window)
        return window

    yield SimpleNamespace(create=create, errors=errors)
    for window in created:
        if window.project._loaded is not None:
            window.project._loaded.close()
        window.deleteLater()
    qt_app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    while QGuiApplication.overrideCursor() is not None:
        QGuiApplication.restoreOverrideCursor()


def _pages(window):
    group = window.stack.widget(0)
    assert isinstance(group, GroupedToolPage)
    return group, {page.tool.id: page for page in group.pages}


def _source(tmp_path, name="curve.csv"):
    path = tmp_path / name
    path.write_text("E(V),j(A/cm2)\n1.4,0.01\n2.0,1\n3.0,4\n", encoding="utf-8")
    return path


def _result(page, label):
    result = ToolResult(
        tables={"Values": pd.DataFrame({"Current density (A/cm²)": [4.3]})},
        text_blocks={"Case": label},
    )
    page.last_result = result
    page.last_artifacts = page.tool.exportable_artifacts(result)
    page.last_run = {"method": "run", "params": deepcopy(page.panel.values())}
    page.results.set_result(result)
    page.export_btn.set_artifacts(page.last_artifacts)


def test_ccd_is_an_lsv_function_and_shares_add_remove_clear_and_order(windows, tmp_path):
    window = windows.create()
    group, pages = _pages(window)
    assert list(pages) == ["lsv", "lsv_generator", "integrated_re_cell", "lsv_ccd"]
    assert group.function_combo.currentText() == "Decoupling"
    decoupling, ccd = pages["lsv"], pages["lsv_ccd"]
    first, second = _source(tmp_path), _source(tmp_path, "second.csv")
    decoupling.file_input.set_files([first, second])
    assert ccd.file_input.files() == [first, second]
    group.function_combo.setCurrentIndex(3)
    ccd.file_input.set_files([second, first])
    assert decoupling.file_input.files() == [second, first]
    ccd.file_input.list.item(0).setSelected(True)
    ccd.file_input.btn_remove.click()
    assert ccd.file_input.files() == decoupling.file_input.files() == [first]
    # Other LSV functions keep their role-specific selections independent.
    role_picker = pages["integrated_re_cell"].extra.inputs["cell"]
    role_picker.set_files([second])
    decoupling.file_input.clear()
    assert ccd.file_input.files() == []
    assert role_picker.files() == [second]


def test_shared_files_do_not_replace_each_functions_results(windows, tmp_path):
    window = windows.create()
    group, pages = _pages(window)
    decoupling, ccd = pages["lsv"], pages["lsv_ccd"]
    _result(decoupling, "Decoupled losses")
    _result(ccd, "CCD batch")
    decoupling.panel.set_values({"hfr": .12})
    ccd.file_input.set_files([_source(tmp_path)])
    group.function_combo.setCurrentIndex(3)
    group.function_combo.setCurrentIndex(0)
    assert decoupling.panel.values()["hfr"] == .12
    assert decoupling.last_result.text_blocks["Case"] == "Decoupled losses"
    assert ccd.last_result.text_blocks["Case"] == "CCD batch"


def test_ccd_roundtrip_keeps_shared_embedded_files_and_independent_results(windows, tmp_path):
    window = windows.create()
    group, pages = _pages(window)
    source = _source(tmp_path)
    pages["lsv"].file_input.set_files([source])
    _result(pages["lsv"], "Decoupled losses")
    _result(pages["lsv_ccd"], "CCD batch")
    group.function_combo.setCurrentIndex(3)
    target = tmp_path / "ccd.ecwb"
    assert window.project.save_to(target), windows.errors
    with load_project(target) as loaded:
        assert len(loaded.files) == 1
        saved = loaded.state["pages"]["lsv"]["pages"]
        assert saved["lsv"]["files"] == saved["lsv_ccd"]["files"]
    source.unlink()

    reopened = windows.create()
    assert reopened.project.open_path(target, prompt=False), windows.errors
    restored, children = _pages(reopened)
    assert restored.stack.currentWidget() is children["lsv_ccd"]
    assert children["lsv"].file_input.files() == children["lsv_ccd"].file_input.files()
    assert children["lsv_ccd"].file_input.files()[0].is_file()
    assert children["lsv"].last_result.text_blocks["Case"] == "Decoupled losses"
    assert children["lsv_ccd"].last_result.text_blocks["Case"] == "CCD batch"
    assert not reopened.project.modified


def test_previous_lsv_project_initializes_ccd_from_decoupling_files(windows, tmp_path):
    old = windows.create()
    _, pages = _pages(old)
    source = _source(tmp_path)
    pages["lsv"].file_input.set_files([source])
    pages["lsv"].panel.set_values({"hfr": .12})
    _result(pages["lsv"], "Prior decoupling output")
    state = old.project.workspace_state()
    saved = state["pages"]["lsv"]
    saved["pages"].pop("lsv_ccd")
    saved["selected_tool"] = "lsv_generator"
    target = tmp_path / "prior_lsv.ecwb"
    save_project(target, state)
    source.unlink()

    current = windows.create()
    _, initial = _pages(current)
    defaults = initial["lsv_ccd"].panel.values()
    assert current.project.open_path(target, prompt=False), windows.errors
    group, restored = _pages(current)
    assert group.stack.currentWidget() is restored["lsv_generator"]
    assert restored["lsv"].panel.values()["hfr"] == .12
    assert restored["lsv"].last_result.text_blocks["Case"] == "Prior decoupling output"
    assert restored["lsv_ccd"].panel.values() == defaults
    assert restored["lsv_ccd"].last_result is None
    assert restored["lsv_ccd"].file_input.files() == restored["lsv"].file_input.files()
    assert restored["lsv_ccd"].file_input.files()[0].is_file()
    assert not current.project.modified


@pytest.mark.parametrize("hfr", [.01, .15])
def test_batch_hfr_default_and_custom_value_roundtrip(windows, tmp_path, hfr):
    window = windows.create()
    group, pages = _pages(window)
    ccd = pages["lsv_ccd"]
    assert ccd.panel.values()["hfr"] == pytest.approx(.01)
    ccd.panel.set_values({"hfr": hfr, "input_type": "CV", "cv_cycle_mode": "Specific",
                          "cv_cycle": 2, "cv_scan": "Back", "ccd_fit_mode": "Manual",
                          "ccd_j_lower": 1.1, "ccd_j_upper": 2.9})
    ccd.file_input.set_files([_source(tmp_path)])
    _result(ccd, "Saved batch selection")
    group.function_combo.setCurrentIndex(3)
    target = tmp_path / "batch_settings.ecwb"
    assert window.project.save_to(target), windows.errors
    with load_project(target) as loaded:
        saved = loaded.state["pages"]["lsv"]["pages"]["lsv_ccd"]
        assert saved["panel"]["values"]["hfr"] == pytest.approx(hfr)

    reopened = windows.create()
    assert reopened.project.open_path(target, prompt=False), windows.errors
    restored_group, restored = _pages(reopened)
    assert restored_group.stack.currentWidget() is restored["lsv_ccd"]
    assert restored["lsv_ccd"].panel.values() == ccd.panel.values()
    assert restored["lsv_ccd"].last_run["params"]["hfr"] == pytest.approx(hfr)
    assert restored["lsv_ccd"].last_result.text_blocks["Case"] == "Saved batch selection"
    assert not reopened.project.modified


@pytest.mark.parametrize("historical_hfr", [None, .18])
def test_historical_ccd_project_maps_hfr_and_keeps_frozen_results(windows, tmp_path, historical_hfr):
    old = windows.create()
    _, pages = _pages(old)
    ccd = pages["lsv_ccd"]
    source = _source(tmp_path)
    ccd.file_input.set_files([source])
    _result(ccd, "CCD output before residual method")
    state = old.project.workspace_state()
    state["pages"]["lsv"]["selected_tool"] = "lsv_ccd"
    saved = state["pages"]["lsv"]["pages"]["lsv_ccd"]
    saved["panel"]["values"] = {
        "electrode_area": 2.5, "current_branch": "Negative",
        "baseline_mode": "Manual", "baseline_lower": 1.9, "baseline_upper": 2.45,
    }
    if historical_hfr is not None:
        saved["panel"]["values"]["reference_hfr"] = historical_hfr
    saved["extra"] = None
    saved["last_run"]["params"] = deepcopy(saved["panel"]["values"])
    frozen_artifacts = deepcopy(saved["result"]["artifacts"])
    target = tmp_path / "historical_ccd.ecwb"
    save_project(target, state)
    source.unlink()

    current = windows.create()
    assert current.project.open_path(target, prompt=False), windows.errors
    group, restored = _pages(current)
    restored_ccd = restored["lsv_ccd"]
    assert group.stack.currentWidget() is restored_ccd
    assert restored_ccd.panel.values()["hfr"] == pytest.approx(.12 if historical_hfr is None else historical_hfr)
    assert restored_ccd.panel.values()["electrode_area"] == 2.5
    assert restored_ccd.panel.values()["current_branch"] == "Negative"
    assert restored_ccd.panel.values()["ccd_fit_mode"] == "Auto"
    assert restored_ccd.panel.values()["cv_cycle_mode"] == "All"
    assert restored_ccd.panel.values()["ccd_j_lower"] == 1.0
    assert restored_ccd.panel.values()["ccd_j_upper"] == 3.0
    assert restored_ccd.file_input.files() == restored["lsv"].file_input.files()
    assert restored_ccd.file_input.files()[0].is_file()
    assert restored_ccd.last_result.text_blocks["Case"] == "CCD output before residual method"
    assert "Historical" in restored_ccd.last_result.text_blocks["Analysis method"]
    assert not restored_ccd.last_result.display_tables
    pd.testing.assert_frame_equal(restored_ccd.last_result.tables["Values"], ccd.last_result.tables["Values"])
    assert restored_ccd.last_run["params"] == saved["last_run"]["params"]
    assert [artifact.serialized_payload for artifact in restored_ccd.last_artifacts] == [
        artifact["bytes"] for artifact in frozen_artifacts]
    assert not current.project.modified


@pytest.mark.parametrize("malformation", [
    "missing_generator", "missing_selected_ccd", "unknown_child", "invalid_selection",
    "inconsistent_shared_files",
])
def test_invalid_ccd_project_keeps_current_workspace_intact(windows, tmp_path, malformation):
    current = windows.create()
    before, children = _pages(current)
    source = _source(tmp_path)
    children["lsv"].file_input.set_files([source])
    _result(children["lsv_ccd"], "Keep this result")
    state = current.project.workspace_state()
    saved = state["pages"]["lsv"]
    if malformation == "missing_generator":
        saved["pages"].pop("lsv_generator")
    elif malformation == "missing_selected_ccd":
        saved["pages"].pop("lsv_ccd")
        saved["selected_tool"] = "lsv_ccd"
    elif malformation == "unknown_child":
        saved["pages"]["unknown"] = saved["pages"].pop("lsv_ccd")
    elif malformation == "invalid_selection":
        saved["selected_tool"] = "unknown"
    else:
        saved["pages"]["lsv_ccd"]["files"] = []
    target = tmp_path / "invalid.ecwb"
    save_project(target, state)
    assert not current.project.open_path(target, prompt=False)
    assert current.stack.widget(0) is before
    assert children["lsv_ccd"].last_result.text_blocks["Case"] == "Keep this result"
    assert children["lsv"].file_input.files() == [source]
    assert len(windows.errors) == 1
