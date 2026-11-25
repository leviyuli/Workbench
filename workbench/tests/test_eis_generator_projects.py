"""Grouped EIS pages preserve established analysis and portable projects."""
from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QMessageBox

from workbench.app import GroupedToolPage, MainWindow
from workbench.core.project import load_project, save_project
from workbench.core.tool_base import Tool, ToolResult
from workbench.tools import build_tools
from workbench.tools.eis.tool import EisTool
from workbench.tools.eis_analysis.tool import EisAnalysisTool


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

    def create(tools):
        window = MainWindow(tools)
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


def _set_result(page, label):
    result = ToolResult(
        tables={"Spectrum": pd.DataFrame({"Frequency (Hz)": [1000.0, 10.0],
                                          "Zre (Ohm cm2)": [0.1, 0.2]})},
        text_blocks={"Case": label},
    )
    page.last_result = result
    page.last_artifacts = Tool.exportable_artifacts(page.tool, result)
    page.last_run = {"method": "run", "params": deepcopy(page.panel.values())}
    page.results.set_result(result)
    page.export_btn.set_artifacts(page.last_artifacts)


def test_eis_group_preserves_existing_analysis_interface(windows):
    tools = build_tools()
    registered = [tool for tool in tools if tool.id == "eis"]
    assert len(registered) == 1
    grouped = registered[0]
    assert isinstance(grouped, EisAnalysisTool)
    assert grouped.name == "EIS (OSIF)"
    assert isinstance(grouped, EisTool)
    assert EisAnalysisTool.run is EisTool.run
    assert grouped.parameter_schema() == EisTool().parameter_schema()
    children = grouped.subtools()
    assert [child.id for child in children] == ["eis", "eis_generator"]
    assert [child.name for child in children] == ["Analysis", "CL EIS generation"]
    assert children[0].parameter_schema() == EisTool().parameter_schema()
    assert children[0].actions() == EisTool().actions()
    assert not children[1].needs_files
    window = windows.create([grouped])
    page = window.stack.widget(0)
    assert isinstance(page, GroupedToolPage)
    assert page.function_combo.currentText() == "Analysis"


def test_legacy_single_page_eis_project_opens_and_resaves_grouped(windows, tmp_path):
    legacy = windows.create([EisTool()])
    original = legacy.stack.widget(0)
    source = tmp_path / "legacy.csv"
    source.write_text("Frequency(Hz),Zre(Ohm),Zim(Ohm)\n1000,0.1,-0.01\n", encoding="utf-8")
    original.file_input.set_files([source])
    original.panel.set_values({"mode": "Simulate", "rcl": 0.382})
    _set_result(original, "legacy analysis")
    original.panel.set_values({"rcl": 0.45})
    original.panel.set_annotations({"rcl": "unfinished input after last result"})
    target = tmp_path / "legacy.ecwb"
    assert legacy.project.save_to(target), windows.errors
    source.unlink()

    current = windows.create([EisAnalysisTool()])
    defaults = current.stack.widget(0).pages[1].panel.values()
    assert current.project.open_path(target, prompt=False), windows.errors
    group = current.stack.widget(0)
    assert group.function_combo.currentText() == "Analysis"
    analysis, generator = group.pages
    assert analysis.panel.values() == original.panel.values()
    assert analysis.panel.save_state()["annotations"] == {
        "rcl": "unfinished input after last result",
    }
    assert analysis.last_run["params"]["rcl"] == 0.382
    assert analysis.file_input.files()[0].is_file()
    assert analysis.file_input.files()[0].name == "legacy.csv"
    assert analysis.last_result.text_blocks["Case"] == "legacy analysis"
    pd.testing.assert_frame_equal(analysis.last_result.tables["Spectrum"],
                                  original.last_result.tables["Spectrum"])
    assert analysis.last_artifacts and all(a.serialized_payload for a in analysis.last_artifacts)
    assert generator.panel.values() == defaults
    assert generator.last_result is None
    assert not current.project.modified

    resaved = tmp_path / "migrated.ecwb"
    assert current.project.save_to(resaved), windows.errors
    with load_project(resaved) as loaded:
        state = loaded.state["pages"]["eis"]
        assert state["selected_tool"] == "eis"
        assert set(state["pages"]) == {"eis", "eis_generator"}
        assert state["pages"]["eis"]["last_run"]["params"]["rcl"] == 0.382
    reopened = windows.create([EisAnalysisTool()])
    assert reopened.project.open_path(resaved, prompt=False), windows.errors
    assert reopened.stack.widget(0).pages[0].last_result.text_blocks["Case"] == "legacy analysis"


@pytest.mark.parametrize("malformation", ["missing_panel", "wrong_tool_id", "invalid_result_tab"])
def test_invalid_legacy_eis_restore_leaves_current_workspace_intact(
    windows, tmp_path, malformation,
):
    legacy = windows.create([EisTool()])
    state = legacy.project.workspace_state()
    saved = state["pages"]["eis"]
    if malformation == "missing_panel":
        saved.pop("panel")
    elif malformation == "wrong_tool_id":
        saved["tool_id"] = "unrecognized"
    else:
        saved["result_tab"] = 999
    target = tmp_path / "invalid.ecwb"
    save_project(target, state)
    current = windows.create([EisAnalysisTool()])
    before = current.stack.widget(0)
    before.pages[0].panel.set_values({"rcl": 0.77})
    _set_result(before.pages[1], "keep generator results")
    cache = {"key": "keep existing analysis cache"}
    before.pages[0].tool._kk_cache = cache
    assert not current.project.open_path(target, prompt=False)
    assert current.stack.widget(0) is before
    assert before.pages[0].panel.values()["rcl"] == 0.77
    assert before.pages[0].tool._kk_cache is cache
    assert before.pages[1].last_result.text_blocks["Case"] == "keep generator results"
    assert current.project.modified
    assert len(windows.errors) == 1


@pytest.mark.parametrize("malformation", ["missing_generator", "unknown_child", "invalid_selection"])
def test_grouped_eis_restore_keeps_strict_child_validation(windows, tmp_path, malformation):
    current = windows.create([EisAnalysisTool()])
    before = current.stack.widget(0)
    state = current.project.workspace_state()
    saved = state["pages"]["eis"]
    if malformation == "missing_generator":
        saved["pages"].pop("eis_generator")
    elif malformation == "unknown_child":
        saved["pages"]["unknown"] = saved["pages"].pop("eis_generator")
    else:
        saved["selected_tool"] = "unknown"
    target = tmp_path / "invalid_group.ecwb"
    save_project(target, state)
    assert not current.project.open_path(target, prompt=False)
    assert current.stack.widget(0) is before
    assert len(windows.errors) == 1


def test_eis_functions_keep_independent_inputs_results_and_selected_page(windows, tmp_path):
    first = windows.create([EisAnalysisTool()])
    group = first.stack.widget(0)
    analysis, generator = group.pages
    analysis.panel.set_values({"mode": "Simulate", "rcl": 0.44})
    _set_result(analysis, "analysis output")
    numeric = next(spec for spec in generator.tool.parameter_schema()
                   if spec.kind == "float" and not spec.read_only)
    new_value = min(numeric.maximum, max(numeric.minimum, numeric.default + numeric.step))
    generator.panel.set_values({numeric.key: new_value})
    _set_result(generator, "generator output")
    group.function_combo.setCurrentIndex(1)
    group.function_combo.setCurrentIndex(0)
    assert analysis.panel.values()["rcl"] == 0.44
    assert analysis.last_result.text_blocks["Case"] == "analysis output"
    assert generator.panel.values()[numeric.key] == new_value
    group.function_combo.setCurrentIndex(1)
    path = tmp_path / "grouped.ecwb"
    assert first.project.save_to(path), windows.errors
    second = windows.create([EisAnalysisTool()])
    assert second.project.open_path(path, prompt=False), windows.errors
    restored = second.stack.widget(0)
    assert restored.function_combo.currentText() == "CL EIS generation"
    assert [page.tool.id for page in second.project.leaf_pages()] == ["eis", "eis_generator"]
    assert restored.pages[0].panel.values() == analysis.panel.values()
    assert restored.pages[1].panel.values() == generator.panel.values()
    assert restored.pages[0].last_result.text_blocks["Case"] == "analysis output"
    assert restored.pages[1].last_result.text_blocks["Case"] == "generator output"
    assert not second.project.modified
