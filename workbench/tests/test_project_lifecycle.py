"""Project shortcuts preserve focused edits and reset only runtime state."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

import workbench.app as app_module
import workbench.project_controller as project_module
from workbench.app import MainWindow
from workbench.core.project import load_project, save_project
from workbench.core.tool_base import ParameterSpec, Tool, ToolResult
from workbench.tools.eis.tool import EisTool
from workbench.tools.ink_calc.tool import InkCalcTool
from workbench.tools.lsv_analysis.tool import LsvAnalysisTool


class LifecycleTool(Tool):
    id = "lifecycle"
    name = "Lifecycle test"
    needs_files = False

    def parameter_schema(self):
        return [ParameterSpec("number", "Number", default=1.0),
                ParameterSpec("label", "Label", kind="str", default="original")]

    def run(self, files, params, extra=None):
        return ToolResult(scalars={"Number": params["number"]})


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance()
    if app is None:
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
            app = QApplication([])
    yield app


@pytest.fixture
def windows(qt_app, monkeypatch):
    created = []
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(args))
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: QMessageBox.StandardButton.Cancel)

    def create(tools=None):
        window = MainWindow([LifecycleTool()] if tools is None else tools)
        created.append(window)
        window.show()
        window.activateWindow()
        qt_app.processEvents()
        return window

    yield SimpleNamespace(create=create, errors=errors)
    for window in created:
        if window.project._loaded is not None:
            window.project._loaded.close()
        window.deleteLater()
    qt_app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    while QGuiApplication.overrideCursor() is not None:
        QGuiApplication.restoreOverrideCursor()


def _pending_number(window, qt_app, text="123.5"):
    spinbox = window.stack.widget(0).panel._widgets["number"]
    spinbox.setFocus()
    qt_app.processEvents()
    spinbox.lineEdit().selectAll()
    QTest.keyClicks(spinbox.lineEdit(), text)
    assert spinbox.lineEdit().text() == text
    assert spinbox.value() == 1.0
    assert window.project.modified
    return spinbox


@pytest.mark.parametrize("trigger", ["shortcut", "menu"])
def test_save_commits_focused_numeric_editor(qt_app, windows, tmp_path, monkeypatch, trigger):
    window = windows.create()
    target = tmp_path / "focused.ecwb"
    monkeypatch.setattr(project_module.QFileDialog, "getSaveFileName",
                        lambda *args: (str(target), ""))
    spinbox = _pending_number(window, qt_app)
    if trigger == "shortcut":
        QTest.keyClick(spinbox.lineEdit(), Qt.Key_S, Qt.ControlModifier)
        qt_app.processEvents()
    else:
        window.project.save_action.trigger()
    assert target.is_file(), windows.errors
    assert spinbox.value() == 123.5
    assert not window.project.modified
    with load_project(target) as loaded:
        assert loaded.state["pages"]["lifecycle"]["panel"]["values"]["number"] == 123.5


@pytest.mark.parametrize("trigger", ["shortcut", "menu"])
def test_new_prompts_for_pending_numeric_edit(qt_app, windows, monkeypatch, trigger):
    window = windows.create()
    page = window.stack.widget(0)
    spinbox = _pending_number(window, qt_app)
    prompts = []
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *args: prompts.append(args) or QMessageBox.StandardButton.Cancel)
    if trigger == "shortcut":
        QTest.keyClick(spinbox.lineEdit(), Qt.Key_N, Qt.ControlModifier)
        qt_app.processEvents()
    else:
        window.project.new_action.trigger()
    assert len(prompts) == 1
    assert window.stack.widget(0) is page
    assert page.panel.values()["number"] == 123.5
    assert window.project.modified


def test_failed_save_preserves_pending_edit_and_dirty_state(qt_app, windows, tmp_path, monkeypatch):
    window = windows.create()
    page = window.stack.widget(0)
    spinbox = _pending_number(window, qt_app)

    def fail_save(*args, **kwargs):
        raise OSError("Disk unavailable")

    monkeypatch.setattr(project_module, "save_project", fail_save)
    assert not window.project.save_to(tmp_path / "unwritten.ecwb")
    assert spinbox.value() == 123.5
    assert window.stack.widget(0) is page
    assert window.project.path is None
    assert window.project.modified
    assert len(windows.errors) == 1
    assert all(action.isEnabled() for action in window.project.actions)
    assert QGuiApplication.overrideCursor() is None
    prompts = []
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *args: prompts.append(args) or QMessageBox.StandardButton.Cancel)
    assert not window.project.request_close()
    assert len(prompts) == 1


def test_save_does_not_parse_rounded_display_over_stored_precision(qt_app, windows, tmp_path):
    window = windows.create()
    page = window.stack.widget(0)
    exact = 1.234567890123
    page.panel.set_values({"number": exact})
    # SigFigSpinBox intentionally displays 1.23 while retaining full precision.
    assert page.panel._widgets["number"].text() == "1.23"
    target = tmp_path / "precision.ecwb"
    assert window.project.save_to(target), windows.errors
    assert page.panel.values()["number"] == exact
    with load_project(target) as loaded:
        assert loaded.state["pages"]["lifecycle"]["panel"]["values"]["number"] == exact


def test_shortcut_save_commits_unfinished_ink_table_editor(qt_app, windows, tmp_path, monkeypatch):
    window = windows.create([InkCalcTool()])
    table = window.stack.widget(0).extra.ionomers.table
    item = table.item(0, 0)
    table.setCurrentItem(item)
    table.editItem(item)
    qt_app.processEvents()
    editor = QApplication.focusWidget()
    assert editor is not None and editor is not table
    editor.selectAll()
    QTest.keyClicks(editor, "unfinished")
    assert item.text() != "unfinished"
    target = tmp_path / "ink.ecwb"
    monkeypatch.setattr(project_module.QFileDialog, "getSaveFileName",
                        lambda *args: (str(target), ""))
    QTest.keyClick(editor, Qt.Key_S, Qt.ControlModifier)
    qt_app.processEvents()
    assert target.is_file(), windows.errors
    with load_project(target) as loaded:
        assert loaded.state["pages"]["ink_calc"]["extra"]["ionomers"][0][0] == "unfinished"


@pytest.mark.parametrize("operation", ["new", "open"])
def test_successful_replacement_clears_eis_runtime_cache(qt_app, windows, tmp_path, operation):
    eis = EisTool()
    window = windows.create([eis])
    target = tmp_path / "eis.ecwb"
    assert window.project.save_to(target), windows.errors
    old_page = window.stack.widget(0)
    eis._kk_cache = {"key": "cached inputs", "kk_by_freq": object()}
    if operation == "new":
        assert window.project.new_project()
    else:
        assert window.project.open_path(target)
    assert window.stack.widget(0) is not old_page
    assert window.stack.widget(0).tool._kk_cache is None
    assert not window.project.modified


@pytest.mark.parametrize("failure", ["archive", "page"])
def test_failed_open_preserves_eis_runtime_cache(qt_app, windows, tmp_path, failure):
    eis = EisTool()
    window = windows.create([eis])
    page = window.stack.widget(0)
    cache = {"key": "cached inputs", "kk_by_freq": object()}
    eis._kk_cache = cache
    target = tmp_path / "broken.ecwb"
    if failure == "archive":
        target.write_text("Not an archive", encoding="utf-8")
    else:
        state = window.project.workspace_state()
        state["pages"]["eis"]["result_tab"] = 999
        save_project(target, state)
    assert not window.project.open_path(target, prompt=False)
    assert window.stack.widget(0) is page
    assert page.tool._kk_cache is cache
    assert len(windows.errors) == 1
    assert not window.project.modified


def test_navigation_does_not_mark_project_modified(qt_app, windows):
    window = windows.create([LsvAnalysisTool(), LifecycleTool()])
    group = window.stack.widget(0)
    integrated = group.pages[2]
    result = ToolResult(tables={"Data": pd.DataFrame({"Current": [0.0]})})
    integrated.last_result = result
    integrated.results.set_result(result)
    assert not window.project.modified
    window.nav.setCurrentRow(1)
    window.nav.setCurrentRow(0)
    group.function_combo.setCurrentIndex(2)
    integrated.extra.tabs.setCurrentIndex(1)
    integrated.extra.eis_tabs.setCurrentIndex(2)
    integrated.results.tabs.setCurrentIndex(1)
    qt_app.processEvents()
    assert not window.project.modified


@pytest.mark.parametrize("filename", [None, "实验 workspace.ecwb"])
def test_cli_optional_project_opens_without_starting_real_event_loop(monkeypatch, filename):
    events = []

    class FakeApplication:
        @staticmethod
        def instance():
            return application

        def setApplicationName(self, name):
            events.append(("app", name))

        def setFont(self, font):
            pass

        def setStyleSheet(self, stylesheet):
            pass

        def exec(self):
            events.append(("exec",))
            return 17

    class FakeWindow:
        def __init__(self, tools):
            self.project = SimpleNamespace(open_path=lambda path, **kwargs:
                                           events.append(("open", path, kwargs)))

        def show(self):
            events.append(("show",))

    application = FakeApplication()
    monkeypatch.setattr(app_module, "QApplication", FakeApplication)
    monkeypatch.setattr(app_module, "MainWindow", FakeWindow)
    monkeypatch.setattr(app_module, "build_tools", lambda: [])
    monkeypatch.setattr(app_module, "qInstallMessageHandler", lambda callback: None)
    monkeypatch.setattr(app_module.theme, "apply_matplotlib_style", lambda: None)
    monkeypatch.setattr(app_module.theme, "stylesheet", lambda: "")
    monkeypatch.setattr(app_module.sys, "argv", ["workbench"] + ([] if filename is None else [filename]))
    with pytest.raises(SystemExit) as outcome:
        app_module.main()
    assert outcome.value.code == 17
    if filename is None:
        assert not [event for event in events if event[0] == "open"]
    else:
        assert ("open", Path(filename), {"prompt": False}) in events
        assert events.index(("show",)) < events.index(("open", Path(filename), {"prompt": False}))
    assert events[-1] == ("exec",)
