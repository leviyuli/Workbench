"""Regression coverage for asynchronous tool actions and their UI state."""

from threading import Event, get_ident
from time import monotonic
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QThreadPool, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import workbench.app as app_module
from workbench.app import ToolPage
from workbench.core.tool_base import Artifact, ParameterSpec, Tool, ToolAction, ToolResult


class StubTool(Tool):
    id = "stub"
    name = "Stub tool"
    files_required = False

    def __init__(self):
        self.release = Event()
        self.calls = []
        self.fail = False

    def parameter_schema(self):
        return [ParameterSpec("label", "Label", kind="str", default="sample")]

    def actions(self):
        return [ToolAction("Preview", "preview")]

    def run(self, files, params, extra=None):
        return self._execute("run", files, params, extra)

    def preview(self, files, params, extra=None):
        return self._execute("preview", files, params, extra)

    def _execute(self, method, files, params, extra):
        self.calls.append((method, files, params, extra, get_ident()))
        if not self.release.wait(timeout=5):
            raise TimeoutError("Test did not release the worker")
        if self.fail:
            raise RuntimeError(f"{method} failed")
        return ToolResult(text_blocks={"Result": method})

    def exportable_artifacts(self, result):
        return [Artifact("Result", "txt", "result.txt", result.text_blocks["Result"])]


@pytest.fixture(scope="session")
def qt_app():
    app = QApplication.instance()
    if app is None:
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
            app = QApplication([])
    yield app


def _wait_until(predicate):
    deadline = monotonic() + 5
    while not predicate() and monotonic() < deadline:
        QTest.qWait(5)
    assert predicate(), "The worker did not complete before the timeout"


@pytest.fixture
def tool_page(qt_app, monkeypatch):
    tool = StubTool()
    statuses = []
    errors = []
    monkeypatch.setattr(
        app_module.QMessageBox, "critical",
        lambda _parent, title, detail: errors.append((title, detail)),
    )
    page = ToolPage(tool, lambda message: statuses.append((get_ident(), message)))
    yield SimpleNamespace(page=page, tool=tool, statuses=statuses, errors=errors)

    tool.release.set()
    assert QThreadPool.globalInstance().waitForDone(5000)
    qt_app.processEvents()
    page.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    while QGuiApplication.overrideCursor() is not None:
        QGuiApplication.restoreOverrideCursor()


@pytest.mark.parametrize("action_index, method", [(0, "run"), (1, "preview")])
@pytest.mark.parametrize("fail", [False, True], ids=["success", "failure"])
def test_async_actions_restore_buttons_and_cursor(tool_page, action_index, method, fail):
    page, tool = tool_page.page, tool_page.tool
    tool.fail = fail
    gui_thread = get_ident()
    original_labels = [button.text() for button in page._action_buttons]
    button = page._action_buttons[action_index]

    button.click()

    assert page._running
    assert all(not action.isEnabled() for action in page._action_buttons)
    assert button.text() == "Running…"
    assert QGuiApplication.overrideCursor().shape() == Qt.CursorShape.WaitCursor
    _wait_until(lambda: bool(tool.calls))
    assert tool.calls[0][:4] == (method, [], {"label": "sample"}, None)
    assert tool.calls[0][4] != gui_thread

    tool.release.set()
    _wait_until(lambda: not page._running)

    assert page._worker is None
    assert all(action.isEnabled() for action in page._action_buttons)
    assert [action.text() for action in page._action_buttons] == original_labels
    assert QGuiApplication.overrideCursor() is None
    assert all(thread == gui_thread for thread, _message in tool_page.statuses)
    if fail:
        assert tool_page.errors == [("Stub tool — error", f"RuntimeError: {method} failed")]
        assert tool_page.statuses[-1][1] == f"Stub tool failed: RuntimeError: {method} failed"
        assert not page.export_btn.isEnabled()
    else:
        assert not tool_page.errors
        assert tool_page.statuses[-1][1] == "Stub tool: done."
        assert page.export_btn.isEnabled()
        assert page.last_artifacts[0].payload == method
        assert method in page.results.summary.toPlainText()


def test_export_directory_uses_files_captured_when_run_started(tool_page, tmp_path):
    page, tool = tool_page.page, tool_page.tool
    original_dir = tmp_path / "original"
    replacement_dir = tmp_path / "replacement"
    original_dir.mkdir()
    replacement_dir.mkdir()
    original = original_dir / "input.csv"
    replacement = replacement_dir / "input.csv"
    original.write_text("value\n1\n", encoding="utf-8")
    replacement.write_text("value\n2\n", encoding="utf-8")
    page.file_input._add_paths([original])

    page.run_btn.click()
    _wait_until(lambda: bool(tool.calls))
    page.file_input.clear()
    page.file_input._add_paths([replacement])
    tool.release.set()
    _wait_until(lambda: not page._running)

    assert tool.calls[0][1] == [original]
    assert page.file_input.files() == [replacement]
    assert page.export_btn._start_dir() == str(original_dir)


def test_worker_submission_failure_restores_buttons_and_cursor(tool_page, monkeypatch):
    page = tool_page.page
    original_labels = [button.text() for button in page._action_buttons]

    def fail_to_start(_worker):
        raise RuntimeError("Worker pool unavailable")

    monkeypatch.setattr(
        app_module, "QThreadPool",
        SimpleNamespace(globalInstance=lambda: SimpleNamespace(start=fail_to_start)),
    )

    page._action_buttons[1].click()

    assert not page._running
    assert page._worker is None
    assert all(button.isEnabled() for button in page._action_buttons)
    assert [button.text() for button in page._action_buttons] == original_labels
    assert QGuiApplication.overrideCursor() is None
    assert not tool_page.tool.calls
    assert tool_page.errors == [("Stub tool — error", "RuntimeError: Worker pool unavailable")]
    assert tool_page.statuses[-1][1] == "Stub tool failed: RuntimeError: Worker pool unavailable"
