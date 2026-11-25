"""Workbook exports stay responsive and publish only completed files."""

from pathlib import Path
from threading import Event, get_ident
from time import monotonic
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QThreadPool, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from shiboken6 import isValid

import workbench.widgets.export_button as export_module
from workbench.core.tool_base import Artifact
from workbench.widgets.export_button import ExportButton


@pytest.fixture(scope="module")
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
    assert predicate(), "The export did not complete before the timeout"


@pytest.fixture
def export_ui(qt_app, monkeypatch, tmp_path):
    target = tmp_path / "result.xlsx"
    messages = []
    dialogs = []

    def select_file(*args):
        dialogs.append(args)
        return str(target), "Excel (*.xlsx)"

    monkeypatch.setattr(export_module.QFileDialog, "getSaveFileName", select_file)
    for method in ("critical", "information"):
        monkeypatch.setattr(
            export_module.QMessageBox, method,
            lambda _parent, title, detail, method=method:
                messages.append((method, title, detail, get_ident())),
        )
    button = ExportButton()
    button.set_input_dir(str(tmp_path))
    release = Event()
    yield SimpleNamespace(button=button, target=target, messages=messages,
                          dialogs=dialogs, release=release)
    release.set()
    assert QThreadPool.globalInstance().waitForDone(5000)
    qt_app.processEvents()
    button.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


@pytest.mark.parametrize("kind", ["xlsx", "workbook"])
@pytest.mark.parametrize("keep_artifacts", [False, True])
def test_workbook_export_is_responsive_and_keeps_selected_artifact(
    export_ui, monkeypatch, kind, keep_artifacts,
):
    ui = export_ui
    artifact = Artifact("Original", kind, "result.xlsx", None, b"complete workbook")
    ui.button.set_artifacts([artifact])
    started = Event()
    worker_threads = []
    write = ExportButton._write

    def delayed_write(selected, path):
        worker_threads.append(get_ident())
        started.set()
        if not ui.release.wait(5):
            raise TimeoutError("Test did not release the export")
        write(selected, path)

    monkeypatch.setattr(ui.button, "_write", delayed_write)
    gui_thread = get_ident()
    ui.button._save(artifact)
    _wait_until(started.is_set)

    assert ui.button.text() == "Exporting…"
    assert not ui.button.isEnabled()
    assert not ui.target.exists()
    assert ui.messages == []
    assert len(worker_threads) == 1
    assert worker_threads[0] != gui_thread
    timer_threads = []
    QTimer.singleShot(0, lambda: timer_threads.append(get_ident()))
    _wait_until(lambda: bool(timer_threads))
    assert timer_threads == [gui_thread]
    assert not ui.release.is_set()

    replacement = Artifact("Replacement", "xlsx", "replacement.xlsx", None, b"replacement")
    ui.button.set_artifacts([replacement] if keep_artifacts else [])
    ui.button.set_input_dir(str(ui.target.parent / "other inputs"))
    assert not ui.button.isEnabled()
    ui.button._save(replacement)
    assert len(ui.dialogs) == 1

    ui.release.set()
    _wait_until(lambda: ui.button._worker is None)
    assert ui.target.read_bytes() == b"complete workbook"
    assert ui.button.text() == "Export ▾"
    assert ui.button.isEnabled() == keep_artifacts
    assert ui.messages == [("information", "Export", f"Saved:\n{ui.target}", gui_thread)]


def test_failed_workbook_export_preserves_destination_and_recovers(export_ui):
    ui = export_ui
    ui.target.write_bytes(b"previous workbook")
    started = Event()

    class FailingWorkbook:
        def save(self, path):
            Path(path).write_bytes(b"incomplete workbook")
            started.set()
            if not ui.release.wait(5):
                raise TimeoutError("Test did not release the export")
            raise OSError("Disk write failed")

    artifact = Artifact("Workbook", "workbook", "result.xlsx", FailingWorkbook())
    ui.button.set_artifacts([artifact])
    ui.button._save(artifact)
    _wait_until(started.is_set)
    assert ui.target.read_bytes() == b"previous workbook"
    assert not ui.button.isEnabled()
    assert not ui.messages
    ui.release.set()
    _wait_until(lambda: ui.button._worker is None)

    assert ui.target.read_bytes() == b"previous workbook"
    assert list(ui.target.parent.iterdir()) == [ui.target]
    assert ui.button.isEnabled()
    assert ui.button.text() == "Export ▾"
    assert ui.messages == [("critical", "Export failed", "Disk write failed", get_ident())]


def test_export_recovers_if_worker_submission_fails(export_ui, monkeypatch):
    ui = export_ui

    def fail_to_start(_worker):
        raise RuntimeError("Worker pool unavailable")

    monkeypatch.setattr(
        export_module, "QThreadPool",
        SimpleNamespace(globalInstance=lambda: SimpleNamespace(start=fail_to_start)),
    )
    artifact = Artifact("Workbook", "xlsx", "result.xlsx", None, b"workbook")
    ui.button.set_artifacts([artifact])
    ui.button._save(artifact)

    assert ui.button._worker is None
    assert ui.button.isEnabled()
    assert ui.button.text() == "Export ▾"
    assert not ui.target.exists()
    assert ui.messages == [("critical", "Export failed", "Worker pool unavailable", get_ident())]


def test_png_export_stays_on_gui_thread(export_ui, monkeypatch):
    ui = export_ui
    writes = []
    monkeypatch.setattr(ui.button, "_write", lambda art, path: writes.append((art, path, get_ident())))
    artifact = Artifact("Figure", "png", "plot.png", object())
    ui.button.set_artifacts([artifact])
    ui.button._save(artifact)

    assert writes == [(artifact, str(ui.target), get_ident())]
    assert ui.button._worker is None
    assert ui.button.isEnabled()
    assert ui.messages[0][0] == "information"


def test_closing_export_button_during_write_is_safe(qt_app, tmp_path, monkeypatch):
    target = tmp_path / "closed.xlsx"
    release = Event()
    started = Event()
    messages = []
    monkeypatch.setattr(export_module.QFileDialog, "getSaveFileName", lambda *args: (str(target), ""))
    monkeypatch.setattr(export_module.QMessageBox, "information", lambda *args: messages.append(args))
    monkeypatch.setattr(export_module.QMessageBox, "critical", lambda *args: messages.append(args))
    button = ExportButton()
    write = button._write

    def delayed_write(artifact, path):
        started.set()
        if not release.wait(5):
            raise TimeoutError("Test did not release the export")
        write(artifact, path)

    monkeypatch.setattr(button, "_write", delayed_write)
    artifact = Artifact("Frozen", "xlsx", target.name, None, b"completed after closing")
    try:
        button._save(artifact)
        _wait_until(started.is_set)
        button.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert not isValid(button)
    finally:
        release.set()
        assert QThreadPool.globalInstance().waitForDone(5000)
        qt_app.processEvents()

    assert target.read_bytes() == b"completed after closing"
    assert messages == []


@pytest.mark.parametrize("kind", ["xlsx", "workbook"])
def test_frozen_workbook_write_failure_is_atomic(tmp_path, monkeypatch, kind):
    target = tmp_path / "frozen.xlsx"
    target.write_bytes(b"previous workbook")
    write_bytes = Path.write_bytes

    def fail_after_partial_write(path, data):
        write_bytes(path, b"partial")
        raise OSError("Disk full")

    monkeypatch.setattr(Path, "write_bytes", fail_after_partial_write)
    with pytest.raises(OSError, match="Disk full"):
        ExportButton._write(Artifact("Frozen", kind, target.name, None, b"new workbook"), str(target))

    assert target.read_bytes() == b"previous workbook"
    assert list(tmp_path.iterdir()) == [target]
