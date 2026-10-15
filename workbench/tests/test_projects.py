"""User-level portable project workflows across persistent tool pages."""
from pathlib import Path
from threading import Event
from time import monotonic

import pandas as pd
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QThreadPool
from PySide6.QtGui import QCloseEvent, QGuiApplication
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

import workbench.project_controller as project_module
from workbench.app import GroupedToolPage, MainWindow
from workbench.core.project import load_project, save_project
from workbench.core.tool_base import Artifact, ParameterSpec, Tool, ToolResult
from workbench.tools import build_tools
from workbench.widgets.export_button import ExportButton
from workbench.widgets.figure_view import SingleFigure, StaticFigure


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def errors(monkeypatch):
    seen = []
    monkeypatch.setattr(QMessageBox, "critical", lambda _p, title, detail: seen.append((title, detail)))
    return seen


def dispose(window, app):
    if window.project._loaded is not None:
        window.project._loaded.close()
    window.deleteLater()
    app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    while QGuiApplication.overrideCursor() is not None:
        QGuiApplication.restoreOverrideCursor()


class ProjectTool(Tool):
    id = "project_test"
    name = "Project test"
    files_required = False

    def __init__(self):
        self.calls = []
        self.release = None

    def parameter_schema(self):
        return [ParameterSpec("label", "Label", kind="str", default="original"),
                ParameterSpec("source", "Source", kind="input_file")]

    def run(self, files, params, extra=None):
        self.calls.append((files, params))
        if self.release is not None and not self.release.wait(5):
            raise TimeoutError("Test did not release worker")
        return ToolResult(tables={"Data": pd.DataFrame({"Value": [1.2345678901234567]})},
                          text_blocks={"Run label": params["label"]},
                          param_updates={"label": "fitted"},
                          param_annotations={"label": "fit note"})


def run_page(page):
    files = page._input_files()
    extra = page.extra.values() if page.extra is not None else None
    page._pending_run = {"method": "run", "params": page.panel.values()}
    result = page.tool.run(files, page.panel.values(), extra)
    page._on_run_finished(result)
    return result


def test_all_pages_portable_round_trip_with_saved_results(qt_app, tmp_path, errors):
    first = MainWindow(build_tools())
    second = MainWindow(build_tools())
    try:
        folder = tmp_path / "实验 source"
        folder.mkdir()
        shared = folder / "sample.txt"
        shared.write_text("E(V),j(A/cm2)\n0,0\n0.2,0.1\n", encoding="utf-8")
        duplicate_dir = folder / "other"
        duplicate_dir.mkdir()
        duplicate = duplicate_dir / shared.name
        duplicate.write_text("E(V),j(A/cm2)\n0,0\n0.4,0.2\n", encoding="utf-8")
        for page in first.project.leaf_pages():
            if page.file_input is not None:
                page.file_input.set_files([shared, duplicate])

        group = first.stack.widget(0)
        assert isinstance(group, GroupedToolPage)
        generator = group.pages[1]
        generator.panel.set_values({"temperature": 310.0})
        run_page(generator)
        generator.panel.set_values({"temperature": 315.0})
        generator.results.tabs.setCurrentIndex(1)

        integrated = group.pages[2]
        inputs = integrated.extra
        source_files = [shared, duplicate]
        for role, values in {
            "anode": [(1.3, 0), (1.5, 0.02), (1.8, 0.1)],
            "cathode": [(0.1, 0), (-0.1, -0.02), (-0.3, -0.1)],
            "cell": [(1.2, 0), (1.6, 0.02), (2.1, 0.1)],
            "calibration": [(0, 0), (-0.2, -0.02), (-0.4, -0.1)],
        }.items():
            path = folder / f"{role}.txt"
            path.write_text("E(V),j(A/cm2)\n" + "\n".join(f"{e},{j}" for e, j in values),
                            encoding="utf-8")
            source_files.append(path)
            inputs.inputs[role].set_files([path])
        for role in ("anode", "cathode", "cell"):
            path = folder / f"{role}_eis.txt"
            path.write_text("Frequency(Hz),Zre(Ohm.cm2),Zim(Ohm.cm2)\n1000,0.1,0.01\n100,0.2,-0.01\n",
                            encoding="utf-8")
            source_files.append(path)
            inputs.eis_inputs[role].set_files([path])
        integrated.panel.set_values({"correction_mode": "3E onset", "include_ir": True})
        original = run_page(integrated)
        assert not original.display_tables and not original.display_summary
        calculated = integrated.panel.values()["calculated_shift"]
        integrated.panel.set_values({"correction_mode": "None", "manual_shift": -0.4})
        inputs.tabs.setCurrentIndex(1)
        inputs.eis_tabs.setCurrentIndex(2)
        integrated.results.tabs.setCurrentIndex(1)
        group.function_combo.setCurrentIndex(2)

        ink = next(p for p in first.project.leaf_pages() if p.tool.id == "ink_calc")
        run_page(ink)
        ink.extra.ionomers.table.item(0, 1).setText("unfinished")
        ink.extra.extras.add_row(["", ""])

        converter = next(p for p in first.project.leaf_pages() if p.tool.id == "txt_xlsx")
        run_page(converter)
        first.nav.setCurrentRow(5)
        project_path = tmp_path / "workspace.ecwb"
        assert first.project.save_to(project_path), errors
        assert not first.project.modified

        for path in source_files:
            path.unlink()
        moved = tmp_path / "moved"
        moved.mkdir()
        moved_project = moved / project_path.name
        project_path.rename(moved_project)
        assert second.project.open_path(moved_project, prompt=False), errors
        assert not second.project.modified
        assert second.nav.currentRow() == 5
        for old_page, new_page in zip(first.project.leaf_pages(), second.project.leaf_pages()):
            assert old_page.panel.values() == new_page.panel.values()
            if old_page.file_input is not None:
                assert [p.name for p in old_page.file_input.files()] == [p.name for p in new_page.file_input.files()]
                assert all(p.is_file() for p in new_page.file_input.files())
        new_group = second.stack.widget(0)
        assert new_group.function_combo.currentIndex() == 2
        new_generator = new_group.pages[1]
        assert new_generator.panel.values()["temperature"] == 315.0
        assert new_generator.last_run["params"]["temperature"] == 310.0
        assert isinstance(new_generator.results.tabs.widget(0), StaticFigure)
        assert new_generator.results.tabs.currentIndex() == 1
        new_integrated = new_group.pages[2]
        assert new_integrated.panel.values()["calculated_shift"] == calculated
        assert new_integrated.panel.values()["manual_shift"] == -0.4
        assert new_integrated.extra.inputs["calibration"].files()[0].is_file()
        assert not new_integrated.extra.tabs.isTabVisible(3)
        assert new_integrated.extra.tabs.currentIndex() == 1
        assert new_integrated.extra.eis_tabs.currentIndex() == 2
        assert new_integrated.results.tabs.count() == len(original.figures)
        assert new_integrated.results.tabs.currentIndex() == 1
        pd.testing.assert_frame_equal(new_integrated.last_result.tables["Reconstructed cell"],
                                      original.tables["Reconstructed cell"])
        assert all(a.serialized_payload for a in new_integrated.last_artifacts)
        assert new_integrated.export_btn._start_dir() == str(moved)
        new_ink = next(p for p in second.project.leaf_pages() if p.tool.id == "ink_calc")
        assert new_ink.extra.ionomers.table.item(0, 1).text() == "unfinished"
        assert new_ink.extra.extras.rows() == [["", ""]]
        assert new_ink.last_result is not None
        new_converter = next(p for p in second.project.leaf_pages() if p.tool.id == "txt_xlsx")
        xlsx = tmp_path / "restored.xlsx"
        ExportButton._write(new_converter.last_artifacts[0], str(xlsx))
        assert xlsx.read_bytes().startswith(b"PK")
        # A second save preserves extracted attachments and static snapshots.
        assert second.project.save_to(moved / "resaved.ecwb"), errors
        # Rerunning a reopened analysis uses embedded data and replaces image views.
        run_page(new_integrated)
        assert isinstance(new_integrated.results.tabs.widget(0), SingleFigure)
        assert new_integrated.export_btn._start_dir() == str(moved)
    finally:
        dispose(first, qt_app)
        dispose(second, qt_app)


def test_file_selector_and_edited_input_remain_separate_from_results(qt_app, tmp_path, errors):
    tool = ProjectTool()
    window = MainWindow([tool])
    try:
        paths = []
        for name in ("first.txt", "second.txt"):
            path = tmp_path / name
            path.write_text(name, encoding="utf-8")
            paths.append(path)
        page = window.stack.widget(0)
        page.file_input.set_files(paths)
        page.panel.set_values({"source": str(paths[1])})
        run_page(page)
        page.panel.set_values({"label": "new unfinished input"})
        page.panel.set_annotations({"label": "current field note"})
        path = tmp_path / "test.ecwb"
        assert window.project.save_to(path), errors
        assert window.project.open_path(path), errors
        loaded = window.stack.widget(0)
        assert len(tool.calls) == 1
        assert loaded.panel.values()["source"] == str(loaded.file_input.files()[1])
        assert loaded.panel.values()["label"] == "new unfinished input"
        assert loaded.panel.save_state()["annotations"] == {"label": "current field note"}
        assert loaded.last_result.text_blocks["Run label"] == "original"
        assert loaded.last_result.param_updates["label"] == "fitted"
        assert not window.project.modified
    finally:
        dispose(window, qt_app)


def test_save_failure_and_invalid_open_leave_workspace_intact(qt_app, tmp_path, errors, monkeypatch):
    window = MainWindow([ProjectTool()])
    try:
        page = window.stack.widget(0)
        page.panel.set_values({"label": "keep me"})
        target = tmp_path / "good.ecwb"
        assert window.project.save_to(target)
        original_bytes = target.read_bytes()
        page.panel.set_values({"label": "new edit"})

        def fail_save(*args, **kwargs):
            raise OSError("Disk unavailable")

        with monkeypatch.context() as patch:
            patch.setattr(project_module, "save_project", fail_save)
            assert not window.project.save_to(target)
        assert target.read_bytes() == original_bytes
        assert window.project.path == target
        assert window.project.modified
        assert window.stack.widget(0) is page
        broken = tmp_path / "broken.ecwb"
        broken.write_text("not a project", encoding="utf-8")
        assert not window.project.open_path(broken, prompt=False)
        assert window.stack.widget(0) is page
        assert page.panel.values()["label"] == "new edit"
        assert window.project.modified
        assert all(action.isEnabled() for action in window.project.actions)
        assert QGuiApplication.overrideCursor() is None
        # A structurally valid archive with invalid page state is also transactional.
        loaded = load_project(target)
        try:
            loaded.state["pages"][page.tool.id]["result_tab"] = 999
            invalid = tmp_path / "invalid_page.ecwb"
            save_project(invalid, loaded.state)
        finally:
            loaded.close()
        assert not window.project.open_path(invalid, prompt=False)
        assert window.stack.widget(0) is page
        assert len(errors) == 3
    finally:
        dispose(window, qt_app)


@pytest.mark.parametrize("operation", ["new", "open", "close"])
@pytest.mark.parametrize("answer", [QMessageBox.StandardButton.Save,
                                     QMessageBox.StandardButton.Discard,
                                     QMessageBox.StandardButton.Cancel])
def test_unsaved_choices(qt_app, tmp_path, errors, monkeypatch, operation, answer):
    window = MainWindow([ProjectTool()])
    try:
        target = tmp_path / "target.ecwb"
        assert window.project.save_to(target)
        old_page = window.stack.widget(0)
        old_page.panel.set_values({"label": "unsaved"})
        monkeypatch.setattr(QMessageBox, "warning", lambda *args: answer)
        if operation == "new":
            outcome = window.project.new_project()
        elif operation == "open":
            outcome = window.project.open_path(target)
        else:
            event = QCloseEvent()
            window.closeEvent(event)
            outcome = event.isAccepted()
        assert outcome == (answer != QMessageBox.StandardButton.Cancel)
        if answer == QMessageBox.StandardButton.Cancel:
            assert window.stack.widget(0) is old_page
            assert window.project.modified
        if answer == QMessageBox.StandardButton.Save:
            project = load_project(target)
            try:
                assert project.state["pages"]["project_test"]["panel"]["values"]["label"] == "unsaved"
            finally:
                project.close()
    finally:
        dispose(window, qt_app)


def test_save_as_menu_defaults_shortcuts_and_cancelled_save(qt_app, tmp_path, errors, monkeypatch):
    window = MainWindow([ProjectTool()])
    try:
        window.stack.widget(0).panel.set_values({"label": "edited"})
        assert window.project.modified
        assert [a.shortcut().toString() for a in window.project.actions] == [
            "Ctrl+N", "Ctrl+O", "Ctrl+S", "Ctrl+Shift+S"]
        monkeypatch.setattr(project_module.QFileDialog, "getSaveFileName",
                            lambda *args: (str(tmp_path / "chosen"), ""))
        window.project.save_action.trigger()
        assert window.project.path == tmp_path / "chosen.ecwb"
        assert "chosen.ecwb" in window.windowTitle()
        assert not window.project.modified
        window.stack.widget(0).panel.set_values({"label": "edited again"})
        monkeypatch.setattr(QMessageBox, "warning", lambda *args: QMessageBox.StandardButton.Save)
        monkeypatch.setattr(project_module.QFileDialog, "getSaveFileName", lambda *args: ("", ""))
        window.project.path = None
        assert not window.project.new_project()
        assert window.project.modified
        assert window.stack.widget(0).panel.values()["label"] == "edited again"
    finally:
        dispose(window, qt_app)


def test_running_calculation_locks_project_actions_and_recovers(qt_app, tmp_path, errors):
    tool = ProjectTool()
    tool.release = Event()
    window = MainWindow([tool])
    try:
        page = window.stack.widget(0)
        page.run_btn.click()
        assert page._running
        assert all(not a.isEnabled() for a in window.project.actions)
        assert not window.project.new_project()
        assert not window.project.save_to(tmp_path / "running.ecwb")
        event = QCloseEvent()
        window.closeEvent(event)
        assert not event.isAccepted()
        tool.release.set()
        deadline = monotonic() + 5
        while page._running and monotonic() < deadline:
            QTest.qWait(5)
        assert not page._running
        assert page.last_result is not None
        assert window.project.modified
        assert all(a.isEnabled() for a in window.project.actions)
        assert window.project.save_to(tmp_path / "finished.ecwb"), errors
    finally:
        tool.release.set()
        QThreadPool.globalInstance().waitForDone(5000)
        dispose(window, qt_app)


def test_file_menu_survives_inspection_and_workspace_replacement(qt_app, tmp_path, errors):
    window = MainWindow([ProjectTool()])
    try:
        target = tmp_path / "menu.ecwb"
        assert window.project.save_to(target)
        assert window.project.open_path(target, prompt=False), errors
        qt_app.processEvents()
        menu = window.menuBar().actions()[0].menu()
        assert menu.title() == "&File"
        assert [a for a in menu.actions() if not a.isSeparator()] == window.project.actions
    finally:
        dispose(window, qt_app)
