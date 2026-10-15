"""Workbench window, shared tool pages and background calculation workers."""
from __future__ import annotations

import argparse
import copy
import sys
import traceback
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, cast

# Direct-file launches need the repository root on the import path.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Share page class identities with ProjectController for direct-file launches.
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    from workbench.app import main
    raise SystemExit(main())

from PySide6.QtCore import (
    QObject, QRunnable, Qt, QThreadPool, Signal, Slot, QtMsgType,
    qInstallMessageHandler,
)
from PySide6.QtGui import QCloseEvent, QFont, QGuiApplication
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPushButton, QScrollArea, QSplitter,
    QStackedWidget, QVBoxLayout, QWidget,
)

from workbench import theme
from workbench.core.tool_base import Artifact, ParameterSpec, Tool, ToolResult
from workbench.tools import build_tools
from workbench.widgets.card import Card
from workbench.widgets.export_button import ExportButton
from workbench.widgets.file_input import FileInputWidget
from workbench.widgets.parameter_panel import ParameterPanel
from workbench.widgets.results_panel import ResultsPanel


def _error_detail(exc: Exception) -> str:
    """Format an exception consistently for status messages and dialogs."""
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


class _RunSignals(QObject):
    """Carries a worker's outcome back to the GUI thread."""
    finished = Signal(object)   # ToolResult
    failed = Signal(str)        # formatted error detail


class _RunWorker(QRunnable):
    """Calculate in a worker; ResultsPanel creates Qt canvases on the GUI thread."""

    def __init__(
        self,
        tool: Tool,
        method: str,
        files: list[Path],
        params: dict[str, Any],
        extra: Any,
    ) -> None:
        super().__init__()
        self.signals = _RunSignals()
        self._tool = tool
        self._method = method
        self._files = files
        self._params = params
        self._extra = extra

    @Slot()
    def run(self) -> None:
        try:
            action = getattr(self._tool, self._method)
            result = action(self._files, self._params, self._extra)
        except Exception as exc:
            self.signals.failed.emit(_error_detail(exc))
            return
        self.signals.finished.emit(result)


class ToolPage(QWidget):
    """Generic page that hosts one Tool's inputs and results."""

    runningChanged = Signal(bool)
    changed = Signal()

    def __init__(
        self,
        tool: Tool,
        status_cb: Callable[[str], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.tool = tool
        self._status_cb = status_cb
        self.last_artifacts: list[Artifact] = []
        self.last_result: ToolResult | None = None
        self.last_run: dict[str, Any] | None = None
        self.project_directory: Path | None = None
        self._pending_run: dict[str, Any] | None = None
        self.file_input: FileInputWidget | None = None
        self._action_buttons: list[QPushButton] = []
        self._running = False
        self._worker: _RunWorker | None = None
        self._active_btn: QPushButton | None = None
        self._active_label = ""
        self._run_input_dir: str | None = None
        self._build()

    def _build(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(14, 10, 14, 14)
        root.setSpacing(12)

        splitter = self.splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_inputs())
        splitter.addWidget(self._build_results())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([480, 720])
        root.addWidget(splitter)

    def _build_inputs(self) -> QScrollArea:
        left_card = Card("Inputs")
        if self.tool.needs_files:
            self.file_input = FileInputWidget(
                accepted_exts=self.tool.accepted_file_types,
                multiple=self.tool.multiple_files,
                accepts_directory=self.tool.accepts_directory,
            )
            left_card.add(self.file_input)

        self.panel = ParameterPanel(
            self.tool.parameter_schema(), on_change=self._sync_extra_parameters,
            on_reset=self._resolve_parameter_reset)
        self.panel.changed.connect(self.changed.emit)
        if self.file_input is not None:
            self.file_input.filesChanged.connect(self.panel.set_input_files)
            self.file_input.filesChanged.connect(lambda _paths: self.changed.emit())
            self.panel.set_input_files(self.file_input.files())
        self.extra = self.tool.extra_widget(self, on_change=self.changed.emit)
        if self.extra is not None and self.tool.extra_before_parameters:
            left_card.add(self.extra)
        left_card.add(self.panel)
        if self.extra is not None and not self.tool.extra_before_parameters:
            left_card.add(self.extra)
        self._sync_extra_parameters()

        left_card.add_stretch()
        self.run_btn = self._action_button("run", self.tool.run_button_label)
        self.run_btn.setObjectName("Primary")

        actions = self.tool.actions()
        if actions:
            btn_row = QHBoxLayout()
            btn_row.setSpacing(8)
            btn_row.addWidget(self.run_btn, 1)
            for action in actions:
                btn_row.addWidget(self._action_button(action.method, action.label), 1)
            left_card.body.addLayout(btn_row)
        else:
            left_card.add(self.run_btn)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        left_scroll.setWidget(left_card)
        left_scroll.setMinimumWidth(400)
        left_scroll.setMaximumWidth(620)
        return left_scroll

    def _action_button(self, method: str, label: str) -> QPushButton:
        button = QPushButton(label)
        button.clicked.connect(lambda _checked=False: self._start(method, button))
        self._action_buttons.append(button)
        return button

    def _build_results(self) -> Card:
        right_card = Card("Results")
        top_row = QHBoxLayout()
        top_row.addStretch(1)
        self.export_btn = ExportButton(self.tool.id)
        top_row.addWidget(self.export_btn)
        right_card.body.addLayout(top_row)
        self.results = ResultsPanel()
        right_card.add(self.results)
        return right_card

    def _input_files(self) -> list[Path]:
        if self.file_input is not None:
            return self.file_input.files()
        extra_files = getattr(self.extra, "files", None)
        if not callable(extra_files):
            return []
        file_provider = cast(Callable[[], Iterable[str | Path]], extra_files)
        return [Path(path) for path in file_provider()]

    def _sync_extra_parameters(self) -> None:
        setter = getattr(getattr(self, "extra", None), "set_parameters", None)
        if callable(setter):
            setter(self.panel.values())

    def _resolve_parameter_reset(self, spec: ParameterSpec) -> dict[str, Any]:
        """Ask the active tool for input-derived values for a reset control."""
        files = self._input_files()
        try:
            method = getattr(self.tool, spec.reset_method)
            updates = method(files, self.panel.values())
        except Exception as exc:
            detail = _error_detail(exc)
            self._status_cb(f"Could not reset {spec.label.lower()}: {detail}")
            QMessageBox.warning(self, "Reset unavailable", detail)
            return {}
        self._status_cb(f"{spec.label} reset from the selected input data.")
        return updates

    def save_state(self) -> dict[str, Any]:
        """Capture current inputs and the last successful result."""
        from workbench.core.project import snapshot_result

        extra_state = None
        if self.extra is not None:
            save = getattr(self.extra, "save_state", None)
            if not callable(save):
                raise ValueError(f"{self.tool.name} does not support saving its custom inputs.")
            extra_state = save()
        return {
            "tool_id": self.tool.id,
            "files": self.file_input.files() if self.file_input is not None else [],
            "panel": self.panel.save_state(),
            "extra": extra_state,
            "result": (snapshot_result(self.last_result, self.last_artifacts)
                       if self.last_result is not None else None),
            "last_run": copy.deepcopy(self.last_run),
            "result_tab": self.results.tabs.currentIndex(),
            "splitter_sizes": self.splitter.sizes(),
        }

    def restore_state(self, state: dict[str, Any], project_dir: Path) -> None:
        """Restore saved inputs and results independently into a fresh page."""
        from workbench.core.project import restore_result

        if not isinstance(state, dict) or state.get("tool_id") != self.tool.id:
            raise ValueError(f"Invalid saved page for {self.tool.name}.")
        if self.file_input is not None:
            self.file_input.set_files(state["files"])
        self.panel.restore_state(state["panel"])
        if self.extra is not None:
            restore = getattr(self.extra, "restore_state", None)
            if not callable(restore):
                raise ValueError(f"{self.tool.name} does not support restoring its custom inputs.")
            restore(state["extra"])
        self._sync_extra_parameters()
        self.project_directory = project_dir
        self.last_run = state.get("last_run")
        saved_result = state.get("result")
        if saved_result is not None:
            self.last_result, self.last_artifacts = restore_result(saved_result)
            self.results.set_result(self.last_result)
            self.export_btn.set_artifacts(self.last_artifacts)
            self.export_btn.set_input_dir(str(project_dir))
        result_tab = state.get("result_tab", 0)
        if not isinstance(result_tab, int) or not 0 <= result_tab < self.results.tabs.count():
            raise ValueError(f"Invalid result tab for {self.tool.name}.")
        self.results.tabs.setCurrentIndex(result_tab)
        sizes = state.get("splitter_sizes", [480, 720])
        if (not isinstance(sizes, list) or len(sizes) != 2
                or any(not isinstance(size, int) or size < 0 for size in sizes)):
            raise ValueError(f"Invalid page layout for {self.tool.name}.")
        self.splitter.setSizes(sizes)

    # ------------------------------------------------------------------ run
    def _start(self, method: str, btn: QPushButton) -> None:
        if self._running:
            return
        files = self._input_files()
        if self.tool.needs_files and self.tool.files_required and not files:
            self._status_cb("Add at least one input file first.")
            QMessageBox.information(self, "No files", "Please add at least one input file.")
            return

        params = self.panel.values()
        extra_values = getattr(self.extra, "values", None)
        extra = extra_values() if callable(extra_values) else None
        self._pending_run = {
            "method": method, "files": [str(path) for path in files],
            "params": copy.deepcopy(params), "extra": copy.deepcopy(extra),
        }

        worker = _RunWorker(self.tool, method, files, params, extra)
        worker.signals.finished.connect(self._on_run_finished)
        worker.signals.failed.connect(self._on_run_failed)

        self._worker = worker  # keep a reference until it completes
        self._run_input_dir = str(files[0].parent) if files else None
        self._running = True
        self.runningChanged.emit(True)
        self._active_btn = btn
        self._active_label = btn.text()
        for button in self._action_buttons:
            button.setEnabled(False)
        btn.setText("Running…")
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._status_cb(f"Running {self.tool.name}…")

        try:
            QThreadPool.globalInstance().start(worker)
        except Exception as exc:
            self._on_run_failed(_error_detail(exc))

    def _finish_run(self) -> None:
        QGuiApplication.restoreOverrideCursor()
        self._running = False
        self.runningChanged.emit(False)
        self._worker = None
        if self._active_btn is not None:
            self._active_btn.setText(self._active_label)
        for button in self._action_buttons:
            button.setEnabled(True)
        self._active_btn = None

    @Slot(str)
    def _on_run_failed(self, detail: str) -> None:
        self._finish_run()
        self._status_cb(f"{self.tool.name} failed: {detail}")
        QMessageBox.critical(self, f"{self.tool.name} — error", detail)

    @Slot(object)
    def _on_run_finished(self, result: object) -> None:
        self._finish_run()
        if not isinstance(result, ToolResult):
            result = ToolResult()
        self.last_result = result
        self.last_run = self._pending_run
        self.results.set_result(result)
        self.last_artifacts = self.tool.exportable_artifacts(result)
        self.export_btn.set_artifacts(self.last_artifacts)
        # The file picker may have changed while the worker was running.
        self.export_btn.set_input_dir(
            str(self.project_directory) if self.project_directory else self._run_input_dir)
        if result.param_updates:
            self.panel.set_values(result.param_updates)
        self.panel.set_annotations(result.param_annotations)
        extra_setter = getattr(self.extra, "set_result_parameters", None)
        if callable(extra_setter):
            extra_setter(result.param_updates, result.param_annotations)

        warning_count = len(result.warnings)
        message = f"{self.tool.name}: done."
        if warning_count:
            message += (
                f"  {warning_count} warning{'s' if warning_count != 1 else ''}"
                + (" — see Summary." if result.display_summary
                   else " — details included in export.")
            )
        self._status_cb(message)
        self.changed.emit()


class GroupedToolPage(QWidget):
    """Keep each tool's inputs and results alive while switching functions."""

    def __init__(
        self, tool: Tool, status_cb: Callable[[str], None],
        parent: QWidget | None = None, subtools: list[Tool] | None = None,
    ) -> None:
        super().__init__(parent)
        self.tool = tool
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        selector = QHBoxLayout()
        selector.setContentsMargins(18, 0, 18, 0)
        selector.addWidget(QLabel("Function"))
        self.function_combo = QComboBox()
        selector.addWidget(self.function_combo, 1)
        selector.addStretch(1)
        root.addLayout(selector)
        self.stack = QStackedWidget()
        self.pages: list[ToolPage] = []
        for child in tool.subtools() if subtools is None else subtools:
            page = ToolPage(child, status_cb, self)
            self.pages.append(page)
            self.function_combo.addItem(child.name)
            self.stack.addWidget(page)
            page.runningChanged.connect(self._update_running_state)
        self.function_combo.currentIndexChanged.connect(self.stack.setCurrentIndex)
        self.function_combo.setCurrentIndex(0)
        root.addWidget(self.stack, 1)

    @Slot(bool)
    def _update_running_state(self, _running: bool) -> None:
        self.function_combo.setEnabled(not any(page._running for page in self.pages))

    def save_state(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool.id,
            "selected_tool": self.pages[self.stack.currentIndex()].tool.id,
            "pages": {page.tool.id: page.save_state() for page in self.pages},
        }

    def restore_state(self, state: dict[str, Any], project_dir: Path) -> None:
        if (not isinstance(state, dict) or state.get("tool_id") != self.tool.id
                or set(state.get("pages", {})) != {page.tool.id for page in self.pages}):
            raise ValueError(f"Incompatible functions in {self.tool.name}.")
        selected = state.get("selected_tool")
        ids = [page.tool.id for page in self.pages]
        if selected not in ids:
            raise ValueError(f"Invalid selected function in {self.tool.name}.")
        for page in self.pages:
            page.restore_state(state["pages"][page.tool.id], project_dir)
        self.function_combo.setCurrentIndex(ids.index(selected))


class MainWindow(QMainWindow):
    def __init__(self, tools: list[Tool]) -> None:
        super().__init__()
        self.setWindowTitle("Workbench")
        self.resize(1280, 820)
        self._tools = tools
        self._build()
        from workbench.project_controller import ProjectController
        self.project = ProjectController(self)

    def _build(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_sidebar())
        layout.addWidget(self._build_content(), 1)

        self.statusBar().showMessage("Ready.")
        self.nav.setCurrentRow(0)

    def _build_sidebar(self) -> QWidget:
        side = QWidget()
        side.setFixedWidth(204)
        side.setStyleSheet(f"background:{theme.SIDEBAR};")
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(0)

        brand = QLabel("Workbench")
        brand.setObjectName("Brand")
        sub = QLabel("Electrochemistry toolkit")
        sub.setObjectName("BrandSub")
        side_layout.addWidget(brand)
        side_layout.addWidget(sub)

        self.nav = QListWidget()
        self.nav.setObjectName("Sidebar")
        for tool in self._tools:
            QListWidgetItem(tool.name, self.nav)
        self.nav.currentRowChanged.connect(self._on_nav)
        side_layout.addWidget(self.nav, 1)
        return side

    def _build_content(self) -> QWidget:
        content = QWidget()
        content_layout = self.content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        header = QWidget()
        header.setStyleSheet(f"background:{theme.BG};")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(18, 14, 18, 8)
        header_layout.setSpacing(2)
        self.header_title = QLabel("")
        self.header_title.setObjectName("HeaderTitle")
        self.header_sub = QLabel("")
        self.header_sub.setObjectName("HeaderSub")
        self.header_sub.setWordWrap(True)
        header_layout.addWidget(self.header_title)
        header_layout.addWidget(self.header_sub)
        content_layout.addWidget(header)

        self.stack = self.create_tool_stack()
        content_layout.addWidget(self.stack, 1)
        return content

    def create_tool_stack(self) -> QStackedWidget:
        """Build pages in isolation so an invalid project cannot replace them."""
        stack = QStackedWidget()
        for tool in self._tools:
            subtools = tool.subtools()
            page = (
                GroupedToolPage(tool, self._set_status, subtools=subtools)
                if subtools else ToolPage(tool, self._set_status)
            )
            stack.addWidget(page)
        return stack

    def install_tool_stack(self, stack: QStackedWidget) -> None:
        previous = self.stack
        self.content_layout.replaceWidget(previous, stack)
        self.stack = stack
        previous.hide()
        previous.deleteLater()

    def _on_nav(self, row: int) -> None:
        if not 0 <= row < len(self._tools):
            return
        self.stack.setCurrentIndex(row)
        tool = self._tools[row]
        self.header_title.setText(tool.name)
        self.header_sub.setText(tool.subtitle)
        self._set_status("Ready.")

    def _set_status(self, text: str) -> None:
        self.statusBar().showMessage(text)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.project.request_close():
            event.accept()
        else:
            event.ignore()


def _filter_qt_messages(mode: QtMsgType, context, message: str) -> None:
    """Drop the benign 'QFont::setPointSize: Point size <= 0' noise that the
    matplotlib Qt canvas emits while rendering; forward everything else."""
    if "QFont::setPointSize" in message:
        return
    sys.stderr.write(message + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Workbench electrochemistry toolkit")
    parser.add_argument("project", nargs="?", type=Path, help="Workbench project (*.ecwb)")
    args = parser.parse_args()
    qInstallMessageHandler(_filter_qt_messages)
    app = cast(QApplication | None, QApplication.instance())
    if app is None:
        app = QApplication(sys.argv)
    app.setApplicationName("Workbench")
    app.setFont(QFont("Segoe UI", 10))
    theme.apply_matplotlib_style()
    app.setStyleSheet(theme.stylesheet())

    window = MainWindow(build_tools())
    window.show()
    if args.project is not None:
        window.project.open_path(args.project, prompt=False)
    sys.exit(app.exec())
