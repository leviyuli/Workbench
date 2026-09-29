"""The workbench main window: a left sidebar selects one of the registered
tools; the central area shows that tool's inputs (left) and results (right).
Every tool is rendered by the same generic ``ToolPage`` so inputs, the Run
button, the Export button and the results area look identical throughout.
"""
from __future__ import annotations

import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

# Allow launching this file directly (e.g. the IDE Run button) as well as the
# canonical ``python -m workbench``. When run directly, the package's parent
# folder isn't on sys.path, so add it before importing the ``workbench`` package.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import (
    QObject, QRunnable, Qt, QThreadPool, Signal, Slot, QtMsgType,
    qInstallMessageHandler,
)
from PySide6.QtGui import QFont, QGuiApplication
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
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
    """Runs ``Tool.run`` off the GUI thread so the window stays responsive.

    Tools build plain ``matplotlib.figure.Figure`` objects (no pyplot, no Qt),
    so producing the result here is thread-safe; the figures are only wrapped in
    a Qt canvas later, on the GUI thread, in ``ResultsPanel.set_result``.
    """

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

        splitter = QSplitter(Qt.Orientation.Horizontal)
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
            self.tool.parameter_schema(), on_reset=self._resolve_parameter_reset)
        if self.file_input is not None:
            self.file_input.filesChanged.connect(self.panel.set_input_files)
            self.panel.set_input_files(self.file_input.files())
        left_card.add(self.panel)

        self.extra = self.tool.extra_widget(self, on_change=lambda: None)
        if self.extra is not None:
            left_card.add(self.extra)

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
        return self.file_input.files() if self.file_input is not None else []

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

        worker = _RunWorker(self.tool, method, files, params, extra)
        worker.signals.finished.connect(self._on_run_finished)
        worker.signals.failed.connect(self._on_run_failed)

        # Run off the GUI thread so the window stays responsive during long fits.
        self._worker = worker  # keep a reference until it completes
        self._run_input_dir = str(files[0].parent) if files else None
        self._running = True
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
        self.results.set_result(result)
        self.last_artifacts = self.tool.exportable_artifacts(result)
        self.export_btn.set_artifacts(self.last_artifacts)
        # The file picker may have changed while the worker was running.
        self.export_btn.set_input_dir(self._run_input_dir)
        if result.param_updates:
            self.panel.set_values(result.param_updates)
        # Show ± std-error notes next to fitted fields (clears stale ones).
        self.panel.set_annotations(result.param_annotations)

        warning_count = len(result.warnings)
        message = f"{self.tool.name}: done."
        if warning_count:
            message += (
                f"  {warning_count} warning{'s' if warning_count != 1 else ''}"
                " — see Summary."
            )
        self._status_cb(message)


class MainWindow(QMainWindow):
    def __init__(self, tools: list[Tool]) -> None:
        super().__init__()
        self.setWindowTitle("Workbench")
        self.resize(1280, 820)
        self._tools = tools
        self._build()

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
        content_layout = QVBoxLayout(content)
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

        self.stack = QStackedWidget()
        for tool in self._tools:
            self.stack.addWidget(ToolPage(tool, self._set_status))
        content_layout.addWidget(self.stack, 1)
        return content

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


def _filter_qt_messages(mode: QtMsgType, context, message: str) -> None:
    """Drop the benign 'QFont::setPointSize: Point size <= 0' noise that the
    matplotlib Qt canvas emits while rendering; forward everything else."""
    if "QFont::setPointSize" in message:
        return
    sys.stderr.write(message + "\n")


def main() -> None:
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
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
