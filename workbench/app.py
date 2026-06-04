"""The workbench main window: a left sidebar selects one of the registered
tools; the central area shows that tool's inputs (left) and results (right).
Every tool is rendered by the same generic ``ToolPage`` so inputs, the Run
button, the Export button and the results area look identical throughout.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# Allow launching this file directly (e.g. the IDE Run button) as well as the
# canonical ``python -m workbench``. When run directly, the package's parent
# folder isn't on sys.path, so add it before importing the ``workbench`` package.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import (
    QObject, QRunnable, Qt, QThreadPool, Signal, Slot, QtMsgType,
    qInstallMessageHandler,
)
from PySide6.QtGui import QFont, QGuiApplication
from PySide6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QPushButton, QScrollArea, QSplitter, QStackedWidget, QVBoxLayout,
    QWidget,
)

from workbench import theme
from workbench.core.tool_base import Tool, ToolResult
from workbench.widgets.card import Card
from workbench.widgets.export_button import ExportButton
from workbench.widgets.file_input import FileInputWidget
from workbench.widgets.parameter_panel import ParameterPanel
from workbench.widgets.results_panel import ResultsPanel
from workbench.tools import build_tools


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

    def __init__(self, tool: Tool, files, params, extra):
        super().__init__()
        self.signals = _RunSignals()
        self._tool = tool
        self._files = files
        self._params = params
        self._extra = extra

    @Slot()
    def run(self) -> None:
        try:
            result = self._tool.run(self._files, self._params, self._extra)
        except Exception as exc:
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.signals.failed.emit(detail)
            return
        self.signals.finished.emit(result)


class ToolPage(QWidget):
    """Generic page that hosts one Tool's inputs and results."""

    def __init__(self, tool: Tool, status_cb, parent: QWidget | None = None):
        super().__init__(parent)
        self.tool = tool
        self._status_cb = status_cb
        self.last_artifacts: list = []
        self._running = False
        self._worker = None
        self._run_label = tool.run_button_label
        self._build()

    def _build(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(18, 12, 18, 18)
        root.setSpacing(14)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # ---- Left: inputs -------------------------------------------------
        left_card = Card("Inputs")
        self.file_input = None
        if self.tool.needs_files:
            self.file_input = FileInputWidget(
                accepted_exts=self.tool.accepted_file_types,
                multiple=self.tool.multiple_files,
                accepts_directory=self.tool.accepts_directory,
            )
            left_card.add(self.file_input)

        self.panel = ParameterPanel(self.tool.parameter_schema())
        left_card.add(self.panel)

        self.extra = self.tool.extra_widget(self, on_change=lambda: None)
        if self.extra is not None:
            left_card.add(self.extra)

        left_card.add_stretch()
        self.run_btn = QPushButton(self.tool.run_button_label)
        self.run_btn.setObjectName("Primary")
        self.run_btn.clicked.connect(self.run)
        left_card.add(self.run_btn)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QScrollArea.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setWidget(left_card)
        left_scroll.setMinimumWidth(400)
        left_scroll.setMaximumWidth(620)

        # ---- Right: results ----------------------------------------------
        right_card = Card("Results")
        top_row = QHBoxLayout()
        top_row.addStretch(1)
        self.export_btn = ExportButton(self.tool.id)
        top_row.addWidget(self.export_btn)
        right_card.body.addLayout(top_row)
        self.results = ResultsPanel()
        right_card.add(self.results)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_card)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([480, 720])

    # ------------------------------------------------------------------ run
    def run(self) -> None:
        if self._running:
            return
        files: list[Path] = self.file_input.files() if self.file_input else []
        if self.tool.needs_files and self.tool.files_required and not files:
            self._status_cb("Add at least one input file first.")
            QMessageBox.information(self, "No files", "Please add at least one input file.")
            return

        params = self.panel.values()
        extra = self.extra.values() if (self.extra is not None and hasattr(self.extra, "values")) else None

        # Run off the GUI thread so the window stays responsive during long fits.
        self._running = True
        self._run_label = self.run_btn.text()
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Running…")
        QGuiApplication.setOverrideCursor(Qt.WaitCursor)
        self._status_cb(f"Running {self.tool.name}…")

        worker = _RunWorker(self.tool, files, params, extra)
        worker.signals.finished.connect(self._on_run_finished)
        worker.signals.failed.connect(self._on_run_failed)
        self._worker = worker  # keep a reference until it completes
        QThreadPool.globalInstance().start(worker)

    def _finish_run(self) -> None:
        QGuiApplication.restoreOverrideCursor()
        self._running = False
        self._worker = None
        self.run_btn.setText(self._run_label)
        self.run_btn.setEnabled(True)

    def _on_run_failed(self, detail: str) -> None:
        self._finish_run()
        self._status_cb(f"{self.tool.name} failed: {detail}")
        QMessageBox.critical(self, f"{self.tool.name} — error", detail)

    def _on_run_finished(self, result) -> None:
        self._finish_run()
        if not isinstance(result, ToolResult):
            result = ToolResult()
        self.results.set_result(result)
        self.last_artifacts = self.tool.exportable_artifacts(result)
        self.export_btn.set_artifacts(self.last_artifacts)
        # Default the export dialog to the folder of this run's input files.
        input_dir = None
        if self.file_input is not None:
            fs = self.file_input.files()
            if fs:
                input_dir = str(Path(fs[0]).parent)
        self.export_btn.set_input_dir(input_dir)
        if result.param_updates:
            self.panel.set_values(result.param_updates)
        # Show ± std-error notes next to fitted fields (clears stale ones).
        self.panel.set_annotations(result.param_annotations)

        n_warn = len(result.warnings)
        msg = f"{self.tool.name}: done."
        if n_warn:
            msg += f"  {n_warn} warning{'s' if n_warn != 1 else ''} — see Summary."
        self._status_cb(msg)


class MainWindow(QMainWindow):
    def __init__(self, tools: list[Tool]):
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

        # ---- Sidebar ------------------------------------------------------
        side = QWidget()
        side.setFixedWidth(212)
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
        layout.addWidget(side)

        # ---- Content (header + stacked pages) ----------------------------
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        header = QWidget()
        header.setStyleSheet(f"background:{theme.BG};")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(20, 16, 20, 8)
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
        layout.addWidget(content, 1)

        self.statusBar().showMessage("Ready.")
        self.nav.setCurrentRow(0)

    def _on_nav(self, row: int) -> None:
        if row < 0:
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
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Workbench")
    app.setFont(QFont("Segoe UI", 10))
    theme.apply_matplotlib_style()
    app.setStyleSheet(theme.stylesheet())

    window = MainWindow(build_tools())
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
