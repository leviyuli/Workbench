"""Export menu backed by the current tool's declared artifacts."""
from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal, Slot
from PySide6.QtWidgets import QFileDialog, QMenu, QMessageBox, QToolButton

from workbench.core import io
from workbench.core.tool_base import Artifact

_FILTERS = {
    "csv": "CSV (*.csv)",
    "xlsx": "Excel (*.xlsx)",
    "workbook": "Excel (*.xlsx)",
    "png": "PNG image (*.png)",
    "txt": "Text (*.txt)",
}


def _default_dir(tool_id: str) -> str:
    base = Path(os.path.expanduser("~")) / "Documents" / "Workbench" / tool_id
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError:
        return os.path.expanduser("~")
    return str(base)


class _ExportSignals(QObject):
    finished = Signal(str, str)  # selected path, error (empty on success)


class _ExportWorker(QRunnable):
    def __init__(self, art: Artifact, path: str, write: Callable[[Artifact, str], None]) -> None:
        super().__init__()
        self.signals = _ExportSignals()
        self._artifact = art
        self._path = path
        self._write = write

    @Slot()
    def run(self) -> None:
        error = ""
        try:
            self._write(self._artifact, self._path)
        except Exception as exc:
            error = str(exc) or type(exc).__name__
        self.signals.finished.emit(self._path, error)


class ExportButton(QToolButton):
    def __init__(self, tool_id: str = "tool", parent=None):
        super().__init__(parent)
        self.setObjectName("Primary")
        self.setText("Export ▾")
        self.setPopupMode(QToolButton.InstantPopup)
        self.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self._tool_id = tool_id
        self._input_dir: str | None = None
        self._artifacts: list[Artifact] = []
        self._worker: _ExportWorker | None = None
        self._menu = QMenu(self)
        self.setMenu(self._menu)
        self.setEnabled(False)

    def set_tool_id(self, tool_id: str) -> None:
        self._tool_id = tool_id

    def set_input_dir(self, path: str | None) -> None:
        """Folder of the run's input files; the export dialog opens here first."""
        self._input_dir = path

    def _start_dir(self) -> str:
        if self._input_dir and os.path.isdir(self._input_dir):
            return self._input_dir
        return _default_dir(self._tool_id)

    def set_artifacts(self, artifacts: list[Artifact]) -> None:
        self._artifacts = artifacts or []
        self._menu.clear()
        for art in self._artifacts:
            self._menu.addAction(art.name, lambda a=art: self._save(a))
        self.setEnabled(bool(self._artifacts) and self._worker is None)

    # ------------------------------------------------------------------ save
    def _save(self, art: Artifact) -> None:
        if self._worker is not None:
            return
        start = os.path.join(self._start_dir(), art.default_filename)
        flt = _FILTERS.get(art.kind, "All files (*.*)")
        path, _ = QFileDialog.getSaveFileName(self, f"Export — {art.name}", start, flt)
        if not path:
            return
        if art.kind in {"xlsx", "workbook"}:
            worker = _ExportWorker(art, path, self._write)
            worker.signals.finished.connect(self._export_finished)
            self._worker = worker
            self.setText("Exporting…")
            self.setEnabled(False)
            try:
                QThreadPool.globalInstance().start(worker)
            except Exception as exc:
                self._export_finished(path, str(exc) or type(exc).__name__)
            return
        try:
            self._write(art, path)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        QMessageBox.information(self, "Export", f"Saved:\n{path}")

    @Slot(str, str)
    def _export_finished(self, path: str, error: str) -> None:
        self._worker = None
        self.setText("Export ▾")
        self.setEnabled(bool(self._artifacts))
        if error:
            QMessageBox.critical(self, "Export failed", error)
        else:
            QMessageBox.information(self, "Export", f"Saved:\n{path}")

    @staticmethod
    def _write(art: Artifact, path: str) -> None:
        if art.serialized_payload is not None:
            if art.kind in {"xlsx", "workbook"}:
                with io.atomic_output(path) as temporary:
                    temporary.write_bytes(art.serialized_payload)
            else:
                Path(path).write_bytes(art.serialized_payload)
        elif art.kind == "csv":
            io.write_csv(art.payload, path)
        elif art.kind == "xlsx":
            io.write_xlsx(art.payload, path)
        elif art.kind == "workbook":
            with io.atomic_output(path) as temporary:
                art.payload.save(temporary)
        elif art.kind == "png":
            io.save_figure(art.payload, path)
        elif art.kind == "txt":
            io.write_txt(art.payload, path)
        else:
            raise ValueError(f"Unknown artifact kind: {art.kind}")
