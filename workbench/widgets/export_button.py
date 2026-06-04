"""One export control with the same look everywhere. It reads the artifacts a
tool declares for the latest result and writes each via the unified writers in
``core.io``. A single click (one artifact) saves directly; multiple artifacts
open a dropdown menu.
"""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt
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
        self.setEnabled(bool(self._artifacts))

    # ------------------------------------------------------------------ save
    def _save(self, art: Artifact) -> None:
        start = os.path.join(self._start_dir(), art.default_filename)
        flt = _FILTERS.get(art.kind, "All files (*.*)")
        path, _ = QFileDialog.getSaveFileName(self, f"Export — {art.name}", start, flt)
        if not path:
            return
        try:
            self._write(art, path)
        except Exception as exc:  # surface, never crash
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        QMessageBox.information(self, "Export", f"Saved:\n{path}")

    @staticmethod
    def _write(art: Artifact, path: str) -> None:
        if art.kind == "csv":
            io.write_csv(art.payload, path)
        elif art.kind == "xlsx":
            io.write_xlsx(art.payload, path)
        elif art.kind == "workbook":
            art.payload.save(path)
        elif art.kind == "png":
            io.save_figure(art.payload, path)
        elif art.kind == "txt":
            io.write_txt(art.payload, path)
        else:
            raise ValueError(f"Unknown artifact kind: {art.kind}")
