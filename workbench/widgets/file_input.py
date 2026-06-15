"""The one unified file-input widget every file-based tool uses.

It answers all the inconsistencies the user called out: it accepts a single
file or many, accepts a whole directory (expanded by the tool's filter),
supports drag-and-drop, and shows a tidy removable list. Tools only set the
accepted extensions and whether directories/multiple files are allowed.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QFileDialog, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QVBoxLayout, QWidget,
)

from workbench.core.io import expand_inputs


class FileInputWidget(QWidget):
    filesChanged = Signal(list)  # emits list[Path]

    def __init__(
        self,
        accepted_exts: tuple[str, ...] = (".csv", ".txt", ".xlsx", ".xls"),
        multiple: bool = True,
        accepts_directory: bool = False,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._exts = tuple(e.lower() for e in accepted_exts)
        self._multiple = multiple
        self._accepts_directory = accepts_directory
        self._paths: list[Path] = []
        self.setAcceptDrops(True)
        self._build()

    # ------------------------------------------------------------------ UI
    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self.btn_add = QPushButton("Add Files…")
        self.btn_add.clicked.connect(self._pick_files)
        btn_row.addWidget(self.btn_add)

        if self._accepts_directory:
            self.btn_dir = QPushButton("Add Folder…")
            self.btn_dir.clicked.connect(self._pick_folder)
            btn_row.addWidget(self.btn_dir)

        btn_row.addStretch(1)
        self.btn_remove = QPushButton("Remove")
        self.btn_remove.setToolTip("Remove the selected file(s) from the list.")
        self.btn_remove.setEnabled(False)
        self.btn_remove.clicked.connect(self._remove_selected)
        btn_row.addWidget(self.btn_remove)
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.clear)
        btn_row.addWidget(self.btn_clear)
        layout.addLayout(btn_row)

        self.list = QListWidget()
        self.list.setObjectName("FileList")
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.setMinimumHeight(96)
        self.list.itemSelectionChanged.connect(self._sync_remove_enabled)
        layout.addWidget(self.list)

        self.hint = QLabel("Drop files or a folder here, or use the buttons above.")
        self.hint.setObjectName("Hint")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)

        # Allow Delete key to remove selected rows.
        self.list.installEventFilter(self)

    # --------------------------------------------------------------- events
    def eventFilter(self, obj, event):  # noqa: N802 (Qt naming)
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QKeyEvent
        if obj is self.list and event.type() == QEvent.KeyPress:
            if isinstance(event, QKeyEvent) and event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
                self._remove_selected()
                return True
        return super().eventFilter(obj, event)

    def dragEnterEvent(self, event):  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):  # noqa: N802
        urls = event.mimeData().urls()
        dropped = [Path(u.toLocalFile()) for u in urls if u.toLocalFile()]
        self._add_paths(dropped)
        event.acceptProposedAction()

    # ---------------------------------------------------------------- logic
    def _pick_files(self) -> None:
        pattern = "Data files (" + " ".join(f"*{e}" for e in self._exts) + ");;All files (*.*)"
        if self._multiple:
            paths, _ = QFileDialog.getOpenFileNames(self, "Select files", "", pattern)
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select file", "", pattern)
            paths = [path] if path else []
        self._add_paths([Path(p) for p in paths if p])

    def _pick_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select folder")
        if folder:
            self._add_paths([Path(folder)])

    def _add_paths(self, raw: list[Path]) -> None:
        files = expand_inputs(raw, self._exts)
        if not files:
            if raw:
                self.hint.setText("No matching files found in the dropped selection.")
            return
        if not self._multiple:
            self._paths = files[:1]
        else:
            existing = {str(p) for p in self._paths}
            for f in files:
                if str(f) not in existing:
                    self._paths.append(f)
                    existing.add(str(f))
        self._refresh()

    def _remove_selected(self) -> None:
        rows = sorted((self.list.row(i) for i in self.list.selectedItems()), reverse=True)
        for r in rows:
            del self._paths[r]
        self._refresh()

    def _sync_remove_enabled(self) -> None:
        self.btn_remove.setEnabled(bool(self.list.selectedItems()))

    def _refresh(self) -> None:
        self.list.clear()
        for p in self._paths:
            item = QListWidgetItem(p.name)
            item.setToolTip(str(p))
            self.list.addItem(item)
        n = len(self._paths)
        self.hint.setText("No files selected." if n == 0
                          else f"{n} file{'s' if n != 1 else ''} selected.")
        self.filesChanged.emit(list(self._paths))

    # --------------------------------------------------------------- public
    def files(self) -> list[Path]:
        return list(self._paths)

    def clear(self) -> None:
        self._paths = []
        self._refresh()
