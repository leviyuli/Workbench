"""A rounded white surface used to group related controls/results."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget


class Card(QFrame):
    """A titled card. Add child widgets to ``self.body`` (a QVBoxLayout)."""

    def __init__(self, title: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 16)
        outer.setSpacing(10)

        if title:
            lbl = QLabel(title)
            lbl.setObjectName("CardTitle")
            outer.addWidget(lbl)

        self.body = QVBoxLayout()
        self.body.setSpacing(8)
        outer.addLayout(self.body)

    def add(self, widget: QWidget) -> QWidget:
        self.body.addWidget(widget)
        return widget

    def add_stretch(self) -> None:
        self.body.addStretch(1)
