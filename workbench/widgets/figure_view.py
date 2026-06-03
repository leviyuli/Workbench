"""Embeds matplotlib Figures with a consistent navigation toolbar. Multiple
figures are shown as inner tabs so every plotting tool looks the same."""
from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QStackedWidget, QTabWidget, QVBoxLayout, QWidget


class SingleFigure(QWidget):
    """One figure + its navigation toolbar."""

    def __init__(self, fig: Figure, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        self.canvas = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(toolbar)


class FigureView(QWidget):
    """Holds zero or more figures. Use ``set_figures({name: Figure})``."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedWidget()
        self.placeholder = QLabel("Run the tool to see plots here.")
        self.placeholder.setObjectName("Hint")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.tabs = QTabWidget()

        self.stack.addWidget(self.placeholder)  # index 0
        self.stack.addWidget(self.tabs)         # index 1
        layout.addWidget(self.stack)

    def set_figures(self, figures: dict[str, Figure]) -> None:
        self.tabs.clear()
        if not figures:
            self.stack.setCurrentIndex(0)
            return
        for name, fig in figures.items():
            self.tabs.addTab(SingleFigure(fig), name)
        self.stack.setCurrentIndex(1)

    def clear(self) -> None:
        self.tabs.clear()
        self.stack.setCurrentIndex(0)
