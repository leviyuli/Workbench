"""Embeds matplotlib Figures with a consistent navigation toolbar. Multiple
figures are shown as inner tabs so every plotting tool looks the same."""
from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QLabel, QStackedWidget, QTabWidget, QVBoxLayout, QWidget

from workbench import theme


def _recolor_toolbar_icons(toolbar: NavigationToolbar2QT, color_hex: str) -> None:
    """Force the navigation-toolbar icons to a fixed color.

    matplotlib tints these icons from the Qt *palette*, which follows the OS
    light/dark setting. Our window is always light via the stylesheet, so on a
    dark-mode OS the icons come out near-white on the cream toolbar and are hard
    to see. Repainting them keeps the icons readable regardless of the OS theme.
    """
    color = QColor(color_hex)
    size = toolbar.iconSize()
    for action in toolbar.actions():
        icon = action.icon()
        if icon.isNull():
            continue
        pm = icon.pixmap(size)
        if pm.isNull():
            continue
        tinted = QPixmap(pm.size())
        tinted.setDevicePixelRatio(pm.devicePixelRatio())
        tinted.fill(Qt.transparent)
        painter = QPainter(tinted)
        painter.drawPixmap(0, 0, pm)                       # icon shape (alpha)
        painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
        painter.fillRect(tinted.rect(), color)             # recolor, keep alpha
        painter.end()
        action.setIcon(QIcon(tinted))


class SingleFigure(QWidget):
    """One figure + its navigation toolbar."""

    def __init__(self, fig: Figure, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        self.canvas = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        _recolor_toolbar_icons(toolbar, theme.TEXT)
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
