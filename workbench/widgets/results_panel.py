"""Uniform results area shared by every tool: a single tab strip holding the
plot tabs first, then a Summary (scalars + text + warnings), then one tab per
result table rendered as a read-only grid. Because all five tools funnel their
output through this widget, numeric and graphical output always look the same.
"""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem, QTextEdit,
    QTabWidget, QVBoxLayout, QWidget,
)

from workbench.core.tool_base import ToolResult
from workbench.widgets.figure_view import SingleFigure


def dataframe_to_table(df: pd.DataFrame) -> QTableWidget:
    table = QTableWidget()
    df = df.reset_index(drop=True)
    n_rows = min(len(df), 5000)            # guard against huge tables
    table.setRowCount(n_rows)
    table.setColumnCount(len(df.columns))
    table.setHorizontalHeaderLabels([str(c) for c in df.columns])
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
    for r in range(n_rows):
        for c, col in enumerate(df.columns):
            val = df.iat[r, c]
            text = f"{val:.6g}" if isinstance(val, float) else (
                "" if pd.isna(val) else str(val))
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(r, c, item)
    table.resizeColumnsToContents()
    return table


class ResultsPanel(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setStyleSheet("font-family:'Cascadia Code','Consolas',monospace;")
        self._show_placeholder()

    def _show_placeholder(self) -> None:
        self.tabs.clear()
        self.summary.setPlainText("No results yet. Configure inputs and run the tool.")
        self.tabs.addTab(self.summary, "Summary")

    def set_result(self, result: ToolResult) -> None:
        self.tabs.clear()

        # 1) Plots first.
        for name, fig in result.figures.items():
            self.tabs.addTab(SingleFigure(fig), name)

        # 2) Summary.
        lines: list[str] = []
        if result.scalars:
            lines.append("RESULTS")
            width = max((len(str(k)) for k in result.scalars), default=0)
            for k, v in result.scalars.items():
                vs = f"{v:.6g}" if isinstance(v, float) else str(v)
                lines.append(f"  {str(k).ljust(width)}  :  {vs}")
            lines.append("")
        for title, block in result.text_blocks.items():
            lines.append(title)
            lines.append(block.rstrip())
            lines.append("")
        if result.warnings:
            lines.append("WARNINGS")
            lines += [f"  • {w}" for w in result.warnings]
            lines.append("")
        if not lines:
            lines.append("Done. Use Export to save outputs.")
        self.summary.setPlainText("\n".join(lines))
        self.tabs.addTab(self.summary, "Summary")

        # 3) Tables.
        for name, df in result.tables.items():
            self.tabs.addTab(dataframe_to_table(df), name)

        self.tabs.setCurrentIndex(0)

    def clear(self) -> None:
        self._show_placeholder()
