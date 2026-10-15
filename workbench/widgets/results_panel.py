"""Result tabs for plots, summaries and read-only tables."""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem, QTextEdit,
    QTabWidget, QVBoxLayout, QWidget,
)

from workbench.core.tool_base import ToolResult
from workbench.widgets.figure_view import SingleFigure, StaticFigure


def dataframe_to_table(df: pd.DataFrame) -> QTableWidget:
    table = QTableWidget()
    df = df.reset_index(drop=True)
    n_rows = min(len(df), 5000)  # Limit UI rendering; exports retain every row.
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
        self._clear_tabs()
        self.summary.setPlainText("No results yet. Configure inputs and run the tool.")
        self.tabs.addTab(self.summary, "Summary")

    def _clear_tabs(self) -> None:
        """Release old result pages while keeping the reusable summary."""
        while self.tabs.count():
            page = self.tabs.widget(0)
            self.tabs.removeTab(0)
            # Removing a tab leaves its page owned by the tab widget.
            if page is not self.summary:
                page.deleteLater()

    def set_result(self, result: ToolResult) -> None:
        self._clear_tabs()

        for name, fig in result.figures.items():
            self.tabs.addTab(SingleFigure(fig), name)
        for name, image in result.figure_images.items():
            if name not in result.figures:
                self.tabs.addTab(StaticFigure(image), name)

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
        if result.display_summary:
            self.tabs.addTab(self.summary, "Summary")
        else:
            self.summary.hide()

        if result.display_tables:
            for name, df in result.tables.items():
                self.tabs.addTab(dataframe_to_table(df), name)

        self.tabs.setCurrentIndex(0)

    def clear(self) -> None:
        self._show_placeholder()
