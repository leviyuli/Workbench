"""Upload or paste current-dependent HFR measurements for spline fitting."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog, QFormLayout, QHBoxLayout, QLabel, QMessageBox,
    QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)

from workbench.tools.lsv.hfr_data import parse_hfr_pairs, parse_hfr_text, read_hfr_file
from workbench.widgets.parameter_panel import SigFigSpinBox


class HfrInputs(QWidget):
    """Keep the complete text input portable, including unfinished edits."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None,
                 on_change: Callable[[], None] | None = None):
        super().__init__(parent)
        self._source = ""
        self._updating = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        upload = QHBoxLayout()
        self.upload_button = QPushButton("Upload HFR file")
        self.upload_button.clicked.connect(self._choose_file)
        upload.addWidget(self.upload_button)
        self.source_label = QLabel("or paste below")
        self.source_label.setObjectName("Hint")
        self.source_label.setWordWrap(True)
        upload.addWidget(self.source_label, 1)
        layout.addLayout(upload)

        help_text = QLabel(
            "First column: |j| in A/cm². Second column: HFR in Ω·cm². "
            "Paste two columns from a spreadsheet or upload a file."
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

        self.editor = QPlainTextEdit()
        self.text_input = self.editor
        self.editor.setPlaceholderText("|j|\tHFR\n0.1\t0.06077\n0.5\t0.06191\n1\t0.05974")
        self.editor.setAccessibleName("HFR measurements")
        self.editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.editor.setTabStopDistance(self.editor.fontMetrics().horizontalAdvance("0.00000") + 18)
        self.editor.setToolTip("Paste current density followed by one HFR column. "
                               "Tabs, commas and spaces are accepted; a header row is optional.")
        self.editor.setMinimumHeight(125)
        self.editor.setMaximumHeight(190)
        self.editor.textChanged.connect(self._text_changed)
        layout.addWidget(self.editor)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(6)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.method_label = QLabel("Cubic smoothing spline")
        form.addRow("Fit", self.method_label)
        self.smoothing_input = SigFigSpinBox(sig=3)
        self.smoothing_input.setRange(0.0, 1e6)
        self.smoothing_input.setSingleStep(0.0001)
        self.smoothing_input.setValue(0.0001)
        self.smoothing_input.setAccessibleName("HFR smoothing strength")
        self.smoothing_input.setToolTip(
            "Larger values give a smoother curve. Zero follows the measured points. "
            "The current range is normalized before fitting so the setting is comparable across tests."
        )
        self.smoothing_input.valueChanged.connect(self._smoothing_changed)
        form.addRow("Smoothing", self.smoothing_input)
        layout.addLayout(form)

        self.status_label = QLabel()
        self.status_label.setObjectName("Hint")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        if on_change is not None:
            self.changed.connect(on_change)
        self._refresh_status()
        self.set_parameters({})

    def set_parameters(self, params: dict) -> None:
        self.setVisible(params.get("hfr_mode", "Constant HFR") == "HFR vs |j|")

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Upload HFR measurements", "",
            "HFR data (*.txt *.csv *.tsv *.xlsx *.xls);;All files (*)",
        )
        if not path:
            return
        try:
            self.load_file(path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Cannot load HFR data", str(exc))

    def load_file(self, path: str | Path) -> None:
        """Copy uploaded data into the editor; no file dependency remains."""
        text = read_hfr_file(path)
        parse_hfr_pairs(text)
        self._set_input(text, self.smoothing_input.value(), Path(path).name)

    def _text_changed(self) -> None:
        if self._updating:
            return
        self._source = ""
        self.source_label.setText("Pasted / edited data")
        self._refresh_status()
        self.changed.emit()

    def _smoothing_changed(self, _value: float) -> None:
        if not self._updating:
            self.changed.emit()

    def _refresh_status(self) -> None:
        text = self.editor.toPlainText()
        if not text.strip():
            self.status_label.setText("Add at least five measured HFR points.")
        else:
            try:
                data = parse_hfr_pairs(text)
            except ValueError as exc:
                # Preserve unfinished pasted text; validation occurs when fitting.
                self.status_label.setText(str(exc))
                return
            points = data.points()
            if points:
                current = [point[0] for point in points]
                count = f"{len(points)} point{'s' if len(points) != 1 else ''}"
                self.status_label.setText(
                    f"{count} · |j| {min(current):g}–{max(current):g} A/cm². "
                    + ("Add at least five points to fit HFR." if len(points) < 5
                       else "The HFR fit plot is included in the results.")
                )
            else:
                self.status_label.setText("The HFR column has no measurements.")

    def _set_input(self, text: str, smoothing: float, source: str) -> None:
        before = self.values()
        self._updating = True
        try:
            self._source = source
            self.editor.setPlainText(text)
            self.smoothing_input.setValue(smoothing)
            self.source_label.setText(source or "or paste below")
            self._refresh_status()
        finally:
            self._updating = False
        if self.values() != before:
            self.changed.emit()

    @staticmethod
    def _validated_rows(rows) -> list[list[str]]:
        if not isinstance(rows, list) or any(
            not isinstance(row, list) or len(row) != 2
            or any(not isinstance(cell, str) for cell in row)
            for row in rows
        ):
            raise ValueError("Saved HFR points must be a list of two text cells per row.")
        return [row.copy() for row in rows]

    def set_rows(self, rows: list[list[str]]) -> None:
        """Accept historical point-editor data without requiring numeric completion."""
        rows = self._validated_rows(rows)
        self._set_input("\n".join("\t".join(row) for row in rows),
                        self.smoothing_input.value(), "")

    def rows(self) -> list[list[str]]:
        """Return historical two-column text used by saved point-editor projects."""
        return [(line.split("\t") + [""])[:2] for line in self.editor.toPlainText().splitlines()]

    def add_row(self, values: list[str] | None = None) -> None:
        values = ["", ""] if values is None or isinstance(values, bool) else values
        self.set_rows([*self.rows(), self._validated_rows([values])[0]])

    def values(self) -> dict:
        return {"hfr_text": self.editor.toPlainText(),
                "hfr_smoothing": self.smoothing_input.value(), "hfr_source": self._source}

    def save_state(self) -> dict:
        return self.values()

    def restore_state(self, state: dict | None) -> None:
        if state is None:
            self._set_input("", 0.0001, "")
            return
        if isinstance(state, dict) and set(state) == {"hfr_points"}:
            self.set_rows(state["hfr_points"])
            return
        keys = {"hfr_text", "hfr_smoothing", "hfr_source"}
        if not isinstance(state, dict) or set(state) not in (keys, keys | {"hfr_column"}):
            raise ValueError("Saved HFR inputs must contain measurement text and smoothing.")
        text, smoothing, source = (state[key] for key in
                                   ("hfr_text", "hfr_smoothing", "hfr_source"))
        column = state.get("hfr_column", 0)
        if (not isinstance(text, str) or not isinstance(source, str)
                or not isinstance(column, int) or isinstance(column, bool) or column < 0
                or not isinstance(smoothing, (int, float)) or isinstance(smoothing, bool)
                or not math.isfinite(smoothing) or not 0 <= smoothing <= 1e6):
            raise ValueError("Saved HFR inputs contain invalid text, column or smoothing.")
        if "hfr_column" in state:
            try:
                data = parse_hfr_text(text)
            except ValueError:
                # Keep historical unfinished text intact until the user completes it.
                pass
            else:
                if len(data.column_names) > 1:
                    points = data.points(column)
                    text = "|j|\t" + data.column_names[column] + "\n" + "\n".join(
                        f"{current!r}\t{hfr!r}" for current, hfr in points)
        self._set_input(text, float(smoothing), source)
