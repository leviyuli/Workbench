"""Schema-driven parameter panel: a tool declares ParameterSpec items and this
builds a consistent labelled form, grouped into sections. ``values()`` returns
the current parameter dict passed to ``Tool.run``.
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QSpinBox, QVBoxLayout, QWidget,
)

from workbench.core.tool_base import ParameterSpec


class SigFigSpinBox(QDoubleSpinBox):
    """A float spin box that DISPLAYS at most ``sig`` significant figures
    (adapting to the value's magnitude) while STORING full precision. No fixed
    decimal count is presented — small and large values both read cleanly,
    e.g. 0.0512, 28.9, 1e+05."""

    def __init__(self, sig: int = 3, parent=None):
        super().__init__(parent)
        self._sig = sig
        self.setDecimals(12)            # internal precision; display is custom
        self.setKeyboardTracking(False)

    def textFromValue(self, value: float) -> str:
        if value == 0:
            return "0"
        return f"{value:.{self._sig}g}"

    def valueFromText(self, text: str) -> float:
        try:
            return float(text)
        except ValueError:
            return self.value()

    def validate(self, text: str, pos: int):
        if text.strip() in ("", "-", "+", ".", "-.", "+.", "e", "E"):
            return (QValidator.Intermediate, text, pos)
        try:
            float(text)
            return (QValidator.Acceptable, text, pos)
        except ValueError:
            # Allow part-way scientific input such as "1e" or "2e-".
            if text and text[-1] in "eE+-.":
                return (QValidator.Intermediate, text, pos)
            return (QValidator.Invalid, text, pos)


class ParameterPanel(QWidget):
    def __init__(self, specs: list[ParameterSpec],
                 on_change: Callable[[], None] | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._specs = specs
        self._on_change = on_change
        self._widgets: dict[str, QWidget] = {}
        self._rows: dict[str, QWidget] = {}      # row container per key, for show_if
        self._build()
        self._apply_visibility()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        # Preserve declaration order of groups.
        groups: list[str] = []
        for s in self._specs:
            if s.group not in groups:
                groups.append(s.group)

        for group in groups:
            specs = [s for s in self._specs if s.group == group]
            if group:
                lbl = QLabel(group)
                lbl.setObjectName("SectionLabel")
                root.addWidget(lbl)
            form = QFormLayout()
            form.setLabelAlignment(Qt.AlignLeft)
            form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
            form.setHorizontalSpacing(12)
            form.setVerticalSpacing(8)
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            for spec in specs:
                widget = self._make_widget(spec)
                self._widgets[spec.key] = widget
                if spec.tooltip:
                    widget.setToolTip(spec.tooltip)
                field = widget
                if spec.suffix:
                    field = QWidget()
                    h = QHBoxLayout(field)
                    h.setContentsMargins(0, 0, 0, 0)
                    h.setSpacing(6)
                    h.addWidget(widget)
                    suf = QLabel(spec.suffix)
                    suf.setObjectName("Hint")
                    h.addWidget(suf)
                row_label = QLabel(spec.label)
                form.addRow(row_label, field)
                self._rows[spec.key] = field
                self._row_labels = getattr(self, "_row_labels", {})
                self._row_labels[spec.key] = row_label
            root.addLayout(form)

    def _make_widget(self, spec: ParameterSpec) -> QWidget:
        widget = self._make_control(spec)
        # Ensure values (and any suffix) stay readable without resizing the panel.
        widget.setMinimumWidth(130)
        return widget

    def _make_control(self, spec: ParameterSpec) -> QWidget:
        kind = spec.kind
        if kind == "float":
            # Show ≤3 significant figures, no preset decimal count.
            w = SigFigSpinBox(sig=3)
            w.setRange(spec.minimum, spec.maximum)
            w.setSingleStep(spec.step)
            w.setValue(float(spec.default))
            w.valueChanged.connect(self._changed)
            return w
        if kind == "int":
            w = QSpinBox()
            w.setRange(int(spec.minimum), int(spec.maximum))
            w.setSingleStep(int(spec.step))
            w.setValue(int(spec.default))
            w.valueChanged.connect(self._changed)
            return w
        if kind == "choice":
            w = QComboBox()
            w.addItems([str(c) for c in spec.choices])
            if spec.default in spec.choices:
                w.setCurrentIndex(spec.choices.index(spec.default))
            w.currentIndexChanged.connect(self._changed)
            return w
        if kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(spec.default))
            w.stateChanged.connect(self._changed)
            return w
        # str
        w = QLineEdit(str(spec.default))
        w.textChanged.connect(self._changed)
        return w

    def _changed(self, *_):
        self._apply_visibility()
        if self._on_change:
            self._on_change()

    def _apply_visibility(self) -> None:
        vals = self.values()
        for spec in self._specs:
            if spec.show_if is None:
                continue
            dep_key, dep_val = spec.show_if
            current = vals.get(dep_key)
            if isinstance(dep_val, (list, tuple, set)):
                visible = current in dep_val
            else:
                visible = current == dep_val
            self._rows[spec.key].setVisible(visible)
            lbl = getattr(self, "_row_labels", {}).get(spec.key)
            if lbl is not None:
                lbl.setVisible(visible)

    # --------------------------------------------------------------- public
    def set_values(self, updates: dict) -> None:
        """Programmatically set control values (e.g. a tool writing back a
        suggested frequency window). Signals are blocked per-widget so we only
        re-evaluate visibility and notify once at the end."""
        for key, value in updates.items():
            w = self._widgets.get(key)
            if w is None:
                continue
            w.blockSignals(True)
            try:
                if isinstance(w, QDoubleSpinBox):
                    w.setValue(float(value))
                elif isinstance(w, QSpinBox):
                    w.setValue(int(value))
                elif isinstance(w, QComboBox):
                    w.setCurrentText(str(value))
                elif isinstance(w, QCheckBox):
                    w.setChecked(bool(value))
                elif isinstance(w, QLineEdit):
                    w.setText(str(value))
            finally:
                w.blockSignals(False)
        self._apply_visibility()
        if self._on_change:
            self._on_change()

    def values(self) -> dict:
        out: dict = {}
        for spec in self._specs:
            w = self._widgets[spec.key]
            if isinstance(w, QDoubleSpinBox):
                out[spec.key] = w.value()
            elif isinstance(w, QSpinBox):
                out[spec.key] = w.value()
            elif isinstance(w, QComboBox):
                out[spec.key] = w.currentText()
            elif isinstance(w, QCheckBox):
                out[spec.key] = w.isChecked()
            elif isinstance(w, QLineEdit):
                out[spec.key] = w.text()
        return out
