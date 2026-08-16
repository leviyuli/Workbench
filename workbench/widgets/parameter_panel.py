"""Schema-driven parameter panel: a tool declares ParameterSpec items and this
builds a consistent labelled form, grouped into sections. ``values()`` returns
the current parameter dict passed to ``Tool.run``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from workbench.core.tool_base import ParameterSpec


class _NoWheel:
    """Mixin that makes a control ignore mouse-wheel scrolling, so hovering a
    field while scrolling the panel can never accidentally change its value.
    The event is passed to the parent (the scroll area) instead, so the panel
    still scrolls as expected."""

    def wheelEvent(self, event):  # noqa: N802 (Qt naming)
        event.ignore()


class SigFigSpinBox(_NoWheel, QDoubleSpinBox):
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


class _IntSpinBox(_NoWheel, QSpinBox):
    """Integer spin box that ignores the mouse wheel (see ``_NoWheel``)."""


class _ChoiceComboBox(_NoWheel, QComboBox):
    """Dropdown that ignores the mouse wheel (see ``_NoWheel``)."""


class _InputFileComboBox(_ChoiceComboBox):
    """A choice populated from the page's current input-file list."""


class ParameterPanel(QWidget):
    def __init__(self, specs: list[ParameterSpec],
                 on_change: Callable[[], None] | None = None,
                 on_reset: Callable[[ParameterSpec], dict | None] | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._specs = specs
        self._on_change = on_change
        self._on_reset = on_reset
        self._widgets: dict[str, QWidget] = {}
        self._rows: dict[str, QWidget] = {}      # row container per key, for show_if
        self._row_labels: dict[str, QLabel] = {}
        self._annotations: dict[str, QLabel] = {}  # per-key note label (e.g. ± SE)
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
                # Every field is wrapped so an annotation (e.g. ± SE) can sit to
                # its right. The control keeps stretch 1, so it stays full-width
                # until an annotation is shown, then yields just enough room.
                field = QWidget()
                h = QHBoxLayout(field)
                h.setContentsMargins(0, 0, 0, 0)
                h.setSpacing(6)
                h.addWidget(widget, 1)
                if spec.suffix:
                    suf = QLabel(spec.suffix)
                    suf.setObjectName("Hint")
                    h.addWidget(suf)
                ann = QLabel("")
                ann.setObjectName("Hint")
                ann.setVisible(False)
                h.addWidget(ann)
                self._annotations[spec.key] = ann
                row_label = QLabel(spec.label)
                form.addRow(row_label, field)
                self._rows[spec.key] = field
                self._row_labels[spec.key] = row_label
            root.addLayout(form)

    def _make_widget(self, spec: ParameterSpec) -> QWidget:
        widget = self._make_control(spec)
        # Ensure values (and any suffix) stay readable without resizing the panel.
        widget.setMinimumWidth(130)
        return widget

    def _make_control(self, spec: ParameterSpec) -> QWidget:
        kind = spec.kind
        if kind == "reset":
            w = QPushButton(str(spec.default or "Reset"))
            w.clicked.connect(
                lambda _checked=False, reset_spec=spec:
                self._reset_values(reset_spec)
            )
            return w
        if kind == "float":
            # Show ≤3 significant figures, no preset decimal count.
            w = SigFigSpinBox(sig=3)
            w.setRange(spec.minimum, spec.maximum)
            w.setSingleStep(spec.step)
            w.setValue(float(spec.default))
            w.valueChanged.connect(self._changed)
            return w
        if kind == "int":
            w = _IntSpinBox()
            w.setRange(int(spec.minimum), int(spec.maximum))
            w.setSingleStep(int(spec.step))
            w.setValue(int(spec.default))
            w.valueChanged.connect(self._changed)
            return w
        if kind == "choice":
            w = _ChoiceComboBox()
            w.addItems([str(c) for c in spec.choices])
            if spec.default in spec.choices:
                w.setCurrentIndex(spec.choices.index(spec.default))
            w.currentIndexChanged.connect(self._changed)
            return w
        if kind == "input_file":
            w = _InputFileComboBox()
            w.addItem("Add an input file first", "")
            w.setEnabled(False)
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
        # A manual edit invalidates any fit annotation (it referred to the old
        # value), so clear them all.
        self._clear_annotations()
        if self._on_change:
            self._on_change()

    def _clear_annotations(self) -> None:
        for ann in self._annotations.values():
            if ann.text():
                ann.setText("")
                ann.setVisible(False)

    def _reset_values(self, spec: ParameterSpec) -> None:
        """Resolve and apply the values associated with a reset control."""
        updates = None
        if spec.reset_method and self._on_reset is not None:
            updates = self._on_reset(spec)
        if updates is not None:
            self._clear_annotations()
            self.set_values(updates)
            return

        # Generic fallback: restore the named controls to schema defaults.
        specs_by_key = {spec.key: spec for spec in self._specs}
        updates = {
            key: specs_by_key[key].default
            for key in spec.reset_keys
            if key in specs_by_key and key in self._widgets
        }
        self._clear_annotations()
        self.set_values(updates)

    def _apply_visibility(self) -> None:
        vals = self.values()
        for spec in self._specs:
            if spec.show_if is None:
                continue
            # Accept a single (key, value) tuple or a list of them (AND-combined).
            conditions = spec.show_if
            if conditions and isinstance(conditions[0], str):
                conditions = [conditions]
            visible = True
            for dep_key, dep_val in conditions:
                current = vals.get(dep_key)
                if isinstance(dep_val, (list, tuple, set)):
                    ok = current in dep_val
                else:
                    ok = current == dep_val
                if not ok:
                    visible = False
                    break
            self._rows[spec.key].setVisible(visible)
            self._row_labels[spec.key].setVisible(visible)

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

    def set_annotations(self, updates: dict) -> None:
        """Show a short note (e.g. '± 0.012 (2.4%)') next to each given field;
        any field not in ``updates`` has its note cleared. Pass ``{}`` to clear."""
        for key, ann in self._annotations.items():
            text = str(updates.get(key, "") or "")
            ann.setText(text)
            ann.setVisible(bool(text))

    def set_input_files(self, paths: list[Path]) -> None:
        """Populate every ``input_file`` control from the shared file picker."""
        normalized = [Path(path) for path in paths]
        name_counts: dict[str, int] = {}
        for path in normalized:
            name_counts[path.name] = name_counts.get(path.name, 0) + 1
        for spec in self._specs:
            if spec.kind != "input_file":
                continue
            w = self._widgets[spec.key]
            if not isinstance(w, _InputFileComboBox):
                continue
            previous = w.currentData()
            w.blockSignals(True)
            try:
                w.clear()
                for path in normalized:
                    label = path.name
                    if name_counts[path.name] > 1:
                        label = f"{path.name} — {path.parent.name}"
                    w.addItem(label, str(path))
                    w.setItemData(w.count() - 1, str(path), Qt.ToolTipRole)
                w.setEnabled(bool(normalized))
                if not normalized:
                    w.addItem("Add an input file first", "")
                elif previous:
                    match = w.findData(previous)
                    if match >= 0:
                        w.setCurrentIndex(match)
            finally:
                w.blockSignals(False)

    def values(self) -> dict:
        out: dict = {}
        for spec in self._specs:
            w = self._widgets[spec.key]
            if isinstance(w, QDoubleSpinBox):
                out[spec.key] = w.value()
            elif isinstance(w, QSpinBox):
                out[spec.key] = w.value()
            elif isinstance(w, QComboBox):
                if spec.kind == "input_file":
                    out[spec.key] = w.currentData() or ""
                else:
                    out[spec.key] = w.currentText()
            elif isinstance(w, QCheckBox):
                out[spec.key] = w.isChecked()
            elif isinstance(w, QLineEdit):
                out[spec.key] = w.text()
        return out
