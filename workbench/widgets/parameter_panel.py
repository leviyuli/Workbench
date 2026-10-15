"""Schema-driven parameter panel: a tool declares ParameterSpec items and this
builds a consistent labelled form, grouped into sections. ``values()`` returns
the current parameter dict passed to ``Tool.run``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from workbench.core.tool_base import ParameterSpec


class _NoWheel:
    """Pass wheel events to the surrounding scroll area."""

    def wheelEvent(self, event):  # noqa: N802 (Qt naming)
        event.ignore()


class SigFigSpinBox(_NoWheel, QDoubleSpinBox):
    """Display ``sig`` significant figures while retaining stored precision."""

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
    changed = Signal()

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
        if spec.read_only:
            if isinstance(widget, (QLineEdit, QDoubleSpinBox, QSpinBox)):
                widget.setReadOnly(True)
            else:
                widget.setEnabled(False)
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
            w.lineEdit().textEdited.connect(lambda _text: self.changed.emit())
            return w
        if kind == "int":
            w = _IntSpinBox()
            w.setRange(int(spec.minimum), int(spec.maximum))
            w.setSingleStep(int(spec.step))
            w.setValue(int(spec.default))
            w.valueChanged.connect(self._changed)
            w.lineEdit().textEdited.connect(lambda _text: self.changed.emit())
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
        self.changed.emit()

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
            self.set_annotations({})
            self.set_values(updates)
            return

        # Restore the named controls to schema defaults.
        specs_by_key = {spec.key: spec for spec in self._specs}
        updates = {
            key: specs_by_key[key].default
            for key in spec.reset_keys
            if key in specs_by_key and key in self._widgets
        }
        self.set_annotations({})
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
        """Update controls and notify once when their values change."""
        before = self.values()
        for key, value in updates.items():
            w = self._widgets.get(key)
            if w is None:
                continue
            previously_blocked = w.blockSignals(True)
            try:
                if isinstance(w, QDoubleSpinBox):
                    w.setValue(float(value))
                elif isinstance(w, QSpinBox):
                    w.setValue(int(value))
                elif isinstance(w, _InputFileComboBox):
                    index = w.findData(str(value) if value else "")
                    if index >= 0:
                        w.setCurrentIndex(index)
                elif isinstance(w, QComboBox):
                    w.setCurrentText(str(value))
                elif isinstance(w, QCheckBox):
                    w.setChecked(bool(value))
                elif isinstance(w, QLineEdit):
                    w.setText(str(value))
            finally:
                w.blockSignals(previously_blocked)
        self._apply_visibility()
        if self.values() != before:
            if self._on_change:
                self._on_change()
            self.changed.emit()

    def set_annotations(self, updates: dict) -> None:
        """Show a short note (e.g. '± 0.012 (2.4%)') next to each given field;
        any field not in ``updates`` has its note cleared. Pass ``{}`` to clear."""
        changed = False
        for key, ann in self._annotations.items():
            text = str(updates.get(key, "") or "")
            changed = changed or text != ann.text()
            ann.setText(text)
            ann.setVisible(bool(text))
        if changed:
            self.changed.emit()

    def set_input_files(self, paths: list[Path]) -> None:
        """Populate every ``input_file`` control from the shared file picker."""
        before = self.values()
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
            previously_blocked = w.blockSignals(True)
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
                w.blockSignals(previously_blocked)
        if self.values() != before:
            self._changed()

    def save_state(self) -> dict:
        """Capture editable and calculated fields, including hidden controls."""
        values = self.values()
        for spec in self._specs:
            if spec.kind == "input_file" and values.get(spec.key):
                values[spec.key] = Path(values[spec.key])
        return {
            "values": values,
            "annotations": {key: label.text() for key, label in self._annotations.items()
                            if label.text()},
        }

    def state(self) -> dict:
        """Alias for callers that use the common widget state convention."""
        return self.save_state()

    def restore_state(self, state: dict) -> None:
        """Restore after input-file choices have been populated by the page."""
        self.set_values(state.get("values", {}))
        self.set_annotations(state.get("annotations", {}))

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
