"""Persistent, independent CL EIS comparison cases for the generator page."""
from __future__ import annotations

from copy import deepcopy
import re

from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QHBoxLayout, QHeaderView, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from workbench.tools.eis_generator.logic import DEFAULT_PARAMETERS, validate_parameters


_CASE_ID = re.compile(r"case_([1-9][0-9]*)\Z")


def _validated_core(params: dict, *, complete: bool = False) -> dict:
    """Ignore plot controls, retaining validated physical and frequency inputs."""
    if not isinstance(params, dict):
        raise ValueError("Case parameters must be a mapping.")
    if complete and set(params) != set(DEFAULT_PARAMETERS):
        raise ValueError("Saved case parameters must contain all eight model inputs.")
    core = {key: deepcopy(params.get(key, default))
            for key, default in DEFAULT_PARAMETERS.items()}
    validated = validate_parameters(core)
    return {key: deepcopy(validated[key]) for key in DEFAULT_PARAMETERS}


class ClEisCasesWidget(QWidget):
    """Keep overlays on the GUI thread and hand workers copied snapshots only."""

    def __init__(self, parent=None, on_change=None):
        super().__init__(parent)
        self._on_change = on_change
        self._parameters = deepcopy(DEFAULT_PARAMETERS)
        self._cases: list[dict] = []
        self._annotated_case = "current"
        self._next_case_number = 1

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        heading = QLabel("Saved comparison cases")
        heading.setObjectName("SectionLabel")
        layout.addWidget(heading)
        self.hint = QLabel(
            "Save a snapshot of the current inputs, then edit inputs to compare. "
            "Press Generate to apply overlay and annotation changes."
        )
        self.hint.setObjectName("Hint")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)

        buttons = QHBoxLayout()
        self.btn_save = QPushButton("Save current case")
        self.btn_remove = QPushButton("Remove selected")
        self.btn_clear = QPushButton("Clear saved")
        self.btn_save.clicked.connect(self._save_clicked)
        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_clear.clicked.connect(self.clear_cases)
        for button in (self.btn_save, self.btn_remove, self.btn_clear):
            buttons.addWidget(button)
        layout.addLayout(buttons)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Show", "Case name", "κᵢ / σₑ / L"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setSortingEnabled(False)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(110)
        self.table.setMaximumHeight(240)
        self.table.itemChanged.connect(self._item_changed)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        layout.addWidget(self.table)

        annotation_row = QHBoxLayout()
        annotation_row.addWidget(QLabel("Annotate case"))
        self.annotation_combo = QComboBox()
        self.annotation_combo.setToolTip(
            "Choose one visible curve for the CL HFR and transmission-line width annotations."
        )
        self.annotation_combo.currentIndexChanged.connect(self._annotation_changed)
        annotation_row.addWidget(self.annotation_combo, 1)
        layout.addLayout(annotation_row)
        self.error_label = QLabel()
        self.error_label.setObjectName("Hint")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        layout.addWidget(self.error_label)
        self._rebuild()

    def _assert_gui_thread(self) -> None:
        if QThread.currentThread() != self.thread():
            raise RuntimeError("CL EIS cases must be accessed on their GUI thread.")

    def _changed(self) -> None:
        self.error_label.clear()
        self.error_label.hide()
        if self._on_change is not None:
            self._on_change()

    def _error(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_label.show()

    def set_parameters(self, params: dict) -> None:
        """Cache a fresh panel snapshot; validate only when saving/generating."""
        self._assert_gui_thread()
        if not isinstance(params, dict):
            raise ValueError("Current parameters must be a mapping.")
        self._parameters = deepcopy(params)

    def add_current_case(self, name: str | None = None) -> str:
        """Validate and save the current inputs without retaining mutable aliases."""
        self._assert_gui_thread()
        params = _validated_core(self._parameters)
        number = self._next_case_number
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise ValueError("A case name must contain text.")
        case_id = f"case_{number}"
        self._cases.append({
            "id": case_id, "name": name.strip() if name is not None else f"Case {number}",
            "visible": True, "params": params,
        })
        self._next_case_number += 1
        self._rebuild()
        self._changed()
        return case_id

    def _save_clicked(self, _checked=False) -> None:
        try:
            self.add_current_case()
        except (TypeError, ValueError, OverflowError) as exc:
            self._error(f"Cannot save case: {exc}")

    def remove_selected(self, _checked=False) -> None:
        self._assert_gui_thread()
        rows = {index.row() for index in self.table.selectedIndexes()}
        if not rows:
            return
        self._cases = [case for row, case in enumerate(self._cases) if row not in rows]
        self._rebuild()
        self._changed()

    def clear_cases(self, _checked=False) -> None:
        self._assert_gui_thread()
        if not self._cases:
            return
        self._cases = []
        self._rebuild()
        self._changed()

    def _update_buttons(self) -> None:
        self.btn_remove.setEnabled(bool(self.table.selectedIndexes()))
        self.btn_clear.setEnabled(bool(self._cases))

    def _rebuild(self) -> None:
        blocked = self.table.blockSignals(True)
        try:
            self.table.setRowCount(len(self._cases))
            for row, case in enumerate(self._cases):
                show = QTableWidgetItem()
                show.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                              | Qt.ItemFlag.ItemIsUserCheckable)
                show.setCheckState(Qt.CheckState.Checked if case["visible"] else Qt.CheckState.Unchecked)
                show.setData(Qt.ItemDataRole.UserRole, case["id"])
                name = QTableWidgetItem(case["name"])
                name.setData(Qt.ItemDataRole.UserRole, case["id"])
                p = case["params"]
                detail = QTableWidgetItem(
                    f'{p["ionic_conductivity"]:.3g} / {p["electronic_conductivity"]:.3g} S/cm'
                    f' / {p["thickness_um"]:.3g} µm'
                )
                detail.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                tooltip = (
                    f'κᵢ = {p["ionic_conductivity"]:g} S/cm; '
                    f'σₑ = {p["electronic_conductivity"]:g} S/cm; '
                    f'L = {p["thickness_um"]:g} µm\n'
                    f'C = {p["capacitance_mf_cm2"]:g} mF/cm²; '
                    f'Rexternal = {p["external_resistance"]:g} Ω cm²\n'
                    f'{p["f_min"]:g}–{p["f_max"]:g} Hz; {p["n_points"]} points'
                )
                for column, item in enumerate((show, name, detail)):
                    item.setToolTip(tooltip)
                    self.table.setItem(row, column, item)
        finally:
            self.table.blockSignals(blocked)
        self._rebuild_annotation_choices()
        self._update_buttons()

    def _rebuild_annotation_choices(self) -> None:
        visible = [case for case in self._cases if case["visible"]]
        valid = {"current", *(case["id"] for case in visible)}
        if self._annotated_case not in valid:
            self._annotated_case = "current"
        blocked = self.annotation_combo.blockSignals(True)
        try:
            self.annotation_combo.clear()
            self.annotation_combo.addItem("Current case", "current")
            for case in visible:
                self.annotation_combo.addItem(case["name"], case["id"])
            self.annotation_combo.setCurrentIndex(self.annotation_combo.findData(self._annotated_case))
        finally:
            self.annotation_combo.blockSignals(blocked)

    def _item_changed(self, item: QTableWidgetItem) -> None:
        row, column = item.row(), item.column()
        if not 0 <= row < len(self._cases):
            return
        case = self._cases[row]
        if column == 0:
            visible = item.checkState() == Qt.CheckState.Checked
            if visible == case["visible"]:
                return
            case["visible"] = visible
        elif column == 1:
            name = item.text().strip()
            blocked = self.table.blockSignals(True)
            try:
                item.setText(name if name else case["name"])
            finally:
                self.table.blockSignals(blocked)
            if not name:
                self._error("A case name must contain text.")
                return
            if name == case["name"]:
                return
            case["name"] = name
        else:
            return
        self._rebuild_annotation_choices()
        self._changed()

    def _annotation_changed(self, _index: int) -> None:
        choice = self.annotation_combo.currentData()
        if choice is None or choice == self._annotated_case:
            return
        self._annotated_case = choice
        self._changed()

    def values(self) -> dict:
        """Return a JSON-friendly deep copy, safe to pass to a worker."""
        self._assert_gui_thread()
        return deepcopy({
            "cases": self._cases,
            "annotated_case": self._annotated_case,
            "next_case_number": self._next_case_number,
        })

    def save_state(self) -> dict:
        return self.values()

    def restore_state(self, state: dict) -> None:
        """Validate the entire payload before replacing any existing state."""
        self._assert_gui_thread()
        if not isinstance(state, dict):
            raise ValueError("Saved CL EIS comparison state must be a mapping.")
        if set(state) - {"cases", "annotated_case", "next_case_number"}:
            raise ValueError("Saved CL EIS comparison state contains unknown fields.")
        cases = state.get("cases", [])
        if not isinstance(cases, list):
            raise ValueError("Saved CL EIS cases must be a list.")
        validated = []
        identifiers = set()
        max_number = 0
        for case in cases:
            if not isinstance(case, dict) or set(case) != {"id", "name", "visible", "params"}:
                raise ValueError("Each saved case requires id, name, visible and params.")
            identifier = case["id"]
            match = _CASE_ID.fullmatch(identifier) if isinstance(identifier, str) else None
            if match is None or identifier in identifiers:
                raise ValueError("Saved case IDs must be unique case_N identifiers.")
            if not isinstance(case["name"], str) or not case["name"].strip():
                raise ValueError("A saved case name must contain text.")
            if not isinstance(case["visible"], bool):
                raise ValueError("Saved case visibility must be true or false.")
            params = _validated_core(case["params"], complete=True)
            validated.append({"id": identifier, "name": case["name"].strip(),
                              "visible": case["visible"], "params": params})
            identifiers.add(identifier)
            max_number = max(max_number, int(match.group(1)))
        next_number = state.get("next_case_number", max_number + 1)
        if isinstance(next_number, bool) or not isinstance(next_number, int) or next_number <= max_number:
            raise ValueError("The next saved-case number must exceed all existing IDs.")
        annotated = state.get("annotated_case", "current")
        if not isinstance(annotated, str):
            raise ValueError("The annotated case must be an ID or current.")
        valid_annotations = {"current", *(case["id"] for case in validated if case["visible"])}
        if annotated not in valid_annotations:
            annotated = "current"
        before = self.values()
        self._cases = validated
        self._next_case_number = next_number
        self._annotated_case = annotated
        self._rebuild()
        if self.values() != before:
            self._changed()
