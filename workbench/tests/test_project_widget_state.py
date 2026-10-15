"""Project state captures user input independently of calculation payloads."""
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QPushButton, QTableWidgetItem

from workbench.core.tool_base import ParameterSpec
from workbench.tools.ink_calc.tool import InkCalcTool, IonomerExtrasWidget
from workbench.tools.integrated_re.inputs import IntegratedReInputs
from workbench.widgets.file_input import FileInputWidget
from workbench.widgets.parameter_panel import ParameterPanel


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance()
    if app is None:
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("QT_QPA_PLATFORM", "offscreen")
            app = QApplication([])
    yield app


def test_file_selection_restores_order_and_duplicate_name_identity(qt_app, tmp_path):
    first = tmp_path / "first" / "same.txt"
    second = tmp_path / "second" / "same.txt"
    widget = FileInputWidget()
    notifications = []
    widget.filesChanged.connect(notifications.append)
    try:
        widget.set_files([second, first, second])
        assert widget.files() == [second, first]
        assert widget.list.item(0).toolTip() == str(second)
        assert len(notifications) == 1
        widget.set_files([second, first])
        assert len(notifications) == 1
        widget.clear()
        assert widget.files() == []
        assert len(notifications) == 2
    finally:
        widget.close()


def test_single_file_selection_keeps_one_file(qt_app, tmp_path):
    widget = FileInputWidget(multiple=False)
    try:
        widget.set_files([tmp_path / "one.csv", tmp_path / "two.csv"])
        assert widget.files() == [tmp_path / "one.csv"]
    finally:
        widget.close()


def test_panel_state_preserves_hidden_values_annotations_and_file_identity(qt_app, tmp_path):
    specs = [
        ParameterSpec("mode", "Mode", kind="choice", default="A", choices=["A", "B"]),
        ParameterSpec("hidden", "Hidden", default=3, show_if=("mode", "B")),
        ParameterSpec("source", "Source", kind="input_file"),
        ParameterSpec("reset", "", kind="reset", reset_keys=["hidden"]),
    ]
    first = tmp_path / "first" / "same.csv"
    second = tmp_path / "second" / "same.csv"
    panel = ParameterPanel(specs)
    restored = ParameterPanel(specs)
    try:
        panel.set_input_files([first, second])
        panel.set_values({"hidden": 1.234567891234, "source": str(second)})
        panel.set_annotations({"hidden": "± 0.002"})
        state = panel.save_state()
        assert state["values"]["source"] == second
        assert isinstance(state["values"]["source"], Path)
        assert "reset" not in state["values"]
        restored.set_input_files([first, second])
        restored.restore_state(state)
        assert restored.values() == panel.values()
        assert restored.save_state() == state
        assert restored._rows["hidden"].isHidden()
    finally:
        panel.close()
        restored.close()


def test_panel_changed_emits_for_edits_annotations_and_actual_updates(qt_app):
    callback = []
    panel = ParameterPanel([ParameterSpec("value", "Value", default=1)],
                           on_change=lambda: callback.append(True))
    changes = []
    panel.changed.connect(lambda: changes.append(True))
    try:
        panel.set_values({"value": 1})
        assert not changes
        panel.set_values({"value": 2})
        panel.set_annotations({"value": "± 1"})
        panel.set_annotations({"value": "± 1"})
        assert len(changes) == 2
        panel._widgets["value"].setValue(3)
        assert len(changes) == 3
        assert len(callback) == 2
        assert panel.save_state()["annotations"] == {}
    finally:
        panel.close()


def test_reset_annotation_change_is_saved_when_numeric_value_is_unchanged(qt_app):
    panel = ParameterPanel([
        ParameterSpec("value", "Value", default=1),
        ParameterSpec("reset", "Reset", kind="reset", reset_keys=["value"]),
    ])
    try:
        panel.set_annotations({"value": "Saved fit note"})
        changes = []
        panel.changed.connect(lambda: changes.append(True))
        panel._widgets["reset"].click()
        assert len(changes) == 1
        assert panel.save_state()["annotations"] == {}
    finally:
        panel.close()


def test_integrated_state_includes_hidden_calibration_and_each_eis_role(qt_app, tmp_path):
    notifications = []
    inputs = IntegratedReInputs(on_change=lambda: notifications.append(True))
    restored = IntegratedReInputs()
    try:
        for role, picker in inputs.inputs.items():
            picker.set_files([tmp_path / f"{role}.txt"])
        for role, picker in inputs.eis_inputs.items():
            picker.set_files([tmp_path / role / "z1.txt", tmp_path / role / "z2.txt"])
        inputs.panels["calibration"].set_values({"calibration_area": 2.35})
        inputs.panels["calibration"].set_annotations({"calibration_area": "Saved note"})
        inputs.set_result_parameters({"cathode_hfr": "0.031234"}, {"cathode_hfr": "2 spectra"})
        inputs.tabs.setCurrentIndex(1)
        inputs.eis_tabs.setCurrentIndex(2)
        assert tmp_path / "calibration.txt" not in inputs.files()
        assert tmp_path / "calibration.txt" in inputs.project_files()
        state = inputs.save_state()
        assert isinstance(state["inputs"]["calibration"][0], Path)
        before = len(notifications)
        inputs.tabs.setCurrentIndex(0)
        inputs.eis_tabs.setCurrentIndex(0)
        assert len(notifications) == before
        restored.restore_state(state)
        assert restored.save_state() == state
        assert restored.project_files() == inputs.project_files()
        assert restored.eis_panels["cathode"].values()["cathode_hfr"] == "0.031234"
    finally:
        inputs.close()
        restored.close()


def test_ink_state_keeps_blank_invalid_and_unfinished_cells(qt_app):
    changes = []
    widget = InkCalcTool().extra_widget(None, lambda: changes.append(True))
    restored = IonomerExtrasWidget()
    try:
        widget.ionomers.set_rows([["", "1e-", "NaN"], [" ", "oops", ""]])
        widget.extras.set_rows([["", "unfinished"], ["carbon", ""]])
        state = widget.save_state()
        restored.restore_state(state)
        assert restored.save_state() == state
        assert restored.values()["extras"][0]["name"] == "Material"
        assert state["extras"][0][0] == ""
        count = len(changes)
        widget.extras.table.setItem(0, 0, QTableWidgetItem("Pt"))
        assert len(changes) == count + 1
        add = next(button for button in widget.extras.findChildren(QPushButton)
                   if button.text() == "+ Add Material")
        add.click()
        assert len(changes) == count + 2
        widget.extras.table.selectRow(widget.extras.table.rowCount() - 1)
        remove = next(button for button in widget.extras.findChildren(QPushButton)
                      if button.text() == "Remove")
        remove.click()
        assert len(changes) == count + 3
    finally:
        widget.close()
        restored.close()
