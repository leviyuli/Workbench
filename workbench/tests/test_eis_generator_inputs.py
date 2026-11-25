"""CL EIS saved overlays keep independent inputs and portable, atomic state."""
from copy import deepcopy
import json

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from workbench.tools.eis_generator.inputs import ClEisCasesWidget


DEFAULTS = {
    "electronic_conductivity": 0.010,
    "ionic_conductivity": 0.002,
    "thickness_um": 5.0,
    "capacitance_mf_cm2": 20.0,
    "external_resistance": 0.050,
    "f_min": 0.1,
    "f_max": 1e6,
    "n_points": 400,
}


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance()
    if app is None:
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("QT_QPA_PLATFORM", "offscreen")
            app = QApplication([])
    yield app


@pytest.fixture
def cases_widget(qt_app):
    widget = ClEisCasesWidget()
    yield widget
    widget.close()


def test_saved_case_and_worker_values_are_independent_snapshots(cases_widget):
    params = dict(DEFAULTS, plot_view="Transport detail")
    cases_widget.set_parameters(params)
    params["thickness_um"] = 99
    assert cases_widget.add_current_case() == "case_1"
    cases_widget.set_parameters(dict(DEFAULTS, thickness_um=10))
    assert cases_widget.add_current_case() == "case_2"
    state = cases_widget.values()
    assert state["cases"][0]["params"] == DEFAULTS
    assert "plot_view" not in state["cases"][0]["params"]
    assert state["cases"][1]["params"]["thickness_um"] == 10
    state["cases"][0]["params"]["thickness_um"] = 500
    state["cases"].clear()
    fresh = cases_widget.values()
    assert len(fresh["cases"]) == 2
    assert fresh["cases"][0]["params"]["thickness_um"] == 5
    assert fresh["annotated_case"] == "current"
    assert fresh["next_case_number"] == 3


def test_edits_visibility_and_annotation_use_stable_ids(cases_widget):
    changes = []
    cases_widget._on_change = lambda: changes.append(True)
    first = cases_widget.add_current_case("same name")
    second = cases_widget.add_current_case("same name")
    combo = cases_widget.annotation_combo
    assert combo.itemText(1) == combo.itemText(2) == "same name"
    combo.setCurrentIndex(combo.findData(second))
    assert cases_widget.values()["annotated_case"] == second
    cases_widget.table.item(1, 1).setText("  renamed  ")
    assert combo.currentData() == second
    assert combo.currentText() == "renamed"
    assert cases_widget.values()["cases"][1]["name"] == "renamed"
    cases_widget.table.item(1, 0).setCheckState(Qt.CheckState.Unchecked)
    assert combo.currentData() == "current"
    assert combo.findData(second) == -1
    assert not cases_widget.values()["cases"][1]["visible"]
    cases_widget.table.item(1, 0).setCheckState(Qt.CheckState.Checked)
    assert combo.findData(second) >= 0
    assert combo.currentData() == "current"
    assert combo.findData(first) >= 0
    assert len(changes) == 6  # two saves, selection, rename and two visibility edits
    before = len(changes)
    cases_widget.table.item(1, 1).setText("   ")
    assert cases_widget.table.item(1, 1).text() == "renamed"
    assert cases_widget.values()["cases"][1]["name"] == "renamed"
    assert len(changes) == before
    assert "name" in cases_widget.error_label.text().lower()


def test_remove_and_clear_reset_annotation_without_reusing_ids(cases_widget):
    cases_widget.add_current_case()
    selected = cases_widget.add_current_case()
    cases_widget.add_current_case()
    cases_widget.annotation_combo.setCurrentIndex(cases_widget.annotation_combo.findData(selected))
    cases_widget.table.selectRow(1)
    assert cases_widget.btn_remove.isEnabled()
    cases_widget.btn_remove.click()
    assert [case["id"] for case in cases_widget.values()["cases"]] == ["case_1", "case_3"]
    assert cases_widget.values()["annotated_case"] == "current"
    assert cases_widget.add_current_case() == "case_4"
    cases_widget.clear_cases()
    assert cases_widget.values() == {
        "cases": [], "annotated_case": "current", "next_case_number": 5,
    }
    assert cases_widget.annotation_combo.count() == 1
    assert not cases_widget.btn_clear.isEnabled()
    assert cases_widget.add_current_case() == "case_5"


def test_json_state_roundtrip_preserves_order_names_visibility_and_selection(cases_widget, qt_app):
    cases_widget.add_current_case("first")
    cases_widget.set_parameters(dict(DEFAULTS, electronic_conductivity=0.02))
    cases_widget.add_current_case("second")
    cases_widget.table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    cases_widget.annotation_combo.setCurrentIndex(1)
    state = json.loads(json.dumps(cases_widget.save_state(), allow_nan=False))
    restored = ClEisCasesWidget()
    changes = []
    restored._on_change = lambda: changes.append(True)
    try:
        restored.restore_state(state)
        assert restored.values() == cases_widget.values()
        assert restored.annotation_combo.currentData() == "case_2"
        assert restored.table.item(0, 0).checkState() == Qt.CheckState.Unchecked
        state["cases"][0]["params"]["electronic_conductivity"] = 999
        assert restored.values() == cases_widget.values()
        assert len(changes) == 1
        restored.restore_state(restored.save_state())
        assert len(changes) == 1
        assert restored.add_current_case() == "case_3"
    finally:
        restored.close()


@pytest.mark.parametrize("annotated", ["case_1", "case_999"])
def test_restore_hidden_or_missing_annotation_returns_to_current(cases_widget, annotated):
    cases_widget.add_current_case()
    state = cases_widget.save_state()
    state["cases"][0]["visible"] = False
    state["annotated_case"] = annotated
    cases_widget.restore_state(state)
    assert cases_widget.values()["annotated_case"] == "current"


@pytest.mark.parametrize("path, invalid", [
    (("cases",), {}),
    (("cases", 0, "id"), "current"),
    (("cases", 0, "name"), "  "),
    (("cases", 0, "visible"), "true"),
    (("cases", 0, "params"), {}),
    (("cases", 0, "params", "thickness_um"), 0),
    (("cases", 0, "params", "ionic_conductivity"), float("nan")),
    (("cases", 0, "params", "electronic_conductivity"), float("inf")),
    (("cases", 0, "params", "f_max"), 0.01),
    (("cases", 0, "params", "n_points"), 20001),
    (("next_case_number",), 1),
    (("next_case_number",), True),
    (("annotated_case",), []),
])
def test_invalid_restore_is_atomic(cases_widget, path, invalid):
    cases_widget.add_current_case()
    before = cases_widget.save_state()
    bad = deepcopy(before)
    target = bad
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = invalid
    changes = []
    cases_widget._on_change = lambda: changes.append(True)
    with pytest.raises(ValueError):
        cases_widget.restore_state(bad)
    assert cases_widget.save_state() == before
    assert not changes


def test_duplicate_ids_and_unknown_fields_are_rejected(cases_widget):
    cases_widget.add_current_case()
    before = cases_widget.save_state()
    bad = deepcopy(before)
    bad["cases"].append(deepcopy(bad["cases"][0]))
    with pytest.raises(ValueError, match="unique"):
        cases_widget.restore_state(bad)
    for bad in (None, dict(before, unknown=True)):
        with pytest.raises(ValueError):
            cases_widget.restore_state(bad)
    assert cases_widget.save_state() == before


def test_bad_current_inputs_do_not_save_and_button_explains_error(cases_widget):
    cases_widget.set_parameters(dict(DEFAULTS, thickness_um=0))
    with pytest.raises(ValueError):
        cases_widget.add_current_case()
    assert not cases_widget.values()["cases"]
    cases_widget.btn_save.click()
    assert "Cannot save case" in cases_widget.error_label.text()
    assert cases_widget.values()["next_case_number"] == 1
    cases_widget.set_parameters(DEFAULTS)
    cases_widget.btn_save.click()
    assert len(cases_widget.values()["cases"]) == 1
    assert not cases_widget.error_label.text()


def test_empty_restore_resets_state_and_starts_clean(cases_widget):
    cases_widget.add_current_case()
    cases_widget.restore_state({})
    assert cases_widget.values() == {
        "cases": [], "annotated_case": "current", "next_case_number": 1,
    }
