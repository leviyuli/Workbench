"""HFR input accepts one uploaded/pasted HFR curve and keeps portable edits."""
import json

import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication, QComboBox

from workbench.tools.lsv.hfr_data import parse_hfr_pairs
from workbench.tools.lsv.inputs import HfrInputs


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance()
    if app is None:
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("QT_QPA_PLATFORM", "offscreen")
            app = QApplication([])
    yield app


def test_paste_mode_switch_and_spline_controls_preserve_input(qt_app):
    changes = []
    widget = HfrInputs(on_change=lambda: changes.append(True))
    try:
        assert widget.isHidden()
        assert widget.values() == {"hfr_text": "", "hfr_smoothing": .0001, "hfr_source": ""}
        assert not widget.findChildren(QComboBox)
        widget.set_parameters({"hfr_mode": "HFR vs |j|"})
        assert not widget.isHidden()
        text = "0.1\t0.06077\n0.5\t0.06191\n1\t0.05974\n1.5\t0.05871\n2\t0.0577"
        widget.editor.setPlainText(text)
        assert len(changes) == 1
        assert "5 points" in widget.status_label.text()
        assert widget.method_label.text() == "Cubic smoothing spline"
        widget.smoothing_input.setValue(.001)
        widget.set_parameters({"hfr_mode": "Constant HFR"})
        assert widget.isHidden()
        widget.set_parameters({"hfr_mode": "HFR vs |j|"})
        assert widget.values()["hfr_text"] == text
        assert widget.values()["hfr_smoothing"] == .001
    finally:
        widget.deleteLater()
        qt_app.processEvents()


def test_upload_extensionless_two_columns(qt_app, tmp_path):
    changes = []
    widget = HfrInputs(on_change=lambda: changes.append(True))
    path = tmp_path / "HFR"
    text = ".1\t.06\n.5\t.061\n1\t.059\n2\t.058\n3\t.057"
    path.write_text(text, encoding="utf-8")
    try:
        widget.load_file(path)
        assert len(changes) == 1
        assert widget.values()["hfr_source"] == "HFR"
        assert widget.values()["hfr_text"] == text
        assert "5 points" in widget.status_label.text()
        assert "3 A/cm²" in widget.status_label.text()
    finally:
        widget.deleteLater()
        qt_app.processEvents()


def test_upload_xlsx_copies_data_for_portable_state(qt_app, tmp_path):
    path = tmp_path / "hfr.xlsx"
    pd.DataFrame({"j": [.1, .5, 1, 2, 3], "HFR": [.06, .061, .059, .058, .057]}).to_excel(
        path, index=False)
    widget = HfrInputs()
    restored = HfrInputs()
    try:
        widget.load_file(path)
        state = json.loads(json.dumps(widget.save_state()))
        path.unlink()
        restored.restore_state(state)
        assert restored.values() == widget.values()
        assert len(parse_hfr_pairs(restored.values()["hfr_text"]).points()) == 5
        assert "5 points" in restored.status_label.text()
    finally:
        widget.deleteLater()
        restored.deleteLater()
        qt_app.processEvents()


def test_partial_paste_survives_state_restore(qt_app):
    widget = HfrInputs()
    restored = HfrInputs()
    try:
        widget.editor.setPlainText("j\tHFR\n.1\t.06\n.5\t1e-")
        state = json.loads(json.dumps(widget.save_state()))
        restored.restore_state(state)
        assert restored.values() == widget.values()
        state["hfr_text"] = "changed by caller"
        assert restored.values()["hfr_text"].endswith("1e-")
    finally:
        widget.deleteLater()
        restored.deleteLater()
        qt_app.processEvents()


def test_multicolumn_upload_rejected_and_paste_reports_two_column_requirement(qt_app, tmp_path):
    path = tmp_path / "HFR"
    text = "j\tFirst\tSecond\n.1\t.06\t.07\n.5\t.061\t.071\n1\t.059\t.069"
    path.write_text(text, encoding="utf-8")
    widget = HfrInputs()
    try:
        widget.set_rows([[".1", ".06"]])
        before = widget.save_state()
        with pytest.raises(ValueError, match="exactly two columns"):
            widget.load_file(path)
        assert widget.save_state() == before
        widget.editor.setPlainText(text)
        assert widget.values()["hfr_text"] == text
        assert "exactly two columns" in widget.status_label.text()
    finally:
        widget.deleteLater()
        qt_app.processEvents()


def test_old_four_field_state_normalizes_selected_test_to_two_columns(qt_app):
    widget = HfrInputs()
    try:
        text = "j\tFirst\tSecond\n.1\t.06\t.07\n.5\t.061\t.071\n1\t.059\t.069\n2\t.058\t\n3\t.057\t.067"
        widget.restore_state({"hfr_text": text, "hfr_column": 1,
                              "hfr_smoothing": .001, "hfr_source": "old.csv"})
        state = widget.save_state()
        assert set(state) == {"hfr_text", "hfr_smoothing", "hfr_source"}
        data = parse_hfr_pairs(state["hfr_text"])
        assert data.column_names == ("Second",)
        assert data.points() == [(.1, .07), (.5, .071), (1., .069), (3., .067)]
        assert state["hfr_source"] == "old.csv"
        assert state["hfr_smoothing"] == .001
    finally:
        widget.deleteLater()
        qt_app.processEvents()


def test_old_four_field_state_preserves_unfinished_raw_text(qt_app):
    widget = HfrInputs()
    try:
        text = "j\tFirst\tSecond\n.1\t.06\t1e-"
        widget.restore_state({"hfr_text": text, "hfr_column": 1,
                              "hfr_smoothing": .0001, "hfr_source": ""})
        assert widget.save_state()["hfr_text"] == text
    finally:
        widget.deleteLater()
        qt_app.processEvents()


def test_legacy_point_state_and_none_preserve_compatibility(qt_app):
    widget = HfrInputs()
    rows = [["0.50", "1e-"], [" ", "NaN"], ["0.1", ""]]
    try:
        widget.restore_state({"hfr_points": rows})
        assert widget.rows() == rows
        rows[0][0] = "changed by caller"
        assert widget.rows()[0][0] == "0.50"
        widget.restore_state(None)
        assert widget.values() == {"hfr_text": "", "hfr_smoothing": .0001, "hfr_source": ""}
    finally:
        widget.deleteLater()
        qt_app.processEvents()


@pytest.mark.parametrize("state", [
    [], {}, {"other": []}, {"hfr_points": "invalid"},
    {"hfr_points": [["0.1"]]}, {"hfr_points": [["0.1", "0.1", "extra"]]},
    {"hfr_points": [[0.1, "0.1"]]}, {"hfr_points": [None]},
    {"hfr_points": [["0.1", "0.1"], ["0.5", .12]]},
    {"hfr_text": "", "hfr_column": True, "hfr_smoothing": .0001, "hfr_source": ""},
    {"hfr_text": "", "hfr_smoothing": float("nan"), "hfr_source": ""},
])
def test_invalid_saved_structure_does_not_replace_existing_input(qt_app, state):
    widget = HfrInputs()
    try:
        widget.set_rows([["0.1", "0.1"]])
        before = widget.save_state()
        with pytest.raises(ValueError, match="HFR"):
            widget.restore_state(state)
        assert widget.save_state() == before
    finally:
        widget.deleteLater()
        qt_app.processEvents()
