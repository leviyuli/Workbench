"""Batch inputs preserve source identity, selected branches and acquisition gaps."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication

from workbench.app import ToolPage
from workbench.core.tool_base import Tool, ToolResult
from workbench.tools.ccd.batch_inputs import migrate_batch_page_state, read_batch_scans
from workbench.tools.ccd.parameters import ccd_parameters


def _source(path, cycles=3, *, absolute=False, missing_last_back=False, labelled=False):
    blocks = []
    for cycle in range(1, cycles + 1):
        voltage = np.linspace(1, 2, 101)
        back = voltage[-2::-1]
        if cycle == cycles and missing_last_back:
            back = np.array([])
        block = pd.DataFrame({
            "E(mV)": 1000 * np.r_[voltage, back],
            "I(mA)" if absolute else "j(mA/cm2)":
                1000 * np.r_[cycle * voltage, cycle * (back + .25)],
        })
        if labelled:
            block.insert(0, "Cycle", cycle)
        blocks.append(block)
    pd.concat(blocks, ignore_index=True).to_csv(path, index=False)
    return path


def test_cv_defaults_to_all_forward_cycles_with_original_source_rows(tmp_path):
    path = _source(tmp_path / "cycles.csv")
    scans = read_batch_scans([path], {"input_type": "CV"})
    assert [scan.code for scan in scans] == ["S01-C01", "S01-C02", "S01-C03"]
    assert [scan.cycle for scan in scans] == [1, 2, 3]
    assert all(scan.direction == "Increasing" and not scan.error for scan in scans)
    assert [int(scan.source_indices[0]) for scan in scans] == [0, 201, 402]
    np.testing.assert_allclose(scans[-1].potential[[0, -1]], [1, 2])
    assert scans[-1].is_density


@pytest.mark.parametrize("mode,number,expected", [("Last", 1, 3), ("Specific", 2, 2)])
def test_selected_back_scan_remains_in_native_acquisition_order(tmp_path, mode, number, expected):
    path = _source(tmp_path / "cycles.csv", absolute=True, labelled=True)
    scans = read_batch_scans([path], {
        "input_type": "CV", "cv_scan": "Back", "cv_cycle_mode": mode, "cv_cycle": number,
    })
    assert len(scans) == 1
    scan = scans[0]
    assert scan.cycle == expected and scan.branch == "Back"
    assert scan.direction == "Decreasing" and not scan.is_density
    assert np.all(np.diff(scan.potential) < 0)
    assert scan.source_indices[0] > 100


def test_unavailable_cycle_and_missing_last_back_do_not_fall_back(tmp_path):
    path = _source(tmp_path / "unfinished.csv", missing_last_back=True)
    absent = read_batch_scans([path], {
        "input_type": "CV", "cv_cycle_mode": "Specific", "cv_cycle": 4,
    })
    assert len(absent) == 1 and absent[0].cycle == 4
    assert "unavailable" in absent[0].error
    last = read_batch_scans([path], {
        "input_type": "CV", "cv_scan": "Back", "cv_cycle_mode": "Last",
    })
    assert len(last) == 1 and last[0].cycle == 3
    assert "back scan" in last[0].error


def test_gap_rows_and_their_original_indices_survive_selection(tmp_path):
    path = _source(tmp_path / "gap.csv", cycles=1)
    frame = pd.read_csv(path)
    frame.loc[40, :] = np.nan
    frame.to_csv(path, index=False)
    scan = read_batch_scans([path], {"input_type": "CV"})[0]
    assert np.isnan(scan.potential[40]) and np.isnan(scan.current[40])
    assert scan.source_indices[40] == 40
    assert len(scan.potential) == 101


def test_lsv_mode_uses_one_initial_branch_and_keeps_other_files_independent(tmp_path):
    first = _source(tmp_path / "first.csv")
    second = tmp_path / "second.csv"
    second.write_text("E(V),I(A)\n1,0.1\n1.1,0.2\n1.2,0.3\n", encoding="utf-8")
    scans = read_batch_scans([first, second], {})
    assert [scan.code for scan in scans] == ["S01", "S02"]
    assert [scan.cycle for scan in scans] == [None, None]
    assert scans[0].is_density and not scans[1].is_density


def test_duplicate_names_and_unreadable_files_keep_distinct_sources(tmp_path):
    first_dir, second_dir = tmp_path / "a", tmp_path / "b"
    first_dir.mkdir()
    second_dir.mkdir()
    first = _source(first_dir / "same.csv", cycles=1)
    second = _source(second_dir / "same.csv", cycles=2)
    bad = tmp_path / "bad.csv"
    bad.write_text("invalid\n", encoding="utf-8")
    scans = read_batch_scans([first, second, bad], {"input_type": "CV", "cv_cycle_mode": "Last"})
    assert [scan.file for scan in scans] == ["same.csv", "same.csv (2)", "bad.csv"]
    assert [scan.source for scan in scans] == ["S01", "S02", "S03"]
    assert [scan.cycle for scan in scans[:2]] == [1, 2]
    assert scans[-1].error


def test_ten_cv_cycles_map_source_indices_to_native_table(tmp_path):
    path = _source(tmp_path / "cycles.csv", cycles=10)
    scans = read_batch_scans([path], {"input_type": "CV"})
    assert len(scans) == 10 and [scan.cycle for scan in scans] == list(range(1, 11))
    final = scans[-1]
    assert final.source_indices[0] > len(final.potential)
    table = pd.read_csv(path).iloc[final.source_indices]
    np.testing.assert_allclose(table["E(mV)"] / 1000, final.potential)
    np.testing.assert_allclose(table["j(mA/cm2)"] / 1000, final.current)


@pytest.mark.parametrize("number", [0, -1, 1.5, "invalid", True])
def test_specific_cycle_rejects_invalid_numbers(tmp_path, number):
    with pytest.raises(ValueError, match="positive integer"):
        read_batch_scans([tmp_path / "unused.csv"], {
            "input_type": "CV", "cv_cycle_mode": "Specific", "cv_cycle": number,
        })


def test_legacy_migration_maps_inputs_without_touching_frozen_artifacts():
    state = {
        "tool_id": "lsv_ccd", "files": [],
        "panel": {"values": {"reference_hfr": .18, "electrode_area": 2.5,
                              "current_branch": "Negative", "baseline_mode": "Manual",
                              "baseline_lower": 1.9, "baseline_upper": 2.45,
                              "ccd_threshold_mv": 75}},
        "extra": None, "last_run": {"params": {"reference_hfr": .18}}, "result_tab": 8,
        "result": {"figures": {"Saved": b"frozen plot"}, "tables": {}, "text_blocks": {},
                   "artifacts": [{"kind": "xlsx", "bytes": b"frozen workbook"}],
                   "display_tables": True},
    }
    original = deepcopy(state)
    migrated = migrate_batch_page_state(state)
    assert state == original
    values = migrated["panel"]["values"]
    assert values["hfr"] == .18 and values["electrode_area"] == 2.5
    assert values["current_branch"] == "Negative"
    assert values["ccd_threshold_mv"] == 75
    assert values["ccd_fit_mode"] == "Auto" and values["cv_cycle_mode"] == "All"
    assert "ccd_j_lower" not in values and "baseline_lower" not in values
    assert migrated["result"]["figures"] == original["result"]["figures"]
    assert migrated["result"]["artifacts"] == original["result"]["artifacts"]
    assert migrated["last_run"] == original["last_run"]
    assert migrated["extra"] == {"hfr_points": []}
    assert migrated["result_tab"] == 0 and not migrated["result"]["display_tables"]
    assert "Historical" in migrated["result"]["text_blocks"]["Analysis method"]


def test_new_batch_states_pass_through_without_reinterpreting_ranges():
    state = {"tool_id": "lsv_ccd", "panel": {"values": {
        "hfr": .12, "ccd_fit_mode": "Manual", "ccd_j_lower": 1., "ccd_j_upper": 3.,
    }}}
    assert migrate_batch_page_state(state) is state


def test_historical_residual_batch_annotation_preserves_frozen_results_and_exports():
    from workbench.core.project import restore_result, snapshot_result
    from workbench.core.tool_base import Artifact

    table = pd.DataFrame({"Curve": ["S01-C01"], "CCD (A/cm²)": [1.23456789012345]})
    old_result = ToolResult(tables={"Current results": table},
        scalars={"Method": "residual-baseline-v1", "Scans decoupled": 1},
        text_blocks={"Statistics": "Original saved interpretation"}, display_tables=False)
    artifact = Artifact("Batch workbook", "xlsx", "batch.xlsx", {"Current results": table})
    snapshot = snapshot_result(old_result, [artifact])
    state = {"tool_id": "lsv_ccd", "panel": {"values": {"hfr": .12,
        "i_lower": .02, "i_upper": .8, "ccd_fit_mode": "Manual",
        "ccd_j_lower": 1., "ccd_j_upper": 3.}}, "result": snapshot,
        "last_run": {"params": {"hfr": .12}}, "result_tab": 2}
    migrated = migrate_batch_page_state(state)

    assert migrated is not state
    assert "Saved Tafel method" not in state["result"]["text_blocks"]
    assert "nested-window" in migrated["result"]["text_blocks"]["Saved Tafel method"]
    assert migrated["panel"] == state["panel"]
    assert migrated["last_run"] == state["last_run"]
    assert migrated["result_tab"] == state["result_tab"]
    assert migrated["result"]["scalars"] == snapshot["scalars"]
    assert migrated["result"]["artifacts"] == snapshot["artifacts"]
    restored, frozen = restore_result(migrated["result"])
    pd.testing.assert_frame_equal(restored.tables["Current results"], table)
    assert frozen[0].serialized_payload == snapshot["artifacts"][0]["bytes"]
    assert migrate_batch_page_state(migrated) is migrated


def test_current_tafel_method_snapshot_does_not_receive_a_historical_annotation():
    state = {"tool_id": "lsv_ccd", "panel": {"values": {"hfr": .12}},
             "result": {"scalars": {"Method": "residual-baseline-v1",
                         "Tafel method": "contiguous-stable-tafel-v1"}, "text_blocks": {}}}
    assert migrate_batch_page_state(state) is state


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance()
    if app is None:
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("QT_QPA_PLATFORM", "offscreen")
            app = QApplication([])
    yield app


def test_compact_schema_keeps_threshold_editable_and_restores_its_value(qt_app):
    from workbench.widgets.parameter_panel import ParameterPanel

    panel = ParameterPanel(ccd_parameters())
    restored = ParameterPanel(ccd_parameters())
    try:
        values = panel.values()
        assert values["input_type"] == "LSV" and values["cv_cycle_mode"] == "All"
        assert values["ccd_fit_mode"] == "Auto" and not values["include_rcl"]
        assert (values["i_lower"], values["i_upper"]) == (.02, .8)
        assert panel._rows["cv_cycle"].isHidden()
        assert panel._rows["ccd_j_lower"].isHidden() and panel._rows["rcl"].isHidden()
        assert values["ccd_threshold_mv"] == 50
        assert not panel._rows["ccd_threshold_mv"].isHidden()
        assert not panel._widgets["ccd_threshold_mv"].isReadOnly()
        assert not {"show_advanced", "smoothing_window", "x_log", "swap_axes"} & values.keys()
        panel.set_values({"input_type": "CV", "cv_cycle_mode": "Specific",
                          "ccd_fit_mode": "Manual", "include_rcl": True,
                          "ccd_threshold_mv": 25})
        assert not panel._rows["cv_cycle"].isHidden()
        assert not panel._rows["ccd_j_lower"].isHidden() and not panel._rows["rcl"].isHidden()
        panel.set_values({"cv_cycle_mode": "All"})
        assert panel._rows["cv_cycle"].isHidden()
        restored.restore_state(panel.save_state())
        assert restored.values()["ccd_threshold_mv"] == 25
    finally:
        panel.deleteLater()
        restored.deleteLater()
        qt_app.processEvents()


def test_tool_page_calls_migration_before_restoring_parameters(qt_app, tmp_path):
    class MigratingTool(Tool):
        id = "migration_test"
        needs_files = False

        def parameter_schema(self):
            return []

        def run(self, files, params, extra=None):
            return ToolResult()

        def migrate_page_state(self, state):
            migrated = deepcopy(state)
            migrated["tool_id"] = self.id
            return migrated

    page = ToolPage(MigratingTool(), lambda _: None)
    try:
        saved = page.save_state()
        saved["tool_id"] = "old_identity"
        page.restore_state(saved, tmp_path)
    finally:
        page.deleteLater()
        qt_app.processEvents()
