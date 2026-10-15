"""The default PtRuC calibration is ready to use and remains user editable."""
import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

from workbench.app import ToolPage
from workbench.core.data_io import read_lsv_curve
from workbench.core.project import load_project, save_project
from workbench.tools.integrated_re import inputs as inputs_module
from workbench.tools.integrated_re.inputs import DEFAULT_CALIBRATION_FILE
from workbench.tools.integrated_re.tool import IntegratedReTool


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance()
    if app is None:
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
            app = QApplication([])
    yield app


@pytest.fixture
def page(qt_app):
    widget = ToolPage(IntegratedReTool(), lambda _: None)
    yield widget
    widget.deleteLater()
    qt_app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def select_correction(page, mode):
    page.panel._widgets["correction_mode"].setCurrentText(mode)


def test_choosing_onset_loads_ptruc_preset_from_any_working_directory(page, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert page.panel.values()["correction_mode"] == "None"
    assert page.extra.inputs["calibration"].files() == []
    select_correction(page, "Manual")
    assert page.extra.inputs["calibration"].files() == []

    select_correction(page, "3E onset")

    values = page.panel.values()
    assert values["calibration_electrode"] == "Cathode"
    assert values["calibration_reference"] == "Hg/HgO (1 M KOH, 25 °C)"
    assert values["calibration_ph"] == 14.0
    assert values["calibration_temperature"] == 298.0
    assert values["onset_current"] == 0.05
    assert values["reference"] == "RHE"
    assert values["ph"] == 0.0
    assert page.extra.tabs.isTabVisible(3)
    assert DEFAULT_CALIBRATION_FILE.is_absolute()
    assert DEFAULT_CALIBRATION_FILE.is_file()
    assert page.extra.inputs["calibration"].files() == [DEFAULT_CALIBRATION_FILE]
    assert page._input_files() == [DEFAULT_CALIBRATION_FILE]
    assert page.extra.values()["parameters"]["calibration_area"] == 0.0


def test_calibration_edits_replacement_and_clear_survive_parameter_changes(page, tmp_path):
    select_correction(page, "3E onset")
    custom = tmp_path / "custom_calibration.csv"
    custom.write_text("E(V),j(A/cm2)\n0,0\n-0.3,-0.1\n", encoding="utf-8")
    page.extra.inputs["calibration"].set_files([custom])
    updates = {"calibration_reference": "SHE", "calibration_temperature": 310.0,
               "calibration_ph": 13.0, "calibration_electrode": "Anode",
               "onset_current": 0.02}
    page.panel.set_values(updates)
    select_correction(page, "None")
    assert custom not in page._input_files()
    select_correction(page, "3E peak")
    select_correction(page, "3E onset")
    assert page.extra.inputs["calibration"].files() == [custom]
    assert {key: page.panel.values()[key] for key in updates} == updates

    page.extra.inputs["calibration"].clear()
    page.panel.set_values({"electrode_axis": "Overpotential"})
    select_correction(page, "None")
    select_correction(page, "3E onset")
    assert page.extra.inputs["calibration"].files() == []
    assert {key: page.panel.values()[key] for key in updates} == updates


def test_custom_calibration_selected_before_first_activation_is_retained(page, tmp_path):
    custom = tmp_path / "existing.csv"
    page.extra.inputs["calibration"].set_files([custom])
    page.panel.set_values({"calibration_reference": "SHE", "onset_current": 0.025})
    select_correction(page, "3E onset")
    assert page.extra.inputs["calibration"].files() == [custom]
    assert page.panel.values()["calibration_reference"] == "SHE"
    assert page.panel.values()["onset_current"] == 0.025


@pytest.mark.parametrize("mode", ["None", "3E onset"])
@pytest.mark.parametrize("clear", [False, True])
def test_saved_custom_or_cleared_calibration_survives_restore(page, qt_app, tmp_path, mode, clear):
    select_correction(page, "3E onset")
    custom = tmp_path / "saved.csv"
    page.extra.inputs["calibration"].set_files([] if clear else [custom])
    page.panel.set_values({"calibration_reference": "SHE", "calibration_ph": 12.0,
                           "calibration_temperature": 305.0, "onset_current": 0.02})
    select_correction(page, mode)
    state = page.save_state()
    restored = ToolPage(IntegratedReTool(), lambda _: None)
    try:
        restored.restore_state(state, tmp_path)
        assert restored.panel.values() == page.panel.values()
        assert restored.extra.save_state() == page.extra.save_state()
        select_correction(restored, "None")
        select_correction(restored, "3E onset")
        assert restored.extra.inputs["calibration"].files() == ([] if clear else [custom])
        assert restored.panel.values()["onset_current"] == 0.02
    finally:
        restored.deleteLater()
        qt_app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_legacy_saved_empty_calibration_is_not_replaced(page, tmp_path):
    select_correction(page, "3E onset")
    page.extra.inputs["calibration"].clear()
    state = page.save_state()
    state["extra"].pop("calibration_default_initialized")
    page.restore_state(state, tmp_path)
    select_correction(page, "None")
    select_correction(page, "3E onset")
    assert page.extra.inputs["calibration"].files() == []


def test_missing_default_file_keeps_manual_upload_available(page, tmp_path, monkeypatch):
    monkeypatch.setattr(inputs_module, "DEFAULT_CALIBRATION_FILE", tmp_path / "missing.txt")
    select_correction(page, "3E onset")
    assert page.extra.tabs.isTabVisible(3)
    assert page.extra.inputs["calibration"].files() == []
    custom = tmp_path / "replacement.csv"
    page.extra.inputs["calibration"].set_files([custom])
    assert page.extra.inputs["calibration"].files() == [custom]


def test_preset_is_embedded_in_project_and_restores_without_installed_source(
        page, tmp_path, monkeypatch):
    select_correction(page, "3E onset")
    expected = DEFAULT_CALIBRATION_FILE.read_bytes()
    archive = tmp_path / "ptruc.ecwb"
    save_project(archive, {"page": page.save_state()})
    monkeypatch.setattr(inputs_module, "DEFAULT_CALIBRATION_FILE", tmp_path / "not_installed.txt")
    with load_project(archive) as loaded:
        page.restore_state(loaded.state["page"], tmp_path)
        embedded = page.extra.inputs["calibration"].files()[0]
        assert embedded != DEFAULT_CALIBRATION_FILE
        assert embedded.name == DEFAULT_CALIBRATION_FILE.name
        assert embedded.read_bytes() == expected
        select_correction(page, "None")
        assert embedded not in page._input_files()
        assert embedded in page.extra.project_files()
        select_correction(page, "3E onset")
        assert page.extra.inputs["calibration"].files() == [embedded]


def test_actual_ptruc_calibration_uses_density_and_recovers_known_shared_offset(page, tmp_path):
    select_correction(page, "3E onset")
    data, error = read_lsv_curve(str(DEFAULT_CALIBRATION_FILE))
    assert error is None
    potential, current, info = data
    assert info["is_density"]
    target = 0.05
    reaction_current = -current
    crossings = np.flatnonzero((reaction_current[:-1] - target)
                               * (reaction_current[1:] - target) < 0)
    assert crossings.size
    onset = min(potential[index] + (target - reaction_current[index])
                * (potential[index + 1] - potential[index])
                / (reaction_current[index + 1] - reaction_current[index])
                for index in crossings)
    rhe_shift = 0.1053 + 2.303 * 8.31446261815324 / 96485.33212 * 298.0 * 14.0
    onset += rhe_shift
    j = np.array([0.0, 0.025, 0.075, 0.1])
    measured = {"anode": (1.4 + 0.8 * j + 0.2, j),
                "cathode": (onset - 0.4 * (j - target) + 0.2, -j)}
    for role, (source_e, source_j) in measured.items():
        path = tmp_path / f"{role}.csv"
        pd.DataFrame({"E(V)": source_e, "j(A/cm2)": source_j}).to_csv(path, index=False)
        page.extra.inputs[role].set_files([path])
    page.panel.set_values({"manual_shift": 0.03})
    result = page.tool.run(page._input_files(), page.panel.values(), page.extra.values())

    assert result.scalars["Shared correction (V)"] == pytest.approx(-0.17)
    assert result.param_updates["calculated_shift"] == "-0.200000"
    assert result.param_updates["applied_shift"] == "-0.170000"
    calibration = result.tables["Calibration LSV"]
    np.testing.assert_allclose(calibration["E (V vs RHE)"], potential + rhe_shift)
    np.testing.assert_allclose(calibration["Current density (A/cm²)"], current)
    features = result.tables["Calibration features"]
    np.testing.assert_allclose(features["Current density (A/cm²)"], -target)
    settings = result.tables["Potential settings"].set_index("Parameter")["Value"]
    assert settings["Three-electrode calibration LSV electrode area (cm²)"] == 0.0
    assert settings["Three-electrode calibration LSV: Temperature (K)"] == 298.0
    assert settings["Three-electrode calibration LSV: pH"] == 14.0
