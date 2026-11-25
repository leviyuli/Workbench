"""The everyday CCD form stays short while saved processing choices survive."""
import pytest
from PySide6.QtWidgets import QApplication

from workbench.tools.ccd.legacy_parameters import ccd_parameters
from workbench.widgets.parameter_panel import ParameterPanel


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance()
    if app is None:
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("QT_QPA_PLATFORM", "offscreen")
            app = QApplication([])
    yield app


def test_everyday_form_hides_processing_controls_and_their_section(qt_app):
    panel = ParameterPanel(ccd_parameters())
    try:
        visible = {key for key, row in panel._rows.items() if not row.isHidden()}
        assert visible == {
            "electrode_area", "baseline_mode", "baseline_lower", "baseline_upper",
            "show_raw", "show_advanced",
        }
        assert panel._groups["Advanced settings"].isHidden()
        panel.set_values({"show_advanced": True})
        assert not panel._groups["Advanced settings"].isHidden()
        assert all(not row.isHidden() for row in panel._rows.values())
        assert not panel._widgets["ccd_threshold_mv"].isReadOnly()
    finally:
        panel.deleteLater()
        qt_app.processEvents()


def test_auto_range_hides_only_manual_limits_and_keeps_them_for_manual_mode(qt_app):
    panel = ParameterPanel(ccd_parameters())
    try:
        panel.set_values({"baseline_lower": 1.85, "baseline_upper": 2.35,
                          "baseline_mode": "Auto"})
        assert panel._rows["baseline_lower"].isHidden()
        assert panel._rows["baseline_upper"].isHidden()
        assert not panel._groups["Analysis"].isHidden()
        panel.set_values({"baseline_mode": "Manual"})
        assert not panel._rows["baseline_lower"].isHidden()
        assert panel.values()["baseline_lower"] == 1.85
        assert panel.values()["baseline_upper"] == 2.35
    finally:
        panel.deleteLater()
        qt_app.processEvents()


def test_hidden_advanced_choices_restore_without_changing_scientific_defaults(qt_app):
    panel = ParameterPanel(ccd_parameters())
    restored = ParameterPanel(ccd_parameters())
    try:
        defaults = panel.values()
        assert defaults["baseline_mode"] == "Manual"
        assert (defaults["baseline_lower"], defaults["baseline_upper"]) == (1.9, 2.45)
        assert defaults["reference_hfr"] == .12
        assert defaults["ccd_threshold_mv"] == 20
        panel.set_values({"show_advanced": True, "reference_hfr": .18,
                          "ccd_threshold_mv": 100,
                          "smoothing_window": 21, "plateau_window_mv": 350})
        panel.set_values({"show_advanced": False})
        restored.restore_state(panel.save_state())
        assert restored.values() == panel.values()
        assert restored._groups["Advanced settings"].isHidden()
        restored.set_values({"show_advanced": True})
        assert restored.values()["reference_hfr"] == .18
        assert restored.values()["ccd_threshold_mv"] == 100
        assert restored.values()["smoothing_window"] == 21
        assert restored.values()["plateau_window_mv"] == 350
    finally:
        panel.deleteLater()
        restored.deleteLater()
        qt_app.processEvents()
