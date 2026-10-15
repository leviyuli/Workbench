"""LSV navigation, complete-curve import and custom-input integration."""
from pathlib import Path
from threading import Event
from time import monotonic, sleep

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from PySide6.QtCore import QCoreApplication, QEvent, QThreadPool
from PySide6.QtGui import QGuiApplication
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QLineEdit, QWidget

import workbench.app as app_module
from workbench.app import GroupedToolPage, MainWindow, ToolPage
from workbench.core.data_io import read_lsv_curve, read_lsv_data
from workbench.core.tool_base import ParameterSpec, Tool, ToolResult
from workbench.tools import build_tools
from workbench.tools.integrated_re.inputs import IntegratedReInputs, ROLES
from workbench.tools.lsv.tool import LsvTool
from workbench.tools.lsv_analysis.tool import LsvAnalysisTool


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance()
    if app is None:
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
            app = QApplication([])
    yield app


def _wait_until(predicate):
    deadline = monotonic() + 5
    while not predicate() and monotonic() < deadline:
        QTest.qWait(5)
        # Give Python-heavy figure construction a scheduling window as well
        # as pumping Qt's event queue.
        sleep(0.001)
    assert predicate(), "Worker did not finish before the timeout"


class _AsyncTool(Tool):
    id = "async_lsv_test"
    name = "First function"
    files_required = False

    def __init__(self, fail=False):
        self.fail = fail
        self.release = Event()
        self.calls = []

    def parameter_schema(self):
        return [ParameterSpec("label", "Label", kind="str", default="sample")]

    def run(self, files, params, extra=None):
        self.calls.append((files, params, extra))
        if not self.release.wait(5):
            raise TimeoutError("Worker not released")
        if self.fail:
            raise ValueError("Invalid spectrum")
        return ToolResult(text_blocks={"Details": params["label"]})


class _Group(_AsyncTool):
    name = "Group"

    def __init__(self, children):
        self.children = children

    def subtools(self):
        return self.children


def _dispose(widget, qt_app):
    widget.deleteLater()
    qt_app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    while QGuiApplication.overrideCursor() is not None:
        QGuiApplication.restoreOverrideCursor()


def test_registry_has_one_lsv_entry_and_preserves_concrete_tools(qt_app):
    tools = build_tools()
    assert [tool.name for tool in tools if tool.id.startswith("lsv")] == ["LSV analysis"]
    grouped = next(tool for tool in tools if isinstance(tool, LsvAnalysisTool))
    assert isinstance(grouped, LsvTool)
    children = grouped.subtools()
    assert [tool.name for tool in children] == [
        "Decoupling", "Generator", "Integrated RE for cell",
    ]
    assert children[0].parameter_schema() == LsvTool().parameter_schema()
    assert [action.method for action in children[0].actions()] == [
        action.method for action in LsvTool().actions()
    ]
    window = MainWindow([grouped])
    try:
        page = window.stack.widget(0)
        assert isinstance(page, GroupedToolPage)
        assert page.function_combo.currentText() == "Decoupling"
        assert page.stack.currentWidget() is page.pages[0]
    finally:
        _dispose(window, qt_app)


def test_switching_keeps_files_parameters_and_results(qt_app, tmp_path):
    first, second = _AsyncTool(), _AsyncTool()
    second.name = "Second function"
    page = GroupedToolPage(_Group([first, second]), lambda _: None)
    path = tmp_path / "anode.csv"
    path.write_text("E(V),j(A/cm2)\n1.3,0\n1.5,0.1\n", encoding="utf-8")
    try:
        first_page = page.pages[0]
        first_page.file_input._add_paths([path])
        first_page.panel.set_values({"label": "anode"})
        first_page.results.set_result(ToolResult(text_blocks={"Saved": "previous result"}))
        page.function_combo.setCurrentIndex(1)
        page.pages[1].panel.set_values({"label": "cathode"})
        page.function_combo.setCurrentIndex(0)
        assert page.stack.currentWidget() is first_page
        assert first_page.file_input.files() == [path]
        assert first_page.panel.values()["label"] == "anode"
        assert "previous result" in first_page.results.summary.toPlainText()
        assert page.pages[1].panel.values()["label"] == "cathode"
    finally:
        _dispose(page, qt_app)


@pytest.mark.parametrize("fail", [False, True])
def test_selector_locked_during_worker_and_recovers(qt_app, monkeypatch, fail):
    first, second = _AsyncTool(fail=fail), _AsyncTool()
    page = GroupedToolPage(_Group([first, second]), lambda _: None)
    errors = []
    monkeypatch.setattr(app_module.QMessageBox, "critical", lambda *args: errors.append(args))
    signals = []
    page.pages[0].runningChanged.connect(signals.append)
    try:
        page.pages[0].run_btn.click()
        assert not page.function_combo.isEnabled()
        _wait_until(lambda: bool(first.calls))
        first.release.set()
        _wait_until(lambda: not page.pages[0]._running)
        assert page.function_combo.isEnabled()
        assert signals == [True, False]
        assert bool(errors) is fail
    finally:
        first.release.set()
        assert QThreadPool.globalInstance().waitForDone(5000)
        _dispose(page, qt_app)


def test_selector_recovers_after_submission_failure(qt_app, monkeypatch):
    from types import SimpleNamespace

    page = GroupedToolPage(_Group([_AsyncTool(), _AsyncTool()]), lambda _: None)
    monkeypatch.setattr(app_module.QMessageBox, "critical", lambda *args: None)

    def fail_start(_worker):
        raise RuntimeError("Pool unavailable")

    monkeypatch.setattr(app_module, "QThreadPool", SimpleNamespace(
        globalInstance=lambda: SimpleNamespace(start=fail_start),
    ))
    try:
        page.pages[0].run_btn.click()
        assert page.function_combo.isEnabled()
        assert not page.pages[0]._running
        assert QGuiApplication.overrideCursor() is None
    finally:
        _dispose(page, qt_app)


class _CustomInputs(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.paths = []
        self.parameter_history = []

    def values(self):
        return {"anode": str(self.paths[0]) if self.paths else ""}

    def files(self):
        return list(self.paths)

    def set_parameters(self, params):
        self.parameter_history.append(dict(params))


class _CustomTool(_AsyncTool):
    needs_files = False
    extra_before_parameters = True

    def extra_widget(self, parent, on_change):
        return _CustomInputs(parent)


def test_custom_inputs_receive_parameters_and_capture_export_directory(qt_app, tmp_path):
    tool = _CustomTool()
    page = ToolPage(tool, lambda _: None)
    original_dir, replacement_dir = tmp_path / "original", tmp_path / "replacement"
    original_dir.mkdir()
    replacement_dir.mkdir()
    original, replacement = original_dir / "input.csv", replacement_dir / "input.csv"
    try:
        assert page.extra.parameter_history == [{"label": "sample"}]
        page.panel.set_values({"label": "updated"})
        assert page.extra.parameter_history[-1] == {"label": "updated"}
        parent_layout = page.extra.parentWidget().layout()
        # Card owns the common input layout; the extra widget precedes parameters.
        body = parent_layout.itemAt(1).layout()
        assert body.indexOf(page.extra) < body.indexOf(page.panel)
        page.extra.paths = [original]
        page.run_btn.click()
        _wait_until(lambda: bool(tool.calls))
        page.extra.paths = [replacement]
        tool.release.set()
        _wait_until(lambda: not page._running)
        assert tool.calls[0][0] == [original]
        assert tool.calls[0][2] == {"anode": str(original)}
        assert page.export_btn._start_dir() == str(original_dir)
    finally:
        tool.release.set()
        assert QThreadPool.globalInstance().waitForDone(5000)
        _dispose(page, qt_app)


def test_real_role_panels_keep_one_file_and_independent_settings(qt_app, tmp_path):
    changes = []
    inputs = IntegratedReInputs(on_change=lambda: changes.append(True))
    selected = {}
    try:
        assert inputs.tabs.count() == 4
        assert list(inputs.inputs) == list(ROLES)
        assert not inputs.tabs.isTabVisible(3)
        for index, role in enumerate(ROLES):
            path, ignored = tmp_path / f"{role}.csv", tmp_path / f"{role}_extra.csv"
            path.write_text("E(V),j(A/cm2)\n0,0\n1,0.1\n", encoding="utf-8")
            ignored.write_text("E(V),j(A/cm2)\n0,0\n1,0.2\n", encoding="utf-8")
            selected[role] = path
            picker = inputs.inputs[role]
            assert not picker._multiple
            assert not picker._accepts_directory
            picker._add_paths([path, ignored])
            assert picker.files() == [path]
            inputs.panels[role].set_values({f"{role}_area": index + 1.0})
        inputs.panels["cathode"].set_values({"cathode_sign": "Invert"})
        inputs.set_parameters({"correction_mode": "3E onset"})
        inputs.tabs.setCurrentIndex(3)
        inputs.tabs.setCurrentIndex(0)
        inputs.tabs.setCurrentIndex(1)
        values = inputs.values()
        for index, role in enumerate(ROLES):
            assert values[role] == str(selected[role])
            assert values["parameters"][f"{role}_area"] == index + 1.0
        assert values["parameters"]["cathode_sign"] == "Invert"
        assert values["parameters"]["anode_sign"] == "Auto"
        assert changes
    finally:
        _dispose(inputs, qt_app)


def test_main_parameters_toggle_calibration_without_discarding_selection(qt_app, tmp_path):
    page = ToolPage(LsvAnalysisTool().subtools()[2], lambda _: None)
    calibration = tmp_path / "calibration.csv"
    calibration.write_text("E(V),j(A/cm2)\n-0.1,0\n-0.2,-0.1\n", encoding="utf-8")
    try:
        assert not page.extra.tabs.isTabVisible(3)
        page.panel.set_values({"correction_mode": "3E onset"})
        assert page.extra.tabs.isTabVisible(3)
        page.extra.inputs["calibration"]._add_paths([calibration])
        page.extra.panels["calibration"].set_values({"calibration_area": 2.0})
        page.extra.tabs.setCurrentIndex(3)
        page.panel.set_values({"correction_mode": "Manual"})
        assert not page.extra.tabs.isTabVisible(3)
        assert page.extra.inputs["calibration"].files() == [calibration]
        assert calibration not in page._input_files()
        page.panel.set_values({"correction_mode": "3E peak"})
        assert page.extra.tabs.isTabVisible(3)
        assert page.extra.values()["calibration"] == str(calibration)
        assert page.extra.values()["parameters"]["calibration_area"] == 2.0
        assert page._input_files() == [calibration]
        parent_layout = page.extra.parentWidget().layout()
        body = parent_layout.itemAt(1).layout()
        assert body.indexOf(page.extra) < body.indexOf(page.panel)
    finally:
        _dispose(page, qt_app)


def test_role_eis_inputs_retain_multiple_files_and_keep_lsv_paths_first(qt_app, tmp_path):
    inputs = IntegratedReInputs()
    lsv_paths, eis_paths = [], []
    try:
        assert list(inputs.eis_inputs) == ["anode", "cathode", "cell"]
        assert list(inputs.eis_panels) == ["anode", "cathode", "cell"]
        assert inputs.eis_tabs.count() == 3
        assert [inputs.eis_tabs.tabText(index) for index in range(3)] == [
            "Anode EIS", "Cathode EIS", "Cell EIS",
        ]
        # EIS roles can be selected without switching away from the current
        # LSV role; the uploaders must not be hidden inside an LSV tab.
        assert not inputs.tabs.isAncestorOf(inputs.eis_tabs)
        for index, role in enumerate(ROLES):
            lsv = tmp_path / f"{role}_lsv.csv"
            lsv.write_text("E(V),j(A/cm2)\n0,0\n1,0.1\n", encoding="utf-8")
            inputs.inputs[role]._add_paths([lsv])
            lsv_paths.append(lsv)
            inputs.panels[role].set_values({f"{role}_area": index + 1})
            if role == "calibration":
                assert "calibration_hfr" not in inputs.panels[role]._widgets
                continue
            spectra = [tmp_path / f"{role}_eis_{n}.csv" for n in (1, 2)]
            for path in spectra:
                path.write_text("f,Re,Im\n10000,0.1,0\n1000,0.2,-0.1\n", encoding="utf-8")
            picker = inputs.eis_inputs[role]
            assert picker._multiple and not picker._accepts_directory
            assert inputs.eis_tabs.widget(index).isAncestorOf(picker)
            assert not inputs.tabs.isAncestorOf(picker)
            assert inputs.eis_tabs.widget(index).isAncestorOf(inputs.eis_panels[role])
            picker._add_paths(spectra)
            eis_paths.extend(spectra)
            inputs.tabs.setCurrentIndex(index)
            readout = inputs.panels[role]._widgets[f"{role}_hfr"]
            assert isinstance(readout, QLineEdit) and readout.isReadOnly()
            assert readout.text() == "Not calculated"
            eis_readout = inputs.eis_panels[role]._widgets[f"{role}_hfr"]
            assert isinstance(eis_readout, QLineEdit) and eis_readout.isReadOnly()
            assert eis_readout.text() == "Not calculated"

        snapshot = inputs.values()
        assert inputs.files() == lsv_paths[:3] + eis_paths
        inputs.tabs.setCurrentIndex(1)
        for index in (2, 0, 1):
            inputs.eis_tabs.setCurrentIndex(index)
            assert inputs.tabs.currentIndex() == 1
        assert inputs.values() == snapshot
        inputs.set_parameters({"correction_mode": "3E onset"})
        assert inputs.files() == lsv_paths + eis_paths
        inputs.set_parameters({"correction_mode": "None"})
        assert all(inputs.eis_tabs.isTabVisible(index) for index in range(3))
        for index, role in enumerate(["anode", "cathode", "cell"]):
            assert snapshot["eis"][role] == [str(path) for path in eis_paths[index * 2:index * 2 + 2]]
            assert inputs.eis_inputs[role].files() == eis_paths[index * 2:index * 2 + 2]
            assert snapshot["parameters"][f"{role}_area"] == index + 1
        assert inputs.inputs["calibration"].files() == [lsv_paths[3]]

        inputs.eis_inputs["anode"].clear()
        assert len(snapshot["eis"]["anode"]) == 2
        assert inputs.values()["eis"]["anode"] == []
        assert inputs.values()["eis"]["cathode"] == snapshot["eis"]["cathode"]
        assert inputs.values()["eis"]["cell"] == snapshot["eis"]["cell"]
    finally:
        _dispose(inputs, qt_app)


def test_role_hfr_updates_are_read_only_and_ignore_unrelated_parameters(qt_app):
    inputs = IntegratedReInputs()
    try:
        inputs.panels["anode"].set_values({"anode_area": 2})
        inputs.set_result_parameters(
            {"anode_hfr": "0.123456", "cathode_hfr": "Not available",
             "anode_area": 99, "unknown_key": "ignored"},
            {"anode_hfr": "2 spectra", "unknown_key": "ignored"},
        )
        assert inputs.panels["anode"].values()["anode_area"] == 2
        for panels in (inputs.panels, inputs.eis_panels):
            assert panels["anode"].values()["anode_hfr"] == "0.123456"
            assert panels["cathode"].values()["cathode_hfr"] == "Not available"
            assert panels["cell"].values()["cell_hfr"] == "Not calculated"
            assert panels["anode"]._annotations["anode_hfr"].text() == "2 spectra"
            for role in ("anode", "cathode", "cell"):
                widget = panels[role]._widgets[f"{role}_hfr"]
                assert isinstance(widget, QLineEdit) and widget.isReadOnly()
        inputs.set_result_parameters({"anode_hfr": "Not available"}, {})
        for panels in (inputs.panels, inputs.eis_panels):
            assert panels["anode"].values()["anode_hfr"] == "Not available"
            assert panels["anode"]._annotations["anode_hfr"].text() == ""
    finally:
        _dispose(inputs, qt_app)


def test_worker_captures_eis_selection_and_syncs_role_hfr_readouts(qt_app, tmp_path, monkeypatch):
    tool = LsvAnalysisTool().subtools()[2]
    status, calls = [], []
    release = Event()
    figure = Figure()
    figure.subplots().plot([0, 1], [0, 1])
    result = ToolResult(
        figures={"Electrodes": figure},
        tables={"HFR detail": pd.DataFrame({"HFR (Ω·cm²)": [0.12, 0.16]})},
        warnings=["Example spectrum note"],
        param_updates={"anode_hfr": "0.14", "cathode_hfr": "Not available",
                       "cell_hfr": "Not available"},
        param_annotations={"anode_hfr": "2 spectra"},
        display_tables=False, display_summary=False,
    )

    def captured_run(files, params, extra=None):
        calls.append((files, params, extra))
        if not release.wait(5):
            raise TimeoutError("Integrated worker not released")
        return result

    monkeypatch.setattr(tool, "run", captured_run)
    page = ToolPage(tool, status.append)
    anode = tmp_path / "anode.csv"
    eis = tmp_path / "eis.csv"
    for path in (anode, eis):
        path.write_text("E(V),j(A/cm2)\n0,0\n1,0.1\n", encoding="utf-8")
    try:
        page.extra.inputs["anode"]._add_paths([anode])
        page.extra.eis_inputs["anode"]._add_paths([eis])
        page.run_btn.click()
        _wait_until(lambda: bool(calls))
        page.extra.eis_inputs["anode"].clear()
        release.set()
        _wait_until(lambda: not page._running)
        assert calls[0][0] == [anode, eis]
        assert calls[0][2]["eis"] == {"anode": [str(eis)], "cathode": [], "cell": []}
        assert page.extra.panels["anode"].values()["anode_hfr"] == "0.14"
        assert page.extra.panels["anode"]._annotations["anode_hfr"].text() == "2 spectra"
        assert page.extra.eis_panels["anode"].values()["anode_hfr"] == "0.14"
        assert page.extra.eis_panels["anode"]._annotations["anode_hfr"].text() == "2 spectra"
        assert page.results.tabs.count() == 1
        assert page.results.tabs.tabText(0) == "Electrodes"
        assert "details included in export" in status[-1]
        assert "Summary" not in status[-1]
        assert {artifact.kind for artifact in page.last_artifacts} == {"csv", "xlsx", "png"}
        assert page.export_btn._start_dir() == str(tmp_path)
    finally:
        release.set()
        assert QThreadPool.globalInstance().waitForDone(5000)
        _dispose(page, qt_app)


def test_three_electrode_worker_syncs_read_only_offset_values(qt_app, tmp_path, monkeypatch):
    page = ToolPage(LsvAnalysisTool().subtools()[2], lambda _: None)
    errors = []
    monkeypatch.setattr(app_module.QMessageBox, "critical", lambda *args: errors.append(args))
    j = np.array([0.0, 0.01, 0.1, 0.3, 0.5])
    data = {
        "anode": (1.4 + 0.8 * j + 0.2, j),
        "cathode": (-0.1 - 0.4 * j + 0.2, -j),
        "calibration": (-0.1 - 0.4 * j, -j),
    }
    try:
        calculated = page.panel._widgets["calculated_shift"]
        applied = page.panel._widgets["applied_shift"]
        fine_adjustment = page.panel._widgets["manual_shift"]
        assert isinstance(calculated, QLineEdit) and calculated.isReadOnly()
        assert isinstance(applied, QLineEdit) and applied.isReadOnly()
        assert isinstance(fine_adjustment, QDoubleSpinBox)
        assert not fine_adjustment.isReadOnly()
        assert calculated.text() == applied.text() == "Not calculated"
        page.panel.set_values({"correction_mode": "3E onset", "manual_shift": 0.03,
                               "calibration_reference": "RHE"})
        for role, (potential, current) in data.items():
            path = tmp_path / f"{role}.csv"
            pd.DataFrame({"E(V)": potential, "j(A/cm2)": current}).to_csv(path, index=False)
            page.extra.inputs[role]._add_paths([path])

        for _ in range(2):
            page.run_btn.click()
            _wait_until(lambda: not page._running)
            assert not errors
            assert calculated.text() == "-0.200000"
            assert applied.text() == "-0.170000"
            assert fine_adjustment.value() == pytest.approx(0.03)
            assert not page.panel._rows["calculated_shift"].isHidden()
            assert not page.panel._rows["applied_shift"].isHidden()
            assert "-0.17" in page.results.summary.toPlainText()

        # Disabling correction must replace previous result values, even while
        # those readouts are hidden; stale readouts remain display-only inputs.
        page.panel.set_values({"correction_mode": "None"})
        page.run_btn.click()
        _wait_until(lambda: not page._running)
        assert not errors
        assert calculated.text() == applied.text() == "+0.000000"
        assert fine_adjustment.value() == pytest.approx(0.03)
        assert page.panel._rows["calculated_shift"].isHidden()
        assert page.panel._rows["applied_shift"].isHidden()
    finally:
        assert QThreadPool.globalInstance().waitForDone(5000)
        _dispose(page, qt_app)


def test_grouped_integrated_run_captures_role_settings_and_export_directory(
        qt_app, tmp_path, monkeypatch):
    page = GroupedToolPage(LsvAnalysisTool(), lambda _: None)
    integrated = page.pages[2]
    original_run = integrated.tool.run
    release = Event()
    calls, results, errors = [], [], []
    monkeypatch.setattr(app_module.QMessageBox, "critical", lambda *args: errors.append(args))

    def captured_run(files, params, extra=None):
        calls.append((files, params, extra))
        if not release.wait(5):
            raise TimeoutError("Integrated worker not released")
        result = original_run(files, params, extra)
        results.append(result)
        return result

    monkeypatch.setattr(integrated.tool, "run", captured_run)
    source = tmp_path / "source"
    source.mkdir()
    replacement = tmp_path / "replacement.csv"
    replacement.write_text("E(V),Current (A)\n1.4,0\n1.5,0.4\n", encoding="utf-8")
    files = {}
    data = {
        "anode": ([1.4, 1.5, 1.6], [0, 0.2, 0.4], 2.0),
        "cathode": ([-0.05, -0.15, -0.25], [0, -0.4, -0.8], 4.0),
        "cell": ([1.45, 1.65, 1.85], [0, 0.2, 0.4], 2.0),
    }
    try:
        page.function_combo.setCurrentIndex(2)
        for role, (potential, current, area) in data.items():
            path = source / f"{role}.csv"
            pd.DataFrame({"E (V)": potential, "Current (A)": current}).to_csv(path, index=False)
            files[role] = path
            integrated.extra.inputs[role]._add_paths([path])
            integrated.extra.panels[role].set_values({f"{role}_area": area,
                                                       f"{role}_sign": "As recorded"})
        integrated.run_btn.click()
        assert not page.function_combo.isEnabled()
        _wait_until(lambda: bool(calls))
        integrated.extra.inputs["anode"]._add_paths([replacement])
        integrated.extra.panels["anode"].set_values({"anode_area": 10.0})
        release.set()
        _wait_until(lambda: not integrated._running)
        assert not errors
        assert page.function_combo.isEnabled()
        assert calls[0][0] == [files["anode"], files["cathode"], files["cell"]]
        assert calls[0][2]["anode"] == str(files["anode"])
        assert calls[0][2]["parameters"]["anode_area"] == 2.0
        assert calls[0][2]["parameters"]["cathode_area"] == 4.0
        assert integrated.export_btn._start_dir() == str(source)
        assert np.allclose(results[0].tables["Anode LSV"]["Current density (A/cm²)"], [0, 0.1, 0.2])
        assert np.allclose(results[0].tables["Cathode LSV"]["Current density (A/cm²)"], [0, -0.1, -0.2])
        assert len(integrated.last_artifacts) > 0
        page.function_combo.setCurrentIndex(0)
        page.function_combo.setCurrentIndex(2)
        assert integrated.extra.inputs["anode"].files() == [replacement]
        assert integrated.extra.panels["anode"].values()["anode_area"] == 10.0
        assert "Integrated RE review" in integrated.results.summary.toPlainText()
    finally:
        release.set()
        assert QThreadPool.globalInstance().waitForDone(5000)
        _dispose(page, qt_app)


@pytest.mark.parametrize("extension", ["csv", "txt", "xlsx"])
def test_raw_reader_preserves_full_sweep_sign_zeros_and_units(tmp_path, extension):
    path = tmp_path / f"complete.{extension}"
    voltage = [100, 200, 300, 200, 100]
    current = [-2, 0, 3, 2, -1]
    frame = pd.DataFrame({"Potential (mV)": voltage, "Current (mA)": current})
    if extension == "xlsx":
        frame.to_excel(path, index=False)
    else:
        frame.to_csv(path, sep="," if extension == "csv" else "\t", index=False)
    parsed, error = read_lsv_curve(path)
    assert error is None
    potential, values, info = parsed
    assert np.allclose(potential, np.array(voltage) / 1000)
    assert np.allclose(values, np.array(current) / 1000)
    assert not info["is_density"]
    assert not info["sweep_trimmed"]
    assert info["n_rows"] == 5
    legacy, error = read_lsv_data([path], reaction="oer")
    assert error is None
    assert legacy[2]["sweep_trimmed"]
    assert np.allclose(legacy[0], [0.3])


def test_raw_reader_reports_missing_and_invalid_inputs(tmp_path):
    assert read_lsv_curve(None) == (None, "No file selected.")
    path = tmp_path / "invalid.csv"
    path.write_text("E(V),j(A/cm2)\nnot,numeric\n", encoding="utf-8")
    parsed, error = read_lsv_curve(path)
    assert parsed is None
    assert "No numeric rows" in error


@pytest.mark.parametrize("extension", ["csv", "xlsx"])
def test_raw_reader_keeps_missing_measurements_as_segment_gaps(tmp_path, extension):
    path = tmp_path / f"gapped.{extension}"
    frame = pd.DataFrame({
        "Potential (mV)": [100, 200, "missing", 400, 500],
        "Current (mA)": [0, 1, 2, "missing", 4],
    })
    if extension == "xlsx":
        frame.to_excel(path, index=False)
    else:
        frame.to_csv(path, index=False)
    parsed, error = read_lsv_curve(path)
    assert error is None
    potential, current, info = parsed
    assert info["n_rows"] == 5
    assert info["invalid_rows"] == 2
    assert np.allclose(potential, [0.1, 0.2, np.nan, 0.4, 0.5], equal_nan=True)
    assert np.allclose(current, [0, 0.001, 0.002, np.nan, 0.004], equal_nan=True)
    assert np.array_equal(np.isfinite(potential) & np.isfinite(current),
                          [True, True, False, False, True])
    # Existing Decoupling import keeps its previous invalid-row filtering.
    legacy, error = read_lsv_data([path], reaction="oer")
    assert error is None
    assert np.allclose(legacy[0], [0.2, 0.5])
    assert np.allclose(legacy[1], [0.001, 0.004])


def test_raw_reader_keeps_infinite_measurements_and_rejects_no_finite_pair(tmp_path):
    path = tmp_path / "infinite.csv"
    path.write_text("E(V),j(A/cm2)\n0,0\ninf,0.1\n0.2,-inf\n0.3,0.2\n", encoding="utf-8")
    parsed, error = read_lsv_curve(path)
    assert error is None
    assert np.array_equal(parsed[0], [0, np.inf, 0.2, 0.3])
    assert np.array_equal(parsed[1], [0, 0.1, -np.inf, 0.2])
    assert parsed[2]["invalid_rows"] == 2
    path.write_text("E(V),j(A/cm2)\ninf,1\n0,inf\n", encoding="utf-8")
    parsed, error = read_lsv_curve(path)
    assert parsed is None
    assert "No numeric rows" in error
