"""Generator plots, interoperable exports and real-result project round trips."""
from copy import deepcopy
from io import BytesIO
from time import monotonic

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from openpyxl import load_workbook
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtGui import QGuiApplication
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from workbench.app import MainWindow
from workbench.core.data_io import read_eis_data
from workbench.core.project import artifact_bytes
from workbench.tools.eis_analysis.tool import EisAnalysisTool
from workbench.tools.eis_generator.logic import DEFAULT_PARAMETERS
from workbench.tools.eis_generator.tool import EisGeneratorTool


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def _extra():
    return {"cases": [
        {"id": "case_1", "name": "Balanced", "visible": True,
         "params": {**DEFAULT_PARAMETERS, "electronic_conductivity": 0.002}},
        {"id": "case_2", "name": "Hidden", "visible": False,
         "params": {**DEFAULT_PARAMETERS, "thickness_um": 100.0}},
    ], "annotated_case": "case_1", "next_case_number": 3}


@pytest.mark.parametrize("view", ["Transport detail", "Full frequency range"])
def test_plot_has_exact_markers_and_all_generated_points(view):
    result = EisGeneratorTool().run([], {**DEFAULT_PARAMETERS, "view": view}, _extra())
    assert result.scalars["Annotated case"] == "Balanced"
    assert result.scalars["Generated cases"] == 2
    fig = result.figures["CL Nyquist"]
    FigureCanvasAgg(fig).draw()
    detail = fig.axes[-1]
    assert detail.get_aspect() == 1.0
    guides = {line.get_gid(): line for line in detail.lines if line.get_gid()}
    assert guides["asymptote:external"].get_xdata()[0] == pytest.approx(0.05)
    assert guides["asymptote:hf_total"].get_xdata()[0] == pytest.approx(0.175)
    assert guides["asymptote:lf_intercept"].get_xdata()[0] == pytest.approx(0.05 + 0.5 / 3)
    assert len(guides["spectrum:current"].get_xdata()) == 400
    assert len(guides["spectrum:case_1"].get_xdata()) == 400
    assert len(fig.axes) == (2 if view == "Full frequency range" else 1)
    if view == "Full frequency range":
        full = fig.axes[0]
        data = result.tables["Spectrum 1"]
        assert full.get_ylim()[1] >= -data["Z''(Ohm.cm²)"].min()
        assert full.get_window_extent().width > 100


def test_exports_have_signed_imaginary_data_and_can_be_reimported(qt_app, tmp_path):
    tool = EisGeneratorTool()
    result = tool.run([], DEFAULT_PARAMETERS, _extra())
    artifacts = tool.exportable_artifacts(result)
    assert artifacts[0].name == "CL EIS results (Excel)"
    assert artifacts[0].kind == "xlsx"
    csv_artifacts = [artifact for artifact in artifacts if artifact.kind == "csv"]
    assert len(csv_artifacts) == 2
    assert len({artifact.default_filename for artifact in artifacts}) == len(artifacts)
    for artifact in csv_artifacts:
        assert list(artifact.payload) == ["Freq(Hz)", "Z'(Ohm.cm²)", "Z''(Ohm.cm²)"]
        assert (artifact.payload["Z''(Ohm.cm²)"] < 0).all()
        target = tmp_path / artifact.default_filename
        target.write_bytes(artifact_bytes(artifact))
        loaded, error = read_eis_data(target)
        assert error is None
        np.testing.assert_allclose(loaded["z_imag"], artifact.payload["Z''(Ohm.cm²)"])
    workbook = next(artifact for artifact in artifacts if artifact.kind == "xlsx")
    book = load_workbook(BytesIO(artifact_bytes(workbook)), read_only=True)
    try:
        assert book.sheetnames == ["Resistance summary", "Spectrum 1", "Spectrum 2"]
        assert book["Spectrum 1"].cell(2, 3).value < 0
    finally:
        book.close()


def test_real_overlay_generation_project_save_open_and_rerun(qt_app, tmp_path, monkeypatch):
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(args))
    windows = [MainWindow([EisAnalysisTool()]), MainWindow([EisAnalysisTool()])]
    try:
        first, second = windows
        group = first.stack.widget(0)
        group.function_combo.setCurrentIndex(1)
        page = group.pages[1]
        page.extra.add_current_case("Original")
        page.panel.set_values({"electronic_conductivity": 0.002})
        page.extra.add_current_case("Balanced")
        page.panel.set_values({"electronic_conductivity": 0.001})
        page.run_btn.click()
        deadline = monotonic() + 10
        while page._running and monotonic() < deadline:
            QTest.qWait(10)
        assert not page._running and not errors
        assert page.last_result.scalars["Generated cases"] == 3
        original_data = page.last_result.tables["Spectrum 1"].copy()
        original_exports = {a.default_filename: artifact_bytes(a)
                            for a in page.last_artifacts if a.kind == "csv"}
        page.panel.set_values({"thickness_um": 7.0})
        saved_cases = deepcopy(page.extra.values())
        target = tmp_path / "generated.ecwb"
        assert first.project.save_to(target), errors
        assert second.project.open_path(target, prompt=False), errors
        restored = second.stack.widget(0).pages[1]
        assert restored.extra.values() == saved_cases
        assert restored.panel.values()["thickness_um"] == 7.0
        assert restored.last_run["params"]["thickness_um"] == 5.0
        pd.testing.assert_frame_equal(restored.last_result.tables["Spectrum 1"], original_data)
        assert {a.default_filename: artifact_bytes(a) for a in restored.last_artifacts
                if a.kind == "csv"} == original_exports
        restored.run_btn.click()
        deadline = monotonic() + 10
        while restored._running and monotonic() < deadline:
            QTest.qWait(10)
        assert not restored._running and not errors
        assert restored.last_result.scalars["Generated cases"] == 3
        assert not np.array_equal(restored.last_result.tables["Spectrum 1"], original_data)
    finally:
        for window in windows:
            if window.project._loaded is not None:
                window.project._loaded.close()
            window.deleteLater()
        qt_app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        while QGuiApplication.overrideCursor() is not None:
            QGuiApplication.restoreOverrideCursor()
