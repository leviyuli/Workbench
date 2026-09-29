"""Result page lifecycle regressions for repeated runs and clearing results."""

import pandas as pd
import pytest
from matplotlib.figure import Figure
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QTableWidget
from shiboken6 import isValid

from workbench.core.tool_base import ToolResult
from workbench.widgets.figure_view import SingleFigure
from workbench.widgets.results_panel import ResultsPanel


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance()
    if app is None:
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
            app = QApplication([])
    yield app


@pytest.fixture
def panel(qt_app):
    widget = ResultsPanel()
    yield widget
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def _result(value):
    figure = Figure()
    figure.subplots().plot([0, 1], [0, value])
    return ToolResult(
        figures={"Plot": figure},
        tables={"Data": pd.DataFrame({"Value": [value]})},
        scalars={"Value": value},
    )


def test_replacing_results_releases_old_plot_and_table_pages(panel):
    summary = panel.summary
    panel.set_result(_result(1))

    for value in (2, 3):
        old_plot = panel.tabs.widget(0)
        old_canvas = old_plot.canvas
        old_table = panel.tabs.widget(2)

        panel.set_result(_result(value))
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

        assert not isValid(old_plot)
        assert not isValid(old_canvas)
        assert not isValid(old_table)
        assert isValid(summary)
        assert panel.tabs.widget(1) is summary
        assert panel.tabs.count() == 3
        assert len(panel.findChildren(SingleFigure)) == 1
        assert len(panel.findChildren(QTableWidget)) == 1
        assert panel.tabs.widget(2).item(0, 0).text() == str(value)


def test_clearing_results_releases_pages_and_preserves_summary(panel):
    summary = panel.summary
    panel.set_result(_result(1))
    old_plot = panel.tabs.widget(0)
    old_canvas = old_plot.canvas
    old_table = panel.tabs.widget(2)

    panel.clear()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    assert not isValid(old_plot)
    assert not isValid(old_canvas)
    assert not isValid(old_table)
    assert isValid(summary)
    assert panel.tabs.count() == 1
    assert panel.tabs.widget(0) is summary
    assert summary.toPlainText() == "No results yet. Configure inputs and run the tool."
    assert not panel.findChildren(SingleFigure)
    assert not panel.findChildren(QTableWidget)

    panel.clear()
    panel.set_result(ToolResult(scalars={"Value": 2}))
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    assert isValid(summary)
    assert panel.tabs.count() == 1
    assert panel.tabs.widget(0) is summary
    assert "Value" in summary.toPlainText()
