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


@pytest.mark.parametrize("display_summary", [False, True])
@pytest.mark.parametrize("display_tables", [False, True])
def test_display_flags_preserve_detail_data(panel, display_summary, display_tables):
    result = _result(7)
    source_table = result.tables["Data"]
    result.display_summary = display_summary
    result.display_tables = display_tables

    panel.set_result(result)

    assert panel.tabs.count() == 1 + int(display_summary) + int(display_tables)
    assert isinstance(panel.tabs.widget(0), SingleFigure)
    assert (panel.tabs.indexOf(panel.summary) >= 0) is display_summary
    assert len(panel.findChildren(QTableWidget)) == int(display_tables)
    assert "Value" in panel.summary.toPlainText()
    assert "7" in panel.summary.toPlainText()
    assert result.tables["Data"] is source_table
    assert result.tables["Data"].iloc[0, 0] == 7


def test_repeated_plots_only_results_release_pages_and_restore_default_view(panel):
    summary = panel.summary
    for value in (1, 2):
        result = _result(value)
        result.display_tables = result.display_summary = False
        previous = panel.tabs.widget(0)
        panel.set_result(result)
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        if previous is not summary:
            assert not isValid(previous)
        assert panel.tabs.count() == 1
        assert len(panel.findChildren(SingleFigure)) == 1
        assert not panel.findChildren(QTableWidget)
        assert isValid(summary)

    panel.set_result(_result(3))
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert panel.tabs.count() == 3
    assert panel.tabs.widget(1) is summary
    assert len(panel.findChildren(SingleFigure)) == 1
    assert len(panel.findChildren(QTableWidget)) == 1

    panel.clear()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert panel.tabs.count() == 1
    assert panel.tabs.widget(0) is summary
    assert not panel.findChildren(SingleFigure)
