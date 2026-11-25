"""Batch views keep physical scales and sample identities easy to compare."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import matplotlib as mpl
from matplotlib.colors import to_hex, to_rgba

from workbench.tools.ccd.batch_plots import BatchCurve, build_batch_views
from workbench.tools.ccd.plot_style import source_colors


def _views(*, count=3, include_rcl=False):
    """Use known plotted values; numerical fitting is covered by engine tests."""
    palette = source_colors(["S01", "S02", "S03"])
    correction = SimpleNamespace(axis_label=lambda: "Potential (V)")
    settings = SimpleNamespace(i_lower=.005, correction=correction)
    rows = [{"Curve": "S01", "Source": "S01", "File": "failed.csv", "Cycle": None,
             "Scan": "Forward", "CCD (A/cm²)": np.nan,
             "Limiting current (A/cm²)": np.nan}]
    breakdown = [{**rows[0], "Target |j| (A/cm²)": 1., "Kinetic (V)": np.nan,
                  "Residual (V)": np.nan, "RCL (V)": np.nan}]
    curves = []
    for number in range(count):
        source = "S02" if number < max(2, count - 2) else "S03"
        cycle = number + 1
        code = f"{source}-C{cycle:02d}"
        branch = "Back" if number % 2 else "Forward"
        scan = SimpleNamespace(code=code, source=source, file=f"{source}.csv",
                               cycle=cycle, branch=branch)
        current = np.linspace(.01, 10., 15)
        residual = .08 * current + .01 * number
        residual[4] = residual[3] - .01
        native = slice(None, None, -1) if branch == "Back" else slice(None)
        d = SimpleNamespace(
            potential=(1.3 + .2 * current)[native], measured_current=current[native],
            current=current[native], current_magnitude=current[native], settings=settings,
            eta_kin=(.3 + .02 * current)[native], residual=residual[native],
            eta_rcl=(.08 + .005 * current)[native],
        )
        mask = np.zeros(len(current), dtype=bool)
        mask[2:6] = True
        baseline = residual.copy()
        baseline[8:] = np.nan
        r = SimpleNamespace(
            current=current, residual=residual, segment_id=np.r_[np.zeros(8), np.ones(7)],
            baseline=baseline, baseline_mask=mask, ccd_index=6,
            plateau_start=12, plateau_end=14, limiting_current=9.5,
        )
        color = to_rgba(palette[source], .4 + .6 * number / max(count - 1, 1))
        curves.append(BatchCurve(scan, d, r, color))
        row = {"Curve": code, "Source": source, "File": scan.file,
               "Cycle": cycle, "Scan": branch, "CCD (A/cm²)": 4.2 + .1 * number,
               "Limiting current (A/cm²)": 8.7 + .05 * number}
        rows.append(row)
        breakdown.append({**row, "Target |j| (A/cm²)": 1.,
                          "Kinetic (V)": .35 + .01 * number,
                          "Residual (V)": 1.2 + .01 * number,
                          "RCL (V)": .09 + .01 * number})
    return build_batch_views(curves, pd.DataFrame(rows), pd.DataFrame(breakdown), include_rcl), curves


def _visible_ticks(ax, axis="y"):
    ticks = ax.get_yticks() if axis == "y" else ax.get_xticks()
    lower, upper = ax.get_ylim() if axis == "y" else ax.get_xlim()
    return ticks[(ticks >= lower - 1e-12) & (ticks <= upper + 1e-12)]


def test_current_comparison_uses_the_whole_operating_range_for_both_metrics():
    (figures, _, _, _), curves = _views()
    axes = figures["Current comparison"].axes
    np.testing.assert_allclose(axes[0].get_ylim(), axes[1].get_ylim())
    np.testing.assert_allclose(axes[0].get_yticks(), axes[1].get_yticks())
    assert axes[0].get_ylim()[0] == 0
    assert axes[0].get_ylim()[1] >= max(np.nanmax(c.decomposition.current_magnitude) for c in curves)
    assert 3 <= len(_visible_ticks(axes[0])) <= 6
    # A narrow cluster of CCD values must not acquire a narrow, exaggerated scale.
    assert np.ptp(axes[0].get_ylim()) > 9


def test_breakdown_has_one_voltage_tick_interval_and_equal_category_spacing():
    (figures, _, _, _), _ = _views(include_rcl=True)
    axes = figures["Breakdown comparison"].axes
    steps = [np.diff(ax.get_yticks()) for ax in axes]
    for differences in steps:
        assert len(differences)
        np.testing.assert_allclose(differences, differences[0])
    np.testing.assert_allclose([differences[0] for differences in steps], steps[0][0])
    for ax in axes[1:]:
        np.testing.assert_allclose(ax.get_xlim(), axes[0].get_xlim())
        np.testing.assert_allclose(ax.get_xticks(), axes[0].get_xticks())
    assert len({tuple(ax.get_ylim()) for ax in axes}) > 1


def test_dots_and_black_mean_bars_leave_sample_sd_in_statistics():
    (figures, sheets, statistics, _), _ = _views()
    index = sheets["Plot series"]
    assert not index.Series.str.contains("sample SD", regex=False).any()
    means = index.loc[index.Series.str.endswith("CCD mean")]
    assert len(means) == 2
    artists = {line.get_gid(): line for figure in figures.values()
               for ax in figure.axes for line in ax.lines if line.get_gid()}
    for record in means.to_dict("records"):
        mean = artists[record["Artist"]]
        assert to_hex(mean.get_color()) == "#000000"
        assert mean.get_marker() in ("", "None", "none")
        np.testing.assert_allclose(np.diff(mean.get_ydata()), 0)
        assert 0 < np.ptp(mean.get_xdata()) <= .5
    points = index.loc[index["Plot sheet"].eq("Plot CCD points") & index.Marker.eq("o")]
    for record in points.to_dict("records"):
        assert artists[record["Artist"]].get_markersize() == 3
    stats = statistics.loc[statistics.Metric.eq("CCD (A/cm²)") & statistics.Source.eq("S02")].iloc[0]
    assert stats["Mean"] == pytest.approx(4.25)
    assert stats["Sample SD"] == pytest.approx(np.std([4.2, 4.3], ddof=1))


def test_ccd_small_multiples_share_ranges_hide_repeated_numbers_and_keep_gaps():
    (figures, sheets, _, _), curves = _views(count=10)
    axes = figures["CCD & limiting current"].axes
    assert len(axes) == len(curves)
    assert len({tuple(ax.get_xlim()) for ax in axes}) == 1
    assert len({tuple(ax.get_ylim()) for ax in axes}) == 1
    assert axes[0].get_ylim()[0] == 0
    assert axes[0].get_ylim()[1] >= 10
    # In a three-column layout only the left column carries Y numbers.
    for number, ax in enumerate(axes):
        assert bool(ax.get_yticklabels()) == (number % 3 == 0)
        assert any(curves[number].scan.code in text.get_text() for text in ax.texts)
    assert not axes[0].get_xticklabels()
    assert axes[-1].get_xticklabels()
    series = sheets["Plot series"]
    scan = series.loc[series["Plot sheet"].eq("Plot CCD") & series.Series.str.endswith(" · scan")]
    assert len(scan) == len(curves)
    artists = {line.get_gid(): line for ax in axes for line in ax.lines if line.get_gid()}
    for record, curve in zip(scan.to_dict("records"), curves):
        artist = artists[record["Artist"]]
        assert np.isnan(artist.get_xdata()).sum() == 1
        assert np.isnan(artist.get_ydata()).sum() == 1
        np.testing.assert_allclose(artist.get_xdata(), np.insert(curve.ccd.residual, 8, np.nan),
                                   equal_nan=True)
        np.testing.assert_allclose(artist.get_ydata(), np.insert(curve.ccd.current, 8, np.nan),
                                   equal_nan=True)
    originals = sheets["Plot series"].loc[sheets["Plot series"]["Plot sheet"].eq("Plot original")]
    for record, curve in zip(originals.to_dict("records"), curves):
        sheet = sheets[record["Plot sheet"]]
        np.testing.assert_allclose(sheet[record["X column"]].dropna(), curve.decomposition.potential)
        np.testing.assert_allclose(sheet[record["Y column"]].dropna(), curve.decomposition.measured_current)


def test_batch_style_is_local_compact_and_uses_ordered_batlow_categories():
    palette = source_colors(["S01", "S02", "S03"])
    assert [to_hex(color) for color in palette.values()] == ["#011959", "#faccfa", "#828231"]
    previous = {key: mpl.rcParams[key] for key in ["axes.grid", "font.size", "axes.linewidth"]}
    (figures, sheets, _, _), curves = _views()
    assert {key: mpl.rcParams[key] for key in previous} == previous
    series = sheets["Plot series"]
    for curve in curves:
        records = series.loc[series.Curve.eq(curve.scan.code)]
        assert records.Color.eq(to_hex(palette[curve.scan.source])).all()
        np.testing.assert_allclose(records.Alpha, to_rgba(curve.color)[3])
    for figure in figures.values():
        assert not figure._suptitle
        for ax in figure.axes:
            assert not ax.get_title()
            assert not ax.spines["top"].get_visible()
            assert not ax.spines["right"].get_visible()
            assert ax.spines["bottom"].get_linewidth() == .5
            assert ax.spines["left"].get_linewidth() == .5
            assert not any(line.get_visible() for line in ax.get_xgridlines() + ax.get_ygridlines())
            for text in [ax.xaxis.label, ax.yaxis.label, *ax.texts,
                         *ax.get_xticklabels(), *ax.get_yticklabels()]:
                if text.get_text():
                    assert text.get_fontsize() == 7


def test_dense_facet_canvas_scrolls_without_forcing_window_size(monkeypatch):
    from PySide6.QtWidgets import QApplication, QScrollArea
    from workbench.widgets.figure_view import SingleFigure

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    (figures, _, _, _), _ = _views(count=10)
    view = SingleFigure(figures["CCD & limiting current"])
    try:
        view.resize(550, 450)
        view.show()
        app.processEvents()
        scroll = view.findChild(QScrollArea)
        assert scroll is not None
        assert scroll.widget() is view.canvas
        assert view.canvas.minimumHeight() >= 700
        assert view.canvas.minimumWidth() == 0
        assert scroll.verticalScrollBar().maximum() > 0
        assert view.height() == 450
    finally:
        view.close()
        view.deleteLater()
        app.processEvents()
