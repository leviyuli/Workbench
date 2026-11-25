"""CV dots and SD describe cycle variation independently for each input file."""
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from matplotlib.container import ErrorbarContainer

from workbench.tools.ccd.legacy import CCD_COLUMN, LIMIT_COLUMN, LegacyCcdTool as CcdTool


METRICS = [("CCD", CCD_COLUMN), ("Limiting current", LIMIT_COLUMN)]


def _write_cv(path, cycles):
    """Write (cycle ID, current multiplier, kind) as independent wide scans."""
    voltage = np.arange(1, 3.6001, .002)
    columns = {}
    for cycle, scale, kind in cycles:
        v = voltage.copy()
        j = np.maximum(4 * (v - 1.6), 0)
        if kind != "linear":
            high = v > 2.6
            j[high] = 5 - np.exp(-4 * (v[high] - 2.6))
        if kind == "short":
            v, j = v[:100], j[:100]
        elif kind == "unfinished":
            selected = v <= 3.0
            v, j = v[selected], j[selected]
        columns[f"Cycle {cycle} E(V)"] = pd.Series(v)
        columns[f"Cycle {cycle} j(A/cm2)"] = pd.Series(scale * j)
    pd.DataFrame(columns).to_csv(path, index=False)


def _artist(fig, gid):
    matches = [artist for ax in fig.axes for artist in ax.get_children()
               if artist.get_gid() == gid]
    assert len(matches) == 1, gid
    return matches[0]


def _errors(fig):
    return [container for ax in fig.axes for container in ax.containers
            if isinstance(container, ErrorbarContainer)]


def test_file_cycle_means_sd_and_plotted_dots_match_individual_scan_results(tmp_path):
    paths = [tmp_path / "first.csv", tmp_path / "second.csv"]
    _write_cv(paths[0], [(1, 1., "plateau"), (2, 1.8, "plateau"), (3, .7, "plateau")])
    _write_cv(paths[1], [(1, 1.4, "plateau"), (2, .8, "plateau"), (3, 2., "plateau")])
    result = CcdTool().run(paths, {})
    rows = result.tables["Current results"]
    cycle_stats = result.tables["Cycle statistics"]
    batch_stats = result.tables["Batch statistics"]
    fig = result.figures["Current statistics"]
    assert len(cycle_stats) == 4
    assert "Cycle trends" in result.figures
    for source, path in zip(["S01", "S02"], paths):
        source_rows = rows[rows["File"] == path.name]
        for metric, column in METRICS:
            values = source_rows[column].to_numpy(dtype=float)
            stats = cycle_stats[(cycle_stats["Source"] == source)
                                & (cycle_stats["Metric"] == metric)].iloc[0]
            assert stats["File"] == path.name
            assert stats["Resolved cycles"] == 3
            assert stats["Mean (A/cm²)"] == pytest.approx(np.mean(values))
            assert stats["Sample SD (A/cm²)"] == pytest.approx(np.std(values, ddof=1))
            bar = _artist(fig, f"cycle-mean:{source}:{metric}")
            assert bar.get_height() == pytest.approx(np.mean(values))
            error = next(container for container in _errors(fig)
                         if container.get_label() == f"cycle-sd:{source}:{metric}")
            interval = error.lines[2][0].get_segments()[0][:, 1]
            np.testing.assert_allclose(interval, np.mean(values) + np.array([-1, 1])
                                       * np.std(values, ddof=1))
            for _, row in source_rows.iterrows():
                dot = _artist(fig, f"cycle-point:{row['Curve']}:{metric}")
                assert np.asarray(dot.get_ydata(), dtype=float) == pytest.approx([row[column]])
    # The replicate table compares files at the same cycle; its n and SD
    # must not be replaced with the descriptive within-file cycle statistics.
    for cycle in [1, 2, 3]:
        for metric, column in METRICS:
            values = rows.loc[rows["Cycle"] == cycle, column].to_numpy(dtype=float)
            stats = batch_stats[(batch_stats["Cycle"] == cycle)
                                & (batch_stats["Metric"] == metric)].iloc[0]
            assert stats["Resolved files"] == 2
            assert stats["Mean (A/cm²)"] == pytest.approx(np.mean(values))
            assert stats["Sample SD (A/cm²)"] == pytest.approx(np.std(values, ddof=1))


def test_metric_specific_resolution_controls_n_sd_and_errorbars(tmp_path):
    path = tmp_path / "partial.csv"
    _write_cv(path, [(1, 1., "plateau"), (2, 1.2, "unfinished"), (3, .8, "linear")])
    result = CcdTool().run([path], {})
    rows = result.tables["Current results"]
    stats = result.tables["Cycle statistics"].set_index("Metric")
    assert stats.loc["CCD", "Resolved cycles"] == 2
    assert stats.loc["Limiting current", "Resolved cycles"] == 1
    assert stats.loc["CCD", "Sample SD (A/cm²)"] == pytest.approx(rows[CCD_COLUMN].std(ddof=1))
    assert pd.isna(stats.loc["Limiting current", "Sample SD (A/cm²)"])
    errors = _errors(result.figures["Current statistics"])
    assert [item.get_label() for item in errors] == ["cycle-sd:S01:CCD"]
    texts = [text.get_text() for ax in result.figures["Current statistics"].axes
             for text in ax.texts]
    assert any("SD undefined" in text for text in texts)


def test_all_unresolved_cycles_keep_missing_means_and_have_no_sd_bars(tmp_path):
    path = tmp_path / "linear.csv"
    _write_cv(path, [(1, 1., "linear"), (2, 1.5, "linear")])
    result = CcdTool().run([path], {})
    stats = result.tables["Cycle statistics"]
    assert stats["Resolved cycles"].eq(0).all()
    assert stats[["Mean (A/cm²)", "Sample SD (A/cm²)"]].isna().all().all()
    fig = result.figures["Current statistics"]
    assert not _errors(fig)
    assert not [artist for ax in fig.axes for artist in ax.get_children()
                if str(artist.get_gid()).startswith("cycle-point:")]
    assert sum("Not resolved" in text.get_text() for ax in fig.axes for text in ax.texts) == 2


def test_cycle_opacity_follows_actual_ids_and_survives_unresolved_cycles(tmp_path):
    path = tmp_path / "gapped.csv"
    _write_cv(path, [(1, 1., "plateau"), (2, 1., "short"),
                    (3, 1.2, "plateau"), (7, 1.6, "plateau"), (9, 1., "linear")])
    result = CcdTool().run([path], {})
    rows = result.tables["Current results"]
    assert rows["Cycle"].tolist() == [1, 2, 3, 7, 9]
    assert pd.isna(rows.loc[rows["Cycle"] == 2, CCD_COLUMN].iloc[0])
    processed = result.tables["Processed curves"]
    scan_colors = {}
    for cycle in [1, 3, 7, 9]:
        data = processed[processed["Cycle"] == cycle]
        matching_lines = [line for line in result.figures["LSV processing"].axes[0].lines
                          if len(line.get_xdata()) == len(data)
                          and np.allclose(line.get_xdata(), data["Measured potential (V)"])
                          and np.allclose(line.get_ydata(), data["Smoothed current (A/cm²)"])]
        assert len(matching_lines) == 1
        line = matching_lines[0]
        scan_colors[cycle] = to_rgba(line.get_color(), line.get_alpha())
    low, high = scan_colors[1][3], scan_colors[9][3]
    assert low < high
    for cycle, rgba in scan_colors.items():
        np.testing.assert_allclose(rgba[:3], scan_colors[1][:3])
        assert rgba[3] == pytest.approx(low + (high - low) * (cycle - 1) / 8)
    for cycle in [1, 3, 7]:
        for metric, _ in METRICS:
            dot = _artist(result.figures["Current statistics"],
                          f"cycle-point:S01-C{cycle:02d}:{metric}")
            assert to_rgba(dot.get_color(), dot.get_alpha()) == pytest.approx(scan_colors[cycle])


def test_identically_named_cv_files_keep_separate_cycle_statistics(tmp_path):
    paths = []
    for folder, scale in [("a", 1.), ("b", 2.)]:
        directory = tmp_path / folder
        directory.mkdir()
        path = directory / "cv.csv"
        _write_cv(path, [(1, scale, "plateau"), (2, 1.3 * scale, "plateau")])
        paths.append(path)
    result = CcdTool().run(paths, {})
    stats = result.tables["Cycle statistics"]
    assert len(stats) == 4
    assert stats["Source"].tolist() == ["S01", "S01", "S02", "S02"]
    for metric, _ in METRICS:
        selected = stats[stats["Metric"] == metric].set_index("Source")
        assert selected.loc["S02", "Mean (A/cm²)"] == pytest.approx(
            2 * selected.loc["S01", "Mean (A/cm²)"])
