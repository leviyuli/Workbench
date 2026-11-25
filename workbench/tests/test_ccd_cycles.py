"""Per-cycle CCD integrates parsing, independent fits, trends and compact exports."""
import numpy as np
import pandas as pd
import pytest

from workbench.tools.ccd.legacy import CCD_COLUMN, LIMIT_COLUMN, LegacyCcdTool as CcdTool
from workbench.tools.ccd.logic import analyze_curve


def _forward(scale=1, linear=False):
    voltage = np.arange(1, 3.6001, .002)
    current = np.maximum(4 * (voltage - 1.6), 0)
    if not linear:
        high = voltage > 2.6
        current[high] = 5 - np.exp(-4 * (voltage[high] - 2.6))
    return voltage, scale * current


@pytest.mark.parametrize("negative", [False, True])
def test_all_cycles_are_fit_independently_without_the_return_scan(tmp_path, negative):
    path = tmp_path / "cv.csv"
    sign = -1 if negative else 1
    voltage, current = _forward()
    scales = [1, 1.4, .8]
    # A return scan with large current and the same polarity must never enter a fit.
    pd.DataFrame({
        "E(V)": np.tile(sign * np.r_[voltage, voltage[-2::-1]], 3),
        "j(A/cm2)": np.concatenate([
            sign * np.r_[current * scale, np.full(len(voltage) - 1, 100)]
            for scale in scales]),
    }).to_csv(path, index=False)
    result = CcdTool().run([path], {})
    rows = result.tables["Current results"]
    assert rows["Cycle"].tolist() == [1, 2, 3]
    assert rows["Curve"].is_unique
    assert result.scalars["Files analyzed"] == 1
    assert result.scalars["Forward scans analyzed"] == 3
    for column in [CCD_COLUMN, LIMIT_COLUMN]:
        assert rows[column].to_numpy() == pytest.approx(rows.iloc[0][column] * np.array(scales))
    assert result.tables["Input curves"]["Current magnitude (A/cm²)"].max() < 8
    statistics = result.tables["Batch statistics"]
    assert statistics["Cycle"].tolist() == [1, 1, 2, 2, 3, 3]
    assert statistics["Resolved files"].tolist() == [1] * 6
    assert statistics["Sample SD (A/cm²)"].isna().all()
    assert len(result.figures["Current statistics"].axes) == 1
    assert len(result.figures["Cycle trends"].axes) == 2


@pytest.mark.parametrize("bad_cycle", ["short", "flat", "invalid"])
def test_one_failed_cycle_does_not_discard_other_cycles(tmp_path, bad_cycle):
    voltage, current = _forward()
    columns = {}
    for cycle, count in [(1, len(voltage)), (2, 200), (3, len(voltage))]:
        v, j = voltage[:count], current[:count]
        if cycle == 2 and bad_cycle == "flat":
            v = np.full(count, 2.0)
        elif cycle == 2 and bad_cycle == "invalid":
            v, j = np.full(count, np.nan), np.full(count, np.nan)
        columns[f"Cycle {cycle} Cell voltage (V)"] = pd.Series(v)
        columns[f"Cycle {cycle} i (A/cm²)"] = pd.Series(j)
    path = tmp_path / "wide.csv"
    pd.DataFrame(columns).to_csv(path, index=False)
    result = CcdTool().run([path], {})
    rows = result.tables["Current results"]
    assert rows["Cycle"].tolist() == [1, 2, 3]
    assert np.isfinite(rows.loc[[0, 2], CCD_COLUMN]).all()
    assert pd.isna(rows.loc[1, CCD_COLUMN])
    assert rows.loc[1, "CCD status"] == "Analysis failed"
    assert result.scalars["Forward scans analyzed"] == 2


def test_all_unresolved_cycles_still_produce_statuses_and_a_trend_plot(tmp_path):
    voltage, current = _forward(linear=True)
    path = tmp_path / "linear_cv.csv"
    pd.DataFrame({f"Cycle {cycle} {label}": values
                  for cycle in [1, 2]
                  for label, values in [("E(V)", voltage), ("j(A/cm2)", current)]}).to_csv(path, index=False)
    result = CcdTool().run([path], {})
    assert result.tables["Current results"][[CCD_COLUMN, LIMIT_COLUMN]].isna().all().all()
    assert result.tables["Batch statistics"]["Resolved files"].eq(0).all()
    assert result.scalars["CCD resolved scans"] == 0
    assert len(result.figures["Current statistics"].axes) == 1
    assert len(result.figures["Cycle trends"].axes) == 2


def test_statistics_compare_files_within_each_cycle_even_with_duplicate_basenames(tmp_path):
    voltage, current = _forward()
    files = []
    for directory, scales in [("a", [1, 2]), ("b", [1.4, 2.8])]:
        folder = tmp_path / directory
        folder.mkdir()
        path = folder / "cv.csv"
        pd.DataFrame({f"Cycle {cycle} {label}": values
                      for cycle, scale in enumerate(scales, 1)
                      for label, values in [("E(V)", voltage), ("j(A/cm2)", scale * current)]
                      }).to_csv(path, index=False)
        files.append(path)
    result = CcdTool().run(files, {})
    rows = result.tables["Current results"]
    assert rows["Curve"].is_unique
    assert len(rows) == 4
    assert result.scalars["Files analyzed"] == 2
    for cycle in [1, 2]:
        for metric, column in [("CCD", CCD_COLUMN), ("Limiting current", LIMIT_COLUMN)]:
            values = rows.loc[rows["Cycle"] == cycle, column]
            stats = result.tables["Batch statistics"]
            stats = stats[(stats["Cycle"] == cycle) & (stats["Metric"] == metric)].iloc[0]
            assert stats["Resolved files"] == 2
            assert stats["Mean (A/cm²)"] == pytest.approx(values.mean())
            assert stats["Sample SD (A/cm²)"] == pytest.approx(values.std(ddof=1))


def test_forward_scans_give_identical_long_and_wide_results(tmp_path):
    scans = [_forward(scale) for scale in (1, 1.4, .8)]
    reference = [analyze_curve(voltage, current) for voltage, current in scans]
    long = pd.concat([
        pd.DataFrame({"Cycle": cycle, "E(V)": voltage, "j(A/cm2)": current})
        for cycle, (voltage, current) in enumerate(scans, start=1)
    ], ignore_index=True)
    wide = pd.DataFrame({
        f"Cycle {cycle} {label}": values
        for cycle, (voltage, current) in enumerate(scans, start=1)
        for label, values in [("E(V)", voltage), ("j(A/cm2)", current)]
    })
    results = []
    for layout, frame in [("long", long), ("wide", wide)]:
        path = tmp_path / f"cycles_{layout}.csv"
        frame.to_csv(path, index=False)
        result = CcdTool().run([path], {})
        results.append(result)
        rows = result.tables["Current results"]
        assert rows["Cycle"].tolist() == [1, 2, 3]
        assert rows["Curve"].tolist() == ["S01-C01", "S01-C02", "S01-C03"]
        assert rows["CCD status"].ne("Analysis failed").all()
        assert rows["CCD status"].tolist() == [curve.ccd_status for curve in reference]
        assert rows["Limiting-current status"].tolist() == [curve.limiting_status for curve in reference]
        for column, attribute in [(CCD_COLUMN, "ccd_current"), (LIMIT_COLUMN, "limiting_current")]:
            np.testing.assert_allclose(
                rows[column].to_numpy(dtype=float),
                np.array([getattr(curve, attribute) for curve in reference], dtype=float), equal_nan=True)
        assert result.scalars["Forward scans analyzed"] == 3
        assert len(CcdTool().exportable_artifacts(result)) == 5
    np.testing.assert_allclose(
        results[0].tables["Current results"][[CCD_COLUMN, LIMIT_COLUMN]],
        results[1].tables["Current results"][[CCD_COLUMN, LIMIT_COLUMN]], equal_nan=True)
