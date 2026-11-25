"""The historical sensitivity study excludes later scans and preserves unresolved groups."""
import numpy as np
import pandas as pd

from workbench.tools.ccd.sensitivity import CCD, MEAN, calculate


def _study_source(path, *, plateau):
    voltage = np.arange(1., 3.6001, .002)
    current = np.maximum(4 * (voltage - 1.6), 0)
    if plateau:
        high = voltage > 2.6
        current[high] = 5 - np.exp(-4 * (voltage[high] - 2.6))
    pd.DataFrame({
        f"Cycle {cycle} {label}": values
        for cycle, scale in enumerate((1., 1.2, .9, 1.1), 1)
        for label, values in (("E(V)", voltage), ("j(A/cm2)", scale * current))
    }).to_csv(path, index=False)
    return path


def test_training_study_uses_only_first_three_and_does_not_replace_missing_ccd(tmp_path):
    first = _study_source(tmp_path / "CV1.txt", plateau=True)
    second = _study_source(tmp_path / "CV3.txt", plateau=False)
    tables, _results, selected, _settings = calculate([first, second])
    assert selected[0][2]["scan_count"] == 4
    rows = tables["Per scan"]
    assert len(rows) == 2 * 3 * 2 * 3
    assert set(rows["Cycle"]) == {1, 2, 3}
    assert set(tables["Input forward scans"]["Cycle"]) == {1, 2, 3}
    assert set(tables["Benchmark lines"]["Cycle"]) == {1, 2, 3}
    absent = rows.loc[rows["Fit mode"].eq("Auto") & rows["File"].eq("CV3.txt")
                      & rows["Threshold (mV)"].eq(100)]
    assert absent[CCD].isna().all()
    assert absent["CCD status"].eq("Not reached within measured range").all()
    resolved = rows.loc[rows["Fit mode"].eq("Auto") & rows["File"].eq("CV1.txt")
                       & rows["Threshold (mV)"].eq(20)]
    assert resolved[CCD].notna().all() and resolved["CCD status"].eq("Detected").all()
    summary = tables["Threshold summary"]
    absent_summary = summary.loc[summary["Fit mode"].eq("Auto") & summary["File"].eq("CV3.txt")
                                & summary["Threshold (mV)"].eq(100)].iloc[0]
    assert absent_summary["Resolved scans"] == 0
    assert np.isnan(absent_summary[MEAN])
    assert set(tables["Window sensitivity"]["Anchor mode"]) == {"Manual", "Auto"}
    assert len(tables["Window sensitivity"]) == len(rows) * 9
