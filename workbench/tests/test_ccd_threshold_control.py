"""An explicit CCD threshold changes onset while retaining the fitted model."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from workbench.tools.ccd.tool import CCD_COLUMN, CcdTool


@pytest.fixture
def curve_file(tmp_path):
    current = np.unique(np.r_[np.geomspace(.001, .1, 301), np.linspace(.1, 6., 4001)])
    residual = .08 * np.maximum(current - .3, 0) + .06 * np.maximum(current - 3., 0) ** 2
    potential = 1.2291 + .06 * np.log10(current / 2e-6) + .12 * current + residual
    path = tmp_path / "known_onset.csv"
    pd.DataFrame({"E(V)": potential, "j(A/cm2)": current}).to_csv(path, index=False)
    return path


def _params(**changes):
    return dict(electrode_setup="Three-electrode", reference="RHE", reaction="OER",
                temperature=298.15, ph=0., hfr=.12, include_rcl=False,
                i_lower=.005, i_upper=.3, target_i=1., ccd_fit_mode="Manual",
                ccd_j_lower=.8, ccd_j_upper=2.5, **changes)


def test_selected_threshold_controls_onset_without_refitting_kinetics_or_baseline(curve_file):
    tool = CcdTool()
    low = tool.run([curve_file], _params(ccd_threshold_mv=20))
    high = tool.run([curve_file], _params(ccd_threshold_mv=100))

    low_onset = low.tables["Current results"].iloc[0][CCD_COLUMN]
    high_onset = high.tables["Current results"].iloc[0][CCD_COLUMN]
    assert low_onset == pytest.approx(3. + np.sqrt(.020 / .06), abs=.04)
    assert high_onset == pytest.approx(3. + np.sqrt(.100 / .06), abs=.04)
    assert high_onset > low_onset
    pd.testing.assert_frame_equal(low.tables["Complete decomposition"], high.tables["Complete decomposition"])
    pd.testing.assert_frame_equal(low.tables["Fit diagnostics"], high.tables["Fit diagnostics"])
    for column in ("Residual baseline (V)", "Excess above baseline (V)", "Baseline fit"):
        pd.testing.assert_series_equal(low.tables["Processed scans"][column],
                                       high.tables["Processed scans"][column])


def test_custom_threshold_is_recorded_in_summary_and_exported_settings(curve_file):
    tool = CcdTool()
    result = tool.run([curve_file], _params(ccd_threshold_mv=75.5))
    assert result.scalars["CCD threshold (mV)"] == 75.5
    assert result.scalars["CCD persistence (V)"] == .10
    assert "sustained 75.5 mV" in result.text_blocks["CCD definition"]
    workbook = next(artifact.payload for artifact in tool.exportable_artifacts(result)
                    if artifact.kind == "xlsx")
    summary = workbook["Summary"]
    settings = summary.loc[summary.Section == "Settings"].set_index("Parameter")["Value"]
    assert settings["ccd_threshold_mv"] == 75.5
    assert "threshold_v" not in settings
    assert settings["persistence_v"] == .10
    selected = result.tables["Processed scans"].loc[lambda frame: frame["CCD point"]]
    assert len(selected) == 1
    assert selected.iloc[0]["Excess above baseline (V)"] >= .0755


@pytest.mark.parametrize("threshold", [0, -1, np.nan, np.inf, -np.inf, "invalid", None, True])
def test_invalid_threshold_is_rejected_before_reading_files(threshold):
    with pytest.raises(ValueError, match="CCD threshold must be a finite value greater than zero mV"):
        CcdTool().run([Path("does_not_exist.csv")], {"ccd_threshold_mv": threshold})
