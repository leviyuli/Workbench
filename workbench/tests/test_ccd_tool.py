"""Batch CCD results keep file identities, physical units and honest uncertainty."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from workbench.tools.ccd.legacy import CCD_COLUMN, LIMIT_COLUMN, LegacyCcdTool as CcdTool


def _write_curve(path: Path, *, scale=1.0, end=3.6, negative=False,
                 absolute_area=None, linear=False):
    voltage = np.arange(1.0, end + .0001, .0005)
    current = np.maximum(4 * (voltage - 1.6), 0)
    if not linear:
        high = voltage > 2.6
        current[high] = 5 - np.exp(-4 * (voltage[high] - 2.6))
    current *= scale
    if negative:
        voltage, current = -voltage, -current
    header = "j(A/cm2)"
    if absolute_area is not None:
        current *= absolute_area
        header = "I(A)"
    pd.DataFrame({"E(V)": voltage, header: current}).to_csv(path, index=False)
    return voltage, current


def _statistics(result, metric):
    return result.tables["Batch statistics"].set_index("Metric").loc[metric]


def test_batch_fits_each_file_and_uses_sample_sd_for_individual_currents(tmp_path):
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    _write_curve(first)
    _write_curve(second, scale=1.4)
    result = CcdTool().run([first, second], {})
    rows = result.tables["Current results"]

    assert rows["File"].tolist() == [first.name, second.name]
    assert rows["Curve"].tolist() == ["S01", "S02"]
    for metric, column in [("CCD", CCD_COLUMN), ("Limiting current", LIMIT_COLUMN)]:
        values = rows[column].to_numpy(dtype=float)
        assert np.isfinite(values).all()
        assert values[1] == pytest.approx(1.4 * values[0])
        statistics = _statistics(result, metric)
        assert statistics["Resolved files"] == 2
        assert statistics["Mean (A/cm²)"] == pytest.approx(np.mean(values))
        assert statistics["Sample SD (A/cm²)"] == pytest.approx(np.std(values, ddof=1))

    bars = result.figures["Current statistics"].axes[0].patches
    assert [bar.get_height() for bar in bars] == pytest.approx([
        _statistics(result, "CCD")["Mean (A/cm²)"],
        _statistics(result, "Limiting current")["Mean (A/cm²)"],
    ])


def test_unresolved_and_unreadable_files_remain_rows_without_zero_imputation(tmp_path):
    valid, line, invalid = (tmp_path / name for name in ["plateau.csv", "line.csv", "bad.csv"])
    _write_curve(valid)
    _write_curve(line, linear=True)
    invalid.write_text("not a numeric LSV\n", encoding="utf-8")
    result = CcdTool().run([valid, line, invalid], {})
    rows = result.tables["Current results"].set_index("File")

    assert len(rows) == 3
    assert result.scalars["Files selected"] == 3
    assert result.scalars["Files analyzed"] == 2
    assert "Analysis failed" in rows.loc[invalid.name, "CCD status"]
    for metric, column in [("CCD", CCD_COLUMN), ("Limiting current", LIMIT_COLUMN)]:
        assert pd.isna(rows.loc[line.name, column])
        assert pd.isna(rows.loc[invalid.name, column])
        statistics = _statistics(result, metric)
        assert statistics["Resolved files"] == 1
        assert statistics["Mean (A/cm²)"] == pytest.approx(rows.loc[valid.name, column])
        assert pd.isna(statistics["Sample SD (A/cm²)"])
    assert any("bad.csv" in warning for warning in result.warnings)


def test_limiting_current_statistics_exclude_an_unfinished_plateau_independently(tmp_path):
    full, truncated = tmp_path / "full.csv", tmp_path / "unfinished.csv"
    _write_curve(full)
    _write_curve(truncated, end=3.0, scale=1.2)
    result = CcdTool().run([full, truncated], {})
    rows = result.tables["Current results"].set_index("File")

    assert np.isfinite(rows.loc[truncated.name, CCD_COLUMN])
    assert pd.isna(rows.loc[truncated.name, LIMIT_COLUMN])
    assert _statistics(result, "CCD")["Resolved files"] == 2
    limiting = _statistics(result, "Limiting current")
    assert limiting["Resolved files"] == 1
    assert limiting["Mean (A/cm²)"] == pytest.approx(rows.loc[full.name, LIMIT_COLUMN])
    assert pd.isna(limiting["Sample SD (A/cm²)"])


def test_absolute_current_is_normalized_once_and_density_is_not_normalized_again(tmp_path):
    density, absolute = tmp_path / "density.csv", tmp_path / "absolute.csv"
    _write_curve(density)
    _write_curve(absolute, absolute_area=.25)
    result = CcdTool().run([density, absolute], {"electrode_area": .25})
    rows = result.tables["Current results"]

    for column in [CCD_COLUMN, LIMIT_COLUMN]:
        assert rows.loc[0, column] == pytest.approx(rows.loc[1, column])
    exported = result.tables["Processed curves"]
    np.testing.assert_allclose(
        exported[exported.Curve == "S01"]["Median current (A/cm²)"],
        exported[exported.Curve == "S02"]["Median current (A/cm²)"],
    )


def test_her_preserves_measured_potential_in_plots_and_exports_with_positive_currents(tmp_path):
    oer, her = tmp_path / "positive.csv", tmp_path / "negative.csv"
    _write_curve(oer)
    _write_curve(her, negative=True)
    tool = CcdTool()
    reference = tool.run([oer], {})
    result = tool.run([her], {})

    for column in [CCD_COLUMN, LIMIT_COLUMN]:
        assert result.tables["Current results"].loc[0, column] == pytest.approx(
            reference.tables["Current results"].loc[0, column])
    processed = result.tables["Processed curves"]
    assert (processed["Measured potential (V)"] < 0).all()
    assert (result.tables["Input curves"]["Current magnitude (A/cm²)"] >= 0).all()
    np.testing.assert_allclose(
        processed["HFR-corrected potential (V)"],
        -reference.tables["Processed curves"]["HFR-corrected potential (V)"],
    )
    np.testing.assert_allclose(
        result.tables["Input curves"]["HFR-corrected potential (V)"],
        -reference.tables["Input curves"]["HFR-corrected potential (V)"],
    )
    # Cathodic iR correction moves the native negative potential toward zero.
    active = processed["Smoothed current (A/cm²)"] > 0
    assert (processed.loc[active, "HFR-corrected potential (V)"]
            > processed.loc[active, "Measured potential (V)"]).all()
    np.testing.assert_allclose(
        processed["Residual increase (mV)"],
        -1000 * (processed["HFR-corrected potential (V)"]
                 - processed["Corrected baseline potential (V)"]),
        atol=1e-10,
    )
    assert all(np.max(line.get_xdata()) < 0
               for line in result.figures["LSV processing"].axes[0].lines)


def test_exported_ccd_and_processed_data_match_the_visible_curve_and_summary(tmp_path):
    source = tmp_path / "curve.csv"
    _write_curve(source)
    tool = CcdTool()
    result = tool.run([source], {})
    processed = result.tables["Processed curves"]
    ccd = processed[processed["CCD point"]]

    assert len(ccd) == 1
    assert ccd.iloc[0]["Smoothed current (A/cm²)"] == pytest.approx(
        result.tables["Current results"].loc[0, CCD_COLUMN])
    assert ccd.iloc[0]["Residual increase (mV)"] >= 20
    lines = result.figures["LSV processing"].axes[0].lines
    assert any(len(line.get_xdata()) == len(processed)
               and np.allclose(line.get_xdata(), processed["Measured potential (V)"])
               and np.allclose(line.get_ydata(), processed["Smoothed current (A/cm²)"])
               for line in lines)

    artifacts = tool.exportable_artifacts(result)
    workbook = next(item for item in artifacts if item.kind == "xlsx")
    pd.testing.assert_frame_equal(workbook.payload["Current results"], result.tables["Current results"])
    assert not any("potential" in column.lower() or "voltage" in column.lower()
                   for column in workbook.payload["Current results"])
    assert len([artifact for artifact in artifacts if artifact.kind == "png"]) == 3


def test_reference_hfr_is_explicit_in_inputs_settings_and_results(tmp_path):
    source = tmp_path / "curve.csv"
    _write_curve(source)
    tool = CcdTool()

    schema = {spec.key: spec for spec in tool.parameter_schema()}
    assert schema["reference_hfr"].default == .12
    result = tool.run([source], {})
    settings = result.tables["Analysis settings"].set_index("Parameter")["Value"]
    assert settings["reference_hfr"] == .12
    assert result.scalars["Reference HFR (Ω·cm²)"] == .12


def test_constant_hfr_changes_corrected_potentials_without_double_subtracting_residual(tmp_path):
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    _write_curve(first)
    _write_curve(second, scale=.8)
    tool = CcdTool()
    low = tool.run([first, second], {"reference_hfr": .12})
    high = tool.run([first, second], {"reference_hfr": .20})

    # Both the corrected LSV and its fitted reference lose the same j·ΔHFR.
    # Changing this reference alone must not manufacture a different CCD or SD.
    for column in [CCD_COLUMN, LIMIT_COLUMN]:
        np.testing.assert_allclose(low.tables["Current results"][column],
                                   high.tables["Current results"][column], rtol=1e-12)
    pd.testing.assert_frame_equal(low.tables["Batch statistics"], high.tables["Batch statistics"])
    for table, current_column in [("Input curves", "Current magnitude (A/cm²)"),
                                   ("Processed curves", "Smoothed current (A/cm²)")]:
        before, after = low.tables[table], high.tables[table]
        np.testing.assert_allclose(
            before["HFR-corrected potential (V)"] - after["HFR-corrected potential (V)"],
            .08 * before[current_column], atol=1e-12,
        )
    processed = low.tables["Processed curves"]
    np.testing.assert_allclose(processed["HFR drop (V)"],
                               .12 * processed["Smoothed current (A/cm²)"], atol=1e-12)
    np.testing.assert_allclose(
        processed["Residual increase (mV)"],
        1000 * (processed["HFR-corrected potential (V)"]
                - processed["Corrected baseline potential (V)"]),
        atol=1e-10,
    )
    np.testing.assert_allclose(processed["Residual increase (mV)"],
                               high.tables["Processed curves"]["Residual increase (mV)"],
                               atol=1e-10)


def test_residual_diagnostic_matches_exported_increase_and_detected_ccd(tmp_path):
    source = tmp_path / "curve.csv"
    _write_curve(source)
    result = CcdTool().run([source], {"reference_hfr": .12})
    processed = result.tables["Processed curves"]
    # The onset diagnostic starts with the fitted middle region, omitting the
    # low-current kinetic segment while exports retain the complete curve.
    visible = processed[processed["Measured potential (V)"] >= 1.9]
    lines = result.figures["Residual onset"].axes[0].lines

    assert any(len(line.get_xdata()) == len(visible)
               and np.allclose(line.get_xdata(), visible["Smoothed current (A/cm²)"])
               and np.allclose(line.get_ydata(), visible["Residual increase (mV)"])
               for line in lines)
    ccd = processed[processed["CCD point"]].iloc[0]
    assert any(len(line.get_xdata()) == 1
               and line.get_xdata()[0] == pytest.approx(ccd["Smoothed current (A/cm²)"])
               and line.get_ydata()[0] == pytest.approx(ccd["Residual increase (mV)"])
               for line in lines)
    assert any(np.allclose(line.get_ydata(), [20, 20])
               for line in lines if len(line.get_ydata()) == 2)


@pytest.mark.parametrize("hfr", [-.01, np.nan, np.inf])
def test_reference_hfr_rejects_nonphysical_values(tmp_path, hfr):
    source = tmp_path / "curve.csv"
    _write_curve(source)

    with pytest.raises(ValueError, match="HFR.*(?:nonnegative|finite)"):
        CcdTool().run([source], {"reference_hfr": hfr})


@pytest.mark.parametrize("negative", [False, True])
def test_benchmark_export_matches_dashed_plot_and_per_scan_fit_equations(tmp_path, negative):
    source = tmp_path / "curve.csv"
    _write_curve(source, negative=negative)
    hfr = .12
    result = CcdTool().run([source], {"reference_hfr": hfr})
    workbook = next(item.payload for item in CcdTool().exportable_artifacts(result)
                    if item.kind == "xlsx")
    benchmark = workbook["Benchmark lines"]
    fits = workbook["Baseline fits"].iloc[0]
    dashed = [line for line in result.figures["LSV processing"].axes[0].lines
              if line.get_linestyle() == "--"]

    assert len(dashed) == 1
    np.testing.assert_allclose(benchmark["Potential (V)"], dashed[0].get_xdata())
    np.testing.assert_allclose(benchmark["Benchmark current density (A/cm²)"], dashed[0].get_ydata())
    np.testing.assert_allclose(
        benchmark["Potential (V)"],
        fits["Measured intercept (V)"]
        + fits["Measured dE/dj (Ω·cm²)"] * benchmark["Benchmark current density (A/cm²)"],
    )
    np.testing.assert_allclose(
        benchmark["iR-corrected benchmark potential (V)"],
        fits["iR-corrected intercept (V)"]
        + fits["iR-corrected dE/dj (Ω·cm²)"] * benchmark["Benchmark current density (A/cm²)"],
    )
    sign = -1 if negative else 1
    np.testing.assert_allclose(
        benchmark["iR-corrected benchmark potential (V)"],
        benchmark["Potential (V)"] - sign * hfr * benchmark["Benchmark current density (A/cm²)"],
    )
    assert fits["Fit start potential (V)"] == pytest.approx(sign * 1.9)
    assert fits["Fit end potential (V)"] == pytest.approx(sign * 2.45)
    assert fits["Fit R²"] == pytest.approx(1)
    assert fits["Fit points"] > 100
    assert benchmark["Baseline region"].any()
    assert not benchmark["Baseline region"].all()
    pd.testing.assert_series_equal(
        workbook["Forward scans"]["iR-corrected baseline at scan current (V)"],
        result.tables["Processed curves"]["Corrected baseline potential (V)"],
        check_names=False,
    )


def test_threshold_changes_ccd_and_remains_consistent_in_exports_and_plot_labels(tmp_path):
    source = tmp_path / "curve.csv"
    _write_curve(source)
    results = [CcdTool().run([source], {"ccd_threshold_mv": value}) for value in [20, 50, 100]]
    ccd = [result.tables["Current results"].loc[0, CCD_COLUMN] for result in results]
    assert ccd[0] < ccd[1] < ccd[2]
    for threshold, result in zip([20, 50, 100], results):
        assert result.scalars["CCD threshold (mV)"] == threshold
        settings = result.tables["Analysis settings"].set_index("Parameter")["Value"]
        assert settings["threshold_v"] == pytest.approx(threshold / 1000)
        processed = result.tables["Processed curves"]
        assert processed.loc[processed["CCD point"], "Residual increase (mV)"].iloc[0] >= threshold
        assert f"{threshold} mV" in result.text_blocks["CCD definition"]
        style_legend = result.figures["LSV processing"].axes[0].artists[0]
        assert f"CCD: {threshold} mV" in [label.get_text() for label in style_legend.get_texts()]
        residual_lines = result.figures["Residual onset"].axes[0].lines
        residual_legend = result.figures["Residual onset"].axes[0].artists[0]
        assert f"{threshold} mV threshold" in [label.get_text() for label in residual_legend.get_texts()]
        assert any(np.allclose(line.get_ydata(), [threshold, threshold])
                   for line in residual_lines if len(line.get_ydata()) == 2)


@pytest.mark.parametrize("threshold", [0, -20, np.nan, np.inf])
def test_invalid_ccd_threshold_cannot_produce_results(tmp_path, threshold):
    source = tmp_path / "curve.csv"
    _write_curve(source)
    with pytest.raises(ValueError, match="CCD threshold"):
        CcdTool().run([source], {"ccd_threshold_mv": threshold})
