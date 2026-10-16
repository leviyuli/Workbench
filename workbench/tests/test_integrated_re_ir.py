"""Integrated RE keeps raw curves and applies independent role-mean HFRs."""

import numpy as np
import pandas as pd
import pytest

from workbench.tools.integrated_re.tool import IntegratedReTool
from workbench.widgets.export_button import ExportButton


CURRENT = "Current density (A/cm²)"
VOLTAGE = "Reconstructed cell voltage (V)"
HFR = "HFR (Ω·cm²)"


def params(**updates):
    values = {spec.key: spec.default
              for spec in IntegratedReTool().parameter_schema()}
    # Synthetic calibration curves in these tests are already versus RHE.
    values.update(calibration_reference="RHE", calibration_ph=0.0,
                  calibration_temperature=298.15)
    values.update(updates)
    return values


def write_curve(path, potential, current, *, current_header="j(A/cm2)"):
    pd.DataFrame({"E(V)": potential, current_header: current}).to_csv(
        path, index=False,
    )
    return str(path)


def write_spectrum(path, rows, *, unit="Ohm.cm2"):
    """Rows hold frequency, real impedance, and signed imaginary impedance."""
    path.write_text(
        f"Frequency (Hz),Re(Z) ({unit}),-Im(Z) ({unit})\n"
        + "".join(f"{f},{real},{-imag}\n" for f, real, imag in rows),
        encoding="utf-8",
    )
    return str(path)


def hfr_file(tmp_path, name, resistance, *, unit="Ohm.cm2"):
    return write_spectrum(tmp_path / name, [
        (10000, resistance - 0.05, 0.1),
        (1000, resistance + 0.05, -0.1),
        (100, resistance + 0.3, -0.5),
    ], unit=unit)


def inputs(tmp_path, *, cell=True, common_offset=0.0):
    ja = np.array([0.0, 0.1, 0.3, 0.5])
    jc = np.array([0.0, 0.15, 0.2, 0.35, 0.5])
    sources = {
        "anode": write_curve(tmp_path / "anode.csv",
                             1.4 + 0.8 * ja + common_offset, ja),
        "cathode": write_curve(tmp_path / "cathode.csv",
                               -0.1 - 0.4 * jc + common_offset, -jc),
        "eis": {
            "anode": [hfr_file(tmp_path, "anode_eis.csv", 0.2)],
            "cathode": [hfr_file(tmp_path, "cathode_eis.csv", 0.3)],
            "cell": [],
        },
    }
    if cell:
        jcell = np.array([0.05, 0.2, 0.4, 0.6])
        sources["cell"] = write_curve(tmp_path / "cell.csv",
                                      1.51 + 1.3 * jcell, jcell)
        sources["eis"]["cell"] = [hfr_file(tmp_path, "cell_eis.csv", 0.7)]
    return sources


def role_row(table, role):
    rows = table.loc[table["Role"].astype(str).str.lower().str.startswith(role)]
    assert len(rows) == 1
    return rows.iloc[0]


def curve_line(figure, role, *, corrected=False):
    candidates = [line for line in figure.axes[0].lines
                  if role.lower() in line.get_label().lower()
                  and line.get_linestyle() == ("--" if corrected else "-")]
    assert len(candidates) == 1
    return candidates[0]


def test_ir_is_optional_and_normal_review_needs_no_eis(tmp_path):
    sources = inputs(tmp_path, cell=False)
    del sources["eis"]
    schema = {spec.key: spec for spec in IntegratedReTool().parameter_schema()}
    assert schema["include_ir"].default is False
    result = IntegratedReTool().run([], params(), extra=sources)
    assert "Reconstructed cell (iR corrected)" not in result.tables
    assert result.display_tables is False
    assert result.display_summary is False
    assert len([line for line in result.figures["Electrode LSVs"].axes[0].lines
                if "anode.csv" in line.get_label()]) == 1


def test_role_means_average_intercepts_without_averaging_spectra(tmp_path):
    sources = inputs(tmp_path, cell=False)
    # The separate HFRs are 0.3 and 0.8. Averaging the spectra instead
    # would give 17/30, which must not be used for iR correction.
    sources["eis"]["anode"] = [
        write_spectrum(tmp_path / "first.csv", [
            (10000, 0.1, 0.1), (1000, 0.5, -0.1), (100, 0.9, -0.6),
        ]),
        write_spectrum(tmp_path / "second.csv", [
            (10000, 0.5, 0.3), (1000, 0.9, -0.1), (100, 1.3, -0.6),
        ]),
    ]
    result = IntegratedReTool().run([], params(include_ir=True), extra=sources)
    row = role_row(result.tables["HFR summary"], "anode")
    assert row["Mean HFR (Ω·cm²)"] == pytest.approx(0.55)
    assert row["Spectra used"] == row["Spectra uploaded"] == 2
    source = result.tables["Anode LSV"]
    np.testing.assert_allclose(source[HFR], 0.55)
    np.testing.assert_allclose(source["iR drop (V)"], source[CURRENT] * 0.55)
    assert float(result.param_updates["anode_hfr"]) == pytest.approx(0.55)


def test_simple_hfr_does_not_invoke_kk_or_equivalent_circuit_fitting(tmp_path, monkeypatch):
    import workbench.tools.eis.logic_eis as eis_logic

    def forbidden(*args, **kwargs):
        pytest.fail("Integrated RE HFR extraction must not run KK or fitting.")

    monkeypatch.setattr(eis_logic.pyimpspec, "perform_kramers_kronig_test", forbidden)
    monkeypatch.setattr(eis_logic.EisLogic, "fit_impedance", forbidden)
    monkeypatch.setattr(eis_logic.EisLogic, "_average_points", forbidden)
    result = IntegratedReTool().run([], params(include_ir=True),
                                    extra=inputs(tmp_path, cell=False))
    assert not result.tables["HFR summary"].empty


@pytest.mark.parametrize("axis", ["E vs RHE", "Overpotential"])
def test_signed_electrode_correction_and_overlay_in_both_axis_modes(tmp_path, axis):
    sources = inputs(tmp_path, cell=False)
    result = IntegratedReTool().run([], params(include_ir=True, electrode_axis=axis),
                                    extra=sources)
    for role, resistance in [("Anode", 0.2), ("Cathode", 0.3)]:
        table = result.tables[f"{role} LSV"]
        shift = 1.2291 if axis == "Overpotential" and role == "Anode" else 0.0
        raw = curve_line(result.figures["Electrode LSVs"], role)
        corrected = curve_line(result.figures["Electrode LSVs"], role, corrected=True)
        np.testing.assert_allclose(raw.get_xdata(), table["E (V vs RHE)"] - shift)
        np.testing.assert_allclose(corrected.get_xdata(),
                                   table["E (V vs RHE)"] - table[CURRENT] * resistance - shift)
        np.testing.assert_allclose(corrected.get_ydata(), raw.get_ydata())
        np.testing.assert_allclose(table["iR-corrected E (V vs RHE)"],
                                   table["E (V vs RHE)"] - table[CURRENT] * resistance)
        np.testing.assert_allclose(table["iR-corrected overpotential (V)"],
                                   table["Overpotential (V)"] - table[CURRENT] * resistance)
    cathode = result.tables["Cathode LSV"]
    reaction = cathode[CURRENT] < 0
    assert np.all(cathode.loc[reaction, "iR-corrected E (V vs RHE)"]
                  > cathode.loc[reaction, "E (V vs RHE)"])


def test_cell_and_electrode_hfrs_are_used_independently(tmp_path):
    result = IntegratedReTool().run([], params(include_ir=True), extra=inputs(tmp_path))
    raw = result.tables["Reconstructed cell"]
    corrected = result.tables["Reconstructed cell (iR corrected)"]
    np.testing.assert_allclose(corrected[CURRENT], raw[CURRENT])
    np.testing.assert_allclose(corrected["Anode E (V vs RHE)"],
                               raw["Anode E (V vs RHE)"] - raw[CURRENT] * 0.2)
    np.testing.assert_allclose(corrected["Cathode E (V vs RHE)"],
                               raw["Cathode E (V vs RHE)"] + raw[CURRENT] * 0.3)
    np.testing.assert_allclose(corrected[VOLTAGE], raw[VOLTAGE] - raw[CURRENT] * 0.5)
    np.testing.assert_allclose(corrected["Cell overpotential (V)"],
                               raw["Cell overpotential (V)"] - raw[CURRENT] * 0.5)
    original_comparison = result.tables["Cell comparison"]
    comparison = result.tables["Cell comparison (iR corrected)"]
    np.testing.assert_allclose(comparison[CURRENT], original_comparison[CURRENT])
    np.testing.assert_allclose(comparison["Measured cell voltage (V)"],
                               original_comparison["Measured cell voltage (V)"]
                               - comparison[CURRENT] * 0.7)
    np.testing.assert_allclose(comparison["Voltage difference (V)"],
                               original_comparison["Voltage difference (V)"]
                               + comparison[CURRENT] * 0.2, atol=1e-14)
    cell = result.tables["Cell LSV"]
    np.testing.assert_allclose(cell["iR-corrected cell voltage (V)"],
                               cell["Cell voltage (V)"] - cell[CURRENT] * 0.7)
    figure = result.figures["Cell comparison"]
    for role, table, column in [("Reconstructed", corrected, VOLTAGE),
                                ("Measured", cell, "iR-corrected cell voltage (V)")]:
        raw_line = curve_line(figure, role)
        corrected_line = curve_line(figure, role, corrected=True)
        np.testing.assert_allclose(corrected_line.get_xdata(), table[column])
        np.testing.assert_allclose(corrected_line.get_ydata(), raw_line.get_ydata())


def test_source_columns_and_uncorrected_results_remain_identical(tmp_path):
    sources = inputs(tmp_path)
    tool = IntegratedReTool()
    original = tool.run([], params(), extra=sources)
    corrected = tool.run([], params(include_ir=True), extra=sources)
    for name in ("Anode LSV", "Cathode LSV", "Cell LSV", "Reconstructed cell", "Cell comparison"):
        before = original.tables[name]
        pd.testing.assert_frame_equal(before, corrected.tables[name][before.columns])


def test_ir_keeps_zero_and_minority_sign_currents_in_measured_sweeps(tmp_path):
    sources = inputs(tmp_path, cell=False)
    ja = np.array([0.05, 0.0, -0.01, 0.3])
    jc = np.array([-0.05, 0.0, 0.01, -0.3])
    sources["anode"] = write_curve(tmp_path / "anode.csv", [1.5, 1.4, 1.35, 1.8], ja)
    sources["cathode"] = write_curve(tmp_path / "cathode.csv", [-0.1, 0.0, 0.05, -0.3], jc)
    result = IntegratedReTool().run([], params(include_ir=True), extra=sources)
    for role, current, resistance in [("Anode", ja, 0.2), ("Cathode", jc, 0.3)]:
        source = result.tables[f"{role} LSV"]
        np.testing.assert_array_equal(source[CURRENT], current)
        np.testing.assert_allclose(source["iR-corrected E (V vs RHE)"],
                                   source["E (V vs RHE)"] - current * resistance)
        corrected = curve_line(result.figures["Electrode LSVs"], role, corrected=True)
        np.testing.assert_array_equal(corrected.get_ydata(), current)


def test_shared_manual_shift_changes_neither_raw_nor_corrected_cell_voltage(tmp_path):
    sources = inputs(tmp_path)
    tool = IntegratedReTool()
    before = tool.run([], params(include_ir=True), extra=sources)
    after = tool.run([], params(include_ir=True, correction_mode="Manual", manual_shift=-0.2),
                     extra=sources)
    for name in ("Reconstructed cell", "Reconstructed cell (iR corrected)",
                 "Cell comparison", "Cell comparison (iR corrected)"):
        np.testing.assert_allclose(after.tables[name][VOLTAGE], before.tables[name][VOLTAGE])
    for role in ("Anode", "Cathode"):
        np.testing.assert_allclose(after.tables[f"{role} LSV"]["iR-corrected E (V vs RHE)"],
                                   before.tables[f"{role} LSV"]["iR-corrected E (V vs RHE)"] - 0.2)


def test_three_electrode_alignment_uses_signed_cathode_ir_and_never_accumulates(tmp_path):
    sources = inputs(tmp_path, cell=False, common_offset=0.2)
    j = np.array([0.0, 0.01, 0.1, 0.3, 0.5])
    sources["calibration"] = write_curve(tmp_path / "calibration.csv", -0.1 - 0.4 * j, -j)
    tool = IntegratedReTool()
    options = params(correction_mode="3E onset", manual_shift=0.03)
    raw = tool.run([], options, extra=sources)
    corrected = tool.run([], {**options, "include_ir": True}, extra=sources)
    again = tool.run([], {**options, "include_ir": True, **corrected.param_updates}, extra=sources)
    assert float(corrected.param_updates["calculated_shift"]) == pytest.approx(-0.215)
    assert float(corrected.param_updates["applied_shift"]) == pytest.approx(-0.185)
    feature = corrected.tables["Calibration features"].set_index("Source").loc["Integrated"]
    assert feature["E before correction (V vs RHE)"] == pytest.approx(0.095)
    assert feature["E after correction (V vs RHE)"] == pytest.approx(-0.09)
    pd.testing.assert_frame_equal(raw.tables["Calibration features"],
                                  corrected.tables["Calibration features"])
    for name in ("Anode LSV", "Cathode LSV", "Reconstructed cell (iR corrected)"):
        pd.testing.assert_frame_equal(corrected.tables[name], again.tables[name])
    sources["eis"]["cathode"] = [hfr_file(tmp_path, "higher_cathode_hfr.csv", 0.6)]
    higher_hfr = tool.run([], options, extra=sources)
    assert float(higher_hfr.param_updates["calculated_shift"]) == pytest.approx(-0.23)
    assert float(higher_hfr.param_updates["applied_shift"]) == pytest.approx(-0.2)
    higher_feature = higher_hfr.tables["Calibration features"].set_index("Source").loc["Integrated"]
    assert higher_feature["E before correction (V vs RHE)"] == pytest.approx(0.11)
    assert higher_feature["E after correction (V vs RHE)"] == pytest.approx(-0.09)


def test_ir_preserves_unshifted_masks_and_extreme_crossing_branch(tmp_path):
    sources = inputs(tmp_path, cell=False)
    sources["anode"] = write_curve(tmp_path / "anode.csv",
                                   [1.4, 1.7, 1.45, 1.8, 2.0], [0.05, 0.4, 0.05, 0.4, 0.6])
    sources["cathode"] = write_curve(tmp_path / "cathode.csv",
                                     [-0.1, -0.4, -0.15, -0.5, -0.8], [-0.05, -0.4, -0.05, -0.4, -0.6])
    options = params(anode_use_window=True, anode_e_min=1.5, anode_e_max=2.0,
                     cathode_use_window=True, cathode_e_min=-0.8, cathode_e_max=-0.2,
                     correction_mode="Manual", manual_shift=-0.2)
    tool = IntegratedReTool()
    raw = tool.run([], options, extra=sources)
    corrected = tool.run([], {**options, "include_ir": True}, extra=sources)
    for role in ("Anode", "Cathode"):
        np.testing.assert_array_equal(corrected.tables[f"{role} LSV"]["In comparison window"],
                                      [False, True, False, True, True])
    uncorrected = raw.tables["Reconstructed cell"]
    ir = corrected.tables["Reconstructed cell (iR corrected)"]
    np.testing.assert_allclose(ir[CURRENT], uncorrected[CURRENT])
    np.testing.assert_allclose(ir[VOLTAGE], uncorrected[VOLTAGE] - ir[CURRENT] * 0.5)
    assert uncorrected.iloc[0]["Anode E (V vs RHE)"] == pytest.approx(1.6)
    assert uncorrected.iloc[0]["Cathode E (V vs RHE)"] == pytest.approx(-0.7)


def test_hfr_is_extracted_even_when_ir_is_disabled(tmp_path):
    sources = inputs(tmp_path, cell=False)
    result = IntegratedReTool().run([], params(), extra=sources)
    assert len(result.tables["HFR per spectrum"]) == 2
    assert role_row(result.tables["HFR summary"], "anode")["Mean HFR (Ω·cm²)"] == pytest.approx(0.2)
    assert float(result.param_updates["cathode_hfr"]) == pytest.approx(0.3)
    assert "Reconstructed cell (iR corrected)" not in result.tables


def test_simple_hfr_records_minimum_real_fallback_when_no_crossing_exists(tmp_path):
    sources = inputs(tmp_path, cell=False)
    sources["eis"]["anode"] = [write_spectrum(tmp_path / "fallback.csv", [
        (100, 0.8, -0.6), (10000, 0.5, -0.3), (1000, 0.2, -0.1),
    ])]
    result = IntegratedReTool().run([], params(include_ir=True), extra=sources)
    assert role_row(result.tables["HFR summary"], "anode")["Mean HFR (Ω·cm²)"] == pytest.approx(0.2)
    details = " ".join(map(str, result.tables["HFR per spectrum"].to_numpy().ravel())).lower()
    assert "fallback" in details or "no x-axis intersection" in details


@pytest.mark.parametrize("role", ["anode", "cathode", "cell"])
def test_enabling_ir_requires_valid_hfr_for_every_plotted_role(tmp_path, role):
    sources = inputs(tmp_path)
    sources["eis"][role] = []
    with pytest.raises(ValueError, match=f"(?i){role}"):
        IntegratedReTool().run([], params(include_ir=True), extra=sources)


def test_disabled_ir_allows_missing_eis_roles_and_retains_uploaded_details(tmp_path):
    sources = inputs(tmp_path)
    sources["eis"]["anode"] = []
    sources["eis"]["cell"] = []
    result = IntegratedReTool().run([], params(), extra=sources)
    assert len(result.tables["HFR per spectrum"]) == 1
    assert "Cell comparison" in result.tables
    assert "Cell comparison (iR corrected)" not in result.tables


def test_cell_eis_without_cell_lsv_is_still_exported(tmp_path):
    sources = inputs(tmp_path, cell=False)
    sources["eis"]["cell"] = [hfr_file(tmp_path, "optional_cell.csv", 0.8)]
    result = IntegratedReTool().run([], params(include_ir=True), extra=sources)
    assert role_row(result.tables["HFR summary"], "cell")["Mean HFR (Ω·cm²)"] == pytest.approx(0.8)
    assert "Cell LSV" not in result.tables
    assert "Cell comparison (iR corrected)" not in result.tables
    assert "Reconstructed cell (iR corrected)" in result.tables


def test_failed_spectrum_is_recorded_and_role_mean_uses_valid_spectra(tmp_path):
    sources = inputs(tmp_path, cell=False)
    sources["eis"]["anode"].append(str(tmp_path / "missing.csv"))
    sources["eis"]["anode"].append(hfr_file(tmp_path, "second_anode.csv", 0.4))
    result = IntegratedReTool().run([], params(include_ir=True), extra=sources)
    row = role_row(result.tables["HFR summary"], "anode")
    assert row["Mean HFR (Ω·cm²)"] == pytest.approx(0.3)
    assert row["Spectra used"] == 2
    assert row["Spectra uploaded"] == 3
    per_spectrum = result.tables["HFR per spectrum"]
    assert len(per_spectrum) == 4
    assert "missing.csv" in " ".join(map(str, per_spectrum.to_numpy().ravel()))
    assert per_spectrum[HFR].isna().sum() == 1
    assert any("missing.csv" in notice for notice in result.warnings)


def test_all_failed_eis_for_a_required_role_reports_that_role(tmp_path):
    sources = inputs(tmp_path, cell=False)
    sources["eis"]["cathode"] = [str(tmp_path / "missing.csv")]
    with pytest.raises(ValueError, match="(?i)cathode"):
        IntegratedReTool().run([], params(include_ir=True), extra=sources)


@pytest.mark.parametrize("unit, scale", [("Ohm", 1.0), ("mOhm", 1000.0)])
def test_absolute_impedance_is_converted_using_its_own_role_area(tmp_path, unit, scale):
    sources = inputs(tmp_path)
    for role, raw_resistance in [("anode", 0.2), ("cathode", 0.3), ("cell", 0.7)]:
        rows = [(10000, (raw_resistance - 0.05) * scale, 0.1 * scale),
                (1000, (raw_resistance + 0.05) * scale, -0.1 * scale),
                (100, (raw_resistance + 0.3) * scale, -0.5 * scale)]
        sources["eis"][role] = [write_spectrum(tmp_path / f"{role}_raw.csv", rows, unit=unit)]
    sources["parameters"] = {"anode_area": 2.0, "cathode_area": 3.0, "cell_area": 4.0}
    result = IntegratedReTool().run([], params(include_ir=True), extra=sources)
    for role, mean in [("anode", 0.4), ("cathode", 0.9), ("cell", 2.8)]:
        assert role_row(result.tables["HFR summary"], role)["Mean HFR (Ω·cm²)"] == pytest.approx(mean)


def test_raw_impedance_without_area_reports_the_role(tmp_path):
    sources = inputs(tmp_path, cell=False)
    sources["eis"]["anode"] = [hfr_file(tmp_path, "raw_anode.csv", 0.2, unit="Ohm")]
    with pytest.raises(ValueError, match="(?i)anode"):
        IntegratedReTool().run([], params(include_ir=True), extra=sources)


def test_area_normalized_impedance_is_not_multiplied_by_area_twice(tmp_path):
    sources = inputs(tmp_path)
    sources["parameters"] = {"anode_area": 2.0, "cathode_area": 3.0, "cell_area": 4.0}
    result = IntegratedReTool().run([], params(include_ir=True), extra=sources)
    for role, mean in [("anode", 0.2), ("cathode", 0.3), ("cell", 0.7)]:
        assert role_row(result.tables["HFR summary"], role)["Mean HFR (Ω·cm²)"] == pytest.approx(mean)


def test_nonoverlapping_cell_still_draws_all_curves_without_extrapolation(tmp_path):
    sources = inputs(tmp_path)
    sources["cell"] = write_curve(tmp_path / "cell.csv", [4, 5], [2, 3])
    result = IntegratedReTool().run([], params(include_ir=True), extra=sources)
    assert result.tables["Cell comparison"].empty
    assert result.tables["Cell comparison (iR corrected)"].empty
    assert len(result.figures["Cell comparison"].axes[0].get_legend_handles_labels()[0]) == 4
    assert any("overlap" in message.lower() for message in result.warnings)


def test_plot_only_result_keeps_analysis_details_out_of_exports(tmp_path):
    tool = IntegratedReTool()
    result = tool.run([], params(include_ir=True), extra=inputs(tmp_path))
    assert result.display_tables is False
    assert result.display_summary is False
    assert set(result.tables) >= {
        "Anode LSV", "Cathode LSV", "Cell LSV", "Reconstructed cell",
        "Reconstructed cell (iR corrected)", "Cell comparison",
        "Cell comparison (iR corrected)", "HFR per spectrum", "HFR summary",
        "EIS spectra", "Run details", "Potential settings",
    }
    details = " ".join(map(str, result.tables["Run details"].to_numpy().ravel())).lower()
    assert "hfr" in details and "correction" in details
    assert any("eis" in name.lower() for name in result.tables)
    artifacts = tool.exportable_artifacts(result)
    workbook = next(artifact for artifact in artifacts if artifact.kind == "workbook")
    assert workbook.payload.sheetnames == ["Cell comparison", "Electrode LSVs"]
    workbook_path = tmp_path / workbook.default_filename
    ExportButton._write(workbook, str(workbook_path))
    written = pd.read_excel(workbook_path, sheet_name="Cell comparison", header=3)
    np.testing.assert_allclose(
        written["Reconstructed V iR corrected (V)"],
        result.tables["Reconstructed cell (iR corrected)"][VOLTAGE],
    )
    csvs = [artifact for artifact in artifacts if artifact.kind == "csv"]
    assert len(csvs) == 2
    assert all("HFR" not in artifact.name and "EIS" not in artifact.name for artifact in artifacts)
    for artifact in csvs:
        destination = tmp_path / artifact.default_filename
        ExportButton._write(artifact, str(destination))
        assert len(pd.read_csv(destination)) == len(artifact.payload)
    figures = [artifact for artifact in artifacts if artifact.kind == "png"]
    assert len(figures) == len(result.figures) == 2
    for artifact in figures:
        destination = tmp_path / artifact.default_filename
        ExportButton._write(artifact, str(destination))
        assert destination.stat().st_size > 1000
