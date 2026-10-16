"""Integrated RE processes measured sweeps and exports accountable comparisons."""

import numpy as np
import pandas as pd
import pytest

from workbench.tools.integrated_re.tool import IntegratedReTool
from workbench.widgets.export_button import ExportButton


CURRENT = "Current density (A/cm²)"
VOLTAGE = "Reconstructed cell voltage (V)"


def params(**updates):
    values = {spec.key: spec.default
              for spec in IntegratedReTool().parameter_schema()}
    # Synthetic calibration curves in these tests are already versus RHE.
    values.update(calibration_reference="RHE", calibration_ph=0.0,
                  calibration_temperature=298.15)
    values.update(updates)
    return values


def write_curve(path, potential, current, *, potential_unit="V",
                current_header="j(A/cm2)"):
    pd.DataFrame({f"E({potential_unit})": potential,
                  current_header: current}).to_csv(
        path, index=False, sep="\t" if path.suffix == ".txt" else ",",
    )
    return str(path)


def linear_pair(tmp_path, *, common_offset=0.0):
    ja = np.array([0.0, 0.1, 0.3, 0.5])
    jc = np.array([0.0, 0.15, 0.2, 0.35, 0.5])
    return {
        "anode": write_curve(tmp_path / "anode.csv",
                             1.4 + 0.8 * ja + common_offset, ja),
        "cathode": write_curve(tmp_path / "cathode.csv",
                               -0.1 - 0.4 * jc + common_offset, -jc),
    }


def add_zero_hfr_eis(tmp_path, sources, role="cathode"):
    path = tmp_path / f"{role}_zero_hfr.csv"
    path.write_text(
        "Frequency (Hz),Re(Z) (Ohm.cm2),Im(Z) (Ohm.cm2)\n"
        "10000,0,0\n1000,0.1,-0.1\n100,0.3,-0.5\n",
        encoding="utf-8",
    )
    sources.setdefault("eis", {})[role] = [str(path)]


def reaction_line(figure, reaction):
    return next(line for line in figure.axes[0].lines
                if reaction.lower() in line.get_label().lower())


def test_integrated_schema_defaults_to_rhe_and_explicit_correction_modes():
    specs = {spec.key: spec for spec in IntegratedReTool().parameter_schema()}
    assert specs["reference"].default == "RHE"
    assert specs["ph"].default == 0.0
    assert specs["temperature"].default == 298.15
    assert specs["calibration_reference"].default == "Hg/HgO (1 M KOH, 25 °C)"
    assert specs["calibration_ph"].default == 14.0
    assert specs["calibration_temperature"].default == 298.0
    assert specs["manual_shift"].default == 0.0
    assert specs["calibration_electrode"].default == "Cathode"
    assert specs["onset_current"].default == 0.05
    assert specs["electrode_axis"].choices == ["E vs RHE", "Overpotential"]
    assert specs["correction_mode"].choices == [
        "None", "Manual", "3E onset", "3E peak",
    ]
    for key in ("calculated_shift", "applied_shift"):
        assert specs[key].kind == "str"
        assert specs[key].read_only
        assert specs[key].default == "Not calculated"
        assert specs[key].suffix == "V"
    assert not specs["manual_shift"].read_only
    assert "target_scale" not in specs
    assert not IntegratedReTool().files_required


@pytest.mark.parametrize("invalid_column, invalid_value", [
    ("current", "missing"), ("potential", "missing"), ("potential", np.inf),
])
def test_invalid_measurement_rows_remain_gaps_in_reconstruction_and_plots(
        tmp_path, invalid_column, invalid_value):
    potential = [1.4, 1.5, 1.6]
    current = [0.0, 0.5, 1.0]
    (potential if invalid_column == "potential" else current)[1] = invalid_value
    sources = {
        "anode": write_curve(tmp_path / "anode.csv", potential, current),
        "cathode": write_curve(tmp_path / "cathode.csv", [-0.1, -0.3, -0.5], [0, -0.5, -1]),
    }
    result = IntegratedReTool().run([], params(), extra=sources)
    source = result.tables["Anode LSV"]
    assert len(source) == 3
    assert source["In comparison window"].tolist() == [True, False, True]
    assert result.tables["Reconstructed cell"][CURRENT].tolist() == [1.0]
    assert any("retained as gaps" in warning for warning in result.warnings)
    line = reaction_line(result.figures["Electrode LSVs"], "Anode")
    assert np.isnan(line.get_xdata()[1])
    assert np.all(np.isfinite(result.figures["Electrode LSVs"].axes[0].get_xlim()))


def test_reconstruction_matches_distinct_current_grids_without_double_equilibrium(
        tmp_path):
    sources = linear_pair(tmp_path)
    measured_j = np.array([0.05, 0.2, 0.4, 0.6])
    sources["cell"] = write_curve(tmp_path / "cell.csv",
                                  1.51 + 1.3 * measured_j, measured_j)
    result = IntegratedReTool().run([], params(), extra=sources)

    reconstructed = result.tables["Reconstructed cell"]
    np.testing.assert_allclose(reconstructed[CURRENT],
                               [0.05, 0.1, 0.15, 0.2, 0.3, 0.35, 0.4, 0.5])
    j = reconstructed[CURRENT].to_numpy()
    np.testing.assert_allclose(reconstructed["Anode E (V vs RHE)"], 1.4 + 0.8 * j)
    np.testing.assert_allclose(reconstructed["Cathode E (V vs RHE)"], -0.1 - 0.4 * j)
    np.testing.assert_allclose(reconstructed[VOLTAGE], 1.5 + 1.2 * j)
    np.testing.assert_allclose(reconstructed["Cell overpotential (V)"],
                               1.5 + 1.2 * j - 1.2291)
    assert result.scalars["Reversible cell voltage (V)"] == pytest.approx(1.2291)

    comparison = result.tables["Cell comparison"]
    jc = comparison[CURRENT].to_numpy()
    assert np.all((jc >= 0.05) & (jc <= 0.6))
    np.testing.assert_allclose(comparison["Measured cell voltage (V)"],
                               1.51 + 1.3 * jc)
    np.testing.assert_allclose(comparison["Voltage difference (V)"],
                               -0.01 - 0.1 * jc, atol=1e-14)


def test_common_manual_shift_preserves_voltage_and_unshifted_branch_windows(tmp_path):
    j = np.array([0.05, 0.15, 0.25, 0.35])
    sources = {
        "anode": write_curve(tmp_path / "anode.csv", 1.4 + j, j),
        "cathode": write_curve(tmp_path / "cathode.csv", -0.1 - 0.5 * j, -j),
    }
    options = params(anode_use_window=True, anode_e_min=1.5, anode_e_max=1.8,
                     cathode_use_window=True, cathode_e_min=-0.3,
                     cathode_e_max=-0.15)
    tool = IntegratedReTool()
    original = tool.run([], options, extra=sources)
    shifted = tool.run([], {**options, "correction_mode": "Manual",
                           "manual_shift": -0.2}, extra=sources)

    for role in ("Anode", "Cathode"):
        before = original.tables[f"{role} LSV"]
        after = shifted.tables[f"{role} LSV"]
        np.testing.assert_allclose(after["E (V vs RHE)"],
                                   before["E (V vs RHE)"] - 0.2)
        np.testing.assert_array_equal(after["In comparison window"],
                                      before["In comparison window"])
        np.testing.assert_array_equal(after["In comparison window"],
                                      [False, True, True, True])
    np.testing.assert_allclose(shifted.tables["Reconstructed cell"][CURRENT],
                               [0.15, 0.25, 0.35])
    np.testing.assert_allclose(shifted.tables["Reconstructed cell"][VOLTAGE],
                               original.tables["Reconstructed cell"][VOLTAGE])
    assert shifted.scalars["Shared correction (V)"] == pytest.approx(-0.2)


def test_overpotential_plot_changes_electrode_axes_only(tmp_path):
    sources = linear_pair(tmp_path)
    tool = IntegratedReTool()
    rhe = tool.run([], params(), extra=sources)
    eta = tool.run([], params(electrode_axis="Overpotential"), extra=sources)
    for role in ("Anode", "Cathode"):
        rhe_line = reaction_line(rhe.figures["Electrode LSVs"], role)
        eta_line = reaction_line(eta.figures["Electrode LSVs"], role)
        shift = 1.2291 if role == "Anode" else 0.0
        np.testing.assert_allclose(eta_line.get_xdata(), rhe_line.get_xdata() - shift)
        np.testing.assert_allclose(eta_line.get_ydata(), rhe_line.get_ydata())
    pd.testing.assert_frame_equal(rhe.tables["Reconstructed cell"],
                                  eta.tables["Reconstructed cell"])
    for result in (rhe, eta):
        ax = result.figures["Electrode LSVs"].axes[0]
        assert ax.get_xlim()[0] < 0 < ax.get_xlim()[1]
        assert "current" in ax.get_ylabel().lower()
        cathode = reaction_line(result.figures["Electrode LSVs"], "Cathode")
        assert np.all(cathode.get_ydata() <= 0)
    assert "RHE" in rhe.figures["Electrode LSVs"].axes[0].get_xlabel()
    assert "overpotential" in eta.figures["Electrode LSVs"].axes[0].get_xlabel().lower()
    cell_ax = eta.figures["Cell comparison"].axes[0]
    assert "voltage" in cell_ax.get_xlabel().lower()
    assert "current" in cell_ax.get_ylabel().lower()


def test_three_electrode_onset_recovers_shared_offset_without_accumulation(tmp_path):
    sources = linear_pair(tmp_path, common_offset=0.2)
    j = np.array([0.0, 0.01, 0.1, 0.3, 0.5])
    sources["calibration"] = write_curve(tmp_path / "calibration.csv",
                                         -0.1 - 0.4 * j, -j)
    add_zero_hfr_eis(tmp_path, sources)
    options = params(correction_mode="3E onset", manual_shift=0.03)
    tool = IntegratedReTool()
    first = tool.run([], options, extra=sources)
    # ToolPage merges param_updates into the sidebar after completion. Those
    # display values must never become a second correction on the next run.
    again = tool.run([], {**options, **first.param_updates}, extra=sources)

    assert first.scalars["Shared correction (V)"] == pytest.approx(-0.17)
    assert {key: first.param_updates[key] for key in ("calculated_shift", "applied_shift")} == {
        "calculated_shift": "-0.200000", "applied_shift": "-0.170000",
    }
    assert "manual_shift" not in first.param_updates
    assert again.param_updates == first.param_updates
    for name, table in first.tables.items():
        pd.testing.assert_frame_equal(table, again.tables[name])
    assert "Calibration alignment" in first.figures
    assert "Calibration features" in first.tables
    c = first.tables["Cathode LSV"]
    np.testing.assert_allclose(c["E (V vs RHE)"],
                               c["E before correction (V vs RHE)"] - 0.17)
    a = first.tables["Anode LSV"]
    np.testing.assert_allclose(a["E (V vs RHE)"],
                               a["E before correction (V vs RHE)"] - 0.17)


def test_integrated_and_calibration_reference_conversion_are_independent(tmp_path):
    j = np.array([0.0, 0.01, 0.1, 0.3])
    nernst = 2.303 * 8.31446261815324 / 96485.33212
    integrated_shift = 0.11 + nernst * 298.15 * 13
    calibration_shift = 0.22 + nernst * 310 * 14
    sources = {
        "anode": write_curve(tmp_path / "anode.txt",
                             (1.4 + 0.8 * j + 0.2 - integrated_shift) * 1000,
                             j * 1000, potential_unit="mV",
                             current_header="j(mA/cm2)"),
        "cathode": write_curve(tmp_path / "cathode.csv",
                               -0.1 - 0.4 * j + 0.2 - integrated_shift, -j),
        "calibration": write_curve(tmp_path / "calibration.csv",
                                   -0.1 - 0.4 * j - calibration_shift, -j),
        "cell": write_curve(tmp_path / "cell.csv", 1.5 + 1.2 * j, j),
    }
    add_zero_hfr_eis(tmp_path, sources)
    result = IntegratedReTool().run([], params(
        reference="Custom E_ref vs SHE", custom_e_ref_vs_she=0.11, ph=13,
        calibration_reference="Custom E_ref vs SHE",
        calibration_custom_e_ref_vs_she=0.22, calibration_ph=14,
        calibration_temperature=310, correction_mode="3E onset",
    ), extra=sources)

    assert result.scalars["Shared correction (V)"] == pytest.approx(-0.2)
    np.testing.assert_allclose(result.tables["Anode LSV"]["E (V vs RHE)"],
                               1.4 + 0.8 * j)
    np.testing.assert_allclose(result.tables["Cathode LSV"]["E (V vs RHE)"],
                               -0.1 - 0.4 * j)
    np.testing.assert_allclose(result.tables["Calibration LSV"]["E (V vs RHE)"],
                               -0.1 - 0.4 * j)
    np.testing.assert_allclose(result.tables["Cell LSV"]["Cell voltage (V)"],
                               1.5 + 1.2 * j)


def test_peak_calibration_matches_interior_feature_in_separate_rhe_windows(tmp_path):
    e = np.array([-0.4, -0.3, -0.2, -0.1, 0.0])
    j = np.array([0.01, 0.05, 0.2, 0.04, 0.005])
    sources = {
        "anode": write_curve(tmp_path / "anode.csv", 1.4 + j + 0.2, j),
        "cathode": write_curve(tmp_path / "cathode.csv", e + 0.2, -j),
        "calibration": write_curve(tmp_path / "calibration.csv", e, -j),
    }
    add_zero_hfr_eis(tmp_path, sources)
    result = IntegratedReTool().run([], params(
        correction_mode="3E peak", manual_shift=0.02,
        integrated_peak_min=-0.15, integrated_peak_max=0.15,
        calibration_peak_min=-0.35, calibration_peak_max=-0.05,
    ), extra=sources)

    assert result.scalars["Shared correction (V)"] == pytest.approx(-0.18)
    assert {key: result.param_updates[key] for key in ("calculated_shift", "applied_shift")} == {
        "calculated_shift": "-0.200000", "applied_shift": "-0.180000",
    }
    assert "Calibration features" in result.tables
    assert "Calibration alignment" in result.figures
    np.testing.assert_allclose(result.tables["Cathode LSV"]["E (V vs RHE)"],
                               e + 0.02)


def test_offset_readouts_do_not_influence_manual_or_uncorrected_runs(tmp_path):
    sources = linear_pair(tmp_path)
    tool = IntegratedReTool()
    stale_readouts = {"calculated_shift": "-0.200000",
                      "applied_shift": "-0.170000"}
    manual = tool.run([], params(correction_mode="Manual", manual_shift=0.03,
                                 **stale_readouts), extra=sources)
    assert manual.scalars["Shared correction (V)"] == pytest.approx(0.03)
    assert {key: manual.param_updates[key] for key in ("calculated_shift", "applied_shift")} == {
        "calculated_shift": "+0.000000", "applied_shift": "+0.030000",
    }
    uncorrected = tool.run([], params(correction_mode="None", manual_shift=0.03,
                                      **stale_readouts), extra=sources)
    assert uncorrected.scalars["Shared correction (V)"] == pytest.approx(0.0)
    assert {key: uncorrected.param_updates[key] for key in ("calculated_shift", "applied_shift")} == {
        "calculated_shift": "+0.000000", "applied_shift": "+0.000000",
    }
    np.testing.assert_allclose(uncorrected.tables["Cathode LSV"]["E (V vs RHE)"],
                               uncorrected.tables["Cathode LSV"][
                                   "E before correction (V vs RHE)"])


def test_peak_calibration_rejects_a_window_endpoint_as_feature(tmp_path):
    sources = linear_pair(tmp_path)
    e = np.array([-0.4, -0.3, -0.2, -0.1])
    sources["calibration"] = write_curve(tmp_path / "calibration.csv", e,
                                         [-0.01, -0.05, -0.1, -0.2])
    add_zero_hfr_eis(tmp_path, sources)
    with pytest.raises(ValueError, match="(?i)(peak|feature)"):
        IntegratedReTool().run([], params(
            correction_mode="3E peak", integrated_peak_min=-0.4,
            integrated_peak_max=0.0, calibration_peak_min=-0.4,
            calibration_peak_max=-0.1,
        ), extra=sources)


def test_import_retains_signed_zeros_and_sweep_order_with_area_and_auto_polarity(tmp_path):
    ea = np.array([1.55, 1.4, 1.42, 1.48, 1.7])
    ec = np.array([-0.15, 0.0, 0.01, -0.1, -0.25])
    ia_ma = np.array([-100.0, 0.0, 5.0, -50.0, -200.0])
    jc_ma = np.array([50.0, 0.0, -5.0, 25.0, 100.0])
    sources = {
        "anode": write_curve(tmp_path / "anode.csv", ea, ia_ma,
                             current_header="I(mA)"),
        "cathode": write_curve(tmp_path / "cathode.txt", ec, jc_ma,
                               current_header="j(mA/cm2)"),
    }
    result = IntegratedReTool().run([], params(), extra={
        **sources, "parameters": {"anode_area": 2.0},
    })
    for role, e, source_j in (("Anode", ea, ia_ma / 2000),
                              ("Cathode", ec, jc_ma / 1000)):
        table = result.tables[f"{role} LSV"]
        np.testing.assert_allclose(table["Source potential (V)"], e)
        np.testing.assert_allclose(table["Source current (A/cm²)"], source_j)
        np.testing.assert_allclose(table[CURRENT], -source_j)
        assert len(table) == 5
        assert set(np.sign(table[CURRENT])) == {-1, 0, 1}
        assert table["File"].unique().tolist() == [f"{role.lower()}." +
                                                  ("csv" if role == "Anode" else "txt")]


def test_manual_polarity_override_is_applied_to_the_whole_curve(tmp_path):
    j = np.array([0.0, 0.1, 0.3])
    sources = {
        "anode": write_curve(tmp_path / "anode.csv", 1.4 + j, -j),
        "cathode": write_curve(tmp_path / "cathode.csv", -0.1 - j, j),
    }
    result = IntegratedReTool().run([], params(anode_sign="Invert",
                                               cathode_sign="Invert"), extra=sources)
    np.testing.assert_allclose(result.tables["Anode LSV"][CURRENT], j)
    np.testing.assert_allclose(result.tables["Cathode LSV"][CURRENT], -j)


def test_cell_is_optional_and_nonoverlapping_cell_has_no_extrapolated_residuals(tmp_path):
    sources = linear_pair(tmp_path)
    tool = IntegratedReTool()
    without = tool.run([], params(), extra=sources)
    assert "Cell comparison" in without.figures
    assert "Cell LSV" not in without.tables
    assert "Cell comparison" not in without.tables

    sources["cell"] = write_curve(tmp_path / "cell.csv", [4.0, 5.0], [2.0, 3.0])
    with_cell = tool.run([], params(), extra=sources)
    pd.testing.assert_frame_equal(without.tables["Reconstructed cell"],
                                  with_cell.tables["Reconstructed cell"])
    assert "Cell LSV" in with_cell.tables
    assert ("Cell comparison" not in with_cell.tables
            or with_cell.tables["Cell comparison"].empty)
    notices = " ".join(with_cell.warnings + list(with_cell.text_blocks.values()))
    assert "overlap" in notices.lower()
    ax = with_cell.figures["Cell comparison"].axes[0]
    assert len(ax.get_legend_handles_labels()[0]) >= 2


def test_nonoverlapping_electrodes_are_reported_instead_of_extrapolated(tmp_path):
    sources = {
        "anode": write_curve(tmp_path / "anode.csv", [1.5, 1.6], [0.1, 0.2]),
        "cathode": write_curve(tmp_path / "cathode.csv", [-0.2, -0.3], [-0.4, -0.5]),
    }
    with pytest.raises(ValueError, match="(?i)overlap"):
        IntegratedReTool().run([], params(), extra=sources)


@pytest.mark.parametrize("role", ["anode", "cathode", "calibration"])
def test_missing_required_inputs_identify_the_role(tmp_path, role):
    sources = linear_pair(tmp_path)
    options = params()
    if role == "calibration":
        options["correction_mode"] = "3E onset"
    else:
        del sources[role]
    with pytest.raises(ValueError, match=f"(?i){role}"):
        IntegratedReTool().run([], options, extra=sources)


def test_absolute_current_without_area_reports_the_electrode_role(tmp_path):
    sources = linear_pair(tmp_path)
    sources["anode"] = write_curve(tmp_path / "absolute.csv", [1.5, 1.7],
                                   [0.1, 0.3], current_header="I(A)")
    with pytest.raises(ValueError, match="(?i)(anode.*area|area.*anode)"):
        IntegratedReTool().run([], params(), extra=sources)


def test_results_export_final_plot_data_and_figures(tmp_path):
    sources = linear_pair(tmp_path)
    tool = IntegratedReTool()
    result = tool.run([], params(correction_mode="Manual", manual_shift=-0.2),
                      extra=sources)
    artifacts = tool.exportable_artifacts(result)
    workbook = next(item for item in artifacts if item.kind == "workbook")
    assert workbook.payload.sheetnames == ["Cell comparison", "Electrode LSVs"]
    ExportButton._write(workbook, str(tmp_path / workbook.default_filename))
    written = pd.read_excel(tmp_path / workbook.default_filename,
                            sheet_name="Electrode LSVs", header=0)
    n_anode = len(result.tables["Anode LSV"])
    np.testing.assert_allclose(written["Anode E (V vs RHE)"].iloc[:n_anode],
                               result.tables["Anode LSV"]["E (V vs RHE)"])
    csv = next(item for item in artifacts if item.kind == "csv"
               and item.name.startswith("Cell comparison"))
    ExportButton._write(csv, str(tmp_path / csv.default_filename))
    np.testing.assert_allclose(pd.read_csv(tmp_path / csv.default_filename)["Reconstructed V (V)"],
                               result.tables["Reconstructed cell"][VOLTAGE])
    pngs = [item for item in artifacts if item.kind == "png"]
    assert len(pngs) == 2
    for artifact in pngs:
        ExportButton._write(artifact, str(tmp_path / artifact.default_filename))
        assert (tmp_path / artifact.default_filename).stat().st_size > 1000
