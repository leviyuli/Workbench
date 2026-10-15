"""LSV uses the correct potential scale for preview, fitting, and exports."""
import numpy as np
import pandas as pd
import pytest

from workbench.core.io import write_xlsx
from workbench.tools.lsv.tool import FARADAY, R_GAS, LsvTool


def _params(**overrides):
    params = {spec.key: spec.default for spec in LsvTool().parameter_schema()}
    params.update(
        electrode_setup="Three-electrode",
        reaction="OER", temperature=298.15, ph=13.0, hfr=0.0, rcl=0.0,
        include_rcl=False, i_lower=0.02, i_upper=0.8, target_i=0.5,
    )
    params.update(overrides)
    return params


def _write_lsv(path, shift=0.0, unit="V"):
    current = np.logspace(-3, 0, 400)
    potential_rhe = 1.2291 + 0.06 * np.log10(current / 1e-6)
    measured = potential_rhe - shift
    scale = 1e3 if unit == "mV" else 1.0
    pd.DataFrame({f"E({unit})": measured * scale, "j(A/cm2)": current}).to_csv(
        path, index=False,
    )
    return potential_rhe, measured


@pytest.mark.parametrize(
    "reference,e_ref,unit",
    [
        ("RHE", 0.0, "V"),
        ("Ag/AgCl (sat. KCl, 25 °C)", 0.197, "mV"),
        ("Custom E_ref vs SHE", 0.1234, "V"),
    ],
)
def test_lsv_preview_and_fit_apply_the_same_reference_shift_once(
    tmp_path, reference, e_ref, unit,
):
    params = _params(reference=reference, custom_e_ref_vs_she=e_ref)
    shift = 0.0 if reference == "RHE" else (
        e_ref + 2.303 * R_GAS * params["temperature"] / FARADAY * params["ph"]
    )
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    expected_rhe, _ = _write_lsv(first, shift, unit)
    _write_lsv(second, shift, unit)
    tool = LsvTool()
    preview = tool.show_plot([first, second], params)
    fit = tool.run([first, second], params)

    axis = preview.figures["LSV originals"].axes[0]
    assert "RHE" in axis.get_xlabel()
    for line in axis.lines:  # two input curves and their average
        np.testing.assert_allclose(line.get_xdata(), expected_rhe)
    for name in ("LSV data (per file)", "Averaged LSV"):
        table = preview.tables[name]
        potential_column = next(column for column in table if "Potential" in column)
        assert "RHE" in potential_column
        for _, frame in table.groupby("File") if "File" in table else [(None, table)]:
            np.testing.assert_allclose(frame[potential_column], expected_rhe)
    np.testing.assert_allclose(
        fit.tables["Curves (stacked)"]["Original LSV (RHE)"], expected_rhe,
    )
    assert fit.scalars["RHE conversion shift (V)"] == pytest.approx(shift)
    assert fit.scalars["b_kin (V/dec)"] == pytest.approx(0.06)
    assert fit.scalars["i0 (A/cm²)"] == pytest.approx(1e-6)


def test_lsv_preview_and_fit_export_all_potential_settings(tmp_path):
    params = _params(reference="Custom E_ref vs SHE", custom_e_ref_vs_she=0.1234)
    shift = 0.1234 + 2.303 * R_GAS * 298.15 / FARADAY * 13.0
    path = tmp_path / "lsv.csv"
    _write_lsv(path, shift)
    tool = LsvTool()
    for method in ("show_plot", "run"):
        result = getattr(tool, method)([path], params)
        artifact = next(item for item in tool.exportable_artifacts(result)
                        if item.kind == "xlsx")
        output = tmp_path / f"{method}.xlsx"
        write_xlsx(artifact.payload, output)
        settings = pd.read_excel(output, sheet_name="Potential settings")
        settings = settings.set_index("Parameter")["Value"].to_dict()
        assert settings == {
            "Electrode setup": "Three-electrode",
            "Potential scale": "RHE",
            "Potential shift (V)": pytest.approx(shift),
            "Reference electrode": "Custom E_ref vs SHE",
            "Reference E vs SHE (V)": pytest.approx(0.1234),
            "pH": 13.0,
            "Temperature (K)": 298.15,
        }


@pytest.mark.parametrize(
    "include_rcl,swap_axes,current_sign,voltage_sign",
    [
        (False, False, 1, 1),
        (False, True, -1, 1),
        (True, False, -1, 1),
        (True, True, 1, -1),
    ],
)
def test_lsv_two_electrode_fits_measured_cell_voltage(
    tmp_path, include_rcl, swap_axes, current_sign, voltage_sign,
):
    path = tmp_path / "cell.csv"
    temperature = 333.15
    reversible = 1.2291 - 0.0008456 * (temperature - 298.15)
    current = np.logspace(-3, 0, 400)
    b_kin, i0, hfr, rcl = 0.06, 2e-6, 0.15, 0.2
    kinetic = b_kin * np.log10(current / i0)
    # Build a cell curve from known effective Tafel and resistance losses.
    catalyst = (
        b_kin / 1.1982 * np.log10(
            1 + (current * np.log(10) * rcl / (2 * b_kin)) ** 1.1982,
        ) if include_rcl else np.zeros_like(current)
    )
    measured = voltage_sign * (reversible + kinetic + hfr * current + catalyst)
    pd.DataFrame({"E(V)": measured, "j(A/cm2)": current_sign * current}).to_csv(
        path, index=False,
    )
    # The parser orders rows by voltage, including negative-polarity cells.
    order = np.argsort(measured)
    measured, current = measured[order], current[order]
    kinetic, catalyst = kinetic[order], catalyst[order]
    params = _params(
        electrode_setup="Two-electrode", temperature=temperature,
        reaction="HER" if current_sign > 0 else "OER",
        reference="inactive invalid reference", ph="inactive invalid pH",
        custom_e_ref_vs_she=float("nan"), hfr=hfr,
        rcl=rcl if include_rcl else 3.0, include_rcl=include_rcl,
        swap_axes=swap_axes,
    )
    tool = LsvTool()
    preview = tool.show_plot([path], params)
    axis = preview.figures["LSV originals"].axes[0]
    assert "cell voltage" in axis.get_xlabel().lower()
    assert "RHE" not in axis.get_xlabel()
    np.testing.assert_allclose(axis.lines[0].get_xdata(), measured)
    table = preview.tables["LSV data (per file)"]
    column = next(column for column in table if "voltage" in column.lower())
    np.testing.assert_allclose(table[column], measured)
    np.testing.assert_allclose(table["Current Density (A/cm²)"], current_sign * current)

    fit = tool.run([path], params)
    assert fit.scalars["b_kin (V/dec)"] == pytest.approx(b_kin, rel=1e-5)
    assert fit.scalars["i0 (A/cm²)"] == pytest.approx(i0, rel=1e-5)
    assert fit.scalars["E_rev (V)"] == pytest.approx(voltage_sign * reversible)
    curves = fit.tables["Curves (stacked)"]
    assert "Original LSV (cell voltage)" in curves
    assert all("RHE" not in column for column in curves)
    np.testing.assert_allclose(curves["Original LSV (cell voltage)"], measured)
    np.testing.assert_allclose(curves["Current Density (A/cm²)"], current_sign * current)
    np.testing.assert_allclose(curves["E_rev"], voltage_sign * reversible)
    np.testing.assert_allclose(curves["+ η_res"], measured, atol=1e-8)
    components = fit.tables["Overpotential components"]
    np.testing.assert_allclose(components["η_kin (V)"], voltage_sign * kinetic, atol=1e-8)
    np.testing.assert_allclose(components["η_ohm (V)"], voltage_sign * hfr * current)
    assert ("η_RCL (V)" in components) == include_rcl
    assert ("Catalyst utilization" in fit.tables) == include_rcl
    assert ("Catalyst utilization" in fit.figures) == include_rcl
    if include_rcl:
        np.testing.assert_allclose(components["η_RCL (V)"], voltage_sign * catalyst,
                                   atol=1e-8)

    curves_axis = fit.figures["Decoupled curves"].axes[0]
    potential_label = curves_axis.get_xlabel() if swap_axes else curves_axis.get_ylabel()
    assert "cell voltage" in potential_label.lower()
    original_line = curves_axis.lines[-1]
    voltage_data = original_line.get_xdata() if swap_axes else original_line.get_ydata()
    current_data = original_line.get_ydata() if swap_axes else original_line.get_xdata()
    np.testing.assert_allclose(voltage_data, measured)
    np.testing.assert_allclose(current_data, current_sign * current)
    for figure in fit.figures.values():
        for axis in figure.axes:
            assert "RHE" not in axis.get_xlabel() + axis.get_ylabel()
    breakdown = fit.figures["Overpotential breakdown"].axes[0]
    assert "cell voltage" in breakdown.get_xlabel().lower()
    assert breakdown.patches[0].get_width() == pytest.approx(reversible)

    for result in (preview, fit):
        settings = result.tables["Potential settings"].set_index("Parameter")["Value"]
        assert settings["Electrode setup"] == "Two-electrode"
        assert settings["Potential scale"] == "Cell voltage"
        assert settings["Potential shift (V)"] == 0.0
        assert result.scalars["Potential shift (V)"] == 0.0
    fitting_info = fit.tables["Fitting info"].set_index("Parameter")["Value"]
    assert fitting_info["E_rev (V)"] == pytest.approx(voltage_sign * reversible)
    for labels in (fitting_info.index, fit.scalars):
        assert not any("RHE" in label or "SHE" in label or "reference" in label.lower()
                       for label in labels)
    diagnostics = fit.text_blocks["Diagnostics"]
    assert "cell voltage" in diagnostics.lower()
    assert "RHE" not in diagnostics and "SHE" not in diagnostics
    workbook = next(artifact.payload for artifact in tool.exportable_artifacts(fit)
                    if artifact.kind == "xlsx")
    pd.testing.assert_frame_equal(workbook["Curves (stacked)"], curves)
    pd.testing.assert_frame_equal(workbook["Potential settings"], fit.tables["Potential settings"])


def test_lsv_schema_exposes_shared_electrode_setup_and_custom_reference_controls():
    specs = {spec.key: spec for spec in LsvTool().parameter_schema()}
    assert specs["electrode_setup"].choices == ["Three-electrode", "Two-electrode"]
    assert specs["electrode_setup"].default == "Two-electrode"
    assert "SHE" in specs["reference"].choices
    assert "Ag/AgCl (3 M KCl, 25 °C)" in specs["reference"].choices
    assert "custom_e_ref_vs_she" in specs
    assert "target_scale" not in specs  # electrode setup determines the fitting scale
    assert specs["temperature"].default == 378.0
    assert specs["temperature"].minimum == 0.01
