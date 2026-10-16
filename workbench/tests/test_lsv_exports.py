"""LSV exports preserve plotted results without duplicate analysis worksheets."""

import numpy as np
import pandas as pd
import pytest

from workbench.core.io import write_xlsx
from workbench.tools.lsv.logic_tafel import calculate_RCL_overpotential
from workbench.tools.lsv.tool import LsvTool
from workbench.tools.lsv_generator.tool import LsvGeneratorTool


def _workbook(tool, result):
    return next(item.payload for item in tool.exportable_artifacts(result) if item.kind == "xlsx")


def _lsv_source(tmp_path, reaction="OER", setup="Three-electrode", include_rcl=True,
                voltage_sign=1):
    params = {spec.key: spec.default for spec in LsvTool().parameter_schema()}
    params.update(electrode_setup=setup, reaction=reaction, reference="RHE", ph=0,
                  temperature=298.15, hfr=0.15, rcl=0.2, include_rcl=include_rcl,
                  i_lower=0.005, i_upper=0.1, target_i=0.7)
    current = np.sort(np.append(np.logspace(-3, 0, 350), 0.7))
    kinetic = 0.06 * np.log10(current / 1e-6)
    catalyst = calculate_RCL_overpotential(current, 0.2, 0.06) if include_rcl else 0.0
    loss = kinetic + catalyst + 0.15 * current + 0.5 * np.maximum(current - 0.5, 0)
    if setup == "Two-electrode":
        potential = voltage_sign * (1.2291 + loss)
    else:
        potential = -loss if reaction == "HER" else 1.2291 + loss
    signed_current = -current if reaction == "HER" else current
    path = tmp_path / "lsv.csv"
    pd.DataFrame({"E(V)": potential, "j(A/cm2)": signed_current}).to_csv(path, index=False)
    return path, params


@pytest.mark.parametrize("multiple", [False, True])
def test_preview_exports_each_plotted_series_once(tmp_path, multiple):
    first, params = _lsv_source(tmp_path)
    paths = [first]
    if multiple:
        second = tmp_path / "replicate.csv"
        second.write_bytes(first.read_bytes())
        paths.append(second)
    tool = LsvTool()
    result = tool.show_plot(paths, params)
    sheets = _workbook(tool, result)

    assert list(sheets) == ["LSV curves", "Conditions"]
    data = sheets["LSV curves"]
    assert list(data) == ["Series", "Potential (V vs RHE)", "Current Density (A/cm²)"]
    lines = result.figures["LSV originals"].axes[0].lines
    assert data["Series"].unique().tolist() == [line.get_label() for line in lines]
    for line in lines:
        series = data[data["Series"] == line.get_label()]
        np.testing.assert_allclose(series.iloc[:, 1], line.get_xdata())
        np.testing.assert_allclose(series.iloc[:, 2] * 1e3, line.get_ydata())
    pd.testing.assert_frame_equal(sheets["Conditions"], result.tables["Potential settings"])


@pytest.mark.parametrize("reaction,setup,include_rcl,voltage_sign,swap_axes", [
    ("OER", "Three-electrode", False, 1, False),
    ("HER", "Three-electrode", True, 1, True),
    ("HER", "Two-electrode", True, 1, False),
    ("OER", "Two-electrode", False, -1, True),
])
def test_decoupling_exports_curves_fit_and_target_breakdown(
        tmp_path, reaction, setup, include_rcl, voltage_sign, swap_axes):
    path, params = _lsv_source(tmp_path, reaction, setup, include_rcl, voltage_sign)
    params.update(x_log=True, swap_axes=swap_axes)
    tool = LsvTool()
    result = tool.run([path], params)
    originals = {name: table.copy(deep=True) for name, table in result.tables.items()}
    sheets = _workbook(tool, result)

    assert list(sheets) == ["Decoupled curves", "Fit results", "Target breakdown"]
    assert "Overpotential components" not in sheets and "Catalyst utilization" not in sheets
    data = sheets["Decoupled curves"]
    lines = result.figures["Decoupled curves"].axes[0].lines
    current = np.abs(data.iloc[:, 0].to_numpy())
    for column, line in zip(data.columns[1:], lines, strict=False):
        plotted_voltage = line.get_xdata() if swap_axes else line.get_ydata()
        plotted_current = line.get_ydata() if swap_axes else line.get_xdata()
        np.testing.assert_allclose(data[column], plotted_voltage)
        np.testing.assert_allclose(current, plotted_current)
    assert len(data.columns) == len(lines) + 1 + int(include_rcl)
    assert ("Catalyst utilization U_J (-)" in data) == include_rcl
    if include_rcl:
        line = result.figures["Catalyst utilization"].axes[0].lines[0]
        order = np.argsort(current)
        np.testing.assert_allclose(current[order], line.get_xdata())
        np.testing.assert_allclose(data["Catalyst utilization U_J (-)"].iloc[order], line.get_ydata())
    breakdown = sheets["Target breakdown"]
    assert breakdown.iloc[0, 0] == 0.7
    np.testing.assert_allclose(breakdown.iloc[0, 1:].to_numpy(dtype=float),
                               [patch.get_width() for patch in result.figures["Overpotential breakdown"].axes[0].patches])
    fitting = sheets["Fit results"].set_index("Parameter")["Value"]
    assert fitting["Tafel slope (V/dec)"] == result.scalars["b_kin (V/dec)"]
    assert fitting["Exchange current density (A/cm²)"] == result.scalars["i0 (A/cm²)"]
    assert fitting["HFR (Ω·cm²)"] == params["hfr"]
    assert ("R_CL (Ω·cm²)" in fitting.index) == include_rcl
    assert "RHE conversion shift (V)" not in fitting.index
    assert "E_ref vs SHE (V)" not in fitting.index
    for name, table in originals.items():
        pd.testing.assert_frame_equal(result.tables[name], table)


@pytest.mark.parametrize("mode", ["OER", "HER", "Two-electrode"])
def test_generator_export_retains_final_curve_losses_and_parameters(tmp_path, mode):
    tool = LsvGeneratorTool()
    params = {spec.key: spec.default for spec in tool.parameter_schema()}
    params.update(mode=mode, oer_b=0.06, oer_i0=1e-6, oer_rcl=0.2,
                  her_b=0.04, her_i0=1e-5, her_rcl=0.1)
    result = tool.run([], params)
    sheets = _workbook(tool, result)

    assert list(sheets) == ["Generated LSV", "Parameters"]
    data = sheets["Generated LSV"]
    assert not {"Current Magnitude (A/cm²)", "E_rev (V)", "Total Modeled Loss (V)"} & set(data)
    line = result.figures["Generated LSV"].axes[0].lines[0]
    np.testing.assert_allclose(data.iloc[:, 0], line.get_xdata())
    np.testing.assert_allclose(data["Current density (A/cm²)"] * 1e3, line.get_ydata())
    generated = result.payload["generated"]
    for role in ("oer", "her"):
        if mode == role.upper() or mode == "Two-electrode":
            np.testing.assert_allclose(data[f"η_kin,{role.upper()} (V)"], generated[f"eta_kin_{role}"])
            np.testing.assert_allclose(data[f"η_RCL,{role.upper()} (V)"], generated[f"eta_rcl_{role}"])
            np.testing.assert_allclose(data[f"{role.upper()} Catalyst Utilization U_J (-)"],
                                       generated[f"utilization_{role}"])
    pd.testing.assert_frame_equal(sheets["Parameters"], result.tables["Parameters and endpoint"])
    target = tmp_path / f"{mode}.xlsx"
    write_xlsx(sheets, target)
    for name, expected in sheets.items():
        pd.testing.assert_frame_equal(pd.read_excel(target, sheet_name=name, keep_default_na=False), expected,
                                      check_dtype=False, rtol=1e-12, atol=1e-14)
    assert sum(item.kind == "png" for item in tool.exportable_artifacts(result)) == 3
