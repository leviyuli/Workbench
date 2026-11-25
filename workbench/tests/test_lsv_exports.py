"""LSV exports preserve plotted results without duplicate analysis worksheets."""

import numpy as np
import pandas as pd
import pytest

from workbench.core.io import write_xlsx
from workbench.tools.lsv.logic_tafel import calculate_RCL_overpotential
from workbench.tools.lsv.tool import LsvTool
from workbench.tools.lsv_generator.tool import LsvGeneratorTool
from workbench.widgets.export_button import ExportButton


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


@pytest.mark.parametrize("sources", ["single", "replicates", "duplicate_names"])
@pytest.mark.parametrize("reaction,setup", [
    ("OER", "Three-electrode"),
    ("HER", "Three-electrode"),
    ("OER", "Two-electrode"),
])
def test_preview_exports_each_plotted_series_once(tmp_path, sources, reaction, setup):
    first, params = _lsv_source(tmp_path, reaction, setup)
    paths = [first]
    if sources != "single":
        second = tmp_path / "a_replicate.csv"
        if sources == "duplicate_names":
            second = tmp_path / "replicate" / first.name
            second.parent.mkdir()
        replicate = pd.read_csv(first).iloc[::2].copy()
        replicate.iloc[:, 1] *= 1.1
        replicate.to_csv(second, index=False)
        paths.append(second)
    tool = LsvTool()
    result = tool.show_plot(paths, params)
    originals = {name: table.copy(deep=True) for name, table in result.tables.items()}
    sheets = _workbook(tool, result)

    assert list(sheets) == ["LSV curves", "Conditions"]
    data = sheets["LSV curves"]
    lines = result.figures["LSV originals"].axes[0].lines
    if sources != "single":
        lines = [lines[-1], *lines[:-1]]
        assert lines[0].get_label() == "Average"
    potential_label = "Cell voltage (V)" if setup == "Two-electrode" else "Potential (V vs RHE)"
    assert list(data) == [
        f"{line.get_label()} · {quantity}"
        for line in lines
        for quantity in (potential_label, "Current Density (A/cm²)")
    ]
    assert data.columns.is_unique
    assert len(data) == max(len(line.get_xdata()) for line in lines)
    for offset, line in enumerate(lines):
        pair = data.iloc[:, 2 * offset:2 * offset + 2]
        assert pair.iloc[:, 0].isna().equals(pair.iloc[:, 1].isna())
        series = pair.dropna()
        np.testing.assert_allclose(series.iloc[:, 0], line.get_xdata())
        np.testing.assert_allclose(series.iloc[:, 1] * 1e3, line.get_ydata())
    pd.testing.assert_frame_equal(sheets["Conditions"], result.tables["Potential settings"])
    for artifact in tool.exportable_artifacts(result):
        if artifact.kind == "xlsx" or artifact.default_filename == "lsv_lsv_curves.csv":
            output = tmp_path / artifact.default_filename
            ExportButton._write(artifact, str(output))
            written = (pd.read_excel(output, sheet_name="LSV curves")
                       if artifact.kind == "xlsx" else pd.read_csv(output))
            pd.testing.assert_frame_equal(written, data, rtol=1e-12, atol=1e-14)
    for name, table in originals.items():
        pd.testing.assert_frame_equal(result.tables[name], table)


def test_all_measured_voltages_extends_preview_and_fit_without_changing_export_layout(tmp_path):
    tool = LsvTool()
    params = {spec.key: spec.default for spec in tool.parameter_schema()}
    params.update(electrode_setup="Two-electrode", temperature=298.15, hfr=0, include_rcl=False,
                  i_lower=0.02, i_upper=0.08, target_i=0.7)
    potential_label = "Cell voltage (V)"
    current_label = "Current Density (A/cm²)"

    short, back = tmp_path / "short.csv", tmp_path / "back.csv"
    short_voltage = np.linspace(1, 2, 201)
    back_voltage = np.linspace(3, 1, 401)
    short_current = 0.9 * 10 ** (short_voltage - 3)
    back_current = 10 ** (back_voltage - 3)
    pd.DataFrame({"E(V)": short_voltage, "j(A/cm2)": short_current}).to_csv(short, index=False)
    pd.DataFrame({"E(V)": back_voltage, "j(A/cm2)": back_current}).to_csv(back, index=False)
    paths = [short, back]

    overlap_preview = tool.show_plot(paths, params)
    overlap = overlap_preview.tables["Averaged LSV"]
    assert overlap[potential_label].max() == pytest.approx(2)
    with pytest.raises(ValueError, match="outside the measured range"):
        tool.run(paths, params)
    overlap_fit = tool.run(paths, {**params, "target_i": 0.07})

    params["average_range"] = "All measured voltages"
    preview = tool.show_plot(paths, params)
    average = preview.tables["Averaged LSV"]
    assert list(average) == list(overlap) == [potential_label, current_label]
    voltage = average[potential_label].to_numpy()
    shared = voltage <= 2
    assert voltage[0] == pytest.approx(1)
    assert voltage[-1] == pytest.approx(3)
    assert np.any(np.isclose(voltage, 2))
    back_at_voltage = np.interp(voltage, back_voltage[::-1], back_current[::-1])
    expected_current = back_at_voltage.copy()
    expected_current[shared] = (
        back_at_voltage[shared] + np.interp(voltage[shared], short_voltage, short_current)
    ) / 2
    np.testing.assert_allclose(average[current_label], expected_current)
    average_line = preview.figures["LSV originals"].axes[0].lines[-1]
    np.testing.assert_allclose(average_line.get_xdata(), voltage)
    np.testing.assert_allclose(average_line.get_ydata(), expected_current * 1e3)

    preview_export = _workbook(tool, preview)
    overlap_preview_export = _workbook(tool, overlap_preview)
    assert list(preview_export["LSV curves"]) == list(overlap_preview_export["LSV curves"])
    pd.testing.assert_frame_equal(preview_export["Conditions"], overlap_preview_export["Conditions"])
    np.testing.assert_allclose(
        preview_export["LSV curves"][f"Average · {current_label}"].dropna(), expected_current,
    )
    fit = tool.run(paths, params)
    assert list(fit.tables["Curves (stacked)"]) == list(overlap_fit.tables["Curves (stacked)"])
    fit_export = _workbook(tool, fit)
    overlap_fit_export = _workbook(tool, overlap_fit)
    exported = fit_export["Decoupled curves"]
    assert list(exported) == list(overlap_fit_export["Decoupled curves"])
    assert fit_export["Summary"]["Parameter"].tolist() == overlap_fit_export["Summary"]["Parameter"].tolist()
    np.testing.assert_allclose(exported["Measured cell voltage (V)"], voltage)
    np.testing.assert_allclose(exported[current_label], expected_current)


def test_disjoint_lsvs_leave_preview_gap_and_cannot_be_fit_as_one_curve(tmp_path):
    tool = LsvTool()
    params = {spec.key: spec.default for spec in tool.parameter_schema()}
    params.update(electrode_setup="Two-electrode", average_range="All measured voltages")
    first, back = tmp_path / "first.csv", tmp_path / "back.csv"
    first.write_text("E(V),j(A/cm2)\n1,0.1\n2,0.2\n", encoding="utf-8")
    back.write_text("E(V),j(A/cm2)\n4,0.4\n3,0.3\n", encoding="utf-8")

    preview = tool.show_plot([first, back], params)
    average = preview.tables["Averaged LSV"]
    voltage = average["Cell voltage (V)"]
    gap = voltage.between(2, 3, inclusive="neither")
    assert gap.any()
    current = average["Current Density (A/cm²)"]
    assert current.isna().equals(gap)
    np.testing.assert_allclose(current[~gap], voltage[~gap] / 10)
    average_line = preview.figures["LSV originals"].axes[0].lines[-1]
    assert np.isnan(average_line.get_ydata()[gap]).all()
    with pytest.raises(ValueError, match="gap in voltage coverage.*Batch decoupling & CCD"):
        tool.run([first, back], params)


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

    assert list(sheets) == ["Summary", "Decoupled curves"]
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
    summary = sheets["Summary"]
    breakdown = summary.loc[summary.Section == "Target breakdown"].set_index("Parameter")["Value"]
    assert breakdown.iloc[0] == 0.7
    np.testing.assert_allclose(breakdown.iloc[1:].to_numpy(dtype=float),
                               [patch.get_width() for patch in result.figures["Overpotential breakdown"].axes[0].patches])
    fitting = summary.loc[summary.Section == "Fit results"].set_index("Parameter")["Value"]
    assert fitting["Tafel slope (V/dec)"] == result.scalars["b_kin (V/dec)"]
    assert fitting["Exchange current density (A/cm²)"] == result.scalars["i0 (A/cm²)"]
    assert fitting["HFR (Ω·cm²)"] == params["hfr"]
    assert ("R_CL (Ω·cm²)" in fitting.index) == include_rcl
    assert "RHE conversion shift (V)" not in fitting.index
    assert "E_ref vs SHE (V)" not in fitting.index
    for name, table in originals.items():
        pd.testing.assert_frame_equal(result.tables[name], table)


def test_decoupling_export_menu_prioritizes_workbook_and_combined_summary(tmp_path):
    path, params = _lsv_source(tmp_path)
    tool = LsvTool()
    result = tool.run([path], params)
    artifacts = tool.exportable_artifacts(result)
    assert [(item.name, item.default_filename) for item in artifacts[:3]] == [
        ("LSV results (Excel)", "lsv_results.xlsx"),
        ("Summary (CSV)", "lsv_summary.csv"),
        ("Decoupled curves (CSV)", "lsv_decoupled_curves.csv"),
    ]
    assert all(item.kind == "png" for item in artifacts[3:])
    summary = artifacts[0].payload["Summary"]
    assert list(summary) == ["Section", "Record", "Parameter", "Value"]
    assert summary.Section.drop_duplicates().tolist() == ["Fit results", "Target breakdown"]
    assert (summary.Record == "").all()
    for artifact in artifacts[:2]:
        target = tmp_path / artifact.default_filename
        ExportButton._write(artifact, str(target))
        exported = (pd.read_excel(target, sheet_name="Summary", keep_default_na=False)
                    if artifact.kind == "xlsx" else pd.read_csv(target, keep_default_na=False))
        pd.testing.assert_frame_equal(exported.iloc[:, :3], summary.iloc[:, :3], check_dtype=False)
        # CSV has one mixed-type Value column, so parse numeric cells explicitly.
        slope = exported.loc[(exported.Section == "Fit results") &
                             (exported.Parameter == "Tafel slope (V/dec)"), "Value"].iloc[0]
        assert float(slope) == pytest.approx(result.scalars["b_kin (V/dec)"], rel=1e-12)


def test_combined_cv_hfr_ccd_export_preserves_records_and_excel_value_types(tmp_path):
    from openpyxl import load_workbook

    path, params = _lsv_source(tmp_path)
    forward = pd.read_csv(path)
    paths = []
    for number in (1, 2):
        folder = tmp_path / str(number)
        folder.mkdir()
        source = folder / "cv.csv"
        pd.concat([forward, forward.iloc[-2::-1]], ignore_index=True).assign(
            Cycle=number,
        ).to_csv(source, index=False)
        paths.append(source)
    params.update(input_type="CV", cv_scan="Forward", hfr_mode="HFR vs |j|", include_ccd=True)
    extra = {"hfr_text": "j,HFR\n.05,.15\n.2,.15\n.4,.15\n.6,.15\n.8,.15",
             "hfr_smoothing": .0001, "hfr_source": "hfr.csv"}
    tool = LsvTool()
    result = tool.run(paths, params, extra)
    originals = {name: table.copy(deep=True) for name, table in result.tables.items()}
    artifacts = tool.exportable_artifacts(result)
    sheets = artifacts[0].payload
    assert list(sheets) == ["Summary", "Decoupled curves", "HFR source data", "HFR fit data", "CCD data"]
    summary = sheets["Summary"]
    assert summary.Section.drop_duplicates().tolist() == [
        "Fit results", "Target breakdown", "CV selection", "HFR points",
        "HFR settings", "CCD results", "CCD settings",
    ]
    selection = summary.loc[summary.Section == "CV selection"]
    assert selection.loc[selection.Parameter == "File", "Value"].tolist() == ["cv.csv", "cv.csv (2)"]
    assert selection.loc[selection.Parameter == "Cycle", ["Record", "Value"]].values.tolist() == [[1, 1], [2, 2]]
    assert [item.name for item in artifacts if item.kind == "csv"] == [
        f"{name} (CSV)" for name in sheets
    ]
    target = tmp_path / "combined.xlsx"
    ExportButton._write(artifacts[0], str(target))
    workbook = load_workbook(target)
    try:
        assert workbook.sheetnames == list(sheets)
        worksheet = workbook["Summary"]
        for section, parameter, data_type in (
            ("Fit results", "Tafel slope (V/dec)", "n"),
            ("CV selection", "Cycle", "n"),
            ("HFR settings", "HFR extended below measurements", "b"),
        ):
            matches = summary.index[(summary.Section == section) & (summary.Parameter == parameter)]
            assert len(matches)
            for index in matches:
                cell = worksheet.cell(int(index) + 2, 4)
                assert cell.data_type == data_type
                assert cell.value == pytest.approx(summary.at[index, "Value"])
    finally:
        workbook.close()
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
