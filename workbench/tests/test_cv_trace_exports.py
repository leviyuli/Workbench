"""CV and time-trace workbooks contain one usable representation of results."""

from io import StringIO

import numpy as np
import pandas as pd
import pytest

from workbench.tools.ecsa.tool import EcsaTool
from workbench.tools.it_integrate.tool import ItIntegrateTool
from workbench.widgets.export_button import ExportButton


def _ecsa_sources(folder):
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for rate, replicate, points in ((20, 1, 11), (40, 1, 9), (40, 2, 13), (80, 1, 15)):
        potential = np.r_[np.linspace(0.0, 0.2, points), np.linspace(0.2, 0.0, points)[1:]]
        j_dl = 0.01 * rate * 1e-3 * (1 + 0.05 * replicate)
        current = np.r_[np.full(points, j_dl), np.full(points - 1, -j_dl)]
        path = folder / f"CV_{rate}mV_{replicate}.txt"
        pd.DataFrame({
            "E(V)": np.r_[potential, potential],
            "j(mA/cm2)": np.r_[current, current * 1.2] * 1e3,
            "t(s)": np.arange(len(potential) * 2) * 0.125,
        }).to_csv(path, sep="\t", index=False)
        paths.append(path)
    return paths


def _ecsa_params(**updates):
    return {
        "mode": "ECSA calculation", "electrode_setup": "Three-electrode",
        "target_scale": "RHE", "reference": "Custom E_ref vs SHE",
        "custom_e_ref_vs_she": 0.25, "ph": 0.0, "Cs": 35.0,
        **updates,
    }


def _assert_plot_pair(frame, offset, line):
    size = len(line.get_xdata())
    np.testing.assert_allclose(frame.iloc[:size, offset], line.get_xdata(), equal_nan=True)
    np.testing.assert_allclose(frame.iloc[:size, offset + 1], line.get_ydata(), equal_nan=True)
    assert frame.iloc[size:, offset:offset + 2].isna().all().all()


@pytest.mark.parametrize("replicate_mode,expected_curves", [("last", 3), ("all", 4)])
def test_ecsa_workbook_keeps_final_fit_and_selected_cv_curves(tmp_path, replicate_mode, expected_curves):
    tool = EcsaTool()
    result = tool.run(_ecsa_sources(tmp_path / "inputs"),
                      _ecsa_params(replicate_mode=replicate_mode))
    assert set(result.tables) == {"Summary", "ECSA_result", "CV_long", "CV_time_series"}
    original = {name: frame.copy(deep=True) for name, frame in result.tables.items()}
    artifacts = tool.exportable_artifacts(result)
    workbook = next(item for item in artifacts if item.kind == "xlsx")
    assert list(workbook.payload) == ["Results", "ECSA fit", "CV overlay"]
    results = workbook.payload["Results"].set_index("Quantity")["Value"]
    assert results["Specific capacitance (µF/cm²)"] == 35.0
    assert "C_dl SE (mF/cm²)" in results and "ECSA SE (cm²)" in results
    assert not any(" SD " in quantity for quantity in (*results.index, *result.scalars))
    for quantity in results.index.difference(["Specific capacitance (µF/cm²)"]):
        assert results[quantity] == pytest.approx(result.scalars[quantity], nan_ok=True)
    assert len(results) == 8

    fit = workbook.payload["ECSA fit"]
    assert fit.columns.tolist() == [
        "Scan rate (V/s)", "j_dl (mA/cm²)", "Fit scan rate (V/s)", "Fit j_dl (mA/cm²)",
    ]
    points = fit[["Scan rate (V/s)", "j_dl (mA/cm²)"]].dropna()
    _, covariance = np.polyfit(points.iloc[:, 0], points.iloc[:, 1], 1, cov=True)
    slope_se = np.sqrt(covariance[0, 0])
    assert results["C_dl SE (mF/cm²)"] == pytest.approx(slope_se)
    assert results["ECSA SE (cm²)"] == pytest.approx(slope_se * 1e-3 / 35e-6)
    for offset, line in zip((0, 2), result.figures["ECSA fit"].axes[0].lines):
        _assert_plot_pair(fit, offset, line)
    assert len(fit) == 100
    overlay = workbook.payload["CV overlay"]
    assert overlay.shape[1] == expected_curves * 2
    for index, line in enumerate(result.figures["CV overlay"].axes[0].lines):
        _assert_plot_pair(overlay, index * 2, line)
    assert all("V vs RHE" in column for column in overlay.columns[::2])
    assert all("mA/cm²" in column for column in overlay.columns[1::2])
    assert all("t(s)" not in column for column in overlay.columns)
    assert any("CV_40mV_1" in column for column in overlay.columns) == (replicate_mode == "all")
    assert any("CV_40mV_2" in column for column in overlay.columns)

    target = tmp_path / workbook.default_filename
    ExportButton._write(workbook, str(target))
    assert pd.ExcelFile(target).sheet_names == list(workbook.payload)
    for name, expected in workbook.payload.items():
        written = pd.read_excel(target, sheet_name=name)
        pd.testing.assert_frame_equal(written, expected, check_dtype=False, rtol=1e-14, atol=1e-14)
    for name, expected in original.items():
        pd.testing.assert_frame_equal(result.tables[name], expected)
    assert [item.kind for item in artifacts] == ["xlsx", "png", "png"]


def test_split_workbook_keeps_cycle_samples_without_settings_sheet(tmp_path):
    path = tmp_path / "measurement.txt"
    path.write_text(
        "E(mV),i(mA),t(s)\n0,1,0\n100,2,1\n0,-1,2\n50,3,3\n100,4,4\n50,-2,5\n0,-3,6\n",
        encoding="utf-8",
    )
    tool = EcsaTool()
    result = tool.run([path], {
        "mode": "Split cycles", "electrode_setup": "Three-electrode",
        "target_scale": "SHE", "reference": "Custom E_ref vs SHE",
        "custom_e_ref_vs_she": 0.2,
    })
    workbook = next(item for item in tool.exportable_artifacts(result) if item.kind == "xlsx")
    assert list(workbook.payload) == ["measurement cycles"]
    pd.testing.assert_frame_equal(workbook.payload["measurement cycles"], result.tables["measurement cycles"])
    target = tmp_path / workbook.default_filename
    ExportButton._write(workbook, str(target))
    written = pd.read_excel(target, sheet_name="measurement cycles")
    for offset, line in zip(range(0, written.shape[1], 2), next(iter(result.figures.values())).axes[0].lines):
        _assert_plot_pair(written, offset, line)


@pytest.mark.parametrize("trace_type", ["i_t", "E_t"])
@pytest.mark.parametrize("designate", [False, True])
def test_trace_workbook_retains_time_and_all_measured_signals(tmp_path, trace_type, designate):
    first, second = tmp_path / "first.txt", tmp_path / "second.txt"
    first.write_text("T(s)\tE(V)\ti(A)\n0\t0.1\t0.002\n2\t0.3\t0.004\n", encoding="utf-8")
    second.write_text("i(A)\tT(s)\tE(V)\n0.005\t0\t0.4\n0.008\t3\t0.7\n", encoding="utf-8")
    tool = ItIntegrateTool()
    result = tool.run([first, second], {
        "trace_type": trace_type, "interval": "1 s", "designate_sources": designate,
        "electrode_setup": "Three-electrode", "target_scale": "RHE",
        "reference": "Custom E_ref vs SHE", "custom_e_ref_vs_she": 0.25, "ph": 0.0,
    })
    artifacts = tool.exportable_artifacts(result)
    workbook = next(item for item in artifacts if item.kind == "xlsx")
    assert list(workbook.payload) == ["Integrated"]
    expected = result.tables["Integrated"]
    pd.testing.assert_frame_equal(workbook.payload["Integrated"], expected)
    assert expected.shape[1] == (6 if designate else 3)
    target = tmp_path / workbook.default_filename
    ExportButton._write(workbook, str(target))
    written = pd.read_excel(target)
    pd.testing.assert_frame_equal(written, expected, check_dtype=False)
    text = next(item for item in artifacts if item.kind == "txt")
    pd.testing.assert_frame_equal(pd.read_csv(StringIO(text.payload), sep="\t"), expected, check_dtype=False)
    primary = "i(A)" if trace_type == "i_t" else "E(V)[vs_RHE]"
    axes = next(iter(result.figures.values())).axes[0]
    groups = ["first.txt · ", "second.txt · "] if designate else [""]
    for prefix, line in zip(groups, axes.lines):
        values = written[[f"{prefix}T(s)", f"{prefix}{primary}"]].dropna()
        np.testing.assert_allclose(values.iloc[:, 0], line.get_xdata())
        np.testing.assert_allclose(values.iloc[:, 1], line.get_ydata())
