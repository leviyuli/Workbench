"""Current-dependent HFR stays consistent across LSV fitting and saved output."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from workbench.core.io import write_xlsx
from workbench.tools.lsv.tool import LsvTool


_POINTS = [[0.8, 0.132], [0.05, 0.102], [0.2, 0.108], [0.4, 0.116], [0.6, 0.124]]
_SORTED_POINTS = np.array(sorted(_POINTS))
_HFR_COLUMN = "HFR (Ω·cm²)"


def _resistance(current):
    return 0.1 + 0.04 * np.abs(current)


def _source(tmp_path, mode="OER", include_rcl=False, area=None, variable=True):
    current = np.unique(np.r_[np.logspace(-3, 0, 500), 0.01, 0.1, 0.3, 0.5])
    kinetic = 0.06 * np.log10(current / 2e-6)
    catalyst = (
        0.06 / 1.1982 * np.log10(
            1 + (current * np.log(10) * 0.2 / (2 * 0.06)) ** 1.1982,
        ) if include_rcl else np.zeros_like(current)
    )
    hfr = _resistance(current) if variable else np.full_like(current, 0.15)
    loss = kinetic + current * hfr + catalyst
    is_cell = mode.startswith("Cell")
    current_sign = -1 if mode in ("HER", "Cell positive") else 1
    voltage_sign = -1 if mode in ("HER", "Cell negative") else 1
    reversible = 0.0 if mode == "HER" else 1.2291
    voltage = voltage_sign * (reversible + loss)
    current_column = "I(A)" if area is not None else "j(A/cm2)"
    imported_current = current_sign * current * (area if area is not None else 1)
    path = tmp_path / "lsv.csv"
    pd.DataFrame({"E(V)": voltage, current_column: imported_current}).to_csv(path, index=False)
    params = {spec.key: spec.default for spec in LsvTool().parameter_schema()}
    params.update(
        electrode_setup="Two-electrode" if is_cell else "Three-electrode",
        reaction="HER" if current_sign < 0 else "OER", reference="RHE",
        temperature=298.15, ph=0.0, hfr_mode="HFR vs |j|" if variable else "Constant HFR",
        hfr=0.9 if variable else 0.15, include_rcl=include_rcl, rcl=0.2,
        i_lower=0.005, i_upper=0.1, target_i=0.3, x_log=True,
        swap_axes=voltage_sign < 0,
        electrode_area=area if area is not None else 0.0,
    )
    return path, params, voltage_sign


def _workbook(tool, result):
    return next(item.payload for item in tool.exportable_artifacts(result) if item.kind == "xlsx")


@pytest.mark.parametrize("mode", ["OER", "HER", "Cell positive", "Cell negative"])
@pytest.mark.parametrize("include_rcl", [False, True])
def test_variable_hfr_corrects_fit_curves_and_target_with_native_polarity(
    tmp_path, mode, include_rcl,
):
    path, params, voltage_sign = _source(tmp_path, mode, include_rcl)
    extra = {"hfr_points": deepcopy(_POINTS)}
    before = deepcopy((params, extra))
    tool = LsvTool()
    result = tool.run([path], params, extra)

    assert (params, extra) == before
    assert result.scalars["b_kin (V/dec)"] == pytest.approx(0.06, rel=2e-4)
    assert result.scalars["i0 (A/cm²)"] == pytest.approx(2e-6, rel=2e-3, abs=0)
    curves = result.tables["Curves (stacked)"]
    components = result.tables["Overpotential components"]
    current = np.abs(curves["Current Density (A/cm²)"].to_numpy())
    expected_hfr = _resistance(current)
    np.testing.assert_allclose(components[_HFR_COLUMN], expected_hfr)
    np.testing.assert_allclose(components["η_ohm (V)"], voltage_sign * current * expected_hfr)
    np.testing.assert_allclose(curves["+ η_ohm"] - curves["E_rev + η_kin"],
                               components["η_ohm (V)"])
    assert np.any(current < _SORTED_POINTS[0, 0])
    assert np.any(current > _SORTED_POINTS[-1, 0])
    np.testing.assert_allclose(result.tables["HFR points"], _SORTED_POINTS)
    assert list(result.tables["HFR points"]) == ["|j| (A/cm²)", _HFR_COLUMN]

    lines = result.figures["Decoupled curves"].axes[0].lines
    for column, line in zip(curves.columns[1:], lines, strict=True):
        plotted_voltage = line.get_xdata() if params["swap_axes"] else line.get_ydata()
        plotted_current = line.get_ydata() if params["swap_axes"] else line.get_xdata()
        np.testing.assert_allclose(plotted_voltage, curves[column])
        np.testing.assert_allclose(plotted_current, current)
    # The spline reproduces the affine resistance, then evaluates j*R(j).
    breakdown = result.payload["export_breakdown"]
    assert breakdown["Ohmic (V)"] == pytest.approx(0.0336)
    assert "HFR fit" in result.figures
    assert not result.display_tables
    patches = result.figures["Overpotential breakdown"].axes[0].patches
    np.testing.assert_allclose([patch.get_width() for patch in patches],
                               list(breakdown.values())[1:], atol=1e-14)


def test_target_hfr_is_evaluated_between_sparse_measured_currents(tmp_path):
    path, params, _ = _source(tmp_path)
    frame = pd.read_csv(path)
    current = frame["j(A/cm2)"].to_numpy()
    frame.loc[(current <= 0.1) | np.isclose(current, 0.5) | np.isclose(current, 1.0)].to_csv(
        path, index=False,
    )
    result = LsvTool().run([path], params, {"hfr_points": _POINTS})
    # Interpolating measured jR between .1 and .5 gives .0352 V instead.
    assert result.payload["export_breakdown"]["Ohmic (V)"] == pytest.approx(0.0336)
    assert result.figures["Overpotential breakdown"].axes[0].patches[2].get_width() == pytest.approx(0.0336)


def test_variable_hfr_exports_points_and_evaluated_resistance_without_mutating_results(tmp_path):
    path, params, _ = _source(tmp_path, include_rcl=True)
    tool = LsvTool()
    result = tool.run([path], params, {"hfr_points": _POINTS})
    originals = {name: table.copy(deep=True) for name, table in result.tables.items()}
    sheets = _workbook(tool, result)
    data = sheets["Decoupled curves"]
    np.testing.assert_allclose(data[_HFR_COLUMN], _resistance(data.iloc[:, 0]))
    summary = sheets["Summary"]
    points = summary.loc[summary.Section == "HFR points"].pivot(
        index="Record", columns="Parameter", values="Value",
    )
    np.testing.assert_allclose(points[result.tables["HFR points"].columns].to_numpy(dtype=float),
                               result.tables["HFR points"].to_numpy())
    settings = summary.loc[summary.Section == "Fit results"].set_index("Parameter")["Value"]
    assert settings["HFR mode"] == "HFR vs |j|"
    assert settings["HFR at target (Ω·cm²)"] == pytest.approx(0.112)
    assert _HFR_COLUMN not in settings.index
    target = tmp_path / "variable_hfr.xlsx"
    write_xlsx(sheets, target)
    for name, expected in sheets.items():
        pd.testing.assert_frame_equal(
            pd.read_excel(target, sheet_name=name, keep_default_na=False), expected,
            check_dtype=False, rtol=1e-12, atol=1e-14,
        )
    for name, table in originals.items():
        pd.testing.assert_frame_equal(result.tables[name], table)


def test_variable_hfr_uses_normalized_current_density_for_absolute_current_data(tmp_path):
    path, params, sign = _source(tmp_path, mode="HER", area=2.5)
    result = LsvTool().run([path], params, {"hfr_points": _POINTS})
    components = result.tables["Overpotential components"]
    current = components["Current Density (A/cm²)"].to_numpy()
    assert np.max(np.abs(current)) == pytest.approx(1.0)
    np.testing.assert_allclose(components[_HFR_COLUMN], _resistance(current))
    np.testing.assert_allclose(components["η_ohm (V)"], sign * np.abs(current) * _resistance(current))
    assert result.scalars["b_kin (V/dec)"] == pytest.approx(0.06)
    assert result.payload["export_breakdown"]["Ohmic (V)"] == pytest.approx(0.0336)


def test_legacy_params_use_constant_hfr_and_ignore_inactive_points(tmp_path):
    path, params, _ = _source(tmp_path, variable=False)
    tool = LsvTool()
    explicit = tool.run([path], params, {"hfr_points": [["unfinished", ""]]})
    params.pop("hfr_mode")
    legacy = tool.run([path], params)
    for name, table in explicit.tables.items():
        pd.testing.assert_frame_equal(legacy.tables[name], table)
    assert "HFR points" not in legacy.tables
    assert _HFR_COLUMN not in legacy.tables["Overpotential components"]
    assert legacy.scalars["b_kin (V/dec)"] == pytest.approx(0.06)
    workbook = _workbook(tool, legacy)
    assert list(workbook) == ["Summary", "Decoupled curves"]
    assert _HFR_COLUMN not in workbook["Decoupled curves"]
    summary = workbook["Summary"]
    assert summary.loc[summary.Section == "Fit results"].set_index("Parameter").loc[_HFR_COLUMN, "Value"] == 0.15


@pytest.mark.parametrize("extra", [None, {}, {"hfr_points": []}])
def test_variable_hfr_requires_points_instead_of_falling_back_to_constant(tmp_path, extra):
    path, params, _ = _source(tmp_path)
    with pytest.raises(ValueError):
        LsvTool().run([path], params, extra)


@pytest.fixture(scope="module")
def qt_app():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("QT_QPA_PLATFORM", "offscreen")
            app = QApplication([])
    yield app


def test_lsv_page_restores_variable_points_and_legacy_constant_state(qt_app, tmp_path):
    from workbench.app import ToolPage

    original = ToolPage(LsvTool(), lambda _: None)
    restored = ToolPage(LsvTool(), lambda _: None)
    legacy = ToolPage(LsvTool(), lambda _: None)
    try:
        original.panel.set_values({"hfr_mode": "HFR vs |j|", "hfr": 0.27})
        rows = [["0.5", "0.12"], ["0.1", "0.1"], ["1e-", ""]]
        original.extra.set_rows(rows)
        state = original.save_state()
        restored.restore_state(state, tmp_path)
        assert restored.panel.values()["hfr_mode"] == "HFR vs |j|"
        assert restored.extra.rows() == rows
        assert restored.extra.values()["hfr_text"] == "\n".join("\t".join(row) for row in rows)
        assert not restored.extra.isHidden()
        old_state = deepcopy(state)
        old_state["panel"]["values"].pop("hfr_mode")
        old_state["extra"] = None
        legacy.restore_state(old_state, tmp_path)
        assert legacy.panel.values()["hfr_mode"] == "Constant HFR"
        assert legacy.panel.values()["hfr"] == pytest.approx(0.27)
        assert legacy.extra.values()["hfr_text"] == ""
        assert legacy.extra.isHidden()
    finally:
        original.close()
        restored.close()
        legacy.close()
