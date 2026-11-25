"""Optional CCD on the fitted Decoupling curve preserves its physical analysis."""
from copy import deepcopy
from io import BytesIO

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication

from workbench.app import ToolPage
from workbench.core.io import write_xlsx
from workbench.core.project import load_project, save_project
from workbench.tools.ccd.tool import CcdTool
from workbench.tools.lsv.tool import LsvTool


def _params(**changes):
    params = {spec.key: spec.default for spec in LsvTool().parameter_schema()}
    params.update(electrode_setup="Three-electrode", reference="RHE", reaction="OER",
                  temperature=298.15, ph=0., hfr=.12, include_rcl=False,
                  i_lower=.005, i_upper=.3, target_i=1., include_ccd=True,
                  ccd_fit_mode="Manual", ccd_j_lower=.8, ccd_j_upper=2.5,
                  ccd_threshold_mv=50.)
    params.update(changes)
    return params


def _write_curve(tmp_path, *, reaction="OER", area=None, rise=True, name="curve.csv"):
    current = np.unique(np.r_[np.geomspace(.001, .1, 301), np.linspace(.1, 6., 4001)])
    residual = .08 * np.maximum(current - .3, 0)
    if rise:
        residual += .06 * np.maximum(current - 3., 0) ** 2
    loss = .06 * np.log10(current / 2e-6) + .12 * current + residual
    potential = -loss if reaction == "HER" else 1.2291 + loss
    signed = -current if reaction == "HER" else current
    column = "j(A/cm2)" if area is None else "Current (A)"
    path = tmp_path / name
    pd.DataFrame({"E(V)": potential, column: signed if area is None else area * signed}).to_csv(
        path, index=False,
    )
    return path


def _assert_ccd_summary(summary, tables):
    results = summary.loc[summary["Section"].eq("CCD results")]
    assert (results["Record"].isna() | results["Record"].eq("")).all()
    pd.testing.assert_series_equal(results.set_index("Parameter")["Value"],
                                   tables["CCD results"].iloc[0],
                                   check_names=False, check_dtype=False, check_index_type=False,
                                   rtol=1e-12, atol=1e-14)
    settings = summary.loc[summary["Section"].eq("CCD settings")]
    assert (settings["Record"].isna() | settings["Record"].eq("")).all()
    pd.testing.assert_frame_equal(settings[["Parameter", "Value"]].reset_index(drop=True),
                                  tables["CCD settings"], check_dtype=False,
                                  rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("reaction", ["OER", "HER"])
def test_single_decoupling_ccd_recovers_known_onset_with_native_voltage_polarity(tmp_path, reaction):
    source = _write_curve(tmp_path, reaction=reaction)
    result = LsvTool().run([source], _params(reaction=reaction))
    expected = 3. + np.sqrt(.05 / .06)
    summary = result.tables["CCD results"].iloc[0]
    assert len(result.tables["CCD results"]) == 1
    assert summary["CCD (A/cm²)"] == pytest.approx(expected, abs=.04)
    assert summary["CCD status"] == "Detected"
    assert summary["Residual baseline status"] == "Resolved"
    assert result.scalars["CCD (A/cm²)"] == summary["CCD (A/cm²)"]
    assert result.scalars["CCD status"] == "Detected"
    assert result.scalars["CCD curve"] == "LSV"
    assert result.scalars["CCD persistence (V)"] == .10
    assert np.sign(summary["CCD potential (V)"]) == (-1 if reaction == "HER" else 1)
    assert "CCD analysis" in result.figures
    processed = result.tables["CCD data"]
    onset = processed.loc[processed["CCD point"]]
    assert len(onset) == 1
    assert onset.iloc[0]["Current density magnitude (A/cm²)"] == summary["CCD (A/cm²)"]
    assert onset.iloc[0]["Signed residual (V)"] == summary["CCD residual (V)"]
    assert onset.iloc[0]["Excess above baseline (V)"] >= .05
    assert processed["Baseline fit"].any()
    np.testing.assert_allclose(processed["Excess above baseline (V)"],
                               processed["Signed residual (V)"] - processed["Residual baseline (V)"],
                               equal_nan=True)


def test_absolute_current_is_normalized_before_ccd_detection(tmp_path):
    density = _write_curve(tmp_path, name="density.csv")
    absolute = _write_curve(tmp_path, area=2.5, name="absolute.csv")
    tool = LsvTool()
    reference = tool.run([density], _params())
    normalized = tool.run([absolute], _params(electrode_area=2.5))
    assert normalized.scalars["CCD (A/cm²)"] == pytest.approx(reference.scalars["CCD (A/cm²)"])
    for column in ("Current density magnitude (A/cm²)", "Signed residual (V)",
                   "Residual baseline (V)", "Excess above baseline (V)"):
        np.testing.assert_allclose(normalized.tables["CCD data"][column],
                                   reference.tables["CCD data"][column], atol=1e-12)


def test_optional_ccd_preserves_existing_decomposition_and_default_exports(tmp_path):
    source = _write_curve(tmp_path)
    tool = LsvTool()
    assert {spec.key: spec.default for spec in tool.parameter_schema()}["include_ccd"] is False
    disabled_params = _params()
    disabled_params.pop("include_ccd")
    disabled = tool.run([source], disabled_params)
    enabled = tool.run([source], _params())
    assert not {"CCD results", "CCD data", "CCD settings"} & disabled.tables.keys()
    assert "CCD analysis" not in disabled.figures
    assert "CCD (A/cm²)" not in disabled.scalars
    for name, frame in disabled.tables.items():
        pd.testing.assert_frame_equal(enabled.tables[name], frame)
    for name, value in disabled.scalars.items():
        assert enabled.scalars[name] == value
    workbook = next(artifact.payload for artifact in tool.exportable_artifacts(disabled)
                    if artifact.kind == "xlsx")
    assert list(workbook) == ["Summary", "Decoupled curves"]
    assert not workbook["Summary"]["Section"].isin(["CCD results", "CCD settings"]).any()


def test_threshold_changes_ccd_without_refitting_kinetics_or_residual_baseline(tmp_path):
    source = _write_curve(tmp_path)
    tool = LsvTool()
    low = tool.run([source], _params(ccd_threshold_mv=20.))
    high = tool.run([source], _params(ccd_threshold_mv=100.))
    assert low.scalars["CCD (A/cm²)"] == pytest.approx(3. + np.sqrt(.020 / .06), abs=.04)
    assert high.scalars["CCD (A/cm²)"] == pytest.approx(3. + np.sqrt(.100 / .06), abs=.04)
    assert high.scalars["CCD (A/cm²)"] > low.scalars["CCD (A/cm²)"]
    pd.testing.assert_frame_equal(low.tables["Fitting info"], high.tables["Fitting info"])
    pd.testing.assert_frame_equal(low.tables["Overpotential components"],
                                  high.tables["Overpotential components"])
    for column in ("Signed residual (V)", "Residual baseline (V)", "Baseline fit"):
        pd.testing.assert_series_equal(low.tables["CCD data"][column], high.tables["CCD data"][column])
    assert high.scalars["CCD threshold (mV)"] == 100.


@pytest.mark.parametrize("mode,status", [("Auto", "Baseline unresolved"),
                                        ("Manual", "Not reached within measured range")])
def test_unresolved_ccd_keeps_decoupling_results_without_fabricating_onset(tmp_path, mode, status):
    source = _write_curve(tmp_path, rise=False)
    params = _params(ccd_fit_mode=mode)
    if mode == "Auto":
        # Every measured current is below the configured Tafel upper bound,
        # so no automatic residual-baseline candidate can exist above it.
        params["i_upper"] = 7.
    result = LsvTool().run([source], params)
    assert "Decoupled curves" in result.figures
    assert not result.tables["Overpotential components"].empty
    assert result.scalars["CCD status"] == status
    assert pd.isna(result.tables["CCD results"].iloc[0]["CCD (A/cm²)"])
    assert not result.tables["CCD data"]["CCD point"].any()


def test_multiple_selected_files_label_ccd_as_the_averaged_lsv(tmp_path):
    sources = [_write_curve(tmp_path, name=f"replicate_{number}.csv") for number in (1, 2)]
    result = LsvTool().run(sources, _params())
    assert result.scalars["CCD curve"] == "Averaged LSV"
    assert result.scalars["CCD (A/cm²)"] == pytest.approx(3. + np.sqrt(.05 / .06), abs=.04)


def test_ccd_workbook_retains_processed_onset_baseline_and_detection_settings(tmp_path):
    source = _write_curve(tmp_path)
    tool = LsvTool()
    result = tool.run([source], _params(ccd_threshold_mv=75.5))
    workbook = next(artifact.payload for artifact in tool.exportable_artifacts(result)
                    if artifact.kind == "xlsx")
    target = tmp_path / "single_lsv_ccd.xlsx"
    write_xlsx(workbook, target)
    restored = pd.read_excel(target, sheet_name=None)
    assert list(workbook) == list(restored) == ["Summary", "Decoupled curves", "CCD data"]
    _assert_ccd_summary(workbook["Summary"], result.tables)
    _assert_ccd_summary(restored["Summary"], result.tables)
    pd.testing.assert_frame_equal(workbook["CCD data"], result.tables["CCD data"])
    pd.testing.assert_frame_equal(restored["CCD data"], result.tables["CCD data"], check_dtype=False,
                                  rtol=1e-12, atol=1e-14)
    assert set(result.tables["CCD settings"].columns) == {"Parameter", "Value"}
    assert result.scalars["CCD threshold (mV)"] == 75.5
    assert restored["CCD data"].loc[restored["CCD data"]["CCD point"],
                                      "Excess above baseline (V)"].iloc[0] >= .0755


def test_batch_schema_keeps_one_copy_of_each_ccd_control():
    specs = CcdTool().parameter_schema()
    keys = [spec.key for spec in specs]
    assert len(keys) == len(set(keys))
    assert "include_ccd" not in keys
    assert {"ccd_fit_mode", "ccd_j_lower", "ccd_j_upper", "ccd_threshold_mv"} <= set(keys)


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance()
    if app is None:
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("QT_QPA_PLATFORM", "offscreen")
            app = QApplication([])
    yield app


def test_ccd_controls_are_conditional_and_survive_page_restore(qt_app, tmp_path):
    original = ToolPage(LsvTool(), lambda _: None)
    restored = ToolPage(LsvTool(), lambda _: None)
    legacy = ToolPage(LsvTool(), lambda _: None)
    try:
        assert original.panel.values()["include_ccd"] is False
        for key in ("ccd_fit_mode", "ccd_threshold_mv", "ccd_j_lower", "ccd_j_upper"):
            assert original.panel._rows[key].isHidden()
        original.panel.set_values({"include_ccd": True})
        assert not original.panel._rows["ccd_fit_mode"].isHidden()
        assert not original.panel._rows["ccd_threshold_mv"].isHidden()
        assert original.panel._rows["ccd_j_lower"].isHidden()
        original.panel.set_values({"ccd_fit_mode": "Manual", "ccd_j_lower": .8,
                                   "ccd_j_upper": 2.5, "ccd_threshold_mv": 75.5})
        assert not original.panel._rows["ccd_j_lower"].isHidden()
        assert not original.panel._rows["ccd_j_upper"].isHidden()
        state = original.save_state()
        restored.restore_state(state, tmp_path)
        assert restored.panel.values() == original.panel.values()
        assert not restored.panel._rows["ccd_j_lower"].isHidden()
        original.panel.set_values({"include_ccd": False})
        assert original.panel._rows["ccd_j_lower"].isHidden()
        original.panel.set_values({"include_ccd": True})
        assert original.panel.values()["ccd_j_lower"] == .8
        assert original.panel.values()["ccd_threshold_mv"] == 75.5

        historical = deepcopy(state)
        for key in ("include_ccd", "ccd_fit_mode", "ccd_j_lower", "ccd_j_upper", "ccd_threshold_mv"):
            historical["panel"]["values"].pop(key)
        legacy.restore_state(historical, tmp_path)
        assert legacy.panel.values()["include_ccd"] is False
        assert legacy.panel._rows["ccd_fit_mode"].isHidden()
    finally:
        for page in (original, restored, legacy):
            page.deleteLater()
        qt_app.processEvents()


def test_ccd_result_and_exports_roundtrip_with_embedded_source(qt_app, tmp_path, monkeypatch):
    # Keep the portable project's temporary extraction directory in the test's
    # sandbox, independently of the host account's temporary-directory access.
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    source = _write_curve(tmp_path)
    original = ToolPage(LsvTool(), lambda _: None)
    restored = ToolPage(LsvTool(), lambda _: None)
    try:
        params = _params(ccd_threshold_mv=75.5)
        original.file_input.set_files([source])
        original.panel.set_values(params)
        original.last_result = original.tool.run([source], params)
        original.last_artifacts = original.tool.exportable_artifacts(original.last_result)
        original.results.set_result(original.last_result)
        expected = original.last_result
        project = tmp_path / "single_lsv_ccd.ecwb"
        save_project(project, {"page": original.save_state()})
        source.unlink()
        with load_project(project) as loaded:
            restored.restore_state(loaded.state["page"], tmp_path)
            assert restored.file_input.files()[0].is_file()
            assert restored.panel.values()["include_ccd"] is True
            assert restored.panel.values()["ccd_threshold_mv"] == 75.5
            assert restored.last_result.scalars["CCD (A/cm²)"] == expected.scalars["CCD (A/cm²)"]
            assert "CCD analysis" in restored.last_result.figure_images
            for name in ("CCD results", "CCD data", "CCD settings"):
                pd.testing.assert_frame_equal(restored.last_result.tables[name], expected.tables[name])
            artifact = next(item for item in restored.last_artifacts if item.kind == "xlsx")
            workbook = pd.read_excel(BytesIO(artifact.serialized_payload), sheet_name=None)
            assert list(workbook) == ["Summary", "Decoupled curves", "CCD data"]
            _assert_ccd_summary(workbook["Summary"], expected.tables)
            pd.testing.assert_frame_equal(workbook["CCD data"], expected.tables["CCD data"],
                                          check_dtype=False, rtol=1e-12, atol=1e-14)
    finally:
        original.deleteLater()
        restored.deleteLater()
        qt_app.processEvents()
