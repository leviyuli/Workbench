"""Batch decomposition, plot exports and status handling share one numerical record."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication, QTableWidget

from workbench.tools.ccd.batch_inputs import read_batch_scans
from workbench.tools.ccd.tool import CCD_COLUMN, LIMIT_COLUMN, CcdTool
from workbench.tools.lsv.analysis import DecouplingSettings, decouple_curve
from workbench.tools.lsv.logic_tafel import calculate_RCL_overpotential
from workbench.tools.lsv.tool import LsvTool
from workbench.widgets.parameter_panel import ParameterPanel
from workbench.widgets.results_panel import ResultsPanel


def _params(**changes):
    params = {spec.key: spec.default for spec in CcdTool().parameter_schema()}
    params.update(electrode_setup="Three-electrode", reference="RHE", reaction="OER",
                  temperature=298.15, ph=0., hfr=.12, include_rcl=False,
                  i_lower=.005, i_upper=.3, target_i=1., ccd_fit_mode="Manual",
                  ccd_j_lower=.8, ccd_j_upper=2.5)
    params.update(changes)
    return params


def _curve(*, slope=.06, onset=3., her=False, include_rcl=False, profile=False, maximum=6.):
    current = np.unique(np.r_[np.logspace(-3, -1, 300), np.linspace(.1, maximum, 4001), 1.])
    current = current[current <= maximum]
    kinetic = slope * np.log10(current / 2e-6)
    resistance = .1 + .02 * current if profile else .12
    catalyst = calculate_RCL_overpotential(current, .2, slope) if include_rcl else 0.
    residual = .08 * np.maximum(current - .3, 0) + .06 * np.maximum(current - onset, 0) ** 2
    sign = -1 if her else 1
    potential = sign * ((0. if her else 1.2291) + kinetic + resistance * current + catalyst + residual)
    return potential, sign * current


def _lsv(path, *, area=None, **options):
    potential, current = _curve(**options)
    column = "j(A/cm2)"
    if area is not None:
        column = "I(mA)"
        current = current * area * 1000
    pd.DataFrame({"E(V)": potential, column: current}).to_csv(path, index=False)
    return path


def _cv(path, cycles=3):
    potential, current = _curve()
    blocks = []
    for cycle in range(1, cycles + 1):
        scale = .9 + .1 * cycle
        blocks.append(pd.DataFrame({
            "Cycle": cycle, "E(V)": np.r_[potential, potential[-2::-1]],
            "j(A/cm2)": np.r_[scale * current, .97 * scale * current[-2::-1]],
        }))
    pd.concat(blocks, ignore_index=True).to_csv(path, index=False)
    return path


def _assert_components_match_engine(result, scan, params, extra=None):
    current = scan.current / params["electrode_area"] if not scan.is_density else scan.current
    engine = decouple_curve(scan.potential, current, DecouplingSettings.from_params(params, extra),
                            reaction="auto", source_indices=scan.source_indices,
                            tafel_method="contiguous-stable")
    complete = result.tables["Complete decomposition"]
    actual = complete.loc[complete.Curve.eq(scan.code)]
    np.testing.assert_array_equal(actual["Source row"], scan.source_indices)
    for field, expected in [
        ("Measured potential (V)", engine.measured_potential),
        ("Reference-converted potential (V)", engine.potential),
        ("Signed current density (A/cm²)", engine.current),
        ("Kinetic overpotential (V)", engine.eta_kin),
        ("Ohmic overpotential (V)", engine.eta_ohm),
        ("RCL overpotential (V)", engine.eta_rcl), ("Signed residual (V)", engine.residual),
    ]:
        np.testing.assert_allclose(actual[field], expected, equal_nan=True, atol=1e-12)
    return engine


def test_lsvs_are_independent_and_match_individual_decoupling(tmp_path):
    first = _lsv(tmp_path / "first.csv")
    second = _lsv(tmp_path / "second.csv", slope=.08, onset=3.5)
    params = _params()
    tool = CcdTool()
    result = tool.run([first, second], params)
    rows = result.tables["Current results"]
    assert rows["Decoupling status"].eq("Resolved").all()
    assert rows[CCD_COLUMN].notna().all()
    assert rows[LIMIT_COLUMN].isna().all()
    assert rows.loc[1, CCD_COLUMN] > rows.loc[0, CCD_COLUMN]
    for scan in read_batch_scans([first, second], params):
        engine = _assert_components_match_engine(result, scan, params)
        fit = result.tables["Fit diagnostics"].set_index("Curve").loc[scan.code]
        assert fit["Tafel b (V/dec)"] == pytest.approx(engine.fit["b_kin"])
        assert fit["Tafel b (V/dec)"] == pytest.approx(.06 if scan.path == first else .08)
    averaged = LsvTool().run([first, second], params)
    assert not np.isclose(averaged.scalars["b_kin (V/dec)"],
                          result.tables["Fit diagnostics"].iloc[0]["Tafel b (V/dec)"])


@pytest.mark.parametrize("mode,cycle,branch,expected", [
    ("All", 1, "Forward", [1, 2, 3]), ("Last", 1, "Back", [3]),
    ("Specific", 2, "Forward", [2]), ("Specific", 2, "Back", [2]),
])
def test_cv_cycle_selection_preserves_native_rows_and_each_scan_fit(tmp_path, mode, cycle, branch, expected):
    path = _cv(tmp_path / "cycles.csv")
    params = _params(input_type="CV", cv_cycle_mode=mode, cv_cycle=cycle, cv_scan=branch)
    result = CcdTool().run([path], params)
    assert result.tables["Current results"].Cycle.tolist() == expected
    assert result.scalars["Scans decoupled"] == len(expected)
    for scan in read_batch_scans([path], params):
        _assert_components_match_engine(result, scan, params)
    raw = result.tables["Raw selected scans"]
    if mode != "All":
        assert raw.iloc[0]["Source row"] > 0
    if branch == "Back":
        assert np.all(np.diff(raw["Measured potential (V)"]) <= 0)
        assert result.tables["Fit diagnostics"]["Reversed toward increasing drive"].all()


def test_mixed_absolute_and_density_her_inputs_share_hfr_profile_and_optional_rcl(tmp_path):
    first = _lsv(tmp_path / "density.csv", her=True, include_rcl=True, profile=True)
    second = _lsv(tmp_path / "absolute.csv", her=True, include_rcl=True, profile=True, area=2.5)
    params = _params(reaction="HER", electrode_area=2.5, include_rcl=True, rcl=.2,
                     hfr_mode="HFR vs |j|")
    extra = {"hfr_points": [[.01, .1002], [.1, .102], [1., .12], [3., .16], [6., .22]]}
    result = CcdTool().run([first, second], params, extra)
    rows = result.tables["Current results"]
    assert rows["Decoupling status"].eq("Resolved").all()
    assert rows.loc[0, CCD_COLUMN] == pytest.approx(rows.loc[1, CCD_COLUMN])
    assert len(result.figures["Decoupling"].axes) == 4
    assert "Plot RCL" in result.tables
    for scan in read_batch_scans([first, second], params):
        _assert_components_match_engine(result, scan, params, extra)
    complete = result.tables["Complete decomposition"]
    assert complete["Signed current density (A/cm²)"].dropna().lt(0).all()
    assert result.tables["Breakdown results"]["Ohmic (V)"].nunique() == 1
    assert result.scalars["Common target ohmic drop (V)"] == pytest.approx(.12)
    assert not result.tables["Plot series"]["Plot sheet"].str.contains("Ohmic|HFR", case=False).any()


def test_failures_and_missing_target_points_do_not_discard_valid_scans(tmp_path):
    good = _lsv(tmp_path / "good.csv")
    short = _lsv(tmp_path / "short.csv", maximum=.4)
    bad = tmp_path / "bad.txt"
    bad.write_text("not a measurement\n", encoding="utf-8")
    result = CcdTool().run([good, short, bad], _params())
    rows = result.tables["Current results"].set_index("File")
    assert result.scalars["Scans decoupled"] == 2
    assert rows.loc[good.name, "Breakdown status"] == "Resolved"
    assert rows.loc[short.name, "Decoupling status"] == "Resolved"
    assert "outside" in rows.loc[short.name, "Breakdown status"]
    assert rows.loc[bad.name, "Decoupling status"] == "Analysis failed"
    assert pd.isna(rows.loc[bad.name, CCD_COLUMN])
    breakdown = result.tables["Breakdown results"].set_index("File")
    assert pd.isna(breakdown.loc[short.name, "Kinetic (V)"])
    all_bad = CcdTool().run([bad], _params())
    assert all_bad.scalars["Scans decoupled"] == 0
    assert all_bad.figures == {}
    assert all_bad.tables["Current results"]["Decoupling status"].eq("Analysis failed").all()
    assert [artifact.kind for artifact in CcdTool().exportable_artifacts(all_bad)] == ["xlsx"]


def test_plot_artists_and_exported_points_means_sd_have_identical_coordinates(tmp_path):
    path = _cv(tmp_path / "cycles.csv")
    result = CcdTool().run([path], _params(input_type="CV"))
    assert list(result.figures) == ["Decoupling", "CCD & limiting current",
                                    "Current comparison", "Breakdown comparison"]
    assert len(result.figures["Decoupling"].axes) == 3
    artists = {line.get_gid(): line for fig in result.figures.values() for ax in fig.axes
               for line in ax.lines if line.get_gid()}
    for row in result.tables["Plot series"].to_dict("records"):
        artist = artists.pop(row["Artist"])
        sheet = result.tables[row["Plot sheet"]]
        for column, values in [(row["X column"], artist.get_xdata()),
                               (row["Y column"], artist.get_ydata())]:
            actual = sheet[column].iloc[:len(values)].to_numpy(dtype=float)
            np.testing.assert_allclose(actual, values, equal_nan=True)
            assert sheet[column].iloc[len(values):].isna().all()
    assert not artists
    stats = result.tables["Current statistics"].set_index("Metric").loc[CCD_COLUMN]
    values = result.tables["Current results"][CCD_COLUMN].dropna().to_numpy()
    assert stats["Mean"] == pytest.approx(values.mean())
    assert stats["Sample SD"] == pytest.approx(values.std(ddof=1))
    index = result.tables["Plot series"]
    mean = index.loc[index.Series.eq("S01 · CCD mean")].iloc[0]
    np.testing.assert_allclose(result.tables[mean["Plot sheet"]][mean["Y column"]].dropna(),
                               [values.mean(), values.mean()])
    assert not index.Series.str.endswith("sample SD").any()
    artifacts = CcdTool().exportable_artifacts(result)
    assert [artifact.kind for artifact in artifacts] == ["xlsx", "png", "png", "png", "png"]
    sheets = artifacts[0].payload
    assert list(sheets)[:3] == ["Summary", "Current results", "Breakdown results"]
    assert not any(name.startswith("Plot ") for name in sheets)
    assert {"Curve data", "CCD data", "Fit diagnostics", "Summary"} <= set(sheets)
    settings = sheets["Summary"].loc[sheets["Summary"].Section == "Settings"].set_index("Parameter")["Value"]
    assert settings["Tafel method"] == result.scalars["Tafel method"]
    assert settings["ccd_threshold_mv"] == result.scalars["CCD threshold (mV)"]
    assert not {"Complete decomposition", "Processed scans", "Raw selected scans"} & set(sheets)
    assert sheets["Current results"].Curve.tolist() == result.tables["Current results"].Curve.tolist()
    assert not any("File position" in column for sheet in sheets.values() for column in sheet)


def test_resolved_limiting_plateaus_keep_markers_and_sample_spread(tmp_path):
    density = np.unique(np.r_[np.logspace(-3, -1, 300), np.linspace(.1, 4.99, 18001)])
    residual = .08 * np.maximum(density - .3, 0) + .1 * np.maximum(density - 3, 0) ** 2 / (5 - density)
    potential = 1.2291 + .06 * np.log10(density / 2e-6) + .12 * density + residual
    voltage = np.linspace(potential[0], 4.8, 6001)
    current = np.interp(voltage, potential, density)
    path = tmp_path / "plateaus.csv"
    pd.DataFrame({
        "Cycle 1 E(V)": voltage, "Cycle 1 j(A/cm2)": .9 * current,
        "Cycle 2 E(V)": voltage, "Cycle 2 j(A/cm2)": 1.1 * current,
    }).to_csv(path, index=False)
    result = CcdTool().run([path], _params(input_type="CV"))
    rows = result.tables["Current results"]
    values = rows[LIMIT_COLUMN].to_numpy(dtype=float)
    assert np.isfinite(values).all()
    assert rows["Limiting-current status"].str.startswith("Resolved plateau").all()
    assert values[1] > values[0]
    statistics = result.tables["Current statistics"].set_index("Metric").loc[LIMIT_COLUMN]
    assert statistics["Mean"] == pytest.approx(values.mean())
    assert statistics["Sample SD"] == pytest.approx(values.std(ddof=1))
    series = result.tables["Plot series"]
    markers = series.loc[series["Plot sheet"].eq("Plot CCD") & series.Marker.eq("^")]
    assert len(markers) == 2
    for marker, expected in zip(markers.to_dict("records"), values):
        exported = result.tables[marker["Plot sheet"]][marker["Y column"]].dropna()
        np.testing.assert_allclose(exported, [expected])
    assert series.Series.eq("S01 · Limiting current mean").any()
    assert not series.Series.str.endswith("sample SD").any()


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance()
    if app is None:
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("QT_QPA_PLATFORM", "offscreen")
            app = QApplication([])
    yield app


def test_compact_default_controls_and_results_hide_tables(qt_app, tmp_path):
    tool = CcdTool()
    specs = tool.parameter_schema()
    defaults = {spec.key: spec.default for spec in specs}
    assert defaults["include_rcl"] is False
    assert defaults["ccd_fit_mode"] == "Auto"
    assert defaults["cv_cycle_mode"] == "All"
    assert defaults["cv_scan"] == "Forward"
    assert defaults["ccd_threshold_mv"] == 50
    assert not {"smoothing_window", "bin_width_mv", "show_advanced"} & set(defaults)
    panel = ParameterPanel(specs)
    result_panel = ResultsPanel()
    try:
        assert panel._rows["ccd_j_lower"].isHidden()
        assert panel._rows["ccd_j_upper"].isHidden()
        panel.set_values({"ccd_fit_mode": "Manual"})
        assert not panel._rows["ccd_j_lower"].isHidden()
        result = tool.run([_lsv(tmp_path / "curve.csv")], _params())
        assert result.scalars["CCD threshold (mV)"] == 50
        assert result.scalars["CCD persistence (V)"] == .10
        settings = result.tables["Settings"].set_index("Parameter")["Value"]
        assert settings["threshold_v"] == .050
        assert settings["persistence_v"] == .10
        assert "sustained 50 mV" in result.text_blocks["CCD definition"]
        assert "confirmed over 100 mV" in result.text_blocks["CCD definition"]
        result_panel.set_result(result)
        assert result.display_summary and not result.display_tables
        assert not result_panel.findChildren(QTableWidget)
        assert result_panel.tabs.count() == 5
    finally:
        panel.deleteLater()
        result_panel.deleteLater()
        qt_app.processEvents()


def test_supplied_ten_cycle_file_last_forward_matches_independent_engine():
    path = Path(__file__).parents[1] / "Test_files" / "CCD_traning" / "CV1_3_2026_11_04_19_11_26.txt"
    if not path.is_file():
        pytest.skip("Historical ten-cycle acquisition is not part of this checkout.")
    params = _params(input_type="CV", cv_cycle_mode="Last", ccd_fit_mode="Auto",
                     electrode_setup="Two-electrode", i_lower=.02, i_upper=.8)
    all_scans = read_batch_scans([path], {**params, "cv_cycle_mode": "All"})
    assert [scan.cycle for scan in all_scans] == list(range(1, 11))
    result = CcdTool().run([path], params)
    assert result.tables["Current results"].Cycle.tolist() == [10]
    scan = read_batch_scans([path], params)[0]
    try:
        engine = decouple_curve(scan.potential, scan.current, DecouplingSettings.from_params(params),
                               source_indices=scan.source_indices, tafel_method="contiguous-stable")
    except ValueError:
        assert result.scalars["Scans decoupled"] == 0
        assert result.tables["Current results"]["Tafel status"].eq("Unresolved").all()
    else:
        assert result.scalars["Scans decoupled"] == 1
        _assert_components_match_engine(result, scan, params)
        assert result.tables["Fit diagnostics"].iloc[0]["Tafel b (V/dec)"] == pytest.approx(engine.fit["b_kin"])


def test_self_contained_ten_cycle_selection_and_partial_failures(tmp_path):
    path = _cv(tmp_path / "ten_cycles.csv", cycles=10)
    params = _params(input_type="CV", cv_cycle_mode="All")
    scans = read_batch_scans([path], params)
    assert [scan.cycle for scan in scans] == list(range(1, 11))
    result = CcdTool().run([path], params)
    assert result.tables["Current results"].Cycle.tolist() == list(range(1, 11))
    assert result.tables["Current results"].iloc[0]["Tafel status"] == "Resolved"
    assert result.scalars["Scans decoupled"] >= 1
    for scan in scans:
        row = result.tables["Current results"].set_index("Curve").loc[scan.code]
        if row["Decoupling status"] == "Resolved":
            _assert_components_match_engine(result, scan, params)
        else:
            assert row["Tafel status"] == "Unresolved"
            assert pd.isna(row[CCD_COLUMN])
    last = CcdTool().run([path], {**params, "cv_cycle_mode": "Last"})
    assert last.tables["Current results"].Cycle.tolist() == [10]


def test_batch_uses_stable_fit_while_single_preserves_previous_selector(tmp_path):
    current = np.geomspace(.005, 2., 2201)
    x = np.log10(current / .005)
    corrected = .15 + .06 * x + .035 * x ** 2
    potential = 1.2291 + corrected + .12 * current
    path = tmp_path / "curved.csv"
    pd.DataFrame({"E(V)": potential, "j(A/cm2)": current}).to_csv(path, index=False)
    params = _params(i_lower=.005, i_upper=.8)
    settings = DecouplingSettings.from_params(params)
    old = decouple_curve(potential, current, settings)
    single = LsvTool().run([path], params)
    assert single.scalars["b_kin (V/dec)"] == pytest.approx(old.fit["b_kin"])
    batch = CcdTool().run([path], params)
    row = batch.tables["Current results"].iloc[0]
    assert row["Tafel method"] == "contiguous-stable-tafel-v1"
    assert row["Tafel status"] == "Resolved"
    scan = read_batch_scans([path], params)[0]
    stable = _assert_components_match_engine(batch, scan, params)
    assert stable.fit["candidate_details"]["current_range"] != old.fit["candidate_details"]["current_range"]
    assert not np.isclose(stable.fit["b_kin"], old.fit["b_kin"])


def test_unresolved_batch_tafel_does_not_fall_back_to_single_selector(tmp_path):
    current = np.geomspace(.02, .16, 401)
    potential = 1.2291 + .45 + .2 * np.log10(current) + .12 * current
    current[200] *= -1
    path = tmp_path / "barrier.csv"
    pd.DataFrame({"E(V)": potential, "j(A/cm2)": current}).to_csv(path, index=False)
    params = _params(i_lower=.02, i_upper=.16, target_i=.1)
    single = LsvTool().run([path], params)
    assert single.scalars["b_kin (V/dec)"] == pytest.approx(.2)
    batch = CcdTool().run([path], params)
    row = batch.tables["Current results"].iloc[0]
    assert row["Tafel status"] == "Unresolved"
    assert row["Decoupling status"] == "Tafel unresolved"
    assert batch.scalars["Scans decoupled"] == 0
    assert pd.isna(row[CCD_COLUMN])
    assert len(batch.tables["Raw selected scans"]) == len(current)
