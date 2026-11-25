"""Decoupling fits exactly the CV branch and cycle selected in the input panel."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication

from workbench.app import ToolPage
from workbench.tools.lsv.tool import LsvTool


def _params(**overrides):
    values = {spec.key: spec.default for spec in LsvTool().parameter_schema()}
    values.update(
        input_type="CV", electrode_setup="Three-electrode", reference="RHE",
        reaction="OER", temperature=298.15,
        ph=0.0, hfr=0.0, rcl=0.0, include_rcl=False,
        i_lower=0.01, i_upper=0.3, target_i=0.5,
    )
    values.update(overrides)
    return values


def _slope(cycle, branch):
    return (0.055 if branch == "Forward" else 0.08) + 0.005 * cycle


def _write_cv(path, cycles=3, *, her=False, area=None):
    """Each branch has a distinct exact Tafel slope and a shared turning point."""
    frames = []
    sign = -1 if her else 1
    reversible = 0.0 if her else 1.2291
    for cycle in range(1, cycles + 1):
        for branch in ("Forward", "Back"):
            eta = np.linspace(0.2, 0.44, 801)
            if branch == "Back":
                eta = eta[-2::-1]
            current = sign * 10 ** ((eta - 0.44) / _slope(cycle, branch))
            column = "j(A/cm2)"
            if area is not None:
                current = current * area * 1000
                column = "I(mA)"
            frames.append(pd.DataFrame({
                "Cycle": cycle,
                "E(V)": sign * (reversible + eta), column: current,
            }))
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return path


def _assert_branch(table, cycle, branch, *, her=False):
    potential_column = next(column for column in table if "Potential" in column)
    sign = -1 if her else 1
    reversible = 0.0 if her else 1.2291
    expected = sign * (
        reversible + 0.44 + _slope(cycle, branch)
        * np.log10(np.abs(table["Current Density (A/cm²)"].to_numpy()))
    )
    np.testing.assert_allclose(table[potential_column], expected, atol=2e-12)


def _assert_exported_selection(workbook, expected, *, preview):
    if preview:
        pd.testing.assert_frame_equal(workbook["CV selection"], expected, check_dtype=False)
        return
    assert "CV selection" not in workbook
    selection = workbook["Summary"].loc[workbook["Summary"]["Section"].eq("CV selection")]
    if len(expected) == 1:
        assert (selection["Record"].isna() | selection["Record"].eq("")).all()
        records = [selection]
    else:
        assert selection["Record"].drop_duplicates().tolist() == list(range(1, len(expected) + 1))
        records = [selection.loc[selection["Record"].eq(number)]
                   for number in range(1, len(expected) + 1)]
    for record, (_, row) in zip(records, expected.iterrows(), strict=True):
        pd.testing.assert_series_equal(record.set_index("Parameter")["Value"], row,
                                       check_names=False, check_dtype=False, check_index_type=False)


@pytest.mark.parametrize("cycle,branch", [(0, "Forward"), (1, "Forward"), (2, "Back"), (0, "Back")])
def test_selected_cycle_and_scan_feed_preview_fit_and_exports(tmp_path, cycle, branch):
    path = _write_cv(tmp_path / "cycles.csv")
    params = _params(cv_cycle=cycle, cv_scan=branch)
    tool = LsvTool()
    chosen = cycle or 3

    preview = tool.show_plot([path], params)
    fit = tool.run([path], params)

    _assert_branch(preview.tables["CV scans (per file)"], chosen, branch)
    assert fit.scalars["b_kin (V/dec)"] == pytest.approx(_slope(chosen, branch), rel=2e-5)
    assert fit.scalars["i0 (A/cm²)"] == pytest.approx(
        10 ** (-0.44 / _slope(chosen, branch)), rel=2e-4, abs=0,
    )
    np.testing.assert_allclose(
        fit.tables["Curves (stacked)"]["Original CV scan (RHE)"],
        preview.tables["CV scans (per file)"].filter(like="Potential").iloc[:, 0],
    )
    expected = pd.DataFrame({
        "File": [path.name], "Cycle": [chosen], "Scan": [branch],
        "Direction": ["Increasing" if branch == "Forward" else "Decreasing"],
    })
    for result in (preview, fit):
        pd.testing.assert_frame_equal(result.tables["CV selection"], expected, check_dtype=False)
        workbook = next(artifact.payload for artifact in tool.exportable_artifacts(result)
                        if artifact.kind == "xlsx")
        _assert_exported_selection(workbook, expected, preview=result is preview)


@pytest.mark.parametrize("branch", ["Forward", "Back"])
def test_last_cycle_is_resolved_separately_for_each_batch_file(tmp_path, branch):
    first = _write_cv(tmp_path / "two_cycles.csv", cycles=2)
    second = _write_cv(tmp_path / "three_cycles.csv", cycles=3)
    result = LsvTool().show_plot([first, second], _params(cv_scan=branch))

    assert result.tables["CV selection"]["Cycle"].tolist() == [2, 3]
    originals = result.tables["CV scans (per file)"]
    for path, cycle in ((first, 2), (second, 3)):
        _assert_branch(originals.loc[originals.File.eq(path.name)], cycle, branch)
    assert result.scalars["Averaged"] == "yes"
    assert "Averaged CV scan" in result.tables


def test_duplicate_filenames_keep_preview_fit_and_export_selection_labels_aligned(tmp_path):
    first_folder, second_folder = tmp_path / "a", tmp_path / "b"
    first_folder.mkdir()
    second_folder.mkdir()
    first = _write_cv(first_folder / "cv.csv", cycles=2)
    second = _write_cv(second_folder / "cv.csv", cycles=3)
    tool = LsvTool()
    params = _params(cv_scan="Back")
    preview = tool.show_plot([first, second], params)
    fit = tool.run([first, second], params)
    originals = preview.tables["CV scans (per file)"]
    labels = originals["File"].drop_duplicates().tolist()

    assert labels == ["cv.csv", "cv.csv (2)"]
    for label, cycle in zip(labels, (2, 3), strict=True):
        _assert_branch(originals.loc[originals.File.eq(label)], cycle, "Back")
    for result in (preview, fit):
        selection = result.tables["CV selection"]
        assert selection["File"].tolist() == labels
        assert selection["Cycle"].tolist() == [2, 3]
        workbook = next(artifact.payload for artifact in tool.exportable_artifacts(result)
                        if artifact.kind == "xlsx")
        _assert_exported_selection(workbook, selection, preview=result is preview)
    preview_workbook = next(artifact.payload for artifact in tool.exportable_artifacts(preview)
                            if artifact.kind == "xlsx")
    for label in labels:
        assert any(column.startswith(f"{label} · ")
                   for column in preview_workbook["CV scans"])


@pytest.mark.parametrize("branch", ["Forward", "Back"])
def test_her_absolute_current_is_normalized_after_branch_selection(tmp_path, branch):
    path = _write_cv(tmp_path / "her.csv", her=True, area=2.5)
    params = _params(reaction="Auto", electrode_area=2.5, cv_cycle=2, cv_scan=branch)
    tool = LsvTool()
    preview = tool.show_plot([path], params)
    fit = tool.run([path], params)

    _assert_branch(preview.tables["CV scans (per file)"], 2, branch, her=True)
    assert fit.scalars["b_kin (V/dec)"] == pytest.approx(_slope(2, branch), rel=2e-5)
    current = fit.tables["Curves (stacked)"]["Current Density (A/cm²)"]
    assert (current < 0).all()
    assert np.max(np.abs(current)) == pytest.approx(1.0)
    assert fit.tables["CV selection"].iloc[0]["Direction"] == (
        "Decreasing" if branch == "Forward" else "Increasing"
    )


@pytest.mark.parametrize("method", ["run", "show_plot"])
def test_missing_requested_cycle_is_reported_with_file_and_cycle(tmp_path, method):
    path = _write_cv(tmp_path / "only_three.csv")
    with pytest.raises(ValueError, match=r"(?is)(?=.*only_three)(?=.*(?:cycle|Cycle))(?=.*4)"):
        getattr(LsvTool(), method)([path], _params(cv_cycle=4))


def test_old_lsv_parameters_retain_lsv_behavior_without_cv_metadata(tmp_path):
    path = _write_cv(tmp_path / "original.csv", cycles=1)
    frame = pd.read_csv(path)
    frame.iloc[:801].drop(columns="Cycle").to_csv(path, index=False)
    explicit = _params(input_type="LSV")
    legacy = {key: value for key, value in explicit.items()
              if key not in {"input_type", "cv_scan", "cv_cycle"}}
    tool = LsvTool()
    for method in ("show_plot", "run"):
        before = getattr(tool, method)([path], explicit)
        after = getattr(tool, method)([path], legacy)
        assert "CV selection" not in after.tables
        for name in before.tables:
            pd.testing.assert_frame_equal(before.tables[name], after.tables[name])
        workbook = next(artifact.payload for artifact in tool.exportable_artifacts(after)
                        if artifact.kind == "xlsx")
        assert "CV selection" not in workbook
        if "Summary" in workbook:
            assert not workbook["Summary"]["Section"].eq("CV selection").any()


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance()
    if app is None:
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("QT_QPA_PLATFORM", "offscreen")
            app = QApplication([])
    yield app


def test_cv_controls_are_conditional_default_to_last_and_survive_page_restore(qt_app, tmp_path):
    original = ToolPage(LsvTool(), lambda _: None)
    restored = ToolPage(LsvTool(), lambda _: None)
    legacy = ToolPage(LsvTool(), lambda _: None)
    try:
        assert original.panel.values()["input_type"] == "LSV"
        assert original.panel._rows["cv_scan"].isHidden()
        assert original.panel._rows["cv_cycle"].isHidden()
        original.panel.set_values({"input_type": "CV"})
        assert not original.panel._rows["cv_scan"].isHidden()
        assert not original.panel._rows["cv_cycle"].isHidden()
        assert original.panel._widgets["cv_cycle"].text() == "Last"
        original.panel.set_values({"cv_cycle": 2, "cv_scan": "Back"})
        state = original.save_state()
        restored.restore_state(state, tmp_path)
        assert restored.panel.values()["input_type"] == "CV"
        assert restored.panel.values()["cv_cycle"] == 2
        assert restored.panel.values()["cv_scan"] == "Back"
        assert not restored.panel._rows["cv_cycle"].isHidden()
        original.panel.set_values({"input_type": "LSV"})
        original.panel.set_values({"input_type": "CV"})
        assert original.panel.values()["cv_cycle"] == 2
        assert original.panel.values()["cv_scan"] == "Back"

        old_state = deepcopy(state)
        for key in ("input_type", "cv_cycle", "cv_scan"):
            old_state["panel"]["values"].pop(key)
        legacy.restore_state(old_state, tmp_path)
        assert legacy.panel.values()["input_type"] == "LSV"
        assert legacy.panel.values()["cv_cycle"] == 0
        assert legacy.panel._rows["cv_scan"].isHidden()
    finally:
        for page in (original, restored, legacy):
            page.deleteLater()
        qt_app.processEvents()
