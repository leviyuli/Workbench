"""Regression coverage for raw CV splitting and the retained ECSA workflow."""

import numpy as np
import pandas as pd
import pytest

from workbench.core.io import write_xlsx
from workbench.tools.ecsa import logic_cv
from workbench.tools.ecsa.tool import EcsaTool
from workbench.widgets.export_button import ExportButton


def _write_raw_cv(path, potential, current=None, header="E (V),i (A),t (s)"):
    potential = np.asarray(potential, dtype=float)
    if current is None:
        current = np.arange(len(potential), dtype=float) + 10.0
    current = np.asarray(current, dtype=float)
    rows = [header] + [
        f"{e:.12g},{i:.12g},{t:.12g}"
        for e, i, t in zip(potential, current, np.arange(len(potential)) * 0.2)
    ]
    path.write_text("\n".join(rows), encoding="utf-8")
    return np.column_stack([potential, current])


def _cycle_pairs(wide):
    """Recover the source order from adjacent E/i pairs, excluding padding."""
    assert wide.shape[1] % 2 == 0
    pairs = []
    for offset in range(0, wide.shape[1], 2):
        pair = wide.iloc[:, offset:offset + 2]
        # Padding must never separate a potential from its measured current.
        assert pair.iloc[:, 0].isna().equals(pair.iloc[:, 1].isna())
        pairs.append(pair.dropna().to_numpy())
    return pairs


def _assert_cycle_figure_matches_table(figure, wide):
    """Each trace must preserve its paired raw E/i samples and source units."""
    assert len(figure.axes) == 1
    axes = figure.axes[0]
    pairs = _cycle_pairs(wide)
    assert len(axes.lines) == len(pairs)
    for line, pair, segment in zip(axes.lines, pairs, wide.attrs["segments"]):
        np.testing.assert_array_equal(line.get_xdata(), pair[:, 0])
        np.testing.assert_array_equal(line.get_ydata(), pair[:, 1])
        assert line.get_label() == segment["label"]
        assert line.get_linestyle() == ("-" if segment["complete"] else "--")
        if len(pair) == 1:
            assert line.get_marker() not in (None, "", " ", "None", "none")
    assert axes.get_xlabel() == f'E ({wide.attrs["potential_unit"] or "raw"})'
    assert axes.get_ylabel() == f'i ({wide.attrs["current_unit"] or "raw"})'
    assert [text.get_text() for text in axes.get_legend().get_texts()] == [
        segment["label"] for segment in wide.attrs["segments"]
    ]


def test_cv_process_defaults_to_splitting_and_gates_ecsa_controls():
    tool = EcsaTool()
    specs = {spec.key: spec for spec in tool.parameter_schema()}

    assert tool.name == "CV process"
    assert specs["mode"].default == "Split cycles"
    assert specs["mode"].choices == ["Split cycles", "ECSA calculation"]
    for key in ("re_type", "target_scale", "ph", "Cs", "n_nearest",
                "cycle_index", "replicate_mode"):
        assert specs[key].show_if == ("mode", "ECSA calculation")


@pytest.mark.parametrize("header", ["E (V),i (A),t (s)", "E,i,t", ""])
def test_split_one_file_with_two_cycles_preserves_every_raw_sample(tmp_path, header):
    source = _write_raw_cv(
        tmp_path / "arbitrary measurement.txt",
        [0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0],
        header=header,
    )
    tool = EcsaTool()

    result = tool.run([tmp_path / "arbitrary measurement.txt"], {})

    assert result.scalars["Files"] == 1
    assert result.scalars["Cycles"] == 2
    assert len(result.tables) == 1
    wide = next(iter(result.tables.values()))
    pairs = _cycle_pairs(wide)
    assert len(pairs) == 2
    assert wide.attrs["cycle_count"] == 2
    np.testing.assert_allclose(np.vstack(pairs), source)
    for pair in pairs:
        changes = np.diff(pair[:, 0])
        assert np.any(changes > 0) and np.any(changes < 0)
    assert len(result.figures) == 1
    _assert_cycle_figure_matches_table(next(iter(result.figures.values())), wide)
    assert not any("scan rate" in warning.lower() for warning in result.warnings)


@pytest.mark.parametrize("current_unit,display_unit", [("mA", "mA"), ("mA/cm2", "mA/cm²")])
def test_split_selects_reordered_named_columns_without_scaling(tmp_path, current_unit, display_unit):
    path = tmp_path / "raw_units.txt"
    potential = np.array([0, 250, 500, 250, 0], dtype=float)
    current = np.array([1, 2, 3, -2, -1], dtype=float)
    rows = [f"time (s);Current ({current_unit});Potential (mV)"] + [
        f"{idx};{i};{e}" for idx, (e, i) in enumerate(zip(potential, current))
    ]
    path.write_text("\n".join(rows), encoding="utf-8")

    result = EcsaTool().run([path], {"mode": "Split cycles"})
    wide = next(iter(result.tables.values()))

    assert wide.shape == (5, 2)
    np.testing.assert_allclose(wide.to_numpy(), np.column_stack([potential, current]))
    assert wide.attrs["potential_unit"] == "mV"
    assert wide.attrs["current_unit"] == display_unit
    assert "mV" in wide.columns[0]
    assert display_unit in wide.columns[1]
    _assert_cycle_figure_matches_table(next(iter(result.figures.values())), wide)


def test_split_accepts_absolute_current_but_ecsa_still_requires_density(tmp_path):
    path = tmp_path / "CV_50mV_1.txt"
    _write_raw_cv(path, [0, 0.5, 1, 0.5, 0])
    tool = EcsaTool()

    split_result = tool.run([path], {"mode": "Split cycles"})

    assert split_result.scalars["Cycles"] == 1
    with pytest.raises(ValueError, match="requires current density"):
        tool.run([path], {"mode": "ECSA calculation"})


def test_split_pads_unequal_cycles_and_csv_exports_only_e_i_pairs(tmp_path):
    path = tmp_path / "unequal.txt"
    source = _write_raw_cv(
        path, [0, 0.5, 1, 0.5, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0],
    )
    tool = EcsaTool()
    result = tool.run([path], {"mode": "Split cycles"})
    csv_artifacts = [a for a in tool.exportable_artifacts(result) if a.kind == "csv"]

    assert len(csv_artifacts) == 1
    artifact = csv_artifacts[0]
    assert artifact.default_filename == "unequal_cycles.csv"
    pairs = _cycle_pairs(artifact.payload)
    assert len(pairs) == 2
    assert len(pairs[0]) < len(pairs[1])
    assert artifact.payload.iloc[len(pairs[0]):, :2].isna().all().all()
    np.testing.assert_allclose(np.vstack(pairs), source)
    _assert_cycle_figure_matches_table(next(iter(result.figures.values())), artifact.payload)

    export_path = tmp_path / artifact.default_filename
    ExportButton._write(artifact, str(export_path))
    exported = pd.read_csv(export_path)
    assert exported.shape == artifact.payload.shape
    assert list(exported.columns) == list(artifact.payload.columns)
    np.testing.assert_allclose(exported, artifact.payload, equal_nan=True)
    assert not any("time" in column.lower() or column.startswith("Unnamed:")
                   for column in exported.columns)

    png_artifact = next(a for a in tool.exportable_artifacts(result) if a.kind == "png")
    assert png_artifact.default_filename == "unequal_cycles.png"
    assert png_artifact.payload is next(iter(result.figures.values()))
    png_path = tmp_path / png_artifact.default_filename
    ExportButton._write(png_artifact, str(png_path))
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_split_duplicate_basenames_keep_independent_figures_and_unique_export_names(tmp_path):
    paths = []
    for folder, offset in (("first", 10), ("second", 20)):
        directory = tmp_path / folder
        directory.mkdir()
        path = directory / "measurement.txt"
        _write_raw_cv(path, [0, 0.5, 1, 0.5, 0], np.arange(5) + offset)
        paths.append(path)
    tool = EcsaTool()

    result = tool.run(paths, {"mode": "Split cycles"})
    artifacts = [a for a in tool.exportable_artifacts(result) if a.kind == "csv"]

    assert result.scalars["Files"] == 2
    assert result.scalars["Cycles"] == 2
    assert len(result.tables) == len(artifacts) == 2
    assert len({a.default_filename.casefold() for a in artifacts}) == 2
    assert all(a.default_filename.endswith("_cycles.csv") for a in artifacts)
    assert {float(a.payload.iloc[0, 1]) for a in artifacts} == {10.0, 20.0}
    assert len(result.figures) == 2
    assert len({id(figure) for figure in result.figures.values()}) == 2
    for table_name, wide in result.tables.items():
        figure_name = table_name.removesuffix(" cycles") + " CV overlay"
        _assert_cycle_figure_matches_table(result.figures[figure_name], wide)
    png_artifacts = [a for a in tool.exportable_artifacts(result) if a.kind == "png"]
    assert len(png_artifacts) == 2
    assert {a.default_filename.casefold() for a in png_artifacts} == {
        a.default_filename.removesuffix(".csv").casefold() + ".png" for a in artifacts
    }
    assert {id(a.payload) for a in png_artifacts} == {
        id(figure) for figure in result.figures.values()
    }


@pytest.mark.parametrize("potential", [
    [0.5, 0.75, 1, 0.5, 0, 0.5, 1, 0.5, 0, 0.25, 0.5],
    [0.5, 0, 0.5, 1, 0.5, 0],  # Leading partial contains one sample.
])
def test_split_preserves_and_labels_partial_scans_around_complete_cycles(tmp_path, potential):
    path = tmp_path / "partial_scans.txt"
    source = _write_raw_cv(path, potential)

    result = EcsaTool().run([path], {"mode": "Split cycles"})
    wide = next(iter(result.tables.values()))

    np.testing.assert_allclose(np.vstack(_cycle_pairs(wide)), source)
    assert wide.attrs["full_cycle_count"] >= 1
    assert wide.attrs["partial_cycle_count"] >= 1
    assert any("partial" in column.lower() for column in wide.columns)
    assert any("partial" in warning.lower() for warning in wide.attrs["warnings"])
    _assert_cycle_figure_matches_table(next(iter(result.figures.values())), wide)


def test_split_preserves_and_labels_an_unmatched_gap_between_full_cycles(tmp_path):
    path = tmp_path / "interior_partial.txt"
    source = _write_raw_cv(path, [0, 1, 0, 0.4, 0, 1, 0])

    result = EcsaTool().run([path], {"mode": "Split cycles"})
    wide = next(iter(result.tables.values()))

    np.testing.assert_allclose(np.vstack(_cycle_pairs(wide)), source)
    assert wide.attrs["full_cycle_count"] == 2
    assert wide.attrs["partial_cycle_count"] == 1
    assert result.scalars["Cycles"] == 2
    assert result.scalars["Partial scans"] == 1
    assert any("partial" in column.lower() for column in wide.columns)
    assert any("partial" in warning.lower() for warning in result.warnings)
    _assert_cycle_figure_matches_table(next(iter(result.figures.values())), wide)


@pytest.mark.parametrize("potential", [
    [0.5, 0.75, 1, 0.5, 0, 0.25, 0.5],
    [0.5, 1, 0.5, 0, 0.5],
    [0.8, 1, 0.5, 0, 0.5, 0.8],
])
def test_split_recognizes_a_complete_cycle_starting_inside_the_window(tmp_path, potential):
    path = tmp_path / "midpoint_start.txt"
    source = _write_raw_cv(path, potential)

    wide = logic_cv.split_cv_cycles(path)

    assert wide.shape == source.shape
    np.testing.assert_allclose(wide.to_numpy(), source)
    assert wide.attrs["cycle_count"] == 1
    assert wide.attrs["full_cycle_count"] == 1
    assert wide.attrs["partial_cycle_count"] == 0
    assert not wide.attrs["warnings"]


@pytest.mark.parametrize("potential", [[0, 0, 0, 0], [0, 0.25, 0.5, 0.75]])
def test_split_rejects_data_without_a_cv_cycle(tmp_path, potential):
    path = tmp_path / "no_cycle.txt"
    _write_raw_cv(path, potential)

    with pytest.raises(ValueError, match="(?i)cycle"):
        logic_cv.split_cv_cycles(path)


def test_split_rejects_files_without_numeric_cv_data(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("E (V),i (A),t (s)\ninvalid,missing,0\n", encoding="utf-8")

    with pytest.raises(ValueError):
        logic_cv.split_cv_cycles(path)


def test_ecsa_subfunction_mode_retains_fit_conversion_and_workbook(tmp_path):
    cdl = 0.01
    potential = np.concatenate([np.linspace(0, 0.2, 51), np.linspace(0.2, 0, 51)[1:]])
    paths = []
    for rate in (20, 40, 60, 80):
        jdl = cdl * rate * 1e-3
        current = np.r_[np.full(51, jdl), np.full(50, -jdl)]
        path = tmp_path / f"CV_{rate}mV_1.txt"
        _write_raw_cv(path, potential, current, header="E (V),j (A/cm2),t (s)")
        paths.append(path)
    tool = EcsaTool()

    result = tool.run(paths, {"mode": "ECSA calculation", "Cs": 40.0})

    assert result.scalars["C_dl (mF/cm²)"] == pytest.approx(cdl * 1e3)
    assert result.scalars["ECSA (cm²)"] == pytest.approx(cdl / 40e-6)
    assert result.scalars["R²"] == pytest.approx(1.0)
    assert result.scalars["Scan rates"] == 4
    assert result.tables["CV_long"]["potential"].min() == pytest.approx(0.098 + 0.0591 * 14)
    assert set(result.figures) == {"ECSA fit", "CV overlay"}
    assert callable(tool.compute_ecsa)

    workbook = next(a for a in tool.exportable_artifacts(result) if a.kind == "xlsx")
    assert workbook.default_filename == "ecsa_results.xlsx"
    assert set(workbook.payload) == {"Summary", "ECSA_result", "CV_long", "CV_time_series"}
    export_path = tmp_path / workbook.default_filename
    write_xlsx(workbook.payload, export_path)
    summary = pd.read_excel(export_path, sheet_name="Summary").set_index("Quantity")["Value"]
    assert summary["ECSA (cm²)"] == pytest.approx(cdl / 40e-6)
