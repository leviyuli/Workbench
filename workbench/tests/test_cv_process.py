"""Regression coverage for raw CV splitting and the retained ECSA workflow."""

import numpy as np
import pandas as pd
import pytest

from workbench.core.io import write_xlsx
from workbench.core.potential import FARADAY, R_GAS, PotentialCorrection
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
    assert axes.get_xlabel() == wide.attrs['potential_label']
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
    for key in ("Cs", "n_nearest", "cycle_index", "replicate_mode"):
        assert specs[key].show_if == ("mode", "ECSA calculation")
    assert specs['electrode_setup'].show_if is None
    assert specs['electrode_setup'].default == 'Two-electrode'
    assert specs['target_scale'].default == 'Original'
    assert not any(condition[0] == 'mode' for condition in specs['reference'].show_if)


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
    assert result.scalars["Electrode setup"] == "Two-electrode"
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
        tool.run([path], {"mode": "ECSA calculation", "electrode_setup": "Three-electrode"})


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

    result = tool.run(paths, {"mode": "ECSA calculation", "Cs": 40.0, "target_scale": "RHE",
                              "electrode_setup": "Three-electrode"})

    assert result.scalars["C_dl (mF/cm²)"] == pytest.approx(cdl * 1e3)
    assert result.scalars["ECSA (cm²)"] == pytest.approx(cdl / 40e-6)
    assert result.scalars["R²"] == pytest.approx(1.0)
    assert result.scalars["Scan rates"] == 4
    assert result.tables["CV_long"]["potential"].min() == pytest.approx(
        0.1053 + 2.303 * R_GAS * 298.15 / FARADAY * 14)
    assert set(result.figures) == {"ECSA fit", "CV overlay"}
    assert callable(tool.compute_ecsa)

    workbook = next(a for a in tool.exportable_artifacts(result) if a.kind == "xlsx")
    assert workbook.default_filename == "ecsa_results.xlsx"
    assert list(workbook.payload) == ["Results", "ECSA fit", "CV overlay"]
    export_path = tmp_path / workbook.default_filename
    write_xlsx(workbook.payload, export_path)
    summary = pd.read_excel(export_path, sheet_name="Results").set_index("Quantity")["Value"]
    assert summary["ECSA (cm²)"] == pytest.approx(cdl / 40e-6)
    assert summary["Specific capacitance (µF/cm²)"] == 40.0


@pytest.mark.parametrize('unit,multiplier', [('V', 1.0), ('mV', 1000.0)])
def test_split_corrects_all_cycles_and_partial_scans_in_source_units(tmp_path, unit, multiplier):
    path = tmp_path / 'corrected.txt'
    potential = np.array([0.5, 0.75, 1, 0.5, 0, 0.5, 1, 0.5, 0, 0.25, 0.5]) * multiplier
    source = _write_raw_cv(path, potential, header=f'E ({unit}),i (mA),t (s)')
    tool = EcsaTool()
    result = tool.run([path], {
        'electrode_setup': 'Three-electrode',
        'target_scale': 'RHE', 'reference': 'Custom E_ref vs SHE',
        'custom_e_ref_vs_she': 0.25, 'ph': 0, 'temperature': 298.15,
    })
    wide = next(iter(result.tables.values()))
    actual = np.vstack(_cycle_pairs(wide))
    np.testing.assert_allclose(actual[:, 0], source[:, 0] + 0.25 * multiplier)
    np.testing.assert_array_equal(actual[:, 1], source[:, 1])
    assert wide.attrs['source_row_count'] == len(source)
    assert wide.attrs['partial_cycle_count'] > 0
    assert wide.attrs['potential_unit'] == unit
    assert wide.attrs['current_unit'] == 'mA'
    assert all(f'{unit} vs RHE' in column for column in wide.columns[::2])
    _assert_cycle_figure_matches_table(next(iter(result.figures.values())), wide)
    csv = next(a for a in tool.exportable_artifacts(result) if a.kind == 'csv')
    export_path = tmp_path / csv.default_filename
    ExportButton._write(csv, str(export_path))
    exported = pd.read_csv(export_path)
    np.testing.assert_allclose(exported, wide, equal_nan=True)
    assert list(exported.columns) == list(wide.columns)
    workbook = next(a for a in tool.exportable_artifacts(result) if a.kind == 'xlsx')
    assert list(workbook.payload) == list(result.tables)
    metadata = result.payload['potential_correction']
    assert metadata['Potential shift (V)'] == pytest.approx(0.25)


@pytest.mark.parametrize('setup,target', [('Two-electrode', 'RHE'), ('Three-electrode', 'Original')])
def test_split_inactive_reference_settings_leave_values_unchanged(tmp_path, setup, target):
    path = tmp_path / 'unchanged.txt'
    source = _write_raw_cv(path, [0, 250, 500, 250, 0], header='E (mV),i (mA),t (s)')
    result = EcsaTool().run([path], {
        'electrode_setup': setup, 'target_scale': target,
        'reference': 'stale selection', 'ph': float('nan'), 'temperature': -1,
    })
    wide = next(iter(result.tables.values()))
    np.testing.assert_array_equal(np.vstack(_cycle_pairs(wide)), source)
    expected_label = 'Cell voltage (mV)' if setup == 'Two-electrode' else 'E (mV)'
    assert wide.attrs['potential_label'] == expected_label
    assert expected_label in wide.columns[0]
    assert 'RHE' not in wide.columns[0]
    _assert_cycle_figure_matches_table(next(iter(result.figures.values())), wide)


@pytest.mark.parametrize('header', ['E', 'V', 'Potential'])
def test_split_unitless_potential_uses_explicit_fallback_and_reports_it(tmp_path, header):
    path = tmp_path / 'unitless.txt'
    source = _write_raw_cv(path, [0, 250, 500, 250, 0], header=f'{header},i,t')
    result = EcsaTool().run([path], {
        'electrode_setup': 'Three-electrode',
        'target_scale': 'SHE', 'reference': 'Ag/AgCl (3 M KCl, 25 °C)',
        'potential_unit': 'mV',
    })
    wide = next(iter(result.tables.values()))
    actual = np.vstack(_cycle_pairs(wide))
    np.testing.assert_allclose(actual[:, 0], source[:, 0] + 210)
    np.testing.assert_array_equal(actual[:, 1], source[:, 1])
    assert 'mV vs SHE' in wide.columns[0]
    assert any('using mV' in warning for warning in result.warnings)


def test_split_rejects_unknown_reference_before_processing_files():
    with pytest.raises(ValueError, match='Unknown reference'):
        EcsaTool().run([], {'electrode_setup': 'Three-electrode',
                            'target_scale': 'RHE', 'reference': 'unknown'})


def test_ecsa_rejects_two_electrode_input_before_fitting():
    with pytest.raises(ValueError, match='requires a three-electrode'):
        EcsaTool().run([], {'mode': 'ECSA calculation', 'electrode_setup': 'Two-electrode'})


def test_split_and_ecsa_use_identical_conversion_with_different_output_units(tmp_path):
    potential = np.r_[np.linspace(0, 200, 51), np.linspace(200, 0, 51)[1:]]
    paths = []
    for rate in (20, 40, 60):
        current = np.r_[np.full(51, rate * 0.01), np.full(50, -rate * 0.01)]
        path = tmp_path / f'CV_{rate}mV_1.txt'
        _write_raw_cv(path, potential, current, header='E (mV),j (mA/cm2),t (s)')
        paths.append(path)
    params = {'electrode_setup': 'Three-electrode',
              'target_scale': 'RHE', 'reference': 'Ag/AgCl (3 M KCl, 25 °C)',
              'ph': 13, 'temperature': 313.15}
    tool = EcsaTool()
    split = tool.run(paths[:1], params)
    ecsa = tool.run(paths, {**params, 'mode': 'ECSA calculation'})
    split_e = np.vstack(_cycle_pairs(next(iter(split.tables.values()))))[:, 0]
    ecsa_e = ecsa.tables['CV_long'].query('scan_rate == 20')['potential'].to_numpy()
    np.testing.assert_allclose(split_e / 1000, ecsa_e)
    assert ecsa.scalars['C_dl (mF/cm²)'] == pytest.approx(10)
    assert ecsa.scalars['Potential shift (V)'] == split.scalars['Potential shift (V)']
    assert 'V vs RHE' in ecsa.figures['CV overlay'].axes[0].get_xlabel()
    assert any('V vs RHE' in column for column in ecsa.tables['CV_time_series'])
    assert any('25 °C' in warning for warning in ecsa.warnings)


@pytest.mark.parametrize('target', ['Original', 'RHE'])
def test_ecsa_normalizes_unitless_mv_independently_of_reference_correction(tmp_path, target):
    path = tmp_path / 'CV_20mV_1.txt'
    _write_raw_cv(path, [0, 100, 200, 100, 0], header='E,j (A/cm2),t (s)')
    correction = PotentialCorrection.from_params({'electrode_setup': 'Three-electrode',
                                                  'target_scale': target, 'reference': 'RHE'})
    cycles = logic_cv.process_one_file(path, potential_correction=correction, potential_unit='mV')
    np.testing.assert_allclose(cycles[0]['E'], [0, 0.1, 0.2, 0.1, 0])
    assert any('using mV' in warning for warning in cycles[0].attrs['warnings'])
