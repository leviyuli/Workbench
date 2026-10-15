"""Integrated-reference matching uses measured segments and common offsets."""

import numpy as np
import pytest

from workbench.tools.integrated_re.logic import (
    COMPARISON_COLUMNS,
    RECONSTRUCTED_COLUMNS,
    find_feature,
    normalize_current,
    potential_at_current,
    reconstruct_cell,
)


@pytest.mark.parametrize(
    ("role", "expected"),
    [("anode", [0.1, 0.0, -0.2, 1.0]), ("cathode", [-0.1, 0.0, 0.2, -1.0]),
     ("cell", [0.1, 0.0, -0.2, 1.0])],
)
def test_auto_polarity_uses_one_multiplier_for_the_whole_curve(role, expected):
    source = np.array([-0.1, 0.0, 0.2, -1.0])
    normalized, multiplier = normalize_current(source, role)
    np.testing.assert_array_equal(normalized, expected)
    np.testing.assert_array_equal(normalized, source * multiplier)
    np.testing.assert_array_equal(source, [-0.1, 0.0, 0.2, -1.0])


def test_manual_polarity_overrides_auto_and_all_zero_curve_is_retained():
    source = [0, 1, -2]
    np.testing.assert_array_equal(normalize_current(source, "anode", "As recorded")[0], source)
    np.testing.assert_array_equal(normalize_current(source, "anode", "Invert")[0], [0, -1, 2])
    np.testing.assert_array_equal(normalize_current([0, 0], "cathode")[0], [0, 0])


def test_acquisition_crossings_choose_extreme_voltage_without_sorting():
    potential = [0, 1, 2, 3]
    current = [0, 1, 0, 1]
    assert potential_at_current(potential, current, [0.5], select="max")[0] == 2.5
    assert potential_at_current(potential, current, [0.5], select="min")[0] == 0.5


def test_constant_current_segments_include_extreme_endpoints_and_singletons():
    np.testing.assert_array_equal(potential_at_current([3, 1], [0.2, 0.2], [0.2]), [3])
    np.testing.assert_array_equal(
        potential_at_current([3, 1], [0.2, 0.2], [0.2], select="min"), [1],
    )
    assert potential_at_current([7], [0.2], [0.2])[0] == 7


@pytest.mark.parametrize("missing", ["mask", "nan"])
def test_invalid_or_cropped_points_do_not_create_artificial_segments(missing):
    potential = np.array([0.0, 1.0, 2.0])
    kwargs = {}
    if missing == "mask":
        kwargs["mask"] = [True, False, True]
    else:
        potential[1] = np.nan
    matched = potential_at_current(potential, [0, 1, 2], [0, 0.5, 1.5, 2], **kwargs)
    assert matched[0] == 0
    assert matched[-1] == 2
    assert np.all(np.isnan(matched[1:3]))


def test_interpolation_preserves_zero_crossings_and_never_extrapolates():
    np.testing.assert_allclose(
        potential_at_current([0, 2], [-1, 1], [-2, 0, 0.5, 2]),
        [np.nan, 1, 1.5, np.nan], equal_nan=True,
    )


def test_reconstruction_uses_union_of_unequal_current_grids():
    reconstructed, comparison, warnings = reconstruct_cell(
        [1.3, 1.5, 1.9], [0, 0.2, 0.6],
        [0, -0.1, -0.4], [0, -0.1, -0.4], erev=1.2,
    )
    assert list(reconstructed.columns) == RECONSTRUCTED_COLUMNS
    np.testing.assert_allclose(reconstructed.iloc[:, 0], [0.1, 0.2, 0.4])
    np.testing.assert_allclose(reconstructed["Anode E (V vs RHE)"], [1.4, 1.5, 1.7])
    np.testing.assert_allclose(reconstructed["Cathode E (V vs RHE)"], [-0.1, -0.2, -0.4])
    np.testing.assert_allclose(reconstructed["Reconstructed cell voltage (V)"], [1.5, 1.7, 2.1])
    np.testing.assert_allclose(reconstructed["Cell overpotential (V)"], [0.3, 0.5, 0.9])
    assert list(comparison.columns) == COMPARISON_COLUMNS
    assert comparison.empty
    assert warnings == []


def test_reconstruction_selects_most_anodic_and_most_cathodic_crossings():
    reconstructed, _, _ = reconstruct_cell(
        [1.2, 1.4, 1.6, 1.8], [0, 1, 0, 1],
        [0, -0.2, -0.4, -0.6], [0, -1, 0, -1],
        cell_potential=[1.4, 2.4], cell_current=[0, 0.5],
    )
    row = reconstructed.loc[reconstructed.iloc[:, 0] == 0.5].iloc[0]
    assert row["Anode E (V vs RHE)"] == pytest.approx(1.7)
    assert row["Cathode E (V vs RHE)"] == pytest.approx(-0.5)
    assert row["Reconstructed cell voltage (V)"] == pytest.approx(2.2)


def test_measured_cell_grid_is_included_and_comparison_is_limited_to_overlap():
    reconstructed, comparison, warnings = reconstruct_cell(
        [1.3, 2.3], [0, 1], [0, -1], [0, -1],
        cell_potential=[1.85, 2.35], cell_current=[0.25, 0.5],
    )
    np.testing.assert_allclose(reconstructed.iloc[:, 0], [0.25, 0.5, 1])
    np.testing.assert_allclose(comparison.iloc[:, 0], [0.25, 0.5])
    np.testing.assert_allclose(comparison["Voltage difference (V)"], [-0.05, -0.05])
    assert warnings == []


def test_measured_cell_reverse_sweep_uses_highest_voltage_crossing():
    _, comparison, _ = reconstruct_cell(
        [1.3, 2.3], [0, 1], [0, -1], [0, -1],
        cell_potential=[1.5, 2.5, 1.8, 2.8], cell_current=[0, 1, 0, 1],
    )
    assert comparison["Measured cell voltage (V)"].iloc[0] == 2.8


def test_no_cell_overlap_keeps_reconstruction_and_warns():
    reconstructed, comparison, warnings = reconstruct_cell(
        [1.3, 2.3], [0, 1], [0, -1], [0, -1],
        cell_potential=[4, 5], cell_current=[2, 3],
    )
    assert not reconstructed.empty
    assert comparison.empty
    assert list(comparison.columns) == COMPARISON_COLUMNS
    assert "numerical comparison is unavailable" in warnings[0]


def test_matching_does_not_stitch_electrode_window_gaps():
    with pytest.raises(ValueError, match="overlapping.*crossings"):
        reconstruct_cell(
            [1.3, 1.5, 1.7], [0, 0.2, 0.4],
            [-0.1, -0.3], [-0.1, -0.3], anode_mask=[True, False, True],
        )


def test_common_shift_preserves_cell_voltage_and_does_not_mutate_sources():
    anode, cathode = np.array([1.3, 1.8]), np.array([0, -0.5])
    current = np.array([0, 0.5])
    before, _, _ = reconstruct_cell(anode, current, cathode, -current)
    shifted, _, _ = reconstruct_cell(anode - 0.2, current, cathode - 0.2, -current)
    np.testing.assert_allclose(
        before["Reconstructed cell voltage (V)"], shifted["Reconstructed cell voltage (V)"],
    )
    np.testing.assert_allclose(anode, [1.3, 1.8])
    np.testing.assert_allclose(cathode, [0, -0.5])


def test_onset_alignment_recovers_known_shared_offset_each_time():
    potential, current = np.array([-0.1, -0.2, -0.4]), np.array([0, 0.01, 0.1])
    for _ in range(2):
        integrated = find_feature(potential + 0.2, current, "onset", select="min")
        calibration = find_feature(potential, current, "onset", select="min")
        assert calibration["potential"] - integrated["potential"] == pytest.approx(-0.2)
        assert integrated["current"] == 0.01
    np.testing.assert_allclose(potential, [-0.1, -0.2, -0.4])


def test_peak_alignment_uses_prominence_and_recovers_known_offset():
    potential = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    current = np.array([0, 0.8, 0, 0.2, 1.1, 0.9, 0.8])
    integrated = find_feature(potential + 0.2, current, "peak", window=(0.1, 0.9))
    calibration = find_feature(potential, current, "peak", window=(-0.1, 0.7))
    assert calibration["potential"] == 0.1
    assert calibration["prominence"] == 0.8
    assert calibration["potential"] - integrated["potential"] == pytest.approx(-0.2)


@pytest.mark.parametrize(
    ("potential", "current", "window"),
    [([0, 1, 2], [3, 2, 1], (0, 2)),
     ([0, 1, 2], [1, 2, 3], (0, 2)),
     ([0, 1, 2], [0, 3, 0], (1, 2)),
     ([0, 1, 2, 3, 4], [0, 1, np.nan, 2, 0], (0, 4))],
)
def test_peak_search_never_uses_window_or_invalid_segment_endpoints(potential, current, window):
    with pytest.raises(ValueError, match="No interior"):
        find_feature(potential, current, "peak", window=window)


def test_peak_search_requires_a_finite_ordered_window():
    for window in (None, (0, 0), (1, 0), (0, np.inf), (0,)):
        with pytest.raises(ValueError, match="window"):
            find_feature([0, 1, 2], [0, 1, 0], "peak", window=window)


def test_onset_window_restricts_crossings_without_stitching():
    feature = find_feature([0, 1, 2, 3], [0, 1, 0, 1], "onset", target_current=0.5,
                           window=(0, 1), select="max")
    assert feature["potential"] == 0.5
    with pytest.raises(ValueError, match="not crossed"):
        find_feature([0, 3, 1], [0, 0.5, 1], "onset", target_current=0.5, window=(0, 1))


@pytest.mark.parametrize("target", [0, -1, np.nan, np.inf])
def test_onset_requires_positive_finite_current_density(target):
    with pytest.raises(ValueError, match="finite and positive"):
        find_feature([0, 1], [0, 1], "onset", target_current=target)


def test_bad_curve_shapes_and_no_reaction_overlap_report_errors():
    with pytest.raises(ValueError, match="equal lengths"):
        potential_at_current([0], [0, 1], [0.5])
    with pytest.raises(ValueError, match="branch mask"):
        potential_at_current([0, 1], [0, 1], [0.5], mask=[True])
    with pytest.raises(ValueError, match="finite one-dimensional"):
        potential_at_current([0, 1], [0, 1], [np.inf])
    with pytest.raises(ValueError, match="no overlapping"):
        reconstruct_cell([1, 2], [0.1, 0.2], [-1, -2], [-0.3, -0.4])
    with pytest.raises(ValueError, match="Cathode.*positive reaction"):
        reconstruct_cell([1, 2], [0.1, 0.2], [-1, -2], [0.1, 0.2])
    with pytest.raises(ValueError, match="finite"):
        reconstruct_cell([1, 2], [0, 1], [0, -1], [0, -1], erev=np.nan)
    with pytest.raises(ValueError, match="supplied together"):
        reconstruct_cell([1, 2], [0, 1], [0, -1], [0, -1], cell_potential=[2, 3])
