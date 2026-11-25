"""Decoupling selects a CV cycle and branch before averaging or sign filtering."""
import numpy as np
import pandas as pd
import pytest

from workbench.core.data_io import read_lsv_data
from workbench.tools.lsv.data import read_decoupling_data


def _cycle(scale=1.0, *, negative=False, back_negative=False):
    forward = np.linspace(1.0, 2.0, 101)
    back = forward[-2::-1]
    potential = np.r_[forward, back]
    current = np.r_[scale * (forward + .25), scale * (back + .75)]
    if back_negative:
        current[len(forward):] *= -1
    if negative:
        potential, current = -potential, -current
    return potential, current


def _source(path, *, scales=(1, 2, 3), layout="acquisition", labels=None,
            negative=False, back_negative=False, absolute=False, milli=False):
    blocks = [_cycle(scale, negative=negative, back_negative=back_negative)
              for scale in scales]
    labels = list(labels if labels is not None else range(1, len(blocks) + 1))
    voltage_name = "E(mV)" if milli else "E(V)"
    current_name = ("I(mA)" if milli else "I(A)") if absolute else (
        "j(mA/cm2)" if milli else "j(A/cm2)"
    )
    unit_scale = 1000 if milli else 1
    if layout == "wide":
        frame = pd.DataFrame({
            f"Cycle {label} {name}": pd.Series(values * unit_scale)
            for label, block in zip(labels, blocks, strict=True)
            for name, values in zip((voltage_name, current_name), block, strict=True)
        })
        frame.loc[len(frame)] = np.nan
    else:
        frame = pd.DataFrame({
            voltage_name: np.concatenate([block[0] for block in blocks]) * unit_scale,
            current_name: np.concatenate([block[1] for block in blocks]) * unit_scale,
        })
        if layout == "long":
            frame.insert(0, "Cycle number", np.repeat(labels, len(blocks[0][0])))
    frame.to_csv(path, index=False)
    return path


def _read(paths, *, branch="Forward", cycle="Last", reaction="auto"):
    data, error = read_decoupling_data(paths, {
        "input_type": "CV", "cv_scan": branch, "cv_cycle": cycle,
    }, reaction=reaction)
    assert error is None, error
    return data


@pytest.mark.parametrize("layout", ["acquisition", "long", "wide"])
@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("branch", ["Forward", "Back"])
def test_last_cycle_and_requested_branch_are_selected_in_all_supported_layouts(
    tmp_path, layout, negative, branch,
):
    path = _source(tmp_path / "cv.csv", layout=layout, negative=negative)
    voltage, current, info = _read([path], branch=branch)
    sign = -1 if negative else 1
    interior = (np.abs(voltage) > 1.05) & (np.abs(voltage) < 1.95)
    offset = .25 if branch == "Forward" else .75
    np.testing.assert_allclose(current[interior], sign * 3 * (np.abs(voltage[interior]) + offset))
    assert np.all(np.diff(voltage) >= 0)
    assert info["reaction"] == ("her" if negative else "oer")
    assert info["is_density"]
    direction = ("Decreasing" if negative else "Increasing")
    if branch == "Back":
        direction = "Increasing" if negative else "Decreasing"
    assert info["cv_selections"] == [{
        "File": "cv.csv", "Cycle": 3, "Scan": branch, "Direction": direction,
    }]


@pytest.mark.parametrize("layout", ["long", "wide"])
def test_explicit_cycle_labels_are_ids_rather_than_ordinal_positions(tmp_path, layout):
    path = _source(tmp_path / "labelled.csv", scales=(1, 4), labels=(2, 7), layout=layout)
    voltage, current, info = _read([path], branch="Back", cycle="2")
    interior = (voltage > 1.05) & (voltage < 1.95)
    np.testing.assert_allclose(current[interior], voltage[interior] + .75)
    assert info["cv_selections"][0]["Cycle"] == 2
    assert _read([path])[2]["cv_selections"][0]["Cycle"] == 7


def test_cycle_and_branch_selection_precede_automatic_reaction_detection(tmp_path):
    path = _source(tmp_path / "different_signs.csv", back_negative=True)
    forward = _read([path], branch="Forward")
    back = _read([path], branch="Back")
    assert forward[2]["reaction"] == "oer"
    assert np.all(forward[1] > 0)
    assert back[2]["reaction"] == "her"
    assert np.all(back[1] < 0)


def test_last_cycle_is_resolved_per_file_before_batch_average(tmp_path):
    first = _source(tmp_path / "first.csv", scales=(1, 2))
    second = _source(tmp_path / "second.csv", scales=(3, 4, 6))
    voltage, current, info = _read([first, second], branch="Back")
    interior = (voltage > 1.05) & (voltage < 1.95)
    np.testing.assert_allclose(current[interior], 4 * (voltage[interior] + .75))
    assert info["averaged"]
    assert [row["Cycle"] for row in info["cv_selections"]] == [2, 3]


def test_millivolts_and_milliamps_convert_once_and_remain_absolute_current(tmp_path):
    path = _source(tmp_path / "units.csv", absolute=True, milli=True)
    voltage, current, info = _read([path], branch="Back")
    assert not info["is_density"]
    assert np.min(voltage) == pytest.approx(1)
    assert np.max(voltage) == pytest.approx(2)
    interior = (voltage > 1.05) & (voltage < 1.95)
    np.testing.assert_allclose(current[interior], 3 * (voltage[interior] + .75))


def test_batch_refuses_mixed_absolute_current_and_current_density(tmp_path):
    absolute = _source(tmp_path / "absolute.csv", absolute=True)
    density = _source(tmp_path / "density.csv")
    data, error = read_decoupling_data([absolute, density], {"input_type": "CV"})
    assert data is None
    assert "mixture" in error and "current-density" in error


def test_missing_requested_cycle_in_any_file_does_not_average_the_others(tmp_path):
    first = _source(tmp_path / "first.csv", scales=(1, 2))
    second = _source(tmp_path / "second.csv", scales=(1, 2, 3))
    data, error = read_decoupling_data([first, second], {
        "input_type": "CV", "cv_cycle": "3", "cv_scan": "Forward",
    })
    assert data is None
    assert "first.csv" in error and "3" in error


@pytest.mark.parametrize("layout", ["acquisition", "long", "wide"])
def test_missing_back_of_last_cycle_does_not_fall_back_to_an_earlier_cycle(tmp_path, layout):
    path = tmp_path / "unfinished.csv"
    complete_v, complete_i = _cycle()
    final_v, final_i = _cycle(2)
    final_v, final_i = final_v[:101], final_i[:101]
    if layout == "wide":
        frame = pd.DataFrame({
            "Cycle 1 E(V)": pd.Series(complete_v), "Cycle 1 j(A/cm2)": pd.Series(complete_i),
            "Cycle 2 E(V)": pd.Series(final_v), "Cycle 2 j(A/cm2)": pd.Series(final_i),
        })
    else:
        frame = pd.DataFrame({"E(V)": np.r_[complete_v, final_v],
                              "j(A/cm2)": np.r_[complete_i, final_i]})
        if layout == "long":
            frame.insert(0, "Cycle", np.r_[np.ones(len(complete_v)), np.full(len(final_v), 2)])
    frame.to_csv(path, index=False)
    assert _read([path], branch="Forward")[2]["cv_selections"][0]["Cycle"] == 2
    data, error = read_decoupling_data([path], {"input_type": "CV", "cv_scan": "Back"})
    assert data is None
    assert "2" in error and "back" in error.lower()


@pytest.mark.parametrize("cycle", ["-1", "1.5", "unknown"])
def test_invalid_cycle_selection_has_an_actionable_error(tmp_path, cycle):
    path = _source(tmp_path / "cv.csv")
    data, error = read_decoupling_data([path], {"input_type": "CV", "cv_cycle": cycle})
    assert data is None
    assert "cycle" in error.lower()


@pytest.mark.parametrize("cycle", [0, "0", "Last"])
def test_last_cycle_sentinel_matches_the_cycle_picker_default(tmp_path, cycle):
    path = _source(tmp_path / "cv.csv")
    actual = _read([path], cycle=cycle)
    expected = _read([path], cycle="Last")
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2]["cv_selections"] == expected[2]["cv_selections"]


def test_legacy_lsv_defaults_and_inactive_cv_settings_use_unchanged_lsv_reader(tmp_path):
    path = tmp_path / "lsv.csv"
    voltage = np.linspace(1, 2, 101)
    pd.DataFrame({"E(V)": voltage, "j(A/cm2)": voltage + .25}).to_csv(path, index=False)
    expected, error = read_lsv_data([path])
    assert error is None
    for params in ({}, {"input_type": "LSV", "cv_scan": "Back", "cv_cycle": "unused"}):
        actual, error = read_decoupling_data([path], params)
        assert error is None, error
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        assert actual[2] == expected[2]
