"""Cycle extraction retains acquired forward scans and their physical units."""
import numpy as np
import pandas as pd
import pytest

from workbench.tools.ccd.inputs import read_ccd_scans


def _read(path):
    result, error = read_ccd_scans(path)
    assert error is None, error
    return result


def _cycle(scale=1., negative=False):
    forward = np.linspace(1, 3, 401)
    reverse = forward[-2::-1]
    voltage = np.r_[forward, reverse]
    current = np.r_[scale * (forward - 1), 20 + reverse]
    if negative:
        voltage, current = -voltage, -current
    return voltage, current, forward, scale * (forward - 1)


@pytest.mark.parametrize("negative", [False, True])
def test_each_cycle_retains_only_forward_data_before_current_filtering(tmp_path, negative):
    blocks = [_cycle(scale, negative) for scale in [1, 2, 3]]
    path = tmp_path / "cv.csv"
    pd.DataFrame({"E(V)": np.concatenate([block[0] for block in blocks]),
                  "j(A/cm2)": np.concatenate([block[1] for block in blocks])}).to_csv(path, index=False)
    scans, info = _read(path)

    assert [scan.cycle for scan in scans] == [1, 2, 3]
    assert info["scan_count"] == 3
    assert info["warnings"] == []
    for scan, block in zip(scans, blocks):
        assert scan.direction == ("Decreasing" if negative else "Increasing")
        assert scan.is_density
        # A duplicated valley belongs to the next scan and does not create a
        # separate cycle; the return's high-current points cannot enter it.
        sign = -1 if negative else 1
        np.testing.assert_allclose(scan.potential[-401:], sign * block[2])
        np.testing.assert_allclose(scan.current[-401:], sign * block[3])
        assert np.max(np.abs(scan.current)) <= 6


def test_wide_cycle_columns_preserve_ids_and_convert_millivolts_milliamps(tmp_path):
    first = _cycle(1)
    second = _cycle(2)
    path = tmp_path / "wide.csv"
    table = pd.DataFrame({
        "Cycle 2 Voltage (mV)": 1000 * first[0],
        "Cycle 2 Current (mA)": 1000 * first[1],
        "Cycle 7 Voltage (mV)": 1000 * second[0],
        "Cycle 7 Current (mA)": 1000 * second[1],
    })
    table.loc[len(table)] = np.nan  # Wide spreadsheet padding.
    table.to_csv(path, index=False)
    scans, info = _read(path)

    assert [scan.cycle for scan in scans] == [2, 7]
    assert info["source_format"] == "wide cycles"
    for scan, block in zip(scans, [first, second]):
        np.testing.assert_allclose(scan.potential, block[2])
        np.testing.assert_allclose(scan.current, block[3])
        assert not scan.is_density


def test_long_cycle_table_preserves_explicit_numbers(tmp_path):
    blocks = [_cycle(scale) for scale in [1, 2]]
    path = tmp_path / "long.csv"
    pd.DataFrame({"Cycle number": np.repeat([4, 8], len(blocks[0][0])),
                  "E(V)": np.concatenate([block[0] for block in blocks]),
                  "j(A/cm2)": np.concatenate([block[1] for block in blocks])}).to_csv(path, index=False)
    scans, info = _read(path)

    assert [scan.cycle for scan in scans] == [4, 8]
    assert info["source_format"] == "long cycles"
    assert len(scans[0].potential) == 401


def test_small_voltage_jitter_and_holds_do_not_create_cycles(tmp_path):
    voltage, current, _, _ = _cycle()
    voltage[100] = voltage[99] - .0001
    voltage[250:254] = voltage[249]
    path = tmp_path / "jitter.csv"
    pd.DataFrame({"E(V)": voltage, "j(A/cm2)": current}).to_csv(path, index=False)
    scans, _ = _read(path)

    assert len(scans) == 1
    np.testing.assert_allclose(scans[0].potential, voltage[:401])
    np.testing.assert_allclose(scans[0].current, current[:401])


def test_invalid_current_rows_remain_inside_the_scan(tmp_path):
    voltage, current, _, _ = _cycle()
    current[100] = np.nan
    path = tmp_path / "gap.csv"
    pd.DataFrame({"E(V)": voltage, "j(A/cm2)": current}).to_csv(path, index=False)
    scans, _ = _read(path)

    assert len(scans) == 1
    assert np.isnan(scans[0].current[100])
    assert len(scans[0].current) == 401


def test_truncated_final_forward_scan_remains_identified_and_warned(tmp_path):
    voltage, current, _, _ = _cycle()
    path = tmp_path / "partial.csv"
    pd.DataFrame({"E(V)": np.r_[voltage, voltage[:80]],
                  "j(A/cm2)": np.r_[current, current[:80]]}).to_csv(path, index=False)
    scans, info = _read(path)

    assert [scan.cycle for scan in scans] == [1, 2]
    assert any("Cycle 2" in warning and "incomplete" in warning for warning in info["warnings"])


def test_wide_cycle_without_matching_current_is_an_error(tmp_path):
    path = tmp_path / "missing.csv"
    pd.DataFrame({"Cycle 1 Voltage (V)": [1, 2, 3],
                  "Cycle 1 Current (A)": [0, 1, 2],
                  "Cycle 2 Voltage (V)": [1, 2, 3]}).to_csv(path, index=False)
    result, error = read_ccd_scans(path)

    assert result is None
    assert "Cycle 2 needs both" in error


@pytest.mark.parametrize("layout", ["wide", "long"])
def test_bad_explicit_cycles_remain_rows_without_discarding_valid_neighbors(tmp_path, layout):
    voltage, current, forward, forward_current = _cycle(negative=True)
    flat = np.full(len(voltage), -1.)
    invalid = np.full(len(voltage), np.nan)
    blocks = [(flat, current), (voltage, current), (invalid, invalid), (voltage, current)]
    path = tmp_path / "bad_cycles.csv"
    if layout == "wide":
        data = {}
        for cycle, (v, i) in enumerate(blocks, start=1):
            data[f"Cycle {cycle} E(V)"] = v
            data[f"Cycle {cycle} j(A/cm2)"] = i
    else:
        data = {"Cycle": np.repeat(np.arange(1, 5), len(voltage)),
                "E(V)": np.concatenate([v for v, _ in blocks]),
                "j(A/cm2)": np.concatenate([i for _, i in blocks])}
    pd.DataFrame(data).to_csv(path, index=False)
    scans, info = _read(path)

    assert [scan.cycle for scan in scans] == [1, 2, 3, 4]
    assert scans[0].error and scans[2].error
    assert scans[0].direction == scans[2].direction == "Unknown"
    assert not scans[1].error and not scans[3].error
    assert info["scan_count"] == 4
    assert info["warnings"] == []
    for scan in [scans[1], scans[3]]:
        assert scan.direction == "Decreasing"
        np.testing.assert_allclose(scan.potential, -forward)
        np.testing.assert_allclose(scan.current, -forward_current)


def test_multiple_forward_scans_with_same_explicit_label_fail_only_that_cycle(tmp_path):
    voltage, current, _, _ = _cycle()
    path = tmp_path / "repeated.csv"
    pd.DataFrame({
        "Cycle": np.r_[np.ones(2 * len(voltage)), np.full(len(voltage), 2)],
        "E(V)": np.tile(voltage, 3), "j(A/cm2)": np.tile(current, 3),
    }).to_csv(path, index=False)
    scans, _ = _read(path)

    assert len(scans) == 2
    assert scans[0].cycle == 1 and "multiple forward scans" in scans[0].error
    assert scans[0].direction == "Increasing"
    assert scans[1].cycle == 2 and not scans[1].error


@pytest.mark.parametrize("layout", ["long", "wide"])
def test_acquired_forward_scans_match_exported_cycle_layouts(tmp_path, layout):
    blocks = [_cycle(scale) for scale in (1, 2, 3)]
    source = tmp_path / "acquisition.txt"
    source.write_text("CV acquisition\n" + pd.DataFrame({
        "E(V)": np.concatenate([block[0] for block in blocks]),
        "j(A/cm2)": np.concatenate([block[1] for block in blocks]),
    }).to_csv(index=False, sep="\t", lineterminator="\n"), encoding="utf-8")
    raw, _ = _read(source)
    assert [scan.cycle for scan in raw] == [1, 2, 3]
    path = tmp_path / f"cycles_{layout}.csv"
    if layout == "wide":
        table = pd.DataFrame({
            f"Cycle {scan.cycle} {label}": pd.Series(values)
            for scan in raw
            for label, values in [("E(V)", scan.potential), ("j(A/cm2)", scan.current)]
        })
    else:
        table = pd.concat([
            pd.DataFrame({"Cycle": scan.cycle, "E(V)": scan.potential,
                          "j(A/cm2)": scan.current}) for scan in raw
        ], ignore_index=True)
    table.to_csv(path, index=False)
    processed, info = _read(path)

    assert len(raw) == len(processed) == info["scan_count"] == 3
    assert info["source_format"] == f"{layout} cycles"
    for first, second in zip(raw, processed):
        assert first.cycle == second.cycle
        assert first.direction == second.direction == "Increasing"
        assert first.is_density and second.is_density
        assert not first.error and not second.error
        np.testing.assert_allclose(first.potential, second.potential, rtol=1e-14, atol=0)
        np.testing.assert_allclose(first.current, second.current, rtol=1e-14, atol=0)
