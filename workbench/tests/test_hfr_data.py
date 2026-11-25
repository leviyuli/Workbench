"""HFR files and pasted tables preserve independent tests and missing cells."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from workbench.tools.lsv.hfr import HfrProfile
from workbench.tools.lsv.hfr_data import parse_hfr_pairs, parse_hfr_text, read_hfr_file


@pytest.mark.parametrize("separator", ["\t", ",", ";", " "])
def test_pasted_current_and_resistance_rows_accept_common_separators(separator):
    text = "\n".join(separator.join(map(str, row)) + "\\" for row in
                     ((.1, .06), (.5, .061), (1, .059), (2, .058), (3, .057)))
    parsed = parse_hfr_text(text)
    assert parsed.column_names == ("Test 1",)
    assert parsed.points() == [(.1, .06), (.5, .061), (1., .059), (2., .058), (3., .057)]
    assert HfrProfile(parsed.points())(1.) > 0


def test_header_and_empty_test_cells_preserve_column_identity():
    parsed = parse_hfr_text("j\tRun A\tRun B\n0.1\t0.06\t0.07\n0.5\t\t0.08\n1\t0.09\t\n")
    assert parsed.column_names == ("Run A", "Run B")
    assert parsed.rows == ((.1, .06, .07), (.5, None, .08), (1., .09, None))
    assert parsed.points(0) == [(.1, .06), (1., .09)]
    assert parsed.points(1) == [(.1, .07), (.5, .08)]
    with pytest.raises(ValueError, match="unavailable"):
        parsed.points(2)


def test_pair_input_accepts_exactly_current_and_one_hfr_column():
    data = parse_hfr_pairs("j\tHFR\n0.1\t0.06\n0.5\t0.061\n")
    assert data.column_names == ("HFR",)
    assert data.points() == [(.1, .06), (.5, .061)]


@pytest.mark.parametrize("text", ["0.1\t0.06\t0.07", "j,HFR A,HFR B\n.1,.06,.07"])
def test_pair_input_rejects_multiple_hfr_columns(text):
    with pytest.raises(ValueError, match="exactly two columns: current density and HFR"):
        parse_hfr_pairs(text)


@pytest.mark.parametrize("text,message", [
    ("", "Paste or upload"),
    ("0.1", "at least two columns"),
    ("j\tHFR", "no measurements"),
    ("0.1\t.1\n\t.2", "row 2 is missing"),
    ("0.1\t.1\n.2\tbad", "row 2, column 2 must be numeric"),
    ("0.1\tnan", "finite and nonnegative"),
    ("-1\t.1", "finite and nonnegative"),
    ("1\t-.1", "finite and nonnegative"),
    ("1\t", "at least two columns"),
    ("1\t.1\n2\t\t", "row 2 has no HFR"),
])
def test_invalid_data_reports_the_problem_row(text, message):
    with pytest.raises(ValueError, match=message):
        parse_hfr_text(text)


@pytest.mark.parametrize("name,counts", [("HFR1", [8, 8, 8]), ("HFR2", [8, 8, 8]),
                                        ("HFR3", [8, 8, 8, 10, 11])])
def test_uploaded_example_files_fit_each_test_independently(name, counts):
    source = Path(__file__).parents[1] / "Test_files" / "HFRvsj" / name
    if not source.exists():
        pytest.skip("Local measurement examples are not installed.")
    data = parse_hfr_text(read_hfr_file(source))
    assert [len(data.points(index)) for index in range(len(data.column_names))] == counts
    for index in range(len(data.column_names)):
        profile = HfrProfile(data.points(index))
        grid = np.linspace(*profile.current_range, 100)
        assert np.all(np.isfinite(profile(grid)))
        assert np.all(profile(grid) >= 0)
    if name == "HFR3":
        assert data.points(4)[-1] == (8., .09763)


@pytest.mark.parametrize("extension", ["csv", "xlsx", "txt", ""])
def test_upload_contents_are_portable_editable_text(tmp_path, extension):
    source = tmp_path / ("hfr" + ("." + extension if extension else ""))
    frame = pd.DataFrame({"j": [.1, .5, 1, 2, 3], "Run A": [.06, .061, .059, .058, .057]})
    if extension == "xlsx":
        frame.to_excel(source, index=False)
    else:
        frame.to_csv(source, sep="," if extension == "csv" else "\t", index=False)
    text = read_hfr_file(source)
    source.unlink()
    data = parse_hfr_text(text)
    assert data.column_names == ("Run A",)
    assert len(data.points()) == 5
