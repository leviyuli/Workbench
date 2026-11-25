"""Streaming exports retain scientific data and publish only complete files."""

from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import pytest
from openpyxl import Workbook, load_workbook

from workbench.core import io as core_io


def test_atomic_output_publishes_only_after_writer_finishes(tmp_path):
    destination = tmp_path / "results.xlsx"
    destination.write_bytes(b"previous result")

    with core_io.atomic_output(destination) as temporary:
        assert temporary.parent == destination.parent
        assert temporary.suffix == destination.suffix
        assert temporary != destination
        temporary.write_bytes(b"finished result")
        assert destination.read_bytes() == b"previous result"

    assert destination.read_bytes() == b"finished result"
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("existing", [False, True])
def test_atomic_output_preserves_destination_and_cleans_failed_write(tmp_path, existing):
    destination = tmp_path / "results.xlsx"
    if existing:
        destination.write_bytes(b"previous result")

    with pytest.raises(RuntimeError, match="write failed"):
        with core_io.atomic_output(destination) as temporary:
            temporary.write_bytes(b"incomplete result")
            raise RuntimeError("write failed")

    assert destination.exists() is existing
    if existing:
        assert destination.read_bytes() == b"previous result"
    assert list(tmp_path.iterdir()) == ([destination] if existing else [])


def test_atomic_output_preserves_destination_if_publish_fails(tmp_path, monkeypatch):
    destination = tmp_path / "results.xlsx"
    destination.write_bytes(b"previous result")

    def fail_replace(source, target):
        raise PermissionError("destination is open")

    monkeypatch.setattr(core_io.os, "replace", fail_replace)
    with pytest.raises(PermissionError, match="destination is open"):
        with core_io.atomic_output(destination) as temporary:
            temporary.write_bytes(b"finished result")

    assert destination.read_bytes() == b"previous result"
    assert list(tmp_path.iterdir()) == [destination]


def test_xlsx_save_failure_preserves_existing_workbook(tmp_path, monkeypatch):
    destination = tmp_path / "results.xlsx"
    destination.write_bytes(b"previous result")

    def fail_save(self, path):
        assert self.write_only
        assert path != destination
        path.write_bytes(b"incomplete ZIP archive")
        raise OSError("disk full")

    monkeypatch.setattr(Workbook, "save", fail_save)
    with pytest.raises(OSError, match="disk full"):
        core_io.write_xlsx({"Results": pd.DataFrame({"j": [1.2]})}, destination)

    assert destination.read_bytes() == b"previous result"
    assert list(tmp_path.iterdir()) == [destination]


def test_xlsx_validates_all_sheets_before_creating_output(tmp_path, monkeypatch):
    destination = tmp_path / "results.xlsx"
    monkeypatch.setattr(core_io, "MAX_TABLE_CELLS", 3)
    with pytest.raises(ValueError, match="too many cells"):
        core_io.write_xlsx({
            "Good": pd.DataFrame({"x": [1]}),
            "Oversized": pd.DataFrame({"x": [1, 2, 3]}),
        }, destination)
    assert not list(tmp_path.iterdir())


def test_xlsx_bad_cell_cleans_stream_files_and_preserves_destination(tmp_path, monkeypatch):
    from openpyxl.utils.exceptions import IllegalCharacterError
    from openpyxl.worksheet import _writer

    destination = tmp_path / "results.xlsx"
    destination.write_bytes(b"previous result")
    stream_files = []
    create_temporary_file = _writer.create_temporary_file

    def record_stream(*args, **kwargs):
        name = create_temporary_file(*args, **kwargs)
        stream_files.append(name)
        return name

    monkeypatch.setattr(_writer, "create_temporary_file", record_stream)
    with pytest.raises(IllegalCharacterError):
        core_io.write_xlsx({"Results": pd.DataFrame({"x": [1, "bad\x01text"]})}, destination)

    assert stream_files
    assert all(not core_io.Path(name).exists() for name in stream_files)
    assert destination.read_bytes() == b"previous result"
    assert list(tmp_path.iterdir()) == [destination]


def test_xlsx_streaming_retains_nullable_values_and_neutralizes_text(tmp_path, monkeypatch):
    def forbid_copy(*args, **kwargs):
        raise AssertionError("XLSX export must not copy/sanitize the whole table")

    monkeypatch.setattr(core_io, "_sanitized_dataframe", forbid_copy)
    values = [
        np.float64(0.123456789012345), np.int64(-2), True,
        datetime(2026, 11, 5, 12, 30), np.nan, pd.NA, pd.NaT,
        "=SUM(A1:A2)", " \t@SUM(1,2)", "'already literal",
        float("inf"), float("-inf"),
    ]
    path = tmp_path / "results.xlsx"
    core_io.write_xlsx({"Results": pd.DataFrame({
        "=User text": pd.Series(values, dtype=object),
        "Row": range(len(values)),
    })}, path)

    workbook = load_workbook(path)
    sheet = workbook.active
    assert sheet["A1"].value == "'=User text"
    assert sheet["A2"].value == pytest.approx(values[0], abs=1e-15)
    assert sheet["A3"].value == -2 and sheet["A3"].data_type == "n"
    assert sheet["A4"].value is True and sheet["A4"].data_type == "b"
    assert sheet["A5"].value == values[3]
    assert [sheet.cell(row, 1).value for row in (6, 7, 8)] == [None] * 3
    assert sheet["A9"].value == "'=SUM(A1:A2)" and sheet["A9"].data_type == "s"
    assert sheet["A10"].value == "' \t@SUM(1,2)"
    assert sheet["A11"].value == "'already literal"
    assert sheet["A12"].value == "inf"
    assert sheet["A13"].value == "-inf"
    workbook.close()


def test_xlsx_streaming_wraps_late_rows_without_resampling_all_data(tmp_path):
    frame = pd.DataFrame({"Condition": ["short"] * 300 + ["first line\nsecond line"]})
    path = tmp_path / "results.xlsx"
    core_io.write_xlsx({"Results": frame}, path)
    workbook = load_workbook(path)
    sheet = workbook.active
    assert sheet.column_dimensions["A"].width == 11
    assert sheet["A302"].alignment.wrap_text
    assert sheet.row_dimensions[302].height == 60
    assert sheet.freeze_panes == "A2"
    workbook.close()


def test_xlsx_streaming_supports_in_memory_project_artifacts():
    output = BytesIO()
    core_io.write_xlsx({"Curve": pd.DataFrame({"j (A/cm²)": [0.0, 1.5]})}, output)
    assert output.getvalue().startswith(b"PK")
    output.seek(0)
    workbook = load_workbook(output)
    assert list(workbook.active.values) == [("j (A/cm²)",), (0,), (1.5,)]
    workbook.close()
