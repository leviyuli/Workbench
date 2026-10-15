"""Portable, versioned Workbench projects using tagged JSON and embedded files.

The UI supplies a state dictionary containing explicit ``Path`` input references.
Those files are embedded, and references are redirected to a temporary project
directory on open. Results are snapshots, independent of editable input state.
"""
from __future__ import annotations

import base64
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from typing import Any
import zipfile

import numpy as np
import pandas as pd

from workbench.core import io
from workbench.core.tool_base import Artifact, ToolResult

FORMAT_NAME = "electrochemistry-workbench"
FORMAT_VERSION = 1
MAX_ARCHIVE_ENTRIES = 50_000
MAX_UNCOMPRESSED_BYTES = 2 * 1024**3
MAX_MANIFEST_BYTES = 256 * 1024**2
_TAG = "$ecwb"


class ProjectError(ValueError):
    """An incomplete, unsupported, or unreadable project file."""


def _dtype_state(dtype) -> Any:
    if isinstance(dtype, pd.CategoricalDtype):
        return {"categorical": list(dtype.categories), "ordered": dtype.ordered}
    return str(dtype)


def _restore_dtype(value):
    if isinstance(value, dict):
        return pd.CategoricalDtype(value["categorical"], ordered=value["ordered"])
    return value


def _index_state(index: pd.Index) -> dict:
    if isinstance(index, pd.RangeIndex):
        return {"kind": "range", "start": index.start, "stop": index.stop,
                "step": index.step, "name": index.name}
    if isinstance(index, pd.MultiIndex):
        return {"kind": "multi", "levels": [_index_state(v) for v in index.levels],
                "codes": [v.tolist() for v in index.codes], "names": list(index.names),
                "sortorder": index.sortorder}
    return {"kind": "index", "values": index.tolist(),
            "dtype": _dtype_state(index.dtype), "name": index.name,
            "freq": getattr(index, "freqstr", None)}


def _restore_index(state: dict) -> pd.Index:
    kind = state["kind"]
    if kind == "range":
        return pd.RangeIndex(state["start"], state["stop"], state["step"],
                             name=state["name"])
    if kind == "multi":
        return pd.MultiIndex(levels=[_restore_index(v) for v in state["levels"]],
                             codes=state["codes"], names=state["names"],
                             sortorder=state["sortorder"], verify_integrity=True)
    if kind != "index":
        raise ProjectError(f"Unknown table index kind: {kind}")
    index = pd.Index(state["values"], dtype=_restore_dtype(state["dtype"]),
                     name=state["name"], tupleize_cols=False)
    if state.get("freq") and isinstance(index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        index.freq = state["freq"]
    return index


def _encode(value: Any, path_ids=None, blob_writer=None) -> Any:
    """Use tagged JSON for types JSON alone cannot preserve losslessly."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if value is pd.NA:
        return {_TAG: "missing"}
    if value is pd.NaT:
        return {_TAG: "nat"}
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {_TAG: "float", "value": "nan" if math.isnan(value) else
                ("inf" if value > 0 else "-inf")}
    enc = lambda item: _encode(item, path_ids, blob_writer)
    if isinstance(value, Path):
        if path_ids is None:
            return {_TAG: "path", "value": str(value)}
        return {_TAG: "file", "id": path_ids[str(value)]}
    if isinstance(value, bytes):
        if blob_writer is not None:
            return {_TAG: "blob", **blob_writer(value)}
        return {_TAG: "bytes", "base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, pd.DataFrame):
        return {_TAG: "dataframe", "columns": enc(_index_state(value.columns)),
                "index": enc(_index_state(value.index)),
                "data": [enc(value.iloc[:, n].tolist()) for n in range(len(value.columns))],
                "dtypes": [enc(_dtype_state(t)) for t in value.dtypes],
                "attrs": enc(value.attrs)}
    if isinstance(value, pd.Timestamp):
        return {_TAG: "timestamp", "value": value.isoformat()}
    if isinstance(value, pd.Timedelta):
        return {_TAG: "pandas_timedelta", "value": value.value}
    if isinstance(value, pd.Period):
        return {_TAG: "period", "value": str(value), "freq": value.freqstr}
    if isinstance(value, np.ndarray):
        return {_TAG: "ndarray", "dtype": str(value.dtype), "shape": list(value.shape),
                "data": enc(value.ravel().tolist())}
    if isinstance(value, np.generic):
        if isinstance(value, (np.datetime64, np.timedelta64)):
            return {_TAG: "numpy_time", "dtype": str(value.dtype),
                    "value": int(value.view("i8"))}
        return enc(value.item())
    if isinstance(value, complex):
        return {_TAG: "complex", "real": enc(value.real), "imag": enc(value.imag)}
    if isinstance(value, Decimal):
        return {_TAG: "decimal", "value": str(value)}
    if isinstance(value, (datetime, date, time)):
        return {_TAG: type(value).__name__, "value": value.isoformat()}
    if isinstance(value, timedelta):
        return {_TAG: "timedelta", "days": value.days, "seconds": value.seconds,
                "microseconds": value.microseconds}
    if isinstance(value, Mapping):
        return {_TAG: "dict", "items": [[enc(k), enc(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple, set)):
        return {_TAG: type(value).__name__, "items": [enc(v) for v in value]}
    raise ProjectError(f"Cannot save project value of type {type(value).__name__}.")


def _decode(value: Any, files=None, blob_reader=None) -> Any:
    if not isinstance(value, dict):
        if isinstance(value, list):
            return [_decode(v, files, blob_reader) for v in value]
        return value
    kind = value.get(_TAG)
    dec = lambda item: _decode(item, files, blob_reader)
    if kind == "dict":
        return {dec(k): dec(v) for k, v in value["items"]}
    if kind in ("list", "tuple", "set"):
        items = [dec(v) for v in value["items"]]
        return {"list": list, "tuple": tuple, "set": set}[kind](items)
    if kind == "float":
        if value["value"] not in ("nan", "inf", "-inf"):
            raise ProjectError("Invalid nonfinite number.")
        return float(value["value"])
    if kind == "missing":
        return pd.NA
    if kind == "nat":
        return pd.NaT
    if kind == "path":
        # Archive states must reference embedded files, never arbitrary paths.
        if files is not None:
            raise ProjectError("Project contains an unembedded input path.")
        return Path(value["value"])
    if kind == "file":
        if files is None or value["id"] not in files:
            raise ProjectError("Project references a missing input file.")
        return files[value["id"]]
    if kind == "bytes":
        return base64.b64decode(value["base64"], validate=True)
    if kind == "blob":
        if blob_reader is None:
            raise ProjectError("Missing project binary data reader.")
        return blob_reader(value)
    if kind == "dataframe":
        columns = _restore_index(dec(value["columns"]))
        index = _restore_index(dec(value["index"]))
        if len(columns) != len(value["data"]) or len(columns) != len(value["dtypes"]):
            raise ProjectError("Project table has inconsistent column metadata.")
        series = []
        for raw, dtype in zip(value["data"], value["dtypes"]):
            data = dec(raw)
            if len(data) != len(index):
                raise ProjectError("Project table has inconsistent row metadata.")
            series.append(pd.Series(data, dtype=_restore_dtype(dec(dtype))))
        frame = pd.concat(series, axis=1) if series else pd.DataFrame(index=range(len(index)))
        frame.columns, frame.index = columns, index
        frame.attrs = dec(value.get("attrs", {_TAG: "dict", "items": []}))
        return frame
    if kind == "timestamp":
        return pd.Timestamp(value["value"])
    if kind == "pandas_timedelta":
        return pd.Timedelta(value["value"], unit="ns")
    if kind == "period":
        return pd.Period(value["value"], freq=value["freq"])
    if kind == "ndarray":
        dtype = np.dtype(value["dtype"])
        if dtype.hasobject and dtype != np.dtype("object"):
            raise ProjectError("Structured object arrays are unsupported.")
        return np.asarray(dec(value["data"]), dtype=dtype).reshape(value["shape"])
    if kind == "numpy_time":
        return np.array(value["value"], dtype="i8").view(value["dtype"])[()]
    if kind == "complex":
        return complex(dec(value["real"]), dec(value["imag"]))
    if kind == "decimal":
        return Decimal(value["value"])
    if kind in ("datetime", "date", "time"):
        return {"datetime": datetime, "date": date, "time": time}[kind].fromisoformat(value["value"])
    if kind == "timedelta":
        return timedelta(days=value["days"], seconds=value["seconds"],
                         microseconds=value["microseconds"])
    raise ProjectError(f"Unknown project value type: {kind!r}")


def encode_value(value: Any) -> Any:
    """Encode a value into JSON-safe, precision-preserving structured data."""
    return _encode(value)


def decode_value(value: Any) -> Any:
    """Decode a tagged JSON value."""
    try:
        return _decode(value)
    except ProjectError:
        raise
    except Exception as exc:
        raise ProjectError(f"Invalid saved value: {exc}") from exc


def artifact_bytes(artifact: Artifact) -> bytes:
    """Freeze an export using the export button's writers."""
    existing = getattr(artifact, "serialized_payload", None)
    if existing is not None:
        return bytes(existing)
    suffix = ".xlsx" if artifact.kind == "workbook" else "." + artifact.kind
    with tempfile.TemporaryDirectory(prefix="ecwb-export-") as folder:
        target = Path(folder) / ("artifact" + suffix)
        if artifact.kind == "csv":
            io.write_csv(artifact.payload, target)
        elif artifact.kind == "xlsx":
            io.write_xlsx(artifact.payload, target)
        elif artifact.kind == "workbook":
            artifact.payload.save(target)
        elif artifact.kind == "png":
            io.save_figure(artifact.payload, target)
        elif artifact.kind == "txt":
            io.write_txt(artifact.payload, target)
        else:
            raise ProjectError(f"Unknown export kind: {artifact.kind}")
        return target.read_bytes()


def snapshot_result(result: ToolResult, artifacts: list[Artifact]) -> dict:
    """Capture displayed results and exports, excluding tool runtime payloads."""
    figures = dict(getattr(result, "figure_images", {}))
    for name, figure in result.figures.items():
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
        figures[name] = buffer.getvalue()
    return {
        "figures": figures, "tables": dict(result.tables), "scalars": dict(result.scalars),
        "text_blocks": dict(result.text_blocks), "warnings": list(result.warnings),
        "param_updates": dict(result.param_updates),
        "param_annotations": dict(result.param_annotations),
        "display_tables": result.display_tables, "display_summary": result.display_summary,
        "artifacts": [{"name": art.name, "kind": art.kind,
                       "default_filename": art.default_filename,
                       "bytes": artifact_bytes(art)} for art in artifacts],
    }


def restore_result(snapshot: dict) -> tuple[ToolResult, list[Artifact]]:
    """Restore image plots and frozen exports without invoking any tool."""
    from PIL import Image

    try:
        figures = snapshot["figures"]
        for data in figures.values():
            if not isinstance(data, bytes):
                raise ProjectError("Saved figure is not binary image data.")
            with Image.open(BytesIO(data)) as picture:
                if picture.format != "PNG":
                    raise ProjectError("Saved figure is not a PNG image.")
                picture.verify()
        if not all(isinstance(v, pd.DataFrame) for v in snapshot["tables"].values()):
            raise ProjectError("Saved result contains an invalid table.")
        result = ToolResult(
            figure_images=dict(figures), tables=dict(snapshot["tables"]),
            scalars=dict(snapshot["scalars"]), text_blocks=dict(snapshot["text_blocks"]),
            warnings=list(snapshot["warnings"]), param_updates=dict(snapshot["param_updates"]),
            param_annotations=dict(snapshot["param_annotations"]),
            display_tables=bool(snapshot["display_tables"]),
            display_summary=bool(snapshot["display_summary"]),
        )
        artifacts = []
        for item in snapshot["artifacts"]:
            if item["kind"] not in ("csv", "xlsx", "workbook", "png", "txt"):
                raise ProjectError(f"Unknown saved export kind: {item['kind']}")
            if not isinstance(item["bytes"], bytes):
                raise ProjectError("Saved export is not binary data.")
            artifacts.append(Artifact(item["name"], item["kind"], item["default_filename"],
                                      None, serialized_payload=item["bytes"]))
        return result, artifacts
    except ProjectError:
        raise
    except Exception as exc:
        raise ProjectError(f"Invalid saved result: {exc}") from exc


def _collect_paths(value: Any) -> Iterable[Path]:
    if isinstance(value, Path):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _collect_paths(key)
            yield from _collect_paths(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _collect_paths(item)
    # DataFrames contain result values, not input selections.


def _safe_name(name: str) -> bool:
    return bool(name and name not in (".", "..") and
                not any(c in name for c in '/\\:\x00') and
                not any(ord(c) < 32 for c in name))


def save_project(path: Path | str, state: dict, files: Iterable[Path] = (), *,
                 source_metadata: Mapping[str, dict] | None = None) -> None:
    """Atomically save all state and inputs; leave an existing target intact on error."""
    path = Path(path)
    if not isinstance(state, dict):
        raise ProjectError("Workspace state must be a dictionary.")
    source_metadata = source_metadata or {}
    ordered_files = dict.fromkeys(str(Path(p)) for p in [*files, *_collect_paths(state)])
    path_ids = {source: f"f{n:06d}" for n, source in enumerate(ordered_files)}
    temporary = None
    try:
        descriptor, temp_name = tempfile.mkstemp(prefix=".ecwb-", suffix=".tmp", dir=path.parent)
        temporary = Path(temp_name)
        os.close(descriptor)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as archive:
            entries = []
            for source, identifier in path_ids.items():
                source_path = Path(source)
                if not source_path.is_file():
                    raise ProjectError(f"Input file is unavailable: {source}")
                metadata = source_metadata.get(source, {})
                name = metadata.get("original_name", source_path.name)
                if not _safe_name(name):
                    raise ProjectError(f"Invalid input filename: {name!r}")
                member = f"inputs/{identifier}/{name}"
                digest = hashlib.sha256()
                size = 0
                with source_path.open("rb") as source_stream, archive.open(member, "w") as target:
                    for chunk in iter(lambda: source_stream.read(1024**2), b""):
                        digest.update(chunk)
                        size += len(chunk)
                        if size > MAX_UNCOMPRESSED_BYTES:
                            raise ProjectError(f"Input file is too large: {name}")
                        target.write(chunk)
                entries.append({"id": identifier, "member": member, "original_name": name,
                                "original_path": metadata.get("original_path", source),
                                "mtime_ns": metadata.get("mtime_ns", source_path.stat().st_mtime_ns),
                                "size": size, "sha256": digest.hexdigest()})
            blobs = {}

            def write_blob(data):
                checksum = hashlib.sha256(data).hexdigest()
                member = f"objects/{checksum}.bin"
                if member not in blobs:
                    archive.writestr(member, data)
                    blobs[member] = True
                return {"member": member, "size": len(data), "sha256": checksum}

            manifest = {"format": FORMAT_NAME, "version": FORMAT_VERSION,
                        "files": entries, "state": _encode(state, path_ids, write_blob)}
            raw_manifest = json.dumps(manifest, ensure_ascii=False, allow_nan=False,
                                      separators=(",", ":")).encode("utf-8")
            if len(raw_manifest) > MAX_MANIFEST_BYTES:
                raise ProjectError("Workspace manifest is too large.")
            archive.writestr("manifest.json", raw_manifest)
            members = archive.infolist()
            if (len(members) > MAX_ARCHIVE_ENTRIES or
                    sum(item.file_size for item in members) > MAX_UNCOMPRESSED_BYTES):
                raise ProjectError("Workspace exceeds the project size limit.")
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except ProjectError:
        raise
    except Exception as exc:
        raise ProjectError(f"Could not save project: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass
class LoadedProject:
    """Own the extracted sources for as long as the UI references their paths."""
    path: Path
    state: dict
    files: dict[str, Path]
    source_metadata: dict[str, dict]
    _directory: tempfile.TemporaryDirectory

    def close(self) -> None:
        self._directory.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def load_project(path: Path | str) -> LoadedProject:
    """Validate and decode before exposing a replacement workspace to the UI."""
    path = Path(path)
    directory = tempfile.TemporaryDirectory(prefix="ecwb-project-")
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = {item.filename for item in members}
            if len(members) > MAX_ARCHIVE_ENTRIES or len(names) != len(members):
                raise ProjectError("Project has too many or duplicate archive entries.")
            if sum(item.file_size for item in members) > MAX_UNCOMPRESSED_BYTES:
                raise ProjectError("Project exceeds the decompressed size limit.")
            for item in members:
                parts = PurePosixPath(item.filename).parts
                if (not parts or item.filename.startswith("/") or
                        any(p in (".", "..") or not _safe_name(p) for p in parts) or
                        stat.S_ISLNK(item.external_attr >> 16)):
                    raise ProjectError("Project contains an unsafe archive path.")
            if "manifest.json" not in names:
                raise ProjectError("Project manifest is missing.")
            if archive.getinfo("manifest.json").file_size > MAX_MANIFEST_BYTES:
                raise ProjectError("Project manifest is too large.")
            manifest = json.loads(archive.read("manifest.json"),
                                  parse_constant=lambda v: (_ for _ in ()).throw(
                                      ProjectError(f"Invalid JSON number: {v}")))
            if manifest.get("format") != FORMAT_NAME:
                raise ProjectError("This file is not a Workbench project.")
            if type(manifest.get("version")) is not int or manifest["version"] != FORMAT_VERSION:
                raise ProjectError(f"Unsupported project format version: {manifest.get('version')}")
            file_ids, files, metadata, binary_cache = {}, {}, {}, {}

            def checked_bytes(entry):
                member = entry["member"]
                if member not in names or archive.getinfo(member).file_size != entry["size"]:
                    raise ProjectError(f"Project data is missing or truncated: {member}")
                # Figures and exports can share a deduplicated binary blob.
                data = binary_cache.get(member)
                if data is None:
                    data = archive.read(member)
                if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                    raise ProjectError(f"Project data checksum failed: {member}")
                if member.startswith("objects/"):
                    binary_cache[member] = data
                return data

            for entry in manifest["files"]:
                identifier, name = entry["id"], entry["original_name"]
                if (not _safe_name(identifier) or not _safe_name(name) or identifier in file_ids or
                        entry["member"] != f"inputs/{identifier}/{name}"):
                    raise ProjectError("Project contains invalid input-file metadata.")
                if not isinstance(entry["original_path"], str):
                    raise ProjectError("Project contains an invalid original file path.")
                target = Path(directory.name) / identifier / name
                target.parent.mkdir()
                target.write_bytes(checked_bytes(entry))
                mtime = entry["mtime_ns"]
                if type(mtime) is not int or mtime < 0:
                    raise ProjectError("Project contains an invalid input modification time.")
                os.utime(target, ns=(mtime, mtime))
                file_ids[identifier] = target
                # File IDs distinguish a re-added original from its embedded copy.
                files.setdefault(entry["original_path"], target)
                metadata[str(target)] = {key: entry[key] for key in
                                         ("original_path", "original_name", "mtime_ns")}
            state = _decode(manifest["state"], file_ids, checked_bytes)
            if not isinstance(state, dict):
                raise ProjectError("Project workspace state must be a dictionary.")
            return LoadedProject(path, state, files, metadata, directory)
    except ProjectError:
        directory.cleanup()
        raise
    except Exception as exc:
        directory.cleanup()
        raise ProjectError(f"Could not open project: {exc}") from exc
