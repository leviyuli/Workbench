"""Identified, acquisition-ordered scans and saved-state migration for batch analysis."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from workbench.core.cv_io import read_cv_scans


@dataclass(frozen=True)
class BatchScan:
    code: str
    source: str
    file: str
    path: Path
    cycle: int | None
    branch: str
    direction: str
    potential: np.ndarray
    current: np.ndarray
    is_density: bool
    source_indices: np.ndarray
    error: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _failed(source: str, label: str, path: Path, branch: str,
            error: str, cycle: int | None = None) -> BatchScan:
    return BatchScan(
        code=f"{source}-C{cycle:02d}" if cycle is not None else source,
        source=source, file=label, path=path, cycle=cycle, branch=branch,
        direction="Unknown", potential=np.array([], dtype=float),
        current=np.array([], dtype=float), is_density=True,
        source_indices=np.array([], dtype=int), error=error,
    )


def read_batch_scans(files, params: dict) -> list[BatchScan]:
    """Return each selected scan separately, retaining failed source identities.

    Unit conversion to V and A or A/cm² belongs to the shared reader. Absolute
    current normalization belongs to the numerical caller. No rows are sorted
    or removed here, so invalid acquisition rows continue to separate segments.
    ``source_indices`` identifies zero-based rows in the source data table,
    before cycle/branch selection, sign filtering, reversal or processing.
    """
    input_type = params.get("input_type", "LSV")
    if input_type not in {"LSV", "CV"}:
        raise ValueError("Input type must be LSV or CV.")
    branch = params.get("cv_scan", "Forward") if input_type == "CV" else "Forward"
    if branch not in {"Forward", "Back"}:
        raise ValueError("CV scan must be Forward or Back.")
    cycle_mode = params.get("cv_cycle_mode", "All")
    cycle_number = None
    if input_type == "CV":
        if cycle_mode not in {"All", "Last", "Specific"}:
            raise ValueError("CV cycles must be All, Last or Specific.")
        if cycle_mode == "Specific":
            value = params.get("cv_cycle", 1)
            try:
                cycle_number = int(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("CV cycle number must be a positive integer.") from exc
            if isinstance(value, bool) or str(value).strip() != str(cycle_number) or cycle_number < 1:
                raise ValueError("CV cycle number must be a positive integer.")
    paths = ([files] if isinstance(files, (str, Path)) else list(files or []))
    if not paths:
        raise ValueError("Add at least one LSV or CV file for batch analysis.")
    result = []
    used_labels = set()
    for number, filepath in enumerate(paths, start=1):
        path = Path(filepath)
        source = f"S{number:02d}"
        label = path.name
        suffix = 2
        while label in used_labels:
            label = f"{path.name} ({suffix})"
            suffix += 1
        used_labels.add(label)
        data, error = read_cv_scans(path, branch=branch)
        if error:
            result.append(_failed(source, label, path, branch, error))
            continue
        scans, info = data
        if not scans:
            result.append(_failed(source, label, path, branch, "No acquired scans were found."))
            continue
        if input_type == "LSV":
            chosen = scans[:1]
        elif cycle_mode == "Last":
            chosen = scans[-1:]
        elif cycle_mode == "Specific":
            chosen = [scan for scan in scans if scan.cycle == cycle_number]
            if not chosen:
                available = ", ".join(str(scan.cycle) for scan in scans)
                result.append(_failed(
                    source, label, path, branch,
                    f"Cycle {cycle_number} is unavailable. Available cycles: {available}.",
                    cycle=cycle_number,
                ))
                continue
        else:
            chosen = scans
        for scan in chosen:
            cycle = scan.cycle if input_type == "CV" else None
            result.append(BatchScan(
                code=f"{source}-C{cycle:02d}" if cycle is not None else source,
                source=source, file=label, path=path, cycle=cycle, branch=branch,
                direction=scan.direction, potential=scan.potential.copy(),
                current=scan.current.copy(), is_density=scan.is_density,
                source_indices=(scan.source_indices.copy() if scan.source_indices is not None
                                else np.arange(len(scan.potential), dtype=int)), error=scan.error,
                warnings=tuple(info.get("warnings", [])),
            ))
    return result


def migrate_batch_page_state(state: dict) -> dict:
    """Migrate inputs and annotate historical fit methods without recalculation."""
    if not isinstance(state, dict) or state.get("tool_id") != "lsv_ccd":
        return state
    values = state.get("panel", {}).get("values", {})
    if not isinstance(values, dict):
        return state
    legacy = "hfr" not in values and any(
        key in values for key in ("reference_hfr", "baseline_mode", "baseline_lower")
    )
    if not legacy:
        snapshot = state.get("result")
        if not isinstance(snapshot, dict):
            return state
        scalars = snapshot.get("scalars", {})
        if (not isinstance(scalars, dict) or "Tafel method" in scalars
                or not str(scalars.get("Method", "")).startswith("residual-baseline")):
            return state
        annotation = (
            "Historical batch result using the nested-window Tafel fit. Saved numerical "
            "results, plots and exports are retained. Run the batch analysis again to use "
            "the contiguous stable-slope Tafel method."
        )
        if snapshot.get("text_blocks", {}).get("Saved Tafel method") == annotation:
            return state
        migrated = deepcopy(state)
        migrated["result"].setdefault("text_blocks", {})["Saved Tafel method"] = annotation
        return migrated
    migrated = deepcopy(state)
    updated = {
        "input_type": "CV", "cv_scan": "Forward", "cv_cycle_mode": "All", "cv_cycle": 1,
        "ccd_fit_mode": "Auto", "include_rcl": False,
        "hfr_mode": "Constant HFR", "hfr": values.get("reference_hfr", .12),
    }
    for key in ("electrode_area", "current_branch", "ccd_threshold_mv"):
        if key in values:
            updated[key] = values[key]
    migrated["panel"]["values"] = updated
    migrated["panel"]["annotations"] = {}
    if migrated.get("extra") is None:
        migrated["extra"] = {"hfr_points": []}
    snapshot = migrated.get("result")
    if isinstance(snapshot, dict):
        snapshot["display_tables"] = False
        snapshot["display_summary"] = True
        snapshot.setdefault("text_blocks", {})["Analysis method"] = (
            "Historical measured-voltage CCD result. Saved plots and exports are unchanged. "
            "Run Batch decoupling & CCD to calculate the residual-based method."
        )
        migrated["result_tab"] = 0
    return migrated
