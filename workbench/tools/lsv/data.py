"""Choose CV cycle branches before applying the ordinary LSV analysis pipeline."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from workbench.core.cv_io import read_cv_scans
from workbench.core.data_io import _combine_lsv_series, read_lsv_data


ALL_MEASURED_VOLTAGES = "All measured voltages"


def read_decoupling_data(filepaths, params: dict, reaction="auto"):
    """Return one selected branch per input, with the same units/signs as LSV.

    ``cv_cycle`` is ``Last`` (or zero), or a positive source cycle number. Last
    resolves independently for each file, before checking that cycle's branch.
    An unavailable requested cycle or branch is an error rather than a fallback.
    """
    input_type = params.get("input_type", "LSV")
    if input_type == "LSV":
        return read_lsv_data(
            filepaths, reaction=reaction,
            use_available_range=params.get("average_range") == ALL_MEASURED_VOLTAGES,
        )
    if input_type != "CV":
        return None, "Input type must be LSV or CV."

    branch = params.get("cv_scan", "Forward")
    if branch not in {"Forward", "Back"}:
        return None, "CV scan must be Forward or Back."
    cycle_value = params.get("cv_cycle", "Last")
    cycle_text = str(cycle_value).strip()
    if cycle_text.lower() == "last" or cycle_text == "0":
        cycle_number = None
    elif re.fullmatch(r"[0-9]+", cycle_text) and int(cycle_text) > 0:
        cycle_number = int(cycle_text)
    else:
        return None, "CV cycle must be Last or a positive integer cycle number."
    if filepaths is None:
        return None, "No file selected."
    paths = ([filepaths] if isinstance(filepaths, (str, Path)) else
             [path for path in filepaths if path])
    if not paths:
        return None, "No file selected."

    series, file_infos, warnings, selections = [], [], [], []
    used_labels = set()
    for filepath in paths:
        data, error = read_cv_scans(filepath, branch=branch)
        if error:
            return None, error
        scans, source_info = data
        if not scans:
            return None, f"{Path(filepath).name}: No CV cycles were found."
        if cycle_number is None:
            scan = scans[-1]
        else:
            scan = next((scan for scan in scans if scan.cycle == cycle_number), None)
            if scan is None:
                available = ", ".join(str(scan.cycle) for scan in scans)
                return None, (f"{Path(filepath).name}: Cycle {cycle_number} is unavailable. "
                              f"Available cycles: {available}.")
        if scan.error:
            return None, f"{Path(filepath).name}: {scan.error}"

        finite = np.isfinite(scan.potential) & np.isfinite(scan.current)
        voltage, current = scan.potential[finite], scan.current[finite]
        if len(voltage) < 2:
            return None, (f"{Path(filepath).name}: Cycle {scan.cycle} {branch.lower()} scan "
                          "needs at least two finite potential/current points.")
        order = np.argsort(voltage, kind="stable")
        file_info = {
            "file": Path(filepath).name, "v_col": scan.potential_column,
            "i_col": scan.current_column, "is_density": scan.is_density,
            "sweep_trimmed": False, "n_rows": len(voltage),
            "cycle": scan.cycle, "scan": branch, "direction": scan.direction,
        }
        file_infos.append(file_info)
        series.append((voltage[order], current[order], file_info))
        warnings.extend(f"{Path(filepath).name}: {warning}"
                        for warning in source_info.get("warnings", []))
        label = Path(filepath).name
        suffix = 2
        while label in used_labels:
            label = f"{Path(filepath).name} ({suffix})"
            suffix += 1
        used_labels.add(label)
        selections.append({"File": label, "Cycle": scan.cycle,
                           "Scan": branch, "Direction": scan.direction})

    combined, error = _combine_lsv_series(
        series, file_infos, warnings, str(reaction).strip().lower(),
    )
    if error:
        return None, error
    voltage, current, info = combined
    info.update({"input_type": "CV", "cv_scan": branch,
                 "cv_cycle": "Last" if cycle_number is None else cycle_number,
                 "cv_selections": selections})
    return (voltage, current, info), None
