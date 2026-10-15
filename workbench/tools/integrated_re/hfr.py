"""Fast measured-spectrum HFR extraction for Integrated RE analysis.

Each file contributes one equally weighted HFR estimate. Impedances from
different files are never averaged, and no KK test or model fit is performed.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ...core.data_io import read_eis_data
from ..eis.logic_eis import EisLogic


ESTIMATE_COLUMNS = [
    "Role", "File", "Source path", "HFR (Ω·cm²)", "HFR method", "Points",
    "Status", "Error", "Source impedance unit", "Area applied (cm²)",
    "Real impedance scale", "Imaginary impedance scale",
]
SPECTRUM_COLUMNS = [
    "Role", "File", "Source path", "Frequency (Hz)",
    "Re(Z) (Ω·cm²)", "Im(Z) (Ω·cm²)",
]
INTERSECTION_METHOD = "High-frequency x-axis intersection"
FALLBACK_METHOD = "Minimum real impedance (no x-axis intersection)"


def extract_role_hfr(files, *, area_cm2=0.0, role="anode"):
    """Return mean HFR, per-file estimates, cleaned spectra, and warnings.

    Finite positive-frequency measurements are sorted from high to low
    frequency. Exact repeated frequencies are averaged only within a file.
    Use the first measured/interpolated Im(Z)=0 crossing, falling back to the
    minimum Re(Z) without extrapolation. Failed files remain in the estimate
    table but do not contribute to the arithmetic mean. No supplied files
    returns ``None``; supplied files with no usable HFR raise ``ValueError``.
    """
    estimates, spectra, warnings = [], [], []
    for source in files:
        path = Path(source)
        label = f"{role.capitalize()} EIS ({path.name})"
        row = {
            "Role": role, "File": path.name, "Source path": str(path),
            "HFR (Ω·cm²)": np.nan, "HFR method": "", "Points": 0,
            "Status": "Failed", "Error": "",
            "Source impedance unit": "", "Area applied (cm²)": np.nan,
            "Real impedance scale": np.nan, "Imaginary impedance scale": np.nan,
        }
        try:
            data, error = read_eis_data(str(path), area_cm2=area_cm2)
            if error:
                raise ValueError(" ".join(error.splitlines()))
            row.update({
                "Source impedance unit": data.get("source_impedance_unit", "unknown"),
                "Area applied (cm²)": (data.get("area_cm2")
                                       if data.get("area_cm2") is not None else np.nan),
                "Real impedance scale": data.get("source_impedance_scale_real", 1.0),
                "Imaginary impedance scale": data.get("source_impedance_scale_imag", 1.0),
            })
            if data.get("warning"):
                warnings.append(f"{label}: {data['warning']}")

            freq = np.asarray(data["frequency"], dtype=float)
            zr = np.asarray(data["z_real"], dtype=float)
            zi = np.asarray(data["z_imag"], dtype=float)
            zc = np.asarray(data["z_complex"], dtype=complex)
            if not (len(freq) == len(zr) == len(zi) == len(zc)):
                raise ValueError("Impedance columns have different lengths.")
            usable = (np.isfinite(freq) & (freq > 0) & np.isfinite(zr)
                      & np.isfinite(zi) & np.isfinite(zc))
            dropped = int(np.count_nonzero(~usable))
            if dropped:
                warnings.append(
                    f"{label}: dropped {dropped} non-finite or "
                    "non-positive-frequency point(s)."
                )
            points = pd.DataFrame({
                "Freq(Hz)": freq[usable], "Z'(Ohm.cm²)": zr[usable],
                "Z''(Ohm.cm²)": zi[usable],
            })
            counts = points["Freq(Hz)"].value_counts()
            repeated = int((counts > 1).sum())
            if repeated:
                before = len(points)
                points = points.groupby("Freq(Hz)", as_index=False).mean()
                warnings.append(
                    f"{label}: averaged repeated measurements at {repeated} "
                    f"duplicate frequency value(s), reducing {before} "
                    f"to {len(points)} points."
                )
            points = points.sort_values("Freq(Hz)", ascending=False).reset_index(drop=True)
            row["Points"] = len(points)
            if not points.empty:
                exported = points.rename(columns={
                    "Freq(Hz)": "Frequency (Hz)",
                    "Z'(Ohm.cm²)": "Re(Z) (Ω·cm²)",
                    "Z''(Ohm.cm²)": "Im(Z) (Ω·cm²)",
                })
                exported.insert(0, "Source path", str(path))
                exported.insert(0, "File", path.name)
                exported.insert(0, "Role", role)
                spectra.append(exported)
            if len(points) < 2:
                raise ValueError(
                    f"Only {len(points)} usable unique frequency point(s); need at least 2."
                )

            hfr, intersection = EisLogic.estimate_hfr(points)
            row["HFR method"] = INTERSECTION_METHOD if intersection else FALLBACK_METHOD
            if not np.isfinite(hfr) or hfr < 0:
                raise ValueError("HFR must be finite and nonnegative.")
            if not intersection:
                warnings.append(
                    f"{label}: no x-axis intersection; HFR uses minimum real "
                    "impedance as a fallback estimate."
                )
            row.update({"HFR (Ω·cm²)": float(hfr), "Status": "OK"})
        except Exception as exc:
            row["Error"] = str(exc)
            warnings.append(f"{label}: {exc}")
        estimates.append(row)

    estimate_table = pd.DataFrame(estimates, columns=ESTIMATE_COLUMNS)
    spectrum_table = (pd.concat(spectra, ignore_index=True)[SPECTRUM_COLUMNS]
                      if spectra else pd.DataFrame(columns=SPECTRUM_COLUMNS))
    if estimate_table.empty:
        return None, estimate_table, spectrum_table, warnings
    valid = estimate_table.loc[estimate_table["Status"] == "OK", "HFR (Ω·cm²)"]
    if valid.empty:
        raise ValueError(
            f"No usable HFR could be extracted for {role} EIS. " + " ".join(warnings)
        )
    return float(valid.mean()), estimate_table, spectrum_table, warnings
