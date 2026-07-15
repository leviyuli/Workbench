from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from workbench.core.io import read_text as _read_bounded_text

from workbench import theme


@dataclass
class Spectrum:
    path: Path
    x: np.ndarray
    intensity: np.ndarray
    technique: str
    source_type: str
    x_column: str
    y_column: str
    warnings: list[str] = field(default_factory=list)

    @property
    def wavenumber(self) -> np.ndarray:
        """Backward-compatible FTIR name used by existing callers/tests."""
        return self.x


FtirSpectrum = Spectrum


def _read_text(path: Path) -> str:
    return _read_bounded_text(path)


def _split_line(line: str, delim: str | None = None) -> list[str]:
    stripped = line.strip()
    if not stripped:
        return []
    if delim:
        return [t.strip() for t in stripped.split(delim)]
    if "\t" in stripped:
        return [t.strip() for t in stripped.split("\t")]
    if "," in stripped:
        return [t.strip() for t in stripped.split(",")]
    if ";" in stripped:
        return [t.strip() for t in stripped.split(";")]
    return stripped.split()


def _sniff_delimiter(lines: list[str]) -> str:
    sample = "\n".join(lines[:40])
    try:
        return csv.Sniffer().sniff(sample, delimiters="\t,; ").delimiter
    except csv.Error:
        pass
    first = next((ln for ln in lines if ln.strip()), "")
    if "\t" in first:
        return "\t"
    if "," in first:
        return ","
    if ";" in first:
        return ";"
    return r"\s+"


def _norm_label(label: object) -> str:
    s = str(label).strip().lower()
    replacements = {
        "\u207b": "-",
        "\u2212": "-",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u00b9": "1",
        "\u2071": "i",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return re.sub(r"[\s_]+", "", s)


def _is_wavenumber(label: object) -> bool:
    s = _norm_label(label)
    return (
        "wavenumber" in s
        or s in {"wn", "nu"}
        or "cm-1" in s
        or "cm^-1" in s
        or "1/cm" in s
    )


def _is_xps_energy(label: object) -> bool:
    s = _norm_label(label)
    return (
        "bindingenergy" in s
        or "kineticenergy" in s
        or s in {"be", "be(ev)", "energy", "energy(ev)", "ev"}
        or (s.startswith("be") and "ev" in s)
    )


def _intensity_kind(label: object) -> str | None:
    s = _norm_label(label)
    if "transmittance" in s or "transmission" in s or s in {"t", "t%", "%t"} or "t%" in s:
        return "T"
    if "absorbance" in s or s in {"a", "a%", "abs", "abs."} or "a%" in s:
        return "A"
    return None


def _is_xps_intensity(label: object) -> bool:
    s = _norm_label(label)
    return (
        "intensity" in s
        or "counts" in s
        or "count/s" in s
        or "counts/s" in s
        or "countpersecond" in s
        or "countspersecond" in s
        or s in {"cps", "cnt", "signal", "y"}
    )


def _find_header_row(lines: list[str], technique: str) -> int:
    for idx, line in enumerate(lines[:120]):
        tokens = _split_line(line)
        if len(tokens) < 2:
            continue
        if technique == "XPS":
            has_x = any(_is_xps_energy(t) for t in tokens)
            has_y = any(_is_xps_intensity(t) for t in tokens)
        else:
            has_x = any(_is_wavenumber(t) for t in tokens)
            has_y = any(_intensity_kind(t) for t in tokens)
        if has_x and has_y:
            return idx
    return -1


def _numeric_frame_from_rows(lines: list[str]) -> pd.DataFrame:
    rows: list[list[float]] = []
    for line in lines:
        tokens = _split_line(line)
        nums: list[float] = []
        for token in tokens:
            try:
                nums.append(float(token))
            except ValueError:
                continue
        if len(nums) >= 2:
            rows.append(nums)
    if not rows:
        return pd.DataFrame()
    width = max(len(r) for r in rows)
    padded = [r + [np.nan] * (width - len(r)) for r in rows]
    return pd.DataFrame(padded)


def _first_numeric_columns(df: pd.DataFrame) -> tuple[int | None, int | None]:
    numeric = []
    for idx in range(df.shape[1]):
        col = pd.to_numeric(df.iloc[:, idx], errors="coerce")
        if col.notna().sum() > 0:
            numeric.append(idx)
        if len(numeric) >= 2:
            return numeric[0], numeric[1]
    return None, None


def read_spectrum_file(path, technique: str = "FTIR") -> Spectrum:
    """Read one FTIR or XPS file and identify its x and intensity columns."""
    path = Path(path)
    technique = str(technique).strip().upper()
    if technique not in {"FTIR", "XPS"}:
        raise ValueError(f"Unknown spectroscopy technique: {technique}")
    ext = path.suffix.lower()
    if ext not in {".csv", ".txt", ".dat"}:
        raise ValueError(f"{path.name}: unsupported extension '{ext}'. Use .csv, .txt, or .dat.")

    text = _read_text(path)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"{path.name}: file is empty.")

    warnings: list[str] = []
    header_idx = _find_header_row(lines, technique)
    if header_idx >= 0:
        delim = _sniff_delimiter(lines[header_idx:header_idx + 20])
        sep = r"\s+" if delim in {" ", r"\s+"} else delim
        df = pd.read_csv(
            io.StringIO(text),
            sep=sep,
            skiprows=header_idx,
            header=0,
            engine="python",
            on_bad_lines="skip",
        )
    else:
        df = _numeric_frame_from_rows(lines)
        assumed = "binding energy and intensity" if technique == "XPS" else "cm-1 and T%"
        warnings.append(f"{path.name}: no {technique} header found; using first two numeric columns as {assumed}.")

    if df.shape[1] < 2:
        raise ValueError(f"{path.name}: file must contain at least two numeric columns.")

    columns = list(df.columns)
    if technique == "XPS":
        x_idx = next((i for i, col in enumerate(columns) if _is_xps_energy(col)), None)
        y_idx = next((i for i, col in enumerate(columns) if _is_xps_intensity(col)), None)
    else:
        x_idx = next((i for i, col in enumerate(columns) if _is_wavenumber(col)), None)
        y_idx = next((i for i, col in enumerate(columns) if _intensity_kind(col)), None)
    if x_idx is None or y_idx is None or x_idx == y_idx:
        x_idx, y_idx = _first_numeric_columns(df)
        if x_idx is None or y_idx is None:
            raise ValueError(f"{path.name}: no usable numeric {technique} columns found.")
        expected = "binding-energy/intensity" if technique == "XPS" else "cm-1/T%/A%"
        warnings.append(f"{path.name}: could not identify {expected} columns; assuming first two numeric columns.")

    x_name = str(columns[x_idx])
    y_name = str(columns[y_idx])
    source_type = "XPS" if technique == "XPS" else (_intensity_kind(y_name) or "T")
    if technique == "FTIR" and _intensity_kind(y_name) is None:
        warnings.append(f"{path.name}: intensity type unknown; assuming T%.")

    x = pd.to_numeric(df.iloc[:, x_idx], errors="coerce")
    y = pd.to_numeric(df.iloc[:, y_idx], errors="coerce")
    mask = x.notna() & y.notna()
    x_arr = x[mask].to_numpy(dtype=float)
    y_arr = y[mask].to_numpy(dtype=float)
    if len(x_arr) == 0:
        raise ValueError(f"{path.name}: no numeric {technique} rows found.")

    order = np.argsort(x_arr)
    return Spectrum(
        path=path,
        x=x_arr[order],
        intensity=y_arr[order],
        technique=technique,
        source_type=source_type,
        x_column=x_name,
        y_column=y_name,
        warnings=warnings,
    )


def read_ftir_file(path) -> Spectrum:
    """Read one FTIR file (compatibility wrapper)."""
    return read_spectrum_file(path, technique="FTIR")


def read_xps_file(path) -> Spectrum:
    """Read one XPS file."""
    return read_spectrum_file(path, technique="XPS")


def convert_t_to_a(t_percent) -> np.ndarray:
    """Convert transmittance percent to absorbance."""
    t = np.asarray(t_percent, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(t > 0, -np.log10(t / 100.0), np.nan)


def normalize_spectrum(y, mode: str) -> np.ndarray:
    y_arr = np.asarray(y, dtype=float)
    mode = str(mode or "None")
    if mode == "None":
        return y_arr.copy()

    out = y_arr.copy()
    finite = np.isfinite(out)
    if not finite.any():
        return out

    vals = out[finite]
    if mode == "Maximum normalization":
        denom = float(np.nanmax(vals))
        if denom == 0:
            return out
        out[finite] = vals / denom
        return out

    if mode == "Min-max normalization":
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
        if hi == lo:
            return out
        out[finite] = (vals - lo) / (hi - lo)
        return out

    raise ValueError(f"Unknown normalization mode: {mode}")


def apply_vertical_offset(y, index: int, offset: float) -> np.ndarray:
    return np.asarray(y, dtype=float) + int(index) * float(offset)


def plot_ftir_spectra(
    curves: list[dict],
    display_mode: str,
    normalization: str,
    line_width: float,
    legend_position: str,
    hide_y_ticks: bool,
    technique: str = "FTIR",
    x_display_min: float | None = None,
    x_display_max: float | None = None,
) -> Figure:
    fig = Figure(figsize=(7.8, 5.6))
    ax = fig.add_subplot(111)
    colors = theme.SERIES_COLORS

    for idx, curve in enumerate(curves):
        ax.plot(
            curve["wavenumber"],
            curve["y_display"],
            linewidth=float(line_width),
            color=colors[idx % len(colors)],
            label=curve["curve"],
        )

    technique = str(technique).upper()
    ax.set_xlabel("Binding energy (eV)" if technique == "XPS" else "Wavenumber (cm-1)")
    if technique == "XPS":
        base_ylabel = "Intensity"
    else:
        base_ylabel = "Transmittance" if str(display_mode).startswith("T") else "Absorbance"
    if normalization != "None":
        base_ylabel += " (normalized)"
    ax.set_ylabel(base_ylabel)
    ax.set_title(f"{technique} spectra")
    if x_display_min is not None and x_display_max is not None:
        # FTIR and binding-energy XPS spectra conventionally run high-to-low.
        ax.set_xlim(float(x_display_max), float(x_display_min))
    else:
        ax.invert_xaxis()
    if hide_y_ticks:
        ax.set_yticklabels([])

    if legend_position != "Hidden":
        if legend_position == "Outside right":
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
        else:
            ax.legend(loc=legend_position.lower())
    fig.tight_layout()
    return fig
