from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.signal import find_peaks, peak_prominences, peak_widths, savgol_filter

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
    background: np.ndarray | None = None
    background_column: str | None = None
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


def _is_xps_background(label: object) -> bool:
    s = _norm_label(label)
    return (
        "background" in s
        or "backgnd" in s
        or "baseline" in s
        or s in {"bg", "bgcounts", "bgcount"}
    )


def _avantage_xps_frame(lines: list[str]) -> pd.DataFrame | None:
    """Read Avantage clipboard text with its split label/unit header rows."""
    header_idx = next(
        (
            idx for idx, line in enumerate(lines[:120])
            if any(_is_xps_energy(token) for token in _split_line(line))
            and any(_is_xps_background(token) for token in _split_line(line))
            and not any(
                _is_xps_intensity(token) and not _is_xps_background(token)
                for token in _split_line(line)
            )
        ),
        None,
    )
    if header_idx is None:
        return None

    frame = _numeric_frame_from_rows(lines[header_idx + 1:])
    if frame.shape[1] < 3:
        return None
    extra_columns = [f"XPS column {index}" for index in range(4, frame.shape[1] + 1)]
    frame.columns = [
        "Binding Energy (eV)",
        "Counts / s",
        "Background Counts / s",
        *extra_columns,
    ]
    return frame


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
    avantage_frame = _avantage_xps_frame(lines) if technique == "XPS" else None
    header_idx = _find_header_row(lines, technique)
    if avantage_frame is not None:
        df = avantage_frame
    elif header_idx >= 0:
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
        background_idx = next(
            (i for i, col in enumerate(columns) if _is_xps_background(col)), None
        )
        y_idx = next(
            (
                i for i, col in enumerate(columns)
                if i != background_idx and _is_xps_intensity(col)
            ),
            None,
        )
    else:
        background_idx = None
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
    background_arr = None
    background_name = None
    if technique == "XPS" and background_idx is not None:
        background = pd.to_numeric(df.iloc[:, background_idx], errors="coerce")
        background_arr = background[mask].to_numpy(dtype=float)
        background_name = str(columns[background_idx])
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
        background=None if background_arr is None else background_arr[order],
        background_column=background_name,
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


def normalize_spectrum(y, mode: str, reference=None) -> np.ndarray:
    """Normalize *y*, optionally using another trace to determine the scale."""
    y_arr = np.asarray(y, dtype=float)
    reference_arr = y_arr if reference is None else np.asarray(reference, dtype=float)
    mode = str(mode or "None")
    if mode == "None":
        return y_arr.copy()

    out = y_arr.copy()
    finite = np.isfinite(out)
    reference_finite = np.isfinite(reference_arr)
    if not finite.any() or not reference_finite.any():
        return out

    vals = out[finite]
    reference_vals = reference_arr[reference_finite]
    if mode == "Maximum normalization":
        denom = float(np.nanmax(reference_vals))
        if denom == 0:
            return out
        out[finite] = vals / denom
        return out

    if mode == "Min-max normalization":
        lo = float(np.nanmin(reference_vals))
        hi = float(np.nanmax(reference_vals))
        if hi == lo:
            return out
        out[finite] = (vals - lo) / (hi - lo)
        return out

    raise ValueError(f"Unknown normalization mode: {mode}")


def mask_xps_background_endpoint_zeros(background) -> tuple[np.ndarray, int]:
    """Mask zero runs at XPS background boundaries when nonzero data follow."""
    out = np.asarray(background, dtype=float).copy()
    finite_indices = np.flatnonzero(np.isfinite(out))
    if not len(finite_indices) or not np.any(out[finite_indices] != 0.0):
        return out, 0

    masked = 0
    for index in finite_indices:
        if out[index] != 0.0:
            break
        out[index] = np.nan
        masked += 1
    for index in finite_indices[::-1]:
        if out[index] != 0.0:
            break
        out[index] = np.nan
        masked += 1
    return out, masked


def apply_vertical_offset(y, index: int, offset: float) -> np.ndarray:
    return np.asarray(y, dtype=float) + int(index) * float(offset)


def find_spectrum_peaks(
    x,
    y,
    *,
    minima: bool = False,
    prominence_percent: float = 5.0,
    noise_multiplier: float = 4.0,
    minimum_width: float = 3.0,
    smoothing_window: int = 0,
) -> dict[str, np.ndarray | float | int]:
    """Find spectroscopic peaks using adjustable, scale-independent filters.

    ``minima`` is used for transmittance spectra, where absorption bands point
    downward. Prominence is the greater of a percentage of the signal range and
    a robust noise multiple. ``smoothing_window=0`` selects a small automatic
    Savitzky-Golay window; a custom window is converted to a valid odd size.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_clean = x_arr[finite]
    y_clean = y_arr[finite]
    if len(x_clean) < 3:
        raise ValueError("Peak detection requires at least three finite data points.")
    if not 0.0 <= float(prominence_percent) <= 100.0:
        raise ValueError("Minimum prominence must be between 0 and 100 percent.")
    if float(noise_multiplier) < 0.0:
        raise ValueError("Noise multiplier cannot be negative.")
    if float(minimum_width) < 1.0:
        raise ValueError("Minimum peak width must be at least one point.")
    if int(smoothing_window) < 0:
        raise ValueError("Smoothing window cannot be negative.")

    signal = -y_clean if minima else y_clean
    n = len(signal)
    if int(smoothing_window) == 0:
        # Approximately 1% of the trace, bounded to preserve narrow peaks.
        window = min(31, max(5, n // 100 * 2 + 1))
    else:
        window = max(3, int(smoothing_window))
        if window % 2 == 0:
            window += 1
    if window >= n:
        window = n if n % 2 else n - 1
    if window >= 3:
        smooth = savgol_filter(signal, window_length=window, polyorder=min(3, window - 2))
    else:
        smooth = signal.copy()

    span = float(np.ptp(smooth))
    residual = signal - smooth
    residual_mad = float(np.median(np.abs(residual - np.median(residual))))
    noise = 1.4826 * residual_mad
    prominence_floor = max(
        float(prominence_percent) / 100.0 * span,
        float(noise_multiplier) * noise,
        np.finfo(float).eps,
    )
    indices, _ = find_peaks(
        smooth,
        prominence=prominence_floor,
        distance=max(1, n // 200),
    )
    # Smoothing must not turn a one-point spike into an apparently broad peak.
    # Refine each location to the nearby maximum in the original signal, then
    # measure and filter widths there.
    if len(indices):
        radius = max(1, window // 2)
        refined = []
        for index in indices:
            lo = max(1, int(index) - radius)
            hi = min(n - 1, int(index) + radius + 1)
            refined.append(lo + int(np.argmax(signal[lo:hi])))
        indices = np.asarray(sorted(set(refined)), dtype=int)
        prominences = peak_prominences(signal, indices)[0]
        widths = peak_widths(signal, indices, rel_height=0.5)[0]
        keep = widths >= float(minimum_width)
        indices = indices[keep]
        prominences = prominences[keep]
        widths = widths[keep]
    else:
        prominences = np.array([])
        widths = np.array([])
    return {
        "x": x_clean,
        "y": y_clean,
        "smoothed_signal": smooth,
        "indices": indices,
        "prominences": prominences,
        "widths": widths,
        "prominence_floor": prominence_floor,
        "noise_estimate": noise,
        "smoothing_window": window,
    }


def plot_detected_peaks(
    x,
    y,
    peak_indices,
    *,
    technique: str,
    source_type: str,
    title: str,
) -> Figure:
    """Plot one spectrum and mark the automatically detected peaks."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    indices = np.asarray(peak_indices, dtype=int)
    fig = Figure(figsize=(7.8, 5.6))
    ax = fig.add_subplot(111)
    ax.plot(x_arr, y_arr, color=theme.SERIES_COLORS[0], linewidth=1.0)
    if len(indices):
        ax.scatter(
            x_arr[indices], y_arr[indices], marker="v" if source_type == "T" else "^",
            color=theme.SERIES_COLORS[1], s=34, zorder=3, label="Detected peaks",
        )
        ax.legend(loc="best")
    technique = str(technique).upper()
    ax.set_xlabel("Binding energy (eV)" if technique == "XPS" else "Wavenumber (cm-1)")
    ax.set_ylabel("Intensity" if technique == "XPS" else (
        "Transmittance" if source_type == "T" else "Absorbance"
    ))
    ax.set_title(title)
    ax.invert_xaxis()
    fig.tight_layout()
    return fig


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
    reference_x: float | None = None,
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
        if curve.get("background_display") is not None:
            ax.plot(
                curve["wavenumber"],
                curve["background_display"],
                linewidth=max(0.8, float(line_width) * 0.9),
                linestyle="--",
                color=colors[idx % len(colors)],
                label="_nolegend_",
            )

    if reference_x is not None:
        ax.axvline(
            float(reference_x), color="gray", linestyle=":", linewidth=1.2,
            zorder=2.5,
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
        # Positive offsets place later input files higher on the stacked plot.
        # FTIR retains the visual top-to-bottom stack order. XPS follows the
        # draggable input-file order, which is more useful for sample series.
        handles, labels = ax.get_legend_handles_labels()
        if technique != "XPS":
            handles.reverse()
            labels.reverse()
        if legend_position == "Outside right":
            ax.legend(
                handles, labels, loc="center left",
                bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0,
            )
        else:
            ax.legend(handles, labels, loc=legend_position.lower())
    fig.tight_layout()
    return fig
