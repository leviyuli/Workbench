"""Per-cycle forward-scan CCD analysis within the LSV analysis tools."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from workbench import theme
from workbench.core.tool_base import ParameterSpec, Tool, ToolResult
from workbench.tools.ccd.logic import CcdResult, CcdSettings, analyze_curve
from workbench.tools.ccd.inputs import read_ccd_scans
from workbench.tools.ccd.legacy_parameters import ccd_parameters
from workbench.tools.ccd.exports import ccd_artifacts


CCD_COLUMN = "CCD (A/cm²)"
LIMIT_COLUMN = "Limiting current (A/cm²)"


@dataclass
class _Curve:
    code: str
    filename: str
    result: CcdResult
    color: object
    cycle: int = 1
    source: str = ""


def _statistics(values) -> tuple[int, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    count = len(values)
    return (count, float(np.mean(values)) if count else np.nan,
            float(np.std(values, ddof=1)) if count > 1 else np.nan)


class LegacyCcdTool(Tool):
    id = "lsv_ccd"
    name = "CCD & limiting current"
    subtitle = "Calculate CCD and limiting current from each CV cycle’s forward scan, or an LSV."
    shared_file_group = "lsv_analysis_inputs"
    run_button_label = "Calculate currents"

    def parameter_schema(self) -> list[ParameterSpec]:
        return ccd_parameters()

    def exportable_artifacts(self, result: ToolResult):
        return ccd_artifacts(result)

    def run(self, files: list[Path], params: dict, extra=None) -> ToolResult:
        if not files:
            raise ValueError("Add a CV or LSV file for CCD analysis.")
        values = {spec.key: spec.default for spec in self.parameter_schema()}
        values.update(params)
        branch = str(values["current_branch"])
        if branch not in {"Auto", "Positive", "Negative"}:
            raise ValueError("Current branch must be Auto, Positive or Negative.")
        area = float(values["electrode_area"])
        if not np.isfinite(area) or area < 0:
            raise ValueError("Electrode area must be finite and nonnegative.")
        reference_hfr = float(values["reference_hfr"])
        if not np.isfinite(reference_hfr) or reference_hfr < 0:
            raise ValueError("Reference HFR must be finite and nonnegative, in Ω·cm².")
        settings = CcdSettings(
            baseline_mode=values["baseline_mode"],
            baseline_lower=float(values["baseline_lower"]),
            baseline_upper=float(values["baseline_upper"]),
            reference_hfr=reference_hfr,
            threshold_v=float(values["ccd_threshold_mv"]) / 1000,
            bin_width_v=float(values["bin_width_mv"]) / 1000,
            smoothing_window=values["smoothing_window"],
            persistence_v=float(values["persistence_mv"]) / 1000,
            plateau_window_v=float(values["plateau_window_mv"]) / 1000,
            plateau_slope_fraction=float(values["plateau_slope_pct"]) / 100,
            plateau_spread_fraction=float(values["plateau_spread_pct"]) / 100,
        )
        curves: list[_Curve] = []
        rows, processed, raw = [], [], []
        warnings = []
        analyzed_files = set()
        has_cycles = False
        colors = (*theme.SERIES_COLORS, *colormaps["tab20"].colors)
        for number, path in enumerate(files, start=1):
            path = Path(path)
            source = f"S{number:02d}"
            data, error = read_ccd_scans(path)
            if error:
                rows.append(_empty_row(source, path.name, None, "", error))
                warnings.append(f"{source} ({path.name}): {error}")
                continue
            scans, info = data
            multiple = len(scans) > 1 or any(scan.cycle != 1 for scan in scans)
            has_cycles |= multiple
            warnings.extend(f"{source} ({path.name}): {note}"
                            for note in info.get("warnings", []))
            for scan in scans:
                code = f"{source}-C{scan.cycle:02d}" if multiple else source
                row = _empty_row(code, path.name, scan.cycle, scan.direction)
                try:
                    if scan.error:
                        raise ValueError(scan.error)
                    potential, current = scan.potential, scan.current.copy()
                    if not scan.is_density:
                        if area <= 0:
                            raise ValueError("Enter the electrode area to convert absolute current to A/cm².")
                        current = current / area
                    finite = np.isfinite(potential) & np.isfinite(current)
                    nonzero = current[finite & (current != 0)]
                    negative = (branch == "Negative" or (branch == "Auto"
                                and np.count_nonzero(nonzero < 0) > np.count_nonzero(nonzero > 0)))
                    keep = (current <= 0) if negative else (current >= 0)
                    # Split cycles before selecting current polarity. Return scans never enter a fit.
                    selected = keep | ~finite
                    result = analyze_curve(potential[selected], current[selected], settings)
                    curve = _Curve(code, path.name, result, colors[len(curves) % len(colors)],
                                   scan.cycle, source)
                    curves.append(curve)
                    analyzed_files.add(number)
                    row.update({CCD_COLUMN: result.ccd_current, LIMIT_COLUMN: result.limiting_current,
                                "CCD status": result.ccd_status,
                                "Limiting-current status": result.limiting_status,
                                "Notes": " ".join(result.warnings)})
                    warnings.extend(f"{code} ({path.name}): {note}" for note in result.warnings)
                    detail, acquired = _curve_tables(curve, reference_hfr)
                    processed.append(detail)
                    raw.append(acquired)
                except ValueError as exc:
                    row["Notes"] = str(exc)
                    warnings.append(f"{code} ({path.name}): {exc}")
                rows.append(row)
        if not curves:
            raise ValueError("No forward scans could be analyzed.\n" + "\n".join(warnings))
        currents = pd.DataFrame(rows)
        currents["Cycle"] = currents["Cycle"].astype("Int64")
        for column in [CCD_COLUMN, LIMIT_COLUMN]:
            currents[column] = pd.to_numeric(currents[column], errors="coerce")
        cycle_statistics = _cycle_statistics(currents) if has_cycles else None
        if has_cycles:
            _color_cycles(curves, currents)
        statistics = []
        groups = currents.groupby("Cycle", sort=True) if has_cycles else [(None, currents)]
        for cycle, group in groups:
            for metric, column in [("CCD", CCD_COLUMN), ("Limiting current", LIMIT_COLUMN)]:
                count, mean, sd = _statistics(group[column])
                record = {"Metric": metric, "Resolved files": count,
                          "Mean (A/cm²)": mean, "Sample SD (A/cm²)": sd}
                if cycle is not None:
                    record = {"Cycle": int(cycle), **record}
                statistics.append(record)
        statistics = pd.DataFrame(statistics)
        scalars = {"Files selected": len(files), "Files analyzed": len(analyzed_files),
                   "Forward scans analyzed": len(curves),
                   "CCD threshold (mV)": 1000 * settings.threshold_v,
                   "Reference HFR (Ω·cm²)": reference_hfr}
        if has_cycles:
            for metric, column in [("CCD", CCD_COLUMN), ("Limiting current", LIMIT_COLUMN)]:
                scalars[f"{metric} resolved scans"] = int(np.isfinite(currents[column]).sum())
            for row in cycle_statistics.to_dict("records"):
                label = f"{row['Source']} {row['Metric']}"
                scalars[f"{label} mean (A/cm²)"] = (
                    row["Mean (A/cm²)"] if row["Resolved cycles"] else "not resolved")
                scalars[f"{label} cycle SD (A/cm²)"] = (
                    row["Sample SD (A/cm²)"] if row["Resolved cycles"] > 1 else "undefined (n < 2)")
        else:
            for row in statistics.to_dict("records"):
                metric = row["Metric"]
                scalars[f"{metric} resolved"] = row["Resolved files"]
                scalars[f"{metric} mean (A/cm²)"] = (
                    row["Mean (A/cm²)"] if row["Resolved files"] else "not resolved")
                scalars[f"{metric} SD (A/cm²)"] = (
                    row["Sample SD (A/cm²)"] if row["Resolved files"] > 1 else "undefined (n < 2)")
        figures = {"LSV processing": _plot_processing(
                       curves, bool(values["show_raw"]), settings.threshold_v),
                   "Current statistics": (_plot_cycle_statistics(curves, currents, cycle_statistics)
                                          if has_cycles else _plot_statistics(curves, statistics)),
                   "Residual onset": _plot_residual(curves, settings.threshold_v)}
        benchmark, fits = _benchmark_tables(curves, reference_hfr, settings.baseline_mode)
        tables = {"Current results": currents, "Batch statistics": statistics,
                    "Processed curves": pd.concat(processed, ignore_index=True),
                    "Input curves": pd.concat(raw, ignore_index=True),
                    "Benchmark lines": benchmark, "Baseline fits": fits,
                    "Analysis settings": pd.DataFrame(
                        {**asdict(settings), "current_branch": branch,
                         "electrode_area_cm2": area,
                         "scan_selection": "Forward scan of each cycle"}.items(),
                        columns=["Parameter", "Value"])}
        if has_cycles:
            figures["Cycle trends"] = _plot_cycles(curves, currents)
            tables["Cycle statistics"] = cycle_statistics
        return ToolResult(
            figures=figures, tables=tables,
            scalars=scalars, warnings=warnings,
            display_tables=False,
            text_blocks={
                "Forward scans": "Each CV cycle is analyzed separately in the initial scan direction. "
                    "Return scans are excluded. Raw CV files and wide Cycle N voltage/current columns "
                    "are supported. LSV files contribute one scan. Curve IDs identify the file and cycle.",
                "CCD definition": f"CCD is the first sustained {1000 * settings.threshold_v:g} mV "
                    "increase above the fitted "
                    "middle-region voltage/current baseline after reference HFR correction. "
                    "The same constant HFR is subtracted from the scan and baseline, so it cancels "
                    "from the excess voltage. The empirical baseline includes any steady linear "
                    "losses in that interval; its slope is not an HFR measurement. This excess "
                    "differs from the residual overpotential in LSV decoupling. "
                    "Inspect the fitted interval for each scan.",
                "Benchmark export": "The analysis workbook's Benchmark lines sheet contains the "
                    "exact dashed-line potential/current coordinates shown in Forward scans. "
                    "Its iR-corrected benchmark potential subtracts HFR × benchmark current, "
                    "for plotting that same reference line after correction. Baseline fits gives "
                    "the actual fitted range and equation E = intercept + (dE/dj) × j for each scan. "
                    "Forward scans also includes the corrected baseline evaluated at the measured "
                    "smoothed current, which is the reference used to calculate excess voltage.",
                "Limiting current": "Median current of the highest sustained, nearly flat plateau "
                    "after CCD. An unfinished or unstable plateau is reported as unresolved.",
                "Current statistics": ("Dots show individual cycles within each file. Bars show their "
                    "mean ± sample SD (n−1); this SD describes cycle-to-cycle spread. Earlier cycles "
                    "are lighter and later cycles darker, with a separate color for each file. Each "
                    "metric omits its unresolved cycles; SD requires at least two resolved cycles. "
                    "Cycle trends retain the scan sequence. Batch statistics separately compare "
                    "matching cycle numbers across files. "
                    if has_cycles else "Points show individual files; bars show their mean and sample SD "
                    "(n−1). Each metric excludes unresolved results; SD requires at least two files. ")
                    + "Select replicates under matching conditions. Raw and processed copies of the "
                    "same measurement are alternative inputs, not independent replicates.",
            },
        )


def _empty_row(code, filename, cycle, direction, note=""):
    return {"Curve": code, "File": filename, "Cycle": cycle, "Scan direction": direction,
            CCD_COLUMN: np.nan, LIMIT_COLUMN: np.nan, "CCD status": "Analysis failed",
            "Limiting-current status": "Analysis failed", "Notes": note}


def _source_rows(currents: pd.DataFrame):
    scans = currents.loc[currents["Cycle"].notna()]
    return scans.groupby(scans["Curve"].str.split("-C").str[0], sort=False)


def _cycle_statistics(currents: pd.DataFrame) -> pd.DataFrame:
    """Describe cycle-to-cycle spread within each file, independently per metric."""
    rows = []
    for source, scans in _source_rows(currents):
        for metric, column in [("CCD", CCD_COLUMN), ("Limiting current", LIMIT_COLUMN)]:
            count, mean, sd = _statistics(scans[column])
            rows.append({"Source": source, "File": scans.iloc[0]["File"], "Metric": metric,
                         "Resolved cycles": count, "Mean (A/cm²)": mean,
                         "Sample SD (A/cm²)": sd})
    return pd.DataFrame(rows)


def _color_cycles(curves: list[_Curve], currents: pd.DataFrame) -> None:
    """Use file hue and cycle opacity consistently across all CV plots."""
    styles = {}
    for index, (source, scans) in enumerate(_source_rows(currents)):
        styles[source] = (theme.SERIES_COLORS[index % len(theme.SERIES_COLORS)],
                          int(scans["Cycle"].min()), int(scans["Cycle"].max()))
    for curve in curves:
        color, first, last = styles[curve.source]
        alpha = .35 + .65 * (curve.cycle - first) / (last - first) if last > first else 1.0
        curve.color = to_rgba(color, alpha)


def _curve_tables(curve: _Curve, reference_hfr: float):
    r = curve.result
    identity = {"Curve": curve.code, "File": curve.filename, "Cycle": curve.cycle}
    ccd_mask = np.zeros(len(r.binned_potential), dtype=bool)
    if r.ccd_index is not None:
        ccd_mask[r.ccd_index] = True
    plateau_mask = np.zeros_like(ccd_mask)
    if r.plateau_start is not None:
        plateau_mask[r.plateau_start:r.plateau_end + 1] = True
    processed = pd.DataFrame({
        **identity,
        "Measured potential (V)": r.potential_sign * r.binned_potential,
        "HFR drop (V)": reference_hfr * r.smoothed_current,
        "HFR-corrected potential (V)": r.potential_sign * r.binned_corrected_potential,
        "Corrected baseline potential (V)": r.potential_sign * r.corrected_baseline_potential,
        "Median current (A/cm²)": r.binned_current,
        "Smoothed current (A/cm²)": r.smoothed_current,
        "Middle-region line (A/cm²)": r.baseline_current,
        "Residual increase (mV)": 1000 * r.excess_voltage,
        "Baseline fit": ((r.binned_potential >= r.baseline_lower)
                         & (r.binned_potential <= r.baseline_upper)),
        "CCD point": ccd_mask, "Plateau": plateau_mask,
    })
    raw = pd.DataFrame({
        **identity,
        "Measured potential (V)": r.potential_sign * r.potential,
        "HFR-corrected potential (V)": r.potential_sign * r.corrected_potential,
        "Current magnitude (A/cm²)": r.current,
    })
    return processed, raw


def _axes(fig, rectangle):
    ax = fig.add_axes(rectangle)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    ax.tick_params(direction="out", labelsize=9)
    return ax


def _benchmark_mask(result: CcdResult) -> np.ndarray:
    """The dashed extension shown in the measured-potential processing plot."""
    return ((result.binned_potential >= result.baseline_lower)
            & (result.baseline_current >= 0)
            & (result.baseline_current <= 1.15 * np.max(result.smoothed_current)))


def _benchmark_tables(curves: list[_Curve], reference_hfr: float, mode: str):
    lines, fits = [], []
    for curve in curves:
        r = curve.result
        identity = {"Curve": curve.code, "File": curve.filename, "Cycle": curve.cycle}
        visible = _benchmark_mask(r)
        lines.append(pd.DataFrame({
            **identity,
            "Potential (V)": r.potential_sign * r.binned_potential[visible],
            "iR-corrected benchmark potential (V)": r.potential_sign * (
                r.binned_potential[visible] - reference_hfr * r.baseline_current[visible]),
            "Benchmark current density (A/cm²)": r.baseline_current[visible],
            "Baseline region": r.binned_potential[visible] <= r.baseline_upper,
        }))
        fit = (r.potential >= r.baseline_lower) & (r.potential <= r.baseline_upper)
        fits.append({
            **identity, "Fit selection": mode,
            "Fit start potential (V)": r.potential_sign * r.baseline_lower,
            "Fit end potential (V)": r.potential_sign * r.baseline_upper,
            "Fit points": int(np.count_nonzero(fit)),
            "Measured dE/dj (Ω·cm²)": r.potential_sign / r.baseline_slope,
            "Measured intercept (V)": r.potential_sign * r.corrected_baseline_intercept,
            "iR-corrected dE/dj (Ω·cm²)": r.potential_sign * r.corrected_baseline_slope,
            "iR-corrected intercept (V)": r.potential_sign * r.corrected_baseline_intercept,
            "Fit R²": r.baseline_r_squared,
        })
    return pd.concat(lines, ignore_index=True), pd.DataFrame(fits)


def _plot_processing(curves: list[_Curve], show_raw: bool, threshold_v: float = .020) -> Figure:
    many = len(curves) > 15
    fig = Figure(figsize=(10 if many else 9, 5.5), facecolor=theme.SURFACE)
    ax = _axes(fig, [.09, .13, .68 if many else .74, .82])
    color_handles = []
    for curve in curves:
        r, color = curve.result, curve.color
        x = r.potential_sign * r.binned_potential
        if show_raw:
            ax.plot(r.potential_sign * r.potential, r.current, color=color, lw=.5,
                    alpha=.22 * to_rgba(color)[3])
        ax.plot(x, r.smoothed_current, color=color, lw=1.35)
        baseline = _benchmark_mask(r)
        ax.plot(x[baseline], r.baseline_current[baseline], color=color, ls="--", lw=1)
        fit = (r.binned_potential >= r.baseline_lower) & (r.binned_potential <= r.baseline_upper)
        ax.plot(x[fit], r.smoothed_current[fit], color=color, lw=2.5,
                alpha=.45 * to_rgba(color)[3])
        if r.ccd_index is not None:
            ax.plot(x[r.ccd_index], r.ccd_current, "o", color=color, ms=5,
                    markeredgecolor=theme.TEXT, markeredgewidth=.4)
        if r.plateau_start is not None:
            span = [x[r.plateau_start], x[r.plateau_end]]
            ax.plot(span, [r.limiting_current] * 2, color=color, lw=2, ls=":")
            ax.plot(np.mean(span), r.limiting_current, "^", color=color, ms=5,
                    markeredgecolor=theme.TEXT, markeredgewidth=.4)
        color_handles.append(Line2D([], [], color=color, label=curve.code, lw=1.5))
    style_handles = [
        Line2D([], [], color=theme.MUTED, lw=1.3, label="Processed forward scan"),
        Line2D([], [], color=theme.MUTED, lw=2.5, alpha=.45, label="Fitted interval"),
        Line2D([], [], color=theme.MUTED, ls="--", lw=1, label="Middle-region fit"),
        Line2D([], [], color=theme.MUTED, marker="o", ls="none", ms=4,
               label=f"CCD: {1000 * threshold_v:g} mV"),
        Line2D([], [], color=theme.MUTED, marker="^", ls=":", ms=4, label="Limiting plateau"),
    ]
    if show_raw:
        style_handles.insert(0, Line2D([], [], color=theme.MUTED, lw=.5, alpha=.4,
                                     label="Raw forward scan"))
    style = ax.legend(handles=style_handles, loc="upper left", frameon=False,
                      fontsize=8, handlelength=1.8, labelspacing=.35)
    ax.add_artist(style)
    ax.legend(handles=color_handles, loc="upper left", bbox_to_anchor=(1.02, 1),
              borderaxespad=0, frameon=False, ncol=2 if many else 1,
              fontsize=8, handlelength=1.2, columnspacing=.8, labelspacing=.35)
    ax.set_xlabel("Measured potential / V", fontsize=10)
    ax.set_ylabel(r"Current magnitude / A cm$^{-2}$", fontsize=10)
    ax.set_ylim(bottom=0)
    return fig


def _plot_statistics(curves: list[_Curve], statistics: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(7.5, 5), facecolor=theme.SURFACE)
    ax = _axes(fig, [.13, .19, .80, .76])
    tick_labels = []
    for index, (metric, field) in enumerate([("CCD", "ccd_current"),
                                           ("Limiting current", "limiting_current")]):
        row = statistics.iloc[index]
        valid = [(curve, getattr(curve.result, field)) for curve in curves
                 if getattr(curve.result, field) is not None]
        count = len(valid)
        tick_labels.append(f"{metric}\nn = {count}")
        if not count:
            ax.text(index, .06, "Not resolved", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", color=theme.MUTED, fontsize=9)
            continue
        ax.bar(index, row["Mean (A/cm²)"], width=.52, color=theme.NEUTRAL_SOFT,
               edgecolor=theme.MUTED, linewidth=.7, zorder=1)
        if count > 1:
            ax.errorbar(index, row["Mean (A/cm²)"], yerr=row["Sample SD (A/cm²)"],
                        color=theme.TEXT, capsize=5, lw=1.4, fmt="none", zorder=4)
        offsets = np.linspace(-.16, .16, count) if count > 1 else [0]
        for offset, (curve, value) in zip(offsets, valid):
            ax.plot(index + offset, value, marker="o" if index == 0 else "^",
                    ls="none", color=curve.color, ms=5, markeredgecolor=theme.TEXT,
                    markeredgewidth=.4, zorder=5)
        if count == 1:
            ax.text(index, .02, "SD undefined", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", color=theme.MUTED, fontsize=8)
    ax.set_xticks([0, 1], tick_labels)
    ax.set_xlim(-.6, 1.6)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(min(0, ymin), max(.1, ymax * 1.10))
    ax.set_ylabel(r"Current density / A cm$^{-2}$", fontsize=10)
    fig.text(.53, .035, "Points: individual files    Bars: mean ± sample SD",
             ha="center", fontsize=9, color=theme.MUTED)
    return fig


def _plot_cycle_statistics(curves: list[_Curve], currents: pd.DataFrame,
                           statistics: pd.DataFrame) -> Figure:
    """Plot each cycle and its file's mean ± sample SD alongside the LSV-style bars."""
    fig = Figure(figsize=(7.5, 5), facecolor=theme.SURFACE, layout="constrained")
    ax = fig.subplots()
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    ax.tick_params(direction="out", labelsize=9)
    sources = statistics["Source"].drop_duplicates().tolist()
    count = len(sources)
    width = .52 / count
    centers = np.linspace(-.28, .28, count) if count > 1 else [0]
    ticks = []
    for index, (metric, field) in enumerate([("CCD", "ccd_current"),
                                           ("Limiting current", "limiting_current")]):
        metric_rows = statistics.loc[statistics["Metric"] == metric]
        total = int(metric_rows["Resolved cycles"].sum())
        ticks.append(f"{metric}\nn = {total} cycles")
        for source, offset in zip(sources, centers):
            row = metric_rows.loc[metric_rows["Source"] == source].iloc[0]
            position = index + offset
            valid = sorted([(curve, getattr(curve.result, field)) for curve in curves
                            if curve.source == source and getattr(curve.result, field) is not None],
                           key=lambda item: item[0].cycle)
            n = len(valid)
            if not n:
                ax.text(position, .05, "Not resolved", transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", color=theme.MUTED, fontsize=8, rotation=90 if count > 2 else 0)
                continue
            bar = ax.bar(position, row["Mean (A/cm²)"], width=width,
                         color=theme.NEUTRAL_SOFT, edgecolor=theme.MUTED, linewidth=.7,
                         zorder=1)[0]
            bar.set_gid(f"cycle-mean:{source}:{metric}")
            if n > 1:
                ax.errorbar(position, row["Mean (A/cm²)"], yerr=row["Sample SD (A/cm²)"],
                            color=theme.TEXT, capsize=5, lw=1.4, fmt="none", zorder=5,
                            label=f"cycle-sd:{source}:{metric}")
            else:
                ax.text(position, .02, "SD undefined", transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", color=theme.MUTED, fontsize=8, rotation=90 if count > 2 else 0)
            if count > 1:
                high = max(value for _, value in valid)
                if n > 1:
                    high = max(high, row["Mean (A/cm²)"] + row["Sample SD (A/cm²)"])
                ax.annotate(f"n={n}", (position, high), xytext=(0, 7),
                            textcoords="offset points", ha="center", fontsize=7, color=theme.MUTED)
            offsets = np.linspace(-.3 * width, .3 * width, n) if n > 1 else [0]
            for jitter, (curve, value) in zip(offsets, valid):
                point, = ax.plot(position + jitter, value, marker="o" if index == 0 else "^",
                                 ls="none", color=curve.color, ms=5, markeredgecolor=theme.TEXT,
                                 markeredgewidth=.35, zorder=6)
                point.set_gid(f"cycle-point:{curve.code}:{metric}")
    if count == 1:
        source, scans = next(iter(_source_rows(currents)))
        first, last = int(scans["Cycle"].min()), int(scans["Cycle"].max())
        handles = [Line2D([], [], marker="o", ls="none", ms=5,
                          color=to_rgba(theme.SERIES_COLORS[0], alpha), label=f"Cycle {cycle}")
                   for cycle, alpha in ([(first, .35), (last, 1.0)] if last > first else [(first, 1.0)])]
    else:
        handles = [Line2D([], [], marker="o", ls="none", ms=5,
                          color=theme.SERIES_COLORS[i % len(theme.SERIES_COLORS)], label=source)
                   for i, source in enumerate(sources)]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, 1.01),
              ncol=min(5, len(handles)), frameon=False, fontsize=8)
    ax.set_xticks([0, 1], ticks)
    ax.set_xlim(-.65, 1.65)
    ax.set_ylim(0, max(.1, ax.get_ylim()[1] * 1.12))
    ax.set_ylabel(r"Current density / A cm$^{-2}$", fontsize=10)
    ax.set_xlabel("Dots: cycles; bars: mean ± sample SD.\n"
                  "Light → dark: increasing cycle number.", fontsize=9)
    return fig


def _plot_cycles(curves: list[_Curve], currents: pd.DataFrame) -> Figure:
    """Keep cycle evolution visible without treating cycles as replicate files."""
    fig = Figure(figsize=(9, 5.8), facecolor=theme.SURFACE, layout="constrained")
    sources = [source for source, _ in _source_rows(currents)]
    cycles = sorted(currents["Cycle"].dropna().unique())
    axes = fig.subplots(2, 1, sharex=True)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
        ax.tick_params(direction="out", labelsize=9)
    handles = []
    curve_by_cycle = {(curve.source, curve.cycle): curve for curve in curves}
    for source_index, source in enumerate(sources):
        selected = currents["Curve"].str.split("-C").str[0] == source
        rows = currents.loc[selected].sort_values("Cycle")
        color = theme.SERIES_COLORS[source_index % len(theme.SERIES_COLORS)]
        for ax, column, marker in zip(axes, [CCD_COLUMN, LIMIT_COLUMN], ["o", "^"]):
            ax.plot(rows["Cycle"].to_numpy(dtype=float), rows[column].to_numpy(dtype=float),
                    color=color, lw=1.0, alpha=.45)
            for _, row in rows.iterrows():
                if np.isfinite(row[column]):
                    curve = curve_by_cycle[(source, int(row["Cycle"]))]
                    ax.plot(row["Cycle"], row[column], marker=marker, color=curve.color,
                            ls="none", ms=4)
        handles.append(Line2D([], [], color=color, lw=1.3, label=source))
    for ax, label, column in zip(axes, ["CCD", "Limiting current"], [CCD_COLUMN, LIMIT_COLUMN]):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(float(min(cycles)) - .4, float(max(cycles)) + .4)
        if len(cycles) <= 20:
            ax.set_xticks(cycles)
        ax.set_ylabel(label + r" / A cm$^{-2}$", fontsize=10)
        ax.set_ylim(0, max(.1, ax.get_ylim()[1] * 1.08))
        if not np.isfinite(currents[column]).any():
            ax.text(.5, .5, "Not resolved", transform=ax.transAxes,
                    ha="center", color=theme.MUTED, fontsize=10)
    axes[1].set_xlabel("Cycle (forward scan)\nGaps indicate unresolved results.", fontsize=9)
    axes[0].legend(handles=handles, loc="best", frameon=False, fontsize=8)
    return fig


def _plot_residual(curves: list[_Curve], threshold_v: float) -> Figure:
    """Show the HFR-corrected residual near the onset, in acquisition order."""
    many = len(curves) > 15
    fig = Figure(figsize=(10 if many else 9, 5.5), facecolor=theme.SURFACE)
    ax = _axes(fig, [.10, .13, .67 if many else .73, .82])
    handles = []
    for curve in curves:
        r = curve.result
        selected = r.binned_potential >= r.baseline_lower
        ax.plot(r.smoothed_current[selected], 1000 * r.excess_voltage[selected],
                color=curve.color, lw=1.35)
        if r.ccd_index is not None:
            ax.plot(r.ccd_current, 1000 * r.excess_voltage[r.ccd_index], "o",
                    color=curve.color, ms=5, markeredgecolor=theme.TEXT,
                    markeredgewidth=.4)
        handles.append(Line2D([], [], color=curve.color, lw=1.5, label=curve.code))
    ax.axhline(0, color=theme.MUTED, lw=.5)
    threshold = ax.axhline(1000 * threshold_v, color=theme.TEXT, ls="--", lw=1,
                           label=f"{1000 * threshold_v:g} mV threshold")
    styles = ax.legend(handles=[threshold, Line2D([], [], color=theme.MUTED,
                        marker="o", ls="none", ms=4, label="CCD")],
                       loc="upper left", frameon=False, fontsize=8)
    ax.add_artist(styles)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1),
              borderaxespad=0, frameon=False, ncol=2 if many else 1,
              fontsize=8, handlelength=1.2, columnspacing=.8, labelspacing=.35)
    ax.set_xlabel(r"Current magnitude / A cm$^{-2}$", fontsize=10)
    ax.set_ylabel("Excess above fitted baseline / mV", fontsize=10)
    # Focus the initial view on the threshold; exports retain the complete residual.
    ax.set_ylim(-1000 * threshold_v, 5000 * threshold_v)
    return fig
