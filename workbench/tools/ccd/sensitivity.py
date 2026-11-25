"""Reproduce the first-three-forward-scan CCD sensitivity comparison.

Run ``python -m workbench.tools.ccd.sensitivity`` from the repository root.
Results retain unresolved onsets as missing values rather than zero currents.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import scipy

from workbench.tools.ccd.inputs import read_ccd_scans
from workbench.tools.ccd.logic import CcdSettings, analyze_curve
from workbench.tools.ccd.legacy import _Curve, _benchmark_tables, _curve_tables


CCD = "CCD (A/cm²)"
MEAN = "Mean CCD (A/cm²)"
SD = "Cycle SD (A/cm²)"
COLORS = ["#4477AA", "#CC6677", "#228833"]
THRESHOLDS = (20, 50, 100)
ROOT = Path(__file__).resolve().parents[2]
DAY_LABELS = {f"CV{i}.txt": f"Day {i}" for i in (1, 2, 3)}


def _first_three(path: Path, area: float):
    data, error = read_ccd_scans(path)
    if error:
        raise ValueError(error)
    scans, info = data
    if len(scans) < 3:
        raise ValueError(f"{path.name}: need three forward scans, found {len(scans)}.")
    selected = []
    for scan in scans[:3]:
        if scan.error:
            raise ValueError(f"{path.name} cycle {scan.cycle}: {scan.error}")
        current = scan.current.copy()
        if not scan.is_density:
            current /= area
        finite = np.isfinite(scan.potential) & np.isfinite(current)
        nonzero = current[finite & (current != 0)]
        negative = np.count_nonzero(nonzero < 0) > np.count_nonzero(nonzero > 0)
        keep = (current <= 0) if negative else (current >= 0)
        keep |= ~finite
        selected.append((scan, scan.potential[keep], current[keep]))
    return selected, info


def _summaries(rows):
    keys = ["Fit mode", "Threshold (mV)", "File"]
    return rows.groupby(keys, sort=False, dropna=False).agg(**{
        "Scans selected": ("Cycle", "size"),
        "Resolved scans": (CCD, "count"), MEAN: (CCD, "mean"), SD: (CCD, "std"),
    }).reset_index()


def _gaps(summary):
    rows = []
    for (mode, threshold), group in summary.groupby(["Fit mode", "Threshold (mV)"], sort=False):
        for (_, first), (_, second) in combinations(group.iterrows(), 2):
            complete = first["Resolved scans"] == 3 and second["Resolved scans"] == 3
            rows.append({"Fit mode": mode, "Threshold (mV)": threshold,
                         "First file": first["File"], "Second file": second["File"],
                         "First resolved scans": first["Resolved scans"],
                         "Second resolved scans": second["Resolved scans"],
                         "Both complete": complete,
                         "Second minus first (A/cm²)": second[MEAN] - first[MEAN] if complete else np.nan,
                         "Second / first": second[MEAN] / first[MEAN] if complete else np.nan})
    return pd.DataFrame(rows)


def _threshold_changes(rows):
    records = []
    for (mode, file, cycle), group in rows.groupby(["Fit mode", "File", "Cycle"], sort=False):
        base = group.loc[group["Threshold (mV)"] == 20].iloc[0]
        for _, row in group.loc[group["Threshold (mV)"] != 20].iterrows():
            records.append({"Fit mode": mode, "File": file, "Cycle": cycle,
                            "Threshold (mV)": row["Threshold (mV)"],
                            "20 mV CCD (A/cm²)": base[CCD], CCD: row[CCD],
                            "Change (A/cm²)": row[CCD] - base[CCD],
                            "Change (%)": 100 * (row[CCD] / base[CCD] - 1),
                            "20 mV status": base["CCD status"],
                            "CCD status": row["CCD status"]})
    return pd.DataFrame(records)


def calculate(files, baseline=(1.90, 2.45), area=1.0):
    """Return analysis tables and main results without writing artifacts."""
    if not np.isfinite(area) or area <= 0:
        raise ValueError("Electrode area must be finite and positive.")
    settings = CcdSettings(baseline_lower=baseline[0], baseline_upper=baseline[1])
    selected = [(Path(path), *_first_three(Path(path), area)) for path in files]
    main, local, forward, benchmark, fits, raw, results = [], [], [], [], [], [], {}
    scenarios = [(mode, lower, upper, True, "") for mode in ("Manual", "Auto")
                 for lower, upper in [baseline]]
    scenarios += [("Manual", lo, hi, False, anchor) for anchor in ("Manual", "Auto")
                  for lo, hi in product((-.05, 0, .05), repeat=2)]
    for mode, lower, upper, primary, anchor in scenarios:
        for threshold in THRESHOLDS:
            config = replace(settings, baseline_mode=mode, baseline_lower=lower,
                             baseline_upper=upper, threshold_v=threshold / 1000)
            for file_index, (path, scans, _info) in enumerate(selected):
                for scan, potential, current in scans:
                    scan_config = config
                    anchor_error = ""
                    if not primary:
                        center = results.get((anchor, 20, path.name, scan.cycle))
                        if center is None:
                            anchor_error = f"{anchor} anchor fit unavailable."
                            scan_config = replace(config, baseline_lower=baseline[0], baseline_upper=baseline[1])
                        else:
                            scan_config = replace(config, baseline_lower=center.baseline_lower + lower,
                                                  baseline_upper=center.baseline_upper + upper)
                    identity = {"Fit mode": mode, "Threshold (mV)": threshold,
                                "File": path.name, "Cycle": scan.cycle}
                    row = {**identity, "Anchor mode": anchor,
                           "Lower shift (mV)": 0 if primary else round(1000 * lower),
                           "Upper shift (mV)": 0 if primary else round(1000 * upper),
                           "Requested fit start (V)": scan_config.baseline_lower,
                           "Requested fit end (V)": scan_config.baseline_upper, CCD: np.nan,
                           "CCD potential (V)": np.nan, "CCD status": "Analysis failed",
                           "Fit start (V)": np.nan, "Fit end (V)": np.nan,
                           "Fit R²": np.nan, "Measured dE/dj (Ω·cm²)": np.nan,
                           "Fit intercept (V)": np.nan, "Notes": ""}
                    try:
                        if anchor_error:
                            raise ValueError(anchor_error)
                        result = analyze_curve(potential, current, scan_config)
                        row.update({CCD: result.ccd_current,
                                    "CCD potential (V)": (result.potential_sign * result.ccd_voltage
                                                           if result.ccd_voltage is not None else np.nan),
                                    "CCD status": result.ccd_status,
                                    "Fit start (V)": result.baseline_lower,
                                    "Fit end (V)": result.baseline_upper,
                                    "Fit R²": result.baseline_r_squared,
                                    "Measured dE/dj (Ω·cm²)": 1 / result.baseline_slope,
                                    "Fit intercept (V)": result.corrected_baseline_intercept,
                                    "Notes": " ".join(result.warnings)})
                        if primary:
                            results[(mode, threshold, path.name, scan.cycle)] = result
                            curve = _Curve(f"S{file_index + 1:02}-C{scan.cycle:02}", path.name,
                                           result, COLORS[file_index % len(COLORS)], scan.cycle)
                            detail, acquired = _curve_tables(curve, config.reference_hfr)
                            line, fit = _benchmark_tables([curve], config.reference_hfr, mode)
                            for table, target in [(detail, forward), (line, benchmark), (fit, fits)]:
                                target.append(table.assign(**{"Fit mode": mode,
                                                               "Threshold (mV)": threshold}))
                            if mode == "Manual" and threshold == 20:
                                raw.append(acquired)
                    except ValueError as exc:
                        row["Notes"] = str(exc)
                    (main if primary else local).append(row)
    rows = pd.DataFrame(main)
    rows[CCD] = pd.to_numeric(rows[CCD])
    windows = pd.DataFrame(local)
    windows[CCD] = pd.to_numeric(windows[CCD])
    summary = _summaries(rows)
    window_summary = windows.groupby(["Anchor mode", "Lower shift (mV)", "Upper shift (mV)",
                                      "Threshold (mV)", "File"], sort=False).agg(**{
        "Resolved scans": (CCD, "count"), MEAN: (CCD, "mean"), SD: (CCD, "std"),
    }).reset_index()
    tables = {"Per scan": rows, "Threshold summary": summary, "Mean gaps": _gaps(summary),
              "Paired threshold changes": _threshold_changes(rows),
              "Window sensitivity": windows, "Window summary": window_summary,
              "Forward scans": pd.concat(forward, ignore_index=True),
              "Benchmark lines": pd.concat(benchmark, ignore_index=True),
              "Baseline fits": pd.concat(fits, ignore_index=True),
              "Input forward scans": pd.concat(raw, ignore_index=True)}
    for table in tables.values():
        if "File" in table:
            table.insert(table.columns.get_loc("File") + 1, "Day", table["File"].map(DAY_LABELS).fillna(""))
    return tables, results, selected, settings


def _figure(frame_count=1, height=4.5):
    # Each frame is 4.5 cm wide; surrounding margins never shrink the frames.
    frame, left, right, bottom, top, gap = 4.5, 1.25, 2.1, 1.05, .6, 1.25
    width = left + frame * frame_count + gap * (frame_count - 1) + right
    full_height = bottom + height + top
    fig = plt.figure(figsize=(width / 2.54, full_height / 2.54))
    axes = [fig.add_axes([(left + index * (frame + gap)) / width, bottom / full_height,
                          frame / width, height / full_height]) for index in range(frame_count)]
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out", length=3, width=.5)
        ax.yaxis.set_major_locator(MaxNLocator(4))
    return fig, axes


def _save(fig, path, expected_frames):
    fig.canvas.draw()
    for ax in fig.axes:
        size = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        assert np.allclose([size.width * 2.54, size.height * 2.54], expected_frames)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(path.with_suffix("." + suffix), dpi=600, facecolor="white", bbox_inches=None)
    plt.close(fig)


def plots(tables, results, output):
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 7, "axes.labelsize": 7, "axes.linewidth": .5,
                         "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.bbox": None}):
        files = list(tables["Per scan"]["File"].unique())
        fig, axes = _figure(2)
        for ax, mode in zip(axes, ("Manual", "Auto")):
            for index, file in enumerate(files):
                group = tables["Threshold summary"].query("`Fit mode` == @mode and File == @file")
                # Conditional means with n < 3 are intentionally not connected.
                complete = group["Resolved scans"].eq(3)
                x = group["Threshold (mV)"].to_numpy() + (index - 1) * 2
                y = group[MEAN].where(complete).to_numpy()
                label = DAY_LABELS.get(file, Path(file).stem)
                ax.plot(x, y, color=COLORS[index], lw=1, label=label)
                valid = group["Resolved scans"].gt(0)
                ax.errorbar(x[valid], group.loc[valid, MEAN], yerr=group.loc[valid, SD],
                            fmt="s", ms=3, color=COLORS[index], lw=.5, capsize=2)
                for position, (_, summary) in zip(x, group.iterrows()):
                    points = tables["Per scan"].loc[
                        tables["Per scan"]["Fit mode"].eq(mode)
                        & tables["Per scan"]["File"].eq(file)
                        & tables["Per scan"]["Threshold (mV)"].eq(summary["Threshold (mV)"])]
                    for offset, value in zip((-.8, 0, .8), points[CCD]):
                        ax.plot(position + offset, value, "o", color=COLORS[index],
                                ms=3, alpha=.55, markeredgewidth=0)
                    if summary["Resolved scans"] < 3:
                        ax.text(position, .04 + index * .08,
                                f"{int(summary['Resolved scans'])}/3", color=COLORS[index],
                                transform=ax.get_xaxis_transform(), ha="center", va="bottom")
            ax.text(.04, .96, mode, transform=ax.transAxes, va="top")
            ax.set_xticks(THRESHOLDS)
            ax.set_xlim(12, 108)
            ax.set_ylim(2, 5.3)
            ax.set_xlabel("CCD threshold / mV")
        axes[0].set_ylabel(r"CCD / A cm$^{-2}$")
        axes[-1].legend(loc="upper left", bbox_to_anchor=(1.05, 1), frameon=False,
                        handlelength=1.5, borderaxespad=0)
        _save(fig, output / "threshold_sensitivity", (4.5, 4.5))

        fig, axes = _figure(3)
        handles = [Line2D([], [], color="black", lw=1, label="Forward scan"),
                   Line2D([], [], color="black", ls="--", lw=1, label="Manual fit"),
                   Line2D([], [], color="#AA3377", ls=":", lw=1, label="Auto fit")]
        for ax, file in zip(axes, files):
            manual = results[("Manual", 20, file, 1)]
            auto = results[("Auto", 20, file, 1)]
            ax.plot(manual.binned_potential, manual.smoothed_current, color="black", lw=1)
            for result, color, style in [(manual, "black", "--"), (auto, "#AA3377", ":")]:
                visible = ((result.binned_potential >= result.baseline_lower)
                           & (result.baseline_current >= 0)
                           & (result.baseline_current <= 1.15 * max(result.smoothed_current)))
                ax.plot(result.binned_potential[visible], result.baseline_current[visible],
                        color=color, ls=style, lw=1)
                ax.axvspan(result.baseline_lower, result.baseline_upper, color=color, alpha=.06)
                if result.ccd_index is not None:
                    ax.plot(result.ccd_voltage, result.ccd_current, "o", color=color, ms=3)
            ax.text(.04, .96, DAY_LABELS.get(file, Path(file).stem) + ", scan 1", transform=ax.transAxes, va="top")
            ax.set_xlim(1.5, 3.6)
            ax.set_ylim(0, 6)
            ax.set_xlabel("Measured potential / V")
        axes[0].set_ylabel(r"$j$ / A cm$^{-2}$")
        axes[-1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.05, 1),
                        frameon=False, handlelength=1.5, borderaxespad=0)
        # Extend right margin for the longer external legend without resizing frames.
        width, height = fig.get_size_inches()
        positions = [ax.get_position().bounds for ax in axes]
        fig.set_size_inches(width + .5, height)
        for ax, (x, y, w, h) in zip(axes, positions):
            ax.set_position([x * width / (width + .5), y, w * width / (width + .5), h])
        _save(fig, output / "baseline_comparison", (4.5, 4.5))


def write_report(tables, output, baseline):
    summary = tables["Threshold summary"]
    windows = tables["Window summary"]
    lines = ["CCD SENSITIVITY: FIRST THREE FORWARD SCANS", "",
             "All current densities use A/cm². User-confirmed mapping: CV1 = Day 1,",
             "CV2 = Day 2, CV3 = Day 3. The confirmed manual fit interval is 1.90–2.45 V.",
             f"Manual interval: {baseline[0]:.3f}–{baseline[1]:.3f} V; Auto selects each scan independently.",
             "Thresholds: 20, 50, 100 mV. All other settings are fixed at app defaults:",
             "10 mV bins, 15-bin cubic Savitzky–Golay smoothing, ten consecutive bins",
             "above threshold (approximately 100 mV confirmation), reference HFR 0.12 Ω·cm².",
             "Constant reference HFR cancels from the excess-voltage residual.", "",
             "MEAN ± SAMPLE SD ACROSS RESOLVED CYCLES", ""]
    for row in summary.to_dict("records"):
        if row["Resolved scans"] > 1:
            value = f"{row[MEAN]:.6f} ± {row[SD]:.6f}"
        elif row["Resolved scans"] == 1:
            value = f"{row[MEAN]:.6f}; SD undefined (n < 2)"
        else:
            value = "unresolved"
        lines.append(f"{row['Fit mode']:6} {row['Threshold (mV)']:3} mV {row['File']}: "
                     f"{value}; n={row['Resolved scans']}/3")
    lines += ["", "BETWEEN-DAY DIFFERENCES", ""]
    for row in tables["Mean gaps"].to_dict("records"):
        complete = row["Both complete"]
        value = f"{row['Second minus first (A/cm²)']:.6f}" if complete else "not comparable (incomplete detection)"
        lines.append(f"{row['Fit mode']:6} {row['Threshold (mV)']:3} mV "
                     f"{row['Second file']} minus {row['First file']}: {value}")
    if set(summary["File"]) == set(DAY_LABELS) and np.allclose(baseline, (1.9, 2.45)):
        lines += ["", "INTERPRETATION FOR THESE THREE FILES", "",
              "For Auto, Day 2 minus Day 1 changes from +0.658669 A/cm² at 20 mV",
              "to +0.041584 at 50 mV and −0.925844 at 100 mV: the ordering reverses.",
              "Day 3 remains above Day 2 at 20 and 50 mV; all Day 3 onsets are unresolved",
              "at 100 mV. Its residual does exceed 100 mV near the end, but not early enough",
              "for ten consecutive bins of confirmation. Thus 100 mV is not simply a",
              "uniform upward shift, and numerical Day 3 gaps cannot be inferred at 100 mV.",
              "For CV1, Auto 20 mV onsets at 2.185–2.245 V lie inside the confirmed",
              "Manual interval 1.90–2.45 V. That interval incorporates the developing",
              "deviation into the reference fit, and detection is allowed only after its",
              "upper endpoint. This explains why Manual can be much later or unresolved.",
              "High R² alone does not establish that a reference interval is appropriate.",
              "These interpretations apply to the supplied files and default processing."]
    lines += ["", "CAUTIONS FOR THE COMPARISON", "",
              "A mean with fewer than three resolved scans is conditional on detection; it",
              "cannot be directly interpreted as the complete three-cycle mean. Unresolved",
              "does not mean zero CCD. The threshold may be crossed without enough remaining",
              "scan span for sustained confirmation. No significance tests were performed.",
              "Cycle SD describes sequential scan variation within one file, not independent",
              "device-to-device replication. The uploaded figure is not reproduced by the",
              "confirmed manual range and the first three scans under the current algorithm.",
              "Its values are closer to Auto, but its exact prior processing is unverified.",
              "Auto searches for a 0.35 V (at least 12-bin) stable positive-slope region whose",
              "median current is 15–65% of the 95th-percentile current, requiring R² ≥ 0.995 and half-window slope",
              "agreement within 20%; among candidates it chooses the highest current slope.",
              "Manual can include curvature and bias the extrapolated reference even when",
              "its R² looks high. Fit choice changes the operational definition of onset.", "",
              "LOCAL MANUAL-RANGE SENSITIVITY", "",
              "Each endpoint was shifted independently by −50, 0, +50 mV (nine intervals)",
              "around both the default Manual range and each scan's frozen Auto range.",
              "The shifted Auto intervals are refitted as Manual without rerunning selection.",
              "The following ranges are only among intervals resolving all three scans:"]
    for (anchor, threshold, file), group in windows.groupby(["Anchor mode", "Threshold (mV)", "File"], sort=False):
        complete = group.loc[group["Resolved scans"].eq(3), MEAN]
        value = f"{complete.min():.6f} to {complete.max():.6f}" if len(complete) else "none"
        lines.append(f"{anchor:6} anchor {threshold:3} mV {file}: complete-window mean range {value}; "
                     f"{len(complete)}/9 intervals complete")
    lines += ["", "ORIGIN RECREATION", "",
              "Open ccd_sensitivity.xlsx. Filter every sheet by Fit mode, Threshold (mV),",
              "File and Cycle before plotting. Forward scans: X = Measured potential (V),",
              "Y = Smoothed current (A/cm²). Benchmark lines: X = Potential (V),",
              "Y = Benchmark current density (A/cm²); use a dashed line. Baseline region",
              "marks the fitted interval; extension rows lie outside that interval.",
              "For an iR-corrected illustration use HFR-corrected potential (V) for the",
              "scan and iR-corrected benchmark potential (V) for the fitted line. The UI",
              "currently plots measured potentials. Corrected baseline potential is evaluated",
              "at the actual smoothed current and is used for residuals; it is not the same",
              "X grid as the benchmark line evaluated at benchmark current.",
              "CCD point flags identify the scatter markers. Input forward scans preserves",
              "the first three polarity-selected acquired scans once per file, with return scans excluded.",
              "Baseline fits includes line equations E = intercept + (dE/dj) × j and R².", "",
              "FIGURE CAPTIONS", "",
              "threshold_sensitivity: Squares and error bars show mean ± sample SD of",
              "resolved sequential cycles; dots show each scan. Labels n/3 identify incomplete",
              "groups. Lines connect only complete groups. Left Manual, right Auto.",
              "baseline_comparison: First forward scan of each file, measured-potential",
              "axis, with default Manual and Auto fitted lines, shaded fit windows and",
              "20 mV CCD markers where resolved. Numeric sheets contain all three scans.",
              "Figures use 4.5 × 4.5 cm frames, Arial 7 pt and 600 dpi PNG plus editable SVG/PDF.", "",
              "REPRODUCIBILITY", "",
              "Run: python -m workbench.tools.ccd.sensitivity",
              "Change manual interval: add --baseline 1.90 2.45; choose --output for destination.",
              "provenance.json records input SHA-256, scan counts, settings and package versions."]
    (output / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs=3, type=Path,
                        default=[ROOT / "Test_files" / "CCD_traning" / f"CV{i}.txt" for i in (1, 2, 3)])
    parser.add_argument("--baseline", type=float, nargs=2, default=(1.9, 2.45), metavar=("LOW", "HIGH"))
    parser.add_argument("--area", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "analysis_outputs" / "ccd_sensitivity")
    args = parser.parse_args(argv)
    tables, results, selected, settings = calculate(args.inputs, tuple(args.baseline), args.area)
    args.output.mkdir(parents=True, exist_ok=True)
    first_three_dir = args.output / "first3_forward"
    first_three_dir.mkdir(exist_ok=True)
    for path, scans, _info in selected:
        # Preserve the full acquired forward branches for convenient UI re-import.
        original = pd.concat([pd.DataFrame({
            "Cycle": scan.cycle, "E(V)": scan.potential,
            "j(A/cm2)": scan.current if scan.is_density else scan.current / args.area,
        }) for scan, _, _ in scans], ignore_index=True)
        original.to_csv(first_three_dir / f"{path.stem}_first3_forward.csv", index=False)
    for name, table in tables.items():
        table.to_csv(args.output / (name.lower().replace(" ", "_") + ".csv"), index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(args.output / "ccd_sensitivity.xlsx", engine="openpyxl") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name, index=False)
            worksheet = writer.sheets[name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column in worksheet.iter_cols(max_row=1):
                worksheet.column_dimensions[column[0].column_letter].width = min(45, max(14, len(str(column[0].value)) + 2))
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "selection": "First three acquired forward scans per file; dominant current polarity",
                "thresholds_mv": THRESHOLDS, "settings": asdict(settings), "area_cm2": args.area,
                "manual_endpoint_shifts_v": [-.05, 0, .05],
                "user_confirmed_day_mapping": DAY_LABELS,
                "software": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "scipy": scipy.__version__,
                             "matplotlib": matplotlib.__version__},
                "inputs": [{"path": str(path.resolve()),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "forward_scans_in_file": info["scan_count"],
                            "selected_cycles": [scan.cycle for scan, _, _ in scans]}
                           for path, scans, info in selected]}
    (args.output / "provenance.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(tables, args.output, tuple(args.baseline))
    plots(tables, results, args.output)
    print(tables["Threshold summary"].to_string(index=False).replace("²", "2"))
    print(f"\nSaved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
