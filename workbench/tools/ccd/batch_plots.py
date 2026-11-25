"""Batch figures and Origin-ready coordinates built from the same series."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib.colors import to_hex, to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator

from workbench.tools.ccd.plot_style import (
    FONT_SIZE, annotate, category_scale, operating_current_scale, shared_value_step,
    source_colors, style_axis, style_text, value_scale,
)


@dataclass
class BatchCurve:
    scan: object
    decomposition: object
    ccd: object
    color: object


@dataclass
class _Series:
    curve: str
    source: str
    file: str
    cycle: object
    scan: str
    role: str
    x: np.ndarray
    y: np.ndarray
    color: object
    marker: str = ""
    linestyle: str = "-"
    width: float = 1.


def _curve_series(item, role, x, y, **style):
    scan = item.scan
    return _Series(scan.code, scan.source, scan.file, scan.cycle, scan.branch,
                   role, np.asarray(x), np.asarray(y), item.color, **style)


def _export_series(name, series, x_label, y_label, index):
    columns = {}
    for number, line in enumerate(series, 1):
        key = f"{line.curve} · {line.role}"
        x_column, y_column = f"{key} · {x_label}", f"{key} · {y_label}"
        columns[x_column], columns[y_column] = pd.Series(line.x), pd.Series(line.y)
        index.append({"Plot sheet": name, "Series": key, "Curve": line.curve,
                      "Source": line.source, "File": line.file, "Cycle": line.cycle,
                      "Scan": line.scan, "X column": x_column, "Y column": y_column,
                      "Color": to_hex(line.color), "Alpha": to_rgba(line.color)[3],
                      "Marker": line.marker, "Line style": line.linestyle,
                      "Artist": f"{name}:{number}"})
    return pd.DataFrame(columns)


def _draw(ax, name, series, x_label, y_label, *, first_number=1):
    for number, line in enumerate(series, first_number):
        artist, = ax.plot(line.x, line.y, color=line.color, marker=line.marker,
                          linestyle=line.linestyle, lw=line.width, ms=3,
                          markeredgewidth=.5,
                          markeredgecolor="black" if line.marker else line.color,
                          label=line.curve if line.role in {"original", "kinetic", "residual", "RCL"}
                          else "_nolegend_")
        artist.set_gid(f"{name}:{number}")
    # Compact scientific labels remain readable in the app's 2x2 panels;
    # workbook columns retain the full quantity names and explicit units.
    display_labels = {
        "Current density magnitude (A/cm²)": r"$|j|$ / A cm$^{-2}$",
        "Signed current density (A/cm²)": r"$j$ / A cm$^{-2}$",
        "Kinetic overpotential (V)": r"$\eta_{\mathrm{kin}}$ / V",
        "Signed residual overpotential (V)": r"$\eta_{\mathrm{res}}$ / V",
        "RCL overpotential (V)": r"$\eta_{\mathrm{RCL}}$ / V",
        "CCD (A/cm²)": r"$|j|$ / A cm$^{-2}$",
        "Limiting current (A/cm²)": r"$|j|$ / A cm$^{-2}$",
        "Kinetic (V)": r"$\eta_{\mathrm{kin}}$ / V",
        "Residual (V)": r"$\eta_{\mathrm{res}}$ / V",
        "RCL (V)": r"$\eta_{\mathrm{RCL}}$ / V",
        "Potential (V)": r"$E$ / V",
        "Cell voltage (V)": r"$E_{\mathrm{cell}}$ / V",
    }
    if x_label.startswith("Potential (V") and x_label.endswith(")"):
        reference = x_label.removeprefix("Potential (V").removesuffix(")").strip()
        display_labels[x_label] = "$E$ / V" + (f" {reference}" if reference else "")
    ax.set_xlabel(display_labels.get(x_label, x_label))
    ax.set_ylabel(display_labels.get(y_label, y_label))
    style_axis(ax)


def _point_series(rows, field, sources, colors, role):
    series, statistics = [], []
    for position, source in enumerate(sources):
        group = rows.loc[rows.Source.eq(source)]
        values = pd.to_numeric(group[field], errors="coerce")
        valid = group.loc[np.isfinite(values)]
        offsets = np.linspace(-.16, .16, len(group)) if len(group) > 1 else [0.0]
        for offset, (_, row) in zip(offsets, group.iterrows()):
            value = row[field]
            if pd.notna(value) and np.isfinite(value):
                color = colors.get(row.Curve, colors[source])
                series.append(_Series(row.Curve, source, row.File, row.Cycle, row.Scan,
                                      role, np.array([position + offset]), np.array([float(value)]),
                                      color, marker="o", linestyle="none"))
        finite = pd.to_numeric(valid[field]).to_numpy(dtype=float)
        mean = float(np.mean(finite)) if len(finite) else np.nan
        sd = float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan
        statistics.append({"Source": source, "File": group.iloc[0].File, "Metric": field,
                           "Resolved scans": len(finite), "Mean": mean, "Sample SD": sd})
        if len(finite):
            first = group.iloc[0]
            series.append(_Series(source, source, first.File, None, "", f"{role} mean",
                                  np.array([position - .2, position + .2]), np.array([mean, mean]),
                                  "black", width=1.))
    return series, statistics


def _panel_grid(rows, columns, *, frame=(4.5, 4.), gap=(1.5, 1.2)):
    """Allocate physical frame sizes in cm, separately from labels and margins."""
    left, right, bottom, top = 1.5, .3, 1.3, .3
    width = left + columns * frame[0] + (columns - 1) * gap[0] + right
    height = bottom + rows * frame[1] + (rows - 1) * gap[1] + top
    fig = Figure(figsize=(width / 2.54, height / 2.54), facecolor="white")

    def add(row, column, *, span=1, **kwargs):
        x = left + column * (frame[0] + gap[0])
        y = bottom + (rows - row - 1) * (frame[1] + gap[1])
        panel_width = span * frame[0] + (span - 1) * gap[0]
        return fig.add_axes([x / width, y / height, panel_width / width, frame[1] / height], **kwargs)

    return fig, add


def _legend(ax, **kwargs):
    legend = ax.legend(frameon=False, fontsize=FONT_SIZE, handlelength=1.5,
                       handletextpad=.4, labelspacing=.25, borderpad=.2, **kwargs)
    for text in legend.get_texts():
        style_text(text)
    return legend


def _ccd_facets(curves, lines, dimensions, current_limits, current_ticks):
    columns = 1 if len(curves) == 1 else 2 if len(curves) <= 4 else 3
    rows = max(1, int(np.ceil(len(curves) / columns)))
    fig, add = _panel_grid(rows, columns, frame=(4., 3.2), gap=(.5, .6))
    fig._workbench_plot_page = "residual_ccd_facets"
    fig._workbench_facet_columns = columns
    # Dense pages scroll vertically in the UI rather than compressing their text.
    if rows > 2:
        fig._workbench_minimum_canvas_size = (0, rows * 170 + 50)
    finite_x = [line.x[np.isfinite(line.x)] for line in lines
                if np.isfinite(line.x).any()]
    x_values = np.concatenate(finite_x) if finite_x else np.array([0., 1.])
    lower, upper = float(x_values.min()), float(x_values.max())
    pad = max((upper - lower) * .04, .005)
    limits = (lower - pad, upper + pad)
    first, number = None, 1
    for position, item in enumerate(curves):
        row, column = divmod(position, columns)
        ax = add(row, column, **({"sharex": first, "sharey": first} if first is not None else {}))
        if first is None:
            first = ax
        selected = [line for line in lines if line.curve == item.scan.code]
        _draw(ax, "Plot CCD", selected, *dimensions, first_number=number)
        number += len(selected)
        ax._workbench_curve_id = item.scan.code
        ax.set_xlim(limits)
        ax.set_ylim(current_limits)
        ax.yaxis.set_major_locator(FixedLocator(current_ticks))
        bottom = position + columns >= len(curves)
        ax.tick_params(labelleft=column == 0, labelbottom=bottom)
        if column:
            ax.set_ylabel("")
        if not bottom:
            ax.set_xlabel("")
        annotate(ax, f"{item.scan.code} · {item.scan.branch}")
        if position == 0:
            handles = [
                Line2D([], [], color="black", lw=.5, label="Residual"),
                Line2D([], [], color="black", lw=1., label="Fit interval"),
                Line2D([], [], color="black", lw=.5, ls="--", label="Baseline"),
                Line2D([], [], color="black", ls="none", marker="o", ms=3,
                       markeredgewidth=.5, label="CCD"),
                Line2D([], [], color="black", lw=.5, ls=":", marker="^", ms=3,
                       markeredgewidth=.5, label="Plateau"),
            ]
            _legend(ax, handles=handles, loc="upper left", bbox_to_anchor=(.01, .88))
        if item.ccd.ccd_index is None:
            annotate(ax, "CCD unresolved", x=.03, y=.12)
    return fig


def build_batch_views(curves, rows, breakdown, include_rcl=False):
    """Return four figures, exact series sheets and descriptive statistics."""
    sources = list(dict.fromkeys(rows.Source))
    colors = source_colors(sources)
    colors.update({item.scan.code: item.color for item in curves})
    series = {name: [] for name in ("Plot original", "Plot kinetic", "Plot residual", "Plot CCD")}
    if include_rcl:
        series["Plot RCL"] = []
    for item in curves:
        d, r = item.decomposition, item.ccd
        series["Plot original"].append(_curve_series(item, "original", d.potential, d.measured_current))
        # The log-kinetic model is not constrained below its fitting search range.
        # Retain complete calculations in the detail sheets, while avoiding the
        # near-zero-current extrapolation dominating the plotted component axes.
        show = d.current_magnitude >= d.settings.i_lower
        visible_current = np.where(show, d.current_magnitude, np.nan)
        series["Plot kinetic"].append(_curve_series(
            item, "kinetic", visible_current, np.where(show, d.eta_kin, np.nan)))
        series["Plot residual"].append(_curve_series(
            item, "residual", visible_current, np.where(show, d.residual, np.nan)))
        if include_rcl:
            series["Plot RCL"].append(_curve_series(
                item, "RCL", visible_current, np.where(show, d.eta_rcl, np.nan)))
        # NaN separators prevent lines joining independent acquisition segments.
        visible = r.current >= d.settings.i_lower
        x, y = np.where(visible, r.residual, np.nan), np.where(visible, r.current, np.nan)
        breaks = np.flatnonzero(np.diff(r.segment_id) != 0) + 1
        x, y = np.insert(x, breaks, np.nan), np.insert(y, breaks, np.nan)
        series["Plot CCD"].append(_curve_series(item, "scan", x, y, width=.5))
        if np.any(r.baseline_mask):
            keep = np.isfinite(r.baseline)
            series["Plot CCD"].append(_curve_series(
                item, "baseline", r.baseline[keep], r.current[keep], linestyle="--", width=.5))
            series["Plot CCD"].append(_curve_series(
                item, "fitted interval", r.residual[r.baseline_mask], r.current[r.baseline_mask]))
        if r.ccd_index is not None:
            index = r.ccd_index
            series["Plot CCD"].append(_curve_series(
                item, "CCD", [r.residual[index]], [r.current[index]], marker="o", linestyle="none"))
        if r.plateau_start is not None:
            start, end = r.plateau_start, r.plateau_end
            span = np.array([r.residual[start], r.residual[end]])
            series["Plot CCD"].append(_curve_series(
                item, "limiting plateau", span, np.full(2, r.limiting_current), linestyle=":"))
            series["Plot CCD"].append(_curve_series(
                item, "limiting current", [float(np.mean(span))], [r.limiting_current],
                marker="^", linestyle="none"))

    dimensions = {
        "Plot original": ("Potential (V)", "Signed current density (A/cm²)"),
        "Plot kinetic": ("Current density magnitude (A/cm²)", "Kinetic overpotential (V)"),
        "Plot residual": ("Current density magnitude (A/cm²)", "Signed residual overpotential (V)"),
        "Plot RCL": ("Current density magnitude (A/cm²)", "RCL overpotential (V)"),
        "Plot CCD": ("Signed residual overpotential (V)", "Current density magnitude (A/cm²)"),
    }
    if curves:
        dimensions["Plot original"] = (
            curves[0].decomposition.settings.correction.axis_label(),
            "Signed current density (A/cm²)",
        )
    index, sheets, figures = [], {}, {}
    fig, add = _panel_grid(2, 2)
    axes = ([add(0, 0), add(0, 1), add(1, 0), add(1, 1)] if include_rcl else
            [add(0, 0, span=2), add(1, 0), add(1, 1)])
    names = (["Plot original", "Plot kinetic", "Plot RCL", "Plot residual"] if include_rcl else
             ["Plot original", "Plot kinetic", "Plot residual"])
    for ax, name in zip(axes, names):
        _draw(ax, name, series[name], *dimensions[name])
        ax._workbench_plot_role = name
        if name == "Plot original":
            annotate(ax, "Original polarization")
        sheets[name] = _export_series(name, series[name], *dimensions[name], index)
    if curves:
        _legend(axes[0], ncol=min(4, len(curves)), loc="upper left", bbox_to_anchor=(.01, .88))
    figures["Decoupling"] = fig

    current_limits, current_ticks = operating_current_scale(curves)
    figures["CCD & limiting current"] = _ccd_facets(
        curves, series["Plot CCD"], dimensions["Plot CCD"], current_limits, current_ticks)
    sheets["Plot CCD"] = _export_series("Plot CCD", series["Plot CCD"], *dimensions["Plot CCD"], index)

    current_stats, breakdown_stats = [], []
    fig, add = _panel_grid(1, 2)
    for position, (field, label) in enumerate(zip(["CCD (A/cm²)", "Limiting current (A/cm²)"],
                                                ["CCD", "Limiting current"])):
        ax = add(0, position)
        name = f"Plot {label} points"
        lines, stats = _point_series(rows, field, sources, colors, label)
        current_stats.extend(stats)
        _draw(ax, name, lines, "Input file", field)
        category_scale(ax, sources)
        ax.set_ylim(current_limits)
        ax.yaxis.set_major_locator(FixedLocator(current_ticks))
        annotate(ax, label)
        if not lines:
            annotate(ax, "No resolved values", x=.15, y=.5)
        sheets[name] = _export_series(name, lines, "File position", field, index)
    figures["Current comparison"] = fig

    fields = ["Kinetic (V)", "Residual (V)"]
    if include_rcl:
        fields.insert(1, "RCL (V)")
    fig, add = _panel_grid(1, len(fields))
    step = shared_value_step([pd.to_numeric(breakdown[field], errors="coerce") for field in fields])
    for position, field in enumerate(fields):
        ax = add(0, position)
        label = field.removesuffix(" (V)")
        name = f"Plot {label} breakdown"
        lines, stats = _point_series(breakdown, field, sources, colors, label)
        breakdown_stats.extend(stats)
        _draw(ax, name, lines, "Input file", field)
        category_scale(ax, sources)
        value_scale(ax, pd.to_numeric(breakdown[field], errors="coerce"), step)
        if not lines:
            annotate(ax, "No resolved points\nat target", x=.15, y=.5)
        sheets[name] = _export_series(name, lines, "File position", field, index)
    figures["Breakdown comparison"] = fig
    # The exact individual and mean-bar coordinates remain available internally.
    sheets["Plot series"] = pd.DataFrame(index)
    return figures, sheets, pd.DataFrame(current_stats), pd.DataFrame(breakdown_stats)
