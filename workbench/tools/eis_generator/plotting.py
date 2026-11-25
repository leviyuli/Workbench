"""Equal-scale Nyquist views with exact mixed-conduction resistance markers."""
from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.layout_engine import LayoutEngine
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import offset_copy

from workbench import theme


_HF = theme.SERIES_COLORS[1]
_WIDTH = theme.SERIES_COLORS[3]


class _NyquistLayout(LayoutEngine):
    """Reserve annotation space in points, independent of window dimensions."""
    _adjust_compatible = True
    _colorbar_gridspec = False

    def __init__(self, detail, labels, title, footer, bounds):
        super().__init__()
        self.detail, self.labels = detail, labels
        self.title, self.footer = title, footer
        self.bounds = bounds
        self._last_size = None

    def set(self, **kwargs):
        self._params.update(kwargs)

    def execute(self, fig):
        width, height = fig.get_size_inches() * 72.0
        fig.subplots_adjust(left=min(52.0 / width, 0.24),
                            right=1.0 - min(12.0 / width, 0.08),
                            top=1.0 - min(48.0 / height, 0.20),
                            bottom=min(132.0 / height, 0.55),
                            wspace=0.34)
        self.title.set_y(1.0 - 10.0 / height)
        for artist, y in zip(self.footer, (32.0, 18.0, 4.0)):
            artist.set_position((52.0 / width, y / height))
        size = (width, height)
        if size != self._last_size:
            # Fit the requested bounds to the available frame with exactly equal
            # units. Expand the data ranges rather than shrinking the axes box.
            # Only resize events reset the framing, preserving toolbar pan/zoom.
            for axes, xlim, ylim in self.bounds:
                frame = axes.get_position(original=True)
                ratio = frame.width * width / (frame.height * height)
                xspan = max(xlim[1] - xlim[0], (ylim[1] - ylim[0]) * ratio)
                center = (xlim[0] + xlim[1]) / 2.0
                axes.set_xlim(center - xspan / 2.0, center + xspan / 2.0)
                axes.set_ylim(ylim[0], ylim[0] + xspan / ratio)
            self._last_size = size
        available = self.detail.get_position(original=True).width * width
        for artist, midpoint, offset in self.labels:
            if available >= 230.0:
                artist.set_transform(offset_copy(self.detail.get_xaxis_transform(),
                                                fig=fig, y=-offset, units="points"))
                artist.set_position((midpoint, 0.0))
                artist.set_ha("center")
            else:
                artist.set_transform(offset_copy(self.detail.transAxes,
                                                fig=fig, y=-offset, units="points"))
                artist.set_position((0.0, 0.0))
                artist.set_ha("left")


def _draw_curves(ax, cases, selected):
    for index, case in enumerate(cases):
        z = case["generated"]["z_total"]
        emphasized = case["id"] == selected
        color = theme.SERIES_COLORS[index % len(theme.SERIES_COLORS)]
        line, = ax.plot(z.real, -z.imag, color=color,
                        linewidth=2.1 if emphasized else 1.3,
                        alpha=1.0 if emphasized else 0.55,
                        label=case["name"], zorder=3 if emphasized else 2)
        line.set_gid(f"spectrum:{case['id']}")
    ax.set_xlabel(r"$Z'$ / Ω cm$^2$")
    ax.set_ylabel(r"$-Z''$ / Ω cm$^2$")
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(labelsize=9)


def _detail_limits(cases):
    resistances = [case["generated"]["resistances"] for case in cases]
    origins = [case["generated"]["parameters"]["external_resistance"] for case in cases]
    low = min(origins)
    high = max(r["lf_intercept"] for r in resistances)
    scale = max((r["r_i"] + r["r_e"]) / 3 for r in resistances)
    span = max(high - low, scale, np.finfo(float).tiny)
    return (max(0.0, low - 0.12 * span), high + 0.15 * span), (0.0, 1.65 * scale)


def _guides_and_arrows(ax, annotated):
    generated = annotated["generated"]
    p, r = generated["parameters"], generated["resistances"]
    for key, value, color in (
        ("external", p["external_resistance"], theme.MUTED),
        ("hf_total", r["hfr_total"], _HF),
        ("lf_intercept", r["lf_intercept"], _WIDTH),
    ):
        guide = ax.axvline(value, color=color, linestyle="--", linewidth=1, alpha=0.85, zorder=1)
        guide.set_gid(f"asymptote:{key}")

    for offset, left, right, color, gid in (
        (38.0, p["external_resistance"], r["hfr_total"], _HF, "cl_hfr"),
        (65.0, r["hfr_total"], r["lf_intercept"], _WIDTH, "cl_width"),
    ):
        transform = offset_copy(ax.get_xaxis_transform(), fig=ax.figure,
                                y=-offset, units="points")
        arrow = ax.annotate("", xy=(right, 0), xytext=(left, 0),
                            xycoords=transform, textcoords=transform,
                            arrowprops={"arrowstyle": "<->", "color": color, "lw": 1.4},
                            annotation_clip=False)
        arrow.set_gid(f"resistance_arrow:{gid}")
    hfr_label = ax.text(0, 0, f"CL HFR = {r['hfr_cl']:.4g} Ω cm²",
                        color=_HF, fontsize=9, va="center")
    width_label = ax.text(0, 0, f"ΔZ′ = {r['width']:.4g} Ω cm²\n= R_CL,app / 3",
                          color=_WIDTH, fontsize=9, va="center")
    hfr_label.set_gid("resistance_label:cl_hfr")
    width_label.set_gid("resistance_label:cl_width")

    z = generated["z_total"]
    for index, marker, text in ((0, "o", "HF sample"), (-1, "s", "LF sample")):
        ax.scatter([z.real[index]], [-z.imag[index]], marker=marker, s=24,
                   facecolor="white", edgecolor=theme.TEXT, linewidth=0.9, zorder=5)
        if ax.get_ylim()[0] <= -z.imag[index] <= ax.get_ylim()[1]:
            ax.annotate(text, (z.real[index], -z.imag[index]), xytext=(7, 8),
                        textcoords="offset points", fontsize=8, color=theme.MUTED)

    visible = np.flatnonzero((-z.imag >= ax.get_ylim()[0]) & (-z.imag <= ax.get_ylim()[1]))
    if len(visible) >= 6:
        first = visible[len(visible) // 3]
        last = visible[min(len(visible) - 1, len(visible) // 3 + max(2, len(visible) // 10))]
        direction = ax.annotate("", xy=(z.real[last], -z.imag[last]),
                                xytext=(z.real[first], -z.imag[first]),
                                arrowprops={"arrowstyle": "->", "color": theme.TEXT, "lw": 1.2})
        direction.set_gid("decreasing_frequency")
    return [(hfr_label, (p["external_resistance"] + r["hfr_total"]) / 2, 48.0),
            (width_label, (r["hfr_total"] + r["lf_intercept"]) / 2, 79.0)]


def plot_cl_nyquist(cases, selected="current", view="Transport detail") -> Figure:
    annotated = next(case for case in cases if case["id"] == selected)
    r = annotated["generated"]["resistances"]
    full = view == "Full frequency range"
    fig = Figure(figsize=(11.2 if full else 8.2, 6.8))
    bounds = []
    if full:
        overall, detail = fig.subplots(1, 2)
        _draw_curves(overall, cases, selected)
        imag = np.concatenate([-case["generated"]["z_total"].imag for case in cases])
        overall.set_ylim(0.0, float(imag.max()) * 1.05)
        bounds.append((overall, overall.get_xlim(), overall.get_ylim()))
        overall.set_title("Full frequency range", fontsize=11)
        overall.legend(loc="upper right", fontsize=8)
        z = annotated["generated"]["z_total"]
        overall.scatter([z.real[0], z.real[-1]], [-z.imag[0], -z.imag[-1]],
                        s=24, facecolor="white", edgecolor=theme.TEXT, zorder=5)
    else:
        detail = fig.add_subplot(111)
    _draw_curves(detail, cases, selected)
    xlim, ylim = _detail_limits(cases)
    detail.set_xlim(*xlim)
    detail.set_ylim(*ylim)
    bounds.append((detail, xlim, ylim))
    detail.set_title("Transport detail", fontsize=11)
    labels = _guides_and_arrows(detail, annotated)
    detail.legend(loc="upper left", fontsize=8)
    title = fig.suptitle(f"CL Nyquist · annotations: {annotated['name']}", fontsize=12)
    metrics = fig.text(0, 0,
             f"R_CL,app = {r['rcl_app']:.4g} Ω cm²   ·   HFR_total = {r['hfr_total']:.4g} Ω cm²",
             fontsize=9, color=theme.TEXT)
    guide_note = fig.text(0, 0,
                         "Guides: external R (gray), HF (blue), LF (ochre). Circle → square: decreasing f.",
                         fontsize=8, color=theme.MUTED)
    view_note = fig.text(0, 0,
                        "Full spectrum and transport detail; all points retained." if full else
                        "Detail clips the capacitive tail; exports retain all points.",
                        fontsize=8, color=theme.MUTED)
    fig.set_layout_engine(_NyquistLayout(detail, labels, title, [metrics, guide_note, view_note], bounds))
    fig.get_layout_engine().execute(fig)
    return fig
