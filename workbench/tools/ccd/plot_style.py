"""Local scientific styling and shared scales for the batch comparison pages."""
from __future__ import annotations

import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import FixedLocator, MaxNLocator, NullLocator


FONT_SIZE = 7
FONT_FAMILY = ("Arial" if any(font.name == "Arial" for font in font_manager.fontManager.ttflist)
               else "DejaVu Sans")
# First 32 entries, in the supplied categorical order, of batlowS from
# Fabio Crameri, Scientific colour maps 8.0.1 (2023), doi:10.5281/zenodo.1243862.
# Original license and attribution: assets/scientific_colour_maps/.
BATLOW_CATEGORICAL = (
    "#011959", "#faccfa", "#828231", "#226061", "#f19d6b", "#114360", "#fdb4b4", "#4d734d",
    "#c09036", "#677b3e", "#fcbfd6", "#a18a2b", "#fca890", "#356a59", "#0d315d", "#175262",
    "#db954b", "#f8a27e", "#2b655e", "#91862d", "#5a7745", "#1b5962", "#fdbac4", "#b18d2f",
    "#08255b", "#134b61", "#416f53", "#e7985a", "#0f3b5f", "#fbc6e8", "#fdaea2", "#747e38",
)


def source_colors(sources):
    """Reserve one categorical color per source, including unsuccessful files."""
    return {source: BATLOW_CATEGORICAL[n % len(BATLOW_CATEGORICAL)]
            for n, source in enumerate(dict.fromkeys(sources))}


def style_text(text):
    text.set_fontfamily(FONT_FAMILY)
    text.set_fontsize(FONT_SIZE)
    text.set_math_fontfamily("dejavusans")
    return text


def style_axis(ax):
    """Override only this axes; the application's other tools keep their theme."""
    ax.set_facecolor("white")
    ax.figure.set_facecolor("white")
    ax.grid(False, which="both", axis="both")
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(.5)
        spine.set_color("black")
    ax.tick_params(which="major", direction="out", length=3, width=.5,
                   labelsize=FONT_SIZE, top=False, right=False, color="black")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        axis.set_minor_locator(NullLocator())
        style_text(axis.label)
        style_text(axis.get_offset_text())
    for text in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        style_text(text)


def annotate(ax, text, *, x=.03, y=.97):
    return style_text(ax.text(x, y, text, transform=ax.transAxes,
                              ha="left", va="top", color="black"))


def tick_step(span):
    """A reproducible 1/2/2.5/5 interval, with roughly four intervals per frame."""
    span = float(span)
    if not np.isfinite(span) or span <= 0:
        return .1
    ticks = MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]).tick_values(0., span)
    return float(ticks[1] - ticks[0])


def operating_current_scale(curves):
    """Cover the complete finite measured current of the analyzed polarity."""
    maximum = max((float(np.nanmax(item.decomposition.current_magnitude))
                   for item in curves
                   if np.isfinite(item.decomposition.current_magnitude).any()), default=1.)
    step = tick_step(maximum)
    upper = max(step, np.ceil(maximum / step - 1e-12) * step)
    ticks = np.arange(int(round(upper / step)) + 1) * step
    return (0., upper), ticks


def shared_value_step(groups):
    """Use one voltage tick interval while allowing independent panel origins."""
    spans = []
    for values in groups:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            spans.append(float(np.ptp(values)))
    return tick_step(max(spans, default=.4))


def value_scale(ax, values, step):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    low, high = (float(np.min(values)), float(np.max(values))) if len(values) else (0., step)
    lower = np.floor(low / step) * step
    upper = np.ceil(high / step) * step
    if upper - lower < 2 * step - 1e-12:
        center = np.floor((low + high) / (2 * step)) * step
        lower, upper = center - step, center + 2 * step
    count = int(round((upper - lower) / step))
    ax.set_ylim(lower, upper)
    ax.yaxis.set_major_locator(FixedLocator(lower + np.arange(count + 1) * step))


def category_scale(ax, sources):
    ax.set_xlim(-.5, max(len(sources) - .5, .5))
    ax.set_xticks(range(len(sources)), sources)
    for text in ax.get_xticklabels():
        style_text(text)
