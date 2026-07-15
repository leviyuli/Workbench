"""Anthropic-Claude visual identity: color tokens, the Qt stylesheet loader,
and a matplotlib style so embedded plots blend with the workbench.

Keeping every color in one place means the five tools never diverge visually.
"""
from __future__ import annotations

import os

import matplotlib as mpl

# --- Color tokens -----------------------------------------------------------
BG = "#FAF9F5"        # warm off-white app background
SURFACE = "#FFFFFF"   # card surfaces
TEXT = "#2D2A26"      # dark slate
MUTED = "#6B6660"     # secondary text
ACCENT = "#D97757"    # Claude coral
ACCENT_HOVER = "#C2613F"
ACCENT_SOFT = "#F5E6DF"   # (reserved) soft coral tint
NEUTRAL_SOFT = "#ECE9E1"  # neutral warm-grey shade behind hovered/selected text
BORDER = "#E8E4DC"
SIDEBAR = "#F2EFE7"   # slightly warmer than BG
DANGER = "#B91C1C"
OK = "#0F766E"

RADIUS = 10
PAD = 18

# A balanced categorical palette for plotted series (print friendly and distinct).
SERIES_COLORS = [
    "#2563EB",  # blue
    "#DC2626",  # red
    "#16A34A",  # green
    "#9333EA",  # purple
    "#EA580C",  # orange
    "#0891B2",  # cyan
    "#C026D3",  # magenta
    "#65A30D",  # lime
    "#4B5563",  # graphite
    "#D97706",  # amber
]

# Overpotential decomposition palette (shared by the LSV tool).
OVERPOT_COLORS = {
    "E_rev": "#374151",
    "eta_kin": "#1D4ED8",
    "eta_ohm": "#0F766E",
    "eta_rcl": "#B45309",
    "eta_res": "#B91C1C",
}
OVERPOT_LABELS = {
    "E_rev": r"$E_\mathrm{rev}$",
    "eta_kin": r"$\eta_\mathrm{kin}$",
    "eta_ohm": r"$\eta_\mathrm{ohm}$",
    "eta_rcl": r"$\eta_\mathrm{RCL}$",
    "eta_res": r"$\eta_\mathrm{res}$",
}


def stylesheet() -> str:
    """Return the application Qt stylesheet, with color tokens substituted in."""
    qss_path = os.path.join(os.path.dirname(__file__), "styles", "claude.qss")
    with open(qss_path, "r", encoding="utf-8") as fh:
        css = fh.read()
    tokens = {
        "BG": BG, "SURFACE": SURFACE, "TEXT": TEXT, "MUTED": MUTED,
        "ACCENT": ACCENT, "ACCENT_HOVER": ACCENT_HOVER, "ACCENT_SOFT": ACCENT_SOFT,
        "NEUTRAL_SOFT": NEUTRAL_SOFT,
        "BORDER": BORDER, "SIDEBAR": SIDEBAR, "DANGER": DANGER,
        "RADIUS": str(RADIUS),
    }
    for key, val in tokens.items():
        css = css.replace("{{%s}}" % key, val)
    return css


def apply_matplotlib_style() -> None:
    """Minimalist scientific matplotlib style that matches the workbench."""
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "regular",
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": TEXT,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlepad": 12,
        "axes.labelpad": 6,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=SERIES_COLORS),
        "grid.color": MUTED,
        "grid.linewidth": 0.4,
        "grid.linestyle": "-",
        "grid.alpha": 0.22,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "legend.frameon": False,
        "figure.facecolor": SURFACE,
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })
