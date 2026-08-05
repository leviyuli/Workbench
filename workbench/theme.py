"""Anthropic-Claude visual identity: color tokens, the Qt stylesheet loader,
and a matplotlib style so embedded plots blend with the workbench.

Keeping every color in one place means the five tools never diverge visually.
"""
from __future__ import annotations

import os

import matplotlib as mpl

# --- Color tokens -----------------------------------------------------------
# The interface is deliberately neutral: near-black type, white surfaces and
# cool gray structure. Claude coral is reserved for small, meaningful accents.
BG = "#F5F5F3"
SURFACE = "#FFFFFF"
TEXT = "#181817"
MUTED = "#666663"
ACCENT = "#D97757"        # Claude coral
ACCENT_HOVER = "#C96345"
ACCENT_SOFT = "#F3E6E1"
NEUTRAL_SOFT = "#E8E8E5"
BORDER = "#D6D6D2"
SIDEBAR = "#EEEEEB"
DANGER = "#B91C1C"
OK = "#0F766E"

RADIUS = 6
PAD = 14

# A muted Anthropic-inspired categorical palette. Coral leads, followed by
# cool blue, sage and other restrained illustration-like hues.
SERIES_COLORS = [
    "#D97757",  # Claude coral
    "#6A9BCC",  # muted blue
    "#788C5D",  # sage
    "#C59A3D",  # ochre
    "#8E75B2",  # plum
    "#4F8F8B",  # teal
    "#B56B7A",  # dusty rose
    "#5F6670",  # slate
    "#9A7557",  # earth
    "#252523",  # charcoal
]

# Overpotential decomposition palette (shared by the LSV tool).
OVERPOT_COLORS = {
    "E_rev": "#252523",
    "eta_kin": "#6A9BCC",
    "eta_ohm": "#4F8F8B",
    "eta_rcl": "#C59A3D",
    "eta_res": "#D97757",
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
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlepad": 12,
        "axes.labelpad": 6,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=SERIES_COLORS),
        "grid.color": BORDER,
        "grid.linewidth": 0.55,
        "grid.linestyle": "-",
        "grid.alpha": 0.8,
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
