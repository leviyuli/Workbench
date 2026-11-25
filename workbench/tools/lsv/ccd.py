"""Optional CCD diagnostics for the curve fitted on the Decoupling page."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from workbench import theme
from workbench.core.tool_base import ParameterSpec, ToolResult
from workbench.tools.ccd.residual import ResidualCcdSettings, analyze_residual_ccd


def ccd_parameters() -> list[ParameterSpec]:
    enabled = ("include_ccd", True)
    manual = [enabled, ("ccd_fit_mode", "Manual")]
    return [
        ParameterSpec(
            "include_ccd", "Include CCD analysis", kind="bool", default=False,
            group="CCD",
            tooltip="Calculate CCD from the fitted curve. With several files selected, "
                    "CCD describes their averaged curve.",
        ),
        ParameterSpec(
            "ccd_fit_mode", "Fit range", kind="choice", default="Auto",
            choices=["Auto", "Manual"], group="CCD", show_if=enabled,
            tooltip="Auto finds a stable residual baseline above the Tafel search range. "
                    "Manual uses the current-density limits below.",
        ),
        ParameterSpec(
            "ccd_j_lower", "Fit from", default=1.0, minimum=0.0, step=.1,
            decimals=4, suffix="A/cm²", group="CCD", show_if=manual,
            tooltip="Lower current-density magnitude for the residual baseline fit.",
        ),
        ParameterSpec(
            "ccd_j_upper", "Fit to", default=3.0, minimum=0.0, step=.1,
            decimals=4, suffix="A/cm²", group="CCD", show_if=manual,
            tooltip="Upper current-density magnitude for the residual baseline fit.",
        ),
        ParameterSpec(
            "ccd_threshold_mv", "Threshold", default=50.0, minimum=.1,
            maximum=1000.0, step=5.0, decimals=1, suffix="mV", group="CCD",
            show_if=enabled,
            tooltip="Excess above the fitted residual baseline required to detect CCD, "
                    "sustained over 100 mV of measured drive.",
        ),
    ]


def analyze_ccd(curve, params: dict, *, source_label: str, averaged: bool) -> ToolResult:
    """Reuse the fitted kinetic model and retain unresolved CCD as a normal result."""
    threshold_error = "CCD threshold must be a finite value greater than zero mV."
    value = params.get("ccd_threshold_mv", 50.0)
    try:
        threshold_mv = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(threshold_error) from None
    if isinstance(value, bool) or not np.isfinite(threshold_mv) or threshold_mv <= 0:
        raise ValueError(threshold_error)
    settings = ResidualCcdSettings(threshold_v=threshold_mv / 1000)
    mode = params.get("ccd_fit_mode", "Auto")
    lower, upper = params.get("ccd_j_lower", 1.0), params.get("ccd_j_upper", 3.0)
    if str(mode).strip().lower() == "manual":
        try:
            lower, upper = float(lower), float(upper)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("Residual-baseline current limits must be finite, nonnegative and increasing.") from None
    result = analyze_residual_ccd(
        curve, mode=mode, current_lower=lower, current_upper=upper, settings=settings,
    )
    label = f"Averaged {source_label}" if averaged else source_label
    ccd = result.ccd_current if result.ccd_current is not None else np.nan
    summary = {
        "CCD curve": label, "CCD (A/cm²)": ccd, "CCD status": result.ccd_status,
        "CCD potential (V)": result.ccd_potential,
        "CCD residual (V)": result.ccd_residual,
        "Residual baseline status": result.baseline_status,
        "Baseline lower (A/cm²)": result.baseline_lower_current,
        "Baseline upper (A/cm²)": result.baseline_upper_current,
        "Baseline slope (Ω·cm²)": result.baseline_slope,
        "Baseline intercept (V)": result.baseline_intercept,
        "Baseline R²": result.baseline_r_squared,
        "Baseline RMS (V)": result.baseline_rmse,
    }
    ccd_mask = np.zeros(len(result.current), dtype=bool)
    if result.ccd_index is not None:
        ccd_mask[result.ccd_index] = True
    metadata = {
        "CCD curve": label, "ccd_fit_mode": mode,
        "ccd_threshold_mv": threshold_mv, **asdict(settings),
        "Tafel method": "nested-window",
        "Residual definition": "total - kinetic - j*HFR - RCL; no clipping",
        "Source row basis": "Zero-based row of the fitted single or averaged curve",
        **curve.settings.correction.metadata(),
    }
    if str(mode).strip().lower() == "manual":
        metadata.update(ccd_j_lower=lower, ccd_j_upper=upper)
    warnings = list(result.warnings)
    if averaged:
        warnings.append("CCD describes the averaged curve. Use Batch decoupling & CCD "
                        "for an independent CCD from each file.")
    return ToolResult(
        figures={"CCD analysis": _plot_ccd(result, label, curve.settings.i_lower)},
        tables={
            "CCD results": pd.DataFrame([summary]),
            "CCD data": pd.DataFrame({
                "Drive (V)": result.drive, "Potential (V)": result.potential,
                "Current density magnitude (A/cm²)": result.current,
                "Median current density (A/cm²)": result.median_current,
                "Signed residual (V)": result.residual,
                "Residual baseline (V)": result.baseline,
                "CCD threshold line (V)": result.baseline + settings.threshold_v,
                "Excess above baseline (V)": result.excess,
                "Baseline fit": result.baseline_mask, "CCD point": ccd_mask,
                "Segment": result.segment_id,
                "Fitted curve row start": result.source_start,
                "Fitted curve row end": result.source_end,
            }),
            "CCD settings": pd.DataFrame(metadata.items(), columns=["Parameter", "Value"]),
        },
        scalars={
            "CCD curve": label, "CCD (A/cm²)": ccd, "CCD status": result.ccd_status,
            "CCD threshold (mV)": threshold_mv, "CCD persistence (V)": settings.persistence_v,
        },
        text_blocks={"CCD definition":
            f"CCD is the first sustained {threshold_mv:g} mV excess above a line fitted "
            "to the signed decoupling residual, confirmed over "
            f"{1000 * settings.persistence_v:g} mV of measured drive. "
            "It uses the kinetic fit from this Decoupling run. The baseline is fitted "
            "above the Tafel search range in Auto mode; inspect it or select a manual "
            "current range. CCD is reported as a positive current-density magnitude. "
            "The baseline can contain other losses and is not an HFR measurement. "
            "Single-curve analysis uses the imported, sorted curve; Batch decoupling & CCD "
            "retains acquisition segments and uses contiguous stable Tafel fitting."},
        warnings=warnings,
    )


def _plot_ccd(result, label: str, current_lower: float) -> Figure:
    fig = Figure(figsize=(7.5, 5.5))
    ax = fig.add_subplot(111)
    visible = result.current >= current_lower
    for number, segment in enumerate(np.unique(result.segment_id[visible])):
        selected = visible & (result.segment_id == segment)
        current = result.current[selected]
        ax.plot(current, result.residual[selected], color=theme.OVERPOT_COLORS["eta_res"],
                label="Signed residual" if number == 0 else "_nolegend_")
        if np.any(np.isfinite(result.baseline[selected])):
            ax.plot(current, result.baseline[selected], color=theme.TEXT, linestyle="--",
                    label="Residual baseline")
            ax.plot(current, result.baseline[selected] + result.settings.threshold_v,
                    color=theme.MUTED, linestyle=":",
                    label=f"Baseline + {1000 * result.settings.threshold_v:g} mV")
    selected = visible & result.baseline_mask
    if np.any(selected):
        ax.plot(result.current[selected], result.residual[selected], linestyle="none",
                marker="o", markersize=4, color=theme.OVERPOT_COLORS["eta_ohm"],
                label="Baseline fit points")
    if result.ccd_index is not None:
        ax.plot([result.ccd_current], [result.ccd_residual], marker="^", markersize=8,
                linestyle="none", color=theme.TEXT,
                label=f"CCD = {result.ccd_current:.3g} A/cm²", zorder=5)
    ax.set_xlabel(r"Current density magnitude  $|j|$  (A cm$^{-2}$)")
    ax.set_ylabel(r"Signed residual overpotential  $\eta_\mathrm{res}$  (V)")
    ax.set_title(f"{label}: {result.ccd_status}")
    if ax.lines:
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig
