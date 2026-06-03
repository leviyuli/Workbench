from __future__ import annotations

import io
import os
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.figure import Figure

from workbench import theme
from workbench.core.tool_base import Artifact, ParameterSpec, Tool, ToolResult
from workbench.tools.ecsa import logic_cv as cvp
from workbench.tools.ecsa.logic_ecsa import compute_cdl_ecsa


class EcsaTool(Tool):
    id = "ecsa"
    name = "ECSA (non-faradaic)"
    subtitle = "Double-layer capacitance & ECSA from a CV scan-rate series. Filenames like CV_50mV_1.txt."
    needs_files = True
    multiple_files = True
    accepts_directory = True
    accepted_file_types = (".txt",)
    run_button_label = "Compute ECSA"

    def parameter_schema(self) -> list[ParameterSpec]:
        return [
            ParameterSpec("re_type", "Reference electrode", kind="choice",
                          choices=["Hg/HgO", "Ag/AgCl"], default="Hg/HgO", group="Potential"),
            ParameterSpec("target_scale", "Target scale", kind="choice",
                          choices=["RHE", "SHE"], default="RHE", group="Potential"),
            ParameterSpec("ph", "pH", kind="float", default=14.0, minimum=0.0,
                          maximum=14.0, step=0.1, decimals=2, group="Potential"),
            ParameterSpec("Cs", "Specific capacitance", kind="float", default=40.0,
                          minimum=0.0, step=1.0, decimals=3, suffix="µF/cm²",
                          tooltip="ECSA = C_dl / Cs.", group="ECSA"),
            ParameterSpec("n_nearest", "Points near midpoint", kind="int", default=5,
                          minimum=1, maximum=999, step=1, group="ECSA"),
            ParameterSpec("cycle_index", "Cycle index", kind="int", default=-1,
                          minimum=-50, maximum=50, step=1,
                          tooltip="-1 = last cycle of each file.", group="ECSA"),
            ParameterSpec("replicate_mode", "Replicates", kind="choice",
                          choices=["last", "all"], default="last", group="ECSA"),
        ]

    def run(self, files: list[Path], params: dict, extra=None) -> ToolResult:
        re_type = params.get("re_type", "Hg/HgO")
        target_scale = params.get("target_scale", "RHE")
        ph = float(params.get("ph", 14.0))
        Cs = float(params.get("Cs", 40.0)) or None
        n_nearest = int(params.get("n_nearest", 5))
        cycle_index = int(params.get("cycle_index", -1))
        replicate_mode = params.get("replicate_mode", "last")

        warnings: list[str] = []
        files_by_rate: dict[float, list] = {}
        file_order: list[str] = []
        per_file_cycles: dict[str, dict] = {}

        # Parse + read each file.
        for fp in files:
            fp = str(fp)
            rate, rep = cvp.parse_scan_rate(fp)
            if rate is None:
                warnings.append(f"Filename doesn't match CV pattern, skipped: {os.path.basename(fp)}")
                continue
            cycles = cvp.process_one_file(fp, re_type, target_scale, ph)
            if not cycles:
                warnings.append(f"No cycles detected, skipped: {os.path.basename(fp)}")
                continue
            per_file_cycles[fp] = cycles
            files_by_rate.setdefault(rate, []).append((rep, fp))
            file_order.append(fp)

        if len(files_by_rate) < 2:
            raise ValueError(
                "Need at least two distinct scan rates with valid cycles. "
                "Check filenames (e.g. CV_50mV_1.txt) and that each file has CV cycles."
            )

        # Build datasets + ECSA fit, capturing any stdout notices as warnings.
        buf = io.StringIO()
        with redirect_stdout(buf):
            chosen = cvp.select_replicates(files_by_rate, replicate_mode)
            long_df = cvp.build_ecsa_dataset(chosen, per_file_cycles, cycle_index)
            ecsa = compute_cdl_ecsa(long_df, Cs_uF_per_cm2=Cs, n_nearest=n_nearest)
            ts_wide = cvp.build_time_series_wide(file_order, per_file_cycles, cycle_index, target_scale)
        for line in buf.getvalue().splitlines():
            if line.strip():
                warnings.append(line.strip())
        warnings.extend(ecsa.get("warnings", []))

        # Tables (flatten the time-series MultiIndex for display/export).
        ts_flat = ts_wide.copy()
        if isinstance(ts_flat.columns, pd.MultiIndex):
            ts_flat.columns = [f"{a} · {b}" for a, b in ts_flat.columns]
        tables = {
            "ECSA_result": ecsa["records"],
            "CV_long": long_df,
            "CV_time_series": ts_flat,
        }

        scalars = {
            "C_dl (mF/cm²)": ecsa["Cdl"] * 1e3,
            "ECSA (cm²)": ecsa["ECSA"] if ecsa["ECSA"] is not None else float("nan"),
            "R²": ecsa["r2"],
            "Intercept (A/cm²)": ecsa["intercept"],
            "Scan rates": len(files_by_rate),
        }

        figures = {
            "ECSA fit": self._plot_ecsa(ecsa),
            "CV overlay": self._plot_cv(chosen, per_file_cycles, cycle_index, target_scale),
        }

        result = ToolResult(figures=figures, tables=tables, scalars=scalars, warnings=warnings)
        return result

    # ------------------------------------------------------------------ plots
    def _plot_ecsa(self, ecsa: dict) -> Figure:
        rec = ecsa["records"]
        vs = rec["scan_rate_mV_per_s"].values * 1e-3
        jdl = rec["jdl_A_per_cm2"].values
        fig = Figure(figsize=(6.5, 5))
        ax = fig.add_subplot(111)
        ax.plot(vs, jdl * 1e3, "o", color=theme.ACCENT, markerfacecolor="white",
                markeredgecolor=theme.ACCENT, markeredgewidth=1.6, label="j_dl")
        xline = np.linspace(vs.min(), vs.max(), 100)
        yline = ecsa["Cdl"] * xline + ecsa["intercept"]
        ax.plot(xline, yline * 1e3, "-", color=theme.TEXT, linewidth=1.4,
                label=f"fit (R²={ecsa['r2']:.4f})")
        ax.set_xlabel(r"Scan rate $\nu$  (V s$^{-1}$)")
        ax.set_ylabel(r"$j_\mathrm{dl}$  (mA cm$^{-2}$)")
        ecsa_txt = f"{ecsa['ECSA']:.3g} cm²" if ecsa["ECSA"] is not None else "n/a"
        ax.set_title(f"C_dl = {ecsa['Cdl'] * 1e3:.4g} mF/cm²   ·   ECSA = {ecsa_txt}")
        ax.legend()
        fig.tight_layout()
        return fig

    def _plot_cv(self, chosen, per_file_cycles, cycle_index, target_scale) -> Figure:
        fig = Figure(figsize=(6.5, 5))
        ax = fig.add_subplot(111)
        rates = sorted(chosen.keys())
        colors = cm.viridis(np.linspace(0, 0.9, max(len(rates), 1)))
        for color, rate in zip(colors, rates):
            for fp in chosen[rate]:
                cycles = per_file_cycles.get(fp)
                if not cycles:
                    continue
                idx = cvp._resolve_cycle_index(cycles, cycle_index, fp)
                cyc = cycles[idx]
                ax.plot(cyc["E"], cyc["i"] * 1e3, linewidth=1.0,
                        color=color, label=f"{rate:g} mV/s")
        ax.set_xlabel(f"E (V vs {target_scale})")
        ax.set_ylabel(r"i  (mA cm$^{-2}$)")
        ax.set_title("Selected CV cycle per scan rate")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        return fig

    def exportable_artifacts(self, result: ToolResult) -> list[Artifact]:
        items = [Artifact(
            name="ECSA workbook (XLSX)", kind="xlsx",
            default_filename="ecsa_results.xlsx",
            payload={k: v for k, v in result.tables.items()},
        )]
        for fname, fig in result.figures.items():
            items.append(Artifact(
                name=f"{fname} (PNG)", kind="png",
                default_filename=f"ecsa_{fname.lower().replace(' ', '_')}.png", payload=fig,
            ))
        return items
