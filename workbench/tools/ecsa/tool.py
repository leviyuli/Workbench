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
    # Keep the existing registry ID so saved selections/export locations work.
    id = "ecsa"
    name = "CV process"
    subtitle = "Split raw CV data into E/i columns per cycle, or calculate non-faradaic ECSA from a scan-rate series."
    needs_files = True
    multiple_files = True
    accepts_directory = True
    accepted_file_types = (".txt",)
    run_button_label = "Process CV"

    def parameter_schema(self) -> list[ParameterSpec]:
        ecsa_params = [
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
        for spec in ecsa_params:
            spec.show_if = ("mode", "ECSA calculation")
            spec.group = "CV process"
        return [
            ParameterSpec(
                "mode", "Function", kind="choice",
                choices=["Split cycles", "ECSA calculation"], default="Split cycles",
                tooltip="Split cycles preserves raw E/i values and units. ECSA requires "
                        "at least two scan rates named like CV_50mV_1.txt.",
                group="CV process",
            ),
            *ecsa_params,
        ]

    def run(self, files: list[Path], params: dict, extra=None) -> ToolResult:
        mode = params.get("mode", "Split cycles")
        if mode == "Split cycles":
            return self.split_cycles(files, params, extra)
        if mode == "ECSA calculation":
            return self.compute_ecsa(files, params, extra)
        raise ValueError(f"Unknown CV function: {mode!r}")

    def split_cycles(self, files: list[Path], params: dict, extra=None) -> ToolResult:
        tables: dict[str, pd.DataFrame] = {}
        figures: dict[str, Figure] = {}
        export_names: dict[str, str] = {}
        figure_export_names: dict[str, str] = {}
        warnings: list[str] = []
        used_names: set[str] = set()
        cycle_count = 0
        partial_count = 0
        for filepath in files:
            filepath = Path(filepath)
            try:
                wide = cvp.split_cv_cycles(filepath)
            except ValueError as exc:
                warnings.append(f"{filepath.name}: {exc}")
                continue
            # Files from separate directories can have the same stem. Keep all
            # result tabs and suggested export filenames distinct on Windows.
            label = filepath.stem
            suffix = 2
            while label.casefold() in used_names:
                label = f"{filepath.stem}_{suffix}"
                suffix += 1
            used_names.add(label.casefold())
            table_name = f"{label} cycles"
            tables[table_name] = wide
            export_names[table_name] = f"{label}_cycles.csv"
            figure_name = f"{label} CV overlay"
            figures[figure_name] = self._plot_split_cv(wide, label)
            figure_export_names[figure_name] = f"{label}_cycles.png"
            cycle_count += wide.attrs["full_cycle_count"]
            partial_count += wide.attrs["partial_cycle_count"]
            warnings.extend(f"{filepath.name}: {warning}"
                            for warning in wide.attrs.get("warnings", []))

        if not tables:
            detail = "\n".join(warnings)
            raise ValueError("No CV cycles could be split. Select a file with E and i "
                             "columns and a cyclic potential sweep."
                             + (f"\n{detail}" if detail else ""))
        scalars = {"Files": len(tables), "Cycles": cycle_count}
        if partial_count:
            scalars["Partial scans"] = partial_count
        return ToolResult(
            figures=figures,
            tables=tables,
            scalars=scalars,
            text_blocks={"Cycle export": "Each adjacent E/i column pair is one cycle. "
                         "Raw values and units are preserved; time is omitted. "
                         "Shorter cycles have blank trailing cells in CSV exports. "
                         "Any partial scans are labeled separately."},
            warnings=warnings,
            payload={"mode": "Split cycles", "csv_filenames": export_names,
                     "png_filenames": figure_export_names},
        )

    def compute_ecsa(self, files: list[Path], params: dict, extra=None) -> ToolResult:
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

        scalars = {
            "C_dl (mF/cm²)": ecsa["Cdl"] * 1e3,
            "C_dl SD (mF/cm²)": ecsa["Cdl_se"] * 1e3,
            "ECSA (cm²)": ecsa["ECSA"] if ecsa["ECSA"] is not None else float("nan"),
            "ECSA SD (cm²)": ecsa["ECSA_se"] if ecsa["ECSA_se"] is not None else float("nan"),
            "R²": ecsa["r2"],
            "Intercept (A/cm²)": ecsa["intercept"],
            "Scan rates": len(files_by_rate),
        }

        # Tables (flatten the time-series MultiIndex for display/export).
        # The Summary sheet carries the scalar results (incl. ECSA SD) into the
        # exported workbook, which otherwise only holds the raw data tables.
        ts_flat = ts_wide.copy()
        if isinstance(ts_flat.columns, pd.MultiIndex):
            ts_flat.columns = [f"{a} · {b}" for a, b in ts_flat.columns]
        summary = pd.DataFrame(
            {"Quantity": list(scalars.keys()), "Value": list(scalars.values())}
        )
        tables = {
            "Summary": summary,
            "ECSA_result": ecsa["records"],
            "CV_long": long_df,
            "CV_time_series": ts_flat,
        }

        figures = {
            "ECSA fit": self._plot_ecsa(ecsa),
            "CV overlay": self._plot_cv(chosen, per_file_cycles, cycle_index, target_scale),
        }

        result = ToolResult(figures=figures, tables=tables, scalars=scalars, warnings=warnings)
        return result

    # ------------------------------------------------------------------ plots
    def _plot_split_cv(self, wide: pd.DataFrame, label: str) -> Figure:
        """Overlay the exported raw E/i pairs in acquisition order."""
        fig = Figure(figsize=(7.5, 5))
        ax = fig.add_subplot(111)
        segments = wide.attrs["segments"]
        n_cycles = wide.attrs["full_cycle_count"]
        colors = (theme.SERIES_COLORS if n_cycles <= len(theme.SERIES_COLORS)
                  else cm.viridis(np.linspace(0, 0.9, n_cycles)))
        cycle_index = 0
        for index, segment in enumerate(segments):
            pair = wide.iloc[:, 2 * index:2 * index + 2].dropna()
            if segment["complete"]:
                color = colors[cycle_index]
                cycle_index += 1
            else:
                color = theme.MUTED
            ax.plot(
                pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy(),
                label=segment["label"], color=color, linewidth=1.2,
                linestyle="-" if segment["complete"] else "--",
                marker="o" if len(pair) == 1 else None, markersize=4,
            )
        ax.set_xlabel(f"E ({wide.attrs['potential_unit'] or 'raw'})")
        ax.set_ylabel(f"i ({wide.attrs['current_unit'] or 'raw'})")
        ax.set_title(f"{label} · Split CV cycles")
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1),
                  ncol=1 if len(segments) <= 15 else 2, borderaxespad=0)
        fig.tight_layout()
        return fig

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
        if result.payload.get("mode") == "Split cycles":
            items = [Artifact(
                name=f"{table_name} (CSV)", kind="csv",
                default_filename=result.payload["csv_filenames"][table_name],
                payload=table,
            ) for table_name, table in result.tables.items()]
            items.append(Artifact(
                name="CV cycles workbook (XLSX)", kind="xlsx",
                default_filename="cv_cycles.xlsx", payload=dict(result.tables),
            ))
            for figure_name, fig in result.figures.items():
                items.append(Artifact(
                    name=f"{figure_name} (PNG)", kind="png",
                    default_filename=result.payload["png_filenames"][figure_name],
                    payload=fig,
                ))
            return items
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
