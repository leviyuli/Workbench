"""DRT (Distribution of Relaxation Times) analysis — model-free EIS diagnostics.

Uses pyimpspec's ``calculate_drt`` to recover the distribution of relaxation
times from impedance spectra. DRT is complementary to equivalent-circuit fitting:
it reveals the number and position of time-constant features without assuming a
circuit topology.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from pathlib import Path

from workbench import theme
from workbench.core.data_io import read_eis_data
from workbench.core.tool_base import Artifact, ParameterSpec, Tool, ToolResult

_DRT_METHODS = ["tr-nnls", "tr-rbf", "bht", "lm", "mrq-fit"]
_DATA = theme.OVERPOT_COLORS["eta_kin"]    # indigo for measured points
_DRT_COLOR = "#c44e52"                      # DRT distribution curve
_PEAK_COLOR = "#e5ae38"                     # identified peaks


class DrtTool(Tool):
    id = "drt"
    name = "DRT analysis"
    subtitle = ("Distribution of Relaxation Times — model-free identification of "
                "time-constant features in impedance spectra. Use before EIS fitting "
                "to determine how many RC processes are present.")
    needs_files = True
    files_required = True
    multiple_files = True
    accepts_directory = False
    accepted_file_types = (".csv", ".txt", ".xlsx", ".xls")
    run_button_label = "Analyze"

    def parameter_schema(self) -> list[ParameterSpec]:
        return [
            ParameterSpec("method", "DRT method", kind="choice",
                          choices=_DRT_METHODS, default="tr-nnls",
                          tooltip="TR-NNLS (Tikhonov regularization with "
                                  "non-negative least squares) is the recommended "
                                  "default. TR-RBF uses radial basis functions. "
                                  "BHT, LM, and MRQ-fit are alternative algorithms.",
                          group="DRT options"),
            ParameterSpec("lambda", "Regularization λ", kind="float",
                          default=0.0, minimum=0.0, maximum=1.0, step=0.001,
                          decimals=6,
                          tooltip="Tikhonov regularization strength for TR methods. "
                                  "0 = auto-select via L-curve (recommended). "
                                  "Higher values produce smoother distributions.",
                          group="DRT options",
                          show_if=("method", ["tr-nnls", "tr-rbf"])),
            ParameterSpec("f_min", "f_min", kind="float", default=0.0, minimum=0.0,
                          step=1.0, decimals=3, suffix="Hz",
                          tooltip="Analysis window lower bound. 0 = no limit.",
                          group="Frequency window"),
            ParameterSpec("f_max", "f_max", kind="float", default=0.0, minimum=0.0,
                          step=1.0, decimals=3, suffix="Hz",
                          tooltip="Analysis window upper bound. 0 = no limit.",
                          group="Frequency window"),
        ]

    @staticmethod
    def _read_and_average(files, f_min, f_max):
        """Read all EIS files, window to freq range, and average. Returns
        ``(freq, z_avg, warnings)``."""
        import pyimpspec

        data_list, warnings = [], []
        for fp in files:
            data, err = read_eis_data(str(fp))
            if err:
                warnings.append(f"{Path(fp).name}: {err.splitlines()[0]}")
            else:
                data_list.append(data)
                if data.get("warning"):
                    warnings.append(f"{Path(fp).name}: {data['warning']}")

        if not data_list:
            raise ValueError("No EIS spectra could be read from the selected files.")

        # Window and average
        rows = []
        for d in data_list:
            freq = np.asarray(d["frequency"], float)
            z = np.asarray(d["z_complex"])
            mask = np.isfinite(freq) & (freq > 0) & np.isfinite(z)
            freq, z = freq[mask], z[mask]
            if f_min is not None:
                m = freq >= f_min
                freq, z = freq[m], z[m]
            if f_max is not None:
                m = freq <= f_max
                freq, z = freq[m], z[m]
            if len(freq) >= 3:
                for fi, zi in zip(freq, z):
                    rows.append({"f": float(fi), "zr": float(np.real(zi)),
                                 "z_im": float(np.imag(zi))})

        if not rows:
            raise ValueError("No data points in the selected frequency window.")

        df = pd.DataFrame(rows).groupby("f", sort=True)
        f_avg = np.asarray(list(df.groups.keys()), float)
        z_avg = df["zr"].mean().values + 1j * df["z_im"].mean().values
        return f_avg, z_avg, warnings

    def run(self, files: list[Path], params: dict, extra=None) -> ToolResult:
        import pyimpspec

        f_min = float(params.get("f_min", 0.0)) or None
        f_max = float(params.get("f_max", 0.0)) or None

        f_avg, z_avg, warnings = self._read_and_average(files, f_min, f_max)
        if len(f_avg) < 5:
            raise ValueError(
                f"Only {len(f_avg)} averaged point(s) in the frequency window — "
                "need at least 5. Widen f_min/f_max or select more files."
            )

        ds = pyimpspec.DataSet(f_avg, z_avg)

        method = str(params.get("method", "tr-nnls"))
        kwargs = {}
        if method in ("tr-nnls", "tr-rbf"):
            lam = float(params.get("lambda", 0.0))
            if lam > 0:
                kwargs["lambda"] = lam

        drt_result = pyimpspec.calculate_drt(ds, method=method, **kwargs)

        # Full DRT distribution via peak analysis
        peaks = drt_result.analyze_peaks(num_peaks=0)
        tau = peaks.get_time_constants()
        gamma = peaks.get_gammas()

        # Identified peaks table
        df_peaks = drt_result.to_peaks_dataframe()

        # Statistics
        df_stats = drt_result.to_statistics_dataframe()

        # Residuals
        f_res, res_re, res_im = drt_result.get_residuals_data()

        # ---- Figures ----
        figures = {}

        # Panel 1: DRT distribution + Nyquist
        fig = Figure(figsize=(10, 6.5))
        ax_drt = fig.add_subplot(221)
        ax_ny = fig.add_subplot(222)
        ax_re = fig.add_subplot(223)
        ax_im = fig.add_subplot(224)

        # DRT: log gamma vs log tau
        ax_drt.loglog(tau, gamma, color=_DRT_COLOR, linewidth=1.6, label="DRT")
        # Shade identified peaks
        if len(df_peaks) > 0:
            peak_tau = df_peaks["tau (s)"].to_numpy()
            peak_gamma = df_peaks["gamma (ohm)"].to_numpy()
            ax_drt.scatter(peak_tau, peak_gamma, c=_PEAK_COLOR, s=40,
                           zorder=5, edgecolors="black", linewidths=0.5,
                           label=f"{len(df_peaks)} peak(s)")
            for i, (t, g) in enumerate(zip(peak_tau, peak_gamma)):
                ax_drt.annotate(
                    f"  {t:.2e} s", (t, g),
                    fontsize=7, color=theme.TEXT, alpha=0.85,
                    va="bottom" if i % 2 == 0 else "top",
                )
        ax_drt.set_xlabel("τ  (s)")
        ax_drt.set_ylabel("γ  (Ω)")
        ax_drt.set_title("DRT distribution", loc="left")
        ax_drt.legend(fontsize=8)

        # Nyquist (measured data only — DRT reconstruction uses measured - residuals
        # which may not preserve the exact impedance; the key DRT output is the
        # gamma-tau distribution above)
        ax_ny.plot(np.real(z_avg), -np.imag(z_avg), "o", markersize=4,
                   markerfacecolor="white", markeredgecolor=_DATA,
                   markeredgewidth=1.2, linestyle="none", label="Measured")
        ax_ny.set_aspect("equal", adjustable="datalim")
        ax_ny.set_xlabel("Z′  (Ω·cm²)")
        ax_ny.set_ylabel("−Z″  (Ω·cm²)")
        ax_ny.set_title("Nyquist", loc="left")
        ax_ny.legend(fontsize=8)

        # Real residuals
        ax_re.semilogx(f_res, res_re, color=_DRT_COLOR, linewidth=1.2)
        ax_re.axhline(0, color=theme.BORDER, linestyle="--", linewidth=0.5)
        ax_re.set_xlabel("Frequency (Hz)")
        ax_re.set_ylabel("ΔRe(Z)  (Ω·cm²)")
        ax_re.set_title("Real residuals", loc="left")

        # Imaginary residuals
        ax_im.semilogx(f_res, res_im, color=_DRT_COLOR, linewidth=1.2)
        ax_im.axhline(0, color=theme.BORDER, linestyle="--", linewidth=0.5)
        ax_im.set_xlabel("Frequency (Hz)")
        ax_im.set_ylabel("ΔIm(Z)  (Ω·cm²)")
        ax_im.set_title("Imaginary residuals", loc="left")

        fig.tight_layout()
        figures["DRT analysis"] = fig

        # ---- Scalars ----
        scalars = {
            "Data points": int(len(f_avg)),
            "DRT method": method,
            "Peaks found": int(len(df_peaks)),
        }

        # ---- Tables ----
        tables = {
            "DRT peaks": df_peaks,
            "DRT statistics": df_stats,
        }

        # ---- Text ----
        text_blocks = {}
        chi2 = df_stats.loc[df_stats["Label"] == "Log pseudo chi-squared", "Value"]
        if len(chi2):
            chi2_val = float(chi2.iloc[0])
            quality = ("Excellent fit" if chi2_val < -4 else
                       "Good fit" if chi2_val < -2 else
                       "Moderate fit — inspect residuals" if chi2_val < 0 else
                       "Poor fit — consider a different method or check data quality")
            text_blocks["Fit quality"] = (
                f"Log pseudo χ² = {chi2_val:.3f}\n{quality}"
            )

        if len(df_peaks) > 0:
            peak_info = "\n".join(
                f"  τ = {row['tau (s)']:.2e} s  →  f = {1.0/(2*np.pi*row['tau (s)']):.3g} Hz"
                for _, row in df_peaks.iterrows()
            )
            text_blocks["Identified time constants"] = peak_info

        return ToolResult(
            figures=figures, tables=tables, scalars=scalars,
            text_blocks=text_blocks, warnings=warnings,
        )

    def exportable_artifacts(self, result: ToolResult) -> list[Artifact]:
        items = []
        if result.tables:
            items.append(Artifact(
                name="DRT results (XLSX)", kind="xlsx",
                default_filename="drt_results.xlsx",
                payload=dict(result.tables),
            ))
        for fname, fig in result.figures.items():
            items.append(Artifact(
                name=f"{fname} (PNG)", kind="png",
                default_filename=f"drt_{fname.lower().replace(' ', '_')}.png",
                payload=fig,
            ))
        return items
