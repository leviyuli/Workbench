"""Distribution of relaxation times and residual diagnostics for EIS spectra."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from pathlib import Path

from workbench import theme
from workbench.core.data_io import read_eis_data
from workbench.core.tool_base import Artifact, ParameterSpec, Tool, ToolResult

_DRT_METHODS = ["tr-nnls", "tr-rbf", "bht", "lm"]
_DATA = theme.OVERPOT_COLORS["eta_kin"]
_DRT_COLOR = "#c44e52"
_PEAK_COLOR = "#e5ae38"
_GAMMA_UNIT = "Ω·cm²"


class DrtTool(Tool):
    id = "drt"
    name = "DRT analysis"
    subtitle = "Inspect relaxation-time distributions and impedance residuals."
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
                                  "BHT and LM are alternative algorithms.",
                          group="DRT options"),
            ParameterSpec("lambda", "Regularization λ", kind="float",
                          default=0.0, minimum=0.0, maximum=1.0, step=0.001,
                          decimals=6,
                          tooltip="Tikhonov regularization strength for TR methods. "
                                  "0 = auto-select for the chosen method. "
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
        """Average selected spectra within the frequency window."""
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
        if method not in _DRT_METHODS:
            raise ValueError(f"Choose a supported DRT method: {', '.join(_DRT_METHODS)}.")
        kwargs = {}
        if method in ("tr-nnls", "tr-rbf"):
            lam = float(params.get("lambda", 0.0))
            if lam > 0:
                kwargs["lambda_value"] = lam
                if method == "tr-rbf":
                    kwargs["cross_validation"] = ""

        drt_result = pyimpspec.calculate_drt(ds, method=method, **kwargs)

        df_peaks = drt_result.to_peaks_dataframe()
        df_peaks = df_peaks.rename(columns=lambda label: label.replace("ohm", _GAMMA_UNIT))
        df_stats = drt_result.to_statistics_dataframe()
        f_res, res_re, res_im = drt_result.get_residuals_data()

        drt_data = drt_result.get_drt_data()
        if method == "bht":
            tau, gamma_re, gamma_im = drt_data
            distributions = [(tau, gamma_re, "Real", "tau, real (s)",
                              f"gamma, real ({_GAMMA_UNIT})"),
                             (tau, gamma_im, "Imaginary", "tau, imag. (s)",
                              f"gamma, imag. ({_GAMMA_UNIT})")]
        elif method == "lm":
            tau_rc, gamma_rc, tau_rl, gamma_rl = drt_data
            distributions = [(tau_rc, gamma_rc, "RC", "tau, RC (s)",
                              f"gamma, RC ({_GAMMA_UNIT})"),
                             (tau_rl, gamma_rl, "RL", "tau, RL (s)",
                              f"gamma, RL ({_GAMMA_UNIT})")]
        else:
            tau, gamma = drt_data
            distributions = [(tau, gamma, "DRT", "tau (s)", f"gamma ({_GAMMA_UNIT})")]

        figures = {}
        fig = Figure(figsize=(10, 6.5))
        ax_drt = fig.add_subplot(221)
        ax_ny = fig.add_subplot(222)
        ax_re = fig.add_subplot(223)
        ax_im = fig.add_subplot(224)

        identified = []
        for index, (tau, gamma, label, tau_column, gamma_column) in enumerate(distributions):
            if len(tau):
                ax_drt.semilogx(tau, gamma, color=_DRT_COLOR if index == 0 else _DATA,
                               linestyle="-" if index == 0 else "--",
                               linewidth=1.6, label=label)
            peak_values = df_peaks[[tau_column, gamma_column]].dropna()
            peak_tau = peak_values[tau_column].to_numpy()
            peak_gamma = peak_values[gamma_column].to_numpy()
            identified.extend((label, t, g) for t, g in zip(peak_tau, peak_gamma))
            if not len(peak_values):
                continue
            ax_drt.scatter(peak_tau, peak_gamma, c=_PEAK_COLOR if index == 0 else _DATA, s=40,
                           zorder=5, edgecolors="black", linewidths=0.5,
                           label=f"{label}: {len(peak_values)} peak(s)")
            for i, (t, g) in enumerate(zip(peak_tau, peak_gamma)):
                ax_drt.annotate(
                    f"  {t:.2e} s", (t, g),
                    fontsize=7, color=theme.TEXT, alpha=0.85,
                    va="bottom" if i % 2 == 0 else "top",
                )
        ax_drt.set_xlabel("τ  (s)")
        ax_drt.set_ylabel(f"γ  ({_GAMMA_UNIT})")
        ax_drt.set_title("DRT distribution", loc="left")
        ax_drt.legend(fontsize=8)

        ax_ny.plot(np.real(z_avg), -np.imag(z_avg), "o", markersize=4,
                   markerfacecolor="white", markeredgecolor=_DATA,
                   markeredgewidth=1.2, linestyle="none", label="Measured")
        ax_ny.set_aspect("equal", adjustable="datalim")
        ax_ny.set_xlabel("Z′  (Ω·cm²)")
        ax_ny.set_ylabel("−Z″  (Ω·cm²)")
        ax_ny.set_title("Nyquist", loc="left")
        ax_ny.legend(fontsize=8)

        ax_re.semilogx(f_res, res_re, color=_DRT_COLOR, linewidth=1.2)
        ax_re.axhline(0, color=theme.BORDER, linestyle="--", linewidth=0.5)
        ax_re.set_xlabel("Frequency (Hz)")
        ax_re.set_ylabel("Real residual (%)")
        ax_re.set_title("Real residuals", loc="left")

        ax_im.semilogx(f_res, res_im, color=_DRT_COLOR, linewidth=1.2)
        ax_im.axhline(0, color=theme.BORDER, linestyle="--", linewidth=0.5)
        ax_im.set_xlabel("Frequency (Hz)")
        ax_im.set_ylabel("Imaginary residual (%)")
        ax_im.set_title("Imaginary residuals", loc="left")

        fig.tight_layout()
        figures["DRT analysis"] = fig

        scalars = {
            "Data points": int(len(f_avg)),
            "DRT method": method,
            "Peaks found": len(identified),
        }

        distribution_columns = {}
        for tau, gamma, _label, tau_column, gamma_column in distributions:
            if not len(tau):
                continue
            distribution_columns[tau_column] = pd.Series(np.asarray(tau, dtype=float))
            distribution_columns[gamma_column] = pd.Series(np.asarray(gamma, dtype=float))
        tables = {
            "DRT distribution": pd.DataFrame(distribution_columns),
            "DRT peaks": df_peaks,
            "DRT statistics": pd.concat([
                pd.DataFrame({"Label": ["DRT method"], "Value": [method]}), df_stats,
            ], ignore_index=True),
            "Measured impedance": pd.DataFrame({
                "Frequency (Hz)": f_avg, "Re(Z) (Ω·cm²)": np.real(z_avg),
                "Im(Z) (Ω·cm²)": np.imag(z_avg),
            }),
            "Residuals": pd.DataFrame({
                "Frequency (Hz)": f_res, "Real residual (%)": res_re,
                "Imaginary residual (%)": res_im,
            }),
        }

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

        if identified:
            peak_info = "\n".join(
                f"  {label}: τ = {t:.2e} s  →  f = {1.0/(2*np.pi*t):.3g} Hz"
                for label, t, _gamma in identified
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
