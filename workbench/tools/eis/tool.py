from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from workbench import theme
from workbench.core.data_io import read_eis_data
from workbench.core.tool_base import Artifact, ParameterSpec, Tool, ToolResult
from workbench.tools.eis.logic_eis import EisLogic

_MODELS = [
    "Transmission Line",
    "Faradaic Transmission Line (CPE)",
    "1-D Linear Diffusion",
    "1-D Spherical Diffusion",
]
# Canonical order: [L_wire, HFR, R_CL, Q_dl, Phi, Theta, R_k]
_CANON = ["L_wire", "HFR", "R_CL", "Q_dl", "Phi", "Theta", "R_k"]
_UNITS = ["H·cm²", "Ω·cm²", "Ω·cm²", "S·sᵠ/cm²", "—", "—", "Ω·cm²"]

_DATA = theme.OVERPOT_COLORS["eta_kin"]   # indigo for measured points
_FIT = theme.DANGER                       # crimson for the model curve


class EisTool(Tool):
    id = "eis"
    name = "EIS (OSIF)"
    subtitle = ("Kramers-Kronig validation, transmission-line fitting, and forward "
                "simulation of impedance spectra. Use 'KK test' for a quick quality check.")
    needs_files = True
    files_required = False        # Simulate mode can run without files
    multiple_files = True
    accepts_directory = False
    accepted_file_types = (".csv", ".txt", ".xlsx", ".xls")
    run_button_label = "Run"

    def parameter_schema(self) -> list[ParameterSpec]:
        return [
            ParameterSpec("mode", "Mode", kind="choice",
                          choices=["Fit", "KK test", "Simulate"], default="Fit",
                          tooltip="Fit: KK-validate + fit loaded spectra. "
                                  "KK test: KK validation only (fast quality check). "
                                  "Simulate: forward model from the values below.",
                          group="Mode"),
            ParameterSpec("model", "Model", kind="choice", choices=_MODELS,
                          default="Transmission Line", group="Mode",
                          show_if=("mode", ["Fit", "Simulate"])),
            ParameterSpec("fit_inductance", "Fit wire inductance (L_wire, Θ)",
                          kind="bool", default=False, group="Mode",
                          show_if=("mode", ["Fit", "Simulate"])),

            ParameterSpec("f_min", "f_min", kind="float", default=0.0, minimum=0.0,
                          step=1.0, decimals=3, suffix="Hz",
                          tooltip="Analysis/fit-window lower bound. 0 = no limit. "
                                  "KK test fills this with the suggested valid range.",
                          group="Frequency window"),
            ParameterSpec("f_max", "f_max", kind="float", default=0.0, minimum=0.0,
                          step=1.0, decimals=3, suffix="Hz",
                          tooltip="Analysis/fit-window upper bound. 0 = no limit. "
                                  "KK test fills this with the suggested valid range.",
                          group="Frequency window"),

            ParameterSpec("hfr", "HFR", kind="float", default=0.0, minimum=0.0,
                          step=0.001, decimals=5, suffix="Ω·cm²",
                          tooltip="0 = auto-seed from the high-frequency intercept (Fit mode).",
                          group="Parameters (init / simulate)",
                          show_if=("mode", ["Fit", "Simulate"])),
            ParameterSpec("rcl", "R_CL", kind="float", default=0.2, minimum=0.0,
                          step=0.01, decimals=5, suffix="Ω·cm²",
                          group="Parameters (init / simulate)",
                          show_if=("mode", ["Fit", "Simulate"])),
            ParameterSpec("qdl", "Q_dl", kind="float", default=0.1, minimum=0.0,
                          step=0.01, decimals=6, suffix="S·sᵠ",
                          group="Parameters (init / simulate)",
                          show_if=("mode", ["Fit", "Simulate"])),
            ParameterSpec("phi", "Phi", kind="float", default=0.9, minimum=0.0,
                          maximum=1.0, step=0.01, decimals=3,
                          group="Parameters (init / simulate)",
                          show_if=("mode", ["Fit", "Simulate"])),
            ParameterSpec("rk", "R_k (faradaic)", kind="float", default=1.0, minimum=0.0,
                          step=0.1, decimals=5, suffix="Ω·cm²",
                          tooltip="Only used by the Faradaic TML model.",
                          group="Parameters (init / simulate)",
                          show_if=("mode", ["Fit", "Simulate"])),
            ParameterSpec("lwire", "L_wire", kind="float", default=2e-5, minimum=0.0,
                          step=1e-6, decimals=9, suffix="H·cm²",
                          group="Parameters (init / simulate)",
                          show_if=("mode", ["Fit", "Simulate"])),
            ParameterSpec("theta", "Theta", kind="float", default=0.95, minimum=0.0,
                          maximum=1.0, step=0.01, decimals=3,
                          group="Parameters (init / simulate)",
                          show_if=("mode", ["Fit", "Simulate"])),

            ParameterSpec("restarts", "Fit restarts", kind="int", default=8,
                          minimum=1, maximum=64, step=1, group="Fit options",
                          show_if=("mode", "Fit")),
            ParameterSpec("max_nfev", "Max evaluations", kind="int", default=20000,
                          minimum=100, maximum=200000, step=1000, group="Fit options",
                          show_if=("mode", "Fit")),

            ParameterSpec("sim_fmin", "Sim f_min", kind="float", default=0.01,
                          minimum=1e-9, step=0.01, decimals=4, suffix="Hz",
                          group="Simulate options", show_if=("mode", "Simulate")),
            ParameterSpec("sim_fmax", "Sim f_max", kind="float", default=1e6,
                          minimum=1.0, step=1000.0, decimals=1, suffix="Hz",
                          group="Simulate options", show_if=("mode", "Simulate")),
            ParameterSpec("sim_points", "Sim points", kind="int", default=100,
                          minimum=2, maximum=5000, step=10,
                          group="Simulate options", show_if=("mode", "Simulate")),
        ]

    # ------------------------------------------------------------------ run
    def run(self, files: list[Path], params: dict, extra=None) -> ToolResult:
        mode = params.get("mode", "Fit")
        if mode == "Simulate":
            return self._run_simulate(files, params)
        if mode == "KK test":
            return self._run_kk(files, params)
        return self._run_fit(files, params)

    # --------------------------------------------------------------- KK test
    def _run_kk(self, files: list[Path], params: dict) -> ToolResult:
        if not files:
            raise ValueError("KK test needs at least one EIS file.")
        data_list, warnings = _read_all(files)
        if not data_list:
            raise ValueError("No EIS spectra could be read from the selected files.")

        logic = EisLogic()
        f_min = float(params.get("f_min", 0.0)) or None
        f_max = float(params.get("f_max", 0.0)) or None
        kk_range = (f_min, f_max) if (f_min or f_max) else None

        df_all, df_avg, suggested, rsd, kk_warn = logic.process_spectra(data_list, kk_range)
        warnings.extend(kk_warn)

        n_total = len(df_all)
        n_valid = int((df_all["kk_err"] <= logic.kk_threshold).sum())
        pct_valid = 100.0 * n_valid / n_total if n_total else 0.0
        hfr_est = float(df_avg["Z'(Ohm.cm²)"].min())

        figures = {"KK diagnostics": _plot_kk(df_all, df_avg, logic.kk_threshold)}
        tables = {"KK diagnostics (raw)": df_all, "Averaged spectrum": df_avg}
        scalars = {
            "Spectra": int(df_all["scan"].nunique()),
            "Points passing KK": f"{n_valid}/{n_total} ({pct_valid:.0f}%)",
            "Drift (RSD)": float(rsd),
            "Suggested f-range (Hz)": f"{suggested[0]:.3g} – {suggested[1]:.3g}",
            "HFR estimate (Ω·cm²)": hfr_est,
        }
        quality = ("Good — most points pass KK." if pct_valid >= 90 else
                   "Mixed — inspect the residual plot; consider narrowing f_min/f_max." if pct_valid >= 60 else
                   "Poor — many points fail KK; the spectrum may be noisy or drifting.")
        quality += ("\nThe suggested KK-valid window and HFR estimate have been "
                    "written into the fields — review them, switch Mode to 'Fit', and Run.")
        # Write the KK-valid window + HFR back into the panel so the user can review
        # and fit on it (mirrors the original OSIF preprocess→fit handoff).
        sug_min, sug_max = suggested
        param_updates = {
            "f_min": round(float(sug_min), 3),
            "f_max": round(float(sug_max), 3),
            "hfr": round(hfr_est, 5),
        }
        return ToolResult(figures=figures, tables=tables, scalars=scalars,
                          text_blocks={"KK quality": quality},
                          param_updates=param_updates, warnings=warnings)

    # ----------------------------------------------------------------- fit
    def _run_fit(self, files: list[Path], params: dict) -> ToolResult:
        if not files:
            raise ValueError("Fit mode needs at least one EIS file. "
                             "Switch Mode to 'Simulate' to model without data.")
        data_list, warnings = _read_all(files)
        if not data_list:
            raise ValueError("No EIS spectra could be read from the selected files.")

        logic = EisLogic()
        logic.fit_n_restarts = int(params.get("restarts", 8))
        logic.fit_max_nfev = int(params.get("max_nfev", 20000))

        f_min = float(params.get("f_min", 0.0)) or None
        f_max = float(params.get("f_max", 0.0)) or None
        kk_range = (f_min, f_max) if (f_min or f_max) else None

        df_all, df_avg, suggested, rsd, kk_warn = logic.process_spectra(data_list, kk_range)
        warnings.extend(kk_warn)

        # The f_min/f_max fields ARE the fit window: process_spectra has already
        # restricted df_avg to that scope, so fit on the whole averaged spectrum.
        f_fit = df_avg["Freq(Hz)"].to_numpy()
        z_exp = (df_avg["Z'(Ohm.cm²)"].to_numpy()
                 + 1j * df_avg["Z''(Ohm.cm²)"].to_numpy())
        if len(f_fit) < 4:
            raise ValueError(
                f"Only {len(f_fit)} point(s) in the window — widen f_min/f_max "
                "or run KK test first to get a suggested range."
            )

        model = params.get("model", "Transmission Line")
        fit_inductance = bool(params.get("fit_inductance", False))
        fit_faradaic = (model == EisLogic.FARADAIC_TML_MODEL)

        hfr = float(params.get("hfr", 0.0))
        if hfr <= 0:
            # Auto-seed from the high-frequency intercept (Z′ at the highest f).
            hfr = max(float(np.real(z_exp)[int(np.argmax(f_fit))]), 1e-4)
        rcl = float(params["rcl"]); qdl = float(params["qdl"]); phi = float(params["phi"])
        rk = float(params["rk"]); lwire = float(params["lwire"]); theta = float(params["theta"])

        if fit_inductance and fit_faradaic:
            init = [lwire, hfr, rcl, qdl, phi, theta, rk]
        elif fit_inductance:
            init = [lwire, hfr, rcl, qdl, phi, theta]
        elif fit_faradaic:
            init = [hfr, rcl, qdl, phi, rk]
        else:
            init = [hfr, rcl, qdl, phi]

        out, err = logic.fit_impedance(model, init, f_fit, z_exp,
                                       fit_inductance=fit_inductance)
        figures = {"KK diagnostics": _plot_kk(df_all, df_avg, logic.kk_threshold)}
        tables = {"KK diagnostics (raw)": df_all, "Averaged spectrum": df_avg}
        scalars = {"Drift (RSD)": float(rsd),
                   "Suggested f-range (Hz)": f"{suggested[0]:.3g} – {suggested[1]:.3g}",
                   "Fit points": int(len(f_fit))}

        if err:
            warnings.append(f"Fit failed: {err}")
            return ToolResult(figures=figures, tables=tables, scalars=scalars,
                              warnings=warnings)

        p, se, _resid, info = out
        z_model = logic.evaluate_model(model, p, f_fit)

        figures["Fit results"] = _plot_panels(f_fit, z_exp, z_model, "Fit", _FIT,
                                              param_box=_param_box(p, model, fit_inductance,
                                                                   fit_faradaic))
        tables["Fitted curve"] = pd.DataFrame({
            "Frequency (Hz)": f_fit,
            "Data Re(Z)": np.real(z_exp), "Data Im(Z)": np.imag(z_exp),
            "Fit Re(Z)": np.real(z_model), "Fit Im(Z)": np.imag(z_model),
            "Fit |Z|": np.abs(z_model),
        })
        tables["Fit parameters"] = _param_table(p, se, model, fit_inductance, fit_faradaic)

        scalars.update({
            "HFR (Ω·cm²)": float(p[1]), "R_CL (Ω·cm²)": float(p[2]),
            "Q_dl": float(p[3]), "Phi": float(p[4]),
            "SSR": float(info.get("ssr", float("nan"))),
            "Converged": bool(info.get("converged", False)),
        })
        if model == "Transmission Line" and not fit_faradaic:
            scalars["R_CL/3 (Ω·cm²)"] = float(p[2]) / 3.0
        if fit_faradaic:
            scalars["R_k (Ω·cm²)"] = float(p[6])
        if fit_inductance:
            scalars["L_wire (H·cm²)"] = float(p[0])

        # Flag high-uncertainty parameters.
        flagged = _flag_high_se(p, se, model, fit_inductance, fit_faradaic,
                                logic.fit_high_uncertainty_pct)
        text = (f"Best of {info['n_restarts']} restarts (#{info['best_attempt']}, "
                f"by {'HFR/R_CL accuracy' if info['ranking_basis'] == 'hfr_rcl_se' else 'lowest SSR'})\n"
                f"max rel. SE on HFR/R_CL = {info['score_hfr_rcl_pct']:.1f}%  ·  "
                f"SSR = {info['ssr']:.3g}")
        if flagged:
            text += "\nHigh uncertainty: " + ", ".join(flagged)

        # Write the fitted values back into the init fields so the next Run
        # refines from this solution, and annotate each with its ± std. error.
        # Only the parameters that were actually fitted are touched (canonical p
        # holds 0/∞ placeholders otherwise).
        keys_idx = {"hfr": 1, "rcl": 2, "qdl": 3, "phi": 4}
        if fit_inductance:
            keys_idx["lwire"] = 0
            keys_idx["theta"] = 5
        if fit_faradaic:
            keys_idx["rk"] = 6
        param_updates = {k: float(p[i]) for k, i in keys_idx.items()}
        param_annotations = {k: _fmt_se(float(p[i]), float(se[i]))
                             for k, i in keys_idx.items()}

        return ToolResult(figures=figures, tables=tables, scalars=scalars,
                          text_blocks={"Fit diagnostics": text},
                          param_updates=param_updates,
                          param_annotations=param_annotations, warnings=warnings)

    # ------------------------------------------------------------- simulate
    def _run_simulate(self, files: list[Path], params: dict) -> ToolResult:
        logic = EisLogic()
        model = params.get("model", "Transmission Line")
        canon = [float(params["lwire"]), float(params["hfr"]), float(params["rcl"]),
                 float(params["qdl"]), float(params["phi"]), float(params["theta"]),
                 float(params["rk"])]
        if not params.get("fit_inductance", False):
            canon[0] = 0.0  # L_wire off
        if model != EisLogic.FARADAIC_TML_MODEL:
            canon[6] = np.inf  # R_k unused

        fmin = float(params.get("sim_fmin", 0.01))
        fmax = float(params.get("sim_fmax", 1e6))
        n = int(params.get("sim_points", 100))
        if fmin <= 0 or fmax <= 0 or fmin >= fmax:
            raise ValueError("Simulate frequency range must satisfy 0 < f_min < f_max.")
        freq = np.logspace(np.log10(fmin), np.log10(fmax), max(n, 2))
        z_sim = logic.evaluate_model(model, canon, freq)

        # Optional overlay: averaged raw measurement within the window.
        f_exp = z_exp = None
        warnings: list[str] = []
        if files:
            data_list, warnings = _read_all(files)
            if data_list:
                f_src, z_src = _raw_average(data_list)
                m = (f_src >= fmin) & (f_src <= fmax)
                if m.any():
                    f_exp, z_exp = f_src[m], z_src[m]

        fig = _plot_panels(freq, z_exp, z_sim, "Simulation", _FIT, sim=True,
                           f_overlay=f_exp, z_overlay=z_exp)
        table = pd.DataFrame({
            "Frequency (Hz)": freq, "Re(Z)": np.real(z_sim),
            "Im(Z)": np.imag(z_sim), "|Z|": np.abs(z_sim),
        })
        scalars = {"Model": model, "Points": len(freq),
                   "f-range (Hz)": f"{freq.min():.3g} – {freq.max():.3g}"}
        return ToolResult(figures={"Simulation": fig}, tables={"Simulated curve": table},
                          scalars=scalars, warnings=warnings)

    # ------------------------------------------------------------- exports
    def exportable_artifacts(self, result: ToolResult) -> list[Artifact]:
        items = []
        if result.tables:
            items.append(Artifact(name="Results (XLSX)", kind="xlsx",
                                  default_filename="eis_results.xlsx",
                                  payload=dict(result.tables)))
        for fname, fig in result.figures.items():
            items.append(Artifact(name=f"{fname} (PNG)", kind="png",
                                  default_filename=f"eis_{fname.lower().replace(' ', '_')}.png",
                                  payload=fig))
        return items


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_all(files):
    data_list, warnings = [], []
    for fp in files:
        data, err = read_eis_data(str(fp))
        if err:
            warnings.append(f"{Path(fp).name}: {err.splitlines()[0]}")
        else:
            data_list.append(data)
    return data_list, warnings


def _raw_average(data_list):
    rows = []
    for d in data_list:
        rows.append(pd.DataFrame({"f": np.asarray(d["frequency"], float),
                                  "zr": np.asarray(d["z_real"], float),
                                  "zi": np.asarray(d["z_imag"], float)}))
    df = pd.concat(rows, ignore_index=True).groupby("f", sort=True)
    f = np.asarray(list(df.groups.keys()), float)
    return f, df["zr"].mean().values + 1j * df["zi"].mean().values


def _fmt_se(val: float, err: float) -> str:
    """Compact '± SE (rel%)' note for a fitted parameter; '± —' if undefined.

    The SD is shown in scientific notation with 3 significant figures so the
    note stays short and fits the parameter-panel annotation label (a raw
    fixed-point SD can run long and get clipped)."""
    if not np.isfinite(err):
        return "± —"
    note = f"± {err:.2e}"
    if val != 0 and np.isfinite(abs(err) / abs(val)):
        note += f" ({abs(err) / abs(val) * 100:.1f}%)"
    return note


def _param_box(p, model, fit_inductance, fit_faradaic) -> str:
    lines = [f"HFR  = {p[1]:.4f} Ω·cm²", f"R_CL = {p[2]:.4f} Ω·cm²",
             f"Q_dl = {p[3]:.3g}", f"φ    = {p[4]:.3f}"]
    if model == "Transmission Line" and not fit_faradaic:
        lines.append(f"R_CL/3 = {p[2] / 3.0:.4f} Ω·cm²")
    if fit_faradaic:
        lines.append(f"R_k  = {p[6]:.4f} Ω·cm²")
    if fit_inductance:
        lines.append(f"L_wire = {p[0]:.3e}")
        lines.append(f"Θ    = {p[5]:.3f}")
    return "\n".join(lines)


def _active_keys(model, fit_inductance, fit_faradaic):
    keys = ["HFR", "R_CL", "Q_dl", "Phi"]
    if fit_inductance:
        keys = ["L_wire"] + keys + ["Theta"]
    if fit_faradaic:
        keys = keys + ["R_k"]
    return keys


def _param_table(p, se, model, fit_inductance, fit_faradaic) -> pd.DataFrame:
    active = _active_keys(model, fit_inductance, fit_faradaic)
    rows = []
    for k in active:
        idx = _CANON.index(k)
        val, err = float(p[idx]), float(se[idx])
        pct = (abs(err) / abs(val) * 100) if val != 0 else float("inf")
        rows.append({"Parameter": k, "Value": val, "Std. error": err,
                     "Rel. SE (%)": pct, "Unit": _UNITS[idx]})
    if model == "Transmission Line" and not fit_faradaic:
        idx = _CANON.index("R_CL")
        rows.append({"Parameter": "R_CL/3", "Value": float(p[idx]) / 3.0,
                     "Std. error": float(se[idx]) / 3.0, "Rel. SE (%)": float("nan"),
                     "Unit": "Ω·cm²"})
    return pd.DataFrame(rows)


def _flag_high_se(p, se, model, fit_inductance, fit_faradaic, threshold):
    flagged = []
    for k in _active_keys(model, fit_inductance, fit_faradaic):
        idx = _CANON.index(k)
        val = float(p[idx])
        pct = (abs(float(se[idx])) / abs(val) * 100) if val != 0 else float("inf")
        if not np.isfinite(pct) or pct > threshold:
            flagged.append(f"{k} ±{pct:.0f}%" if np.isfinite(pct) else f"{k} ±inf")
    return flagged


def _plot_kk(df_all, df_avg, kk_threshold) -> Figure:
    fig = Figure(figsize=(10, 4.2))
    ax_n = fig.add_subplot(121)
    ax_b = fig.add_subplot(122)

    valid = df_all["kk_err"] <= kk_threshold
    ax_n.scatter(df_all.loc[valid, "zr"], -df_all.loc[valid, "zi"],
                 c=_DATA, alpha=0.45, s=18, edgecolor="none", label="KK valid")
    ax_n.scatter(df_all.loc[~valid, "zr"], -df_all.loc[~valid, "zi"],
                 c=_FIT, marker="x", s=22, linewidth=1.2, label="KK invalid")
    ax_n.plot(df_avg["Z'(Ohm.cm²)"], -df_avg["Z''(Ohm.cm²)"],
              color=theme.TEXT, linewidth=1.6, label="averaged")
    ax_n.set_aspect("equal", adjustable="datalim")
    ax_n.set_xlabel("Z′  (Ω·cm²)")
    ax_n.set_ylabel("−Z″  (Ω·cm²)")
    ax_n.set_title("Nyquist · KK validation", loc="left")
    ax_n.legend(fontsize=8)

    ax_b.scatter(df_all["f"], df_all["kk_err"] * 100, c=theme.MUTED, alpha=0.6,
                 s=12, edgecolor="none")
    ax_b.axhline(kk_threshold * 100, color=theme.OVERPOT_COLORS["eta_rcl"],
                 linestyle="--", linewidth=1.2, label=f"threshold ({kk_threshold*100:.0f}%)")
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel("Frequency (Hz)")
    ax_b.set_ylabel("KK residual (%)")
    ax_b.set_title("Point-wise KK error", loc="left")
    ax_b.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _plot_panels(freq, z_data, z_model, model_label, model_color, *,
                 param_box=None, sim=False, f_overlay=None, z_overlay=None) -> Figure:
    """2x2: Nyquist (equal aspect), Bode |Z|, Re(Z) vs f, Im(Z) vs f."""
    fig = Figure(figsize=(10, 6.5))
    ax_ny = fig.add_subplot(221)
    ax_bode = fig.add_subplot(222)
    ax_re = fig.add_subplot(223)
    ax_im = fig.add_subplot(224)

    data_f = f_overlay if sim else freq
    data_z = z_overlay if sim else z_data

    # Nyquist — equal unit length on both axes.
    if data_z is not None:
        ax_ny.plot(np.real(data_z), -np.imag(data_z), "o", markersize=4,
                   markerfacecolor="white", markeredgecolor=_DATA,
                   markeredgewidth=1.2, linestyle="none", label="Data")
    ax_ny.plot(np.real(z_model), -np.imag(z_model), color=model_color,
               linewidth=1.8, label=model_label)
    ax_ny.set_aspect("equal", adjustable="datalim")
    ax_ny.set_xlabel("Z′  (Ω·cm²)")
    ax_ny.set_ylabel("−Z″  (Ω·cm²)")
    ax_ny.set_title("Nyquist", loc="left")
    ax_ny.legend(loc="upper left", fontsize=8)
    if param_box:
        ax_ny.text(0.97, 0.05, param_box, transform=ax_ny.transAxes,
                   ha="right", va="bottom", fontsize=8, color=theme.TEXT,
                   family="monospace",
                   bbox=dict(boxstyle="round,pad=0.4", fc="white",
                             ec=theme.BORDER, lw=0.6, alpha=0.9))

    if data_z is not None:
        ax_bode.loglog(data_f, np.abs(data_z), "o", markersize=4, markerfacecolor="white",
                       markeredgecolor=_DATA, markeredgewidth=1.2, linestyle="none", label="Data")
    ax_bode.loglog(freq, np.abs(z_model), color=model_color, linewidth=1.6, label=model_label)
    ax_bode.set_xlabel("Frequency (Hz)")
    ax_bode.set_ylabel("|Z|  (Ω·cm²)")
    ax_bode.set_title("Bode · |Z|", loc="left")
    ax_bode.legend(fontsize=8)

    if data_z is not None:
        ax_re.semilogx(data_f, np.real(data_z), "o", markersize=4, markerfacecolor="white",
                       markeredgecolor=_DATA, markeredgewidth=1.2, linestyle="none")
    ax_re.semilogx(freq, np.real(z_model), color=model_color, linewidth=1.6)
    ax_re.set_xlabel("Frequency (Hz)")
    ax_re.set_ylabel("Re(Z)  (Ω·cm²)")
    ax_re.set_title("Re(Z)", loc="left")

    if data_z is not None:
        ax_im.semilogx(data_f, -np.imag(data_z), "o", markersize=4, markerfacecolor="white",
                       markeredgecolor=_DATA, markeredgewidth=1.2, linestyle="none")
    ax_im.semilogx(freq, -np.imag(z_model), color=model_color, linewidth=1.6)
    ax_im.set_xlabel("Frequency (Hz)")
    ax_im.set_ylabel("−Im(Z)  (Ω·cm²)")
    ax_im.set_title("−Im(Z)", loc="left")

    fig.tight_layout()
    return fig
