from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from workbench import theme
from workbench.core.data_io import read_eis_data
from workbench.core.tool_base import Artifact, ParameterSpec, Tool, ToolResult
from workbench.tools.eis.logic_eis import EisLogic

_MODELS_NF = [
    "Transmission Line",
    "1-D Linear Diffusion",
    "1-D Spherical Diffusion",
]
_MODELS_F = [
    "Faradaic Transmission Line (CPE)",
]
_MAX_RC = 4  # maximum number of interfacial RC units in the faradaic model
# Canonical order: [L_wire, HFR, R_CL, Q_dl, Phi, Theta, R_k]
_CANON = ["L_wire", "HFR", "R_CL", "Q_dl", "Phi", "Theta", "R_k"]
_UNITS = ["H·cm²", "Ω·cm²", "Ω·cm²", "S·sᵠ/cm²", "—", "—", "Ω·cm²"]

_DATA = theme.OVERPOT_COLORS["eta_kin"]   # indigo for measured points
_FIT = theme.DANGER                       # crimson for the model curve


class EisTool(Tool):
    id = "eis"
    name = "EIS (OSIF)"
    subtitle = ("Kramers-Kronig validation, transmission-line fitting, and forward "
                "simulation of impedance spectra. Use 'Batch KK test' for separate HFR results per file.")
    needs_files = True
    files_required = False        # Simulate mode can run without files
    multiple_files = True
    accepts_directory = False
    accepted_file_types = (".csv", ".txt", ".xlsx", ".xls")
    run_button_label = "Run"

    # Fit reuses KK residuals while the files and frequency coverage match.
    _kk_cache = None

    def reset_runtime_state(self) -> None:
        self._kk_cache = None

    def parameter_schema(self) -> list[ParameterSpec]:
        specs = [
            ParameterSpec("mode", "Mode", kind="choice",
                          choices=["Fit", "KK test", "Batch KK test", "Simulate"], default="KK test",
                          tooltip="Fit: KK-validate + fit loaded spectra. "
                                  "KK test: KK validation only (fast quality check). "
                                  "Batch KK test: validate each file separately and compare HFR. "
                                  "Simulate: forward model from the values below.",
                          group="Mode"),
            ParameterSpec("model_family", "Model family", kind="choice",
                          choices=["Non-faradaic", "Faradaic"], default="Non-faradaic",
                          tooltip="Non-faradaic: blocking / diffusion porous-electrode "
                                  "models (no charge transfer). Faradaic: transmission "
                                  "line with one or more R∥CPE charge-transfer units.",
                          group="Mode", show_if=("mode", ["Fit", "Simulate"])),
            ParameterSpec("model_nf", "Model", kind="choice", choices=_MODELS_NF,
                          default="Transmission Line", group="Mode",
                          show_if=[("mode", ["Fit", "Simulate"]),
                                   ("model_family", "Non-faradaic")]),
            ParameterSpec("model_f", "Model", kind="choice", choices=_MODELS_F,
                          default=_MODELS_F[0], group="Mode",
                          show_if=[("mode", ["Fit", "Simulate"]),
                                   ("model_family", "Faradaic")]),
            ParameterSpec("n_rc", "RC units", kind="int", default=1,
                          minimum=1, maximum=_MAX_RC, step=1,
                          tooltip="Number of parallel pore-wall R∥CPE units. Unit 1 uses "
                                  "R_k and Q_dl. With multiple units, resistance enters "
                                  "only as 1/R_k + Σ 1/R_i; individual resistances "
                                  "cannot be determined separately from the spectrum.",
                          group="Mode",
                          show_if=[("mode", ["Fit", "Simulate"]),
                                   ("model_family", "Faradaic")]),
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
            ParameterSpec("freq_reset", "Frequency range", kind="reset", default="Reset",
                          reset_keys=["f_min", "f_max"],
                          reset_method="input_frequency_bounds",
                          tooltip="Set f_min and f_max to the actual bounds of the "
                                  "selected input spectra.",
                          group="Frequency window", show_if=("mode", ["KK test", "Batch KK test"])),

            ParameterSpec("kk_threshold_pct", "KK threshold", kind="float", default=10.0,
                          minimum=0.1, maximum=100.0, step=1.0, decimals=1, suffix="%",
                          tooltip="Per-point Kramers-Kronig residual cut-off (% of |Z|). "
                                  "Points above this relative complex-residual magnitude are "
                                  "excluded from fitting.",
                          group="KK test", show_if=("mode", ["Fit", "KK test", "Batch KK test"])),

            ParameterSpec("hfr", "HFR", kind="float", default=0.0, minimum=0.0,
                          step=0.001, decimals=5, suffix="Ω·cm²",
                          tooltip="KK test uses the high-frequency x-axis intersection, "
                                  "or minimum real impedance if none is present. "
                                  "0 = auto-seed from the highest-frequency real impedance "
                                  "(Fit mode).",
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
            ParameterSpec("rk", "R_k (unit 1)", kind="float", default=1.0, minimum=0.0,
                          step=0.1, decimals=5, suffix="Ω·cm²",
                          tooltip="Charge-transfer resistance of the distributed "
                                  "transmission-line rung (RC unit 1). Faradaic only.",
                          group="Parameters (init / simulate)",
                          show_if=[("mode", ["Fit", "Simulate"]),
                                   ("model_family", "Faradaic")]),
            ParameterSpec("lwire", "L_wire", kind="float", default=2e-5, minimum=0.0,
                          step=1e-6, decimals=9, suffix="H·cm²",
                          group="Parameters (init / simulate)",
                          show_if=("mode", ["Fit", "Simulate"])),
            ParameterSpec("theta", "Theta", kind="float", default=0.95, minimum=0.0,
                          maximum=1.0, step=0.01, decimals=3,
                          group="Parameters (init / simulate)",
                          show_if=("mode", ["Fit", "Simulate"])),
        ]

        # Extra pore-wall R∥CPE units appear when selected by n_rc.
        for i in range(2, _MAX_RC + 1):
            gate = [("mode", ["Fit", "Simulate"]), ("model_family", "Faradaic"),
                    ("n_rc", list(range(i, _MAX_RC + 1)))]
            grp = f"RC unit {i} (init / simulate)"
            specs += [
                ParameterSpec(f"r{i}", f"R_{i}", kind="float", default=0.5, minimum=0.0,
                              step=0.1, decimals=5, suffix="Ω·cm²", group=grp, show_if=gate),
                ParameterSpec(f"q{i}", f"Q_{i}", kind="float", default=0.1, minimum=0.0,
                              step=0.01, decimals=6, suffix="S·sᵠ", group=grp, show_if=gate),
                ParameterSpec(f"phi{i}", f"Phi_{i}", kind="float", default=0.9, minimum=0.0,
                              maximum=1.0, step=0.01, decimals=3, group=grp, show_if=gate),
            ]

        specs += [
            ParameterSpec("restarts", "Fit restarts", kind="int", default=6,
                          minimum=1, maximum=64, step=1, group="Fit options",
                          show_if=("mode", "Fit")),
            ParameterSpec("max_nfev", "Max evaluations", kind="int", default=20000,
                          minimum=100, maximum=200000, step=1000, group="Fit options",
                          show_if=("mode", "Fit")),
            ParameterSpec("weighting", "Weighting", kind="choice",
                          choices=["Modulus (1/|Z|)", "None"], default="Modulus (1/|Z|)",
                          tooltip="Modulus weights each residual by 1/|Z| so small "
                                  "high-frequency features (HFR, R_CL) are not swamped by "
                                  "the large low-frequency arcs — recommended for wide-window "
                                  "multi-RC fits. None = plain |ΔZ| least squares.",
                          group="Fit options", show_if=("mode", "Fit")),

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
        return specs

    @staticmethod
    def _resolve_model(params: dict) -> str:
        """The active model name given the family selector."""
        if params.get("model_family", "Non-faradaic") == "Faradaic":
            return params.get("model_f", _MODELS_F[0])
        return params.get("model_nf", _MODELS_NF[0])

    @staticmethod
    def _file_key(files: list[Path]) -> tuple:
        """Identity of an input file set (resolved path + mtime, sorted) so a
        cached KK result can be matched to exactly these files and invalidated
        when any of them changes on disk."""
        key = []
        for p in files:
            p = Path(p)
            try:
                stat = p.stat()
                key.append((str(p.resolve()), stat.st_mtime_ns, stat.st_size))
            except OSError:
                key.append((str(p), None, None))
        return tuple(sorted(key))

    @staticmethod
    def _range_covers(cached_range, requested_range) -> bool:
        """Whether a cached KK run covers every requested fit frequency."""
        def bounds(value):
            if value is None:
                return -np.inf, np.inf
            low, high = value
            return (-np.inf if low is None else float(low),
                    np.inf if high is None else float(high))

        cached_low, cached_high = bounds(cached_range)
        requested_low, requested_high = bounds(requested_range)
        return cached_low <= requested_low and cached_high >= requested_high

    @staticmethod
    def _lookup_kk_errors(frequencies, kk_by_freq, *, rtol=1e-4, atol=1e-12):
        """Map tolerant averaged-frequency bins onto raw per-frequency KK errors."""
        known_f = kk_by_freq.index.to_numpy(dtype=float)
        known_e = kk_by_freq.to_numpy(dtype=float)
        out = np.full(len(frequencies), np.nan, dtype=float)
        for idx, frequency in enumerate(np.asarray(frequencies, dtype=float)):
            matches = np.isclose(known_f, frequency, rtol=rtol, atol=atol)
            if np.any(matches):
                out[idx] = float(np.mean(known_e[matches]))
        return out

    # ------------------------------------------------------------------ run
    def run(self, files: list[Path], params: dict, extra=None) -> ToolResult:
        mode = params.get("mode", "Fit")
        if mode == "Simulate":
            return self._run_simulate(files, params)
        if mode == "KK test":
            return self._run_kk(files, params)
        if mode == "Batch KK test":
            return self._run_batch_kk(files, params)
        return self._run_fit(files, params)

    def input_frequency_bounds(self, files: list[Path], params=None) -> dict:
        """Return the positive frequency extent across all readable inputs."""
        if not files:
            raise ValueError("Add at least one EIS input file first.")
        data_list, warnings = _read_all(files)
        frequencies = [
            np.asarray(data["frequency"], dtype=float)
            for data in data_list
        ]
        if not frequencies:
            detail = f" {warnings[0]}" if warnings else ""
            raise ValueError(f"No EIS spectra could be read from the selected files.{detail}")
        combined = np.concatenate(frequencies)
        usable = combined[np.isfinite(combined) & (combined > 0)]
        if usable.size == 0:
            raise ValueError("The selected EIS files contain no positive frequencies.")
        return {"f_min": float(usable.min()), "f_max": float(usable.max())}

    # --------------------------------------------------------------- KK test
    def _run_kk(self, files: list[Path], params: dict) -> ToolResult:
        if not files:
            raise ValueError("KK test needs at least one EIS file.")
        data_list, warnings = _read_all(files)
        if not data_list:
            raise ValueError("No EIS spectra could be read from the selected files.")

        logic = EisLogic()
        logic.kk_threshold = float(params.get("kk_threshold_pct", 10.0)) / 100.0
        f_min = float(params.get("f_min", 0.0)) or None
        f_max = float(params.get("f_max", 0.0)) or None
        kk_range = (f_min, f_max) if (f_min or f_max) else None

        df_all, df_avg, suggested, rsd, kk_warn = logic.process_spectra(data_list, kk_range)
        warnings.extend(kk_warn)

        # Cache residuals so changes to the KK threshold require no new KK test.
        self._kk_cache = {
            "key": self._file_key(files),
            "kk_by_freq": df_all.groupby("f")["kk_err"].mean(),
            "range": kk_range,
        }

        n_total = len(df_all)
        n_valid = int((df_all["kk_err"] <= logic.kk_threshold).sum())
        pct_valid = 100.0 * n_valid / n_total if n_total else 0.0
        hfr_est, intersection_found = logic.estimate_hfr(df_avg)
        hfr_method = ("High-frequency x-axis intersection" if intersection_found else
                      "Minimum real impedance (no x-axis intersection in selected window)")

        figures = {"KK diagnostics": _plot_kk(df_all, df_avg, logic.kk_threshold)}
        tables = {"KK diagnostics (raw)": df_all, "Averaged spectrum": df_avg}
        scalars = {
            "Spectra": int(df_all["scan"].nunique()),
            "Points passing KK": f"{n_valid}/{n_total} ({pct_valid:.0f}%)",
            "Drift (RSD)": float(rsd),
            "Suggested f-range (Hz)": f"{suggested[0]:.3g} – {suggested[1]:.3g}",
            "HFR estimate (Ω·cm²)": hfr_est,
            "HFR method": hfr_method,
        }
        quality = ("Good — most points pass KK." if pct_valid >= 90 else
                   "Mixed — inspect the residual plot; consider narrowing f_min/f_max." if pct_valid >= 60 else
                   "Poor — many points fail KK; the spectrum may be noisy or drifting.")
        quality += ("\nThe suggested KK-valid window and HFR estimate have been "
                    "written into the fields — review them, switch Mode to 'Fit', and Run.")
        sug_min, sug_max = suggested
        param_updates = {
            "f_min": round(float(sug_min), 3),
            "f_max": round(float(sug_max), 3),
            "hfr": round(hfr_est, 5),
        }
        return ToolResult(figures=figures, tables=tables, scalars=scalars,
                          text_blocks={"KK quality": quality},
                          param_updates=param_updates, warnings=warnings)

    # --------------------------------------------------------- Batch KK test
    def _run_batch_kk(self, files: list[Path], params: dict) -> ToolResult:
        if not files:
            raise ValueError("Batch KK test needs at least one EIS file.")

        logic = EisLogic()
        logic.kk_threshold = float(params.get("kk_threshold_pct", 10.0)) / 100.0
        f_min = float(params.get("f_min", 0.0)) or None
        f_max = float(params.get("f_max", 0.0)) or None
        kk_range = (f_min, f_max) if (f_min or f_max) else None
        logic._normalize_frequency_range(kk_range)

        rows, raw_spectra, warnings = [], [], []
        diagnostics, spectra, quality_notes = {}, {}, []
        for number, fp in enumerate(files, start=1):
            path = Path(fp)
            label = f"Sample {number} ({path.name})"
            row = {
                "Sample": number, "File": path.name,
                "HFR (Ω·cm²)": np.nan, "HFR method": "",
                "Points": 0, "Points passing KK": 0, "KK pass (%)": np.nan,
                "Suggested f_min (Hz)": np.nan, "Suggested f_max (Hz)": np.nan,
                "Status": "Failed", "Error": "",
            }
            try:
                data, error = read_eis_data(str(path))
                if error:
                    raise ValueError(" ".join(error.splitlines()))
                if data.get("warning"):
                    warnings.append(f"{label}: {data['warning']}")
                df_all, suggested, kk_warn = logic.validate_spectra([data], kk_range)
                warnings.extend(f"{label}: {warning}" for warning in kk_warn)
                # Use this file's measured points directly for HFR and plotting.
                spectrum = df_all[["f", "zr", "zi"]].rename(columns={
                    "f": "Freq(Hz)", "zr": "Z'(Ohm.cm²)", "zi": "Z''(Ohm.cm²)",
                }).sort_values("Freq(Hz)", ascending=False).reset_index(drop=True)
                hfr, intersection_found = logic.estimate_hfr(spectrum)
            except Exception as exc:
                row["Error"] = str(exc)
                warnings.append(f"{label}: {exc}")
                rows.append(row)
                continue

            n_total = len(df_all)
            n_valid = int((df_all["kk_err"] <= logic.kk_threshold).sum())
            pct_valid = 100.0 * n_valid / n_total
            row.update({
                "HFR (Ω·cm²)": hfr,
                "HFR method": ("High-frequency x-axis intersection" if intersection_found else
                               "Minimum real impedance (no x-axis intersection in selected window)"),
                "Points": n_total, "Points passing KK": n_valid, "KK pass (%)": pct_valid,
                "Suggested f_min (Hz)": float(suggested[0]),
                "Suggested f_max (Hz)": float(suggested[1]), "Status": "OK",
            })
            rows.append(row)
            quality = "Good" if pct_valid >= 90 else "Mixed" if pct_valid >= 60 else "Poor"
            quality_notes.append(
                f"{label}: {quality}; {n_valid}/{n_total} points pass KK ({pct_valid:.0f}%). "
                f"HFR = {hfr:.6g} Ω·cm²."
            )
            figure = _plot_kk(df_all, spectrum, logic.kk_threshold, spectrum_label="spectrum")
            figure.suptitle(label)
            figure.tight_layout(rect=(0, 0, 1, 0.92))
            diagnostics[f"KK sample {number}"] = figure
            spectra[f"Spectrum {number}"] = spectrum
            df_all = df_all.copy()
            df_all["scan"] = number
            df_all.insert(0, "File", path.name)
            df_all.insert(0, "Sample", number)
            raw_spectra.append(df_all)

        if not raw_spectra:
            raise ValueError("No EIS spectra could be processed in Batch KK test. "
                             + " ".join(warnings))

        summary = pd.DataFrame(rows)
        figures = {"HFR by sample": _plot_batch_hfr(summary), **diagnostics}
        tables = {"HFR summary": summary,
                  "KK diagnostics (raw)": pd.concat(raw_spectra, ignore_index=True),
                  **spectra}
        return ToolResult(
            figures=figures, tables=tables,
            scalars={"Files": len(files), "Spectra processed": len(raw_spectra),
                     "Failed samples": len(files) - len(raw_spectra),
                     "KK threshold (%)": logic.kk_threshold * 100},
            text_blocks={
                "Batch KK test": "Each file is analyzed separately. Sample numbers follow the "
                                 "input file order; see HFR summary for filenames and HFR methods.",
                "KK quality per sample": "\n".join(quality_notes),
            },
            warnings=warnings,
        )

    # ----------------------------------------------------------------- fit
    def _run_fit(self, files: list[Path], params: dict) -> ToolResult:
        if not files:
            raise ValueError("Fit mode needs at least one EIS file. "
                             "Switch Mode to 'Simulate' to model without data.")
        data_list, warnings = _read_all(files)
        if not data_list:
            raise ValueError("No EIS spectra could be read from the selected files.")

        logic = EisLogic()
        logic.kk_threshold = float(params.get("kk_threshold_pct", 10.0)) / 100.0
        logic.fit_n_restarts = int(params.get("restarts", 8))
        logic.fit_max_nfev = int(params.get("max_nfev", 20000))

        f_min = float(params.get("f_min", 0.0)) or None
        f_max = float(params.get("f_max", 0.0)) or None
        kk_range = (f_min, f_max) if (f_min or f_max) else None

        df_avg, rsd, avg_warn = logic.average_spectra(data_list, kk_range)
        warnings.extend(avg_warn)
        if kk_range is None:
            warnings.append(
                "No frequency window set — KK-validating and fitting the full spectrum."
            )

        # Cached validation must cover the complete requested fit range.
        cache = getattr(self, "_kk_cache", None)
        cache_usable = (
            cache is not None
            and cache["key"] == self._file_key(files)
            and "range" in cache
            and self._range_covers(cache.get("range"), kk_range)
        )
        if not cache_usable:
            df_all, df_avg, _suggested, rsd, kk_warn = logic.process_spectra(
                data_list, kk_range)
            warnings.extend(warning for warning in kk_warn if warning not in warnings)
            cache = {
                "key": self._file_key(files),
                "kk_by_freq": df_all.groupby("f")["kk_err"].mean(),
                "range": kk_range,
            }
            self._kk_cache = cache
            warnings.append(
                "KK validation refreshed for the current fit frequency range."
            )

        kk_errors = self._lookup_kk_errors(
            df_avg["Freq(Hz)"].to_numpy(), cache["kk_by_freq"],
            rtol=logic.frequency_rtol, atol=logic.frequency_atol,
        )
        kk_tested = np.isfinite(kk_errors)
        kk_keep = kk_tested & (kk_errors <= logic.kk_threshold)
        n_untested = int(np.sum(~kk_tested))
        n_kk_drop = int(np.sum(kk_tested & ~kk_keep))
        if n_untested:
            warnings.append(
                f"{n_untested} point(s) had no matching KK result and were excluded from the fit."
            )
        if n_kk_drop:
            warnings.append(
                f"{n_kk_drop} KK-invalid point(s) excluded from the fit "
                f"(residual > {logic.kk_threshold * 100:.0f}%)."
            )
        df_fit = df_avg[kk_keep]

        f_fit = df_fit["Freq(Hz)"].to_numpy()
        z_exp = (df_fit["Z'(Ohm.cm²)"].to_numpy()
                 + 1j * df_fit["Z''(Ohm.cm²)"].to_numpy())
        if len(f_fit) < 4:
            raise ValueError(
                f"Only {len(f_fit)} point(s) in the fit window — widen "
                "f_min/f_max or run KK test to set a valid window."
            )

        model = self._resolve_model(params)
        fit_inductance = bool(params.get("fit_inductance", False))
        fit_faradaic = (model == EisLogic.FARADAIC_TML_MODEL)
        n_rc = int(params.get("n_rc", 1)) if fit_faradaic else 1

        hfr = float(params.get("hfr", 0.0))
        if hfr <= 0:
            # Auto-seed from the high-frequency intercept (Z′ at the highest f).
            hfr = max(float(np.real(z_exp)[int(np.argmax(f_fit))]), 1e-4)
        rcl = float(params["rcl"]); qdl = float(params["qdl"]); phi = float(params["phi"])
        rk = float(params["rk"]); lwire = float(params["lwire"]); theta = float(params["theta"])

        # Seed default-valued extra units from −Z″ peaks; retain edited values.
        seeds = _seed_extra_arcs(f_fit, z_exp, n_rc) if (fit_faradaic and n_rc >= 2) else {}
        extras = []
        for i in range(2, n_rc + 1):
            sr, sq, sphi = seeds.get(i, (0.5, 0.1, 0.9))
            r_i = float(params.get(f"r{i}", 0.5))
            q_i = float(params.get(f"q{i}", 0.1))
            phi_i = float(params.get(f"phi{i}", sphi))
            extras += [sr if r_i == 0.5 else r_i,
                       sq if q_i == 0.1 else q_i,
                       phi_i]

        if fit_inductance and fit_faradaic:
            init = [lwire, hfr, rcl, qdl, phi, theta, rk] + extras
        elif fit_inductance:
            init = [lwire, hfr, rcl, qdl, phi, theta]
        elif fit_faradaic:
            init = [hfr, rcl, qdl, phi, rk] + extras
        else:
            init = [hfr, rcl, qdl, phi]

        weighting = "modulus" if str(params.get("weighting", "")).startswith("Modulus") else "none"

        out, err = logic.fit_impedance(model, init, f_fit, z_exp,
                                       fit_inductance=fit_inductance, weighting=weighting)
        figures = {}
        tables = {"Averaged spectrum": df_avg}
        scalars = {"Drift (RSD)": float(rsd),
                   "Fit points": int(len(f_fit))}

        if err:
            warnings.append(f"Fit failed: {err}")
            return ToolResult(figures=figures, tables=tables, scalars=scalars,
                              warnings=warnings)

        p, se, _resid, info = out
        z_model = logic.evaluate_model(model, p, f_fit)

        figures["Fit results"] = _plot_panels(f_fit, z_exp, z_model, "Fit", _FIT,
                                              param_box=_param_box(p, model, fit_inductance,
                                                                   fit_faradaic, n_rc))
        tables["Fitted curve"] = pd.DataFrame({
            "Frequency (Hz)": f_fit,
            "Data Re(Z)": np.real(z_exp), "Data Im(Z)": np.imag(z_exp),
            "Fit Re(Z)": np.real(z_model), "Fit Im(Z)": np.imag(z_model),
            "Fit |Z|": np.abs(z_model),
        })
        tables["Fit parameters"] = _param_table(p, se, model, fit_inductance, fit_faradaic, n_rc)

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
            for i in range(2, n_rc + 1):
                scalars[f"R_{i} (Ω·cm²)"] = float(p[7 + (i - 2) * 3])
        if fit_inductance:
            scalars["L_wire (H·cm²)"] = float(p[0])

        flagged = _flag_high_se(p, se, model, fit_inductance, fit_faradaic,
                                logic.fit_high_uncertainty_pct, n_rc)
        text = (f"Best of {info['n_restarts']} restarts (#{info['best_attempt']}, "
                f"by {'HFR/R_CL accuracy' if info['ranking_basis'] == 'hfr_rcl_se' else 'lowest SSR'})\n"
                f"max rel. SE on HFR/R_CL = {info['score_hfr_rcl_pct']:.1f}%  ·  "
                f"SSR = {info['ssr']:.3g}")
        if flagged:
            text += "\nHigh uncertainty: " + ", ".join(flagged)

        # Update fitted fields, omitting canonical 0/∞ placeholders.
        keys_idx = {"hfr": 1, "rcl": 2, "qdl": 3, "phi": 4}
        if fit_inductance:
            keys_idx["lwire"] = 0
            keys_idx["theta"] = 5
        if fit_faradaic:
            keys_idx["rk"] = 6
            for i in range(2, n_rc + 1):
                base = 7 + (i - 2) * 3
                keys_idx[f"r{i}"] = base
                keys_idx[f"q{i}"] = base + 1
                keys_idx[f"phi{i}"] = base + 2
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
        model = self._resolve_model(params)
        canon = [float(params["lwire"]), float(params["hfr"]), float(params["rcl"]),
                 float(params["qdl"]), float(params["phi"]), float(params["theta"]),
                 float(params["rk"])]
        if not params.get("fit_inductance", False):
            canon[0] = 0.0  # L_wire off
        if model != EisLogic.FARADAIC_TML_MODEL:
            canon[6] = np.inf  # R_k unused
        else:
            for i in range(2, int(params.get("n_rc", 1)) + 1):
                canon += [float(params.get(f"r{i}", 0.5)),
                          float(params.get(f"q{i}", 0.1)),
                          float(params.get(f"phi{i}", 0.9))]

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
            if data.get("warning"):
                warnings.append(f"{Path(fp).name}: {data['warning']}")
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


def _seed_extra_arcs(f, z, n_rc):
    """Estimate extra pore-wall R∥CPE parameters from −Z″ peaks.

    Semicircle estimates seed R ≈ 2× peak height and Q ≈ 1/(R·ω_p^φ).
    Remaining units use logarithmically spaced low frequencies. Return
    ``{unit_index: (R_i, Q_i, Phi_i)}``, or defaults for unusable inputs.
    """
    n_extra = max(0, int(n_rc) - 1)
    if n_extra == 0:
        return {}
    try:
        f = np.asarray(f, float)
        zim = -np.imag(np.asarray(z))
        order = np.argsort(f)
        f, zim = f[order], zim[order]

        # Local maxima of −Z″, strongest first.
        peaks = [(zim[k], f[k]) for k in range(1, len(zim) - 1)
                 if zim[k] > zim[k - 1] and zim[k] >= zim[k + 1] and zim[k] > 0]
        peaks.sort(reverse=True)

        phi0 = 0.9
        span = float(np.ptp(np.real(z)))          # Re(Z) span ~ total arc width
        seeds = {}
        for j in range(n_extra):
            if j < len(peaks):
                height, fp = peaks[j]
                r_i = max(2.0 * float(height), 1e-6)
            else:                                  # no distinct peak left
                fp = float(f[min(len(f) - 1, len(f) // (j + 2))])
                r_i = max(span / (n_extra + 1), 1e-6)
            wp = 2.0 * np.pi * max(float(fp), 1e-9)
            q_i = float(np.clip(1.0 / (r_i * wp ** phi0), 1e-9, 1e3))
            seeds[j + 2] = (r_i, q_i, phi0)
        return seeds
    except Exception:
        return {}


def _fmt_pct(pct: float) -> str:
    """Compact relative-SE percentage. A poorly constrained parameter (e.g.
    L_wire) can yield a percentage with many digits; switch to approximate
    scientific notation past 1e4% so the note stays short."""
    if not np.isfinite(pct):
        return "∞%"
    if pct >= 1e4:
        return f"~{pct:.1e}%"
    return f"{pct:.1f}%"


def _fmt_se(val: float, err: float) -> str:
    """Compact '± SE (rel%)' note for a fitted parameter; '± —' if undefined.

    The SD is shown in scientific notation with 3 significant figures so the
    note stays short and fits the parameter-panel annotation label (a raw
    fixed-point SD can run long and get clipped)."""
    if not np.isfinite(err):
        return "± —"
    note = f"± {err:.2e}"
    if val != 0 and np.isfinite(abs(err) / abs(val)):
        note += f" ({_fmt_pct(abs(err) / abs(val) * 100)})"
    return note


def _active_params(model, fit_inductance, fit_faradaic, n_rc=1):
    """Active fitted parameters as ``(display_key, canonical_index, unit)`` in
    display order. Extends past the 7-element core with one (R_i, Q_i, Phi_i)
    triple per extra RC unit (i = 2..n_rc) of the faradaic model."""
    items = []
    if fit_inductance:
        items.append(("L_wire", 0, _UNITS[0]))
    items += [("HFR", 1, _UNITS[1]), ("R_CL", 2, _UNITS[2]),
              ("Q_dl", 3, _UNITS[3]), ("Phi", 4, _UNITS[4])]
    if fit_inductance:
        items.append(("Theta", 5, _UNITS[5]))
    if fit_faradaic:
        items.append(("R_k", 6, _UNITS[6]))
        for i in range(2, n_rc + 1):
            base = 7 + (i - 2) * 3
            items += [(f"R_{i}", base, "Ω·cm²"),
                      (f"Q_{i}", base + 1, "S·sᵠ/cm²"),
                      (f"Phi_{i}", base + 2, "—")]
    return items


def _param_box(p, model, fit_inductance, fit_faradaic, n_rc=1) -> str:
    lines = [f"HFR  = {p[1]:.4f} Ω·cm²", f"R_CL = {p[2]:.4f} Ω·cm²",
             f"Q_dl = {p[3]:.3g}", f"φ    = {p[4]:.3f}"]
    if model == "Transmission Line" and not fit_faradaic:
        lines.append(f"R_CL/3 = {p[2] / 3.0:.4f} Ω·cm²")
    if fit_faradaic:
        lines.append(f"R_k  = {p[6]:.4f} Ω·cm²")
        for i in range(2, n_rc + 1):
            lines.append(f"R_{i}  = {p[7 + (i - 2) * 3]:.4f} Ω·cm²")
    if fit_inductance:
        lines.append(f"L_wire = {p[0]:.3e}")
        lines.append(f"Θ    = {p[5]:.3f}")
    return "\n".join(lines)


def _param_table(p, se, model, fit_inductance, fit_faradaic, n_rc=1) -> pd.DataFrame:
    rows = []
    for k, idx, unit in _active_params(model, fit_inductance, fit_faradaic, n_rc):
        val, err = float(p[idx]), float(se[idx])
        pct = (abs(err) / abs(val) * 100) if val != 0 else float("inf")
        rows.append({"Parameter": k, "Value": val, "Std. error": err,
                     "Rel. SE (%)": pct, "Unit": unit})
    if model == "Transmission Line" and not fit_faradaic:
        rows.append({"Parameter": "R_CL/3", "Value": float(p[2]) / 3.0,
                     "Std. error": float(se[2]) / 3.0, "Rel. SE (%)": float("nan"),
                     "Unit": "Ω·cm²"})
    return pd.DataFrame(rows)


def _flag_high_se(p, se, model, fit_inductance, fit_faradaic, threshold, n_rc=1):
    flagged = []
    for k, idx, _unit in _active_params(model, fit_inductance, fit_faradaic, n_rc):
        val = float(p[idx])
        pct = (abs(float(se[idx])) / abs(val) * 100) if val != 0 else float("inf")
        if not np.isfinite(pct) or pct > threshold:
            flagged.append(f"{k} ±{_fmt_pct(pct)}")
    return flagged


def _plot_batch_hfr(summary) -> Figure:
    fig = Figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    samples = summary["Sample"].to_numpy()
    ax.plot(samples, summary["HFR (Ω·cm²)"].to_numpy(), "o-",
            color=_DATA, markersize=6, linewidth=1.4)
    ax.set_xticks(samples)
    ax.set_xlim(0.5, float(samples[-1]) + 0.5)
    ax.set_xlabel("Sample number")
    ax.set_ylabel("HFR (Ω·cm²)")
    ax.set_title("HFR by sample", loc="left")
    ax.grid(axis="y", alpha=0.2)
    for number in summary.loc[summary["Status"] == "Failed", "Sample"]:
        ax.annotate("failed", (number, 0.02), xycoords=("data", "axes fraction"),
                    ha="center", color=_FIT, fontsize=8)
    fig.tight_layout()
    return fig


def _plot_kk(df_all, df_avg, kk_threshold, *, spectrum_label="averaged") -> Figure:
    fig = Figure(figsize=(10, 4.2))
    ax_n = fig.add_subplot(121)
    ax_b = fig.add_subplot(122)

    valid = df_all["kk_err"] <= kk_threshold
    ax_n.scatter(df_all.loc[valid, "zr"], -df_all.loc[valid, "zi"],
                 c=_DATA, alpha=0.45, s=18, edgecolor="none", label="KK valid")
    ax_n.scatter(df_all.loc[~valid, "zr"], -df_all.loc[~valid, "zi"],
                 c=_FIT, marker="x", s=22, linewidth=1.2, label="KK invalid")
    ax_n.plot(df_avg["Z'(Ohm.cm²)"], -df_avg["Z''(Ohm.cm²)"],
              color=theme.TEXT, linewidth=1.6, label=spectrum_label)
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
    ax_b.legend(fontsize=8, loc="upper left")

    # Plain −Z″ vs f on a twin axis to help eyeball the number of arcs (and hence
    # how many RC units to fit) by hand.
    f_avg = df_avg["Freq(Hz)"].to_numpy()
    zi_avg = -df_avg["Z''(Ohm.cm²)"].to_numpy()
    ax_z = ax_b.twinx()
    ax_z.plot(f_avg, zi_avg, color=_DATA, linewidth=1.4, alpha=0.8, label="−Z″")
    ax_z.set_ylabel("−Z″  (Ω·cm²)")
    ax_z.legend(fontsize=8, loc="upper right")
    ax_b.set_title("Point-wise KK error", loc="left")
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
