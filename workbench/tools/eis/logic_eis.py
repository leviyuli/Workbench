"""EIS preprocessing (Kramers-Kronig via pyimpspec), transmission-line models
and multi-start fitting.
"""
import io
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import pyimpspec
import scipy.optimize

class EisLogic:
    FARADAIC_TML_MODEL = "Faradaic Transmission Line (CPE)"
    CANONICAL_KEYS = ["Lwire", "HFR", "Rcl", "Qdl", "Phi", "Theta", "Rk"]

    def __init__(self):
        self.kk_threshold = 0.1
        self.outlier_z_threshold = 1.5
        # Nominally identical frequencies from separate instrument scans often
        # differ slightly after instrument rounding. Treat values within 0.01% as one
        # frequency bin instead of reporting them as unrelated zero-variance
        # points.
        self.frequency_rtol = 1e-4
        self.frequency_atol = 1e-12
        self.min_kk_points = 3
        self.fit_max_nfev = 20000
        self.fit_n_restarts = 6
        self.fit_ssr_tolerance = 2.0
        self.fit_high_uncertainty_pct = 50.0
        # Parallelism policy for the restarts. None -> "auto": run serially in the
        # calling thread for the common (fast) case, and only spread over CPU
        # cores for genuinely heavy jobs (many parameters × many restarts).
        # Spawning worker processes on Windows re-imports scipy/impedance per
        # worker and easily dwarfs a small fit, so serial is faster there.
        # Set to an int to force that many workers (1 -> always serial).
        self.fit_n_jobs = None

    # --- Preprocessing & KK ---
    @staticmethod
    def estimate_hfr(df_avg):
        """Return (HFR, intersection_found) from a windowed averaged spectrum.

        Follow adjacent points from high to low frequency and interpolate Z'
        at the first Z'' = 0 intersection. Exact on-axis points also count.
        With no intersection, retain the minimum-Z' estimate without extrapolation.
        """
        ordered = df_avg.sort_values('Freq(Hz)', ascending=False)
        zr = ordered["Z'(Ohm.cm²)"].to_numpy(dtype=float)
        zi = ordered["Z''(Ohm.cm²)"].to_numpy(dtype=float)
        for i in range(len(zi)):
            if zi[i] == 0.0:
                return float(zr[i]), True
            if i + 1 < len(zi) and zi[i + 1] != 0.0 \
                    and np.signbit(zi[i]) != np.signbit(zi[i + 1]):
                fraction = -zi[i] / (zi[i + 1] - zi[i])
                return float(zr[i] + fraction * (zr[i + 1] - zr[i])), True
        return float(ordered["Z'(Ohm.cm²)"].min()), False

    def _window_scans(self, data_list, freq_range):
        """Trim each scan to the [f_min, f_max] window and drop scans left with
        too few in-scope points. Returns ``(scans, warnings)`` where each scan is
        a dict ``{'i', 'freq', 'zr', 'zi', 'zc'}`` (1-based index + windowed
        arrays). Shared by the KK test (``process_spectra``) and the fit-path
        averaging (``average_spectra``) so both window the data identically."""
        f_min, f_max = self._normalize_frequency_range(freq_range)
        use_range = f_min is not None or f_max is not None
        scans = []
        warnings = []
        for i, data in enumerate(data_list):
            freq = np.asarray(data['frequency'])
            z_real = np.asarray(data['z_real'])
            z_imag = np.asarray(data['z_imag'])
            z_complex = np.asarray(data['z_complex'])

            if not (len(freq) == len(z_real) == len(z_imag) == len(z_complex)):
                warnings.append(f"Scan {i + 1}: skipped; impedance columns have different lengths.")
                continue
            usable = (
                np.isfinite(freq) & (freq > 0)
                & np.isfinite(z_real) & np.isfinite(z_imag) & np.isfinite(z_complex)
            )
            dropped = int(np.sum(~usable))
            if dropped:
                warnings.append(
                    f"Scan {i + 1}: dropped {dropped} non-finite or non-positive-frequency point(s)."
                )
                freq = freq[usable]
                z_real = z_real[usable]
                z_imag = z_imag[usable]
                z_complex = z_complex[usable]

            if use_range:
                in_scope = np.ones(len(freq), dtype=bool)
                if f_min is not None:
                    in_scope &= freq >= f_min
                if f_max is not None:
                    in_scope &= freq <= f_max
                freq = freq[in_scope]
                z_real = z_real[in_scope]
                z_imag = z_imag[in_scope]
                z_complex = z_complex[in_scope]

            if len(freq) < self.min_kk_points:
                warnings.append(
                    f"Scan {i + 1}: skipped; {len(freq)} in-scope point(s), "
                    f"need at least {self.min_kk_points}."
                )
                continue

            scans.append({'i': i + 1, 'freq': freq, 'zr': z_real,
                          'zi': z_imag, 'zc': z_complex})
        return scans, warnings

    def _require_scans(self, scans, warnings, freq_range,
                       what="processed for KK validation"):
        """Raise a clear error (mentioning the window when one is set) if no scan
        survived windowing/validation."""
        if scans:
            return
        f_min, f_max = self._normalize_frequency_range(freq_range)
        detail = " ".join(warnings)
        if f_min is not None or f_max is not None:
            raise ValueError(
                f"No spectra could be {what} inside the selected "
                f"frequency range. {detail}".strip()
            )
        raise ValueError(f"No spectra could be {what}. {detail}".strip())

    def _frequency_groups(self, frequencies):
        """Return tolerant frequency-bin ids for an array of positive values."""
        values = np.asarray(frequencies, dtype=float)
        group_ids = np.empty(len(values), dtype=int)
        if not len(values):
            return group_ids
        representatives: list[float] = []
        counts: list[int] = []
        for idx in np.argsort(values):
            value = float(values[idx])
            if representatives and np.isclose(
                    value, representatives[-1], rtol=self.frequency_rtol,
                    atol=self.frequency_atol):
                gid = len(representatives) - 1
                counts[gid] += 1
                representatives[gid] += (value - representatives[gid]) / counts[gid]
            else:
                representatives.append(value)
                counts.append(1)
                gid = len(representatives) - 1
            group_ids[idx] = gid
        return group_ids

    def _average_points(self, df_pts):
        """Average impedance across scans by frequency, removing cross-scan
        outliers (modified z-score) when ≥3 scans are present. Returns ``(df_avg, rsd)``.
        ``df_pts`` must have columns ``f``, ``zr``, ``zi``, ``scan``."""
        df_pts = df_pts.copy()
        df_pts['_f_group'] = self._frequency_groups(df_pts['f'].to_numpy())
        if df_pts['scan'].nunique() >= 3:
            def calc_z(x):
                median = float(x.median())
                deviation = np.abs(x.to_numpy(dtype=float) - median)
                mad = float(np.median(deviation))
                if mad == 0:
                    scores = np.where(deviation == 0, 0.0, np.inf)
                else:
                    scores = 0.67448975 * deviation / mad
                return pd.Series(scores, index=x.index)
            z_zr = df_pts.groupby('_f_group')['zr'].transform(calc_z)
            z_zi = df_pts.groupby('_f_group')['zi'].transform(calc_z)
            df_clean = df_pts[(z_zr <= self.outlier_z_threshold) & (z_zi <= self.outlier_z_threshold)]
        else:
            df_clean = df_pts.copy()

        grouped = df_clean.groupby('_f_group', sort=False)
        f_avg = grouped['f'].mean().values
        zr_avg = grouped['zr'].mean().values
        zi_avg = grouped['zi'].mean().values
        zr_std = grouped['zr'].std().fillna(0).values
        zi_std = grouped['zi'].std().fillna(0).values
        scan_counts = grouped['scan'].nunique().to_numpy()

        mag = np.sqrt(zr_avg ** 2 + zi_avg ** 2)
        std_total = np.sqrt(zr_std ** 2 + zi_std ** 2)
        if df_pts['scan'].nunique() < 2:
            rsd = 0.0
        else:
            repeated = (scan_counts >= 2) & np.isfinite(mag) & (mag > 0)
            rsd = float(np.mean(std_total[repeated] / mag[repeated])) \
                if np.any(repeated) else float('nan')

        df_avg = pd.DataFrame({
            'Freq(Hz)': f_avg,
            "Z'(Ohm.cm²)": zr_avg,
            "Z''(Ohm.cm²)": zi_avg,
            "Z'_std": zr_std,
            "Z''_std": zi_std,
        })
        return df_avg, rsd

    def average_spectra(self, data_list, freq_range=None):
        """Window and average spectra into ``df_avg`` without running the KK test.

        Returns ``(df_avg, rsd, warnings)``. The fit path can combine this with a
        covering cached KK result; it refreshes KK validation when that cache is
        absent, stale, or narrower than the requested fit range.
        """
        scans, warnings = self._window_scans(data_list, freq_range)
        self._require_scans(scans, warnings, freq_range, what="averaged")
        df_pts = pd.concat(
            [pd.DataFrame({'f': s['freq'], 'zr': s['zr'],
                           'zi': s['zi'], 'scan': s['i']}) for s in scans],
            ignore_index=True,
        )
        df_avg, rsd = self._average_points(df_pts)
        return df_avg, rsd, warnings

    def process_spectra(self, data_list, freq_range=None):
        f_min, f_max = self._normalize_frequency_range(freq_range)
        scans, warnings = self._window_scans(data_list, freq_range)

        all_scans = []
        for s in scans:
            try:
                with redirect_stdout(io.StringIO()):
                    ds = pyimpspec.DataSet(s['freq'], s['zc'])
                    kk_result = pyimpspec.perform_kramers_kronig_test(
                        ds, test='complex', num_procs=1,
                    )
                f_res, res_real, res_imag = kk_result.get_residuals_data()
                res_real = np.asarray(res_real, dtype=float) / 100.0
                res_imag = np.asarray(res_imag, dtype=float) / 100.0
                f_res = np.asarray(f_res, dtype=float)
                z_res = np.asarray(ds.get_impedances(), dtype=complex)
                point_errors = np.sqrt(res_real ** 2 + res_imag ** 2)
                all_scans.append(pd.DataFrame({
                    'f': f_res, 'zr': np.real(z_res), 'zi': np.imag(z_res),
                    'scan': s['i'], 'kk_err': point_errors,
                    'in_scope': True,
                    'range_f_min': f_min if f_min is not None else np.nan,
                    'range_f_max': f_max if f_max is not None else np.nan,
                }))
            except Exception as e:
                warnings.append(f"Scan {s['i']}: KK validation failed ({e}).")

        self._require_scans(all_scans, warnings, freq_range)
        df_all = pd.concat(all_scans, ignore_index=True)

        valid_mask = df_all['kk_err'] <= self.kk_threshold
        valid_freqs = df_all.loc[valid_mask, 'f']
        freq_max_suggested = valid_freqs.max() if not valid_freqs.empty else df_all['f'].max()
        freq_min_suggested = valid_freqs.min() if not valid_freqs.empty else df_all['f'].min()

        df_avg, rsd = self._average_points(df_all[['f', 'zr', 'zi', 'scan']])
        return df_all, df_avg, (freq_min_suggested, freq_max_suggested), rsd, warnings

    def _normalize_frequency_range(self, freq_range):
        if freq_range is None:
            return None, None
        f_min, f_max = freq_range
        if f_min is not None:
            f_min = float(f_min)
            if not np.isfinite(f_min):
                raise ValueError("f_min must be a finite number.")
        if f_max is not None:
            f_max = float(f_max)
            if not np.isfinite(f_max):
                raise ValueError("f_max must be a finite number.")
        if f_min is not None and f_max is not None and f_min > f_max:
            raise ValueError("f_min must be less than or equal to f_max.")
        return f_min, f_max

    # --- Fitting Models ---
    def JPcoth(self, x):
        # coth(x) = 1/tanh(x). np.tanh saturates to a finite ±1 for large |x|,
        # whereas the explicit (e^x+e^-x)/(e^x-e^-x) form overflows to inf/inf =
        # nan at high ω (large complex x) — which wasted optimizer steps. The
        # errstate silences the benign "overflow in tanh" as it saturates.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            t = np.tanh(x)
            coth = 1.0 / t
        return np.where(t == 0, np.inf, coth)

    def JPtanh(self, x):
        with np.errstate(over="ignore", invalid="ignore"):
            return np.tanh(x)

    def is_faradaic_model(self, model_name):
        return model_name == self.FARADAIC_TML_MODEL

    def evaluate_model(self, model_name, params, freq):
        # Canonical core is 7 long: [L_wire, HFR, R_CL, Q_dl, Phi, Theta, R_k].
        # Anything beyond it is a flat list of extra (R_i, Q_i, Phi_i) triples —
        # additional Faradaic R∥CPE branches placed in parallel with unit 1
        # inside Z_k (all at the pore wall). Only the faradaic model consumes
        # these extras.
        params = list(params)
        if len(params) == 6:
            params = params + [np.inf]
        Lwire, HFR, Rcl, Qdl, Phi, Theta, Rk = params[:7]
        extras = params[7:]
        omega = 1j * 2 * np.pi * freq
        Z_L = Lwire * (omega ** Theta) if Lwire != 0 else 0
        cpe_admittance = Qdl * (omega ** Phi)

        if model_name == "Transmission Line":
            x = np.sqrt(Rcl * cpe_admittance)
            prefactor = np.sqrt(Rcl / cpe_admittance)
            Z = Z_L + HFR + prefactor * self.JPcoth(x)
        elif self.is_faradaic_model(model_name):
            # All Faradaic RC units (1..N) are in parallel at the pore
            # wall, inside Z_k.  Unit 1 uses the shared double-layer CPE;
            # units 2..N each contribute their own R_i ∥ Q_i branch.
            admittance = cpe_admittance + (1.0 / Rk)
            for j in range(0, len(extras) - 2, 3):
                R_i, Q_i, Phi_i = extras[j], extras[j + 1], extras[j + 2]
                admittance += (1.0 / R_i) + Q_i * (omega ** Phi_i)
            Zk = 1.0 / admittance
            Z = Z_L + HFR + np.sqrt(Rcl * Zk) * self.JPcoth(np.sqrt(Rcl / Zk))
        elif model_name == "1-D Linear Diffusion":
            x = np.sqrt(Rcl * cpe_admittance)
            prefactor = np.sqrt(Rcl / cpe_admittance)
            Z = Z_L + HFR + prefactor * self.JPtanh(x)
        elif model_name == "1-D Spherical Diffusion":
            x = np.sqrt(Rcl * cpe_admittance)
            Z = Z_L + HFR + Rcl / (x * self.JPcoth(x) - 1)
        else:
            raise ValueError(f"Unknown impedance model: {model_name}")
        return Z

    def _perturb_init_params(self, init_params, lower_bounds, upper_bounds, rng,
                             fit_inductance=False, fit_faradaic=False):
        def log_jitter(value, decades=1.0):
            base = max(abs(value), 1e-9)
            factor = 10.0 ** rng.uniform(-decades, decades)
            return base * factor

        def perturb_extras(extras):
            # Each extra RC unit is a (R_i, Q_i, Phi_i) triple.
            out = []
            for j in range(0, len(extras) - 2, 3):
                out.append(float(np.clip(log_jitter(extras[j], decades=1.0), 1e-12, 1e12)))
                out.append(float(np.clip(log_jitter(extras[j + 1], decades=1.0), 1e-9, 1e3)))
                out.append(float(rng.uniform(0.5, 0.95)))
            return out

        if fit_inductance and fit_faradaic:
            lwire_init, hfr_init, rcl_init, qdl_init, phi_init, theta_init, rk_init = init_params[:7]
            lwire_lo, hfr_lo = lower_bounds[0], lower_bounds[1]
            lwire_hi, hfr_hi = upper_bounds[0], upper_bounds[1]
            lwire = float(np.clip(log_jitter(lwire_init, decades=1.0), 1e-9, lwire_hi))
            hfr = float(rng.uniform(hfr_lo, hfr_hi))
            rcl = float(np.clip(log_jitter(rcl_init, decades=1.0), 1e-6, 1e6))
            qdl = float(np.clip(log_jitter(qdl_init, decades=1.0), 1e-9, 1e3))
            phi = float(rng.uniform(0.5, 0.95))
            theta = float(rng.uniform(0.6, 1.0))
            rk = float(np.clip(log_jitter(rk_init, decades=1.0), 1e-12, 1e12))
            return [lwire, hfr, rcl, qdl, phi, theta, rk] + perturb_extras(init_params[7:])

        if fit_inductance:
            lwire_init, hfr_init, rcl_init, qdl_init, phi_init, theta_init = init_params
            lwire_lo, hfr_lo = lower_bounds[0], lower_bounds[1]
            lwire_hi, hfr_hi = upper_bounds[0], upper_bounds[1]
            lwire = float(np.clip(log_jitter(lwire_init, decades=1.0), 1e-9, lwire_hi))
            hfr = float(rng.uniform(hfr_lo, hfr_hi))
            rcl = float(np.clip(log_jitter(rcl_init, decades=1.0), 1e-6, 1e6))
            qdl = float(np.clip(log_jitter(qdl_init, decades=1.0), 1e-9, 1e3))
            phi = float(rng.uniform(0.5, 0.95))
            theta = float(rng.uniform(0.6, 1.0))
            return [lwire, hfr, rcl, qdl, phi, theta]

        if fit_faradaic:
            hfr_init, rcl_init, qdl_init, phi_init, rk_init = init_params[:5]
            hfr_lo, hfr_hi = lower_bounds[0], upper_bounds[0]
            hfr = float(rng.uniform(hfr_lo, hfr_hi))
            rcl = float(np.clip(log_jitter(rcl_init, decades=1.0), 1e-6, 1e6))
            qdl = float(np.clip(log_jitter(qdl_init, decades=1.0), 1e-9, 1e3))
            phi = float(rng.uniform(0.5, 0.95))
            rk = float(np.clip(log_jitter(rk_init, decades=1.0), 1e-12, 1e12))
            return [hfr, rcl, qdl, phi, rk] + perturb_extras(init_params[5:])

        hfr_init, rcl_init, qdl_init, phi_init = init_params
        hfr_lo, hfr_hi = lower_bounds[0], upper_bounds[0]
        hfr = float(rng.uniform(hfr_lo, hfr_hi))
        rcl = float(np.clip(log_jitter(rcl_init, decades=1.0), 1e-6, 1e6))
        qdl = float(np.clip(log_jitter(qdl_init, decades=1.0), 1e-9, 1e3))
        phi = float(rng.uniform(0.5, 0.95))
        return [hfr, rcl, qdl, phi]

    def fit_impedance(self, model_name, init_params, freq, z_exp, max_nfev=None,
                      n_restarts=None, random_seed=0, fit_inductance=False,
                      weighting="modulus"):
        # ``weighting``: "modulus" weights each residual by 1/|Z| (so small
        # high-frequency features such as HFR & R_CL are not drowned by the large
        # low-frequency arcs); "none" is plain |ΔZ| least squares.
        if max_nfev is None:
            max_nfev = self.fit_max_nfev
        max_nfev = int(max_nfev)
        if max_nfev <= 0:
            raise ValueError("Maximum function evaluations must be a positive integer.")

        if n_restarts is None:
            n_restarts = self.fit_n_restarts
        n_restarts = max(1, int(n_restarts))

        fit_faradaic = self.is_faradaic_model(model_name)

        if fit_inductance and fit_faradaic:
            if len(init_params) < 7 or (len(init_params) - 7) % 3 != 0:
                raise ValueError(
                    "init_params for Faradaic TML with inductance must be "
                    "[L_wire, HFR, R_CL, Q_dl, Phi, Theta, R_k] plus 3 elements "
                    "(R_i, Q_i, Phi_i) per extra RC unit."
                )
            hfr_init = init_params[1]
            lower_bounds = [0.0, 0.8 * hfr_init, 1e-12, 1e-12, 0.0, 0.0, 1e-12]
            upper_bounds = [1.0, 1.2 * hfr_init, np.inf, np.inf, 1.0, 1.0, np.inf]
            for _ in range((len(init_params) - 7) // 3):
                lower_bounds += [1e-12, 1e-12, 0.0]
                upper_bounds += [np.inf, np.inf, 1.0]
        elif fit_inductance:
            if len(init_params) != 6:
                raise ValueError(
                    "init_params must have 6 elements when fit_inductance=True: "
                    "[L_wire, HFR, R_CL, Q_dl, Phi, Theta]."
                )
            hfr_init = init_params[1]
            lower_bounds = [0.0, 0.8 * hfr_init, 1e-12, 1e-12, 0.0, 0.0]
            upper_bounds = [1.0, 1.2 * hfr_init, np.inf, np.inf, 1.0, 1.0]
        elif fit_faradaic:
            if len(init_params) < 5 or (len(init_params) - 5) % 3 != 0:
                raise ValueError(
                    "init_params for Faradaic TML must be [HFR, R_CL, Q_dl, Phi, "
                    "R_k] plus 3 elements (R_i, Q_i, Phi_i) per extra RC unit."
                )
            hfr_init = init_params[0]
            lower_bounds = [0.9 * hfr_init, 1e-12, 1e-12, 0.0, 1e-12]
            upper_bounds = [1.1 * hfr_init, np.inf, np.inf, 1.0, np.inf]
            for _ in range((len(init_params) - 5) // 3):
                lower_bounds += [1e-12, 1e-12, 0.0]
                upper_bounds += [np.inf, np.inf, 1.0]
        else:
            if len(init_params) != 4:
                raise ValueError(
                    "init_params must have 4 elements when fit_inductance=False: "
                    "[HFR, R_CL, Q_dl, Phi]."
                )
            hfr_init = init_params[0]
            lower_bounds = [0.9 * hfr_init, 1e-12, 1e-12, 0.0]
            upper_bounds = [1.1 * hfr_init, np.inf, np.inf, 1.0]

        rng = np.random.default_rng(random_seed)
        # Pre-generate every restart's starting point sequentially so the set of
        # attempts — and therefore the chosen best — is identical no matter what
        # order the parallel optimizations finish in.
        trial_inits = [list(init_params)]
        for _ in range(1, n_restarts):
            trial_inits.append(self._perturb_init_params(
                init_params, lower_bounds, upper_bounds, rng,
                fit_inductance=fit_inductance, fit_faradaic=fit_faradaic,
            ))

        tasks = [(model_name, ti, lower_bounds, upper_bounds, freq, z_exp, max_nfev,
                  fit_inductance, fit_faradaic, i, weighting)
                 for i, ti in enumerate(trial_inits)]

        # The restarts are independent and each scipy.least_squares call is
        # GIL-bound, so they *can* be spread over CPU cores with *processes*
        # (threads gave no speedup). But process spawn on Windows re-imports
        # scipy per worker (seconds) and dwarfs a small fit, so only
        # parallelize genuinely heavy jobs; run serially otherwise. Closures
        # aren't picklable, hence the module-level worker.
        if self.fit_n_jobs is not None:
            n_jobs = int(self.fit_n_jobs)
        else:
            n_params = len(trial_inits[0])
            heavy = n_params >= 10 and n_restarts >= 6   # ~n_rc>=3, or n_rc>=2 + L
            n_jobs = (os.cpu_count() or 1) if heavy else 1
        n_workers = max(1, min(n_jobs, n_restarts, os.cpu_count() or 1))
        outcomes = None
        if n_workers > 1:
            try:
                with ProcessPoolExecutor(max_workers=n_workers) as ex:
                    outcomes = list(ex.map(_fit_attempt, tasks))
            except Exception:
                outcomes = None  # any pool failure -> fall back to in-process
        if outcomes is None:
            outcomes = [_fit_attempt(t) for t in tasks]

        attempts = []
        attempt_messages = []
        failure_count = 0
        for ok, payload in outcomes:
            if ok:
                attempts.append(payload)
            else:
                failure_count += 1
                attempt_messages.append(payload)

        if not attempts:
            return None, ("All fitting attempts failed. " + "; ".join(attempt_messages[:4]))

        hfr_idx = 1 if fit_inductance else 0
        rcl_idx = 2 if fit_inductance else 1

        # Number of branch parameters in the canonical *core* (before any extra
        # RC triples). Extras only exist on the faradaic model.
        core_len = 7 if (fit_inductance and fit_faradaic) else \
            6 if fit_inductance else 5 if fit_faradaic else 4
        multi_rc = fit_faradaic and len(init_params) > core_len

        def resistance_score(a):
            p, s = a["params"], a["se"]
            rel_errors = [
                (s[hfr_idx] / abs(p[hfr_idx])) if p[hfr_idx] != 0 else np.inf,
                (s[rcl_idx] / abs(p[rcl_idx])) if p[rcl_idx] != 0 else np.inf,
            ]
            score = max(rel_errors)
            return score if np.isfinite(score) else np.inf

        best_ssr = min(a["ssr"] for a in attempts)
        ssr_gate = best_ssr * float(self.fit_ssr_tolerance)
        eligible = [a for a in attempts if a["ssr"] <= ssr_gate]
        ranking_basis = "hfr_rcl_se"
        if not eligible:
            eligible = attempts
            ranking_basis = "ssr_fallback"

        if multi_rc:
            # All RC branches sit in parallel inside Z_k, so their spectral
            # signatures overlap heavily with the transmission-line shape; a
            # solution that pins HFR/R_CL but botches the arcs is plausible.
            # Rank multi-RC purely by overall SSR so arc quality drives the
            # selection. Modulus weighting still keeps the HF (HFR/R_CL)
            # features emphasized inside that SSR.
            best = min(attempts, key=lambda a: a["ssr"])
            ranking_basis = "ssr_multi_rc"
        else:
            best = min(eligible, key=resistance_score)
        res = best["res"]
        se = best["se"]
        ssr = best["ssr"]

        if fit_inductance and fit_faradaic:
            params_canonical = np.asarray(res.x, dtype=float)
            se_canonical = np.asarray(se, dtype=float)
        elif fit_inductance:
            params_canonical = np.array(
                [res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], res.x[5], np.inf], dtype=float)
            se_canonical = np.array([se[0], se[1], se[2], se[3], se[4], se[5], 0.0], dtype=float)
        elif fit_faradaic:
            params_canonical = np.array(
                [0.0, res.x[0], res.x[1], res.x[2], res.x[3], 0.0, res.x[4]]
                + list(res.x[5:]), dtype=float)
            se_canonical = np.array(
                [0.0, se[0], se[1], se[2], se[3], 0.0, se[4]] + list(se[5:]), dtype=float)
        else:
            params_canonical = np.array(
                [0.0, res.x[0], res.x[1], res.x[2], res.x[3], 0.0, np.inf], dtype=float)
            se_canonical = np.array([0.0, se[0], se[1], se[2], se[3], 0.0, 0.0], dtype=float)

        info = {
            "ssr": ssr, "ssr_best": best_ssr, "best_attempt": best["attempt_idx"] + 1,
            "n_restarts": n_restarts, "n_eligible": len(eligible), "n_failed": failure_count,
            "converged": bool(res.success), "message": res.message, "nfev": int(res.nfev),
            "ranking_basis": ranking_basis, "score_hfr_rcl_pct": resistance_score(best) * 100.0,
            "fit_inductance": bool(fit_inductance), "fit_faradaic": bool(fit_faradaic),
        }
        # res.fun is the residual vector at the solution (== recomputing cost).
        return (params_canonical, se_canonical, np.asarray(res.fun), info), None


def _to_canonical(p, fit_inductance, fit_faradaic):
    """Map a branch-reduced parameter vector to the 7-element canonical core
    [L_wire, HFR, R_CL, Q_dl, Phi, Theta, R_k] plus any extra RC triples. Single
    source of truth, shared by the per-restart fit worker."""
    if fit_inductance and fit_faradaic:
        return list(p)                                   # already canonical core + extras
    if fit_inductance:
        return list(p) + [np.inf]                        # R_k unused
    if fit_faradaic:
        return [0.0, p[0], p[1], p[2], p[3], 0.0, p[4]] + list(p[5:])
    return [0.0, p[0], p[1], p[2], p[3], 0.0, np.inf]


def _fit_attempt(args):
    """Run one restart of the impedance fit. Defined at module level (and thus
    picklable) so the independent restarts can be dispatched across processes.
    Returns ``(True, result_dict)`` on success or ``(False, message)``."""
    (model_name, trial_init, lower_bounds, upper_bounds, freq, z_exp, max_nfev,
     fit_inductance, fit_faradaic, idx, weighting) = args
    logic = EisLogic()

    # Modulus weighting (1/|Z|): each residual becomes a relative error, so the
    # small high-frequency features are not swamped by the large low-frequency
    # arcs. ``weight`` is constant across the fit, so precompute it once.
    weight = (1.0 / np.abs(z_exp)) if weighting == "modulus" else None

    def cost_func(params):
        canon = _to_canonical(params, fit_inductance, fit_faradaic)
        Z_model = logic.evaluate_model(model_name, canon, freq)
        resid_real = np.real(Z_model) - np.real(z_exp)
        resid_imag = np.imag(Z_model) - np.imag(z_exp)
        if weight is not None:
            resid_real = resid_real * weight
            resid_imag = resid_imag * weight
        return np.concatenate((resid_real, resid_imag))

    try:
        res = scipy.optimize.least_squares(
            cost_func, trial_init, bounds=(lower_bounds, upper_bounds),
            method='trf', xtol=1e-8, ftol=1e-8, gtol=1e-8, x_scale='jac',
            max_nfev=max_nfev,
        )
    except Exception as e:
        return False, f"attempt {idx + 1}: exception ({e})"
    ssr = float(res.fun @ res.fun)
    if not np.isfinite(ssr):
        return False, f"attempt {idx + 1}: non-finite residual"
    dof = len(res.fun) - len(res.x)
    if dof > 0:
        cov_matrix = np.linalg.pinv(res.jac.T @ res.jac)
        se = np.sqrt(np.maximum(np.diag(cov_matrix * (ssr / dof)), 0.0))
    else:
        se = np.full(len(res.x), np.nan, dtype=float)
    return True, {"res": res, "ssr": ssr, "se": se,
                  "params": res.x, "attempt_idx": idx}
