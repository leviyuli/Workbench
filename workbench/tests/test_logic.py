"""Logic-only tests (no Qt). Run with: pytest workbench/tests"""
import numpy as np
import pandas as pd

from workbench.tools.ink_calc.logic import calculate_ink_recipe
from workbench.tools.ecsa.logic_ecsa import compute_cdl_ecsa
from workbench.tools.lsv.logic_tafel import iterative_tafel_fit
from workbench.tools.lsv.tool import LsvTool
from workbench.tools.it_integrate.logic import integrate_potentiostatic_tests
from workbench.tools.eis.logic_eis import EisLogic
from workbench.tools.ftir.logic import (
    apply_vertical_offset,
    convert_t_to_a,
    normalize_spectrum,
    read_ftir_file,
)
from workbench.tools.ftir.tool import FtirTool


def test_ink_recipe_area_mode():
    params = {
        "mode": "area", "length": 5.0, "width": 5.0, "loading": 1.0,
        "ink_density": 2.0,
        "ionomers": [{"ratio": 20.0, "conc": 5.0, "density": 0.78}],
        "extras": [],
    }
    res = calculate_ink_recipe(params)
    assert res["catalyst_mg"] == 25.0                 # 1 * 5 * 5
    # m_i = 25 * 0.2 / 0.8 = 6.25
    assert abs(res["total_ionomer_mg"] - 6.25) < 1e-9
    assert res["ink_volume_ml"] == 12.5               # 25 / 2
    assert res["adjusted"] is not None                 # area mode -> loss computed


def test_ecsa_linear_recovery():
    # Construct synthetic CV cycles with a known C_dl.
    Cdl_true = 0.01  # F/cm²
    rows = []
    E = np.linspace(0.0, 0.2, 201)
    for rate_mV in (20, 40, 60, 80):
        v = rate_mV * 1e-3
        jdl = Cdl_true * v
        # anodic +jdl, cathodic -jdl across the window
        i = np.where(E <= 0.1, jdl, -jdl)
        for e, ii in zip(E, i):
            rows.append({"scan_rate": rate_mV, "potential": e, "current": ii})
    long_df = pd.DataFrame(rows)
    out = compute_cdl_ecsa(long_df, Cs_uF_per_cm2=40.0, n_nearest=5)
    assert abs(out["Cdl"] - Cdl_true) < 1e-3
    assert out["r2"] > 0.99
    assert out["ECSA"] is not None


def test_tafel_fit_runs_on_synthetic():
    # Synthetic Tafel: eta = b*log10(i/i0); build E = E_rev + eta
    E_rev, b, i0 = 1.23, 0.06, 1e-6
    i = np.logspace(-3, 0, 400)
    V = E_rev + b * np.log10(i / i0)
    out, err = iterative_tafel_fit(V, i, E_rev, HFR=0.0, R_CL=0.0,
                                   i_lower=0.005, i_upper=0.5)
    assert out is not None
    fit = out[0]
    assert abs(fit["b_kin"] - b) < 0.02


def test_lsv_her_decoupling_end_to_end(tmp_path):
    # Synthetic HER LSV: negative current, V = -b*log10(|i|/i0) below E_rev = 0.
    b, i0 = 0.06, 1e-6
    i_mag = np.logspace(-3, 0, 400)        # 1 mA/cm² .. 1 A/cm²
    i = -i_mag                              # cathodic (negative) current
    V = -b * np.log10(i_mag / i0)           # native potential, all < 0
    csv = tmp_path / "her.csv"
    lines = ["E(V),j(A/cm2)"] + [f"{v},{ii}" for v, ii in zip(V, i)]
    csv.write_text("\n".join(lines), encoding="utf-8")

    params = {
        "reaction": "Auto", "reference": "RHE", "ph": 0.0, "temperature": 298.15,
        "hfr": 0.0, "rcl": 0.0, "i_lower": 0.02, "i_upper": 0.8,
        "target_i": 0.5, "x_log": False,
    }
    res = LsvTool().run([csv], params)

    # Auto-detected HER and recovered the Tafel slope magnitude.
    assert any("HER" in w for w in res.warnings)
    assert abs(res.scalars["b_kin (V/dec)"] - b) < 0.02

    # Native-sign display: original LSV and the kinetic curve sit below E_rev = 0.
    curves = res.tables["Curves (stacked)"]
    assert (curves["Original LSV (RHE)"] < 0).all()
    assert (curves["E_rev + η_kin"] <= curves["E_rev"] + 1e-9).all()


def test_it_integrate_potential_mode(tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("T(s)\tE(V)\n0\t1.0\n1\t1.1\n2\t1.2\n", encoding="utf-8")
    f2.write_text("T(s)\tE(V)\n0\t2.0\n1\t2.1\n2\t2.2\n", encoding="utf-8")
    out = integrate_potentiostatic_tests([str(f1), str(f2)], interval_s=None, trace_type="E_t")
    assert out["headers"] == ["T(s)", "E(V)"]
    # Second file's time should be shifted to continue after the first.
    t = out["data"][:, 0]
    assert t[-1] > 2.0
    assert out["n_out"] == 6


def test_it_integrate_current_mode_without_potential_column(tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("i(A/cm2)\tT(s)\n1.0\t0\n1.1\t1\n", encoding="utf-8")
    f2.write_text("i(A/cm2)\tT(s)\n2.0\t0\n2.1\t1\n", encoding="utf-8")
    out = integrate_potentiostatic_tests([str(f1), str(f2)], interval_s=None, trace_type="i_t")
    assert out["headers"] == ["i(A/cm2)", "T(s)"]
    assert out["time_header"] == "T(s)"
    assert np.allclose(out["data"][:, 1], [0.0, 1.0, 2.0, 3.0])
    assert out["n_out"] == 4


def test_ftir_header_detection_and_t_to_a(tmp_path):
    csv = tmp_path / "ftir_t.csv"
    csv.write_text("Wavenumber (cm-1),T%\n4000,100\n3000,10\n", encoding="utf-8")
    spec = read_ftir_file(csv)
    assert spec.source_type == "T"
    assert spec.x_column == "Wavenumber (cm-1)"
    assert spec.y_column == "T%"
    assert np.allclose(spec.wavenumber, [3000, 4000])
    assert np.allclose(convert_t_to_a(np.array([100.0, 10.0])), [0.0, 1.0])


def test_ftir_absorbance_header_detection(tmp_path):
    txt = tmp_path / "ftir_a.txt"
    txt.write_text("cm-1\tA%\n1000\t0.2\n900\t0.4\n", encoding="utf-8")
    spec = read_ftir_file(txt)
    assert spec.source_type == "A"
    assert np.allclose(spec.wavenumber, [900, 1000])
    assert np.allclose(spec.intensity, [0.4, 0.2])


def test_ftir_invalid_t_normalization_and_offset():
    converted = convert_t_to_a(np.array([100.0, 0.0, -5.0, 10.0]))
    assert np.allclose(converted[[0, 3]], [0.0, 1.0])
    assert np.isnan(converted[1]) and np.isnan(converted[2])
    assert np.allclose(
        normalize_spectrum(np.array([2.0, 4.0]), "Maximum normalization"),
        [0.5, 1.0],
    )
    assert np.allclose(
        normalize_spectrum(np.array([2.0, 4.0]), "Min-max normalization"),
        [0.0, 1.0],
    )
    assert np.allclose(apply_vertical_offset(np.array([1.0, 2.0]), 2, 0.5), [2.0, 3.0])


def test_ftir_tool_end_to_end_and_data_only_exports(tmp_path):
    t_file = tmp_path / "t.dat"
    a_file = tmp_path / "a.txt"
    t_file.write_text("cm-1 T%\n4000 100\n3000 10\n", encoding="utf-8")
    a_file.write_text("cm-1\tA%\n4000\t0.1\n3000\t0.3\n", encoding="utf-8")
    params = {
        "display_mode": "A",
        "normalization": "None",
        "curve_names": "T curve,A curve",
        "line_width": 1.5,
        "vertical_offset": 0.2,
        "legend_position": "Best",
        "hide_y_ticks": False,
    }

    tool = FtirTool()
    result = tool.run([t_file, a_file], params)
    plot_data = result.tables["FTIR plot data"]
    assert set(plot_data["Curve"]) == {"T curve", "A curve"}
    t_rows = plot_data[plot_data["Curve"] == "T curve"].sort_values("Wavenumber (cm-1)")
    a_rows = plot_data[plot_data["Curve"] == "A curve"].sort_values("Wavenumber (cm-1)")
    assert np.allclose(t_rows["Y before offset"], [1.0, 0.0])
    assert np.allclose(a_rows["Y display"], [0.5, 0.3])

    artifacts = tool.exportable_artifacts(result)
    assert [a.kind for a in artifacts] == ["csv", "xlsx"]
    assert all(a.kind not in {"png", "pdf", "svg"} for a in artifacts)


def test_eis_two_rc_roundtrip():
    # Synthesize a faradaic TLM with TWO RC units (one distributed rung + one
    # series R||CPE arc) and check the fit recovers the planted resistances.
    logic = EisLogic()
    logic.fit_n_jobs = 1  # keep the test in-process (no worker-process spawning)
    model = EisLogic.FARADAIC_TML_MODEL
    f = np.logspace(5, -2, 140)
    # canonical: [L_wire, HFR, R_CL, Q_dl, Phi, Theta, R_k] + [R_2, Q_2, Phi_2]
    canon = [0.0, 0.5, 0.6, 0.10, 0.85, 1.0, 3.0, 0.5, 0.05, 0.90]
    z = logic.evaluate_model(model, canon, f)

    # init from the truth-ish values: [HFR, R_CL, Q_dl, Phi, R_k, R_2, Q_2, Phi_2]
    init = [0.5, 0.6, 0.10, 0.85, 3.0, 0.5, 0.05, 0.90]
    out, err = logic.fit_impedance(model, init, f, z, n_restarts=6, random_seed=0)
    assert err is None
    p, se, _resid, info = out
    assert len(p) == 10                      # 7 core + one extra triple
    assert abs(p[2] - 0.6) < 0.06            # R_CL
    assert abs(p[6] - 3.0) < 0.3             # R_k
    assert abs(p[7] - 0.5) < 0.05            # R_2 (series arc)


def test_eis_stable_coth_matches_analytic():
    # The overflow-safe coth/tanh must equal the analytic values and stay finite
    # where the old (e^x±e^-x) form overflowed to inf/nan at high ω (large |x|).
    logic = EisLogic()
    x = np.array([0.05 + 0.02j, 1.0 + 0.5j, 8.0 + 3.0j, 40.0 + 10.0j])
    assert np.allclose(logic.JPcoth(x), 1.0 / np.tanh(x))
    assert np.allclose(logic.JPtanh(x), np.tanh(x))
    assert np.all(np.isfinite(logic.JPcoth(np.array([300.0 + 300.0j]))))


def test_eis_faradaic_limits():
    # Two physics limits that anchor the faradaic model to Eikerling–Kornyshev.
    logic = EisLogic()
    f = np.logspace(5, -2, 120)
    hfr, rcl, qdl, phi, rk = 0.5, 0.6, 0.10, 0.85, 3.0

    # (1) R_k -> inf collapses the faradaic TML onto the blocking "Transmission Line".
    z_far = logic.evaluate_model(
        EisLogic.FARADAIC_TML_MODEL, [0.0, hfr, rcl, qdl, phi, 1.0, np.inf], f)
    z_block = logic.evaluate_model(
        "Transmission Line", [0.0, hfr, rcl, qdl, phi, 1.0, np.inf], f)
    assert np.allclose(z_far, z_block, rtol=1e-6, atol=1e-9)

    # (2) DC limit == HFR + sqrt(R_CL*R_k)*coth(sqrt(R_CL/R_k))  (E–K Eq. 11/26).
    z_lf = logic.evaluate_model(
        EisLogic.FARADAIC_TML_MODEL, [0.0, hfr, rcl, qdl, phi, 1.0, rk],
        np.array([1e-9]))[0]
    r_diff = hfr + np.sqrt(rcl * rk) / np.tanh(np.sqrt(rcl / rk))
    assert abs(np.real(z_lf) - r_diff) < 1e-3
    assert abs(np.imag(z_lf)) < 1e-3


def test_eis_seed_extra_arcs_locates_planted_arc():
    # Data-driven arc seeding returns finite, in-range guesses for the extra unit.
    from workbench.tools.eis.tool import _seed_extra_arcs
    logic = EisLogic()
    f = np.logspace(5, -2, 200)
    canon = [0.0, 0.5, 0.3, 0.10, 0.90, 1.0, 2.0, 0.5, 0.5, 0.90]  # + one series arc
    z = logic.evaluate_model(EisLogic.FARADAIC_TML_MODEL, canon, f)
    seeds = _seed_extra_arcs(f, z, n_rc=2)
    assert 2 in seeds
    r2, q2, phi2 = seeds[2]
    assert np.isfinite(r2) and r2 > 0
    assert 1e-9 <= q2 <= 1e3
    assert _seed_extra_arcs(f, z, n_rc=1) == {}   # no extras requested


def test_eis_modulus_weighting_runs():
    # Modulus weighting (the default) must still recover the planted parameters
    # on a clean spectrum.
    logic = EisLogic()
    logic.fit_n_jobs = 1
    model = EisLogic.FARADAIC_TML_MODEL
    f = np.logspace(5, -2, 140)
    canon = [0.0, 0.5, 0.6, 0.10, 0.85, 1.0, 3.0, 0.5, 0.05, 0.90]
    z = logic.evaluate_model(model, canon, f)
    init = [0.5, 0.6, 0.10, 0.85, 3.0, 0.5, 0.05, 0.90]
    out, err = logic.fit_impedance(model, init, f, z, n_restarts=4, random_seed=0,
                                   weighting="modulus")
    assert err is None
    p = out[0]
    assert abs(p[1] - 0.5) < 0.03 and abs(p[2] - 0.6) < 0.06   # HFR, R_CL
