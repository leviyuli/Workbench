"""Logic-only tests (no Qt). Run with: pytest workbench/tests"""
import numpy as np
import pandas as pd

from workbench.tools.ink_calc.logic import calculate_ink_recipe
from workbench.tools.ecsa.logic_ecsa import compute_cdl_ecsa
from workbench.tools.lsv.logic_tafel import iterative_tafel_fit
from workbench.tools.it_integrate.logic import integrate_potentiostatic_tests


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


def test_it_integrate(tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("T(s)\tE(V)\n0\t1.0\n1\t1.1\n2\t1.2\n", encoding="utf-8")
    f2.write_text("T(s)\tE(V)\n0\t2.0\n1\t2.1\n2\t2.2\n", encoding="utf-8")
    out = integrate_potentiostatic_tests([str(f1), str(f2)], interval_s=None)
    assert out["headers"] == ["T(s)", "E(V)"]
    # Second file's time should be shifted to continue after the first.
    t = out["data"][:, 0]
    assert t[-1] > 2.0
    assert out["n_out"] == 6
