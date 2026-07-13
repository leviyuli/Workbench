# EIS faradaic model — validation notes

Audit of the transmission-line (TML) faradaic model used by the EIS tool, against
Eikerling & Kornyshev, *Electrochemical impedance of the cathode catalyst layer in
polymer electrolyte fuel cells*, J. Electroanal. Chem. 475 (1999) 107–123 (the E–K
paper). Implementation lives in
[`logic_eis.py`](logic_eis.py) (`EisLogic.evaluate_model`).

## The model we compute

Faradaic branch (`FARADAIC_TML_MODEL`), with `ω = i·2π·f`:

```
Y_cpe = Q_dl · ω^Phi                     # CPE double-layer admittance
Z_k   = 1 / (1/R_k + Y_cpe)              # R_k ∥ CPE   (one charge-transfer unit)
Z     = Z_L + HFR + √(R_CL · Z_k) · coth(√(R_CL / Z_k))
        + Σ_{i=2..N} 1 / (1/R_i + Q_i · ω^Phi_i)   # extra series R‖CPE arcs
```

`Z_L = L_wire·ω^Theta` is an optional series wire inductance; `HFR` is the
series high-frequency (membrane + contact) resistance.

## Mapping to Eikerling–Kornyshev (linear regime)

The single-unit core (drop the extras and `Z_L`) is **exactly** E–K's small-signal
CL impedance, their Eq. (14):

```
Z = √(R_p · Z_ct) · coth(√(R_p / Z_ct)),    Z_ct = R_ct / (1 + iω/ω_c),   ω_c = 1/(R_ct·C_dl)
```

| Our symbol | E–K symbol | Meaning |
|-----------|-----------|---------|
| `R_CL`    | `R_p`     | ionic/proton transport resistance of the catalyst layer |
| `R_k`     | `R_ct`    | charge-transfer resistance |
| `Q_dl, Phi` | `C_dl`  | double-layer capacitance, generalized to a CPE |
| `HFR`     | (external) | series membrane/contact resistance, added outside the CL |

With `Phi = 1` and `Q_dl = C_dl`, `Z_k = R_k/(1 + iω R_k Q_dl) = R_ct/(1+iω/ω_c)`
and the two forms are identical. So the CPE is only a phenomenological *depression*
of the ideal double layer; the porous-electrode physics is E–K's.

Consistency checks (all hold in the code):

- **ω → 0 (DC):** `Z → HFR + √(R_CL·R_k)·coth(√(R_CL/R_k))` — E–K Eq. (11)/(26),
  the differential resistance `√(R_p R_ct) coth(l/λ_p)`.
- **R_k → ∞ (blocking):** reduces to the non-faradaic `"Transmission Line"` model
  (`Z = HFR + √(R_CL/Y_cpe)·coth(√(R_CL·Y_cpe))`), the De Levie blocking pore whose
  DC limit is `HFR + R_CL/3`.
- **ω ≫ ω_c (high frequency):** the coth argument grows and the response tends to
  the 45° Warburg-like branch `∝ √(R_p/C_dl)·ω^(−1/2)` — E–K Eq. (27).

**Verdict:** the single charge-transfer-unit faradaic TML is a valid, physically
grounded implementation of the E–K macrohomogeneous CL impedance in the linear
(small-signal) regime.

## Q3 — why changing R_CL reshapes the whole arc

`R_CL` (= `R_p`) appears **both** in the prefactor `√(R_CL·Z_k)` **and** inside
`coth(√(R_CL/Z_k))`. The Nyquist shape is governed by the ratio
`l/λ_p = √(R_CL/R_k)`. Therefore changing `R_CL` simultaneously:

- lengthens/shortens the 45° high-frequency branch (`∝ √(R_p/C_dl)`), and
- changes the curvature and diameter of the low-frequency arc
  (`R_diff = √(R_CL·R_k)·coth(√(R_CL/R_k))`).

This is **expected and correct** for a transmission line — `R_CL` is a
*distributed* transport resistance, not an independent series resistor. The only
element that shifts the spectrum purely along the real axis without deforming the
arc is `HFR`. (Observing that "only the slope near the HFR changes" usually means
`R_CL ≪ R_k`, where `l/λ_p` is small and the arc is already near a perfect
semicircle, so the visible effect concentrates in the HF branch.)

## Q4 caveat — the multi-RC (units 2..N) are phenomenological

The extra series `R_i ∥ CPE_i` arcs are **not** part of the E–K model. E–K's own
route to a second spectral feature is finite **oxygen diffusion** (their Eq. (47)),
which has a different functional form (a single depressed relaxation from the
diffusion term), not a lumped R‖CPE. The series-arc approach is a reasonable
engineering stand-in for an additional interfacial/contact or transport relaxation,
but it should be read as a **phenomenological Voigt-type add-on**, not as CL
physics. Because these arcs overlap the TML response in frequency, their parameters
are weakly separable — hence the data-driven initialization and SSR-based ranking
added for multi-RC fits (see `_seed_extra_arcs` in [`tool.py`](tool.py) and the
`ssr_multi_rc` branch in `EisLogic.fit_impedance`).
