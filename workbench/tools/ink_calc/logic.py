"""AEM catalyst-ink recipe math, lifted verbatim (numerically) from
Ink_Calculator2.py's ``calculate()`` with all Tkinter UI code stripped out.

Pure in, pure out: takes a params dict, returns a results dict. No I/O.
"""
from __future__ import annotations


def calculate_ink_recipe(params: dict) -> dict:
    mode = params.get("mode", "area")
    loading = float(params.get("loading", 0.0))
    ink_dens = float(params.get("ink_density", 2.0))
    L = float(params.get("length", 0.0))
    W = float(params.get("width", 0.0))
    avail_cat = float(params.get("avail_cat", 0.0))
    ionomers = params.get("ionomers", []) or []
    extras = params.get("extras", []) or []

    # --- 1. Catalyst mass on substrate ---
    if mode == "area":
        m_cat = loading * L * W
    else:
        m_cat = avail_cat

    # --- 2. Theoretical components ---
    total_iono = 0.0
    total_disp_mass = 0.0
    total_disp_vol_ml = 0.0
    ionomer_results = []
    for idx, iono in enumerate(ionomers, start=1):
        pct = float(iono.get("ratio", 0.0)) / 100.0
        m_i = (m_cat * pct) / (1 - pct) if pct < 1 else 0.0
        total_iono += m_i
        disp_frac = float(iono.get("conc", 0.0)) / 100.0
        m_disp_i = m_i / disp_frac if disp_frac > 0 else 0.0
        total_disp_mass += m_disp_i
        density = float(iono.get("density", 0.0))
        v_disp_ml = m_disp_i / 1000.0 / density if density > 0 else 0.0
        total_disp_vol_ml += v_disp_ml
        ionomer_results.append({
            "type": idx,
            "active_mg": m_i,
            "dispersion_mg": m_disp_i,
            "volume_ul": v_disp_ml * 1000.0,
        })

    # Extras (other solids by % of total solids)
    sum_r = sum(float(e.get("ratio", 0.0)) / 100.0 for e in extras)
    base_solids = m_cat + total_iono
    total_solids = base_solids / (1 - sum_r) if sum_r < 1 else 0.0
    extra_results = []
    for e in extras:
        r = float(e.get("ratio", 0.0)) / 100.0
        extra_results.append({
            "name": e.get("name", "Material"),
            "mass_mg": total_solids * r,
        })

    v_ink = m_cat / ink_dens if ink_dens > 0 else 0.0

    # --- 3. Spray-loss adjustment (area mode only) ---
    adjusted = None
    if mode == "area" and L > 0 and W > 0:
        loss_factor = min(0.212 * ((1.0 / L) + (1.0 / W)), 0.99)
        scale_factor = 1.0 / (1.0 - loss_factor)
        adjusted = {
            "loss_factor": loss_factor,
            "scale_factor": scale_factor,
            "required_catalyst_mg": m_cat * scale_factor,
            "ionomers": [
                {"type": r["type"], "required_volume_ul": r["volume_ul"] * scale_factor}
                for r in ionomer_results
            ],
        }

    return {
        "catalyst_mg": m_cat,
        "ink_volume_ml": v_ink,
        "total_ionomer_mg": total_iono,
        "total_dispersion_mass_mg": total_disp_mass,
        "total_dispersion_volume_ul": total_disp_vol_ml * 1000.0,
        "ionomers": ionomer_results,
        "extras": extra_results,
        "adjusted": adjusted,
    }
