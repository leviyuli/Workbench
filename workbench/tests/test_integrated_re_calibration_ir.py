"""Reference calibration compares both electrodes on the same iR-free basis."""

import numpy as np
import pandas as pd
import pytest

from workbench.tools.integrated_re.exports import plot_tables
from workbench.tools.integrated_re.tool import IntegratedReTool


CURRENT = "Current density (A/cm²)"
BEFORE = "E before correction (V vs RHE)"
AFTER = "E (V vs RHE)"


def _curve(path, potential, current):
    pd.DataFrame({"E(V)": potential, "j(A/cm2)": current}).to_csv(path, index=False)
    return path


def _eis(path, resistance):
    pd.DataFrame({
        "Frequency (Hz)": [10000, 1000, 100],
        "Re(Z) (Ohm.cm2)": [resistance, resistance + 0.1, resistance + 0.3],
        "Im(Z) (Ohm.cm2)": [0.0, -0.1, -0.3],
    }).to_csv(path, index=False)
    return path


def _case(folder, role, mode, *, include_ir=False):
    q = np.array([0.0, 0.02, 0.08, 0.04, 0.01])
    sign = -1 if role == "cathode" else 1
    j = sign * q
    calibration_e = (0.0 if sign < 0 else 1.5) + sign * np.arange(len(q)) * 0.04
    corrected_e = calibration_e + 0.18
    raw_e = corrected_e + j * 3.0
    sources = {
        "anode": _curve(folder / "anode.csv", 1.5 + q, q),
        "cathode": _curve(folder / "cathode.csv", -0.1 - q, -q),
        "calibration": _curve(folder / "calibration.csv", calibration_e, j),
        "eis": {role: [_eis(folder / "first_eis.csv", 2.0),
                       _eis(folder / "second_eis.csv", 4.0)]},
    }
    sources[role] = _curve(folder / f"{role}.csv", raw_e, j)
    if include_ir:
        other = "anode" if role == "cathode" else "cathode"
        sources["eis"][other] = [_eis(folder / "other_eis.csv", 0.9)]
    params = {spec.key: spec.default for spec in IntegratedReTool().parameter_schema()}
    params.update(correction_mode=mode, calibration_electrode=role.capitalize(),
                  calibration_reference="RHE", calibration_ph=0.0,
                  manual_shift=0.04, include_ir=include_ir,
                  onset_current=0.05,
                  integrated_peak_min=corrected_e[2] - 0.045,
                  integrated_peak_max=corrected_e[2] + 0.045,
                  calibration_peak_min=calibration_e[2] - 0.045,
                  calibration_peak_max=calibration_e[2] + 0.045)
    return params, sources, raw_e, corrected_e, calibration_e, j


@pytest.mark.parametrize("role", ["anode", "cathode"])
@pytest.mark.parametrize("mode", ["3E onset", "3E peak"])
@pytest.mark.parametrize("include_ir", [False, True])
def test_calibration_uses_signed_mean_hfr_and_preserves_measured_curves(
        tmp_path, role, mode, include_ir):
    params, sources, raw_e, corrected_e, calibration_e, j = _case(
        tmp_path, role, mode, include_ir=include_ir,
    )
    tool = IntegratedReTool()
    result = tool.run([], params, sources)

    assert result.scalars["Shared correction (V)"] == pytest.approx(-0.14)
    assert result.param_updates["calculated_shift"] == "-0.180000"
    assert result.param_updates["applied_shift"] == "-0.140000"
    integrated = result.tables["Calibration integrated LSV"]
    np.testing.assert_allclose(integrated[BEFORE], corrected_e, atol=1e-14)
    np.testing.assert_allclose(integrated[AFTER], calibration_e + 0.04, atol=1e-14)
    np.testing.assert_allclose(integrated[CURRENT], j)
    measured = result.tables[f"{role.capitalize()} LSV"]
    np.testing.assert_allclose(measured[BEFORE], raw_e)
    np.testing.assert_allclose(measured[AFTER], raw_e - 0.14, atol=1e-14)
    np.testing.assert_allclose(result.tables["Calibration LSV"][AFTER], calibration_e)
    if include_ir:
        np.testing.assert_allclose(measured["iR-corrected E (V vs RHE)"],
                                   calibration_e + 0.04, atol=1e-14)
    else:
        assert "Reconstructed cell (iR corrected)" not in result.tables

    alignment = result.figures["Calibration alignment"].axes[0]
    np.testing.assert_allclose(alignment.lines[0].get_xdata(), corrected_e, atol=1e-14)
    np.testing.assert_allclose(alignment.lines[1].get_xdata(), calibration_e + 0.04, atol=1e-14)
    assert "iR corrected" in alignment.get_title(loc="left")
    exported = plot_tables(result)["Calibration alignment"]
    np.testing.assert_allclose(exported["Integrated E iR corrected (V vs RHE)"], corrected_e, atol=1e-14)
    np.testing.assert_allclose(exported["Integrated E iR + RE corrected (V vs RHE)"],
                               calibration_e + 0.04, atol=1e-14)
    np.testing.assert_allclose(exported["3E E iR corrected (V vs RHE)"], calibration_e)


@pytest.mark.parametrize("role", ["anode", "cathode"])
@pytest.mark.parametrize("mode", ["3E onset", "3E peak"])
def test_3e_calibration_requires_selected_electrode_eis_with_overlays_off(tmp_path, role, mode):
    params, sources, *_ = _case(tmp_path, role, mode)
    other = "anode" if role == "cathode" else "cathode"
    sources["eis"] = {other: [_eis(tmp_path / "wrong_role_eis.csv", 0.2)]}
    with pytest.raises(ValueError, match=f"{role.capitalize()}.*EIS.*3E reference correction"):
        IntegratedReTool().run([], params, sources)


@pytest.mark.parametrize("mode", ["None", "Manual"])
def test_non_calibration_modes_keep_eis_optional(tmp_path, mode):
    params, sources, *_ = _case(tmp_path, "cathode", mode)
    sources.pop("calibration")
    sources.pop("eis")
    result = IntegratedReTool().run([], params, sources)
    assert result.scalars["Shared correction (V)"] == pytest.approx(0.04 if mode == "Manual" else 0)
    assert "Calibration integrated LSV" not in result.tables
