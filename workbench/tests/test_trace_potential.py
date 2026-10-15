"""Reference conversion must agree across trace previews and export formats."""

from io import StringIO

import numpy as np
import pandas as pd
import pytest

from workbench.core.potential import PotentialCorrection
from workbench.tools.it_integrate.logic import integrate_potentiostatic_tests
from workbench.tools.it_integrate.tool import ItIntegrateTool


def _params(**overrides):
    return {
        "trace_type": "E_t", "interval": "none",
        "electrode_setup": "Three-electrode",
        "reference": "Custom E_ref vs SHE", "custom_e_ref_vs_she": 0.25,
        "target_scale": "RHE", "ph": 0.0,
        **overrides,
    }


def test_trace_original_scale_preserves_values_headers_and_text(tmp_path):
    source = tmp_path / "original.txt"
    source.write_text("T(s)\tE(V)\ti(A)\n0\t0.1\t0.002\n1\t0.2\t0.003\n")
    tool = ItIntegrateTool()

    result = tool.run([source], {
        "trace_type": "E_t", "interval": "none",
        "electrode_setup": "Three-electrode", "target_scale": "Original",
    })

    assert list(result.tables["Integrated"].columns) == ["T(s)", "E(V)", "i(A)"]
    np.testing.assert_allclose(
        result.tables["Integrated"].to_numpy(), [[0, 0.1, 0.002], [1, 0.2, 0.003]],
    )
    assert result.payload["text"] == (
        "T(s)\tE(V)\ti(A)\n0.00000\t0.1\t0.002\n1.00000\t0.2\t0.003\n"
    )
    assert result.figures["E-t"].axes[0].get_ylabel() == "E(V)"
    specs = {spec.key: spec for spec in tool.parameter_schema()}
    assert specs["electrode_setup"].default == "Two-electrode"
    assert specs["target_scale"].default == "Original"
    assert specs["potential_unit"].choices == ["V", "mV"]


@pytest.mark.parametrize(
    "unit,values,expected", [
        ("V", [0.1, 0.2], [0.35, 0.45]),
        ("mV", [100, 200], [350, 450]),
    ],
)
def test_trace_units_preview_and_both_exports_agree(tmp_path, unit, values, expected):
    source = tmp_path / "trace.txt"
    source.write_text(
        f"T(s)\tE({unit})\ti(A)\n0\t{values[0]}\t0.002\n1\t{values[1]}\t0.003\n"
    )
    tool = ItIntegrateTool()
    # Header units take precedence over the UI fallback.
    result = tool.run([source], _params(potential_unit="mV" if unit == "V" else "V"))
    frame = result.tables["Integrated"]
    potential_header = f"E({unit})[vs_RHE]"

    np.testing.assert_allclose(frame[potential_header], expected)
    np.testing.assert_allclose(frame["T(s)"], [0, 1])
    np.testing.assert_allclose(frame["i(A)"], [0.002, 0.003])
    axis = result.figures["E-t"].axes[0]
    np.testing.assert_allclose(axis.lines[0].get_ydata(), expected)
    assert "RHE" in axis.get_ylabel()
    assert unit in axis.get_ylabel()
    artifacts = {artifact.kind: artifact.payload for artifact in tool.exportable_artifacts(result)}
    pd.testing.assert_frame_equal(artifacts["xlsx"]["Integrated"], frame)
    settings = artifacts["xlsx"]["Potential settings"].set_index("Setting")["Value"]
    assert settings["Reference electrode"] == "Custom E_ref vs SHE"
    assert settings["Potential shift (V)"] == 0.25
    assert settings["pH"] == 0.0
    assert settings["Potential columns corrected"] == 1
    pd.testing.assert_frame_equal(
        pd.read_csv(StringIO(artifacts["txt"]), sep="\t"), frame, check_dtype=False,
    )


def test_it_source_groups_correct_potentials_but_keep_current_preview(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("E(V)\ti(A)\tT(s)\n0.1\t1\t0\n0.3\t3\t2\n")
    second.write_text("E(V)\ti(A)\tT(s)\n0.4\t4\t0\n0.6\t6\t2\n")
    result = ItIntegrateTool().run(
        [first, second],
        _params(trace_type="i_t", interval="1 s", designate_sources=True),
    )
    frame = result.tables["Integrated"]

    np.testing.assert_allclose(frame["first.txt · E(V)[vs_RHE]"], [0.35, 0.45, 0.55])
    np.testing.assert_allclose(frame["second.txt · E(V)[vs_RHE]"], [0.65, 0.75, 0.85])
    np.testing.assert_allclose(frame["first.txt · T(s)"], [0, 1, 2])
    np.testing.assert_allclose(frame["second.txt · T(s)"], [4, 5, 6])
    axis = result.figures["i-t"].axes[0]
    assert axis.get_ylabel() == "i(A)"
    np.testing.assert_allclose(axis.lines[0].get_ydata(), [1, 2, 3])
    np.testing.assert_allclose(axis.lines[1].get_ydata(), [4, 5, 6])
    pd.testing.assert_frame_equal(
        pd.read_csv(StringIO(result.payload["text"]), sep="\t"), frame, check_dtype=False,
    )


def test_two_electrode_trace_bypasses_reference_and_ph(tmp_path):
    source = tmp_path / "cell.txt"
    source.write_text("T(s)\tE(mV)\n0\t100\n1\t200\n")
    result = ItIntegrateTool().run(
        [source], _params(electrode_setup="Two-electrode", ph=14),
    )

    frame = result.tables["Integrated"]
    np.testing.assert_allclose(frame["E(mV)[cell_voltage]"], [100, 200])
    axis = result.figures["E-t"].axes[0]
    np.testing.assert_allclose(axis.lines[0].get_ydata(), [100, 200])
    assert "cell" in axis.get_ylabel().lower()
    assert "RHE" not in axis.get_ylabel()


@pytest.mark.parametrize("header", ["E", "V", "Potential"])
def test_trace_unknown_units_use_selected_fallback_and_report_it(tmp_path, header):
    source = tmp_path / "unknown.txt"
    source.write_text(f"T(s)\t{header}\n0\t100\n1\t200\n")
    result = ItIntegrateTool().run([source], _params(potential_unit="mV"))

    np.testing.assert_allclose(result.tables["Integrated"][f"{header}(mV)[vs_RHE]"], [350, 450])
    assert any("fallback (mV)" in warning for warning in result.warnings)
    assert "mV" in result.figures["E-t"].axes[0].get_ylabel()
    exported = tmp_path / "exported.txt"
    exported.write_text(result.payload["text"])
    reimported = integrate_potentiostatic_tests([exported], trace_type="E_t")
    assert reimported["warnings"] == []
    np.testing.assert_allclose(reimported["data"][:, 1], [350, 450])


def test_trace_does_not_shift_cell_voltage_alongside_reference_potential(tmp_path):
    source = tmp_path / "mixed.txt"
    source.write_text("T(s)\tEwe/V\tEwe-Ece/V\ti(A)\n0\t0.1\t1.5\t2\n1\t0.2\t1.6\t3\n")
    result = ItIntegrateTool().run([source], _params())
    frame = result.tables["Integrated"]

    np.testing.assert_allclose(frame["Ewe/V[vs_RHE]"], [0.35, 0.45])
    np.testing.assert_allclose(frame["Ewe-Ece/V"], [1.5, 1.6])
    np.testing.assert_allclose(frame["i(A)"], [2, 3])


def test_trace_can_convert_directly_to_she_and_reimport_export(tmp_path):
    source = tmp_path / "she.txt"
    source.write_text("T(s)\tE(V)\n0\t0.1\n1\t0.2\n")
    correction = PotentialCorrection.from_params(_params(target_scale="SHE", ph=7))
    result = integrate_potentiostatic_tests(
        [source], trace_type="E_t", potential_correction=correction,
    )
    np.testing.assert_allclose(result["data"][:, 1], [0.35, 0.45])

    exported = tmp_path / "exported.txt"
    exported.write_text(result["text"])
    reimported = integrate_potentiostatic_tests([exported], trace_type="E_t")
    assert reimported["warnings"] == []
    np.testing.assert_allclose(reimported["data"], result["data"])


def test_current_only_trace_reports_that_no_potential_was_corrected(tmp_path):
    source = tmp_path / "current.txt"
    source.write_text("T(s)\ti(A)\n0\t1\n1\t2\n")
    result = ItIntegrateTool().run([source], _params(trace_type="i_t"))

    np.testing.assert_allclose(result.tables["Integrated"].to_numpy(), [[0, 1], [1, 2]])
    assert any("No reference potential columns" in warning for warning in result.warnings)
    assert "no potential correction applied" in result.text_blocks["Potential scale"]
    assert result.payload["potential_correction"]["Potential shift (V)"] == 0.0
    assert result.payload["potential_correction"]["Potential columns corrected"] == 0
