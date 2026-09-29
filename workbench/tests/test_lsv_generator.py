import numpy as np
import pytest

from workbench.tools.lsv.logic_tafel import calculate_RCL_overpotential
from workbench.tools.lsv_generator.logic import (
    HER_PRESETS,
    NOMINAL_MIN_CURRENT,
    OER_PRESETS,
    generate_lsv,
    oer_reversible_potential,
)
from workbench.tools.lsv_generator.tool import LsvGeneratorTool


OER = {"b": 0.060, "i0": 1e-6, "rcl": 0.20}
HER = {"b": 0.040, "i0": 1e-5, "rcl": 0.10}


def test_oer_generation_reconstructs_the_decoupling_equation_exactly():
    result = generate_lsv(
        "OER", 298.15, 0.05, "Current density",
        current_limit=1.0, oer_params=OER,
    )
    j = result["current_magnitude"]
    eta_kin = OER["b"] * np.log10(j / OER["i0"])
    eta_rcl = calculate_RCL_overpotential(j, OER["rcl"], OER["b"])
    eta_ohm = j * 0.05
    expected = oer_reversible_potential(298.15) + eta_kin + eta_rcl + eta_ohm

    assert len(j) == 400
    assert np.allclose(result["eta_kin_oer"], eta_kin)
    assert np.allclose(result["eta_rcl_oer"], eta_rcl)
    assert np.allclose(result["eta_ohm"], eta_ohm)
    assert np.allclose(result["potential"], expected)
    assert np.all(result["signed_current"] > 0)


def test_her_generation_uses_native_negative_current_and_potential():
    result = generate_lsv(
        "HER", 298.15, 0.02, "Current density",
        current_limit=0.5, her_params=HER,
    )

    expected_loss = (
        result["eta_kin_her"] + result["eta_rcl_her"] + result["eta_ohm"]
    )
    assert np.allclose(result["potential"], -expected_loss)
    assert np.allclose(result["signed_current"], -result["current_magnitude"])
    assert np.all(result["potential"] <= 0)
    assert result["metadata"]["e_rev"] == 0.0


def test_two_electrode_voltage_is_the_sum_of_both_electrodes_and_one_hfr():
    result = generate_lsv(
        "Two-electrode", 323.15, 0.03, "Current density",
        current_limit=2.0, oer_params=OER, her_params=HER,
    )
    expected_loss = (
        result["eta_kin_oer"] + result["eta_rcl_oer"]
        + result["eta_kin_her"] + result["eta_rcl_her"]
        + result["eta_ohm"]
    )
    assert np.allclose(result["eta_total"], expected_loss)
    assert np.allclose(
        result["potential"], oer_reversible_potential(323.15) + expected_loss
    )
    assert np.allclose(result["eta_ohm"], result["current_magnitude"] * 0.03)


@pytest.mark.parametrize(
    ("mode", "target", "kwargs"),
    [
        ("OER", 1.75, {"oer_params": OER}),
        ("HER", -0.35, {"her_params": HER}),
        ("Two-electrode", 1.85, {"oer_params": OER, "her_params": HER}),
    ],
)
def test_voltage_endpoint_is_solved_for_every_configuration(mode, target, kwargs):
    result = generate_lsv(
        mode, 298.15, 0.02, "Voltage", voltage_limit=target, **kwargs
    )
    assert result["metadata"]["limit_mode"] == "voltage"
    assert result["potential"][-1] == pytest.approx(target, abs=1e-10)
    assert result["current_magnitude"][-1] > result["current_magnitude"][0]


def test_voltage_solver_checks_the_safety_bound_after_a_partial_decade():
    params = {"b": 0.060, "i0": 0.2, "rcl": 0.0}
    target_current = 5e5
    target_voltage = (
        oer_reversible_potential(298.15)
        + params["b"] * np.log10(target_current / params["i0"])
    )
    result = generate_lsv(
        "OER", 298.15, 0.0, "Voltage",
        voltage_limit=target_voltage, oer_params=params,
    )
    assert result["metadata"]["current_end"] == pytest.approx(
        target_current, rel=1e-9
    )


def test_actual_start_is_largest_active_i0_and_losses_stay_nonnegative():
    result = generate_lsv(
        "Two-electrode", 298.15, 0.0, "Current density",
        current_limit=1.0,
        oer_params={"b": 0.05, "i0": 2e-5, "rcl": 0.0},
        her_params={"b": 0.03, "i0": 4e-4, "rcl": 0.0},
    )
    assert result["metadata"]["current_start"] == pytest.approx(4e-4)
    assert result["metadata"]["current_start"] > NOMINAL_MIN_CURRENT
    assert np.min(result["eta_kin_oer"]) >= 0
    assert np.min(result["eta_kin_her"]) >= -1e-15


def test_current_endpoint_is_included_exactly():
    result = generate_lsv(
        "OER", 298.15, 0.0, "Current density",
        current_limit=0.713, oer_params=OER,
    )
    assert result["current_magnitude"][-1] == pytest.approx(0.713)
    assert result["metadata"]["requested_endpoint"] == pytest.approx(0.713)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"temperature_k": 0.0}, "Temperature must be greater"),
        ({"hfr": -0.1}, "HFR must be nonnegative"),
        ({"oer_params": {"b": 0.0, "i0": 1e-6, "rcl": 0}}, "Tafel slope"),
        ({"oer_params": {"b": 0.05, "i0": 0.0, "rcl": 0}}, "exchange current"),
        ({"oer_params": {"b": 0.05, "i0": 1e-6, "rcl": -1}}, "R_CL"),
        ({"current_limit": 1e-6}, "effective starting current"),
    ],
)
def test_invalid_generator_inputs_are_rejected(changes, message):
    args = {
        "mode": "OER",
        "temperature_k": 298.15,
        "hfr": 0.0,
        "limit_mode": "Current density",
        "current_limit": 1.0,
        "oer_params": OER,
    }
    args.update(changes)
    with pytest.raises(ValueError, match=message):
        generate_lsv(**args)


def test_voltage_limit_must_be_beyond_start_in_the_reaction_direction():
    oer_start = generate_lsv(
        "OER", 298.15, 0, "Current density",
        current_limit=0.1, oer_params=OER,
    )["metadata"]["potential_start"]
    with pytest.raises(ValueError, match="must exceed"):
        generate_lsv(
            "OER", 298.15, 0, "Voltage",
            voltage_limit=oer_start, oer_params=OER,
        )

    her_start = generate_lsv(
        "HER", 298.15, 0, "Current density",
        current_limit=0.1, her_params=HER,
    )["metadata"]["potential_start"]
    with pytest.raises(ValueError, match="must be below"):
        generate_lsv(
            "HER", 298.15, 0, "Voltage",
            voltage_limit=her_start, her_params=HER,
        )


def test_presets_load_into_editable_fields_and_keep_condition_metadata():
    tool = LsvGeneratorTool()
    oer_name = next(iter(OER_PRESETS))
    her_name = next(iter(HER_PRESETS))
    assert tool.load_oer_preset([], {"oer_preset": oer_name}) == {
        "oer_b": OER_PRESETS[oer_name].b,
        "oer_i0": OER_PRESETS[oer_name].i0,
        "temperature": OER_PRESETS[oer_name].reference_temperature_k,
    }
    assert tool.load_her_preset([], {"her_preset": her_name}) == {
        "her_b": HER_PRESETS[her_name].b,
        "her_i0": HER_PRESETS[her_name].i0,
        "temperature": HER_PRESETS[her_name].reference_temperature_k,
    }
    assert OER_PRESETS[oer_name].conditions
    assert OER_PRESETS[oer_name].source.startswith("http")


def test_every_preset_loads_its_reference_temperature():
    tool = LsvGeneratorTool()
    for name, preset in OER_PRESETS.items():
        loaded = tool.load_oer_preset([], {"oer_preset": name})
        assert loaded["temperature"] == pytest.approx(preset.reference_temperature_k)
    for name, preset in HER_PRESETS.items():
        loaded = tool.load_her_preset([], {"her_preset": name})
        assert loaded["temperature"] == pytest.approx(preset.reference_temperature_k)

    assert OER_PRESETS["NiFe-LDH (1 M KOH)"].reference_temperature_k == 298.15
    assert OER_PRESETS["Commercial IrO₂ (PEM MEA)"].reference_temperature_k == 353.15


def test_schema_exposes_reaction_specific_fields_and_voltage_limits():
    specs = {spec.key: spec for spec in LsvGeneratorTool().parameter_schema()}
    assert specs["oer_b"].show_if == ("mode", ["OER", "Two-electrode"])
    assert specs["her_b"].show_if == ("mode", ["HER", "Two-electrode"])
    assert specs["oer_voltage_limit"].show_if == [
        ("limit_mode", "Voltage"), ("mode", "OER")
    ]
    assert specs["cell_voltage_limit"].show_if == [
        ("limit_mode", "Voltage"), ("mode", "Two-electrode")
    ]
    assert specs["oer_load_preset"].reset_method == "load_oer_preset"
    assert specs["her_load_preset"].reset_method == "load_her_preset"
    assert "temperature" in specs["oer_load_preset"].reset_keys
    assert "temperature" in specs["her_load_preset"].reset_keys


def _tool_params(**updates):
    params = {
        "mode": "OER",
        "temperature": 298.15,
        "hfr": 0.01,
        "oer_preset": next(iter(OER_PRESETS)),
        "oer_b": OER_PRESETS[next(iter(OER_PRESETS))].b,
        "oer_i0": OER_PRESETS[next(iter(OER_PRESETS))].i0,
        "oer_rcl": 0.1,
        "her_preset": next(iter(HER_PRESETS)),
        "her_b": HER_PRESETS[next(iter(HER_PRESETS))].b,
        "her_i0": HER_PRESETS[next(iter(HER_PRESETS))].i0,
        "her_rcl": 0.1,
        "limit_mode": "Current density",
        "current_limit": 1.0,
        "oer_voltage_limit": 2.0,
        "her_voltage_limit": -0.5,
        "cell_voltage_limit": 2.0,
    }
    params.update(updates)
    return params


def test_tool_returns_tables_three_plots_provenance_and_exports():
    tool = LsvGeneratorTool()
    result = tool.run([], _tool_params())

    data = result.tables["Generated LSV data"]
    assert len(data) == 400
    assert "Potential (V vs RHE)" in data
    assert "η_kin,OER (V)" in data
    assert "η_RCL,OER (V)" in data
    assert "η_ohm (V)" in data
    assert "Total Modeled Loss (V)" in data
    assert len(result.figures) == 3
    assert "values match the preset" in result.text_blocks["Preset provenance"]
    assert any("condition-specific" in warning for warning in result.warnings)

    artifacts = tool.exportable_artifacts(result)
    assert any(artifact.kind == "csv" for artifact in artifacts)
    assert any(artifact.kind == "xlsx" for artifact in artifacts)
    assert sum(artifact.kind == "png" for artifact in artifacts) == 3


def test_modified_preset_is_reported_and_full_cell_plot_uses_cell_voltage_label():
    result = LsvGeneratorTool().run(
        [], _tool_params(mode="Two-electrode", oer_b=0.071)
    )
    assert "values modified from the preset" in result.text_blocks["Preset provenance"]
    polarization_ax = result.figures["Generated LSV"].axes[0]
    assert "Cell voltage" in polarization_ax.get_xlabel()
    assert "Cell Voltage (V)" in result.tables["Generated LSV data"]
    assert "Potential (V vs RHE)" not in result.tables["Generated LSV data"]
