from copy import deepcopy

import pytest

from workbench.tools.ink_calc.logic import calculate_ink_recipe
from workbench.tools.ink_calc.tool import InkCalcTool


VALID_AREA = {
    "mode": "area",
    "length": 5.0,
    "width": 4.0,
    "loading": 0.5,
    "avail_cat": 12.5,
    "ink_density": 2.0,
    "ionomers": [{"ratio": 20.0, "conc": 5.0, "density": 0.78}],
    "extras": [{"name": "Carbon", "ratio": 10.0}],
}


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"length": 0}, "Active length must be above zero"),
        ({"loading": -0.1}, "Loading must be above zero"),
        ({"ink_density": 0}, "Ink concentration must be above zero"),
        ({"loading": float("inf")}, "Loading must be finite"),
        (
            {"ionomers": [{"ratio": -1, "conc": 5, "density": 0.78}]},
            "ratio must be at least 0% and below 100%",
        ),
        (
            {"ionomers": [{"ratio": 100, "conc": 5, "density": 0.78}]},
            "ratio must be at least 0% and below 100%",
        ),
        (
            {"ionomers": [{"ratio": 20, "conc": 0, "density": 0.78}]},
            "concentration must be above zero when used",
        ),
        (
            {"ionomers": [{"ratio": 20, "conc": 101, "density": 0.78}]},
            "concentration must be between 0% and 100%",
        ),
        (
            {"ionomers": [{"ratio": 20, "conc": 5, "density": 0}]},
            "density must be above zero when used",
        ),
        (
            {"extras": [{"name": "A", "ratio": 50}, {"name": "B", "ratio": 50}]},
            "must total less than 100%",
        ),
        (
            {"extras": [{"name": "A", "ratio": "not a number"}]},
            "ratio must be a number",
        ),
    ],
)
def test_invalid_recipes_are_rejected(changes, message):
    params = deepcopy(VALID_AREA)
    params.update(changes)
    with pytest.raises(ValueError, match=message):
        calculate_ink_recipe(params)


def test_catalyst_mode_requires_available_catalyst():
    params = deepcopy(VALID_AREA)
    params.update(mode="catalyst", avail_cat=0)
    with pytest.raises(ValueError, match="Available catalyst must be above zero"):
        calculate_ink_recipe(params)


def test_unused_zero_ratio_ionomer_may_have_zero_carrier_values():
    params = deepcopy(VALID_AREA)
    params["ionomers"].append({"ratio": 0, "conc": 0, "density": 0})
    result = calculate_ink_recipe(params)
    assert result["ionomers"][-1]["active_mg"] == 0
    assert result["ionomers"][-1]["volume_ul"] == 0


def test_adjusted_recipe_scales_every_component_total():
    result = calculate_ink_recipe(VALID_AREA)
    adjusted = result["adjusted"]
    assert adjusted is not None
    scale = adjusted["scale_factor"]

    for base_key, adjusted_key in (
        ("catalyst_mg", "required_catalyst_mg"),
        ("ink_volume_ml", "ink_volume_ml"),
        ("total_ionomer_mg", "total_ionomer_mg"),
        ("total_dispersion_mass_mg", "total_dispersion_mass_mg"),
        ("total_dispersion_volume_ul", "total_dispersion_volume_ul"),
        ("total_extra_mg", "total_extra_mg"),
        ("total_solids_mg", "total_solids_mg"),
    ):
        assert adjusted[adjusted_key] == pytest.approx(result[base_key] * scale)

    assert adjusted["ionomers"][0]["active_mg"] == pytest.approx(
        result["ionomers"][0]["active_mg"] * scale
    )
    assert adjusted["extras"][0]["mass_mg"] == pytest.approx(
        result["extras"][0]["mass_mg"] * scale
    )


def test_desktop_exports_include_primary_and_adjusted_totals():
    result = InkCalcTool().run([], VALID_AREA)
    summary = result.tables["Recipe summary"]
    quantities = set(summary["Quantity"])

    assert {"Catalyst", "Ink volume", "Total solids"} <= quantities
    assert {"Required catalyst", "Required ink volume", "Required total solids"} <= quantities
    workbook_artifact = next(
        artifact
        for artifact in InkCalcTool().exportable_artifacts(result)
        if artifact.kind == "xlsx"
    )
    assert "Recipe summary" in workbook_artifact.payload


def test_catalyst_mode_does_not_report_a_contradictory_required_batch():
    params = deepcopy(VALID_AREA)
    params.update(mode="catalyst", avail_cat=8.0)
    result = InkCalcTool().run([], params)

    assert "Required catalyst (mg)" not in result.scalars
    assert not any(
        str(quantity).startswith("Required")
        for quantity in result.tables["Recipe summary"]["Quantity"]
    )
    assert "spray-loss scaling is not applied" in result.text_blocks["Adjusted batch"]
