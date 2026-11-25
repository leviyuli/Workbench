"""Spectrum export labels follow case names without exposing internal IDs."""
import pandas as pd
import pytest

from workbench.core.tool_base import ToolResult
from workbench.tools.eis_generator.tool import EisGeneratorTool


def _result_with_names(names):
    rows = []
    tables = {}
    for index, name in enumerate(names):
        table_name = f"Spectrum {index + 1}"
        rows.append({"Case": name, "Case ID": "current" if index == 0 else f"case_{index}",
                     "Spectrum": table_name})
        tables[table_name] = pd.DataFrame({"Freq(Hz)": [1000.0], "Z'(Ohm.cm²)": [index + 0.1],
                                           "Z''(Ohm.cm²)": [-0.1]})
    return ToolResult(tables={"Resistance summary": pd.DataFrame(rows), **tables})


def _csv_artifacts(names):
    return [artifact for artifact in EisGeneratorTool().exportable_artifacts(_result_with_names(names))
            if artifact.kind == "csv"]


def test_current_and_saved_spectra_use_friendly_case_names():
    artifacts = _csv_artifacts(["Current", "Balanced"])
    assert [artifact.name for artifact in artifacts] == [
        "Current spectrum (CSV)", "Balanced spectrum (CSV)",
    ]
    assert [artifact.default_filename for artifact in artifacts] == [
        "eis_generator_current.csv", "eis_generator_balanced.csv",
    ]
    assert "case_1" not in artifacts[1].name
    assert "case_1" not in artifacts[1].default_filename


def test_duplicate_names_and_current_label_get_deterministic_collision_suffixes():
    names = ["Current", "Balanced", "Balanced", "current", "Balanced"]
    artifacts = _csv_artifacts(names)
    assert [artifact.name for artifact in artifacts] == [
        "Current spectrum (CSV)", "Balanced spectrum (CSV)",
        "Balanced spectrum (2) (CSV)", "current spectrum (2) (CSV)",
        "Balanced spectrum (3) (CSV)",
    ]
    assert [artifact.default_filename for artifact in artifacts] == [
        "eis_generator_current.csv", "eis_generator_balanced.csv",
        "eis_generator_balanced_2.csv", "eis_generator_current_2.csv",
        "eis_generator_balanced_3.csv",
    ]
    assert [(a.name, a.default_filename) for a in _csv_artifacts(names)] == [
        (a.name, a.default_filename) for a in artifacts
    ]


def test_unicode_and_punctuation_slug_collisions_keep_unique_payloads():
    names = ["Current", "A-B", "A B", "a_b", "离子", "离子", "!!!", "???"]
    artifacts = _csv_artifacts(names)
    filenames = [artifact.default_filename for artifact in artifacts]
    assert filenames == [
        "eis_generator_current.csv", "eis_generator_a_b.csv", "eis_generator_a_b_2.csv",
        "eis_generator_a_b_3.csv", "eis_generator_离子.csv", "eis_generator_离子_2.csv",
        "eis_generator_spectrum.csv", "eis_generator_spectrum_2.csv",
    ]
    assert len({name.casefold() for name in filenames}) == len(filenames)
    for index, artifact in enumerate(artifacts):
        assert artifact.payload.iloc[0, 1] == index + 0.1


@pytest.mark.parametrize("malformed", ["", "  ", None, 123])
def test_empty_or_malformed_names_use_spectrum_fallback(malformed):
    artifact = _csv_artifacts([malformed])[0]
    assert artifact.name == "Spectrum (CSV)"
    assert artifact.default_filename == "eis_generator_spectrum.csv"


def test_natural_numbered_labels_cannot_overwrite_a_collision_filename():
    artifacts = _csv_artifacts(["Balanced", "Balanced", "Balanced 2"])
    assert [artifact.default_filename for artifact in artifacts] == [
        "eis_generator_balanced.csv", "eis_generator_balanced_2.csv",
        "eis_generator_balanced_2_2.csv",
    ]
