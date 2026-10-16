from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from workbench import theme
from workbench.core.tool_base import Artifact, ParameterSpec, Tool, ToolResult
from workbench.tools.lsv_generator.logic import (
    HER_PRESETS,
    OER_PRESETS,
    CatalystPreset,
    generate_lsv,
)


CUSTOM_PRESET = "Custom"
DEFAULT_OER_PRESET = next(iter(OER_PRESETS))
DEFAULT_HER_PRESET = next(iter(HER_PRESETS))


class LsvGeneratorTool(Tool):
    id = "lsv_generator"
    name = "LSV Generator"
    subtitle = (
        "Generate HER, OER or two-electrode polarization curves from Tafel, "
        "ohmic and catalyst-layer parameters."
    )
    needs_files = False
    run_button_label = "Generate LSV"

    def parameter_schema(self) -> list[ParameterSpec]:
        active_oer = ["OER", "Two-electrode"]
        active_her = ["HER", "Two-electrode"]
        oer_default = OER_PRESETS[DEFAULT_OER_PRESET]
        her_default = HER_PRESETS[DEFAULT_HER_PRESET]
        return [
            ParameterSpec(
                "mode", "Configuration", kind="choice",
                choices=["OER", "HER", "Two-electrode"], default="OER",
                tooltip=(
                    "Half-cell potentials are reported versus RHE. Two-electrode "
                    "results are reported as full-cell voltage."
                ),
                group="Model",
            ),
            ParameterSpec(
                "temperature", "Temperature", kind="float", default=298.15,
                minimum=1e-9, step=1.0, suffix="K",
                tooltip="Used to calculate the OER/full-cell reversible potential.",
                group="Model",
            ),
            ParameterSpec(
                "hfr", "Series HFR", kind="float", default=0.01,
                minimum=0.0, step=0.001, suffix="Ω·cm²",
                tooltip=(
                    "Area-specific series resistance. In two-electrode mode this "
                    "is one shared cell HFR and is counted once."
                ),
                group="Model",
            ),

            ParameterSpec(
                "oer_preset", "OER example", kind="choice",
                choices=[*OER_PRESETS, CUSTOM_PRESET], default=DEFAULT_OER_PRESET,
                tooltip="Condition-specific editable example; not a universal constant.",
                group="OER electrode", show_if=("mode", active_oer),
            ),
            ParameterSpec(
                "oer_load_preset", "OER parameters", kind="reset",
                default="Load preset",
                reset_keys=["oer_b", "oer_i0", "temperature"],
                reset_method="load_oer_preset",
                tooltip=(
                    "Copy the selected example's b, i₀ and reported reference "
                    "temperature into the editable fields."
                ),
                group="OER electrode", show_if=("mode", active_oer),
            ),
            ParameterSpec(
                "oer_b", "Tafel slope b", kind="float", default=oer_default.b,
                minimum=1e-12, step=0.001, suffix="V/dec",
                group="OER electrode", show_if=("mode", active_oer),
            ),
            ParameterSpec(
                "oer_i0", "Exchange current i₀", kind="float",
                default=oer_default.i0, minimum=1e-15, step=1e-6,
                suffix="A/cm²", group="OER electrode",
                show_if=("mode", active_oer),
            ),
            ParameterSpec(
                "oer_rcl", "R_CL", kind="float", default=0.0,
                minimum=0.0, step=0.001, suffix="Ω·cm²",
                group="OER electrode", show_if=("mode", active_oer),
            ),

            ParameterSpec(
                "her_preset", "HER example", kind="choice",
                choices=[*HER_PRESETS, CUSTOM_PRESET], default=DEFAULT_HER_PRESET,
                tooltip="Condition-specific editable example; not a universal constant.",
                group="HER electrode", show_if=("mode", active_her),
            ),
            ParameterSpec(
                "her_load_preset", "HER parameters", kind="reset",
                default="Load preset",
                reset_keys=["her_b", "her_i0", "temperature"],
                reset_method="load_her_preset",
                tooltip=(
                    "Copy the selected example's b, i₀ and reported reference "
                    "temperature into the editable fields."
                ),
                group="HER electrode", show_if=("mode", active_her),
            ),
            ParameterSpec(
                "her_b", "Tafel slope b", kind="float", default=her_default.b,
                minimum=1e-12, step=0.001, suffix="V/dec",
                group="HER electrode", show_if=("mode", active_her),
            ),
            ParameterSpec(
                "her_i0", "Exchange current i₀", kind="float",
                default=her_default.i0, minimum=1e-15, step=1e-5,
                suffix="A/cm²", group="HER electrode",
                show_if=("mode", active_her),
            ),
            ParameterSpec(
                "her_rcl", "R_CL", kind="float", default=0.0,
                minimum=0.0, step=0.001, suffix="Ω·cm²",
                group="HER electrode", show_if=("mode", active_her),
            ),

            ParameterSpec(
                "limit_mode", "Stop at", kind="choice",
                choices=["Current density", "Voltage"], default="Current density",
                group="Endpoint",
            ),
            ParameterSpec(
                "current_limit", "Maximum |j|", kind="float", default=1.0,
                minimum=1e-12, step=0.1, suffix="A/cm²", group="Endpoint",
                show_if=("limit_mode", "Current density"),
            ),
            ParameterSpec(
                "oer_voltage_limit", "OER terminal potential", kind="float",
                default=2.0, step=0.05, suffix="V vs RHE", group="Endpoint",
                show_if=[("limit_mode", "Voltage"), ("mode", "OER")],
            ),
            ParameterSpec(
                "her_voltage_limit", "HER terminal potential", kind="float",
                default=-0.5, step=0.05, suffix="V vs RHE", group="Endpoint",
                show_if=[("limit_mode", "Voltage"), ("mode", "HER")],
            ),
            ParameterSpec(
                "cell_voltage_limit", "Terminal cell voltage", kind="float",
                default=2.0, minimum=0.0, step=0.05, suffix="V", group="Endpoint",
                show_if=[("limit_mode", "Voltage"), ("mode", "Two-electrode")],
            ),
        ]

    def load_oer_preset(self, files: list[Path], params: dict) -> dict:
        preset = OER_PRESETS.get(str(params.get("oer_preset", "")))
        return {} if preset is None else {
            "oer_b": preset.b,
            "oer_i0": preset.i0,
            "temperature": preset.reference_temperature_k,
        }

    def load_her_preset(self, files: list[Path], params: dict) -> dict:
        preset = HER_PRESETS.get(str(params.get("her_preset", "")))
        return {} if preset is None else {
            "her_b": preset.b,
            "her_i0": preset.i0,
            "temperature": preset.reference_temperature_k,
        }

    def run(self, files: list[Path], params: dict, extra=None) -> ToolResult:
        mode = str(params.get("mode", "OER"))
        limit_mode = str(params.get("limit_mode", "Current density"))
        oer_params = None
        her_params = None
        if mode in {"OER", "Two-electrode"}:
            oer_params = {
                "b": params["oer_b"], "i0": params["oer_i0"],
                "rcl": params["oer_rcl"],
            }
        if mode in {"HER", "Two-electrode"}:
            her_params = {
                "b": params["her_b"], "i0": params["her_i0"],
                "rcl": params["her_rcl"],
            }

        voltage_limit = None
        if limit_mode == "Voltage":
            voltage_limit = {
                "OER": params.get("oer_voltage_limit"),
                "HER": params.get("her_voltage_limit"),
                "Two-electrode": params.get("cell_voltage_limit"),
            }[mode]

        generated = generate_lsv(
            mode=mode,
            temperature_k=params["temperature"],
            hfr=params["hfr"],
            limit_mode=limit_mode,
            current_limit=params.get("current_limit"),
            voltage_limit=voltage_limit,
            oer_params=oer_params,
            her_params=her_params,
        )
        metadata = generated["metadata"]
        data_table = _generated_table(generated, metadata["mode"])
        endpoint_table = _parameter_table(params, generated, oer_params, her_params)

        figures = {
            "Generated LSV": _plot_polarization(generated, metadata["mode"]),
            "Potential construction": _plot_construction(generated, metadata["mode"]),
            "Losses and utilization": _plot_diagnostics(generated, metadata["mode"]),
        }
        potential_name = (
            "Terminal cell voltage (V)"
            if metadata["mode"] == "two-electrode"
            else "Terminal potential (V vs RHE)"
        )
        scalars = {
            "Configuration": mode,
            "Points": metadata["n_points"],
            "Starting |j| (A/cm²)": metadata["current_start"],
            "Terminal |j| (A/cm²)": metadata["current_end"],
            potential_name: metadata["potential_end"],
            "E_rev (V)": metadata["e_rev"],
        }

        preset_notes: list[str] = []
        used_literature_example = False
        if oer_params is not None:
            note, used = _preset_note(
                "OER", str(params.get("oer_preset", CUSTOM_PRESET)),
                oer_params, OER_PRESETS,
            )
            preset_notes.append(note)
            used_literature_example |= used
        if her_params is not None:
            note, used = _preset_note(
                "HER", str(params.get("her_preset", CUSTOM_PRESET)),
                her_params, HER_PRESETS,
            )
            preset_notes.append(note)
            used_literature_example |= used

        warnings = []
        if used_literature_example:
            warnings.append(
                "Catalyst presets are editable, condition-specific examples and must "
                "not be interpreted as universal material constants."
            )
        if metadata["current_start"] > 1e-6:
            warnings.append(
                "The curve starts at the largest active exchange current density "
                f"({metadata['current_start']:.6g} A/cm²), above the nominal "
                "10⁻⁶ A/cm² floor, so the one-direction Tafel losses stay nonnegative."
            )

        return ToolResult(
            figures=figures,
            tables={
                "Generated LSV data": data_table,
                "Parameters and endpoint": endpoint_table,
            },
            scalars=scalars,
            text_blocks={"Preset provenance": "\n\n".join(preset_notes)},
            warnings=warnings,
            payload={"generated": generated},
        )

    def exportable_artifacts(self, result: ToolResult) -> list[Artifact]:
        data = result.tables["Generated LSV data"].drop(columns=[
            "Current Magnitude (A/cm²)", "E_rev (V)", "Total Modeled Loss (V)",
        ]).rename(columns={"Signed Current Density (A/cm²)": "Current density (A/cm²)"})
        potential = "Cell Voltage (V)" if "Cell Voltage (V)" in data else "Potential (V vs RHE)"
        data = data[[potential, *[column for column in data if column != potential]]]
        sheets = {"Generated LSV": data,
                  "Parameters": result.tables["Parameters and endpoint"].copy()}
        artifacts = [Artifact(
            name=f"{name} (CSV)", kind="csv",
            default_filename=f"lsv_generator_{name.lower().replace(' ', '_')}.csv", payload=table,
        ) for name, table in sheets.items()]
        artifacts.append(Artifact(
            name="Generated LSV results (Excel)", kind="xlsx",
            default_filename="lsv_generator_results.xlsx", payload=sheets,
        ))
        artifacts.extend(item for item in super().exportable_artifacts(result) if item.kind == "png")
        return artifacts


def _generated_table(generated: dict, mode: str) -> pd.DataFrame:
    potential_header = (
        "Cell Voltage (V)" if mode == "two-electrode" else "Potential (V vs RHE)"
    )
    columns = {
        "Current Magnitude (A/cm²)": generated["current_magnitude"],
        "Signed Current Density (A/cm²)": generated["signed_current"],
        potential_header: generated["potential"],
        "E_rev (V)": generated["e_rev"],
    }
    if mode in {"oer", "two-electrode"}:
        columns.update({
            "η_kin,OER (V)": generated["eta_kin_oer"],
            "η_RCL,OER (V)": generated["eta_rcl_oer"],
            "OER Catalyst Utilization U_J (-)": generated["utilization_oer"],
        })
    if mode in {"her", "two-electrode"}:
        columns.update({
            "η_kin,HER (V)": generated["eta_kin_her"],
            "η_RCL,HER (V)": generated["eta_rcl_her"],
            "HER Catalyst Utilization U_J (-)": generated["utilization_her"],
        })
    columns.update({
        "η_ohm (V)": generated["eta_ohm"],
        "Total Modeled Loss (V)": generated["eta_total"],
    })
    return pd.DataFrame(columns)


def _parameter_table(params: dict, generated: dict, oer_params, her_params) -> pd.DataFrame:
    m = generated["metadata"]
    rows: list[tuple[str, object, str]] = [
        ("Configuration", params.get("mode"), ""),
        ("Temperature", m["temperature_k"], "K"),
        ("Series HFR", m["hfr"], "Ω·cm²"),
        ("Endpoint mode", params.get("limit_mode"), ""),
        ("Requested endpoint", m["requested_endpoint"],
         "A/cm²" if m["limit_mode"] == "current" else "V"),
        ("Starting current magnitude", m["current_start"], "A/cm²"),
        ("Terminal current magnitude", m["current_end"], "A/cm²"),
        ("Starting potential / cell voltage", m["potential_start"], "V"),
        ("Terminal potential / cell voltage", m["potential_end"], "V"),
        ("Reversible potential", m["e_rev"], "V"),
        ("Generated points", m["n_points"], ""),
    ]
    if oer_params is not None:
        rows.extend([
            ("OER preset selection", params.get("oer_preset"), ""),
            ("OER Tafel slope", float(oer_params["b"]), "V/dec"),
            ("OER exchange current density", float(oer_params["i0"]), "A/cm²"),
            ("OER R_CL", float(oer_params["rcl"]), "Ω·cm²"),
        ])
    if her_params is not None:
        rows.extend([
            ("HER preset selection", params.get("her_preset"), ""),
            ("HER Tafel slope", float(her_params["b"]), "V/dec"),
            ("HER exchange current density", float(her_params["i0"]), "A/cm²"),
            ("HER R_CL", float(her_params["rcl"]), "Ω·cm²"),
        ])
    return pd.DataFrame(rows, columns=["Quantity", "Value", "Unit"])


def _preset_note(
    electrode_name: str,
    selected_name: str,
    current_params: dict,
    presets: dict[str, CatalystPreset],
) -> tuple[str, bool]:
    preset = presets.get(selected_name)
    if preset is None:
        return (
            f"{electrode_name}: Custom parameters\n"
            "No literature preset is attached to these values.",
            False,
        )
    matches = np.isclose(float(current_params["b"]), preset.b, rtol=1e-9, atol=0.0)
    matches &= np.isclose(
        float(current_params["i0"]), preset.i0, rtol=1e-9, atol=0.0
    )
    state = "values match the preset" if matches else "values modified from the preset"
    return (
        f"{electrode_name}: {selected_name} ({state})\n"
        f"  Literature pair: b = {preset.b:.6g} V/dec; "
        f"i₀ = {preset.i0:.6g} A/cm²\n"
        f"  Reference temperature: {preset.reference_temperature_k:.2f} K\n"
        f"  Conditions: {preset.conditions}\n"
        f"  Source: {preset.source}",
        True,
    )


def _plot_polarization(generated: dict, mode: str) -> Figure:
    fig = Figure(figsize=(7.5, 5.5))
    ax = fig.add_subplot(111)
    potential = generated["potential"]
    current_ma = generated["signed_current"] * 1e3
    ax.plot(potential, current_ma, color=theme.ACCENT, linewidth=2.2)
    ax.scatter(
        [potential[-1]], [current_ma[-1]], s=46, zorder=4,
        color=theme.TEXT, edgecolor="white", linewidth=0.8, label="Endpoint",
    )
    ax.set_xlabel(
        "Cell voltage  (V)" if mode == "two-electrode"
        else "Potential  (V vs RHE)"
    )
    ax.set_ylabel(r"Current density  $j$  (mA cm$^{-2}$)")
    ax.set_title({
        "oer": "Generated OER polarization curve",
        "her": "Generated HER polarization curve",
        "two-electrode": "Generated two-electrode polarization curve",
    }[mode])
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _plot_construction(generated: dict, mode: str) -> Figure:
    fig = Figure(figsize=(7.5, 5.5))
    ax = fig.add_subplot(111)
    j = generated["current_magnitude"]
    e_rev = generated["e_rev"]

    if mode == "oer":
        stages = [
            (e_rev, r"$E_\mathrm{rev}$"),
            (e_rev + generated["eta_kin_oer"], r"+ $\eta_\mathrm{kin}$"),
            (
                e_rev + generated["eta_kin_oer"] + generated["eta_rcl_oer"],
                r"+ $\eta_\mathrm{RCL}$",
            ),
            (generated["potential"], r"+ $\eta_\mathrm{ohm}$ (total)"),
        ]
    elif mode == "her":
        stages = [
            (e_rev, r"$E_\mathrm{rev}$"),
            (-generated["eta_kin_her"], r"− $\eta_\mathrm{kin}$"),
            (
                -(generated["eta_kin_her"] + generated["eta_rcl_her"]),
                r"− $\eta_\mathrm{RCL}$",
            ),
            (generated["potential"], r"− $\eta_\mathrm{ohm}$ (total)"),
        ]
    else:
        s1 = e_rev + generated["eta_kin_oer"]
        s2 = s1 + generated["eta_rcl_oer"]
        s3 = s2 + generated["eta_kin_her"]
        s4 = s3 + generated["eta_rcl_her"]
        stages = [
            (e_rev, r"$E_\mathrm{rev,cell}$"),
            (s1, r"+ $\eta_\mathrm{kin,OER}$"),
            (s2, r"+ $\eta_\mathrm{RCL,OER}$"),
            (s3, r"+ $\eta_\mathrm{kin,HER}$"),
            (s4, r"+ $\eta_\mathrm{RCL,HER}$"),
            (generated["potential"], r"+ $\eta_\mathrm{ohm}$ (total)"),
        ]

    colors = theme.SERIES_COLORS
    for index, (values, label) in enumerate(stages):
        style = {"linestyle": (0, (4, 3))} if index == 0 else {}
        ax.plot(j, values, color=colors[index % len(colors)], label=label, **style)
        if index > 0:
            ax.fill_between(
                j, stages[index - 1][0], values,
                color=colors[index % len(colors)], alpha=0.10, linewidth=0,
            )
    ax.set_xscale("log")
    ax.set_xlabel(r"Current-density magnitude  $|j|$  (A cm$^{-2}$)")
    ax.set_ylabel("Cell voltage  (V)" if mode == "two-electrode"
                  else "Potential  (V vs RHE)")
    ax.set_title("Cumulative potential construction")
    ax.legend(loc="best", ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


def _plot_diagnostics(generated: dict, mode: str) -> Figure:
    fig = Figure(figsize=(8.0, 7.0))
    loss_ax = fig.add_subplot(211)
    util_ax = fig.add_subplot(212, sharex=loss_ax)
    j = generated["current_magnitude"]

    if mode == "oer":
        losses = [
            generated["eta_kin_oer"] * 1e3,
            generated["eta_rcl_oer"] * 1e3,
            generated["eta_ohm"] * 1e3,
        ]
        labels = [r"$\eta_\mathrm{kin}$", r"$\eta_\mathrm{RCL}$", r"$\eta_\mathrm{ohm}$"]
        colors = [theme.OVERPOT_COLORS[k] for k in ("eta_kin", "eta_rcl", "eta_ohm")]
    elif mode == "her":
        losses = [
            generated["eta_kin_her"] * 1e3,
            generated["eta_rcl_her"] * 1e3,
            generated["eta_ohm"] * 1e3,
        ]
        labels = [r"$\eta_\mathrm{kin}$", r"$\eta_\mathrm{RCL}$", r"$\eta_\mathrm{ohm}$"]
        colors = [theme.OVERPOT_COLORS[k] for k in ("eta_kin", "eta_rcl", "eta_ohm")]
    else:
        losses = [
            generated["eta_kin_oer"] * 1e3,
            generated["eta_rcl_oer"] * 1e3,
            generated["eta_kin_her"] * 1e3,
            generated["eta_rcl_her"] * 1e3,
            generated["eta_ohm"] * 1e3,
        ]
        labels = [
            r"$\eta_\mathrm{kin,OER}$", r"$\eta_\mathrm{RCL,OER}$",
            r"$\eta_\mathrm{kin,HER}$", r"$\eta_\mathrm{RCL,HER}$",
            r"$\eta_\mathrm{ohm}$",
        ]
        colors = [
            theme.SERIES_COLORS[1], theme.SERIES_COLORS[3],
            theme.SERIES_COLORS[4], theme.SERIES_COLORS[0],
            theme.SERIES_COLORS[5],
        ]
    loss_ax.stackplot(j, *losses, labels=labels, colors=colors, alpha=0.78)
    loss_ax.plot(j, generated["eta_total"] * 1e3, color=theme.TEXT,
                 linewidth=1.5, label="Total")
    loss_ax.set_xscale("log")
    loss_ax.set_ylabel("Modeled loss  (mV)")
    loss_ax.set_title("Loss contributions and catalyst utilization")
    loss_ax.legend(loc="upper left", ncol=2, fontsize=8)

    if mode in {"oer", "two-electrode"}:
        util_ax.plot(j, generated["utilization_oer"],
                     color=theme.SERIES_COLORS[3], label="OER $U_J$")
    if mode in {"her", "two-electrode"}:
        util_ax.plot(j, generated["utilization_her"],
                     color=theme.SERIES_COLORS[4], label="HER $U_J$")
    util_ax.set_xscale("log")
    util_ax.set_xlabel(r"Current-density magnitude  $|j|$  (A cm$^{-2}$)")
    util_ax.set_ylabel(r"Catalyst utilization  $U_J$")
    util_ax.set_ylim(0, 1.05)
    util_ax.legend(loc="best")
    fig.tight_layout()
    return fig
