"""Compact experimental and fit-range controls for batch decoupling."""
from copy import deepcopy

from workbench.core.tool_base import ParameterSpec
from workbench.tools.lsv.tool import LsvTool


def ccd_parameters() -> list[ParameterSpec]:
    """Share physical inputs with Decoupling and expose the CCD threshold."""
    inputs = [
        ParameterSpec(
            "input_type", "Input type", kind="choice", default="LSV",
            choices=["LSV", "CV"], group="Input data",
            tooltip="Each LSV is decoupled independently. CV inputs can contribute every cycle.",
        ),
        ParameterSpec(
            "cv_scan", "Scan", kind="choice", default="Forward",
            choices=["Forward", "Back"], group="Input data",
            show_if=("input_type", "CV"),
            tooltip="Forward follows the initial sweep direction; Back is its return scan. "
                    "CCD is evaluated toward increasing electrochemical drive on the chosen branch.",
        ),
        ParameterSpec(
            "cv_cycle_mode", "Cycles", kind="choice", default="All",
            choices=["All", "Last", "Specific"], group="Input data",
            show_if=("input_type", "CV"),
            tooltip="All analyzes every cycle. Last resolves the last acquired cycle separately "
                    "for each file. Specific uses the source cycle number below.",
        ),
        ParameterSpec(
            "cv_cycle", "Cycle number", kind="int", default=1, minimum=1,
            maximum=1_000_000, group="Input data",
            show_if=[("input_type", "CV"), ("cv_cycle_mode", "Specific")],
            tooltip="Labelled CV files use their source cycle numbers; unlabelled cycles start at 1.",
        ),
        ParameterSpec(
            "current_branch", "Current sign", kind="choice", default="Auto",
            choices=["Auto", "Positive", "Negative"], group="Input data",
            tooltip="Auto uses the dominant current sign in each scan. CCD currents are "
                    "reported as current-density magnitudes.",
        ),
    ]
    omitted = {"input_type", "average_range", "cv_scan", "cv_cycle", "x_log", "swap_axes",
               "include_ccd", "ccd_fit_mode", "ccd_j_lower", "ccd_j_upper", "ccd_threshold_mv"}
    common = deepcopy([spec for spec in LsvTool().parameter_schema() if spec.key not in omitted])
    for spec in common:
        if spec.key == "include_rcl":
            spec.default = False
        elif spec.key in {"i_lower", "i_upper"}:
            spec.tooltip = ("Manual current-density search limits for each scan's contiguous "
                            "stable Tafel interval. The selected region must span at least "
                            "0.5 decades and satisfy the internal slope-stability checks.")
        elif spec.key == "target_i":
            spec.tooltip = ("Shared target current density for per-curve breakdown points. "
                            "A curve outside this range keeps its decomposition and CCD results, "
                            "but has no breakdown point.")
    residual = [
        ParameterSpec(
            "ccd_fit_mode", "Fit range", kind="choice", default="Auto",
            choices=["Auto", "Manual"], group="Residual baseline",
            tooltip="Auto finds a contiguous stable residual-overpotential/current region above "
                    "the Tafel search range. Manual uses the current-density limits below.",
        ),
        ParameterSpec(
            "ccd_j_lower", "Fit from", default=1.0, minimum=0.0, step=.1,
            decimals=4, suffix="A/cm²", group="Residual baseline",
            show_if=("ccd_fit_mode", "Manual"),
            tooltip="Lower current-density magnitude for the residual baseline fit.",
        ),
        ParameterSpec(
            "ccd_j_upper", "Fit to", default=3.0, minimum=0.0, step=.1,
            decimals=4, suffix="A/cm²", group="Residual baseline",
            show_if=("ccd_fit_mode", "Manual"),
            tooltip="Upper current-density magnitude for the residual baseline fit.",
        ),
    ]
    detection = [
        ParameterSpec(
            "ccd_threshold_mv", "Threshold", default=50.0, minimum=.1,
            maximum=1000.0, step=5.0, decimals=1, suffix="mV", group="CCD",
            tooltip="Excess above the fitted residual baseline required to detect CCD. "
                    "It must remain above this threshold over 100 mV of measured drive. "
                    "A higher threshold identifies a later onset.",
        ),
    ]
    target = [spec for spec in common if spec.key == "target_i"]
    common = [spec for spec in common if spec.key != "target_i"]
    return [*inputs, *common, *residual, *detection, *target]
