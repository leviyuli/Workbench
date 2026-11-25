"""CCD controls with a short everyday form and optional processing settings."""
from workbench.core.tool_base import ParameterSpec


def ccd_parameters() -> list[ParameterSpec]:
    advanced = ("show_advanced", True)
    return [
        ParameterSpec(
            "electrode_area", "Electrode area", default=1.0, minimum=0.0,
            step=.01, decimals=4, suffix="cm²", group="Analysis",
            tooltip="Default electrode area: 1 cm². Used to convert absolute current (A) "
                    "to current density. Data already in A/cm² are used directly.",
        ),
        ParameterSpec(
            "baseline_mode", "Fit range", kind="choice", default="Manual",
            choices=["Manual", "Auto"], group="Analysis",
            tooltip="The straight middle part of each forward scan defines the reference line. "
                    "Manual uses the voltage range below; Auto finds a straight region for each cycle. "
                    "CCD is where the curve stays above this line by the selected voltage "
                    "increase (20 mV by default). Check the plotted fit.",
        ),
        ParameterSpec(
            "baseline_lower", "Fit from", default=1.90, step=.05,
            decimals=3, suffix="V", group="Analysis",
            show_if=("baseline_mode", "Manual"),
            tooltip="Start of the straight middle region in the measured forward scan. "
                    "Review the plotted fit for each scan before using this range. For a negative-going "
                    "scan, enter −E (for example, 0.2 to 0.5 V).",
        ),
        ParameterSpec(
            "baseline_upper", "Fit to", default=2.45, step=.05,
            decimals=3, suffix="V", group="Analysis",
            show_if=("baseline_mode", "Manual"),
            tooltip="End of the straight middle region. The range is applied separately "
                    "to the forward scan of every CV cycle.",
        ),
        ParameterSpec(
            "show_raw", "Show raw scans", kind="bool", default=True, group="Display",
            tooltip="Overlay the original forward-scan points behind the processed curves.",
        ),
        ParameterSpec(
            "show_advanced", "Show advanced settings", kind="bool", default=False,
            group="Display", tooltip="Adjust current selection, resistance correction, "
                    "noise filtering and plateau detection. Defaults suit the supplied data.",
        ),
        ParameterSpec(
            "current_branch", "Current sign", kind="choice", default="Auto",
            choices=["Auto", "Positive", "Negative"], group="Advanced settings",
            show_if=advanced,
            tooltip="Auto uses the dominant current sign in each forward scan. "
                    "Reported CCD and limiting current densities are positive magnitudes.",
        ),
        ParameterSpec(
            "reference_hfr", "Reference HFR", default=.12, minimum=0.0,
            step=.01, decimals=5, suffix="Ω·cm²", group="Advanced settings",
            show_if=advanced,
            tooltip="High-frequency resistance measured before the current rises. "
                    "Subtract current density × this resistance from voltage. A constant "
                    "value shifts both the curve and fitted line, preserving the CCD current.",
        ),
        ParameterSpec(
            "ccd_threshold_mv", "CCD voltage increase", default=20.0,
            minimum=.1, maximum=2000, step=10, suffix="mV", decimals=1,
            group="Advanced settings",
            show_if=advanced,
            tooltip="First sustained voltage increase above the fitted middle-region "
                    "reference line after HFR correction. Default: 20 mV. Compare 50 or "
                    "100 mV to assess sensitivity; keep the fit range and processing fixed.",
        ),
        ParameterSpec(
            "bin_width_mv", "Voltage averaging step", default=10.0, minimum=.1,
            maximum=1000, step=1, decimals=1, suffix="mV", group="Advanced settings",
            show_if=advanced,
            tooltip="Group nearby voltages into steps of this size and use the median "
                    "current in each step to reduce isolated spikes.",
        ),
        ParameterSpec(
            "smoothing_window", "Smoothing width", kind="int", default=15,
            minimum=5, maximum=501, step=2, suffix="steps", group="Advanced settings",
            show_if=advanced,
            tooltip="Number of voltage steps used to smooth the curve. Use an odd "
                    "number of at least 5; larger values remove more detail. "
                    "The filter is Savitzky–Golay, polynomial order 3.",
        ),
        ParameterSpec(
            "persistence_mv", "Confirm CCD over", default=100.0, minimum=10,
            maximum=2000, step=10, decimals=0, suffix="mV", group="Advanced settings",
            show_if=advanced,
            tooltip="The voltage increase must stay above the selected CCD threshold over this scan span "
                    "so a short spike does not trigger CCD.",
        ),
        ParameterSpec(
            "plateau_window_mv", "Plateau must span", default=200.0,
            minimum=50, maximum=3000, step=25, decimals=0, suffix="mV",
            group="Advanced settings", show_if=advanced,
            tooltip="Minimum voltage span of a nearly flat high-current region after CCD. "
                    "Limiting current is the median current in that plateau. "
                    "If no plateau is observed, the result stays unresolved.",
        ),
        ParameterSpec(
            "plateau_slope_pct", "Plateau slope limit", default=15.0,
            minimum=.1, maximum=99, step=1, decimals=1, suffix="% of fit slope",
            group="Advanced settings", show_if=advanced,
            tooltip="Maximum absolute plateau slope as a percentage of the middle-region "
                    "fit slope. A lower value requires a flatter plateau.",
        ),
        ParameterSpec(
            "plateau_spread_pct", "Plateau variation limit", default=5.0,
            minimum=.1, maximum=99, step=.5, decimals=1, suffix="%",
            group="Advanced settings", show_if=advanced,
            tooltip="Maximum spread of the central 80% of plateau current values "
                    "as a percentage of the plateau level. Lower values reject noisier plateaus.",
        ),
    ]
