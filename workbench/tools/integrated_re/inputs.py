"""Independent, persistent upload panels for the integrated-RE measurement roles."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget

from workbench.core.tool_base import ParameterSpec
from workbench.widgets.file_input import FileInputWidget
from workbench.widgets.parameter_panel import ParameterPanel


ROLES = {
    "anode": "Anode LSV",
    "cathode": "Cathode LSV",
    "cell": "Two-electrode cell LSV",
    "calibration": "Three-electrode calibration LSV",
}

DEFAULT_CALIBRATION_FILE = (Path(__file__).resolve().parents[2] / "Test_files"
                            / "LSV_PtRuC_ircorrected.txt")


def role_parameter_schema(role: str) -> list[ParameterSpec]:
    specs = [
        ParameterSpec(f"{role}_area", "Electrode area", default=0.0,
                      minimum=0.0, step=0.01, decimals=4, suffix="cm²",
                      tooltip="Required for absolute-current LSV data or EIS in raw Ω. "
                              "Not needed for current-density LSVs and Ω·cm² EIS.",
                      group="Input settings"),
        ParameterSpec(f"{role}_sign", "Current sign", kind="choice", default="Auto",
                      choices=["Auto", "As recorded", "Invert"], group="Input settings",
                      tooltip="Auto orients the largest-|j| point to this electrode's reaction. "
                              "Invert reverses the entire source current, including baseline signs."),
    ]
    if role != "calibration":
        scale = "cell voltage" if role == "cell" else "unshifted RHE potential"
        specs += [
            ParameterSpec(f"{role}_use_window", "Limit comparison window", kind="bool",
                          default=False, group="Input settings",
                          tooltip=f"Select the reconstruction branch using {scale}. "
                                  "The full curve remains visible in the plot."),
            ParameterSpec(f"{role}_e_min", "Window E min", default=0.0 if role == "cell" else -2.0,
                          step=0.05, decimals=4, suffix="V", group="Input settings",
                          show_if=(f"{role}_use_window", True)),
            ParameterSpec(f"{role}_e_max", "Window E max", default=5.0 if role == "cell" else 3.0,
                          step=0.05, decimals=4, suffix="V", group="Input settings",
                          show_if=(f"{role}_use_window", True)),
            ParameterSpec(f"{role}_hfr", "Average HFR", kind="str",
                          default="Not calculated", read_only=True, suffix="Ω·cm²",
                          group="HFR from EIS",
                          tooltip="Calculated from this measurement's EIS files. "
                                  "Multiple spectra contribute only to the average HFR."),
        ]
    return specs


class IntegratedReInputs(QWidget):
    def __init__(self, parent=None, on_change=None):
        super().__init__(parent)
        self._on_change = on_change
        self._calibration_active = False
        self._calibration_electrode = "Cathode"
        self._calibration_default_initialized = False
        self.inputs: dict[str, FileInputWidget] = {}
        self.eis_inputs: dict[str, FileInputWidget] = {}
        self.panels: dict[str, ParameterPanel] = {}
        self.eis_panels: dict[str, ParameterPanel] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        heading = QLabel("LSV inputs")
        heading.setObjectName("SectionTitle")
        layout.addWidget(heading)
        self.calibration_notice = QLabel()
        self.calibration_notice.setObjectName("Hint")
        self.calibration_notice.setWordWrap(True)
        self.calibration_notice.setVisible(False)
        layout.addWidget(self.calibration_notice)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        for role, title in ROLES.items():
            page = QWidget()
            body = QVBoxLayout(page)
            label = QLabel(title + (" · optional" if role == "cell" else " · required"))
            label.setWordWrap(True)
            body.addWidget(label)
            picker = FileInputWidget(multiple=False, accepts_directory=False)
            picker.btn_add.setText(f"Add {role} LSV…" if role != "calibration" else "Add 3E LSV…")
            picker.list.setMaximumHeight(120)
            picker.hint.setText("Select one already iR-corrected 3E LSV, or drop it here."
                                if role == "calibration" else
                                f"Select one {title.lower()} file, or drop it here.")
            picker.filesChanged.connect(self._changed)
            body.addWidget(picker)
            panel = ParameterPanel(role_parameter_schema(role), on_change=self._changed)
            body.addWidget(panel)
            self.inputs[role] = picker
            self.panels[role] = panel
            self.tabs.addTab(page, {"anode": "Anode", "cathode": "Cathode",
                                    "cell": "Cell (optional)", "calibration": "3E calibration"}[role])
        self.tabs.setTabVisible(3, False)

        self.eis_heading = QLabel("EIS inputs for HFR · optional")
        self.eis_heading.setObjectName("SectionTitle")
        layout.addWidget(self.eis_heading)
        self.eis_tabs = QTabWidget()
        layout.addWidget(self.eis_tabs)
        for role in ("anode", "cathode", "cell"):
            title = f"{role.capitalize()} EIS"
            page = QWidget()
            body = QVBoxLayout(page)
            label = QLabel(f"{title} spectra · separate HFR average")
            label.setWordWrap(True)
            body.addWidget(label)
            picker = FileInputWidget(multiple=True, accepts_directory=False)
            picker.btn_add.setText(f"Add {role} EIS…")
            picker.list.setMaximumHeight(120)
            picker.hint.setText(f"Drop {role} EIS spectra here. Only these files contribute to {role} HFR.")
            picker.filesChanged.connect(self._changed)
            body.addWidget(picker)
            self.eis_inputs[role] = picker
            hfr_spec = next(spec for spec in role_parameter_schema(role) if spec.key == f"{role}_hfr")
            panel = ParameterPanel([hfr_spec])
            body.addWidget(panel)
            self.eis_panels[role] = panel
            help_text = QLabel(
                f"Multiple {role} spectra are used only to calculate average HFR. "
                f"For raw Ω data, enter the electrode area in the {role.capitalize()} LSV input settings."
            )
            help_text.setObjectName("Hint")
            help_text.setWordWrap(True)
            body.addWidget(help_text)
            self.eis_tabs.addTab(page, title)

    def _changed(self, *_):
        if self._on_change:
            self._on_change()

    def set_parameters(self, params: dict) -> None:
        self._calibration_active = params.get("correction_mode") in {"3E onset", "3E peak"}
        electrode = params.get("calibration_electrode", self._calibration_electrode)
        self._calibration_electrode = "Anode" if electrode == "Anode" else "Cathode"
        self._update_calibration_notice()
        self.tabs.setTabVisible(3, self._calibration_active)
        if self._calibration_active and not self._calibration_default_initialized:
            # Preserve later replacements and clears across correction toggles.
            self._calibration_default_initialized = True
            picker = self.inputs["calibration"]
            if not picker.files():
                if DEFAULT_CALIBRATION_FILE.is_file():
                    picker.set_files([DEFAULT_CALIBRATION_FILE])
                else:
                    picker.hint.setText("Bundled PtRuC calibration is unavailable. Add a 3E LSV.")

    def _update_calibration_notice(self) -> None:
        electrode = self._calibration_electrode
        self.calibration_notice.setText(
            "The default PtRuC calibration LSV is already iR-corrected. "
            f"{electrode} EIS is required; its average HFR is applied to the measured "
            f"{electrode.lower()} LSV before 3E calibration."
        )
        self.calibration_notice.setVisible(self._calibration_active)
        detail = f"{electrode} required for calibration" if self._calibration_active else "optional"
        self.eis_heading.setText(f"EIS inputs for HFR · {detail}")

    def values(self) -> dict:
        values = {}
        settings = {}
        for role, picker in self.inputs.items():
            paths = picker.files()
            values[role] = str(paths[0]) if paths else ""
            settings.update(self.panels[role].values())
        values["parameters"] = settings
        values["eis"] = {role: [str(path) for path in picker.files()]
                         for role, picker in self.eis_inputs.items()}
        return values

    def files(self) -> list[Path]:
        lsv_paths = [path for role, picker in self.inputs.items()
                     if role != "calibration" or self._calibration_active
                     for path in picker.files()]
        return lsv_paths + [path for picker in self.eis_inputs.values()
                            for path in picker.files()]

    def project_files(self) -> list[Path]:
        """Include retained inputs even when calibration is temporarily hidden."""
        return [path for picker in (*self.inputs.values(), *self.eis_inputs.values())
                for path in picker.files()]

    def save_state(self) -> dict:
        return {
            "inputs": {role: picker.files() for role, picker in self.inputs.items()},
            "eis_inputs": {role: picker.files() for role, picker in self.eis_inputs.items()},
            "panels": {role: panel.save_state() for role, panel in self.panels.items()},
            "eis_panels": {role: panel.save_state() for role, panel in self.eis_panels.items()},
            "calibration_active": self._calibration_active,
            "calibration_electrode": self._calibration_electrode,
            "calibration_default_initialized": self._calibration_default_initialized,
            "input_tab": self.tabs.currentIndex(),
            "eis_tab": self.eis_tabs.currentIndex(),
        }

    def restore_state(self, state: dict) -> None:
        """Restore independent role uploads, settings, readouts and navigation."""
        # Older projects already contain their chosen inputs, even if empty.
        self._calibration_default_initialized = bool(state.get("calibration_default_initialized", True))
        for key, pickers in (("inputs", self.inputs), ("eis_inputs", self.eis_inputs)):
            selections = state.get(key, {})
            for role, picker in pickers.items():
                picker.set_files(selections.get(role, []))
        for key, panels in (("panels", self.panels), ("eis_panels", self.eis_panels)):
            saved = state.get(key, {})
            for role, panel in panels.items():
                panel.restore_state(saved.get(role, {}))
        self._calibration_active = bool(state.get("calibration_active", False))
        self._calibration_electrode = "Anode" if state.get("calibration_electrode") == "Anode" else "Cathode"
        self._update_calibration_notice()
        self.tabs.setTabVisible(3, self._calibration_active)
        for tabs, key in ((self.tabs, "input_tab"), (self.eis_tabs, "eis_tab")):
            index = state.get(key, 0)
            if isinstance(index, int) and 0 <= index < tabs.count() and tabs.isTabVisible(index):
                tabs.setCurrentIndex(index)

    def set_result_parameters(self, updates: dict, annotations: dict) -> None:
        """Populate calculated readouts without changing measurement settings."""
        for role in self.eis_inputs:
            key = f"{role}_hfr"
            for panel in (self.panels[role], self.eis_panels[role]):
                if key in updates:
                    panel.set_values({key: updates[key]})
                panel.set_annotations({key: annotations.get(key, "")})
