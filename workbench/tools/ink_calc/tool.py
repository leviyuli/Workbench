from __future__ import annotations

from typing import Callable

import pandas as pd
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from workbench.core.tool_base import Artifact, ParameterSpec, Tool, ToolResult
from workbench.tools.ink_calc.logic import calculate_ink_recipe


class _EditableTable(QWidget):
    """A small add/remove table backed by a QTableWidget."""

    changed = Signal()

    def __init__(self, headers: list[str], default_row: list[str], add_label: str):
        super().__init__()
        self._default_row = default_row
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.table = QTableWidget(0, len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMaximumHeight(170)
        self.table.itemChanged.connect(lambda *_: self.changed.emit())
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        add = QPushButton(add_label)
        add.clicked.connect(self.add_row)
        rem = QPushButton("Remove")
        rem.clicked.connect(self._remove_selected)
        btn_row.addWidget(add)
        btn_row.addWidget(rem)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

    def add_row(self, values: list[str] | None = None) -> None:
        # QPushButton.clicked supplies its checked state to this optional arg.
        if isinstance(values, bool):
            values = None
        values = values if values is not None else self._default_row
        previously_blocked = self.table.blockSignals(True)
        r = self.table.rowCount()
        try:
            self.table.insertRow(r)
            for c, val in enumerate(values[:self.table.columnCount()]):
                self.table.setItem(r, c, QTableWidgetItem(str(val)))
        finally:
            self.table.blockSignals(previously_blocked)
        self.changed.emit()

    def _remove_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedItems()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)
        if rows:
            self.changed.emit()

    def set_rows(self, rows: list[list[str]]) -> None:
        """Restore raw cell text, retaining unfinished and blank entries."""
        if rows == self.rows():
            return
        previously_blocked = self.table.blockSignals(True)
        try:
            self.table.setRowCount(len(rows))
            for r, row in enumerate(rows):
                for c in range(self.table.columnCount()):
                    value = row[c] if c < len(row) else ""
                    self.table.setItem(r, c, QTableWidgetItem(str(value)))
        finally:
            self.table.blockSignals(previously_blocked)
        self.changed.emit()

    def rows(self) -> list[list[str]]:
        out = []
        for r in range(self.table.rowCount()):
            row = []
            for c in range(self.table.columnCount()):
                item = self.table.item(r, c)
                row.append(item.text() if item else "")
            out.append(row)
        return out


class IonomerExtrasWidget(QWidget):
    def __init__(self, parent: QWidget | None = None,
                 on_change: Callable[[], None] | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        lab1 = QLabel("Ionomers")
        lab1.setObjectName("SectionLabel")
        layout.addWidget(lab1)
        self.ionomers = _EditableTable(
            ["Ratio (%)", "Conc (wt%)", "Dens (g/mL)"],
            ["0", "5.0", "0.78"], "+ Add Ionomer",
        )
        self.ionomers.add_row()
        layout.addWidget(self.ionomers)

        lab2 = QLabel("Other materials (e.g. PTFE, carbon)")
        lab2.setObjectName("SectionLabel")
        layout.addWidget(lab2)
        self.extras = _EditableTable(
            ["Material", "Ratio (% of solids)"],
            ["Material", "0"], "+ Add Material",
        )
        layout.addWidget(self.extras)
        if on_change is not None:
            self.ionomers.changed.connect(on_change)
            self.extras.changed.connect(on_change)

    def save_state(self) -> dict:
        return {"ionomers": self.ionomers.rows(), "extras": self.extras.rows()}

    def restore_state(self, state: dict) -> None:
        self.ionomers.set_rows(state.get("ionomers", []))
        self.extras.set_rows(state.get("extras", []))

    def values(self) -> dict:
        # Preserve raw text for numeric validation in the calculation layer.
        ionomers = [
            {"ratio": r[0], "conc": r[1], "density": r[2]}
            for r in self.ionomers.rows()
        ]
        extras = [
            {"name": r[0] or "Material", "ratio": r[1]}
            for r in self.extras.rows()
        ]
        return {"ionomers": ionomers, "extras": extras}


class InkCalcTool(Tool):
    id = "ink_calc"
    name = "Ink Calculator"
    subtitle = "AEM catalyst-ink recipe with spray-loss adjustment. No files needed — enter values."
    needs_files = False
    run_button_label = "Calculate"

    def parameter_schema(self) -> list[ParameterSpec]:
        return [
            ParameterSpec("mode", "Mode", kind="choice",
                          choices=["area", "catalyst"], default="area",
                          tooltip="'area': catalyst from loading × area. 'catalyst': enter mass directly.",
                          group="Base"),
            ParameterSpec("length", "Active length", kind="float", default=5.0,
                          minimum=1e-9, step=0.1, suffix="cm",
                          show_if=("mode", "area"), group="Base"),
            ParameterSpec("width", "Active width", kind="float", default=5.0,
                          minimum=1e-9, step=0.1, suffix="cm",
                          show_if=("mode", "area"), group="Base"),
            ParameterSpec("loading", "Loading", kind="float", default=0.5,
                          minimum=1e-9, step=0.1, suffix="mg/cm²",
                          show_if=("mode", "area"), group="Base"),
            ParameterSpec("avail_cat", "Available catalyst", kind="float", default=12.5,
                          minimum=1e-9, step=0.1, suffix="mg",
                          show_if=("mode", "catalyst"), group="Base"),
            ParameterSpec("ink_density", "Ink concentration", kind="float", default=2.0,
                          minimum=1e-9, step=0.1, suffix="mg cat/mL", group="Base"),
        ]

    def extra_widget(self, parent, on_change: Callable[[], None]):
        return IonomerExtrasWidget(parent, on_change=on_change)

    def exportable_artifacts(self, result: ToolResult) -> list[Artifact]:
        original = result.tables["Recipe summary"]
        values = original.set_index("Quantity")["Value"]
        required = {
            "Catalyst": "Required catalyst", "Ink volume": "Required ink volume",
            "Total ionomer": "Required ionomer", "Total dispersion mass": "Required dispersion mass",
            "Total dispersion volume": "Required dispersion volume", "Total other solids": "Required other solids",
            "Total solids": "Required total solids",
        }
        rows = []
        adjusted = "Required catalyst" in values
        for quantity, unit in original[["Quantity", "Unit"]].itertuples(index=False, name=None):
            if quantity not in required:
                continue
            row = {"Quantity": quantity, "Base recipe": values[quantity]}
            if adjusted:
                row["Required batch"] = values[required[quantity]]
            row["Unit"] = unit
            rows.append(row)
        if adjusted:
            for quantity in ("Spray loss", "Batch scale factor"):
                row = original.loc[original["Quantity"] == quantity].iloc[0]
                rows.append({"Quantity": quantity, "Base recipe": None,
                             "Required batch": row["Value"], "Unit": row["Unit"]})
        tables = {"Recipe summary": pd.DataFrame(rows)}
        if "Ionomers" in result.tables:
            ionomers = result.tables["Ionomers"].rename(columns={
                "Active ionomer (mg)": "Base active (mg)",
                "Dispersion (mg)": "Base dispersion (mg)",
                "Volume (µL)": "Base volume (µL)",
            })
            if "Adjusted ionomers" in result.tables:
                batch = result.tables["Adjusted ionomers"].drop(columns="Type").rename(columns={
                    "Required active ionomer (mg)": "Batch active (mg)",
                    "Required dispersion (mg)": "Batch dispersion (mg)",
                    "Required volume (µL)": "Batch volume (µL)",
                })
                ionomers = pd.concat([ionomers.reset_index(drop=True), batch.reset_index(drop=True)], axis=1)
            tables["Ionomers"] = ionomers
        if "Other materials" in result.tables:
            tables["Other materials"] = result.tables["Other materials"].copy()
        return super().exportable_artifacts(ToolResult(tables=tables))

    def run(self, files, params: dict, extra=None) -> ToolResult:
        combined = dict(params)
        if extra:
            combined.update(extra)
        res = calculate_ink_recipe(combined)

        scalars = {
            "Catalyst (mg)": res["catalyst_mg"],
            "Ink volume (mL)": res["ink_volume_ml"],
            "Total ionomer (mg)": res["total_ionomer_mg"],
            "Total dispersion mass (mg)": res["total_dispersion_mass_mg"],
            "Total dispersion volume (µL)": res["total_dispersion_volume_ul"],
            "Total other solids (mg)": res["total_extra_mg"],
            "Total solids (mg)": res["total_solids_mg"],
        }

        adj = res["adjusted"]
        if adj:
            scalars.update({
                "Spray loss (%)": adj["loss_factor"] * 100.0,
                "Batch scale factor": adj["scale_factor"],
                "Required catalyst (mg)": adj["required_catalyst_mg"],
                "Required ink volume (mL)": adj["ink_volume_ml"],
                "Required ionomer (mg)": adj["total_ionomer_mg"],
                "Required dispersion mass (mg)": adj["total_dispersion_mass_mg"],
                "Required dispersion volume (µL)": adj["total_dispersion_volume_ul"],
                "Required other solids (mg)": adj["total_extra_mg"],
                "Required total solids (mg)": adj["total_solids_mg"],
            })

        summary_rows = [
            ("Catalyst", res["catalyst_mg"], "mg"),
            ("Ink volume", res["ink_volume_ml"], "mL"),
            ("Total ionomer", res["total_ionomer_mg"], "mg"),
            ("Total dispersion mass", res["total_dispersion_mass_mg"], "mg"),
            ("Total dispersion volume", res["total_dispersion_volume_ul"], "µL"),
            ("Total other solids", res["total_extra_mg"], "mg"),
            ("Total solids", res["total_solids_mg"], "mg"),
        ]
        if adj:
            summary_rows.extend([
                ("Spray loss", adj["loss_factor"] * 100.0, "%"),
                ("Batch scale factor", adj["scale_factor"], "×"),
                ("Required catalyst", adj["required_catalyst_mg"], "mg"),
                ("Required ink volume", adj["ink_volume_ml"], "mL"),
                ("Required ionomer", adj["total_ionomer_mg"], "mg"),
                ("Required dispersion mass", adj["total_dispersion_mass_mg"], "mg"),
                ("Required dispersion volume", adj["total_dispersion_volume_ul"], "µL"),
                ("Required other solids", adj["total_extra_mg"], "mg"),
                ("Required total solids", adj["total_solids_mg"], "mg"),
            ])

        iono_df = pd.DataFrame([
            {"Type": r["type"], "Active ionomer (mg)": r["active_mg"],
             "Dispersion (mg)": r["dispersion_mg"], "Volume (µL)": r["volume_ul"]}
            for r in res["ionomers"]
        ])
        tables = {
            "Recipe summary": pd.DataFrame(
                summary_rows, columns=["Quantity", "Value", "Unit"]
            )
        }
        if not iono_df.empty:
            tables["Ionomers"] = iono_df
        if res["extras"]:
            adjusted_extras = adj["extras"] if adj else []
            rows = []
            for index, extra_row in enumerate(res["extras"]):
                row = {"Material": extra_row["name"], "Mass (mg)": extra_row["mass_mg"]}
                if adj:
                    row["Required mass (mg)"] = adjusted_extras[index]["mass_mg"]
                rows.append(row)
            tables["Other materials"] = pd.DataFrame(rows)

        text = []
        if adj:
            text.append(
                f"Spray-loss adjustment (area mode):\n"
                f"  Geometric loss : {adj['loss_factor'] * 100:.1f} %\n"
                f"  Req. batch catalyst : {adj['required_catalyst_mg']:.2f} mg\n"
                f"  Req. ink volume : {adj['ink_volume_ml']:.3f} mL\n"
                f"  Req. total solids : {adj['total_solids_mg']:.2f} mg"
            )
            for a in adj["ionomers"]:
                text.append(f"  Type {a['type']} required volume : {a['volume_ul']:.1f} µL")
            adj_df = pd.DataFrame([
                {
                    "Type": a["type"],
                    "Required active ionomer (mg)": a["active_mg"],
                    "Required dispersion (mg)": a["dispersion_mg"],
                    "Required volume (µL)": a["volume_ul"],
                }
                for a in adj["ionomers"]
            ])
            if not adj_df.empty:
                tables["Adjusted ionomers"] = adj_df
        else:
            text.append(
                "Catalyst mode uses the entered catalyst mass; spray-loss scaling is not applied."
            )

        return ToolResult(
            tables=tables, scalars=scalars,
            text_blocks={"Adjusted batch": "\n".join(text)},
        )
