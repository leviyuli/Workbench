from __future__ import annotations

from typing import Callable

import pandas as pd
from PySide6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from workbench.core.tool_base import ParameterSpec, Tool, ToolResult
from workbench.tools.ink_calc.logic import calculate_ink_recipe


def _to_float(text: str, default: float = 0.0) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


class _EditableTable(QWidget):
    """A small add/remove table backed by a QTableWidget."""

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
        values = values if values is not None else self._default_row
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c, val in enumerate(values):
            self.table.setItem(r, c, QTableWidgetItem(str(val)))

    def _remove_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedItems()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

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
    def __init__(self, parent: QWidget | None = None):
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

    def values(self) -> dict:
        ionomers = [
            {"ratio": _to_float(r[0]), "conc": _to_float(r[1]), "density": _to_float(r[2])}
            for r in self.ionomers.rows()
        ]
        extras = [
            {"name": r[0] or "Material", "ratio": _to_float(r[1])}
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
            ParameterSpec("length", "Active length", kind="float", default=0.0,
                          minimum=0.0, step=0.1, suffix="cm",
                          show_if=("mode", "area"), group="Base"),
            ParameterSpec("width", "Active width", kind="float", default=0.0,
                          minimum=0.0, step=0.1, suffix="cm",
                          show_if=("mode", "area"), group="Base"),
            ParameterSpec("loading", "Loading", kind="float", default=0.0,
                          minimum=0.0, step=0.1, suffix="mg/cm²",
                          show_if=("mode", "area"), group="Base"),
            ParameterSpec("avail_cat", "Available catalyst", kind="float", default=0.0,
                          minimum=0.0, step=0.1, suffix="mg",
                          show_if=("mode", "catalyst"), group="Base"),
            ParameterSpec("ink_density", "Ink density", kind="float", default=2.0,
                          minimum=0.0, step=0.1, suffix="mg cat/mL", group="Base"),
        ]

    def extra_widget(self, parent, on_change: Callable[[], None]):
        return IonomerExtrasWidget(parent)

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
        }

        iono_df = pd.DataFrame([
            {"Type": r["type"], "Active ionomer (mg)": r["active_mg"],
             "Dispersion (mg)": r["dispersion_mg"], "Volume (µL)": r["volume_ul"]}
            for r in res["ionomers"]
        ])
        tables = {"Ionomers": iono_df} if not iono_df.empty else {}
        if res["extras"]:
            tables["Other materials"] = pd.DataFrame([
                {"Material": e["name"], "Mass (mg)": e["mass_mg"]} for e in res["extras"]
            ])

        text = []
        adj = res["adjusted"]
        if adj:
            text.append(
                f"Spray-loss adjustment (area mode):\n"
                f"  Geometric loss : {adj['loss_factor'] * 100:.1f} %\n"
                f"  Req. batch catalyst : {adj['required_catalyst_mg']:.2f} mg"
            )
            for a in adj["ionomers"]:
                text.append(f"  Type {a['type']} required volume : {a['required_volume_ul']:.1f} µL")
            adj_df = pd.DataFrame([
                {"Type": a["type"], "Required volume (µL)": a["required_volume_ul"]}
                for a in adj["ionomers"]
            ])
            if not adj_df.empty:
                tables["Adjusted volumes"] = adj_df
        else:
            text.append("Spray-loss adjustment is only computed in 'area' mode with L, W > 0.")

        return ToolResult(
            tables=tables, scalars=scalars,
            text_blocks={"Adjusted batch": "\n".join(text)},
        )
