"""The plugin contract every tool implements, plus the small dataclasses the
generic UI uses to render parameters, results, and export options.

The MainWindow never imports a concrete tool; it only consumes ``Tool``,
``ParameterSpec``, ``ToolResult`` and ``Artifact``. Adding a sixth tool is a
one-folder change registered in ``tools/__init__.py``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from PySide6.QtWidgets import QWidget


# ---------------------------------------------------------------------------
# Parameter declaration
# ---------------------------------------------------------------------------
@dataclass
class ParameterSpec:
    """One control in the auto-generated parameter panel."""
    key: str
    label: str
    kind: str = "float"          # float | int | str | choice | bool
    default: Any = 0.0
    minimum: float = -1e12
    maximum: float = 1e12
    step: float = 1.0
    decimals: int = 4
    choices: list[str] = field(default_factory=list)
    suffix: str = ""
    tooltip: str = ""
    group: str = "Parameters"
    # Optional visibility gate. Either a single (dep_key, dep_value) condition or a
    # list of such conditions that are AND-combined (the control shows only when all
    # hold). For one condition the control is shown when params[dep_key] equals
    # dep_value, OR — when dep_value is a list/tuple/set — when params[dep_key] is a
    # member of it.
    show_if: Optional[Any] = None


# ---------------------------------------------------------------------------
# Results & exportable artifacts
# ---------------------------------------------------------------------------
@dataclass
class ToolResult:
    """Everything a tool produces from one run."""
    figures: dict[str, Any] = field(default_factory=dict)       # name -> matplotlib Figure
    tables: dict[str, Any] = field(default_factory=dict)        # name -> pandas DataFrame
    scalars: dict[str, Any] = field(default_factory=dict)       # name -> value
    text_blocks: dict[str, str] = field(default_factory=dict)   # name -> monospace text
    warnings: list[str] = field(default_factory=list)
    # Values to write back into the parameter panel after the run (param key ->
    # new value), e.g. EIS KK-test populating the suggested f_min/f_max window.
    param_updates: dict[str, Any] = field(default_factory=dict)
    # Small per-field notes shown next to a parameter control after the run
    # (param key -> short text), e.g. the ± standard error of a fitted value.
    param_annotations: dict[str, str] = field(default_factory=dict)
    # Per-result data a tool may need when building artifacts (kept private to it).
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolAction:
    """A secondary action button rendered beside the main Run button. ``method``
    names a Tool method with the same ``(files, params, extra)`` signature as
    ``run`` that returns a ToolResult (e.g. LSV's 'Show plot')."""
    label: str
    method: str


@dataclass
class Artifact:
    """One exportable item. ``kind`` selects how ExportButton writes it.

    kind == "csv"      -> payload is a DataFrame
    kind == "xlsx"     -> payload is dict[sheet_name, DataFrame]
    kind == "workbook" -> payload is an openpyxl Workbook (preserves cell types)
    kind == "png"      -> payload is a matplotlib Figure
    kind == "txt"      -> payload is a str (written verbatim)
    """
    name: str
    kind: str
    default_filename: str
    payload: Any


# ---------------------------------------------------------------------------
# The Tool contract
# ---------------------------------------------------------------------------
class Tool(ABC):
    # Identity (override as class attributes)
    id: str = "tool"
    name: str = "Tool"
    subtitle: str = ""
    icon: str = ""                       # optional path to an SVG in styles/icons

    # Input behaviour
    needs_files: bool = True          # show the file picker
    files_required: bool = True       # block Run when no files are selected
    multiple_files: bool = True
    accepts_directory: bool = False
    accepted_file_types: tuple[str, ...] = (".csv", ".txt", ".xlsx", ".xls")
    run_button_label: str = "Run"

    @abstractmethod
    def parameter_schema(self) -> list[ParameterSpec]:
        """Declare the parameter controls for this tool."""

    @abstractmethod
    def run(self, files: list[Path], params: dict, extra: Any = None) -> ToolResult:
        """Execute the tool. ``extra`` is whatever ``extra_widget`` returns via
        its ``.values()`` method (or None)."""

    def exportable_artifacts(self, result: ToolResult) -> list[Artifact]:
        """Default: export every table as CSV and every figure as PNG."""
        items: list[Artifact] = []
        for tname, df in result.tables.items():
            items.append(Artifact(
                name=f"{tname} (CSV)", kind="csv",
                default_filename=f"{self.id}_{_slug(tname)}.csv", payload=df,
            ))
        if result.tables:
            items.append(Artifact(
                name="All tables (Excel)", kind="xlsx",
                default_filename=f"{self.id}_results.xlsx", payload=dict(result.tables),
            ))
        for fname, fig in result.figures.items():
            items.append(Artifact(
                name=f"{fname} (PNG)", kind="png",
                default_filename=f"{self.id}_{_slug(fname)}.png", payload=fig,
            ))
        return items

    def extra_widget(self, parent: QWidget, on_change: Callable[[], None]) -> Optional[QWidget]:
        """Optional custom input widget (e.g. the ink ionomer table). The widget
        must expose ``.values()`` returning a JSON-friendly object."""
        return None

    def actions(self) -> list["ToolAction"]:
        """Optional secondary action buttons shown next to Run (e.g. LSV's
        'Show plot'). Each names a method with the same signature as ``run``
        that returns a ToolResult."""
        return []


def _slug(text: str) -> str:
    keep = []
    for ch in text.lower():
        keep.append(ch if ch.isalnum() else "_")
    out = "".join(keep).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "data"
