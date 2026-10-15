"""Tool interface and shared parameter, result and export data structures."""
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
    kind: str = "float"          # float | int | str | choice | bool | input_file | reset
    default: Any = 0.0
    minimum: float = -1e12
    maximum: float = 1e12
    step: float = 1.0
    decimals: int = 4
    choices: list[str] = field(default_factory=list)
    suffix: str = ""
    tooltip: str = ""
    group: str = "Parameters"
    # For ``kind="reset"``, keys whose controls should be restored to their
    # declared defaults when the button is clicked.
    reset_keys: list[str] = field(default_factory=list)
    # Optional Tool method used to calculate reset values from current inputs.
    # It receives ``(files, params)`` and returns a parameter-update mapping.
    reset_method: str = ""
    # Optional visibility gate. Either a single (dep_key, dep_value) condition or a
    # list of such conditions that are AND-combined (the control shows only when all
    # hold). For one condition the control is shown when params[dep_key] equals
    # dep_value, OR — when dep_value is a list/tuple/set — when params[dep_key] is a
    # member of it.
    show_if: Optional[Any] = None
    # Display a result value without allowing the user to edit it.
    read_only: bool = False


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
    # Keep detail data available to exporters while optionally showing plots only.
    display_tables: bool = True
    display_summary: bool = True
    # Portable PNG snapshots restored from project files. Fresh calculations
    # continue supplying live matplotlib Figures above.
    figure_images: dict[str, bytes] = field(default_factory=dict)


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
    # Project files preserve the original export bytes, including workbook
    # formatting and tool-specific text layouts, without pickling the payload.
    serialized_payload: bytes | None = None


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
    extra_before_parameters: bool = False

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
        for fname, image in result.figure_images.items():
            if fname not in result.figures:
                items.append(Artifact(
                    name=f"{fname} (PNG)", kind="png",
                    default_filename=f"{self.id}_{_slug(fname)}.png", payload=None,
                    serialized_payload=image,
                ))
        return items

    def extra_widget(self, parent: QWidget, on_change: Callable[[], None]) -> Optional[QWidget]:
        """Optional custom input widget (e.g. the ink ionomer table). The widget
        must expose ``.values()`` returning a JSON-friendly object. It may also
        expose ``.files()`` for export-directory selection and
        ``.set_parameters(params)`` to respond to parameter changes, plus
        ``.set_result_parameters(updates, annotations)`` for result readouts.
        Custom inputs implement ``.save_state()`` and ``.restore_state(state)``
        for project persistence; use Path values for embedded input references."""
        return None

    def subtools(self) -> list["Tool"]:
        """Optional persistent subpages displayed under one sidebar entry."""
        return []

    def reset_runtime_state(self) -> None:
        """Discard nonpersistent caches when a workspace is opened or cleared."""

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
