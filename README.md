# Workbench

A single PySide6 desktop app that unifies five electrochemistry tools behind one
Claude-styled interface. Every tool shares the same way to pick files, set
parameters, view plots/tables, and export results.

## Tools

| Tool | What it does | Input | Export |
|------|--------------|-------|--------|
| **LSV Decoupling** | Iterative Tafel fit → kinetic / ohmic / catalyst-layer / residual overpotentials | .csv / .txt / .xlsx (one or many; auto unit + column detect) | CSV, XLSX, PNG |
| **EIS (OSIF)** | Kramers-Kronig cleanup + transmission-line model fitting | .csv / .txt / .xlsx (one or many) | CSV, XLSX, PNG |
| **ECSA (non-faradaic)** | C_dl & ECSA from a CV scan-rate series | .txt named like `CV_50mV_1.txt` (many / folder) | XLSX (CV_long, ECSA_result, CV_time_series), PNG |
| **i–t Integrate** | Stitch repeated potentiostatic tests; optional resample | .txt (many / folder) | TXT (original format), XLSX |
| **TXT → XLSX** | Bundle tab-delimited .txt files into one workbook (a sheet per file) | .txt (many / folder) | XLSX |
| **Ink Calculator** | AEM catalyst-ink recipe with spray-loss adjustment | form fields (no files) | CSV, XLSX |

## How the tools were unified

- **One file picker** (`widgets/file_input.py`): accepts a single file, many
  files, or a whole folder, plus drag-and-drop. Each tool only declares the
  extensions it accepts and whether folders/multi-select are allowed.
- **One reader** (`core/io.py` `read_table` / `read_text_grid`): tolerant of
  delimiter (tab/comma/semicolon/whitespace), encoding, and metadata rows.
  Domain-specific readers (LSV unit scaling, EIS column extraction, CV cycle
  detection) live in `core/data_io.py` and `tools/*/logic*.py`, copied verbatim
  from the originals so results match exactly.
- **One parameter panel** (`widgets/parameter_panel.py`): tools declare
  `ParameterSpec`s; the form is generated and grouped automatically.
- **One results area** (`widgets/results_panel.py`): plots, a summary, and
  tables in a consistent tab strip.
- **One Export button** (`widgets/export_button.py`): identical everywhere; the
  menu is populated from each tool's declared artifacts.

Adding a tool = drop a folder under `workbench/tools/<id>/` with a `Tool`
subclass and register it in `workbench/tools/__init__.py`.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m workbench
```

## Package (optional)

```powershell
pyinstaller --noconfirm --windowed --name Workbench `
  --add-data "workbench/styles;workbench/styles" workbench/__main__.py
```

## Tests

```powershell
pip install pytest
pytest workbench/tests
```
