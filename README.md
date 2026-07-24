# Workbench

A single PySide6 desktop app that unifies lab-data tools behind one
Claude-styled interface. Every tool shares the same way to pick files, set
parameters, view plots/tables, and export results.

## Tools

| Tool | What it does | Input | Export |
|------|--------------|-------|--------|
| **LSV Decoupling** | Iterative Tafel fit → kinetic / ohmic / catalyst-layer / residual overpotentials | .csv / .txt / .xlsx (one or many; auto unit + column detect) | CSV, XLSX, PNG |
| **EIS (OSIF)** | Kramers-Kronig cleanup + transmission-line model fitting | .csv / .txt / .xlsx (one or many) | CSV, XLSX, PNG |
| **ECSA (non-faradaic)** | C_dl & ECSA from a CV scan-rate series | .txt named like `CV_50mV_1.txt` (many / folder) | XLSX (CV_long, ECSA_result, CV_time_series), PNG |
| **i–t Integrate** | Stitch repeated current-time or potential-time tests; optional resample | .txt (many / folder) | TXT (original format), XLSX |
| **TXT → XLSX** | Bundle .txt / .csv / .xlsx files into one workbook (a sheet per input file; xlsx sheets expanded) | .txt / .csv / .xlsx (many / folder) | XLSX |
| **Ink Calculator** | AEM catalyst-ink recipe with spray-loss adjustment | form fields (no files) | CSV, XLSX |
| **FTIR / XPS Display** | Stack processed FTIR or XPS spectra with optional normalization, offsets, and a display-only x range | .csv / .txt / .dat (many) | CSV, XLSX |

## What each tool does

### LSV Decoupling
Decomposes a linear-sweep polarization curve into its loss contributions. It reads
one or more LSV files (auto-detecting the voltage and current columns, converting
mV→V and mA/µA→A, trimming a cyclic sweep to its forward branch, and averaging
several files onto a common voltage grid), then runs an **iterative Tafel fit** that
peels off ohmic (HFR) and catalyst-layer (R_CL) drops to isolate the kinetic slope.
- **Set:** reference electrode + pH + temperature (→ E_rev), HFR, R_CL, the Tafel
  fit current range, and a "breakdown target" current density.
- **Get:** a *Decoupled curves* plot (cumulative E_rev + η_kin + η_ohm + η_RCL +
  η_res over the raw LSV), an *Overpotential breakdown* bar at the target current,
  and tables of the stacked curves, per-point components, and the fit summary
  (Tafel slope b, exchange current i₀, intercept, R², N).

### EIS (OSIF)
Validates and fits impedance spectra. It has three **modes** (pick one, then Run):
- **KK test** — runs a Kramers-Kronig consistency check and shows valid/invalid
  points plus a residual-vs-frequency plot. It then **writes the suggested valid
  frequency window and an HFR estimate back into the f_min/f_max/HFR fields**.
- **Fit** — KK-validates, then fits a transmission-line model over the f_min–f_max
  window. Produces a Nyquist (equal-aspect), Bode |Z|, and Re(Z)/Im(Z) panels, plus
  a parameter table with standard errors (HFR, R_CL, Q_dl, φ, optionally R_k and the
  R_CL/3 low-frequency intercept).
- **Simulate** — draws the forward model from the parameters you type over a
  frequency grid, overlaying your measured data if any is loaded.
- **Models:** Transmission Line, Faradaic TML (CPE), 1-D Linear / Spherical
  Diffusion, with optional wire inductance (L_wire, Θ).
- **Typical workflow:** *KK test → review/adjust the window → switch to Fit → Run.*

### ECSA (non-faradaic)
Extracts the double-layer capacitance C_dl and electrochemically active surface area
from a series of CVs taken at different scan rates. Files are named so the rate is
recognized (e.g. `CV_50mV_1.txt` = 50 mV/s, replicate 1); the reader tolerates
varied headers/delimiters, detects cycles, and converts potential to RHE/SHE.
- **Set:** reference electrode, target scale, pH, specific capacitance Cs, points
  near the midpoint, cycle index (-1 = last), and replicate handling.
- **Get:** the ECSA fit (j_dl vs scan rate; slope = C_dl, ECSA = C_dl/Cs, R²), a CV
  overlay, and three export sheets — `CV_long`, `ECSA_result`, `CV_time_series`.

### i–t Integrate
Joins repeated current-time (`i_t`, e.g. steady voltammetry or chronoamperometry)
or potential-time (`E_t`, e.g. steady-current or chronopotentiometry) runs into
one continuous record, shifting each file's time column so time runs unbroken
across the join, with an optional linear **resample** to a fixed step (none /
1 s / 10 s / 1 min / 1 hour / 1 day / custom). Select the trace type in the
tool options; files need a header row containing time plus the selected signal.
- Optional **Designate data source** mode colors and labels each file separately
  in the plot and exports adjacent column groups (for standard E/j/time input,
  three columns per source). The default remains the original vertically
  stacked E/j/time output. Stitching and source-column order follow the files'
  order in the input window.
- **Get:** the stitched i–t or E–t plot and table; export as `.txt` in the original
  tab-delimited format (`%.5f` for time, `%.8g` for the rest) or as `.xlsx`.

### TXT → XLSX
Bundles many `.txt` / `.csv` / `.xlsx` files into a single Excel workbook — **one
sheet per input file** (each sheet of an input `.xlsx` is expanded into its own
output sheet) — converting numeric cells (default format `0.####`) while keeping
text as text. Sheet order follows file name or modification time.

### Ink Calculator
Computes an AEM catalyst-ink recipe and a spray-loss-adjusted batch — no files, just
form inputs. Work either **by active area** (loading × length × width) or **by
available catalyst mass**, add any number of ionomers (ratio %, concentration wt %,
density) and other solids (e.g. PTFE, carbon).
- **Get:** catalyst mass, ink volume, total ionomer, dispersion mass/volume, a
  per-ionomer breakdown, and — in area mode — the geometric spray-loss factor with
  the required batch catalyst and dispersion volumes. Export as CSV or XLSX.

### FTIR / XPS Display
Displays already-processed spectra without background subtraction or baseline
correction. FTIR mode detects wavenumber plus T or A columns and can convert T to
absorbance using `A = -log10(T / 100)`. XPS mode detects binding energy plus
intensity/count columns. Both use the conventional high-to-low x-axis direction.
- **Set:** technique, a file for automatic peak detection, optional normalization, vertical offset, whether y tick
  labels are hidden, an optional dotted-gray vertical reference line, and an optional displayed x minimum/maximum. Curve order
  follows the draggable file list. Peak filters expose relative prominence,
  noise rejection, minimum width, and smoothing-window adjustments.
- **Get:** the stacked spectra, full-range export data, and a peak-detection page
  listing the selected spectrum's peak x-coordinates. FTIR transmittance bands
  are detected as minima; absorbance and XPS peaks are detected as maxima. A custom x range only
  changes the figure viewport; it never removes rows from CSV or Excel output.
  Plot-data exports place each sample in an adjacent X/Y-display column pair,
  with the sample name included in both column headers.

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
