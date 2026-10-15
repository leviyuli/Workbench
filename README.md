# Workbench

A PySide6 desktop toolkit for electrochemistry and spectroscopy data. Shared
controls handle file selection, parameters, plots, tables and exports.

## Quick start

Run these commands from the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m workbench
```

You can also run `python workbench/app.py`, including with the IDE Run button.
To reopen a workspace at launch, use `python -m workbench "experiment.ecwb"`.
The current dependency set is tested with Python 3.14 on Windows.

## Workbench project files

Use **File → Save / Save As** (`Ctrl+S` / `Ctrl+Shift+S`) to save the whole
workspace as a portable **`.ecwb`** file. It includes uploaded data, settings and
the latest successful results from every page and LSV subfunction, including
hidden calibration inputs, EIS spectra and unfinished Ink Calculator entries.

Open a project with **File → Open** (`Ctrl+O`). Embedded input files remain
available for another run after moving the project. Results reopen as saved plot
images and complete tables, with their exports retained. Run an analysis again
for interactive plots. Current inputs and the last successful result are saved
independently, so unfinished edits survive alongside the previous calculation.

**File → New** (`Ctrl+N`) starts an empty workspace. Changes are marked in the
window title, and New/Open/Close offer Save, Discard or Cancel when needed.
Project actions wait until running calculations finish. Saving is manual;
exports default to the saved project's folder.

## Tools

| Tool | What it does | Input | Export |
|------|--------------|-------|--------|
| **LSV analysis** | Decoupling, forward generation, and Integrated RE electrode/cell comparison | .csv / .txt / .xlsx / .xls, or form fields for Generator | CSV, XLSX, PNG |
| **EIS (OSIF)** | Kramers-Kronig validation, batch HFR analysis and transmission-line model fitting | .csv / .txt / .xlsx / .xls (one or many) | XLSX, PNG |
| **DRT analysis** | Relaxation-time distribution, peaks and impedance diagnostics | .csv / .txt / .xlsx / .xls (one or many) | XLSX, PNG |
| **CV process** | Split and overlay CV cycles with optional RE correction; non-faradaic C_dl & ECSA calculation | .txt with E/i columns and optional time (one / many / folder) | CSV (E/i pairs per cycle), XLSX, PNG |
| **i–t Integrate** | Stitch repeated current-time or potential-time tests; optional resample | .txt (many / folder) | TXT (original format), XLSX |
| **TXT → XLSX** | Bundle .txt / .csv / .xlsx files into one workbook (a sheet per input file; xlsx sheets expanded) | .txt / .csv / .xlsx (many / folder) | XLSX |
| **Ink Calculator** | AEM catalyst-ink recipe with spray-loss adjustment | form fields (no files) | CSV, XLSX |
| **FTIR / XPS Display** | Stack processed FTIR or XPS spectra with optional normalization, offsets, and a display-only x range | .csv / .txt / .dat (many) | CSV, XLSX |

## What each tool does

### Shared potential settings
CV splitting, ECSA, LSV preview/decoupling and i–t/E–t processing use one
reference-electrode conversion and the same **Potential** controls:

1. **Two-electrode** is the default wherever the electrode-setup selector appears.
   It retains the measured cell voltage and hides RE correction controls.
   Select **Three-electrode** explicitly for working-electrode potential and
   reference correction. CV splitting and trace/LSV previews support both setups.
   ECSA requires three-electrode data. LSV decoupling supports both working-electrode
   potential and two-electrode cell voltage.
2. For CV and traces, choose **Output potential scale → RHE** (or SHE) to apply
   correction. After selecting Three-electrode, **Original** is the default output
   scale in both CV functions and trace processing.
   LSV preview and fitting use RHE for three-electrode data and measured cell voltage
   for two-electrode data.
3. Select the **Reference electrode**, including filling solution, then enter
   pH and temperature. **Custom E_ref vs SHE** accepts a calibrated offset in V.
   RHE inputs receive no additional shift when the output is RHE.

The shared formula is `E_RHE = E_measured + E_ref(vs SHE) + 2.303 R T/F × pH`.
Preset offsets are specified at 25 °C; other temperatures change the RHE pH term
and produce a reminder to use a calibrated custom reference offset when needed.
The Hg/HgO (1 M KOH) preset is 0.1053 V vs SHE. Ag/AgCl saturated KCl (0.197 V)
and 3 M KCl (0.210 V) are separate choices. Use **Custom E_ref vs SHE** for a
measured reference offset.
See [Pine Research's conversion reference](https://pineresearch.com/tools/reference-electrode-conversion-calculator/)
for filling-solution and temperature conventions.

CV splitting and traces preserve source V/mV units and all current/time values;
their plots and exports show the selected scale. When potential units are absent,
**Unspecified potential unit** selects V or mV and the result reports the assumption.
LSV and ECSA normalize potential to V for analysis. LSV Generator distinguishes
half-cell RHE potential from full-cell voltage. EIS and DRT operate on impedance.

### LSV analysis
Select **Decoupling**, **Generator**, or **Integrated RE for cell** in the
**Function** selector. Each function retains its own inputs, parameters, results
and exports when you switch. Decoupling is selected initially.

#### Decoupling
Decomposes a linear-sweep polarization curve into its loss contributions. It reads
one or more LSV files (auto-detecting the voltage and current columns, converting
mV→V and mA/µA→A, trimming a cyclic sweep to its forward branch, and averaging
several files onto a common voltage grid), then runs an **iterative Tafel fit** that
peels off ohmic (HFR) and catalyst-layer (R_CL) drops to isolate the kinetic slope.
- **Set:** electrode setup, temperature, HFR, R_CL, the Tafel fit current range,
  and a "breakdown target" current density. For three-electrode data, also set the
  reference electrode (including filling solution) and pH for RHE conversion.
- **Show plot** applies the same RHE conversion as fitting, including per-file and
  averaged curves. With Two-electrode selected, it previews cell voltage.
- **Two-electrode fitting** uses measured cell voltage directly, without a reference
  shift, and fits current magnitudes with the temperature-dependent reversible
  water-splitting cell voltage. The reported Tafel and R_CL parameters describe
  effective cell contributions. Current sign and voltage polarity are preserved
  in the stacked curves; the breakdown displays voltage magnitudes.
- **Include RCL analysis** is enabled by default. Uncheck it under **Resistances**
  to fit with only HFR correction and report kinetic, ohmic, and residual
  overpotentials. R_CL is hidden and ignored; RCL contributions and catalyst
  utilization are omitted from plots, tables, and exports.
- **Get:** a *Decoupled curves* plot (cumulative E_rev + η_kin + η_ohm + η_RCL +
  η_res over the raw LSV), with optional X/Y-axis exchange, a *Catalyst
  utilization* plot and Excel sheet (the U_J used for η_RCL), an *Overpotential
  breakdown* bar at the target current,
  and tables of the stacked curves, per-point components, and the fit summary
  (Tafel slope b, exchange current i₀, intercept, R², N).

#### Generator
Builds synthetic polarization curves from the same Tafel, ohmic and catalyst-layer
equations used by LSV Decoupling, without requiring an input file.
- **Set:** HER, OER or two-electrode mode; temperature; series HFR; separate
  electrode Tafel slope, exchange current density and R_CL; then stop at either a
  chosen current density or a terminal potential/cell voltage.
- **Presets:** load editable, condition-specific examples for NiFe-LDH, IrO₂ and
  Pt/C. Loading a preset also sets its reference temperature. Every result records
  the source conditions and whether the loaded b/i₀ pair was subsequently modified.
- **Get:** a conventional generated LSV, a cumulative potential-construction plot,
  loss/utilization diagnostics, the complete pointwise model table and an endpoint
  summary. HER current and potential use their native negative signs; two-electrode
  results use positive current and full-cell voltage.

#### Integrated RE for cell
Quickly reviews the two electrode LSVs measured against an integrated reference
and compares their reconstructed cell voltage with a two-electrode measurement.
- **Inputs:** separate **Anode**, **Cathode**, and optional **Cell** LSV upload tabs,
  each accepting one LSV file. A separate **EIS inputs for HFR** section provides
  **Anode EIS**, **Cathode EIS**, and **Cell EIS** tabs, each accepting multiple
  spectra and keeping its HFR average independent. Set each input's area for
  absolute-current data; current-density data uses its recorded units. Auto
  current polarity makes OER positive,
  HER negative and cell current positive using one multiplier for the full curve.
  **As recorded** and **Invert** override it without folding baseline sign changes.
- **HFR from EIS:** extracts HFR separately from each uploaded spectrum using
  its first high-frequency imaginary-zero crossing, with minimum real impedance
  as a reported fallback. Each role's **Average HFR** is the arithmetic mean of its
  valid per-file HFR estimates. Use the EIS tool for KK validation and model fits. Raw-Ω
  impedance requires the role's positive electrode area to obtain Ω·cm²;
  already area-normalized spectra are not multiplied again. Failed files are
  recorded and excluded from the mean. If uploaded spectra all fail, the role
  reports an error; leaving optional EIS inputs empty is allowed.
- **Include iR correction:** adds dashed corrected curves to the existing
  electrode and cell-comparison plots while retaining the original solid curves.
  The signed formula is `E_corrected = E - j × HFR`, so HER shifts toward more
  positive potential. Reconstructed cell voltage subtracts
  `|j| × (HFR_anode + HFR_cathode)`; measured cell voltage uses its own HFR.
  Enable this after supplying EIS for both electrodes and, when cell LSV is
  loaded, for the cell. Branch windows stay tied to unshifted measured data.
  Three-electrode reference calibration continues to use the measured curves
  before iR correction.
- **Reference:** choose the shared integrated reference, pH and temperature.
  Inputs default to already-RHE at 298.15 K. The electrode plot offers
  **E vs RHE** or **Overpotential**; the latter subtracts the OER equilibrium
  potential from the anode while HER equilibrium remains 0 V.
- **Reconstruction:** at matching positive current magnitudes, calculate
  `V_cell = E_anode(RHE) - E_cathode(RHE)`. This is full cell voltage;
  cell overpotential is `V_cell - E_rev(T)`. Multiple crossings use the most
  anodic anode and most cathodic cathode potentials. Optional comparison windows
  select branches using unshifted RHE potentials (or cell voltage for Cell).
  Interpolation stays within measured segments and current overlap; gaps are
  retained. The comparison plot uses voltage on X and current density on Y.
- **Correction:** **Manual** adds the entered shared offset to both electrodes.
  **3E onset** or **3E peak** reveals a separate calibration upload tab with its
  own area/polarity settings and independent reference conversion controls.
  Select **3E onset** to use the default PtRuC cathode calibration: the bundled
  `workbench/Test_files/LSV_PtRuC.txt` loads automatically on first activation,
  with **Hg/HgO (1 M KOH)**, **pH 14**, **298 K**, and **0.05 A/cm²** onset current.
  The file already contains current density, so no calibration area is required.
  You can replace the calibration file and edit its settings; toggling correction
  preserves those changes. Choose the matching electrode. Onset aligns the selected
  branches at the entered current density; peak aligns genuine interior
  reaction-current peaks in separately specified RHE search windows.
  The calculated 3E offset is `E_3E(feature) - E_integrated(feature)`. The applied
  shared offset adds the manual fine adjustment to that value. A shared offset
  cancels from reconstructed cell voltage.
  After Run, read-only **Calculated 3E offset** and **Applied shared offset**
  fields show the automatic and total corrections from the last successful run.
  The editable fine adjustment remains separate, so rerunning applies the
  calculated correction once.
- **Outputs:** only electrode and cell-comparison plots, plus optional calibration
  alignment, are displayed in the results view. All details remain available in
  CSV/XLSX exports: source/converted/iR-corrected LSVs, measured and reconstructed
  comparisons, per-spectrum and mean HFR, cleaned EIS points, settings, calibration
  features, and run notes/warnings. Plot exports remain available as PNG.

### EIS (OSIF)
Validates and fits impedance spectra. It has four **modes** (pick one, then Run):

- **KK test** — runs a Kramers-Kronig consistency check and shows valid/invalid
  points plus a residual-vs-frequency plot. It then **writes the suggested valid
  frequency window and an HFR estimate back into the f_min/f_max/HFR fields**.
  HFR uses the highest-frequency x-axis intersection of the averaged Nyquist
  spectrum in the selected window, interpolating linearly between adjacent points
  that straddle zero imaginary impedance (or using an exact on-axis point).
  If there is no intersection, it uses the minimum real impedance; the result
  reports which method was used.
- **Batch KK test** — treats each uploaded file as a separate spectrum and runs
  KK validation without combining files or fitting a transmission-line model.
  Shows the existing Nyquist/residual diagnostics for each sample and an **HFR
  by sample** plot with numbered samples in input-file order. The **HFR summary**
  table maps numbers to filenames and reports each HFR method, KK pass rate and
  suggested frequency window. HFR uses each sample's measured spectrum in the
  selected f_min–f_max window with the same intersection/minimum-real method.
  A failed file is reported under its original sample number while the other
  files continue. Export the summary and diagnostics as XLSX and the plots as PNG.
- **Fit** — KK-validates, then fits a transmission-line model over the f_min–f_max
  window. Produces a Nyquist (equal-aspect), Bode |Z|, and Re(Z)/Im(Z) panels, plus
  a parameter table with standard errors (HFR, R_CL, Q_dl, φ, optionally R_k and the
  R_CL/3 low-frequency intercept).
- **Simulate** — draws the forward model from the parameters you type over a
  frequency grid, overlaying your measured data if any is loaded.
- **Models:** Transmission Line, Faradaic TML (CPE), 1-D Linear / Spherical
  Diffusion, with optional wire inductance (L_wire, Θ).
  Additional Faradaic RC/CPE units enter the pore-wall admittance in parallel;
  only their combined resistance conductance is identifiable.
- **Typical workflow:** *KK test → review/adjust the window → switch to Fit → Run.*

EIS and DRT expect area-normalized impedance in Ω·cm². Multiply raw Ω data by
the electrode area before importing into these tools.

### DRT analysis

Estimate a relaxation-time distribution from uploaded impedance spectra with
TR-NNLS, TR-RBF, BHT or LM. Choose a frequency window and, for the TR
methods, automatic regularization (`λ = 0`) or a fixed positive λ. Multiple
files are averaged at matching frequencies. Results show native distributions,
identified peaks, measured Nyquist data, relative residuals in percent and fit
statistics. BHT real/imaginary and LM RC/RL components are plotted separately.
Export the tables as XLSX and the plot as PNG.

### CV process
Choose a function, then click **Process CV**. The reader accepts common text
delimiters and metadata before the E/i header.
- **Split cycles** (default): load a single raw CV, several
  files, or a folder. Export one CSV per input with adjacent columns
  `Cycle 1 E …`, `Cycle 1 i …`, `Cycle 2 E …`, `Cycle 2 i …`, etc.
  Original mode preserves raw potential/current values and units; RHE/SHE mode
  shifts only potential and labels its scale. The time column is
  omitted, and shorter cycles are padded with blank cells. Arbitrary filenames,
  absolute current and current density are accepted. Incomplete
  scans are retained and labeled as partial. Each input also gets an E/i overlay
  plot with labeled, colored cycles and dashed partial scans, using the original
  units. Use the plot toolbar to zoom/pan and export the plot as PNG.
  An XLSX export collects all the split tables and potential settings.
- **ECSA calculation**: calculate non-faradaic C_dl and ECSA from at least two
  distinct scan rates. Name files like `CV_50mV_1.txt` (50 mV/s, replicate 1) and
  supply current density. Use the shared potential settings for Original/RHE/SHE,
  reference electrode, pH and temperature, then set
  specific capacitance Cs, midpoint sampling, cycle index (-1 = last), and
  replicate handling. Results include the ECSA fit, CV overlay, and workbook
  sheets `Summary`, `CV_long`, `ECSA_result`, and `CV_time_series`.

### i–t Integrate
Joins repeated current-time (`i_t`, e.g. steady voltammetry or chronoamperometry)
or potential-time (`E_t`, e.g. steady-current or chronopotentiometry) runs into
one continuous record, shifting each file's time column so time runs unbroken
across the join, with an optional linear **resample** to a fixed step (none /
1 s / 10 s / 1 min / 1 hour / 1 day / custom). Select the trace type in the
tool options; files need a header row containing time plus the selected signal.
- The shared potential settings optionally correct E columns in either trace
  mode; current and time are unchanged. Stacked and source-grouped exports use
  the same corrected values and label their potential scale.
- Optional **Designate data source** mode colors and labels each file separately
  in the plot and exports adjacent column groups (for standard E/j/time input,
  three columns per source). The default remains the original vertically
  stacked E/j/time output. Stitching and source-column order follow the files'
  order in the input window.
- Empty or malformed files and files with nonfinite or decreasing time values
  are skipped with a warning. The first usable file defines the column layout.
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
Displays processed spectra. FTIR mode detects wavenumber plus T or A columns and can convert T to
absorbance using `A = -log10(T / 100)`. XPS mode detects binding energy plus
intensity/count columns and, when present, plots a `Background Counts`/background/baseline
column as a dashed curve beneath the corresponding signal. Both use the conventional
high-to-low x-axis direction.
- **Set:** technique, optional normalization, vertical offset, whether y tick
  labels are hidden, an optional dotted-gray vertical reference line, and an optional displayed x minimum/maximum. Curve order
  follows the draggable file list.
- **Get:** the stacked spectra and full-range export data. A custom x range only
  changes the figure viewport; it never removes rows from CSV or Excel output.
  Plot-data exports place each sample in an adjacent X/Y-display column pair,
  with the sample name included in both column headers.

## Development

- **One file picker** (`widgets/file_input.py`): accepts a single file, many
  files, or a whole folder, plus drag-and-drop. Each tool only declares the
  extensions it accepts and whether folders/multi-select are allowed.
- **Shared readers** (`core/io.py`, `core/data_io.py`): handle delimiters,
  encodings and metadata, then extract domain-specific columns and units.
- **Shared potential conversion** (`core/potential.py`): defines electrode
  setups, reference presets and RHE/SHE conversions.
- **One parameter panel** (`widgets/parameter_panel.py`): tools declare
  `ParameterSpec`s; the form is generated and grouped automatically.
- **Results** (`widgets/results_panel.py`): plot, summary and table tabs. Tools
  can display plots while retaining detailed tables for export.
- **One Export button** (`widgets/export_button.py`): identical everywhere; the
  menu is populated from each tool's declared artifacts.
- **Projects** (`core/project.py`, `project_controller.py`): versioned `.ecwb`
  archives preserve inputs, widget state, result snapshots and export bytes.

Add a tool by placing a `Tool` subclass under `workbench/tools/<id>/` and
registering it in `workbench/tools/__init__.py`.

## Package (optional)

```powershell
python -m pip install pyinstaller
pyinstaller --noconfirm --windowed --name Workbench `
  --paths . --add-data "workbench/styles;workbench/styles" `
  --add-data "workbench/Test_files/LSV_PtRuC.txt;workbench/Test_files" `
  workbench/__main__.py
```

The bundled PtRuC file supplies the default Integrated RE calibration.

## Tests

```powershell
python -m pip install pytest
python -m pytest workbench/tests
```

For tests without a display, set `$env:QT_QPA_PLATFORM = 'offscreen'` first.
