# XRD Analysis Dashboard with arPLS

Tkinter/Matplotlib application for loading multi-sheet XRD workbooks, fitting a
reference background, optionally applying arPLS, comparing scan positions, and
exporting cleaned spectra.

## Requirements and installation

- Python 3.10 or newer
- Tk 8.6 or newer
- Packages listed in `requirements.txt`

On macOS, use a Miniconda Python that includes a current Tk. Avoid virtual
environments created from Apple's Command Line Tools Python when it provides the
legacy Tk 8.5, which renders several widgets incorrectly.

From this directory:

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -c "import sys, tkinter as tk; print(sys.version); print(tk.TkVersion)"
```

Tkinter comes from the Python/Conda installation rather than pip.

## Run

The preferred launcher from this directory is:

```bash
python main.py
```

From the repository root, use package execution:

```bash
python -m xrd_analysis_dashboard_arpls.main
```

`main_optimized.py` and `xrd_analysis_dashboard_arpls.py` remain as compatibility
launchers for older scripts and commands.

## UI workflow

1. Choose the target and background workbooks, set the display Q range, and
   select **Load / reload data**.
2. Group by `th` or `samz`, choose the primary value, and select one or more
   positions to compare.
3. Choose an anchor/mask preset or edit the ranges, then select **Fitting Only**,
   **arPLS Only**, or **Fitting + arPLS** and run the analysis.
4. Review the normalized/background and cleaned-signal plots. Plot controls
   adjust overlays, offsets, references, labels, and colors without refitting.
   Left-click adds a Q marker; right-click near one removes it.
5. Choose the export format, mode, and Q or 2Theta x-axis, then export the
   selected positions.

Loading, fitting, and exporting run in background workers; Tk widgets and file
dialogs remain on the main thread.

## Input workbook contract

Target and background files use the same multi-sheet layout:

- At least one sheet must contain a numeric `Q` column (the name is matched
  case-insensitively after trimming whitespace).
- Other usable numeric columns are intensity profiles.
- Sheet names become `th` values and intensity-column names become `samz` values.
- Each profile needs at least two finite Q/intensity pairs. Duplicate Q values
  are reduced and different Q grids are merged and interpolated. Analysis
  requires at least four target Q points and four unique points in each usable
  background reference.
- The requested display Q range must contain target data. The background must
  also cover the slightly expanded Q support required by the fitting transforms.

An optional `ReferencePeaks.xlsx` supplies phase/HKL reference peaks. It is
searched for beside the target workbook, beside the application, and in the
current working directory.

## Exports

| Selection | Output |
| --- | --- |
| Excel, merged | One workbook with `Cleaned_Signals` and `Fitting_Parameters` sheets |
| Excel, individual | One workbook per position with `Cleaned_Signal` and `Fitting_Parameters` sheets |
| XYE | One tab-separated, headerless `.xye` file per position |

Exports can retain Q or convert it to 2Theta using the wavelength entered in the
UI. XYE export is always individual. Files are written through temporary sibling
files and moved into place only after a successful write.

## Module map

- `main.py` — preferred command-line entry point.
- `app.py` — Tkinter widgets, application state, and worker coordination.
- `analysis.py` — validation, parsing, coordinate conversion, fit-job
  orchestration, and shared result schemas.
- `data_io.py` — target/background workbook and reference-peak readers.
- `plotting.py` — pure Matplotlib figure rendering, independent of Tk canvas
  updates and click markers.
- `exporting.py` — atomic merged/individual Excel and XYE writers.
- `theme.py` — accessible palette, fonts, and ttk state styles.
- `fitting.py` — compatibility fitting API; `fitting_optimized.py` implements
  reference-background fitting and `fitting_arpls.py` implements arPLS.
- `main_optimized.py` and `xrd_analysis_dashboard_arpls.py` — legacy import and
  launch compatibility.

## Data preparation utilities

`preprocess_xdart_for_dashboard.py` converts xdart IQ CSV/XYE folders into
normalized target and fused-silica background workbooks plus a processing
manifest:

```bash
python preprocess_xdart_for_dashboard.py --help
```

`despike_background_workbook.py` replaces a narrow Q interval in one background
sheet by linear interpolation and records correction metadata:

```bash
python despike_background_workbook.py --help
```

## Tests

From this directory:

```bash
python -m unittest discover -s tests -v
```
