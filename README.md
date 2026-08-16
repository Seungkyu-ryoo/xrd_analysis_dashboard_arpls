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

All commands below assume the repository root (the directory containing
`main.py`). From this directory:

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -c "import sys, tkinter as tk; print(sys.version); print(tk.TkVersion)"
```

Tkinter comes from the Python/Conda installation rather than pip.

## Run

The preferred launcher is intentionally kept at the repository root:

```bash
python main.py
```

The equivalent package command is:

```bash
python -m xrd_dashboard
```

`main_optimized.py`, `fitting.py`, and `fitting_arpls.py` are small compatibility
shims for older beamtime scripts. New code should import from `xrd_dashboard`.

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

The bundled `xrd_dashboard/resources/ReferencePeaks.xlsx` supplies phase/HKL
reference peaks. A file beside the target workbook takes priority, followed by
the bundled file, the legacy repository-root location, and the current working
directory.

## Exports

| Selection | Output |
| --- | --- |
| Excel, merged | One workbook with `Cleaned_Signals` and `Fitting_Parameters` sheets |
| Excel, individual | One workbook per position with `Cleaned_Signal` and `Fitting_Parameters` sheets |
| XYE | One tab-separated, headerless `.xye` file per position |

Exports can retain Q or convert it to 2Theta using the wavelength entered in the
UI. XYE export is always individual. Files are written through temporary sibling
files and moved into place only after a successful write.

## Project structure

```text
.
├── main.py                       # preferred launcher
├── xrd_dashboard/
│   ├── analysis/                 # fitting, arPLS, validation, job orchestration
│   │   ├── baseline.py
│   │   ├── fitting.py
│   │   └── pipeline.py
│   ├── data/                     # workbook readers and result writers
│   │   ├── workbooks.py
│   │   └── exporting.py
│   ├── ui/                       # Tk application, plots, and visual theme
│   │   ├── app.py
│   │   ├── plotting.py
│   │   └── theme.py
│   ├── tools/                    # standalone data-preparation commands
│   ├── resources/                # bundled reference-peak workbook
│   └── paths.py                  # stable project/resource locations
├── tests/                        # regression tests
├── main_optimized.py             # legacy batch-import shim
├── fitting.py                    # legacy fitting shim
└── fitting_arpls.py              # legacy arPLS shim
```

The `__init__.py` files expose the small public API for each package; detailed
implementations stay in the role-specific modules shown above.

## Data preparation utilities

`xrd_dashboard.tools.preprocess_xdart` converts xdart IQ CSV/XYE folders into
normalized target and fused-silica background workbooks plus a processing
manifest:

```bash
python -m xrd_dashboard.tools.preprocess_xdart --help
```

`xrd_dashboard.tools.despike_background` replaces a narrow Q interval in one background
sheet by linear interpolation and records correction metadata:

```bash
python -m xrd_dashboard.tools.despike_background --help
```

## Tests

From this directory:

```bash
python -m unittest discover -s tests -v
```
