# XRD Analysis Dashboard With arPLS

Interactive Tkinter dashboard for cleaning and inspecting combinatorial XRD datasets. The app loads target and background multi-sheet Excel workbooks, fits a reference background, optionally applies arPLS baseline correction, visualizes cleaned profiles, and exports selected spectra.

## Features

- Multi-sheet Excel parsing for `th` / `samz` scan grids.
- Target/background comparison with selectable scan positions.
- Anchor-weighted reference background fitting.
- Optional arPLS residual baseline subtraction.
- Presets for HZO/TiN peak masks and anchor regions.
- Reference peak overlay support through an optional `ReferencePeaks.xlsx` file.
- Export to merged Excel, individual Excel files, or `.xye` text files.

## Files

```text
xrd_analysis_dashboard_arpls.py  # Tkinter GUI application
fitting.py                       # Reference background optimization
fitting_arpls.py                 # arPLS baseline helper
requirements.txt
```

## Usage

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the dashboard:

```powershell
python xrd_analysis_dashboard_arpls.py
```

The app expects target/background Excel workbooks with a `Q` column in each sheet. Signal columns are treated as `samz` positions, and sheet names are treated as `th` positions.

## Optional Data

Place `ReferencePeaks.xlsx` next to the script if you want reference phase markers overlaid in the plots. Raw experimental data files are not included in this repository.
