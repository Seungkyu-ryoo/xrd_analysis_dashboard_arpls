"""Prepare xdart IQ CSV files for the XRD analysis dashboard.

The transformation intentionally matches the March 2026 organizer archived in
``Archieve/Phase_analysis_closed_loop_BO/organize_1d_files.py``:

1. Sort each spectrum by Q.
2. Remove the intensity offset using the first (lowest-Q) intensity.
3. Divide by the maximum offset-corrected intensity in Q = 1.5--1.6 A^-1.

The output is one multi-sheet workbook per combi condition and one workbook for
the fused-silica reference background.  Every data sheet contains ``Q`` in the
first column, which is the format consumed by the dashboard.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


COMBI_FOLDER_RE = re.compile(r"^(?P<prefix>Combi_.+)_(?P<samz>-?\d+)samz$")
SCAN_INDEX_RE = re.compile(r"_(?P<scan>\d{4})\.(?:csv|xye)$", re.IGNORECASE)

# Recovered from the archived March 2026 HDF5_Angle_Map sheets.
DEFAULT_COMBI_ANGLE_MAP = {1: 0.200, 2: 0.230, 3: 0.260}

# Recovered from archived background-fitting records (Best_BG_Used).
DEFAULT_BACKGROUND_ANGLE_MAP = {1: 0.180, 2: 0.300}


def _natural_group_key(prefix: str) -> tuple[int, str]:
    match = re.search(r"Combi_num(\d+)", prefix, flags=re.IGNORECASE)
    return (int(match.group(1)) if match else 10**9, prefix.lower())


def discover_combi_groups(input_dir: Path) -> dict[str, list[tuple[int, Path]]]:
    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for folder in input_dir.iterdir():
        if not folder.is_dir():
            continue
        match = COMBI_FOLDER_RE.match(folder.name)
        if match:
            groups[match.group("prefix")].append((int(match.group("samz")), folder))
    return {
        prefix: sorted(rows, key=lambda row: row[0])
        for prefix, rows in sorted(groups.items(), key=lambda row: _natural_group_key(row[0]))
    }


def _scan_index(csv_path: Path) -> int:
    match = SCAN_INDEX_RE.search(csv_path.name)
    if not match:
        raise ValueError(f"Cannot parse scan index from {csv_path.name}")
    return int(match.group("scan"))


def _find_iq_files(folder: Path) -> list[Path]:
    """Return one IQ file per scan, preferring CSV and falling back to XYE."""
    by_scan: dict[int, Path] = {}
    for pattern in ("iq_*.xye", "iq_*.csv"):
        for path in folder.glob(pattern):
            by_scan[_scan_index(path)] = path
    return [by_scan[index] for index in sorted(by_scan)]


def _read_and_normalize(
    csv_path: Path,
    norm_q_min: float,
    norm_q_max: float,
    offset_q_min: float | None = None,
    offset_q_max: float | None = None,
) -> tuple[pd.Series, dict[str, float | int | str]]:
    # xdart writes comma-separated CSV and tab/space-separated XYE with the same
    # three columns.  The regex accepts both without changing their values.
    raw = pd.read_csv(
        csv_path,
        sep=r"\s*,\s*|\s+",
        engine="python",
        header=None,
        names=["Q", "intensity", "2theta"],
        usecols=[0, 1, 2],
    )
    raw["Q"] = pd.to_numeric(raw["Q"], errors="coerce")
    raw["intensity"] = pd.to_numeric(raw["intensity"], errors="coerce")
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna(subset=["Q", "intensity"])
    raw = raw.sort_values("Q").drop_duplicates(subset="Q", keep="first")
    if len(raw) < 2:
        raise ValueError(f"Fewer than two finite Q/intensity rows in {csv_path}")
    if not raw["Q"].is_monotonic_increasing:
        raise ValueError(f"Q is not increasing in {csv_path}")

    if offset_q_min is None and offset_q_max is None:
        offset_value = float(raw["intensity"].iloc[0])
        offset_reference_q = float(raw["Q"].iloc[0])
        offset_method = "intensity_at_lowest_Q"
    elif offset_q_min is not None and offset_q_max is not None:
        offset_mask = raw["Q"].between(offset_q_min, offset_q_max, inclusive="both")
        if not offset_mask.any():
            raise ValueError(
                f"No Q points in offset window {offset_q_min:g}--{offset_q_max:g} "
                f"for {csv_path}"
            )
        offset_window = raw.loc[offset_mask, ["Q", "intensity"]]
        offset_index = offset_window["intensity"].idxmin()
        offset_value = float(raw.loc[offset_index, "intensity"])
        offset_reference_q = float(raw.loc[offset_index, "Q"])
        offset_method = "minimum_intensity_in_Q_window"
    else:
        raise ValueError("offset_q_min and offset_q_max must be provided together")

    offset_corrected = raw["intensity"] - offset_value
    window_mask = raw["Q"].between(norm_q_min, norm_q_max, inclusive="both")
    if not window_mask.any():
        raise ValueError(
            f"No Q points in normalization window {norm_q_min:g}--{norm_q_max:g} "
            f"for {csv_path}"
        )
    normalization_factor = float(offset_corrected.loc[window_mask].max())
    if not np.isfinite(normalization_factor) or normalization_factor <= 0:
        raise ValueError(
            f"Non-positive normalization factor {normalization_factor!r} for {csv_path}"
        )

    normalized = pd.Series(
        offset_corrected.to_numpy(dtype=float) / normalization_factor,
        index=raw["Q"].to_numpy(dtype=float),
        dtype=float,
    )
    normalized.index.name = "Q"
    qc = {
        "source_file": str(csv_path),
        "source_format": csv_path.suffix.lower().lstrip("."),
        "scan_index": _scan_index(csv_path),
        "n_points": int(len(normalized)),
        "q_min": float(normalized.index.min()),
        "q_max": float(normalized.index.max()),
        "offset_method": offset_method,
        "offset_q_min": offset_q_min,
        "offset_q_max": offset_q_max,
        "offset_reference_q": offset_reference_q,
        "offset_subtracted": offset_value,
        "normalization_q_min": float(norm_q_min),
        "normalization_q_max": float(norm_q_max),
        "normalization_factor": normalization_factor,
        "normalized_first_value": float(normalized.iloc[0]),
        "normalized_window_max": float(normalized.loc[window_mask.to_numpy()].max()),
        "status": "ok",
    }
    return normalized, qc


def _merge_series(series: Iterable[pd.Series]) -> pd.DataFrame:
    merged = pd.concat(list(series), axis=1).sort_index()
    merged.index.name = "Q"
    return merged.reset_index()


def _metadata_frame(
    *,
    dataset_type: str,
    source_dir: Path,
    output_path: Path,
    norm_q_min: float,
    norm_q_max: float,
    angle_map: dict[int, float],
    offset_q_min: float | None = None,
    offset_q_max: float | None = None,
) -> pd.DataFrame:
    if offset_q_min is None:
        offset_method = "intensity - intensity_at_lowest_Q"
    else:
        offset_method = (
            "intensity - minimum_intensity_in_Q_window "
            f"[{offset_q_min:g}, {offset_q_max:g}]"
        )
    rows = [
        ("created_at", datetime.now().isoformat(timespec="seconds")),
        ("dataset_type", dataset_type),
        ("source_dir", str(source_dir)),
        ("output_workbook", str(output_path)),
        ("offset_method", offset_method),
        ("offset_q_min", offset_q_min),
        ("offset_q_max", offset_q_max),
        ("normalization_method", "offset_corrected / max_in_Q_window"),
        ("normalization_q_min", norm_q_min),
        ("normalization_q_max", norm_q_max),
        ("scan_to_angle_map", ", ".join(f"{k}:{v:.3f}" for k, v in angle_map.items())),
    ]
    return pd.DataFrame(rows, columns=["key", "value"])


def _write_workbook(
    output_path: Path,
    metadata: pd.DataFrame,
    source_records: list[dict[str, object]],
    sheets: dict[float | str, pd.DataFrame],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.tmp.xlsx")
    with pd.ExcelWriter(temporary_path, engine="openpyxl") as writer:
        metadata.to_excel(writer, sheet_name="Metadata", index=False)
        pd.DataFrame(source_records).to_excel(writer, sheet_name="Source_Files", index=False)
        for sheet_key, data in sheets.items():
            sheet_name = f"{sheet_key:.3f}" if isinstance(sheet_key, float) else str(sheet_key)
            data.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    temporary_path.replace(output_path)


def process_combi_group(
    prefix: str,
    folders: list[tuple[int, Path]],
    output_dir: Path,
    norm_q_min: float,
    norm_q_max: float,
    angle_map: dict[int, float],
    offset_q_min: float | None = None,
    offset_q_max: float | None = None,
) -> tuple[Path, list[dict[str, object]]]:
    by_sheet: dict[float | str, list[pd.Series]] = defaultdict(list)
    source_records: list[dict[str, object]] = []
    for samz, folder in folders:
        csv_files = _find_iq_files(folder)
        if not csv_files:
            source_records.append(
                {
                    "dataset": prefix,
                    "folder_name": folder.name,
                    "samz": samz,
                    "source_file": "",
                    "status": "skipped_no_iq_csv_or_xye",
                }
            )
            continue
        for csv_path in csv_files:
            scan_index = _scan_index(csv_path)
            sheet_key: float | str = angle_map.get(scan_index, f"scan_{scan_index}")
            normalized, qc = _read_and_normalize(
                csv_path,
                norm_q_min,
                norm_q_max,
                offset_q_min=offset_q_min,
                offset_q_max=offset_q_max,
            )
            by_sheet[sheet_key].append(normalized.rename(str(samz)))
            source_records.append(
                {
                    "dataset": prefix,
                    "folder_name": folder.name,
                    "samz": samz,
                    "sheet": f"{sheet_key:.3f}" if isinstance(sheet_key, float) else sheet_key,
                    **qc,
                }
            )

    if not by_sheet:
        raise RuntimeError(f"No usable IQ CSV/XYE spectra found for {prefix}")
    sheets = {key: _merge_series(values) for key, values in sorted(by_sheet.items(), key=str)}
    output_path = output_dir / "targets" / f"{prefix}_norm.xlsx"
    metadata = _metadata_frame(
        dataset_type="target_combi",
        source_dir=folders[0][1].parent,
        output_path=output_path,
        norm_q_min=norm_q_min,
        norm_q_max=norm_q_max,
        angle_map=angle_map,
        offset_q_min=offset_q_min,
        offset_q_max=offset_q_max,
    )
    _write_workbook(output_path, metadata, source_records, sheets)
    return output_path, source_records


def process_background(
    background_dir: Path,
    output_dir: Path,
    norm_q_min: float,
    norm_q_max: float,
    angle_map: dict[int, float],
) -> tuple[Path, list[dict[str, object]]]:
    csv_files = _find_iq_files(background_dir)
    if not csv_files:
        raise FileNotFoundError(f"No iq_*.csv or iq_*.xye files in {background_dir}")

    label_match = re.search(r"_(\d+)$", background_dir.name)
    reference_label = label_match.group(1) if label_match else background_dir.name
    sheets: dict[float | str, pd.DataFrame] = {}
    source_records: list[dict[str, object]] = []
    for csv_path in csv_files:
        scan_index = _scan_index(csv_path)
        sheet_key: float | str = angle_map.get(scan_index, f"scan_{scan_index}")
        normalized, qc = _read_and_normalize(csv_path, norm_q_min, norm_q_max)
        sheets[sheet_key] = _merge_series([normalized.rename(reference_label)])
        source_records.append(
            {
                "dataset": background_dir.name,
                "folder_name": background_dir.name,
                "reference_label": reference_label,
                "sheet": f"{sheet_key:.3f}" if isinstance(sheet_key, float) else sheet_key,
                **qc,
            }
        )

    output_path = output_dir / "background" / f"{background_dir.name}_norm.xlsx"
    metadata = _metadata_frame(
        dataset_type="fused_silica_background",
        source_dir=background_dir,
        output_path=output_path,
        norm_q_min=norm_q_min,
        norm_q_max=norm_q_max,
        angle_map=angle_map,
    )
    _write_workbook(output_path, metadata, source_records, sheets)
    return output_path, source_records


def _validate_workbook(
    path: Path,
    norm_q_min: float,
    norm_q_max: float,
    offset_q_min: float | None = None,
    offset_q_max: float | None = None,
) -> list[dict[str, object]]:
    issues: list[str] = []
    data_sheet_count = 0
    spectrum_count = 0
    missing_intensity_cells = 0
    with pd.ExcelFile(path) as workbook:
        for sheet_name in workbook.sheet_names:
            data = pd.read_excel(workbook, sheet_name=sheet_name)
            q_columns = [column for column in data.columns if str(column).strip().lower() == "q"]
            if not q_columns:
                continue
            data_sheet_count += 1
            q = pd.to_numeric(data[q_columns[0]], errors="coerce").to_numpy(dtype=float)
            if not np.all(np.isfinite(q)) or not np.all(np.diff(q) > 0):
                issues.append(f"{sheet_name}: Q is not finite and strictly increasing")
            for column in (column for column in data.columns if column != q_columns[0]):
                y = pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float)
                spectrum_count += 1
                finite = np.isfinite(y)
                missing_intensity_cells += int(np.count_nonzero(~finite))
                if np.count_nonzero(finite) < 2:
                    issues.append(f"{sheet_name}/{column}: fewer than two finite intensities")
                    continue
                # Different xdart integrations can have different Q grids.  The
                # organizer preserves those gaps, and the dashboard interpolates
                # them after loading.  Validate each trace on its own finite grid.
                if offset_q_min is None:
                    if not np.isclose(y[finite][0], 0.0, atol=1e-12):
                        issues.append(f"{sheet_name}/{column}: first finite value is not zero")
                else:
                    offset_window = finite & (q >= offset_q_min) & (q <= offset_q_max)
                    if not np.any(offset_window) or not np.isclose(
                        np.min(y[offset_window]), 0.0, atol=1e-12
                    ):
                        issues.append(
                            f"{sheet_name}/{column}: offset-window minimum is not zero"
                        )
                window = finite & (q >= norm_q_min) & (q <= norm_q_max)
                if not np.any(window) or not np.isclose(np.max(y[window]), 1.0, atol=1e-10):
                    issues.append(f"{sheet_name}/{column}: normalization-window max is not one")
    return [
        {
            "workbook": str(path),
            "data_sheets": data_sheet_count,
            "spectra": spectrum_count,
            "missing_intensity_cells": missing_intensity_cells,
            "validation": "ok" if not issues else "failed",
            "issues": " | ".join(issues),
        }
    ]


def _write_run_summary(
    output_dir: Path,
    file_records: list[dict[str, object]],
    validation_records: list[dict[str, object]],
    input_dir: Path,
    norm_q_min: float,
    norm_q_max: float,
    offset_q_min: float | None,
    offset_q_max: float | None,
) -> None:
    summary_path = output_dir / "processing_manifest.xlsx"
    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        pd.DataFrame(file_records).to_excel(writer, sheet_name="Source_Files", index=False)
        pd.DataFrame(validation_records).to_excel(writer, sheet_name="Output_Validation", index=False)

    if offset_q_min is None:
        offset_description = "the intensity at the lowest Q value"
    else:
        offset_description = (
            f"the minimum intensity in Q = {offset_q_min:g}–{offset_q_max:g} Å⁻¹"
        )

    readme = f"""# March 2026 XRD dashboard-ready data

Source: `{input_dir}`

Processing applied to every IQ spectrum:

1. Sort by Q and remove duplicate Q rows.
2. Subtract {offset_description} (offset removal).
3. Normalize by the maximum offset-corrected intensity in Q = {norm_q_min:g}–{norm_q_max:g} Å⁻¹.

Angle sheets:

- Combi scans: 0001 → 0.200, 0002 → 0.230, 0003 → 0.260.
- Fused-silica scans: 0001 → 0.180, 0002 → 0.300.

Use a workbook under `targets/` as the target file in
`xrd_analysis_dashboard_arpls`, and use the workbook under `background/` as the
background file. `processing_manifest.xlsx` contains source-level normalization
factors and output validation results.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--norm-q-min", type=float, default=1.5)
    parser.add_argument("--norm-q-max", type=float, default=1.6)
    parser.add_argument(
        "--offset-q-min",
        type=float,
        help="Use the minimum intensity in this Q-window as the target offset.",
    )
    parser.add_argument(
        "--offset-q-max",
        type=float,
        help="Upper bound paired with --offset-q-min.",
    )
    parser.add_argument(
        "--prefix",
        action="append",
        dest="prefixes",
        help="Process only this exact combi prefix; may be provided more than once.",
    )
    parser.add_argument(
        "--skip-background", action="store_true", help="Do not create the fused-silica workbook."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace outputs created by an earlier run in a non-empty output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if args.norm_q_min >= args.norm_q_max:
        raise ValueError("--norm-q-min must be less than --norm-q-max")
    if (args.offset_q_min is None) != (args.offset_q_max is None):
        raise ValueError("--offset-q-min and --offset-q-max must be provided together")
    if args.offset_q_min is not None and args.offset_q_min >= args.offset_q_max:
        raise ValueError("--offset-q-min must be less than --offset-q-max")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    groups = discover_combi_groups(input_dir)
    if args.prefixes:
        missing = sorted(set(args.prefixes) - set(groups))
        if missing:
            raise ValueError(f"Requested combi prefixes were not found: {missing}")
        groups = {prefix: groups[prefix] for prefix in args.prefixes}
    if not groups:
        raise RuntimeError(f"No combi samz folders found in {input_dir}")

    output_paths: list[Path] = []
    file_records: list[dict[str, object]] = []
    for index, (prefix, folders) in enumerate(groups.items(), start=1):
        print(f"[{index}/{len(groups)}] {prefix}: {len(folders)} samz folders", flush=True)
        path, records = process_combi_group(
            prefix,
            folders,
            output_dir,
            args.norm_q_min,
            args.norm_q_max,
            DEFAULT_COMBI_ANGLE_MAP,
            offset_q_min=args.offset_q_min,
            offset_q_max=args.offset_q_max,
        )
        output_paths.append(path)
        file_records.extend(records)

    if not args.skip_background:
        candidates = sorted(path for path in input_dir.glob("Fused_Silica*") if path.is_dir())
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected exactly one Fused_Silica directory, found {len(candidates)}: {candidates}"
            )
        print(f"[background] {candidates[0].name}", flush=True)
        path, records = process_background(
            candidates[0],
            output_dir,
            args.norm_q_min,
            args.norm_q_max,
            DEFAULT_BACKGROUND_ANGLE_MAP,
        )
        output_paths.append(path)
        file_records.extend(records)

    validation_records: list[dict[str, object]] = []
    for index, path in enumerate(output_paths, start=1):
        print(f"[validate {index}/{len(output_paths)}] {path.name}", flush=True)
        validation_records.extend(
            _validate_workbook(
                path,
                args.norm_q_min,
                args.norm_q_max,
                offset_q_min=args.offset_q_min,
                offset_q_max=args.offset_q_max,
            )
        )
    failures = [row for row in validation_records if row["validation"] != "ok"]
    _write_run_summary(
        output_dir,
        file_records,
        validation_records,
        input_dir,
        args.norm_q_min,
        args.norm_q_max,
        args.offset_q_min,
        args.offset_q_max,
    )
    if failures:
        raise RuntimeError(f"Validation failed for {len(failures)} workbook(s)")

    print(
        f"Complete: {len(groups)} target workbook(s), "
        f"{0 if args.skip_background else 1} background workbook, "
        f"{len(file_records)} source spectra -> {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
