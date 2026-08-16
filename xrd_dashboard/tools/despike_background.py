"""Linearly replace a narrow artifact in one background-workbook Q interval."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def interpolate_interval(
    data: pd.DataFrame, q_min: float, q_max: float
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    corrected = data.copy()
    q_columns = [column for column in corrected.columns if str(column).strip().lower() == "q"]
    if len(q_columns) != 1:
        raise ValueError(f"Expected exactly one Q column; found {q_columns}")
    q_column = q_columns[0]
    q = pd.to_numeric(corrected[q_column], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(q)) or not np.all(np.diff(q) > 0):
        raise ValueError("Q must be finite and strictly increasing")

    replace_mask = (q >= q_min) & (q <= q_max)
    left_candidates = np.flatnonzero(q < q_min)
    right_candidates = np.flatnonzero(q > q_max)
    if not np.any(replace_mask):
        raise ValueError(f"No Q points found in {q_min:g}--{q_max:g}")
    if len(left_candidates) == 0 or len(right_candidates) == 0:
        raise ValueError("Interpolation interval must have data on both sides")
    left_index = int(left_candidates[-1])
    right_index = int(right_candidates[0])

    logs: list[dict[str, object]] = []
    for column in (column for column in corrected.columns if column != q_column):
        values = pd.to_numeric(corrected[column], errors="coerce").to_numpy(
            dtype=float, copy=True
        )
        if not np.isfinite(values[left_index]) or not np.isfinite(values[right_index]):
            raise ValueError(f"{column}: non-finite interpolation boundary")
        before = values.copy()
        values[replace_mask] = np.interp(
            q[replace_mask],
            [q[left_index], q[right_index]],
            [values[left_index], values[right_index]],
        )
        corrected[column] = values
        local = (q >= q[left_index]) & (q <= q[right_index])
        logs.append(
            {
                "signal_column": str(column),
                "requested_q_min": q_min,
                "requested_q_max": q_max,
                "left_boundary_q": q[left_index],
                "left_boundary_value": before[left_index],
                "right_boundary_q": q[right_index],
                "right_boundary_value": before[right_index],
                "replaced_points": int(np.count_nonzero(replace_mask)),
                "before_local_max": float(np.max(before[local])),
                "after_local_max": float(np.max(values[local])),
                "before_max_adjacent_jump": float(np.max(np.abs(np.diff(before[local])))),
                "after_max_adjacent_jump": float(np.max(np.abs(np.diff(values[local])))),
            }
        )
    return corrected, logs


def correct_workbook(
    input_path: Path,
    output_path: Path,
    sheet_name: str,
    q_min: float,
    q_max: float,
    overwrite: bool = False,
) -> list[dict[str, object]]:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    sheets = pd.read_excel(input_path, sheet_name=None)
    if sheet_name not in sheets:
        raise ValueError(f"Sheet {sheet_name!r} not found; available: {list(sheets)}")

    original = sheets[sheet_name]
    corrected, logs = interpolate_interval(original, q_min, q_max)
    sheets[sheet_name] = corrected

    metadata = sheets.get("Metadata")
    correction_rows = pd.DataFrame(
        [
            ("despike_created_at", datetime.now().isoformat(timespec="seconds")),
            ("despike_source_workbook", str(input_path)),
            ("despike_sheet", sheet_name),
            ("despike_method", "linear interpolation between nearest outside points"),
            ("despike_q_min", q_min),
            ("despike_q_max", q_max),
        ],
        columns=["key", "value"],
    )
    if metadata is not None and list(metadata.columns) == ["key", "value"]:
        sheets["Metadata"] = pd.concat([metadata, correction_rows], ignore_index=True)
    else:
        sheets["Correction_Metadata"] = correction_rows
    sheets["Despike_Log"] = pd.DataFrame(logs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.tmp.xlsx")
    with pd.ExcelWriter(temporary_path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=str(name)[:31], index=False)
    temporary_path.replace(output_path)

    reloaded = pd.read_excel(output_path, sheet_name=sheet_name)
    q_column = next(column for column in original if str(column).strip().lower() == "q")
    q = pd.to_numeric(original[q_column], errors="coerce").to_numpy(dtype=float)
    outside = ~((q >= q_min) & (q <= q_max))
    for column in (column for column in original if column != q_column):
        before = pd.to_numeric(original[column], errors="coerce").to_numpy(dtype=float)
        after = pd.to_numeric(reloaded[column], errors="coerce").to_numpy(dtype=float)
        if not np.allclose(before[outside], after[outside], rtol=0, atol=1e-14, equal_nan=True):
            raise RuntimeError(f"Unexpected change outside correction interval in {column}")
    return logs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sheet", default="0.300")
    parser.add_argument("--q-min", required=True, type=float)
    parser.add_argument("--q-max", required=True, type=float)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not np.isfinite(args.q_min) or not np.isfinite(args.q_max) or args.q_min >= args.q_max:
        raise ValueError("Require finite --q-min < --q-max")
    logs = correct_workbook(
        args.input.expanduser().resolve(),
        args.output.expanduser().resolve(),
        args.sheet,
        args.q_min,
        args.q_max,
        overwrite=args.overwrite,
    )
    for record in logs:
        print(
            f"{record['signal_column']}: replaced {record['replaced_points']} points; "
            f"local max {record['before_local_max']:.8g} -> {record['after_local_max']:.8g}; "
            f"max jump {record['before_max_adjacent_jump']:.8g} -> "
            f"{record['after_max_adjacent_jump']:.8g}"
        )
    print(f"Wrote corrected background: {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
