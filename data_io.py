"""Workbook readers used by the XRD dashboard.

The functions here are independent of Tkinter so loading and validation can run
in worker threads and be covered by ordinary unit tests.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:  # Support package imports and execution from this directory.
    from .analysis import numeric_sort_key, validate_q_range
except ImportError:  # pragma: no cover - exercised by direct-script imports.
    from analysis import numeric_sort_key, validate_q_range


REFERENCE_PHASES = ("m-HfO2", "t-ZrO2", "o-HfO2", "HfTiO2", "TiN")
REFERENCE_COLORS = ("#C2414B", "#2F855A", "#3157A4", "#C05621", "#7B2CBF")
REFERENCE_FILENAME = "ReferencePeaks.xlsx"


def read_multisheet(
    path: os.PathLike[str] | str,
    q_min: float,
    q_max: float,
) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    """Read a multi-sheet workbook into the dashboard's position-column schema.

    Each usable sheet needs a numeric ``Q`` column.  Every other usable numeric
    column becomes ``th:<sheet>_samz:<column>``.  Different Q grids are merged
    and interpolated onto their combined sorted index.
    """

    validate_q_range(q_min, q_max)
    sheets = pd.read_excel(path, sheet_name=None)
    all_series: list[pd.Series] = []
    th_values: set[str] = set()
    samz_values: set[str] = set()
    position_names: set[str] = set()

    for sheet_name, frame in sheets.items():
        frame = frame.copy()
        frame.columns = frame.columns.astype(str).str.strip()
        q_columns = [column for column in frame.columns if column.lower() == "q"]
        if not q_columns:
            warnings.warn(
                f"Sheet '{sheet_name}' was skipped because it has no Q column.",
                stacklevel=2,
            )
            continue

        q_column = q_columns[0]
        q_values = pd.to_numeric(frame[q_column], errors="coerce").to_numpy(
            dtype=float
        )
        accepted_in_sheet = False
        for column in (column for column in frame.columns if column != q_column):
            lowered = column.lower()
            if "unnamed" in lowered or "nan" in lowered:
                continue
            intensity = pd.to_numeric(frame[column], errors="coerce").to_numpy(
                dtype=float
            )
            valid = np.isfinite(q_values) & np.isfinite(intensity)
            if np.count_nonzero(valid) < 2:
                warnings.warn(
                    f"Column '{column}' in sheet '{sheet_name}' was skipped because "
                    "it has fewer than two finite Q/intensity pairs.",
                    stacklevel=2,
                )
                continue

            position_name = f"th:{sheet_name}_samz:{column}"
            if position_name in position_names:
                raise ValueError(f"Duplicate scan position: {position_name}")
            position_names.add(position_name)
            series = pd.Series(
                intensity[valid],
                index=q_values[valid],
                name=position_name,
                dtype=float,
            )
            series = series[~series.index.duplicated(keep="first")].sort_index()
            all_series.append(series)
            samz_values.add(str(column))
            accepted_in_sheet = True
        if accepted_in_sheet:
            th_values.add(str(sheet_name))

    if not all_series:
        raise ValueError(
            f"No usable sheet with a 'Q' column found in {Path(path).name}"
        )

    merged = (
        pd.concat(all_series, axis=1)
        .sort_index()
        .interpolate(method="index")
        .ffill()
        .bfill()
    )
    merged.insert(0, "Q", merged.index.to_numpy(dtype=float))
    merged.reset_index(drop=True, inplace=True)
    merged = merged[(merged["Q"] >= q_min) & (merged["Q"] <= q_max)]
    if merged.empty:
        raise ValueError(
            f"No finite data points remain inside Q range {q_min:g}–{q_max:g} "
            f"in {Path(path).name}."
        )

    return (
        merged,
        merged["Q"].to_numpy(dtype=float),
        sorted(th_values, key=numeric_sort_key),
        sorted(samz_values, key=numeric_sort_key),
    )


def _format_hkl(row: pd.Series) -> str:
    try:
        h_value, k_value, l_value = int(row["h"]), int(row["k"]), int(row["l"])
    except (TypeError, ValueError):
        return ""

    def component(value: int) -> str:
        return f"\\bar{{{abs(value)}}}" if value < 0 else str(value)

    return (
        f"$({component(h_value)}{component(k_value)}{component(l_value)})$"
    )


def load_reference_peaks(
    path: os.PathLike[str] | str,
    q_min: float,
    q_max: float,
    known_phases: Iterable[str] = REFERENCE_PHASES,
) -> tuple[dict[str, pd.DataFrame], Optional[str]]:
    """Parse the five-column phase blocks in ``ReferencePeaks.xlsx``.

    Malformed blocks are isolated and summarized in the returned warning text;
    a completely unreadable workbook returns an empty mapping and an error.
    """

    try:
        raw = pd.read_excel(path)
    except Exception as exc:
        return {}, f"Could not read {path}: {exc}"

    reference_data: dict[str, pd.DataFrame] = {}
    issues = []
    phase_names = tuple(known_phases)
    for block_index in range(len(raw.columns) // 5):
        try:
            columns = raw.columns[block_index * 5 : block_index * 5 + 5]
            q_column, h_column, k_column, l_column, intensity_column = columns
            phase_name = str(intensity_column).strip()
            phase_name = next(
                (known for known in phase_names if known in phase_name), phase_name
            )

            phase = raw[
                [q_column, h_column, k_column, l_column, intensity_column]
            ].copy()
            phase.columns = ["Q", "h", "k", "l", "Intensity"]
            phase["Q"] = pd.to_numeric(phase["Q"], errors="coerce")
            phase["Intensity"] = pd.to_numeric(phase["Intensity"], errors="coerce")
            phase = phase.dropna(subset=["Q", "Intensity"])
            phase = phase[
                np.isfinite(phase["Q"]) & np.isfinite(phase["Intensity"])
            ]
            phase = phase[
                (phase["Q"] >= q_min)
                & (phase["Q"] <= q_max)
                & (phase["Intensity"] > 1)
            ].copy()
            if phase.empty:
                continue

            phase = phase.sort_values("Intensity", ascending=False)
            phase["Q_round"] = phase["Q"].round(2)
            phase = phase.drop_duplicates("Q_round", keep="first").sort_values("Q")
            phase["Intensity_Norm"] = phase["Intensity"] / phase["Intensity"].max()
            phase["hkl"] = phase.apply(_format_hkl, axis=1)
            reference_data[phase_name] = phase
        except Exception as exc:
            issues.append(f"reference block {block_index + 1}: {exc}")

    return reference_data, "; ".join(issues) if issues else None


__all__ = [
    "REFERENCE_COLORS",
    "REFERENCE_FILENAME",
    "REFERENCE_PHASES",
    "load_reference_peaks",
    "read_multisheet",
]
