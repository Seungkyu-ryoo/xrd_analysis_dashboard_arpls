"""Deterministic, atomic writers for dashboard export results.

Fitting and UI concerns intentionally stay outside this module.  Callers pass a
snapshot of completed fit results and may report isolated per-position failures
through ``on_error`` without allowing a partial individual file to appear.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional

import numpy as np
import pandas as pd

from ..analysis.pipeline import (
    make_param_record,
    safe_filename_component,
    split_position,
)


ErrorCallback = Callable[[str, BaseException], None]


@dataclass(frozen=True)
class ExportError:
    """One position that could not be included in an export."""

    position: str
    exception_type: str
    message: str


@dataclass(frozen=True)
class ExportSummary:
    """Outcome of an individual-file or merged-workbook export."""

    fmt: str
    save_mode: str
    requested: int
    succeeded: int
    output_paths: tuple[Path, ...]
    errors: tuple[ExportError, ...]

    @property
    def failed(self) -> int:
        """Number of requested positions not written successfully."""

        return self.requested - self.succeeded


@contextmanager
def _atomic_sibling(destination: Path) -> Iterator[Path]:
    """Yield a temporary sibling and atomically replace ``destination`` on success."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.tmp-",
        suffix=destination.suffix,
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        yield temporary_path
        os.replace(temporary_path, destination)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _individual_destinations(
    directory: Path,
    positions: tuple[str, ...],
    base_name: str,
    extension: str,
) -> dict[str, Path]:
    """Plan unique filenames, including on case-insensitive filesystems."""

    safe_base = safe_filename_component(base_name)
    used_names: set[str] = set()
    destinations: dict[str, Path] = {}
    for position in positions:
        th_value, samz_value = split_position(position)
        stem = (
            f"{safe_base}_th_{safe_filename_component(th_value)}_"
            f"samz_{safe_filename_component(samz_value)}"
        )
        filename = f"{stem}{extension}"
        suffix_number = 2
        while filename.casefold() in used_names:
            filename = f"{stem}__{suffix_number}{extension}"
            suffix_number += 1
        used_names.add(filename.casefold())
        destinations[position] = directory / filename
    return destinations


def _as_x_values(values: Iterable[float]) -> np.ndarray:
    if isinstance(values, np.ndarray):
        array = np.asarray(values, dtype=float)
    else:
        array = np.asarray(tuple(values), dtype=float)
    if array.ndim != 1:
        raise ValueError("Export X values must be a one-dimensional array.")
    return array


def _cleaned_signal(
    position: str,
    result: Mapping[str, Any],
    expected_length: int,
) -> np.ndarray:
    signal = np.asarray(result["clean_target_raw"], dtype=float)
    if signal.ndim != 1 or len(signal) != expected_length:
        raise ValueError(
            f"'{position}' cleaned signal length does not match the export X axis."
        )
    return signal


def _error_record(
    position: str,
    error: BaseException,
    errors: list[ExportError],
    on_error: Optional[ErrorCallback],
) -> None:
    errors.append(
        ExportError(
            position=position,
            exception_type=type(error).__name__,
            message=str(error),
        )
    )
    if on_error is not None:
        on_error(position, error)


def _write_xye(path: Path, x_label: str, x_values: np.ndarray, signal: np.ndarray) -> None:
    frame = pd.DataFrame({x_label: x_values, "Intensity": signal})
    with _atomic_sibling(path) as temporary_path:
        frame.to_csv(
            temporary_path,
            sep="\t",
            header=False,
            index=False,
        )


def _write_individual_excel(
    path: Path,
    x_label: str,
    x_values: np.ndarray,
    position: str,
    signal: np.ndarray,
    result: Mapping[str, Any],
) -> None:
    signal_frame = pd.DataFrame({x_label: x_values, position: signal})
    parameter_frame = pd.DataFrame([make_param_record(position, result)])
    with _atomic_sibling(path) as temporary_path:
        with pd.ExcelWriter(temporary_path, engine="openpyxl") as writer:
            signal_frame.to_excel(writer, sheet_name="Cleaned_Signal", index=False)
            parameter_frame.to_excel(
                writer,
                sheet_name="Fitting_Parameters",
                index=False,
            )


def _write_merged_excel(
    path: Path,
    signal_frame: pd.DataFrame,
    parameter_frame: pd.DataFrame,
) -> None:
    with _atomic_sibling(path) as temporary_path:
        with pd.ExcelWriter(temporary_path, engine="openpyxl") as writer:
            signal_frame.to_excel(writer, sheet_name="Cleaned_Signals", index=False)
            parameter_frame.to_excel(
                writer,
                sheet_name="Fitting_Parameters",
                index=False,
            )


def write_export(
    save_target: os.PathLike[str] | str,
    fmt: str,
    save_mode: str,
    x_label: str,
    x_values: Iterable[float],
    positions: Iterable[str],
    fit_results: Mapping[str, Mapping[str, Any]],
    base_name: str,
    on_error: Optional[ErrorCallback] = None,
) -> ExportSummary:
    """Write cleaned spectra using the dashboard's established export schemas.

    ``save_target`` is a directory for XYE and individual Excel exports, and a
    workbook path for merged Excel export.  XYE is always treated as individual
    mode, matching the dashboard UI.  Individual-position failures are isolated
    and returned in the summary; a failure to write a merged workbook is raised
    after its temporary file has been removed.

    Sanitized individual names are compared case-insensitively.  Collisions are
    resolved in input order with ``__2``, ``__3``, and so on.
    """

    normalized_format = str(fmt).strip().lower()
    normalized_mode = str(save_mode).strip().lower()
    if normalized_format not in {"xye", "excel"}:
        raise ValueError("Export format must be 'xye' or 'excel'.")
    if normalized_mode not in {"individual", "merged"}:
        raise ValueError("Export save mode must be 'individual' or 'merged'.")
    effective_mode = "individual" if normalized_format == "xye" else normalized_mode

    position_order = tuple(str(position) for position in positions)
    if len(set(position_order)) != len(position_order):
        raise ValueError("Export positions must be unique.")
    x_array = _as_x_values(x_values)
    target = Path(save_target)
    errors: list[ExportError] = []
    output_paths: list[Path] = []

    if effective_mode == "individual":
        extension = ".xye" if normalized_format == "xye" else ".xlsx"
        destinations = _individual_destinations(
            target,
            position_order,
            base_name,
            extension,
        )
        for position in position_order:
            try:
                result = fit_results[position]
                signal = _cleaned_signal(position, result, len(x_array))
                destination = destinations[position]
                if normalized_format == "xye":
                    _write_xye(destination, x_label, x_array, signal)
                else:
                    _write_individual_excel(
                        destination,
                        x_label,
                        x_array,
                        position,
                        signal,
                        result,
                    )
                output_paths.append(destination)
            except Exception as error:
                _error_record(position, error, errors, on_error)

        return ExportSummary(
            fmt=normalized_format,
            save_mode=effective_mode,
            requested=len(position_order),
            succeeded=len(output_paths),
            output_paths=tuple(output_paths),
            errors=tuple(errors),
        )

    signal_columns: dict[str, np.ndarray] = {x_label: x_array}
    parameter_records: list[dict[str, Any]] = []
    included_positions: list[str] = []
    for position in position_order:
        try:
            result = fit_results[position]
            signal_columns[position] = _cleaned_signal(
                position,
                result,
                len(x_array),
            )
            parameter_records.append(make_param_record(position, result))
            included_positions.append(position)
        except Exception as error:
            _error_record(position, error, errors, on_error)

    if not parameter_records:
        raise RuntimeError("No profile could be fitted for export.")

    try:
        _write_merged_excel(
            target,
            pd.DataFrame(signal_columns),
            pd.DataFrame(parameter_records),
        )
    except Exception as error:
        if on_error is not None:
            on_error("<merged>", error)
        raise

    output_paths.append(target)
    return ExportSummary(
        fmt=normalized_format,
        save_mode=effective_mode,
        requested=len(position_order),
        succeeded=len(included_positions),
        output_paths=tuple(output_paths),
        errors=tuple(errors),
    )


__all__ = ["ExportError", "ExportSummary", "write_export"]
