"""Pure analysis helpers shared by the dashboard and batch callers.

This module deliberately has no Tkinter or Matplotlib dependencies.  Keeping
the numerical orchestration here makes it usable from tests and command-line
workflows without constructing a GUI.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable, Mapping, Optional

import numpy as np

from .baseline import run_arpls
from .fitting import FLEX_BOUNDS, RIGID_BOUNDS


MAX_FIT_WORKERS = 4
BACKGROUND_Q_GUARD = 0.02
_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def numeric_sort_key(value: Any) -> tuple[int, Any]:
    """Sort numeric-looking values numerically and other values alphabetically."""

    try:
        return 0, float(value)
    except (TypeError, ValueError):
        return 1, str(value).casefold()


def parse_anchor_lines(text: str) -> list[tuple[float, float, float]]:
    """Parse ``minimum, maximum, weight`` rows from the anchor editor."""

    anchors = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise ValueError(
                f"Anchor line {line_number}: expected 'min, max, weight'."
            )
        start, end, weight = map(float, parts)
        if not all(np.isfinite((start, end, weight))):
            raise ValueError(f"Anchor line {line_number}: values must be finite.")
        if start >= end:
            raise ValueError(f"Anchor line {line_number}: min must be less than max.")
        if weight <= 0:
            raise ValueError(f"Anchor line {line_number}: weight must be positive.")
        anchors.append((start, end, weight))
    return anchors


def parse_peak_lines(text: str) -> list[tuple[float, float]]:
    """Parse ``minimum, maximum`` rows from the peak-mask editor."""

    peaks = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Peak line {line_number}: expected 'min, max'.")
        start, end = map(float, parts)
        if not all(np.isfinite((start, end))):
            raise ValueError(f"Peak line {line_number}: values must be finite.")
        if start >= end:
            raise ValueError(f"Peak line {line_number}: min must be less than max.")
        peaks.append((start, end))
    return peaks


def q_to_2theta(q_array: Iterable[float], wavelength: float) -> np.ndarray:
    """Convert reciprocal-space Q values to degrees 2-theta."""

    values = np.asarray(q_array, dtype=float)
    sin_theta = np.clip((values * wavelength) / (4.0 * np.pi), -1.0, 1.0)
    return 2.0 * np.degrees(np.arcsin(sin_theta))


def split_position(position: str) -> tuple[str, str]:
    """Split ``th:<value>_samz:<value>`` into its two display values."""

    body = position[3:] if position.startswith("th:") else position
    th_value, separator, samz_value = body.partition("_samz:")
    return (th_value, samz_value) if separator else (body, "")


def safe_filename_component(value: Any) -> str:
    """Return a non-empty filename component valid on common platforms."""

    cleaned = _INVALID_FILENAME_CHARS.sub("_", str(value)).strip(" .")
    return cleaned or "unnamed"


def validate_q_range(q_min: float, q_max: float) -> None:
    """Validate a finite, increasing display Q range."""

    if not np.isfinite(q_min) or not np.isfinite(q_max):
        raise ValueError("Q range values must be finite numbers.")
    if q_min >= q_max:
        raise ValueError("Q minimum must be less than Q maximum.")


def expanded_background_q_range(
    q_min: float,
    q_max: float,
    guard: float = BACKGROUND_Q_GUARD,
) -> tuple[float, float]:
    """Return the reference support needed by every allowed Q transform."""

    validate_q_range(q_min, q_max)
    if not np.isfinite(guard) or guard < 0:
        raise ValueError("Background Q guard must be a finite non-negative number.")

    transformed_limits = [
        q_scale * q_value - dq
        for q_value in (q_min, q_max)
        for bounds in (RIGID_BOUNDS, FLEX_BOUNDS)
        for q_scale in bounds[1]
        for dq in bounds[2]
    ]
    return min(transformed_limits) - guard, max(transformed_limits) + guard


def validate_background_q_support(
    q_values: Iterable[float], required_min: float, required_max: float
) -> None:
    """Reject background data that cannot support all fitting transforms."""

    values = np.asarray(q_values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Background Q data must be a non-empty finite 1D array.")
    available_min = float(np.min(values))
    available_max = float(np.max(values))
    if available_min > required_min or available_max < required_max:
        raise ValueError(
            "Background data does not cover the Q support required by fitting "
            f"({required_min:g}–{required_max:g}). Available support is "
            f"{available_min:g}–{available_max:g}."
        )


def compute_position(
    position: str,
    params: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit and clean one target position using the requested processing mode."""

    q_target = np.asarray(context["q_target"], dtype=float)
    target_raw = context["df_target"][position].to_numpy(dtype=float)
    if target_raw.shape != q_target.shape:
        raise ValueError(f"'{position}' intensity and Q arrays have different lengths.")
    if not np.all(np.isfinite(target_raw)):
        raise ValueError(f"'{position}' contains NaN or infinite intensity values.")
    target_max = float(np.max(target_raw))
    if not np.isfinite(target_max) or target_max <= 0:
        raise ValueError(f"'{position}' has a non-positive maximum intensity.")

    mode = params["method"]
    lam = params["lam"]
    if mode == "Fitting Only":
        return context["fitter"].fit(
            target_raw, params["anchors"], params["peaks"]
        )

    if mode == "arPLS Only":
        arpls_result = run_arpls(q_target, target_raw, lam=lam)
        return {
            "target_norm": target_raw / target_max,
            "best_bg_fit": arpls_result["bkg_arpls"] / target_max,
            "clean_target_raw": arpls_result["cleaned_y"],
            "best_ref_name": "arPLS Only",
            "best_rigid_params": [0, 0, 0, 0],
            "best_flex_params": [0, 0, 0, 0, 0, 0],
        }

    if mode == "Fitting + arPLS":
        fit_result = context["fitter"].fit(
            target_raw, params["anchors"], params["peaks"]
        )
        arpls_result = run_arpls(
            q_target, fit_result["clean_target_raw"], lam=lam
        )
        total_background = (
            fit_result["best_bg_fit"] * target_max
        ) + arpls_result["bkg_arpls"]
        fit_result["clean_target_raw"] = arpls_result["cleaned_y"]
        fit_result["best_bg_fit"] = total_background / target_max
        fit_result["best_ref_name"] = f"{fit_result['best_ref_name']} + arPLS"
        return fit_result

    raise ValueError(f"Unknown processing mode: {mode!r}")


CompletionCallback = Callable[
    [int, int, Any, str, Optional[dict[str, Any]], Optional[BaseException]], None
]


def compute_fit_jobs(
    jobs: Iterable[tuple[Any, str]],
    params: Mapping[str, Any],
    context: Mapping[str, Any],
    on_complete: Optional[CompletionCallback] = None,
    max_workers: int = MAX_FIT_WORKERS,
) -> tuple[dict[Any, dict[str, Any]], dict[Any, tuple[str, BaseException]]]:
    """Compute ``(cache_key, position)`` jobs with a bounded worker pool."""

    queued_jobs = list(jobs)
    if not queued_jobs:
        return {}, {}
    worker_count = min(max(1, int(max_workers)), len(queued_jobs))
    results: dict[Any, dict[str, Any]] = {}
    errors: dict[Any, tuple[str, BaseException]] = {}
    with ThreadPoolExecutor(
        max_workers=worker_count, thread_name_prefix="xrd-fit"
    ) as executor:
        futures = {
            executor.submit(compute_position, position, params, context): (
                key,
                position,
            )
            for key, position in queued_jobs
        }
        for done, future in enumerate(as_completed(futures), 1):
            key, position = futures[future]
            try:
                results[key] = future.result()
                error = None
            except Exception as exc:  # Keep per-profile failures isolated.
                errors[key] = position, exc
                error = exc
            if on_complete is not None:
                on_complete(
                    done,
                    len(queued_jobs),
                    key,
                    position,
                    results.get(key),
                    error,
                )
    return results, errors


def make_param_record(position: str, result: Mapping[str, Any]) -> dict[str, Any]:
    """Build the stable fitting-parameter schema used by Excel exports."""

    th_value, samz_value = split_position(position)
    rigid = result["best_rigid_params"]
    flexible = result["best_flex_params"]
    return {
        "Sheet(th)": th_value,
        "Position(samz)": samz_value,
        "Full_Position": position,
        "Best_BG_Used": result["best_ref_name"],
        "Rigid_Scale": rigid[0],
        "Rigid_Q_Scale": rigid[1],
        "Rigid_dq": rigid[2],
        "Rigid_c0": rigid[3],
        "Flex_Scale": flexible[0],
        "Flex_Q_Scale": flexible[1],
        "Flex_dq": flexible[2],
        "Flex_c0": flexible[3],
        "Flex_c1": flexible[4],
        "Flex_c2": flexible[5],
    }


__all__ = [
    "BACKGROUND_Q_GUARD",
    "MAX_FIT_WORKERS",
    "compute_fit_jobs",
    "compute_position",
    "expanded_background_q_range",
    "make_param_record",
    "numeric_sort_key",
    "parse_anchor_lines",
    "parse_peak_lines",
    "q_to_2theta",
    "safe_filename_component",
    "split_position",
    "validate_background_q_support",
    "validate_q_range",
]
