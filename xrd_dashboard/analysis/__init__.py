"""Numerical fitting, baseline correction, and job orchestration."""

from .baseline import run_arpls
from .fitting import FLEX_BOUNDS, RIGID_BOUNDS, BackgroundFitter
from .pipeline import (
    BACKGROUND_Q_GUARD,
    MAX_FIT_WORKERS,
    compute_fit_jobs,
    compute_position,
    expanded_background_q_range,
    make_param_record,
    numeric_sort_key,
    parse_anchor_lines,
    parse_peak_lines,
    q_to_2theta,
    safe_filename_component,
    split_position,
    validate_background_q_support,
    validate_q_range,
)

__all__ = [
    "BACKGROUND_Q_GUARD",
    "FLEX_BOUNDS",
    "MAX_FIT_WORKERS",
    "RIGID_BOUNDS",
    "BackgroundFitter",
    "compute_fit_jobs",
    "compute_position",
    "expanded_background_q_range",
    "make_param_record",
    "numeric_sort_key",
    "parse_anchor_lines",
    "parse_peak_lines",
    "q_to_2theta",
    "run_arpls",
    "safe_filename_component",
    "split_position",
    "validate_background_q_support",
    "validate_q_range",
]
