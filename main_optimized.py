"""Backward-compatible dashboard entry point.

New code should import focused helpers from ``xrd_dashboard`` or run
``main.py``. These re-exports preserve existing beamtime scripts that import
public names from the historical ``main_optimized`` module.
"""

if __package__:
    from .xrd_dashboard.analysis import (
        BackgroundFitter,
        MAX_FIT_WORKERS,
        compute_fit_jobs,
        compute_position,
        expanded_background_q_range,
        make_param_record,
        parse_anchor_lines,
        parse_peak_lines,
        q_to_2theta,
        safe_filename_component,
        split_position,
        validate_background_q_support,
        validate_q_range,
    )
    from .xrd_dashboard.data import load_reference_peaks, read_multisheet
    from .xrd_dashboard.ui.app import (
        XRDDashboard,
        enable_high_dpi,
        ensure_supported_tk,
        main,
    )
else:
    from xrd_dashboard.analysis import (
        BackgroundFitter,
        MAX_FIT_WORKERS,
        compute_fit_jobs,
        compute_position,
        expanded_background_q_range,
        make_param_record,
        parse_anchor_lines,
        parse_peak_lines,
        q_to_2theta,
        safe_filename_component,
        split_position,
        validate_background_q_support,
        validate_q_range,
    )
    from xrd_dashboard.data import load_reference_peaks, read_multisheet
    from xrd_dashboard.ui.app import (
        XRDDashboard,
        enable_high_dpi,
        ensure_supported_tk,
        main,
    )


__all__ = [
    "BackgroundFitter",
    "MAX_FIT_WORKERS",
    "XRDDashboard",
    "compute_fit_jobs",
    "compute_position",
    "enable_high_dpi",
    "ensure_supported_tk",
    "expanded_background_q_range",
    "load_reference_peaks",
    "main",
    "make_param_record",
    "parse_anchor_lines",
    "parse_peak_lines",
    "q_to_2theta",
    "read_multisheet",
    "safe_filename_component",
    "split_position",
    "validate_background_q_support",
    "validate_q_range",
]


if __name__ == "__main__":
    main()
