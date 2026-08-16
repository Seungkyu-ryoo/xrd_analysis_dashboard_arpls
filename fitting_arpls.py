"""Compatibility wrapper for legacy ``from fitting_arpls import run_arpls``."""

if __package__:
    from .xrd_dashboard.analysis import run_arpls
else:
    from xrd_dashboard.analysis import run_arpls

__all__ = ["run_arpls"]
