"""Validated arPLS baseline subtraction."""

import numpy as np
from pybaselines import Baseline


def run_arpls(x_data, y_data, lam=1e5):
    """Return the arPLS baseline, cleaned signal, and solver metadata.

    ``lam`` controls baseline smoothness; values around ``1e4`` to ``1e6``
    are typical for the dashboard's XRD profiles.
    """
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    if x_data.ndim != 1 or y_data.ndim != 1:
        raise ValueError("arPLS input arrays must be one-dimensional.")
    if x_data.shape != y_data.shape:
        raise ValueError("arPLS Q and intensity arrays have different lengths.")
    if len(x_data) < 3:
        raise ValueError("arPLS requires at least three data points.")
    if not np.all(np.isfinite(x_data)) or not np.all(np.isfinite(y_data)):
        raise ValueError("arPLS input contains NaN or infinite values.")
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("arPLS lambda must be a finite positive number.")
    if np.any(np.diff(x_data) <= 0):
        raise ValueError("arPLS Q values must be strictly increasing.")

    baseline_fitter = Baseline(x_data)

    bkg_arpls, params = baseline_fitter.arpls(y_data, lam=lam)
    y_cleaned = y_data - bkg_arpls

    return {"bkg_arpls": bkg_arpls, "cleaned_y": y_cleaned, "params": params}
