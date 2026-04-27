"""arPLS baseline subtraction helper."""

import numpy as np
from pybaselines import Baseline


def run_arpls(x_data, y_data, lam=1e5):
    """
    Apply pybaselines' arPLS algorithm for baseline subtraction.

    Parameters:
        x_data (np.array): Q values.
        y_data (np.array): Raw intensity or residual signal.
        lam (float): arPLS smoothness parameter. Typical range: 1e4 to 1e6.

    Returns:
        dict: Baseline, cleaned signal, and pybaselines parameter metadata.
    """
    baseline_fitter = Baseline(x_data)
    bkg_arpls, params = baseline_fitter.arpls(y_data, lam=lam)
    y_cleaned = y_data - bkg_arpls

    return {"bkg_arpls": bkg_arpls, "cleaned_y": y_cleaned, "params": params}
