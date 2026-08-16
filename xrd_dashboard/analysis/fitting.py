"""Reusable two-stage reference-background fitter.

Cubic interpolators for the reference pool are built once when a dataset is
loaded, then reused for every selected target profile.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d

RIGID_BOUNDS = [(0.5, 2.0), (0.95, 1.05), (-0.05, 0.05), (-0.3, 0.3)]
FLEX_BOUNDS = [
    (0.5, 2.0),
    (0.95, 1.05),
    (-0.05, 0.05),
    (-0.2, 0.2),
    (-0.1, 0.1),
    (-0.05, 0.05),
]


class BackgroundFitter:
    """Reference background fitting:
    1. rigid search over background references
    2. flexible refinement including polynomial baseline terms
    """

    def __init__(self, df_bg, q_bg, bg_cols, q_target):
        self.q_target = np.asarray(q_target, dtype=float)
        q_bg = np.asarray(q_bg, dtype=float)
        if self.q_target.ndim != 1 or q_bg.ndim != 1:
            raise ValueError("Q arrays must be one-dimensional.")
        if not np.all(np.isfinite(self.q_target)):
            raise ValueError("Target Q array contains NaN or infinite values.")
        if len(self.q_target) < 4:
            raise ValueError("At least four target Q points are required for fitting.")

        self.refs = []  # (name, interpolator, eval at unshifted q_target)
        skipped = []
        for name in bg_cols:
            ref_raw = df_bg[name].values.astype(float)
            if len(ref_raw) != len(q_bg):
                skipped.append(f"{name}: Q/intensity length mismatch")
                continue
            valid = np.isfinite(q_bg) & np.isfinite(ref_raw)
            if np.count_nonzero(valid) < 4:
                skipped.append(f"{name}: fewer than four finite data points")
                continue

            q_valid = q_bg[valid]
            ref_valid = ref_raw[valid]
            order = np.argsort(q_valid)
            q_valid, ref_valid = q_valid[order], ref_valid[order]
            q_valid, unique_indices = np.unique(q_valid, return_index=True)
            ref_valid = ref_valid[unique_indices]
            if len(q_valid) < 4:
                skipped.append(f"{name}: fewer than four unique Q points")
                continue

            ref_max = np.max(ref_valid)
            if not np.isfinite(ref_max) or ref_max <= 0:
                skipped.append(f"{name}: non-positive or invalid maximum")
                continue
            f_ref = interp1d(
                q_valid,
                ref_valid / ref_max,
                kind="cubic",
                fill_value="extrapolate",
                bounds_error=False,
            )
            self.refs.append((name, f_ref, f_ref(self.q_target)))
        if not self.refs:
            raise ValueError(
                "No valid reference background column is available. "
                + "; ".join(skipped)
            )

    def fit(self, target_raw, anchor_list, peak_list):
        q_target = self.q_target
        target_raw = np.asarray(target_raw, dtype=float)
        if target_raw.shape != q_target.shape:
            raise ValueError("Target intensity and Q arrays have different lengths.")
        if not np.all(np.isfinite(target_raw)):
            raise ValueError("Target signal contains NaN or infinite values.")
        target_max = np.max(target_raw)
        if not np.isfinite(target_max) or target_max <= 0:
            raise ValueError("Target signal has a non-positive maximum.")
        target_norm = target_raw / target_max

        weights = np.ones(len(q_target), dtype=float)
        for start, end, w in anchor_list:
            weights[(q_target >= start) & (q_target <= end)] = w

        is_bg = np.ones(len(q_target), dtype=bool)
        for start, end in peak_list:
            is_bg &= ~((q_target >= start) & (q_target <= end))
        if not np.any(is_bg):
            raise ValueError(
                "All points are masked as peaks. Need some background region."
            )
        if np.count_nonzero(is_bg) < 2:
            raise ValueError("At least two unmasked background points are required.")

        q_bg_pts = q_target[is_bg]
        t_bg = target_norm[is_bg]
        w_bg = weights[is_bg]

        # --- Stage 1: rigid search over all references ---
        best_rigid_error = np.inf
        best_ref_name = None
        best_rigid_params = None
        best_f_ref = None
        optimizer_errors = []

        for ref_name, f_ref, base_eval in self.refs:

            def objective_rigid(params, f_ref=f_ref):
                scale, q_scale, dq, c0 = params
                y_fit = scale * f_ref(q_scale * q_bg_pts - dq) + c0
                diff = t_bg - y_fit
                return np.mean(w_bg * diff**2)

            try:
                init_scale, init_c0 = np.polyfit(base_eval[is_bg], t_bg, 1)
            except Exception:
                init_scale, init_c0 = 1.0, 0.0

            res_rigid = minimize(
                objective_rigid,
                [init_scale, 1.0, 0.0, init_c0],
                bounds=RIGID_BOUNDS,
                method="L-BFGS-B",
            )

            if not res_rigid.success or not np.isfinite(res_rigid.fun):
                optimizer_errors.append(f"{ref_name}: {res_rigid.message}")
                continue

            if res_rigid.fun < best_rigid_error:
                best_rigid_error = res_rigid.fun
                best_ref_name = ref_name
                best_rigid_params = res_rigid.x
                best_f_ref = f_ref

        if best_ref_name is None or best_rigid_params is None:
            details = "; ".join(optimizer_errors) or "no finite optimizer result"
            raise RuntimeError(f"Rigid background fitting failed: {details}")

        # --- Stage 2: flexible refinement on the winning reference ---
        def model_flex(params):
            scale, q_scale, dq, c0, c1, c2 = params
            q_adjusted = q_scale * q_target - dq
            return (
                scale * best_f_ref(q_adjusted)
                + c0
                + c1 * q_target
                + c2 * (q_target**2)
            )

        def objective_flex(params):
            scale, q_scale, dq, c0, c1, c2 = params
            y_fit = (
                scale * best_f_ref(q_scale * q_bg_pts - dq)
                + c0
                + c1 * q_bg_pts
                + c2 * (q_bg_pts**2)
            )
            diff = t_bg - y_fit
            # Penalize over-subtraction more strongly
            asym_penalty = np.where(diff < 0, 10.0, 1.0)
            return np.mean(w_bg * asym_penalty * diff**2)

        p0_flex = list(best_rigid_params) + [0.0, 0.0]
        res_flex = minimize(
            objective_flex, p0_flex, bounds=FLEX_BOUNDS, method="L-BFGS-B"
        )
        if not res_flex.success or not np.isfinite(res_flex.fun):
            raise RuntimeError(f"Flexible background fitting failed: {res_flex.message}")

        best_bg_fit = model_flex(res_flex.x)
        if not np.all(np.isfinite(best_bg_fit)):
            raise RuntimeError("Flexible background fitting produced invalid values.")
        clean_target_raw = (target_norm - best_bg_fit) * target_max

        return {
            "target_norm": target_norm,
            "best_bg_fit": best_bg_fit,
            "clean_target_raw": clean_target_raw,
            "best_ref_name": best_ref_name,
            "best_rigid_params": best_rigid_params,
            "best_flex_params": res_flex.x,
        }
