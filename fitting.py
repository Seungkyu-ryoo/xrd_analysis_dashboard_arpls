import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d


def run_optimization(
    target_pos, anchor_list, peak_list, df_target, df_bg, q_target, q_bg, bg_cols
):
    """
    Reference background fitting with
    1. rigid search over background references
    2. flexible refinement including polynomial baseline terms
    """

    target_raw = df_target[target_pos].values.astype(float)
    target_max = np.max(target_raw)

    if target_max <= 0:
        raise ValueError(f"Target signal at {target_pos} has non-positive maximum.")

    target_norm = target_raw / target_max

    current_weights = np.ones(len(q_target), dtype=float)
    for start, end, w in anchor_list:
        mask = (q_target >= start) & (q_target <= end)
        current_weights[mask] = w

    is_bg = np.ones(len(q_target), dtype=bool)
    for start, end in peak_list:
        is_bg &= ~((q_target >= start) & (q_target <= end))

    if not np.any(is_bg):
        raise ValueError("All points are masked as peaks. Need some background region.")

    best_rigid_error = np.inf
    best_ref_name = None
    best_rigid_params = None

    for ref_name in bg_cols:
        ref_raw = df_bg[ref_name].values.astype(float)
        ref_max = np.max(ref_raw)

        if ref_max <= 0:
            continue

        ref_norm_full = ref_raw / ref_max
        f_ref = interp1d(
            q_bg,
            ref_norm_full,
            kind="cubic",
            fill_value="extrapolate",
            bounds_error=False,
        )

        def objective_rigid(params):
            scale, q_scale, dq, c0 = params
            q_adjusted = q_scale * q_target - dq
            y_fit = scale * f_ref(q_adjusted) + c0
            diff = target_norm[is_bg] - y_fit[is_bg]
            weighted_errors = current_weights[is_bg] * (diff**2)
            return np.mean(weighted_errors)

        ref_eval = f_ref(q_target)[is_bg]
        try:
            init_scale, init_c0 = np.polyfit(ref_eval, target_norm[is_bg], 1)
        except Exception:
            init_scale, init_c0 = 1.0, 0.0

        p0_rigid = [init_scale, 1.0, 0.0, init_c0]
        bounds_rigid = [(0.5, 2.0), (0.95, 1.05), (-0.05, 0.05), (-0.3, 0.3)]

        res_rigid = minimize(
            objective_rigid,
            p0_rigid,
            bounds=bounds_rigid,
            method="L-BFGS-B",
        )

        if res_rigid.fun < best_rigid_error:
            best_rigid_error = res_rigid.fun
            best_ref_name = ref_name
            best_rigid_params = res_rigid.x

    if best_ref_name is None or best_rigid_params is None:
        raise ValueError("Failed to find a valid reference background.")

    ref_raw = df_bg[best_ref_name].values.astype(float)
    ref_norm_full = ref_raw / np.max(ref_raw)
    f_ref_best = interp1d(
        q_bg,
        ref_norm_full,
        kind="cubic",
        fill_value="extrapolate",
        bounds_error=False,
    )

    def model_flex(params):
        scale, q_scale, dq, c0, c1, c2 = params
        q_adjusted = q_scale * q_target - dq
        return scale * f_ref_best(q_adjusted) + c0 + c1 * q_target + c2 * (q_target**2)

    def objective_flex(params):
        y_fit = model_flex(params)
        diff = target_norm[is_bg] - y_fit[is_bg]

        # Penalize over-subtraction more strongly
        asym_penalty = np.where(diff < 0, 10.0, 1.0)
        weighted_errors = current_weights[is_bg] * asym_penalty * (diff**2)
        return np.mean(weighted_errors)

    p0_flex = [
        best_rigid_params[0],
        best_rigid_params[1],
        best_rigid_params[2],
        best_rigid_params[3],
        0.0,
        0.0,
    ]

    bounds_flex = [
        (0.5, 2.0),
        (0.95, 1.05),
        (-0.05, 0.05),
        (-0.2, 0.2),
        (-0.1, 0.1),
        (-0.05, 0.05),
    ]

    res_flex = minimize(
        objective_flex,
        p0_flex,
        bounds=bounds_flex,
        method="L-BFGS-B",
    )

    best_bg_fit = model_flex(res_flex.x)

    clean_target_norm = target_norm - best_bg_fit
    clean_target_raw = clean_target_norm * target_max

    return {
        "target_norm": target_norm,
        "best_bg_fit": best_bg_fit,
        "clean_target_raw": clean_target_raw,
        "best_ref_name": best_ref_name,
        "best_rigid_params": best_rigid_params,
        "best_flex_params": res_flex.x,
    }
