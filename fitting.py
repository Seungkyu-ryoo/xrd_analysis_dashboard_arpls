"""Compatibility API for the original dashboard fitting entry point."""

if __package__:
    from .xrd_dashboard.analysis import BackgroundFitter
else:
    from xrd_dashboard.analysis import BackgroundFitter


def run_optimization(
    target_pos,
    anchor_list,
    peak_list,
    df_target,
    df_bg,
    q_target,
    q_bg,
    bg_cols,
):
    """Run the validated fitter while preserving the legacy call signature."""
    fitter = BackgroundFitter(df_bg, q_bg, bg_cols, q_target)
    target_raw = df_target[target_pos].to_numpy(dtype=float)
    return fitter.fit(target_raw, anchor_list, peak_list)
