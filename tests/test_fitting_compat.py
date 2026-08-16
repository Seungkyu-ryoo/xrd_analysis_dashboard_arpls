from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from xrd_analysis_dashboard_arpls.fitting import run_optimization  # noqa: E402
from xrd_analysis_dashboard_arpls.fitting_optimized import (  # noqa: E402
    BackgroundFitter,
)


class FittingCompatibilityTests(unittest.TestCase):
    def test_legacy_wrapper_matches_background_fitter(self):
        q_target = np.linspace(1.5, 3.5, 61)
        q_background = np.linspace(1.35, 3.75, 101)
        reference_a = (
            1.4 + 0.18 * np.sin(2.7 * q_background) + 0.04 * q_background**2
        )
        reference_b = (
            1.2 + 0.12 * np.cos(3.1 * q_background) + 0.08 * q_background
        )
        interpolated = np.interp(
            1.01 * q_target - 0.008, q_background, reference_a
        )
        peak = 0.4 * np.exp(-0.5 * ((q_target - 2.25) / 0.06) ** 2)
        target = 800.0 * (1.1 * interpolated + 0.015 + peak)

        position = "th:0.2_samz:1"
        target_frame = pd.DataFrame({"Q": q_target, position: target})
        background_frame = pd.DataFrame(
            {
                "Q": q_background,
                "reference_a": reference_a,
                "reference_b": reference_b,
            }
        )
        background_columns = ["reference_a", "reference_b"]
        anchors = [(1.5, 1.7, 20.0), (3.2, 3.45, 20.0)]
        peak_masks = [(2.1, 2.4)]

        compatibility_result = run_optimization(
            position,
            anchors,
            peak_masks,
            target_frame,
            background_frame,
            q_target,
            q_background,
            background_columns,
        )
        direct_result = BackgroundFitter(
            background_frame, q_background, background_columns, q_target
        ).fit(target, anchors, peak_masks)

        self.assertEqual(
            compatibility_result["best_ref_name"], direct_result["best_ref_name"]
        )
        for key in (
            "target_norm",
            "best_bg_fit",
            "clean_target_raw",
            "best_rigid_params",
            "best_flex_params",
        ):
            with self.subTest(key=key):
                np.testing.assert_allclose(
                    compatibility_result[key], direct_result[key], rtol=1e-12, atol=1e-12
                )


if __name__ == "__main__":
    unittest.main()
