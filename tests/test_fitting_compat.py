from __future__ import annotations

import inspect
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pybaselines.utils import ParameterWarning


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from fitting import run_optimization  # noqa: E402
from fitting_arpls import run_arpls  # noqa: E402
from xrd_dashboard.analysis import BackgroundFitter  # noqa: E402


class FittingCompatibilityTests(unittest.TestCase):
    def test_legacy_function_signatures_and_arpls_result_schema(self):
        self.assertEqual(
            tuple(inspect.signature(run_optimization).parameters),
            (
                "target_pos",
                "anchor_list",
                "peak_list",
                "df_target",
                "df_bg",
                "q_target",
                "q_bg",
                "bg_cols",
            ),
        )
        self.assertEqual(
            tuple(inspect.signature(run_arpls).parameters),
            ("x_data", "y_data", "lam"),
        )

        q_values = np.linspace(1.5, 3.5, 101)
        signal = 10.0 + q_values + np.exp(-((q_values - 2.4) / 0.08) ** 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ParameterWarning)
            result = run_arpls(q_values, signal, lam=1e4)
        self.assertEqual(set(result), {"bkg_arpls", "cleaned_y", "params"})
        np.testing.assert_allclose(
            result["bkg_arpls"] + result["cleaned_y"], signal
        )

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
