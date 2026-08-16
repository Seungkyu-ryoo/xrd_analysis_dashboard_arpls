from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from xrd_analysis_dashboard_arpls.analysis import (  # noqa: E402
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
import xrd_analysis_dashboard_arpls.analysis as analysis_module  # noqa: E402


class ParsingTests(unittest.TestCase):
    def test_anchor_and_peak_parsers_accept_blank_lines_and_whitespace(self):
        self.assertEqual(
            parse_anchor_lines(" 1.5, 1.7, 20 \n\n2.8,3.0,5.5 "),
            [(1.5, 1.7, 20.0), (2.8, 3.0, 5.5)],
        )
        self.assertEqual(
            parse_peak_lines(" 1.9, 2.2\n\n3.1,3.3 "),
            [(1.9, 2.2), (3.1, 3.3)],
        )

    def test_anchor_parser_rejects_bad_shape_order_weight_and_nonfinite_values(self):
        invalid_rows = (
            "1,2",
            "2,1,10",
            "1,2,0",
            "1,nan,10",
        )
        for row in invalid_rows:
            with self.subTest(row=row), self.assertRaises(ValueError):
                parse_anchor_lines(row)

    def test_peak_parser_rejects_bad_shape_order_and_nonfinite_values(self):
        invalid_rows = ("1,2,3", "2,1", "1,inf")
        for row in invalid_rows:
            with self.subTest(row=row), self.assertRaises(ValueError):
                parse_peak_lines(row)


class ValidationAndRangeTests(unittest.TestCase):
    def test_q_range_and_expanded_background_support(self):
        validate_q_range(1.6, 3.7)
        expanded_min, expanded_max = expanded_background_q_range(1.6, 3.7)
        self.assertAlmostEqual(expanded_min, 1.45)
        self.assertAlmostEqual(expanded_max, 3.955)

        validate_background_q_support(
            [expanded_max, 2.0, expanded_min], expanded_min, expanded_max
        )

    def test_q_range_rejects_reversed_equal_and_nonfinite_bounds(self):
        for q_min, q_max in ((2.0, 2.0), (3.0, 2.0), (np.nan, 2.0), (1.0, np.inf)):
            with self.subTest(q_min=q_min, q_max=q_max), self.assertRaises(ValueError):
                validate_q_range(q_min, q_max)

    def test_background_support_rejects_invalid_arrays_and_short_coverage(self):
        invalid_values = ([], [[1.0, 2.0]], [1.0, np.nan, 3.0])
        for values in invalid_values:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_background_q_support(values, 1.0, 3.0)

        with self.assertRaises(ValueError):
            validate_background_q_support([1.1, 2.0, 3.0], 1.0, 3.0)
        with self.assertRaises(ValueError):
            validate_background_q_support([1.0, 2.0, 2.9], 1.0, 3.0)

    def test_q_to_2theta_and_numeric_sort_key(self):
        wavelength = 1.5406
        converted = q_to_2theta([0.0, 4.0 * np.pi / wavelength], wavelength)
        np.testing.assert_allclose(converted, [0.0, 180.0], atol=1e-12)
        self.assertEqual(
            sorted(["10", "sample", "2", "Alpha"], key=numeric_sort_key),
            ["2", "10", "Alpha", "sample"],
        )


class PositionAndExportSchemaTests(unittest.TestCase):
    def test_position_split_and_filename_sanitizing(self):
        self.assertEqual(split_position("th:0.230_samz:-4"), ("0.230", "-4"))
        self.assertEqual(split_position("unstructured"), ("unstructured", ""))
        self.assertEqual(safe_filename_component("a/b:c*"), "a_b_c_")
        self.assertEqual(safe_filename_component(" . "), "unnamed")

    def test_parameter_record_keeps_the_stable_excel_schema(self):
        result = {
            "best_ref_name": "reference-1 + arPLS",
            "best_rigid_params": [1.0, 1.01, -0.02, 0.03],
            "best_flex_params": [1.1, 0.99, 0.01, -0.02, 0.04, -0.005],
        }
        record = make_param_record("th:0.230_samz:-4", result)

        expected_columns = [
            "Sheet(th)",
            "Position(samz)",
            "Full_Position",
            "Best_BG_Used",
            "Rigid_Scale",
            "Rigid_Q_Scale",
            "Rigid_dq",
            "Rigid_c0",
            "Flex_Scale",
            "Flex_Q_Scale",
            "Flex_dq",
            "Flex_c0",
            "Flex_c1",
            "Flex_c2",
        ]
        self.assertEqual(list(record), expected_columns)
        self.assertEqual(record["Sheet(th)"], "0.230")
        self.assertEqual(record["Position(samz)"], "-4")
        self.assertEqual(record["Best_BG_Used"], "reference-1 + arPLS")
        self.assertEqual(record["Flex_c2"], -0.005)


class _MockFitter:
    def __init__(self):
        self.call_count = 0

    def fit(self, target_raw, anchors, peaks):
        self.call_count += 1
        if target_raw[0] == 99.0:
            raise RuntimeError("intentional isolated failure")
        target_raw = np.asarray(target_raw, dtype=float)
        target_max = float(np.max(target_raw))
        return {
            "target_norm": target_raw / target_max,
            "best_bg_fit": np.zeros_like(target_raw),
            "clean_target_raw": target_raw.copy(),
            "best_ref_name": "mock-reference",
            "best_rigid_params": np.array([1.0, 1.0, 0.0, 0.0]),
            "best_flex_params": np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        }


class ComputeFitJobsTests(unittest.TestCase):
    def test_success_and_failure_are_reported_independently(self):
        context = {
            "q_target": np.array([1.0, 2.0, 3.0]),
            "df_target": pd.DataFrame(
                {
                    "good": [1.0, 2.0, 3.0],
                    "bad": [99.0, 2.0, 3.0],
                }
            ),
            "fitter": _MockFitter(),
        }
        params = {
            "method": "Fitting Only",
            "lam": 0.0,
            "anchors": [(1.0, 1.5, 2.0)],
            "peaks": [(2.0, 2.5)],
        }
        completions = []

        results, errors = compute_fit_jobs(
            [("good-key", "good"), ("bad-key", "bad")],
            params,
            context,
            on_complete=lambda *event: completions.append(event),
            max_workers=2,
        )

        self.assertEqual(set(results), {"good-key"})
        np.testing.assert_allclose(results["good-key"]["clean_target_raw"], [1, 2, 3])
        self.assertEqual(set(errors), {"bad-key"})
        failed_position, failure = errors["bad-key"]
        self.assertEqual(failed_position, "bad")
        self.assertIsInstance(failure, RuntimeError)
        self.assertIn("intentional isolated failure", str(failure))

        self.assertEqual(len(completions), 2)
        self.assertEqual(sorted(event[0] for event in completions), [1, 2])
        self.assertTrue(all(event[1] == 2 for event in completions))
        completion_by_key = {event[2]: event for event in completions}
        self.assertIsNotNone(completion_by_key["good-key"][4])
        self.assertIsNone(completion_by_key["good-key"][5])
        self.assertIsNone(completion_by_key["bad-key"][4])
        self.assertIsInstance(completion_by_key["bad-key"][5], RuntimeError)

    def test_empty_job_list_is_a_noop(self):
        results, errors = compute_fit_jobs([], {}, {}, max_workers=2)
        self.assertEqual(results, {})
        self.assertEqual(errors, {})


class ComputePositionModeTests(unittest.TestCase):
    def setUp(self):
        self.q_values = np.array([1.0, 2.0, 3.0])
        self.fitter = _MockFitter()
        self.context = {
            "q_target": self.q_values,
            "df_target": pd.DataFrame({"scan": [2.0, 4.0, 6.0]}),
            "fitter": self.fitter,
        }

    @staticmethod
    def _fake_arpls(_q_values, intensity, lam):
        baseline = np.full_like(np.asarray(intensity, dtype=float), lam / 100.0)
        return {
            "bkg_arpls": baseline,
            "cleaned_y": np.asarray(intensity, dtype=float) - baseline,
            "params": {},
        }

    def test_fitting_only_does_not_call_arpls(self):
        params = {"method": "Fitting Only", "lam": 0.0, "anchors": [], "peaks": []}
        with patch.object(
            analysis_module, "run_arpls", side_effect=AssertionError("unexpected")
        ):
            result = compute_position("scan", params, self.context)
        self.assertEqual(self.fitter.call_count, 1)
        np.testing.assert_allclose(result["clean_target_raw"], [2.0, 4.0, 6.0])

    def test_arpls_only_skips_reference_fitting(self):
        params = {"method": "arPLS Only", "lam": 10.0, "anchors": [], "peaks": []}
        with patch.object(analysis_module, "run_arpls", side_effect=self._fake_arpls):
            result = compute_position("scan", params, self.context)
        self.assertEqual(self.fitter.call_count, 0)
        self.assertEqual(result["best_ref_name"], "arPLS Only")
        np.testing.assert_allclose(result["clean_target_raw"], [1.9, 3.9, 5.9])

    def test_fitting_plus_arpls_combines_both_backgrounds(self):
        params = {
            "method": "Fitting + arPLS",
            "lam": 10.0,
            "anchors": [],
            "peaks": [],
        }
        with patch.object(analysis_module, "run_arpls", side_effect=self._fake_arpls):
            result = compute_position("scan", params, self.context)
        self.assertEqual(self.fitter.call_count, 1)
        self.assertEqual(result["best_ref_name"], "mock-reference + arPLS")
        np.testing.assert_allclose(result["clean_target_raw"], [1.9, 3.9, 5.9])
        np.testing.assert_allclose(result["best_bg_fit"], np.full(3, 0.1 / 6.0))

    def test_unknown_mode_is_rejected(self):
        params = {"method": "unknown", "lam": 1.0, "anchors": [], "peaks": []}
        with self.assertRaisesRegex(ValueError, "Unknown processing mode"):
            compute_position("scan", params, self.context)


if __name__ == "__main__":
    unittest.main()
