from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from xrd_analysis_dashboard_arpls.exporting import write_export  # noqa: E402


def fit_result(values):
    return {
        "clean_target_raw": np.asarray(values, dtype=float),
        "best_ref_name": "reference + arPLS",
        "best_rigid_params": [1.0, 1.0, 0.0, 0.0],
        "best_flex_params": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    }


class ExportWriterTests(unittest.TestCase):
    def test_xye_export_resolves_sanitized_filename_collisions(self):
        positions = ("th:a/b_samz:c", r"th:a\b_samz:c")
        results = {position: fit_result([10, 20, 30]) for position in positions}
        with tempfile.TemporaryDirectory() as directory:
            summary = write_export(
                directory,
                "xye",
                "merged",  # XYE is intentionally normalized to individual mode.
                "Q",
                [1.0, 2.0, 3.0],
                positions,
                results,
                "sample",
            )

            self.assertEqual(summary.succeeded, 2)
            self.assertEqual(summary.failed, 0)
            self.assertEqual(summary.save_mode, "individual")
            names = sorted(path.name for path in summary.output_paths)
            self.assertEqual(
                names,
                ["sample_th_a_b_samz_c.xye", "sample_th_a_b_samz_c__2.xye"],
            )
            values = np.loadtxt(summary.output_paths[0])
            np.testing.assert_allclose(values[:, 0], [1.0, 2.0, 3.0])
            np.testing.assert_allclose(values[:, 1], [10.0, 20.0, 30.0])
            self.assertFalse(any(".tmp-" in path.name for path in Path(directory).iterdir()))

    def test_individual_excel_preserves_sheet_schema(self):
        position = "th:0.2_samz:3"
        with tempfile.TemporaryDirectory() as directory:
            summary = write_export(
                directory,
                "excel",
                "individual",
                "Q",
                [1.0, 2.0, 3.0],
                [position],
                {position: fit_result([4, 5, 6])},
                "sample",
            )

            with pd.ExcelFile(summary.output_paths[0]) as workbook:
                self.assertEqual(
                    workbook.sheet_names,
                    ["Cleaned_Signal", "Fitting_Parameters"],
                )
            signal = pd.read_excel(summary.output_paths[0], sheet_name="Cleaned_Signal")
            self.assertEqual(list(signal), ["Q", position])

    def test_merged_excel_is_atomic_and_isolates_missing_results(self):
        good = "th:0.2_samz:3"
        missing = "th:0.2_samz:4"
        errors = []
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "merged.xlsx"
            summary = write_export(
                destination,
                "excel",
                "merged",
                "2Theta",
                [20.0, 21.0, 22.0],
                [good, missing],
                {good: fit_result([4, 5, 6])},
                "sample",
                on_error=lambda position, error: errors.append((position, error)),
            )

            self.assertEqual(summary.succeeded, 1)
            self.assertEqual(summary.failed, 1)
            self.assertEqual(errors[0][0], missing)
            with pd.ExcelFile(destination) as workbook:
                self.assertEqual(
                    workbook.sheet_names,
                    ["Cleaned_Signals", "Fitting_Parameters"],
                )
            signal = pd.read_excel(destination, sheet_name="Cleaned_Signals")
            self.assertEqual(list(signal), ["2Theta", good])
            self.assertFalse(any(".tmp-" in path.name for path in Path(directory).iterdir()))


if __name__ == "__main__":
    unittest.main()
