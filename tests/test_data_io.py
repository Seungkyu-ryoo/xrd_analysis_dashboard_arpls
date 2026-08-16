from __future__ import annotations

import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from xrd_dashboard.data import read_multisheet  # noqa: E402


class ReadMultisheetTests(unittest.TestCase):
    def test_temporary_workbook_uses_stable_th_and_samz_column_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workbook_path = Path(temp_dir) / "input.xlsx"
            with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
                pd.DataFrame(
                    {
                        " Q ": [1.0, 2.0, 3.0],
                        2: [20.0, 21.0, 22.0],
                        10: [100.0, 101.0, 102.0],
                        "Unnamed: 9": [0.0, 0.0, 0.0],
                    }
                ).to_excel(writer, sheet_name="10", index=False)
                pd.DataFrame(
                    {
                        "Q": [1.5, 2.5, 3.5],
                        1: [5.0, 6.0, 7.0],
                    }
                ).to_excel(writer, sheet_name="2", index=False)
                pd.DataFrame(
                    {"key": ["source"], "value": ["synthetic"]}
                ).to_excel(writer, sheet_name="Metadata", index=False)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                merged, q_values, th_values, samz_values = read_multisheet(
                    workbook_path, 1.0, 3.5
                )

        self.assertEqual(
            list(merged.columns),
            [
                "Q",
                "th:10_samz:2",
                "th:10_samz:10",
                "th:2_samz:1",
            ],
        )
        self.assertEqual(th_values, ["2", "10"])
        self.assertEqual(samz_values, ["1", "2", "10"])
        np.testing.assert_allclose(q_values, [1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        np.testing.assert_allclose(merged["Q"].to_numpy(), q_values)
        self.assertTrue(
            np.isfinite(merged.drop(columns="Q").to_numpy(dtype=float)).all()
        )
        self.assertTrue(
            any("Metadata" in str(item.message) for item in caught),
            "The non-data metadata sheet should be explicitly reported as skipped.",
        )


if __name__ == "__main__":
    unittest.main()
