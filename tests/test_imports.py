from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_DIR.parent


class ImportSmokeTests(unittest.TestCase):
    REQUIRED_MAIN_NAMES = (
        "BackgroundFitter",
        "XRDDashboard",
        "compute_fit_jobs",
        "expanded_background_q_range",
        "load_reference_peaks",
        "main",
        "make_param_record",
        "parse_anchor_lines",
        "parse_peak_lines",
        "q_to_2theta",
        "read_multisheet",
        "safe_filename_component",
        "split_position",
        "validate_q_range",
    )

    def _assert_importable(self, module_name: str, working_directory: Path) -> None:
        names = repr(self.REQUIRED_MAIN_NAMES)
        code = (
            f"import {module_name} as dashboard; "
            f"required = {names}; "
            "missing = [name for name in required if not hasattr(dashboard, name)]; "
            "assert not missing, missing"
        )
        with tempfile.TemporaryDirectory() as mpl_config:
            environment = os.environ.copy()
            environment["MPLCONFIGDIR"] = mpl_config
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            completed = subprocess.run(
                [sys.executable, "-B", "-c", code],
                cwd=working_directory,
                env=environment,
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )

    def test_package_import_exposes_dashboard_public_names(self):
        self._assert_importable(
            "xrd_analysis_dashboard_arpls.main_optimized", PROJECT_PARENT
        )

    def test_direct_import_exposes_dashboard_public_names(self):
        self._assert_importable("main_optimized", PROJECT_DIR)

    def test_package_exports_background_fitter_and_arpls(self):
        if str(PROJECT_PARENT) not in sys.path:
            sys.path.insert(0, str(PROJECT_PARENT))
        import xrd_analysis_dashboard_arpls as dashboard

        self.assertEqual(
            set(dashboard.__all__), {"BackgroundFitter", "run_arpls"}
        )
        self.assertTrue(callable(dashboard.BackgroundFitter))
        self.assertTrue(callable(dashboard.run_arpls))


if __name__ == "__main__":
    unittest.main()
