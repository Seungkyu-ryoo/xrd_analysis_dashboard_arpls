from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from xrd_dashboard.data import load_reference_peaks  # noqa: E402
from xrd_dashboard.paths import BUNDLED_REFERENCE_PATH  # noqa: E402


class ProjectStructureTests(unittest.TestCase):
    def test_implementation_modules_are_grouped_by_role(self):
        old_flat_modules = (
            "analysis.py",
            "app.py",
            "data_io.py",
            "exporting.py",
            "fitting_optimized.py",
            "plotting.py",
            "theme.py",
            "preprocess_xdart_for_dashboard.py",
            "despike_background_workbook.py",
            "xrd_analysis_dashboard_arpls.py",
        )
        for filename in old_flat_modules:
            with self.subTest(filename=filename):
                self.assertFalse((PROJECT_ROOT / filename).exists())

        self.assertTrue((PROJECT_ROOT / "main.py").is_file())
        for package in ("analysis", "data", "ui", "tools"):
            with self.subTest(package=package):
                self.assertTrue(
                    (PROJECT_ROOT / "xrd_dashboard" / package / "__init__.py").is_file()
                )

    def test_bundled_reference_workbook_is_valid(self):
        self.assertTrue(BUNDLED_REFERENCE_PATH.is_file())
        reference_data, warning = load_reference_peaks(
            BUNDLED_REFERENCE_PATH, 1.6, 3.7
        )
        self.assertIsNone(warning)
        self.assertTrue(reference_data)

    def test_main_and_tool_entry_points_import_or_show_help(self):
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        commands = (
            [sys.executable, "-B", "-c", "import main; assert callable(main.main)"],
            [
                sys.executable,
                "-B",
                "-c",
                "import xrd_dashboard.__main__ as entry; assert callable(entry.main)",
            ],
            [
                sys.executable,
                "-B",
                "-m",
                "xrd_dashboard.tools.preprocess_xdart",
                "--help",
            ],
            [
                sys.executable,
                "-B",
                "-m",
                "xrd_dashboard.tools.despike_background",
                "--help",
            ],
        )
        for command in commands:
            with self.subTest(command=" ".join(command)):
                completed = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
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


if __name__ == "__main__":
    unittest.main()
