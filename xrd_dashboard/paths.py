"""Stable filesystem locations shared by the dashboard modules."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
BUNDLED_REFERENCE_PATH = RESOURCE_DIR / "ReferencePeaks.xlsx"

__all__ = ["BUNDLED_REFERENCE_PATH", "PROJECT_ROOT", "RESOURCE_DIR"]
