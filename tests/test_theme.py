from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from xrd_dashboard.ui import theme  # noqa: E402


def _relative_luminance(hex_colour: str) -> float:
    """Return WCAG 2 relative luminance for a six-digit hex colour."""

    if len(hex_colour) != 7 or not hex_colour.startswith("#"):
        raise ValueError(f"Expected a six-digit hex colour, got {hex_colour!r}.")
    channels = [int(hex_colour[index : index + 2], 16) / 255.0 for index in (1, 3, 5)]
    linear = [
        channel / 12.92
        if channel <= 0.04045
        else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast_ratio(foreground: str, background: str) -> float:
    lighter, darker = sorted(
        (_relative_luminance(foreground), _relative_luminance(background)),
        reverse=True,
    )
    return (lighter + 0.05) / (darker + 0.05)


class ThemeContrastTests(unittest.TestCase):
    def test_normal_and_hover_button_colours_meet_wcag_aa(self):
        button_pairs = {
            "primary normal": (theme.BUTTON_FG, theme.COLOR_PRIMARY),
            "primary hover": (theme.BUTTON_FG, theme.COLOR_HOVER_P),
            "warning normal": (theme.BUTTON_FG, theme.COLOR_WARN),
            "warning hover": (theme.BUTTON_FG, theme.COLOR_HOVER_W),
            "success normal": (theme.BUTTON_FG, theme.COLOR_SUCCESS),
            "success hover": (theme.BUTTON_FG, theme.COLOR_HOVER_S),
            "secondary normal": (theme.FG_TEXT, theme.COLOR_SECONDARY),
            "secondary hover": (
                theme.FG_TEXT,
                theme.COLOR_HOVER_SECONDARY,
            ),
        }

        for state, (foreground, background) in button_pairs.items():
            with self.subTest(state=state):
                ratio = _contrast_ratio(foreground, background)
                self.assertGreaterEqual(
                    ratio,
                    4.5,
                    msg=(
                        f"{state} contrast is {ratio:.2f}:1 for "
                        f"{foreground} on {background}; expected at least 4.5:1."
                    ),
                )

    def test_input_disabled_and_status_text_remain_readable(self):
        text_pairs = {
            "input": (theme.INPUT_FG, theme.INPUT_BG),
            "disabled": (theme.DISABLED_FG, theme.DISABLED_BG),
            "ready status": (theme.STATUS_READY, theme.STATUS_READY_BG),
            "busy status": (theme.STATUS_BUSY, theme.STATUS_BUSY_BG),
            "info status": (theme.STATUS_INFO, theme.STATUS_INFO_BG),
            "error status": (theme.STATUS_ERROR, theme.STATUS_ERROR_BG),
        }
        for state, (foreground, background) in text_pairs.items():
            with self.subTest(state=state):
                self.assertGreaterEqual(
                    _contrast_ratio(foreground, background),
                    4.5,
                    msg=f"{state} text does not meet WCAG AA contrast.",
                )


if __name__ == "__main__":
    unittest.main()
