from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

import matplotlib


matplotlib.use("Agg", force=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from xrd_analysis_dashboard_arpls.data_io import REFERENCE_PHASES  # noqa: E402
from xrd_analysis_dashboard_arpls.plotting import (  # noqa: E402
    PlotOptions,
    render_dashboard_figure,
)


def _synthetic_plot_items(q_values: np.ndarray, count: int):
    items = []
    for index in range(count):
        centre = 2.0 + 0.02 * index
        peak = np.exp(-((q_values - centre) / 0.08) ** 2)
        background = 0.12 + 0.015 * (q_values - q_values.min())
        cleaned = (80.0 + index) * peak + 0.5 * np.sin(q_values * 7.0)
        result = {
            "target_norm": background + peak,
            "best_bg_fit": background,
            "clean_target_raw": cleaned,
        }
        items.append((f"th:0.260_samz:{index}", str(index), result))
    return items


def _synthetic_reference_data():
    return {
        REFERENCE_PHASES[0]: pd.DataFrame(
            {
                "Q": [1.92, 2.15, 2.31],
                "Intensity_Norm": [0.55, 1.0, 0.7],
                "hkl": ["$(111)$", "$(200)$", "$(220)$"],
            }
        )
    }


class DashboardPlottingTests(unittest.TestCase):
    def _render_without_warnings(self, count: int, options: PlotOptions):
        q_values = np.linspace(1.5, 3.5, 240)
        figure = Figure(figsize=(11, 7), dpi=100)
        canvas = FigureCanvasAgg(figure)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            axes = render_dashboard_figure(
                figure,
                q_values,
                _synthetic_plot_items(q_values, count),
                _synthetic_reference_data(),
                options,
            )
            canvas.draw()
        self.assertEqual(
            caught,
            [],
            msg="Unexpected plotting warnings: "
            + " | ".join(str(item.message) for item in caught),
        )
        self.addCleanup(figure.clear)
        return figure, axes

    def test_two_results_without_reference_use_two_labeled_axes_and_legend(self):
        figure, axes = self._render_without_warnings(
            2,
            PlotOptions(show_reference=False, legend_mode="Auto"),
        )

        self.assertEqual(len(axes), 2)
        self.assertEqual(len(figure.axes), 2)
        main_ax, subtracted_ax = axes
        self.assertEqual(
            main_ax.get_title(), "Normalized profiles and fitted backgrounds"
        )
        self.assertEqual(
            subtracted_ax.get_title(), "Background-subtracted profiles"
        )
        self.assertEqual(main_ax.get_xlabel(), r"Q ($\AA^{-1}$)")
        self.assertEqual(subtracted_ax.get_xlabel(), r"Q ($\AA^{-1}$)")
        self.assertEqual(main_ax.get_ylabel(), "Normalized intensity + offset")
        self.assertEqual(subtracted_ax.get_ylabel(), "Intensity (a.u.) + offset")

        legend = main_ax.get_legend()
        self.assertIsNotNone(legend)
        legend_labels = {text.get_text() for text in legend.get_texts()}
        self.assertTrue(
            {"th=0.260, samz=0", "th=0.260, samz=1"}.issubset(legend_labels)
        )

    def test_fifteen_results_with_reference_have_useful_colorbar_ticks(self):
        figure, axes = self._render_without_warnings(
            15,
            PlotOptions(
                show_reference=True,
                show_hkl=True,
                legend_mode="Auto",
                colorbar_label="Selected scan",
            ),
        )

        self.assertEqual(len(axes), 4)
        self.assertEqual(len(figure.axes), 6)
        main_ax, subtracted_ax, main_reference_ax, subtracted_reference_ax = axes
        self.assertEqual(main_ax.get_xlabel(), "")
        self.assertEqual(subtracted_ax.get_xlabel(), "")
        self.assertEqual(main_reference_ax.get_xlabel(), r"Q ($\AA^{-1}$)")
        self.assertEqual(subtracted_reference_ax.get_xlabel(), r"Q ($\AA^{-1}$)")
        self.assertEqual(
            [label.get_text() for label in main_reference_ax.get_yticklabels()],
            list(REFERENCE_PHASES),
        )

        colorbar_axes = [axis for axis in figure.axes if axis not in axes]
        self.assertEqual(len(colorbar_axes), 2)
        for colorbar_ax in colorbar_axes:
            with self.subTest(colorbar_axis=id(colorbar_ax)):
                ticks = colorbar_ax.get_yticks()
                labels = [label.get_text() for label in colorbar_ax.get_yticklabels()]
                self.assertEqual(len(ticks), 6)
                self.assertTrue(np.all(np.diff(ticks) > 0))
                self.assertAlmostEqual(float(ticks[0]), 0.0)
                self.assertAlmostEqual(float(ticks[-1]), 14.0)
                self.assertEqual(len(labels), len(ticks))
                self.assertTrue(all(labels))
                self.assertEqual(labels[0], "th=0.260, samz=0")
                self.assertEqual(labels[-1], "th=0.260, samz=14")
        self.assertIn("Selected scan", {axis.get_ylabel() for axis in colorbar_axes})


if __name__ == "__main__":
    unittest.main()
