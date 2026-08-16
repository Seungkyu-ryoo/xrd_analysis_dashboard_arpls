"""Pure Matplotlib rendering for the XRD analysis dashboard.

The UI owns data selection, fitting, click markers, and canvas updates.  This
module only mutates the supplied :class:`matplotlib.figure.Figure`, which keeps
the rendering code usable in Tk, headless tests, and exported figures alike.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..analysis.pipeline import split_position
from ..data.workbooks import REFERENCE_PHASES
from .theme import (
    BG_APP,
    BG_CARD,
    BORDER_COLOR,
    COLOR_MUTED,
    FG_TEXT,
    REFERENCE_COLORS,
)


LegendMode = Literal["Auto", "Legend", "Colorbar", "None"]
OffsetMode = Literal["auto", "manual"]
FitResult: TypeAlias = Mapping[str, Any]
PlotItem: TypeAlias = tuple[str, str, FitResult]

# Okabe-Ito colours, with the low-contrast yellow omitted, followed by Tol
# accents.  The palette remains distinguishable without relying on markers.
ACCESSIBLE_CURVE_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
    "#332288",
    "#44AA99",
    "#882255",
    "#117733",
    "#AA4499",
)

BACKGROUND_FIT_COLOR = "#334155"
ANCHOR_COLOR = "#0F766E"
MASK_COLOR = "#64748B"
REFERENCE_ROW_COLORS = ("#FFFFFF", "#E2E8F0")


@dataclass(frozen=True, slots=True)
class PlotOptions:
    """Immutable snapshot of every option that affects a dashboard figure.

    ``anchors`` contains ``(q_min, q_max, weight)`` tuples; the weight belongs
    to fitting and is retained here so the renderer can consume the same
    validated structure.  ``masks`` contains ``(q_min, q_max)`` tuples.

    ``cmap_name=None`` uses the accessible qualitative palette above.  Passing
    a Matplotlib colormap name preserves the dashboard's selectable colour-map
    behaviour.  ``show_top_xaxis`` retains the legacy option name: it shows Q
    tick labels on the two main plots when reference panels are present.
    """

    q_min: float | None = None
    q_max: float | None = None
    cmap_name: str | None = None
    legend_mode: LegendMode = "Auto"
    show_reference: bool = True
    show_hkl: bool = True
    show_top_xaxis: bool = False
    show_background: bool = True
    show_anchors: bool = True
    show_masks: bool = True
    anchors: tuple[tuple[float, float, float], ...] = ()
    masks: tuple[tuple[float, float], ...] = ()
    offset_mode: OffsetMode = "auto"
    normalized_offset: float = 0.5
    raw_offset: float = 500.0
    width_ratio: float = 1.0
    height_ratio: float = 2.5
    title_font_size: int = 12
    axis_title_font_size: int = 11
    axis_tick_font_size: int = 10
    legend_font_size: int = 8
    colorbar_label: str = "Selected scan"

    def __post_init__(self) -> None:
        """Normalize nested sequences and reject invalid layout settings."""

        object.__setattr__(
            self,
            "anchors",
            tuple(tuple(float(value) for value in anchor) for anchor in self.anchors),
        )
        object.__setattr__(
            self,
            "masks",
            tuple(tuple(float(value) for value in mask) for mask in self.masks),
        )
        if self.offset_mode not in ("auto", "manual"):
            raise ValueError("offset_mode must be 'auto' or 'manual'.")
        if self.legend_mode not in ("Auto", "Legend", "Colorbar", "None"):
            raise ValueError(
                "legend_mode must be Auto, Legend, Colorbar, or None."
            )
        if not np.isfinite(self.width_ratio) or self.width_ratio <= 0:
            raise ValueError("width_ratio must be a finite positive number.")
        if not np.isfinite(self.height_ratio) or self.height_ratio <= 0:
            raise ValueError("height_ratio must be a finite positive number.")
        for name in (
            "title_font_size",
            "axis_title_font_size",
            "axis_tick_font_size",
            "legend_font_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")


def _display_label(position: str, fallback: str) -> str:
    """Return an unambiguous human-readable scan label."""

    th_value, samz_value = split_position(position)
    if samz_value:
        return f"th={th_value}, samz={samz_value}"
    return fallback or position


def _curve_colours(count: int, cmap_name: str | None) -> tuple[np.ndarray, Colormap]:
    """Return line colours and a matching discrete colormap for colorbars."""

    if cmap_name is None and count <= len(ACCESSIBLE_CURVE_COLORS):
        sampled = np.asarray(ACCESSIBLE_CURVE_COLORS[:count], dtype=object)
        rgba = np.asarray([mcolors.to_rgba(colour) for colour in sampled])
    else:
        requested = cmap_name or "cividis"
        try:
            source = matplotlib.colormaps[requested]
        except KeyError as exc:
            raise ValueError(f"Unknown Matplotlib colormap: {requested!r}") from exc
        locations = np.linspace(0.08, 0.92, count) if count > 1 else np.array([0.5])
        rgba = np.asarray(source(locations))
    return rgba, ListedColormap(rgba, name="dashboard_curves")


def _q_limits(q_values: np.ndarray, options: PlotOptions) -> tuple[float, float]:
    q_min = float(np.min(q_values)) if options.q_min is None else float(options.q_min)
    q_max = float(np.max(q_values)) if options.q_max is None else float(options.q_max)
    if not np.isfinite(q_min) or not np.isfinite(q_max) or q_min >= q_max:
        raise ValueError("Plot Q limits must be finite and increasing.")
    return q_min, q_max


def _as_curve(result: FitResult, key: str, q_values: np.ndarray) -> np.ndarray:
    """Read and validate one curve from a cached fit result."""

    values = np.asarray(result[key], dtype=float)
    if values.shape != q_values.shape:
        raise ValueError(
            f"Fit result {key!r} has shape {values.shape}; expected {q_values.shape}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Fit result {key!r} contains NaN or infinite values.")
    return values


def _draw_reference_peaks(
    ax: Axes,
    reference_data: Mapping[str, Any],
    options: PlotOptions,
    q_min: float,
    q_max: float,
) -> None:
    """Draw phase rows, normalized peak stems, and collision-aware HKL labels."""

    phase_count = len(REFERENCE_PHASES)
    ax.set_facecolor(BG_CARD)
    ax.set_ylim(0, phase_count * 2)
    ax.set_xlim(q_min, q_max)
    yticks: list[float] = []
    yticklabels: list[str] = []
    q_span = q_max - q_min
    text_width_q = max(q_span * 0.014, 0.018)

    for index, phase in enumerate(REFERENCE_PHASES):
        y_base = index * 2.0
        row_colour = REFERENCE_ROW_COLORS[index % len(REFERENCE_ROW_COLORS)]
        ax.axhspan(
            y_base,
            y_base + 2.0,
            facecolor=row_colour,
            edgecolor="none",
            zorder=0,
        )
        ax.axhline(y_base, color=BORDER_COLOR, linewidth=1.0, zorder=1)
        yticks.append(y_base + 1.0)
        yticklabels.append(phase)

        phase_data = reference_data.get(phase)
        if phase_data is None or phase_data.empty:
            continue

        colour = REFERENCE_COLORS[index % len(REFERENCE_COLORS)]
        y_tops = phase_data["Intensity_Norm"].to_numpy(dtype=float) * 0.9 + y_base
        q_peaks = phase_data["Q"].to_numpy(dtype=float)
        ax.vlines(
            q_peaks,
            ymin=y_base,
            ymax=y_tops,
            color=colour,
            linewidth=2.0,
            zorder=10,
        )
        if not options.show_hkl:
            continue

        placed_rectangles: list[tuple[float, float, float, float]] = []
        text_height_y = 0.45
        maximum_y = y_base + 1.95
        horizontal_offsets = (0.0, q_span * 0.008, -q_span * 0.008,
                              q_span * 0.016, -q_span * 0.016)
        vertical_offsets = (0.05, 0.25, 0.45)

        for (_, row), peak_y_top in zip(phase_data.iterrows(), y_tops):
            q_value = float(row["Q"])
            best_x, best_y = q_value, float(peak_y_top) + 0.05
            found_place = False
            for vertical_offset in vertical_offsets:
                for horizontal_offset in horizontal_offsets:
                    candidate_x = q_value + horizontal_offset
                    candidate_y = float(peak_y_top) + vertical_offset
                    if not q_min + 0.01 * q_span <= candidate_x <= q_max - 0.01 * q_span:
                        continue
                    if candidate_y + text_height_y > maximum_y:
                        continue
                    candidate = (
                        candidate_x - text_width_q / 2.0,
                        candidate_x + text_width_q / 2.0,
                        candidate_y,
                        candidate_y + text_height_y,
                    )
                    overlaps = any(
                        candidate[0] < placed[1]
                        and candidate[1] > placed[0]
                        and candidate[2] < placed[3]
                        and candidate[3] > placed[2]
                        for placed in placed_rectangles
                    )
                    if overlaps:
                        continue
                    best_x, best_y = candidate_x, candidate_y
                    placed_rectangles.append(candidate)
                    found_place = True
                    break
                if found_place:
                    break

            if not found_place:
                best_y = min(best_y, maximum_y - text_height_y)
                placed_rectangles.append(
                    (
                        best_x - text_width_q / 2.0,
                        best_x + text_width_q / 2.0,
                        best_y,
                        best_y + text_height_y,
                    )
                )

            ax.text(
                best_x,
                best_y,
                str(row.get("hkl", "")),
                color=colour,
                fontsize=max(7, options.axis_tick_font_size - 1),
                rotation=90,
                verticalalignment="bottom",
                horizontalalignment="center",
                zorder=20,
            )
            if abs(best_x - q_value) > 0.01 * q_span or best_y - peak_y_top > 0.15:
                ax.plot(
                    [q_value, best_x],
                    [float(peak_y_top) + 0.02, best_y - 0.01],
                    color=colour,
                    linewidth=0.6,
                    alpha=0.65,
                    zorder=5,
                )

    ax.axhline(phase_count * 2.0, color=BORDER_COLOR, linewidth=1.0, zorder=1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(
        yticklabels,
        fontsize=options.axis_tick_font_size,
        fontweight="bold",
        color=FG_TEXT,
    )


def _colorbar_ticks(item_count: int) -> np.ndarray:
    """Choose useful, non-overlapping colorbar indices (up to six)."""

    if item_count <= 6:
        return np.arange(item_count, dtype=int)
    return np.unique(np.rint(np.linspace(0, item_count - 1, 6)).astype(int))


def render_dashboard_figure(
    figure: Figure,
    q_values: Sequence[float] | np.ndarray,
    plot_items: Sequence[PlotItem],
    reference_data: Mapping[str, Any] | None,
    options: PlotOptions,
) -> tuple[Axes, ...]:
    """Render cached XRD fits into ``figure`` and return the created axes.

    Each plot item is ``(position, selection_label, fit_result)``.  A fit result
    must expose ``target_norm``, ``best_bg_fit``, and ``clean_target_raw`` arrays
    matching ``q_values``.  The returned tuple is ``(main, subtracted)`` when
    references are hidden and ``(main, subtracted, main_ref, subtracted_ref)``
    otherwise.  Click markers and ``canvas.draw*`` calls deliberately remain the
    caller's responsibility.
    """

    q_array = np.asarray(q_values, dtype=float)
    if q_array.ndim != 1 or q_array.size < 2 or not np.all(np.isfinite(q_array)):
        raise ValueError("q_values must be a finite one-dimensional array.")
    if not plot_items:
        raise ValueError("At least one plot item is required.")
    q_min, q_max = _q_limits(q_array, options)

    prepared = []
    for position, fallback_label, result in plot_items:
        prepared.append(
            (
                position,
                _display_label(position, fallback_label),
                _as_curve(result, "target_norm", q_array),
                _as_curve(result, "best_bg_fit", q_array),
                _as_curve(result, "clean_target_raw", q_array),
            )
        )

    figure.clear()
    figure.patch.set_facecolor(BG_APP)
    if options.show_reference:
        layout = gridspec.GridSpec(
            2,
            2,
            figure=figure,
            height_ratios=[options.height_ratio, 1.0],
            width_ratios=[options.width_ratio, 1.0],
            hspace=0.1,
            wspace=0.25,
        )
        main_ax = figure.add_subplot(layout[0, 0])
        subtracted_ax = figure.add_subplot(layout[0, 1])
        main_reference_ax = figure.add_subplot(layout[1, 0], sharex=main_ax)
        subtracted_reference_ax = figure.add_subplot(
            layout[1, 1], sharex=subtracted_ax
        )
        axes: tuple[Axes, ...] = (
            main_ax,
            subtracted_ax,
            main_reference_ax,
            subtracted_reference_ax,
        )
    else:
        layout = gridspec.GridSpec(
            1,
            2,
            figure=figure,
            width_ratios=[options.width_ratio, 1.0],
            wspace=0.25,
        )
        main_ax = figure.add_subplot(layout[0, 0])
        subtracted_ax = figure.add_subplot(layout[0, 1])
        main_reference_ax = subtracted_reference_ax = None
        axes = (main_ax, subtracted_ax)

    for ax in axes:
        ax.set_facecolor(BG_CARD)
        ax.set_xlim(q_min, q_max)

    max_raw_range = max(float(np.ptp(item[4])) for item in prepared)
    if options.offset_mode == "manual":
        normalized_offset = float(options.normalized_offset)
        raw_offset = float(options.raw_offset)
    else:
        normalized_offset = float(np.ptp(prepared[-1][3])) * 0.65
        raw_offset = max_raw_range * 1.1

    colours, discrete_cmap = _curve_colours(len(prepared), options.cmap_name)
    raw_cumulative_offset = 0.0
    raw_overlay = np.isclose(raw_offset, 0.0)
    raw_min, raw_max = np.inf, -np.inf
    curve_handles: list[Line2D] = []
    curve_labels: list[str] = []

    for index, (_position, label, normalized, background, cleaned) in enumerate(prepared):
        colour = colours[index]
        normalized_y = normalized + index * normalized_offset
        (curve_line,) = main_ax.plot(
            q_array,
            normalized_y,
            color=colour,
            alpha=0.95,
            linewidth=1.7,
            zorder=10,
        )
        curve_handles.append(curve_line)
        curve_labels.append(label)
        if options.show_background:
            main_ax.plot(
                q_array,
                background + index * normalized_offset,
                color=BACKGROUND_FIT_COLOR,
                linestyle="--",
                alpha=0.85,
                linewidth=1.2,
                zorder=5,
            )

        raw_y = cleaned if raw_overlay else cleaned + raw_cumulative_offset
        subtracted_ax.plot(
            q_array,
            raw_y,
            color=colour,
            linewidth=1.8,
            zorder=10,
        )
        raw_min = min(raw_min, float(np.min(raw_y)))
        raw_max = max(raw_max, float(np.max(raw_y)))
        if not raw_overlay:
            raw_cumulative_offset += raw_offset

    normalized_min, normalized_max = main_ax.get_ylim()
    normalized_range = normalized_max - normalized_min
    if np.isclose(normalized_range, 0.0):
        normalized_range = max(abs(normalized_max), 1.0) * 0.1
    main_ax.set_ylim(
        normalized_min,
        normalized_max + normalized_range * 0.15,
    )
    raw_range = raw_max - raw_min
    raw_padding = (
        max(abs(raw_max), 1.0) * 0.1
        if np.isclose(raw_range, 0.0)
        else raw_range * 0.15
    )
    subtracted_ax.set_ylim(raw_min - raw_padding, raw_max + raw_padding)

    references = reference_data or {}
    if options.show_reference:
        assert main_reference_ax is not None and subtracted_reference_ax is not None
        _draw_reference_peaks(main_reference_ax, references, options, q_min, q_max)
        _draw_reference_peaks(
            subtracted_reference_ax, references, options, q_min, q_max
        )

    for ax in axes:
        if options.show_masks:
            for start, end in options.masks:
                ax.axvspan(start, end, color=MASK_COLOR, alpha=0.16, zorder=0)
        if options.show_anchors:
            for start, end, _weight in options.anchors:
                ax.axvspan(start, end, color=ANCHOR_COLOR, alpha=0.13, zorder=0)
        ax.grid(axis="x", color=BORDER_COLOR, linewidth=0.6, alpha=0.55, zorder=-1)
        ax.tick_params(
            axis="both",
            colors=FG_TEXT,
            labelsize=options.axis_tick_font_size,
        )
        for spine in ax.spines.values():
            spine.set_color(BORDER_COLOR)
            spine.set_linewidth(1.2)

    main_ax.set_title(
        "Normalized profiles and fitted backgrounds",
        fontsize=options.title_font_size,
        fontweight="bold",
        color=FG_TEXT,
        pad=10,
    )
    subtracted_ax.set_title(
        "Background-subtracted profiles",
        fontsize=options.title_font_size,
        fontweight="bold",
        color=FG_TEXT,
        pad=10,
    )
    main_ax.set_ylabel(
        "Normalized intensity + offset",
        fontsize=options.axis_title_font_size,
        fontweight="bold",
        color=FG_TEXT,
    )
    subtracted_ax.set_ylabel(
        "Intensity (a.u.) + offset",
        fontsize=options.axis_title_font_size,
        fontweight="bold",
        color=FG_TEXT,
    )

    q_label = r"Q ($\AA^{-1}$)"
    if options.show_reference:
        assert main_reference_ax is not None and subtracted_reference_ax is not None
        for ax in (main_reference_ax, subtracted_reference_ax):
            ax.set_xlabel(
                q_label,
                fontsize=options.axis_title_font_size,
                fontweight="bold",
                color=FG_TEXT,
            )
        for ax in (main_ax, subtracted_ax):
            ax.tick_params(labelbottom=options.show_top_xaxis)
            ax.set_xlabel(q_label if options.show_top_xaxis else "")
            if options.show_top_xaxis:
                ax.xaxis.label.set_fontsize(options.axis_title_font_size)
                ax.xaxis.label.set_fontweight("bold")
                ax.xaxis.label.set_color(FG_TEXT)
    else:
        for ax in (main_ax, subtracted_ax):
            ax.set_xlabel(
                q_label,
                fontsize=options.axis_title_font_size,
                fontweight="bold",
                color=FG_TEXT,
            )

    legend_mode = options.legend_mode
    use_legend = legend_mode == "Legend" or (
        legend_mode == "Auto" and len(prepared) <= 12
    )
    use_colorbar = legend_mode == "Colorbar" or (
        legend_mode == "Auto" and len(prepared) > 12
    )
    overlay_handles: list[Line2D | Patch] = []
    overlay_labels: list[str] = []
    if options.show_background:
        overlay_handles.append(
            Line2D([0], [0], color=BACKGROUND_FIT_COLOR, linestyle="--", linewidth=1.4)
        )
        overlay_labels.append("Fitted background")
    if options.show_anchors and options.anchors:
        overlay_handles.append(Patch(facecolor=ANCHOR_COLOR, alpha=0.2, edgecolor="none"))
        overlay_labels.append("Anchor range")
    if options.show_masks and options.masks:
        overlay_handles.append(Patch(facecolor=MASK_COLOR, alpha=0.22, edgecolor="none"))
        overlay_labels.append("Masked peak range")

    if use_legend:
        main_ax.legend(
            curve_handles[::-1] + overlay_handles,
            curve_labels[::-1] + overlay_labels,
            loc="upper right",
            fontsize=options.legend_font_size,
            framealpha=0.94,
        )
    elif use_colorbar:
        norm = mcolors.Normalize(vmin=0, vmax=max(1, len(prepared) - 1))
        scalar_map = ScalarMappable(cmap=discrete_cmap, norm=norm)
        scalar_map.set_array([])
        tick_indices = _colorbar_ticks(len(prepared))
        tick_labels = [curve_labels[index] for index in tick_indices]
        for ax in (main_ax, subtracted_ax):
            colorbar = figure.colorbar(
                scalar_map,
                ax=ax,
                shrink=0.8,
                aspect=25,
                pad=0.03,
            )
            colorbar.set_ticks(tick_indices)
            colorbar.set_ticklabels(tick_labels)
            colorbar.ax.tick_params(
                colors=FG_TEXT,
                labelsize=options.axis_tick_font_size,
            )
        colorbar.set_label(
            options.colorbar_label,
            fontsize=options.axis_title_font_size,
            color=FG_TEXT,
        )
        if overlay_handles:
            main_ax.legend(
                overlay_handles,
                overlay_labels,
                loc="upper right",
                fontsize=options.legend_font_size,
                framealpha=0.94,
            )

    vertical_spacing = 0.4 if options.show_top_xaxis else 0.15
    # Explicit margins work for ordinary axes and Matplotlib's non-subplot
    # colorbar axes.  ``tight_layout`` warns for the latter and can break the
    # alignment between each main plot and its reference panel.
    figure.subplots_adjust(
        left=0.10,
        right=0.96,
        bottom=0.10,
        top=0.91,
        wspace=0.30,
        hspace=vertical_spacing,
    )
    if options.show_reference and use_colorbar:
        assert main_reference_ax is not None and subtracted_reference_ax is not None
        main_position = main_ax.get_position()
        reference_position = main_reference_ax.get_position()
        main_reference_ax.set_position(
            [
                main_position.x0,
                reference_position.y0,
                main_position.width,
                reference_position.height,
            ]
        )
        subtracted_position = subtracted_ax.get_position()
        subtracted_reference_position = subtracted_reference_ax.get_position()
        subtracted_reference_ax.set_position(
            [
                subtracted_position.x0,
                subtracted_reference_position.y0,
                subtracted_position.width,
                subtracted_reference_position.height,
            ]
        )

    return axes


__all__ = [
    "ACCESSIBLE_CURVE_COLORS",
    "FitResult",
    "PlotItem",
    "PlotOptions",
    "render_dashboard_figure",
]
