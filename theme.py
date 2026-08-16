"""Accessible light theme shared by the XRD dashboard's Tk widgets.

Tk can inherit foreground colours from the operating-system theme even when a
widget supplies its own light background.  Keeping the palette and Ttk state
maps here makes every text/background pair explicit and prevents unreadable
light-on-light controls on macOS, Windows, and Linux.
"""

from __future__ import annotations

import sys
from tkinter import ttk


# Surfaces and text ---------------------------------------------------------
BG_APP = "#F8FAFC"
BG_PANEL = "#F1F5F9"
BG_CARD = "#FFFFFF"
INPUT_BG = "#FFFFFF"
INPUT_FG = "#172033"
FG_TEXT = "#172033"
BORDER_COLOR = "#CBD5E1"
COLOR_MUTED = "#475569"

# Actions.  White text on each normal/hover action colour is WCAG AA (4.5:1)
# or better.  Secondary buttons use the dark foreground declared below.
COLOR_PRIMARY = "#1D4ED8"
COLOR_HOVER_P = "#1E40AF"
COLOR_WARN = "#9A3412"
COLOR_HOVER_W = "#7C2D12"
COLOR_SUCCESS = "#047857"
COLOR_HOVER_S = "#065F46"
COLOR_SECONDARY = "#E2E8F0"
COLOR_HOVER_SECONDARY = "#CBD5E1"
BUTTON_FG = "#FFFFFF"

# Inputs, disabled controls, and selections --------------------------------
INPUT_SELECTION_BG = "#BFDBFE"
INPUT_SELECTION_FG = "#172033"
DISABLED_BG = "#E2E8F0"
DISABLED_FG = "#475569"

# Status and log colours ----------------------------------------------------
STATUS_READY = COLOR_SUCCESS
STATUS_BUSY = COLOR_WARN
STATUS_INFO = COLOR_PRIMARY
STATUS_ERROR = "#B91C1C"
STATUS_READY_BG = "#D1FAE5"
STATUS_BUSY_BG = "#FEF3C7"
STATUS_INFO_BG = "#DBEAFE"
STATUS_ERROR_BG = "#FEE2E2"
STATUS_NEUTRAL_BG = DISABLED_BG

LOG_BG = "#111827"
LOG_FG = "#F8FAFC"
LOG_MUTED = "#CBD5E1"
LOG_SUCCESS = "#6EE7B7"
LOG_WARNING = "#FDE68A"
LOG_ERROR = "#FCA5A5"


# Fonts --------------------------------------------------------------------
if sys.platform == "darwin":
    _UI_FONT = "Helvetica Neue"
    _MONO_FONT = "Menlo"
elif sys.platform == "win32":
    _UI_FONT = "Segoe UI"
    _MONO_FONT = "Consolas"
else:
    _UI_FONT = "DejaVu Sans"
    _MONO_FONT = "DejaVu Sans Mono"

FONT_H1 = (_UI_FONT, 17, "bold")
FONT_H2 = (_UI_FONT, 10, "bold")
FONT_BODY = (_UI_FONT, 10)
FONT_SMALL = (_UI_FONT, 9)
FONT_MONO = (_MONO_FONT, 9)


def _configure_choice_style(style: ttk.Style, name: str, background: str) -> None:
    """Configure a radio/check style with readable state colours."""

    style.configure(
        name,
        background=background,
        foreground=FG_TEXT,
        font=FONT_BODY,
        focuscolor=COLOR_PRIMARY,
        indicatorbackground=INPUT_BG,
        indicatorforeground=COLOR_PRIMARY,
        padding=(2, 2),
    )
    style.map(
        name,
        background=[
            ("disabled", background),
            ("active", BG_PANEL),
            ("!disabled", background),
        ],
        foreground=[
            ("disabled", DISABLED_FG),
            ("active", FG_TEXT),
            ("!disabled", FG_TEXT),
        ],
        indicatorbackground=[
            ("disabled", DISABLED_BG),
            ("selected", COLOR_PRIMARY),
            ("!selected", INPUT_BG),
        ],
        indicatorforeground=[
            ("disabled", DISABLED_FG),
            ("selected", BUTTON_FG),
            ("!selected", COLOR_PRIMARY),
        ],
    )


def _configure_button_style(
    style: ttk.Style,
    name: str,
    *,
    background: str,
    hover: str,
    foreground: str = BUTTON_FG,
) -> None:
    """Configure one action button with explicit hover/disabled colours."""

    style.configure(
        name,
        background=background,
        foreground=foreground,
        font=FONT_H2,
        bordercolor=background,
        lightcolor=background,
        darkcolor=background,
        focuscolor=background,
        relief="flat",
        padding=(12, 7),
    )
    style.map(
        name,
        background=[
            ("disabled", DISABLED_BG),
            ("pressed", hover),
            ("active", hover),
            ("!disabled", background),
        ],
        foreground=[
            ("disabled", DISABLED_FG),
            ("!disabled", foreground),
        ],
        bordercolor=[
            ("disabled", BORDER_COLOR),
            ("pressed", hover),
            ("active", hover),
            ("!disabled", background),
        ],
        lightcolor=[
            ("disabled", DISABLED_BG),
            ("active", hover),
            ("!disabled", background),
        ],
        darkcolor=[
            ("disabled", DISABLED_BG),
            ("active", hover),
            ("!disabled", background),
        ],
    )


def configure_ttk_styles(root) -> ttk.Style:
    """Apply the dashboard's explicit, accessible Ttk styles.

    ``clam`` is preferred because it honours custom field and indicator colours
    consistently.  The configured :class:`~tkinter.ttk.Style` is returned for
    callers that need to add a narrowly scoped component style.
    """

    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")

    style.configure(
        ".",
        background=BG_APP,
        foreground=FG_TEXT,
        font=FONT_BODY,
        focuscolor=COLOR_PRIMARY,
    )

    style.configure(
        "TCombobox",
        background=INPUT_BG,
        fieldbackground=INPUT_BG,
        foreground=INPUT_FG,
        arrowcolor=FG_TEXT,
        bordercolor=BORDER_COLOR,
        lightcolor=INPUT_BG,
        darkcolor=INPUT_BG,
        selectbackground=INPUT_SELECTION_BG,
        selectforeground=INPUT_SELECTION_FG,
        padding=(6, 4),
    )
    style.map(
        "TCombobox",
        background=[
            ("disabled", DISABLED_BG),
            ("readonly", INPUT_BG),
            ("active", BG_PANEL),
        ],
        fieldbackground=[
            ("disabled", DISABLED_BG),
            ("readonly", INPUT_BG),
            ("!disabled", INPUT_BG),
        ],
        foreground=[
            ("disabled", DISABLED_FG),
            ("readonly", INPUT_FG),
            ("!disabled", INPUT_FG),
        ],
        arrowcolor=[
            ("disabled", DISABLED_FG),
            ("!disabled", FG_TEXT),
        ],
        bordercolor=[
            ("invalid", STATUS_ERROR),
            ("focus", COLOR_PRIMARY),
            ("!focus", BORDER_COLOR),
        ],
        selectbackground=[("!disabled", INPUT_SELECTION_BG)],
        selectforeground=[("!disabled", INPUT_SELECTION_FG)],
    )

    for input_style in ("TEntry", "TSpinbox"):
        style.configure(
            input_style,
            background=INPUT_BG,
            fieldbackground=INPUT_BG,
            foreground=INPUT_FG,
            insertcolor=FG_TEXT,
            arrowcolor=FG_TEXT,
            bordercolor=BORDER_COLOR,
            lightcolor=INPUT_BG,
            darkcolor=INPUT_BG,
            selectbackground=INPUT_SELECTION_BG,
            selectforeground=INPUT_SELECTION_FG,
            padding=(6, 4),
        )
        style.map(
            input_style,
            fieldbackground=[
                ("disabled", DISABLED_BG),
                ("readonly", BG_PANEL),
                ("!disabled", INPUT_BG),
            ],
            foreground=[
                ("disabled", DISABLED_FG),
                ("!disabled", INPUT_FG),
            ],
            arrowcolor=[
                ("disabled", DISABLED_FG),
                ("!disabled", FG_TEXT),
            ],
            bordercolor=[
                ("invalid", STATUS_ERROR),
                ("focus", COLOR_PRIMARY),
                ("!focus", BORDER_COLOR),
            ],
            selectbackground=[("!disabled", INPUT_SELECTION_BG)],
            selectforeground=[("!disabled", INPUT_SELECTION_FG)],
        )

    _configure_choice_style(style, "Card.TRadiobutton", BG_CARD)
    _configure_choice_style(style, "Toolbar.TRadiobutton", BG_APP)
    _configure_choice_style(style, "Toolbar.TCheckbutton", BG_APP)

    _configure_button_style(
        style,
        "Primary.TButton",
        background=COLOR_PRIMARY,
        hover=COLOR_HOVER_P,
    )
    _configure_button_style(
        style,
        "Success.TButton",
        background=COLOR_SUCCESS,
        hover=COLOR_HOVER_S,
    )
    _configure_button_style(
        style,
        "Warning.TButton",
        background=COLOR_WARN,
        hover=COLOR_HOVER_W,
    )
    _configure_button_style(
        style,
        "Secondary.TButton",
        background=COLOR_SECONDARY,
        hover=COLOR_HOVER_SECONDARY,
        foreground=FG_TEXT,
    )

    for scrollbar_style in ("Vertical.TScrollbar", "Horizontal.TScrollbar"):
        style.configure(
            scrollbar_style,
            background=COLOR_SECONDARY,
            troughcolor=BG_PANEL,
            arrowcolor=FG_TEXT,
            bordercolor=BORDER_COLOR,
            lightcolor=COLOR_SECONDARY,
            darkcolor=COLOR_SECONDARY,
            width=14,
        )
        style.map(
            scrollbar_style,
            background=[
                ("pressed", COLOR_MUTED),
                ("active", COLOR_HOVER_SECONDARY),
                ("!disabled", COLOR_SECONDARY),
            ],
            arrowcolor=[
                ("disabled", DISABLED_FG),
                ("!disabled", FG_TEXT),
            ],
        )

    style.configure(
        "TProgressbar",
        background=COLOR_PRIMARY,
        troughcolor=DISABLED_BG,
        bordercolor=BORDER_COLOR,
        lightcolor=COLOR_PRIMARY,
        darkcolor=COLOR_PRIMARY,
    )

    style.configure("TPanedwindow", background=BORDER_COLOR, sashwidth=6)
    style.configure("Sash", background=BORDER_COLOR, sashthickness=6)

    # The Combobox pop-down is a Tk Listbox rather than a Ttk element, so its
    # colours must be set through the option database as well.
    root.option_add("*TCombobox*Listbox.background", INPUT_BG)
    root.option_add("*TCombobox*Listbox.foreground", INPUT_FG)
    root.option_add("*TCombobox*Listbox.selectBackground", INPUT_SELECTION_BG)
    root.option_add("*TCombobox*Listbox.selectForeground", INPUT_SELECTION_FG)

    return style


__all__ = [
    "BG_APP",
    "BG_PANEL",
    "BG_CARD",
    "INPUT_BG",
    "INPUT_FG",
    "FG_TEXT",
    "BORDER_COLOR",
    "COLOR_PRIMARY",
    "COLOR_HOVER_P",
    "COLOR_WARN",
    "COLOR_HOVER_W",
    "COLOR_SUCCESS",
    "COLOR_HOVER_S",
    "COLOR_SECONDARY",
    "COLOR_HOVER_SECONDARY",
    "COLOR_MUTED",
    "BUTTON_FG",
    "INPUT_SELECTION_BG",
    "INPUT_SELECTION_FG",
    "DISABLED_BG",
    "DISABLED_FG",
    "STATUS_READY",
    "STATUS_BUSY",
    "STATUS_INFO",
    "STATUS_ERROR",
    "STATUS_READY_BG",
    "STATUS_BUSY_BG",
    "STATUS_INFO_BG",
    "STATUS_ERROR_BG",
    "STATUS_NEUTRAL_BG",
    "LOG_BG",
    "LOG_FG",
    "LOG_MUTED",
    "LOG_SUCCESS",
    "LOG_WARNING",
    "LOG_ERROR",
    "FONT_H1",
    "FONT_H2",
    "FONT_BODY",
    "FONT_SMALL",
    "FONT_MONO",
    "configure_ttk_styles",
]
