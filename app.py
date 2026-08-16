"""Tkinter controller for interactive XRD background analysis.

Numerical processing, workbook parsing, plotting, and theme configuration live
in focused sibling modules.  This module coordinates widgets and background
workers while keeping every Tk and file-dialog operation on the main thread.
"""

import os
import sys
import queue
import threading
import warnings

import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

if __package__:  # Support package/module execution from the repository root.
    from .analysis import (
        MAX_FIT_WORKERS,
        compute_fit_jobs,
        expanded_background_q_range,
        numeric_sort_key,
        parse_anchor_lines,
        parse_peak_lines,
        q_to_2theta,
        safe_filename_component,
        validate_background_q_support,
        validate_q_range,
    )
    from .data_io import (
        REFERENCE_FILENAME,
        load_reference_peaks,
        read_multisheet,
    )
    from .fitting_optimized import BackgroundFitter
    from .exporting import write_export
    from .plotting import PlotOptions, render_dashboard_figure
    from .theme import (
        BG_APP,
        BG_CARD,
        BG_PANEL,
        BORDER_COLOR,
        COLOR_MUTED,
        COLOR_PRIMARY,
        COLOR_SUCCESS,
        COLOR_WARN,
        DISABLED_FG,
        FG_TEXT,
        FONT_BODY,
        FONT_H1,
        FONT_H2,
        FONT_MONO,
        FONT_SMALL,
        INPUT_BG,
        INPUT_FG,
        INPUT_SELECTION_BG,
        INPUT_SELECTION_FG,
        LOG_BG,
        LOG_FG,
        STATUS_ERROR,
        STATUS_ERROR_BG,
        STATUS_INFO,
        STATUS_INFO_BG,
        STATUS_BUSY_BG,
        STATUS_NEUTRAL_BG,
        STATUS_READY_BG,
        configure_ttk_styles,
    )
else:  # Support direct imports and `python app.py` from this directory.
    from analysis import (
        MAX_FIT_WORKERS,
        compute_fit_jobs,
        expanded_background_q_range,
        numeric_sort_key,
        parse_anchor_lines,
        parse_peak_lines,
        q_to_2theta,
        safe_filename_component,
        validate_background_q_support,
        validate_q_range,
    )
    from data_io import (
        REFERENCE_FILENAME,
        load_reference_peaks,
        read_multisheet,
    )
    from fitting_optimized import BackgroundFitter
    from exporting import write_export
    from plotting import PlotOptions, render_dashboard_figure
    from theme import (
        BG_APP,
        BG_CARD,
        BG_PANEL,
        BORDER_COLOR,
        COLOR_MUTED,
        COLOR_PRIMARY,
        COLOR_SUCCESS,
        COLOR_WARN,
        DISABLED_FG,
        FG_TEXT,
        FONT_BODY,
        FONT_H1,
        FONT_H2,
        FONT_MONO,
        FONT_SMALL,
        INPUT_BG,
        INPUT_FG,
        INPUT_SELECTION_BG,
        INPUT_SELECTION_FG,
        LOG_BG,
        LOG_FG,
        STATUS_ERROR,
        STATUS_ERROR_BG,
        STATUS_INFO,
        STATUS_INFO_BG,
        STATUS_BUSY_BG,
        STATUS_NEUTRAL_BG,
        STATUS_READY_BG,
        configure_ttk_styles,
    )

CMAP_OPTIONS = {
    "Accessible (recommended)": None,
    "Categorical (tab20)": "tab20",
    "Sequential (viridis)": "viridis",
    "Colorblind-safe (cividis)": "cividis",
    "Grayscale": "gray",
}

PRESETS = {
    "HZO, TiN": {
        "anchors": "1.50, 1.60, 100.0\n1.80, 1.90, 100.0\n2.72, 2.78, 100.0\n3.25, 3.30, 100.0\n4.40, 4.50, 100.0",
        "masks": "1.65, 1.8\n1.90, 2.65\n2.8, 3.3\n3.35, 3.6\n3.9, 4.5",
    },
    "HZO fitting": {
        "anchors": "1.60, 1.75, 100.0\n1.85, 1.90, 100.0\n2.27, 2.33, 100.0\n2.72, 2.78, 100.0\n3.15, 3.21, 100.0\n3.64, 3.70, 100.0",
        "masks": "1.90, 2.25\n2.33, 2.52\n3.35, 3.6",
    },
    "TiN fitting": {
        "anchors": "1.60, 1.75, 100.0\n2.626, 2.828, 100.0\n3.64, 3.70, 100.0\n4.48, 4.6, 100.0",
        "masks": "2.46, 2.626\n2.828, 3.252\n4, 4.5",
    },
    "TiN fitting 2": {
        "anchors": "1.60, 1.75, 100.0\n2.74, 2.828, 100.0\n3.64, 3.70, 100.0\n4.48, 4.6, 100.0",
        "masks": "2.46, 2.74\n2.828, 3.252\n4, 4.5",
    },
}

REF_FILENAME = REFERENCE_FILENAME


def _is_secondary_click(event):
    """Normalize right-click events across Tk 8.6 and Tk 9 on macOS.

    Tk 9 reports the secondary mouse button as native button 3 on every
    platform.  Matplotlib versions that still apply the legacy macOS 2/3 swap
    consequently expose that click as button 2.  Prefer Tk's native button
    number when it is available so a real middle click is not misclassified.
    """
    button = getattr(event, "button", None)
    modifiers = set(getattr(event, "modifiers", ()) or ())

    if sys.platform == "darwin" and button == 1 and "ctrl" in modifiers:
        return True

    if sys.platform != "darwin" or tk.TkVersion < 9:
        return button == 3

    gui_event = getattr(event, "guiEvent", None)
    native_button = getattr(gui_event, "num", None)
    if native_button is not None:
        return native_button == 3

    # Synthetic/non-Tk events do not carry guiEvent.  Accept both the legacy
    # and corrected Matplotlib representations in that case.
    return button in (2, 3)


def enable_high_dpi():
    """Per-monitor DPI awareness on Windows so the UI is not blurry on
    scaled displays."""
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass


def ensure_supported_tk(root):
    """Reject Apple's legacy Tk 8.5, which misrenders modern dashboard widgets."""
    patchlevel = str(root.tk.call("info", "patchlevel"))
    try:
        major_minor = tuple(int(part) for part in patchlevel.split(".")[:2])
    except ValueError:
        major_minor = (0, 0)
    if major_minor >= (8, 6):
        return True

    message = (
        f"Unsupported Tk {patchlevel} detected. This dashboard requires Tk 8.6 "
        "or newer. Recreate .venv with the Miniconda Python documented in README.md."
    )
    root.withdraw()
    messagebox.showerror("Unsupported Python/Tk environment", message, parent=root)
    print(f"❌ {message}", file=sys.stderr)
    root.destroy()
    return False


# =========================================================
# Application
# =========================================================
class XRDDashboard:
    def __init__(self, root):
        self.root = root

        # --- data state ---
        self.df_target = None
        self.df_bg = None
        self.q_target = None
        self.q_bg = None
        self.global_ths = []
        self.global_samzs = []
        self.bg_cols = []
        self.fitter = None
        self.q_min, self.q_max = 1.6, 3.7
        self.loaded_target_path = None
        self.loaded_bg_path = None
        self._data_inputs_dirty = False

        # --- caches ---
        self.fit_cache = {}        # key -> fit result dict
        self.fit_failed = {}       # key -> error message (don't retry)
        self.ref_cache = {}        # (q_min, q_max) -> ref peak dict
        self._data_rev = 0         # bumped on every data (re)load

        # --- worker plumbing ---
        self._queue = queue.Queue()
        self._computing = False
        self._exporting = False
        self._loading = False
        self._pending_update = False
        self._render_job = None
        self._update_job = None
        self._closing = False

        # --- plot state ---
        self.current_selection_order = []
        self.clicked_q_values = []
        self.marker_objects = {}
        self.ax1 = self.ax2 = self.ax3 = self.ax4 = None

        self._build_ui()
        self._sync_action_states()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_queue()

        # Auto-load default files AFTER the window is shown
        if os.path.exists(self.target_file_var.get()) and os.path.exists(
            self.bg_file_var.get()
        ):
            self.root.after(300, self.load_data)

    # =====================================================
    # UI construction
    # =====================================================
    def _build_ui(self):
        root = self.root
        root.title("XRD Background Analysis")
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        width = min(1650, max(1100, int(screen_width * 0.92)))
        height = min(950, max(700, int(screen_height * 0.88)))
        x_offset = max(0, (screen_width - width) // 2)
        y_offset = max(0, (screen_height - height) // 2)
        root.geometry(f"{width}x{height}+{x_offset}+{y_offset}")
        root.minsize(1100, 700)
        root.configure(bg=BG_APP)

        self.style = configure_ttk_styles(root)

        main_paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)

        # Left control panel keeps a stable width regardless of content
        self.control_frame = tk.Frame(main_paned, width=370, bg=BG_PANEL)
        self.control_frame.pack_propagate(False)
        main_paned.add(self.control_frame, weight=0)

        self._build_left_panel()

        # Right plot panel
        self.plot_frame = tk.Frame(main_paned, bg=BG_APP)
        main_paned.add(self.plot_frame, weight=3)
        self._build_plot_panel()

    # ---------------- left panel ----------------
    def _build_left_panel(self):
        cf = self.control_frame

        title_frame = tk.Frame(cf, bg=BG_PANEL)
        title_frame.pack(side=tk.TOP, fill=tk.X, pady=(20, 10), padx=20)
        tk.Label(
            title_frame,
            text="XRD Background Analysis",
            font=FONT_H1,
            bg=BG_PANEL,
            fg=COLOR_PRIMARY,
        ).pack(anchor=tk.W)
        tk.Label(
            title_frame,
            text="Load  →  Select  →  Process  →  Export",
            font=FONT_BODY,
            bg=BG_PANEL,
            fg=COLOR_MUTED,
        ).pack(anchor=tk.W)
        tk.Frame(cf, bg=BORDER_COLOR, height=1).pack(
            side=tk.TOP, fill=tk.X, padx=20, pady=(0, 10)
        )

        # Bottom action buttons
        btn_frame = tk.Frame(cf, bg=BG_PANEL)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=(10, 15))

        self.btn_run = self._styled_btn(
            btn_frame,
            "Run analysis",
            "Primary.TButton",
            self._retry_analysis,
        )
        self.btn_run.pack(fill=tk.X, pady=(0, 8), ipady=3)

        sub_btn_frame = tk.Frame(btn_frame, bg=BG_PANEL)
        sub_btn_frame.pack(fill=tk.X)
        self._styled_btn(
            sub_btn_frame,
            "Clear markers",
            "Secondary.TButton",
            self.clear_markers,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.btn_export = self._styled_btn(
            sub_btn_frame,
            "Export data",
            "Success.TButton",
            self.export_all,
        )
        self.btn_export.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

        # A scrollable stack is more robust than five vertically nested panes:
        # each card keeps its requested height and can never overlap another
        # card, even with a small window or a platform-specific font scale.
        scroll_host = tk.Frame(cf, bg=BG_PANEL)
        scroll_host.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=(15, 8))

        self.sidebar_canvas = tk.Canvas(
            scroll_host,
            bg=BG_PANEL,
            highlightthickness=0,
            bd=0,
            yscrollincrement=18,
        )
        sidebar_scroll = ttk.Scrollbar(
            scroll_host, orient=tk.VERTICAL, command=self.sidebar_canvas.yview
        )
        self.sidebar_canvas.configure(yscrollcommand=sidebar_scroll.set)
        sidebar_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.sidebar_inner = tk.Frame(self.sidebar_canvas, bg=BG_PANEL)
        self._sidebar_window = self.sidebar_canvas.create_window(
            (0, 0), window=self.sidebar_inner, anchor=tk.NW
        )
        self.sidebar_inner.bind("<Configure>", self._sync_sidebar_scrollregion)
        self.sidebar_canvas.bind("<Configure>", self._sync_sidebar_width)

        self._build_card_file(self.sidebar_inner)
        self._build_card_selection(self.sidebar_inner)
        self._build_card_params(self.sidebar_inner)
        self._build_card_save(self.sidebar_inner)
        self._build_card_log(self.sidebar_inner)

        # Trackpad/mouse-wheel scrolling works over ordinary controls. Text and
        # list widgets retain their own native scrolling behavior.
        self.root.bind_all("<MouseWheel>", self._on_sidebar_mousewheel, add="+")
        self.root.bind_all("<Button-4>", self._on_sidebar_mousewheel, add="+")
        self.root.bind_all("<Button-5>", self._on_sidebar_mousewheel, add="+")

    def _sync_sidebar_scrollregion(self, _event=None):
        self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))

    def _sync_sidebar_width(self, event):
        self.sidebar_canvas.itemconfigure(self._sidebar_window, width=event.width)

    def _on_sidebar_mousewheel(self, event):
        widget = event.widget
        node = widget
        while node is not None and node is not self.sidebar_inner:
            node = getattr(node, "master", None)
        if node is None or isinstance(widget, (tk.Text, tk.Listbox, ttk.Combobox)):
            return None

        if getattr(event, "num", None) == 4:
            units = -1
        elif getattr(event, "num", None) == 5:
            units = 1
        elif event.delta:
            units = -int(event.delta / 120) if abs(event.delta) >= 120 else (
                -1 if event.delta > 0 else 1
            )
        else:
            return None
        self.sidebar_canvas.yview_scroll(units, "units")
        return "break"

    def _styled_btn(self, parent, text, style, cmd):
        return ttk.Button(
            parent,
            text=text,
            command=cmd,
            style=style,
            cursor="hand2",
        )

    def _card(self, parent, title):
        border = tk.Frame(parent, bg=BORDER_COLOR, padx=1, pady=1)
        card = tk.Frame(border, bg=BG_CARD, padx=12, pady=12)
        card.pack(fill=tk.BOTH, expand=True)
        tk.Label(card, text=title, font=FONT_H2, bg=BG_CARD, fg=FG_TEXT).pack(
            anchor=tk.W, pady=(0, 8)
        )
        return border, card

    def _build_card_file(self, parent):
        border, card = self._card(parent, "1. Data Load & Setup")
        border.pack(fill=tk.X, pady=(0, 10))

        project_dir = os.path.dirname(os.path.abspath(__file__))
        self.target_file_var = tk.StringVar(
            value=os.path.join(project_dir, "Combi2.xlsx")
        )
        self.bg_file_var = tk.StringVar(
            value=os.path.join(project_dir, "Combi16.xlsx")
        )
        self.q_min_var = tk.StringVar(value="1.60")
        self.q_max_var = tk.StringVar(value="3.70")

        def browse(var, title):
            def _do(*args):
                current = var.get().strip()
                filepath = filedialog.askopenfilename(
                    title=title,
                    initialdir=(
                        os.path.dirname(current) if os.path.dirname(current) else None
                    ),
                    filetypes=[
                        ("Excel workbooks", "*.xlsx *.xls"),
                        ("All files", "*.*"),
                    ],
                )
                if filepath:
                    var.set(filepath)

            return _do

        browse_target = browse(self.target_file_var, "Select Target Data File")
        browse_bg = browse(self.bg_file_var, "Select Background Data File")

        for label, var, cb in (
            ("Target file", self.target_file_var, browse_target),
            ("Background", self.bg_file_var, browse_bg),
        ):
            row = tk.Frame(card, bg=BG_CARD)
            row.pack(fill=tk.X, pady=2)
            ttk.Button(
                row,
                text=label,
                width=13,
                command=cb,
                style="Secondary.TButton",
                cursor="hand2",
            ).pack(side=tk.LEFT, padx=(0, 5))
            ent = ttk.Entry(
                row,
                textvariable=var,
                font=FONT_BODY,
                state="readonly",
            )
            ent.pack(side=tk.LEFT, fill=tk.X, expand=True)
            ent.bind("<Button-1>", cb)

            def show_filename(*_args, widget=ent):
                widget.after_idle(widget.xview_moveto, 1.0)

            var.trace_add("write", show_filename)
            show_filename()

        row3 = tk.Frame(card, bg=BG_CARD)
        row3.pack(fill=tk.X, pady=(4, 8))
        tk.Label(
            row3,
            text="Q range (Å⁻¹)",
            width=13,
            bg=BG_CARD,
            fg=FG_TEXT,
            font=FONT_BODY,
            anchor="w",
        ).pack(
            side=tk.LEFT, padx=(0, 5), ipady=2
        )
        ttk.Entry(
            row3,
            textvariable=self.q_min_var,
            font=FONT_BODY,
            width=8,
            justify="center",
        ).pack(side=tk.LEFT)
        tk.Label(
            row3, text="to", font=FONT_BODY, bg=BG_CARD, fg=COLOR_MUTED
        ).pack(side=tk.LEFT, padx=5)
        ttk.Entry(
            row3,
            textvariable=self.q_max_var,
            font=FONT_BODY,
            width=8,
            justify="center",
        ).pack(side=tk.LEFT)

        self.btn_apply = ttk.Button(
            card,
            text="Load / reload data",
            style="Primary.TButton",
            cursor="hand2",
            command=self.load_data,
        )
        self.btn_apply.pack(fill=tk.X, pady=(5, 0))
        for variable in (
            self.target_file_var,
            self.bg_file_var,
            self.q_min_var,
            self.q_max_var,
        ):
            variable.trace_add("write", self._on_data_input_change)

    def _build_card_selection(self, parent):
        border, card = self._card(parent, "2. Scan selection")
        border.pack(fill=tk.X, pady=(0, 10))

        self.view_mode_var = tk.StringVar(value="th")

        rb_frame = tk.Frame(card, bg=BG_CARD)
        rb_frame.pack(fill=tk.X, pady=(0, 2))
        for text, val, padx in (("Sort by th", "th", 0), ("Sort by samz", "samz", 10)):
            ttk.Radiobutton(
                rb_frame,
                text=text,
                variable=self.view_mode_var,
                value=val,
                command=self.update_ui_mode,
                style="Card.TRadiobutton",
                cursor="hand2",
            ).pack(side=tk.LEFT, padx=padx)

        combo_frame = tk.Frame(card, bg=BG_CARD)
        combo_frame.pack(fill=tk.X, pady=2)
        self.primary_label = tk.Label(
            combo_frame,
            text="Select:",
            bg=BG_CARD,
            fg=FG_TEXT,
            font=FONT_BODY,
            width=12,
            anchor="w",
        )
        self.primary_label.pack(side=tk.LEFT)
        self.primary_combo = ttk.Combobox(combo_frame, state="readonly")
        self.primary_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.primary_combo.bind("<<ComboboxSelected>>", self.update_secondary_list)

        header = tk.Frame(card, bg=BG_CARD)
        header.pack(fill=tk.X, pady=(5, 0))

        self.sort_mode_var = tk.StringVar(value="Ascending")
        sort_combo = ttk.Combobox(
            header,
            textvariable=self.sort_mode_var,
            values=["Ascending", "Descending", "Selection Order"],
            state="readonly",
            width=12,
            font=FONT_SMALL,
        )
        sort_combo.pack(side=tk.LEFT, padx=(0, 5))
        # Sorting only changes drawing order -> redraw from cache, no refit
        sort_combo.bind("<<ComboboxSelected>>", self.schedule_render)

        ttk.Button(
            header,
            text="Clear",
            style="Secondary.TButton",
            cursor="hand2",
            command=self._clear_list_action,
        ).pack(side=tk.RIGHT, padx=(2, 0))
        ttk.Button(
            header,
            text="Select all",
            style="Secondary.TButton",
            cursor="hand2",
            command=self._select_all_action,
        ).pack(side=tk.RIGHT)

        self.selection_count_label = tk.Label(
            header,
            text="0 selected",
            bg=BG_CARD,
            fg=COLOR_MUTED,
            font=FONT_SMALL,
        )
        self.selection_count_label.pack(side=tk.LEFT, padx=(4, 0))

        list_inner = tk.Frame(card, bg=BORDER_COLOR, bd=1)
        list_inner.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_inner, orient=tk.VERTICAL)
        self.secondary_listbox = tk.Listbox(
            list_inner,
            selectmode=tk.EXTENDED,
            exportselection=False,
            yscrollcommand=scrollbar.set,
            font=FONT_BODY,
            bg=INPUT_BG,
            fg=INPUT_FG,
            selectbackground=INPUT_SELECTION_BG,
            selectforeground=INPUT_SELECTION_FG,
            disabledforeground=DISABLED_FG,
            height=7,
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
        )
        scrollbar.config(command=self.secondary_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.secondary_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.secondary_listbox.bind("<<ListboxSelect>>", self.track_selection)

    def _build_card_params(self, parent):
        border, card = self._card(parent, "3. Background correction")
        border.pack(fill=tk.X, pady=(0, 10))

        preset_frame = tk.Frame(card, bg=BG_CARD)
        preset_frame.pack(fill=tk.X, pady=(0, 5))
        tk.Label(
            preset_frame,
            text="Preset:",
            bg=BG_CARD,
            fg=FG_TEXT,
            font=FONT_BODY,
            width=7,
            anchor="w",
        ).pack(side=tk.LEFT)
        self.preset_combo = ttk.Combobox(
            preset_frame, state="readonly", values=list(PRESETS.keys()), font=FONT_BODY
        )
        self.preset_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset_change)

        tk.Label(
            card,
            text="Anchor regions — min, max, weight (one per line)",
            font=FONT_SMALL,
            bg=BG_CARD,
            fg=COLOR_MUTED,
        ).pack(anchor=tk.W)
        self.anchor_text = tk.Text(
            card,
            height=4,
            font=FONT_MONO,
            relief=tk.SOLID,
            bd=1,
            bg=INPUT_BG,
            fg=INPUT_FG,
            insertbackground=FG_TEXT,
            selectbackground=INPUT_SELECTION_BG,
            selectforeground=INPUT_SELECTION_FG,
        )
        self.anchor_text.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        tk.Label(
            card,
            text="Peak masks — min, max (one per line)",
            font=FONT_SMALL,
            bg=BG_CARD,
            fg=COLOR_MUTED,
        ).pack(anchor=tk.W)
        self.peak_text = tk.Text(
            card,
            height=3,
            font=FONT_MONO,
            relief=tk.SOLID,
            bd=1,
            bg=INPUT_BG,
            fg=INPUT_FG,
            insertbackground=FG_TEXT,
            selectbackground=INPUT_SELECTION_BG,
            selectforeground=INPUT_SELECTION_FG,
        )
        self.peak_text.pack(fill=tk.BOTH, expand=True)

        tk.Frame(card, bg=BORDER_COLOR, height=1).pack(fill=tk.X, pady=5)

        self.processing_mode_var = tk.StringVar(value="Fitting + arPLS")
        self.arpls_lam_var = tk.StringVar(value="100000")

        proc_frame1 = tk.Frame(card, bg=BG_CARD)
        proc_frame1.pack(fill=tk.X, pady=1)
        tk.Label(
            proc_frame1,
            text="Method:",
            bg=BG_CARD,
            fg=FG_TEXT,
            font=FONT_BODY,
            width=7,
            anchor="w",
        ).pack(side=tk.LEFT)
        proc_combo = ttk.Combobox(
            proc_frame1,
            textvariable=self.processing_mode_var,
            values=["Fitting Only", "arPLS Only", "Fitting + arPLS"],
            state="readonly",
            font=FONT_BODY,
        )
        proc_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        proc_combo.bind("<<ComboboxSelected>>", self._on_processing_mode_change)

        proc_frame2 = tk.Frame(card, bg=BG_CARD)
        proc_frame2.pack(fill=tk.X, pady=1)
        tk.Label(
            proc_frame2,
            text="arPLS λ:",
            bg=BG_CARD,
            fg=FG_TEXT,
            font=FONT_BODY,
            width=7,
            anchor="w",
        ).pack(side=tk.LEFT)
        self.arpls_entry = ttk.Entry(
            proc_frame2,
            textvariable=self.arpls_lam_var,
            font=FONT_BODY,
            width=12,
        )
        self.arpls_entry.pack(side=tk.LEFT)
        self.arpls_entry.bind("<Return>", self.schedule_update)

        self.param_hint_label = tk.Label(
            card,
            text="",
            bg=BG_CARD,
            fg=STATUS_ERROR,
            font=FONT_SMALL,
            justify=tk.LEFT,
            anchor="w",
            wraplength=310,
        )
        self.param_hint_label.pack(fill=tk.X, pady=(5, 0))
        self.anchor_text.bind("<KeyRelease>", self._on_analysis_input_change)
        self.peak_text.bind("<KeyRelease>", self._on_analysis_input_change)
        self.arpls_lam_var.trace_add("write", self._on_analysis_input_change)

        self.preset_combo.current(0)
        self._on_preset_change()
        self._on_processing_mode_change(schedule=False)

    def _build_card_log(self, parent):
        border, card = self._card(parent, "Activity log")
        border.pack(fill=tk.X, pady=(0, 10))
        log_inner = tk.Frame(card, bg=BG_CARD)
        log_inner.pack(fill=tk.BOTH, expand=True)
        x_scroll = ttk.Scrollbar(log_inner, orient=tk.HORIZONTAL)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        y_scroll = ttk.Scrollbar(log_inner, orient=tk.VERTICAL)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text = tk.Text(
            log_inner,
            height=6,
            font=FONT_MONO,
            bg=LOG_BG,
            fg=LOG_FG,
            insertbackground=LOG_FG,
            selectbackground=COLOR_PRIMARY,
            selectforeground="#FFFFFF",
            relief=tk.FLAT,
            bd=0,
            wrap=tk.NONE,
            state=tk.DISABLED,
            xscrollcommand=x_scroll.set,
            yscrollcommand=y_scroll.set,
        )
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        x_scroll.config(command=self.log_text.xview)
        y_scroll.config(command=self.log_text.yview)

    def _build_card_save(self, parent):
        border, card = self._card(parent, "4. Export options")
        border.pack(fill=tk.X, pady=(0, 10))

        self.save_format_var = tk.StringVar(value="excel")
        self.save_mode_var = tk.StringVar(value="merged")
        self.x_axis_var = tk.StringVar(value="q")
        self.wavelength_var = tk.StringVar(value="1.5406")

        opt1 = tk.Frame(card, bg=BG_CARD)
        opt1.pack(fill=tk.X, pady=1)
        tk.Label(
            opt1,
            text="Format:",
            bg=BG_CARD,
            fg=FG_TEXT,
            font=FONT_BODY,
            width=8,
            anchor="w",
        ).pack(side=tk.LEFT)
        for text, val in (("Excel", "excel"), ("XYE", "xye")):
            ttk.Radiobutton(
                opt1,
                text=text,
                variable=self.save_format_var,
                value=val,
                style="Card.TRadiobutton",
                cursor="hand2",
            ).pack(side=tk.LEFT)

        opt2 = tk.Frame(card, bg=BG_CARD)
        opt2.pack(fill=tk.X, pady=1)
        tk.Label(
            opt2,
            text="Mode:",
            bg=BG_CARD,
            fg=FG_TEXT,
            font=FONT_BODY,
            width=8,
            anchor="w",
        ).pack(side=tk.LEFT)
        self.rb_merged = ttk.Radiobutton(
            opt2,
            text="Merged",
            variable=self.save_mode_var,
            value="merged",
            style="Card.TRadiobutton",
            cursor="hand2",
        )
        self.rb_merged.pack(side=tk.LEFT)
        ttk.Radiobutton(
            opt2,
            text="Individual",
            variable=self.save_mode_var,
            value="individual",
            style="Card.TRadiobutton",
            cursor="hand2",
        ).pack(side=tk.LEFT)

        tk.Frame(card, bg=BORDER_COLOR, height=1).pack(fill=tk.X, pady=4)

        opt3 = tk.Frame(card, bg=BG_CARD)
        opt3.pack(fill=tk.X, pady=1)
        tk.Label(
            opt3,
            text="X-axis:",
            bg=BG_CARD,
            fg=FG_TEXT,
            font=FONT_BODY,
            width=8,
            anchor="w",
        ).pack(side=tk.LEFT)
        for text, val in (("Q", "q"), ("2Theta", "2theta")):
            ttk.Radiobutton(
                opt3,
                text=text,
                variable=self.x_axis_var,
                value=val,
                style="Card.TRadiobutton",
                cursor="hand2",
            ).pack(side=tk.LEFT)

        opt4 = tk.Frame(card, bg=BG_CARD)
        opt4.pack(fill=tk.X, pady=1)
        tk.Label(
            opt4,
            text="Wavelength:",
            bg=BG_CARD,
            fg=FG_TEXT,
            font=FONT_BODY,
            width=10,
            anchor="w",
        ).pack(side=tk.LEFT)
        self.entry_wl = ttk.Entry(
            opt4,
            textvariable=self.wavelength_var,
            font=FONT_BODY,
            width=10,
            state=tk.DISABLED,
        )
        self.entry_wl.pack(side=tk.LEFT)
        tk.Label(
            opt4, text=" Å", bg=BG_CARD, fg=FG_TEXT, font=FONT_BODY
        ).pack(side=tk.LEFT)

        self.save_format_var.trace_add("write", self._on_format_change)
        self.x_axis_var.trace_add("write", self._on_xaxis_change)

    # ---------------- right (plot) panel ----------------
    def _build_plot_panel(self):
        pf = self.plot_frame

        status_row = tk.Frame(pf, bg=BG_APP, padx=18)
        status_row.pack(fill=tk.X, pady=(10, 4))
        tk.Label(
            status_row,
            text="Analysis preview",
            bg=BG_APP,
            fg=FG_TEXT,
            font=FONT_H1,
        ).pack(side=tk.LEFT)
        self.status_label = tk.Label(
            status_row,
            text="No data loaded",
            font=FONT_H2,
            bg=STATUS_NEUTRAL_BG,
            fg=COLOR_MUTED,
            padx=10,
            pady=4,
            anchor="center",
        )
        self.status_label.pack(side=tk.RIGHT)
        self.progress = ttk.Progressbar(
            status_row, mode="determinate", length=150, maximum=1, value=0
        )
        self.progress.pack(side=tk.RIGHT, padx=(8, 10))

        display_row = tk.Frame(pf, bg=BG_APP, padx=18, pady=3)
        display_row.pack(fill=tk.X)
        self.cmap_var = tk.StringVar()
        tk.Label(
            display_row, text="Colors", font=FONT_H2, bg=BG_APP, fg=FG_TEXT
        ).pack(side=tk.LEFT)
        cmap_combo = ttk.Combobox(
            display_row,
            textvariable=self.cmap_var,
            values=list(CMAP_OPTIONS.keys()),
            state="readonly",
            width=16,
        )
        cmap_combo.current(0)
        cmap_combo.pack(side=tk.LEFT, padx=(5, 12))
        cmap_combo.bind("<<ComboboxSelected>>", self.schedule_render)

        tk.Label(
            display_row, text="Guide", font=FONT_H2, bg=BG_APP, fg=FG_TEXT
        ).pack(side=tk.LEFT)
        self.legend_mode_var = tk.StringVar(value="Auto")
        legend_combo = ttk.Combobox(
            display_row,
            textvariable=self.legend_mode_var,
            values=["Auto", "Legend", "Colorbar", "None"],
            state="readonly",
            width=9,
        )
        legend_combo.pack(side=tk.LEFT, padx=(5, 12))
        legend_combo.bind("<<ComboboxSelected>>", self.schedule_render)

        overlay_row = tk.Frame(pf, bg=BG_APP, padx=18, pady=3)
        overlay_row.pack(fill=tk.X)
        tk.Label(
            overlay_row,
            text="Overlays",
            font=FONT_H2,
            bg=BG_APP,
            fg=FG_TEXT,
        ).pack(side=tk.LEFT, padx=(0, 6))
        self.show_top_xaxis_var = tk.BooleanVar(value=False)
        self.show_bg_var = tk.BooleanVar(value=True)
        self.show_anchor_var = tk.BooleanVar(value=True)
        self.show_mask_var = tk.BooleanVar(value=True)
        for text, variable in (
            ("Main Q labels", self.show_top_xaxis_var),
            ("Background", self.show_bg_var),
            ("Anchors", self.show_anchor_var),
            ("Masks", self.show_mask_var),
        ):
            self._check(overlay_row, text, variable)

        layout_row = tk.Frame(pf, bg=BG_APP, padx=18, pady=3)
        layout_row.pack(fill=tk.X)
        tk.Label(
            layout_row, text="Offsets", font=FONT_H2, bg=BG_APP, fg=FG_TEXT
        ).pack(side=tk.LEFT)
        self.offset_mode_var = tk.StringVar(value="auto")
        for text, value in (("Auto", "auto"), ("Manual", "manual")):
            ttk.Radiobutton(
                layout_row,
                text=text,
                variable=self.offset_mode_var,
                value=value,
                style="Toolbar.TRadiobutton",
                cursor="hand2",
                command=self._on_offset_mode_change,
            ).pack(side=tk.LEFT, padx=(5, 0))

        self.norm_offset_var = tk.StringVar(value="0.5")
        self.raw_offset_var = tk.StringVar(value="500")
        self.manual_offset_widgets = []
        for label, variable, width in (
            ("Normalized", self.norm_offset_var, 6),
            ("Raw", self.raw_offset_var, 7),
        ):
            label_widget = tk.Label(
                layout_row,
                text=label,
                bg=BG_APP,
                fg=FG_TEXT,
                font=FONT_BODY,
            )
            label_widget.pack(side=tk.LEFT, padx=(10, 3))
            entry = ttk.Entry(layout_row, textvariable=variable, width=width)
            entry.pack(side=tk.LEFT)
            self.manual_offset_widgets.extend((label_widget, entry))
            variable.trace_add("write", self.schedule_render)

        tk.Frame(layout_row, bg=BORDER_COLOR, width=1).pack(
            side=tk.LEFT, fill=tk.Y, padx=12
        )
        self.show_ref_var = tk.BooleanVar(value=True)
        self.show_hkl_var = tk.BooleanVar(value=True)
        self._check(
            layout_row,
            "Reference peaks",
            self.show_ref_var,
            self._on_reference_visibility_change,
        )
        self.hkl_check = self._check(layout_row, "HKL labels", self.show_hkl_var)

        sizing_row = tk.Frame(pf, bg=BG_APP, padx=18)
        sizing_row.pack(fill=tk.X, pady=(3, 7))
        self.width_ratio_var = tk.DoubleVar(value=1.0)
        self.height_ratio_var = tk.DoubleVar(value=2.5)
        for label, variable, start, end in (
            ("Plot width ratio", self.width_ratio_var, 0.3, 3.0),
            ("Peak-panel height", self.height_ratio_var, 0.5, 5.0),
        ):
            tk.Label(
                sizing_row,
                text=label,
                bg=BG_APP,
                fg=FG_TEXT,
                font=FONT_BODY,
            ).pack(side=tk.LEFT, padx=(0 if label.startswith("Plot") else 10, 3))
            scale = tk.Scale(
                sizing_row,
                variable=variable,
                from_=start,
                to=end,
                resolution=0.1,
                orient=tk.HORIZONTAL,
                length=105,
                showvalue=True,
                bg=BG_APP,
                fg=FG_TEXT,
                troughcolor=BORDER_COLOR,
                activebackground=COLOR_PRIMARY,
                highlightthickness=0,
                font=FONT_SMALL,
                command=self.schedule_render,
            )
            scale.pack(side=tk.LEFT)
            if label.startswith("Peak"):
                self.reference_height_scale = scale

        font_row = tk.Frame(pf, bg=BG_APP, padx=18)
        font_row.pack(fill=tk.X, pady=(0, 4))
        tk.Label(
            font_row,
            text="Font sizes",
            bg=BG_APP,
            fg=FG_TEXT,
            font=FONT_H2,
        ).pack(side=tk.LEFT, padx=(0, 4))

        self.title_font_var = tk.IntVar(value=12)
        self.axis_title_font_var = tk.IntVar(value=11)
        self.axis_tick_font_var = tk.IntVar(value=10)
        self.legend_font_var = tk.IntVar(value=9)
        for label, variable in (
            ("Title", self.title_font_var),
            ("Axis", self.axis_title_font_var),
            ("Ticks", self.axis_tick_font_var),
            ("Legend", self.legend_font_var),
        ):
            tk.Label(
                font_row,
                text=label,
                bg=BG_APP,
                fg=FG_TEXT,
                font=FONT_BODY,
            ).pack(side=tk.LEFT, padx=(10, 3))
            ttk.Spinbox(
                font_row,
                from_=6,
                to=30,
                textvariable=variable,
                width=4,
            ).pack(side=tk.LEFT)
            variable.trace_add("write", self.schedule_render)

        tk.Label(
            pf,
            text="Plot markers: left-click to add · right-click near a marker to remove",
            bg=BG_APP,
            fg=COLOR_MUTED,
            font=FONT_SMALL,
            anchor="e",
        ).pack(fill=tk.X, padx=20, pady=(0, 3))

        # --- Canvas + toolbar (packed exactly once each) ---
        self.fig = Figure(figsize=(9, 6), dpi=100)
        self.fig.patch.set_facecolor(BG_APP)
        self.canvas = FigureCanvasTkAgg(self.fig, master=pf)
        self.toolbar = NavigationToolbar2Tk(self.canvas, pf, pack_toolbar=False)
        self.toolbar.config(background=BG_APP)
        for child in self.toolbar.winfo_children():
            try:
                child.configure(
                    background=BG_APP,
                    foreground=FG_TEXT,
                    activebackground=BG_PANEL,
                    activeforeground=FG_TEXT,
                    highlightbackground=BG_APP,
                )
            except tk.TclError:
                pass
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X, padx=10)
        self.canvas.get_tk_widget().pack(
            side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 5)
        )
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self._draw_empty_state()
        self._on_offset_mode_change(render=False)
        self._on_reference_visibility_change(render=False)

    def _check(self, parent, text, var, command=None):
        widget = ttk.Checkbutton(
            parent,
            text=text,
            variable=var,
            style="Toolbar.TCheckbutton",
            cursor="hand2",
            command=command or self.schedule_render,
        )
        widget.pack(side=tk.LEFT, padx=(0, 6))
        return widget

    # =====================================================
    # Small UI callbacks
    # =====================================================
    def log(self, msg):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _clear_log(self):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)

    def set_status(self, text, color=COLOR_SUCCESS):
        backgrounds = {
            COLOR_SUCCESS: STATUS_READY_BG,
            COLOR_WARN: STATUS_BUSY_BG,
            STATUS_INFO: STATUS_INFO_BG,
            STATUS_ERROR: STATUS_ERROR_BG,
            COLOR_MUTED: STATUS_NEUTRAL_BG,
        }
        self.status_label.config(
            text=text,
            fg=color,
            bg=backgrounds.get(color, STATUS_NEUTRAL_BG),
        )

    def _draw_empty_state(
        self,
        message="Load target and background workbooks to begin.",
        title="No data loaded",
    ):
        """Show a useful instruction instead of a blank or stale figure."""
        self.fig.clear()
        self.marker_objects.clear()
        self.ax1 = self.ax2 = self.ax3 = self.ax4 = None
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(BG_CARD)
        ax.axis("off")
        ax.text(
            0.5,
            0.54,
            title,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color=FG_TEXT,
        )
        ax.text(
            0.5,
            0.46,
            message,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color=COLOR_MUTED,
        )
        self.canvas.draw_idle()

    def _sync_action_states(self):
        """Enable actions only when their prerequisites are available."""
        if not all(
            hasattr(self, name)
            for name in ("btn_apply", "btn_run", "btn_export")
        ):
            return
        busy = self._loading or self._computing or self._exporting
        has_data = self.df_target is not None and not self._data_inputs_dirty
        has_selection = bool(self.current_selection_order)
        self.btn_apply.configure(state=tk.DISABLED if busy else tk.NORMAL)
        action_state = tk.NORMAL if has_data and has_selection and not busy else tk.DISABLED
        self.btn_run.configure(state=action_state)
        self.btn_export.configure(state=action_state)

    def _on_data_input_change(self, *_args):
        if self.df_target is None:
            return
        self._data_inputs_dirty = True
        self.set_status("Reload required", STATUS_INFO)
        self._sync_action_states()

    def _retry_analysis(self):
        """Explicit Run action retries profiles that previously failed."""
        self.fit_failed.clear()
        self.schedule_update()

    def _on_analysis_input_change(self, *_args):
        if self.df_target is not None:
            self.set_status("Settings changed · run analysis", STATUS_INFO)

    def _on_processing_mode_change(self, _event=None, schedule=True):
        method = self.processing_mode_var.get()
        fitting_enabled = method != "arPLS Only"
        arpls_enabled = method != "Fitting Only"
        self.preset_combo.configure(
            state="readonly" if fitting_enabled else tk.DISABLED
        )
        for editor in (self.anchor_text, self.peak_text):
            editor.configure(
                state=tk.NORMAL if fitting_enabled else tk.DISABLED,
                fg=INPUT_FG if fitting_enabled else DISABLED_FG,
            )
        self.arpls_entry.configure(state=tk.NORMAL if arpls_enabled else tk.DISABLED)
        if schedule:
            self.schedule_update()

    def _on_offset_mode_change(self, render=True):
        manual = self.offset_mode_var.get() == "manual"
        for widget in self.manual_offset_widgets:
            if isinstance(widget, ttk.Entry):
                widget.configure(state=tk.NORMAL if manual else tk.DISABLED)
            else:
                widget.configure(fg=FG_TEXT if manual else DISABLED_FG)
        if render:
            self.schedule_render()

    def _on_reference_visibility_change(self, render=True):
        visible = self.show_ref_var.get()
        self.hkl_check.configure(state=tk.NORMAL if visible else tk.DISABLED)
        self.reference_height_scale.configure(
            state=tk.NORMAL if visible else tk.DISABLED,
            fg=FG_TEXT if visible else DISABLED_FG,
        )
        if render:
            self.schedule_render()

    def _on_preset_change(self, *args):
        sel = self.preset_combo.get()
        if sel in PRESETS:
            self.anchor_text.delete(1.0, tk.END)
            self.anchor_text.insert(tk.END, PRESETS[sel]["anchors"])
            self.peak_text.delete(1.0, tk.END)
            self.peak_text.insert(tk.END, PRESETS[sel]["masks"])
            self._on_analysis_input_change()

    def _on_format_change(self, *args):
        if self.save_format_var.get() == "xye":
            self.save_mode_var.set("individual")
            self.rb_merged.config(state=tk.DISABLED)
        else:
            self.rb_merged.config(state=tk.NORMAL)

    def _on_xaxis_change(self, *args):
        if self.x_axis_var.get() == "2theta":
            self.entry_wl.config(state=tk.NORMAL)
        else:
            self.entry_wl.config(state=tk.DISABLED)

    def update_ui_mode(self, *args):
        mode = self.view_mode_var.get()
        self.primary_combo.set("")
        if mode == "th":
            self.primary_label.config(text="Select th:")
            self.primary_combo["values"] = self.global_ths
            if self.global_ths:
                self.primary_combo.current(0)
        else:
            self.primary_label.config(text="Select samz:")
            self.primary_combo["values"] = self.global_samzs
            if self.global_samzs:
                self.primary_combo.current(0)
        self.update_secondary_list()

    def update_secondary_list(self, *args):
        self.secondary_listbox.delete(0, tk.END)
        self.current_selection_order.clear()
        self.selection_count_label.config(text="0 selected")
        self._sync_action_states()
        if self.df_target is not None:
            self.set_status("Choose scans", STATUS_INFO)
            self._draw_empty_state(
                "Choose one or more scan positions, then click Run analysis.",
                "Choose scans to compare",
            )
        if not self.primary_combo.get():
            return
        items = self.global_samzs if self.view_mode_var.get() == "th" else self.global_ths
        for s in items:
            self.secondary_listbox.insert(tk.END, s)

    def track_selection(self, event=None):
        sel_items = [
            self.secondary_listbox.get(i) for i in self.secondary_listbox.curselection()
        ]
        for item in sel_items:
            if item not in self.current_selection_order:
                self.current_selection_order.append(item)
        self.current_selection_order = [
            x for x in self.current_selection_order if x in sel_items
        ]
        count = len(self.current_selection_order)
        self.selection_count_label.config(
            text=f"{count} selected" if count != 1 else "1 selected"
        )
        self._sync_action_states()
        if event is not None and self.df_target is not None:
            self.set_status("Selection changed", STATUS_INFO)
            self._draw_empty_state(
                "Review the selection, then click Run analysis.",
                "Selection changed",
            )

    def _clear_list_action(self):
        self.secondary_listbox.selection_clear(0, tk.END)
        self.track_selection(event=True)

    def _select_all_action(self):
        self.secondary_listbox.selection_set(0, tk.END)
        self.track_selection(event=True)

    # =====================================================
    # Worker-thread plumbing
    # =====================================================
    def _post(self, event):
        self._queue.put(event)

    def _poll_queue(self):
        if self._closing:
            return
        try:
            while True:
                try:
                    event = self._queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    self._handle_event(event)
                except Exception as exc:
                    self.log(f"❌ UI event failed: {exc}")
                    self.set_status("UI error", STATUS_ERROR)
        finally:
            if not self._closing:
                self.root.after(60, self._poll_queue)

    def _on_close(self):
        """Prevent an incomplete workbook or half-finished background task."""
        if self._exporting:
            messagebox.showwarning(
                "Export in progress",
                "Data is still being written. Please wait until export finishes "
                "before closing the dashboard.",
                parent=self.root,
            )
            return
        if self._loading or self._computing:
            messagebox.showwarning(
                "Analysis in progress",
                "Data loading or fitting is still running. Please wait until it "
                "finishes before closing the dashboard.",
                parent=self.root,
            )
            return
        self._closing = True
        self.root.destroy()

    def _handle_event(self, event):
        kind = event[0]
        if kind == "log":
            self.log(event[1])
        elif kind == "progress":
            done, total = event[1], event[2]
            self.progress.stop()
            self.progress.configure(mode="determinate", maximum=max(1, total), value=done)
            self.set_status(f"Fitting {done}/{total}…", COLOR_WARN)
        elif kind == "fit_result":
            self.fit_cache[event[1]] = event[2]
            while len(self.fit_cache) > 512:
                self.fit_cache.pop(next(iter(self.fit_cache)))
        elif kind == "fit_failed":
            key, pos, err = event[1], event[2], event[3]
            self.fit_failed[key] = err
            while len(self.fit_failed) > 128:
                self.fit_failed.pop(next(iter(self.fit_failed)))
            self.log(f"❌ Fit failed for {pos}: {err}")
        elif kind == "compute_done":
            self._computing = False
            self._set_busy(False)
            if event[1] == self._data_rev:
                if self._pending_update:
                    self._pending_update = False
                    self._do_update()
                else:
                    self.render()
        elif kind == "data_loaded":
            self._on_data_loaded(event[1])
        elif kind == "data_failed":
            self._loading = False
            self._set_busy(False)
            self.log(f"❌ Load failed: {event[1]}")
            self.set_status("Load failed", STATUS_ERROR)
            self._draw_empty_state(str(event[1]), "Could not load data")
        elif kind == "export_done":
            self._exporting = False
            self._set_busy(False)
            self.log(event[1])
            if str(event[1]).startswith("❌"):
                self.set_status("Export failed", STATUS_ERROR)
            elif str(event[1]).startswith("⚠"):
                self.set_status("Export completed with errors", COLOR_WARN)
            else:
                self.set_status("Export complete", COLOR_SUCCESS)

    def _set_busy(self, busy, msg=None):
        if busy:
            self.set_status(msg or "Working…", COLOR_WARN)
            self.progress.configure(mode="indeterminate", maximum=1, value=0)
            self.progress.start(12)
        else:
            if not (self._computing or self._loading or self._exporting):
                self.progress.stop()
                self.progress.configure(mode="determinate", maximum=1, value=0)
                self.set_status("Ready", COLOR_SUCCESS)
        self._sync_action_states()

    # =====================================================
    # Data loading
    # =====================================================
    def load_data(self):
        if self._loading or self._computing or self._exporting:
            self.log("⚠ Busy — please wait for the current task to finish.")
            return
        try:
            q_min = float(self.q_min_var.get())
            q_max = float(self.q_max_var.get())
        except ValueError:
            self.log("❌ Error: Please enter numbers for the Q range.")
            self.set_status("Invalid Q range", STATUS_ERROR)
            return
        try:
            validate_q_range(q_min, q_max)
            bg_q_min, bg_q_max = expanded_background_q_range(q_min, q_max)
            required_bg_q_min, required_bg_q_max = expanded_background_q_range(
                q_min, q_max, guard=0.0
            )
        except ValueError as exc:
            self.log(f"❌ Error: {exc}")
            self.set_status("Invalid Q range", STATUS_ERROR)
            return

        t_path = self.target_file_var.get().strip()
        b_path = self.bg_file_var.get().strip()
        if not os.path.exists(t_path) or not os.path.exists(b_path):
            self.log("⚠ File not found. Please check the paths.")
            self.set_status("File not found", STATUS_ERROR)
            return

        self._loading = True
        self._set_busy(True, "Loading data…")
        self.log("Loading data… (merging multi-sheets in background)")

        def work():
            caught_warnings = []
            try:
                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always")
                    df_target, q_target, ths, samzs = read_multisheet(
                        t_path, q_min, q_max
                    )
                    df_bg, q_bg, _, _ = read_multisheet(
                        b_path, bg_q_min, bg_q_max
                    )
                    validate_background_q_support(
                        q_bg, required_bg_q_min, required_bg_q_max
                    )
                bg_cols = [c for c in df_bg.columns if c != "Q"]
                # Build all reference interpolators once, off the UI thread
                fitter = BackgroundFitter(df_bg, q_bg, bg_cols, q_target)
                self._post(
                    (
                        "data_loaded",
                        {
                            "df_target": df_target,
                            "df_bg": df_bg,
                            "q_target": q_target,
                            "q_bg": q_bg,
                            "ths": ths,
                            "samzs": samzs,
                            "bg_cols": bg_cols,
                            "fitter": fitter,
                            "q_min": q_min,
                            "q_max": q_max,
                            "bg_q_min": bg_q_min,
                            "bg_q_max": bg_q_max,
                            "target_path": os.path.abspath(t_path),
                            "bg_path": os.path.abspath(b_path),
                        },
                    )
                )
            except Exception as e:
                self._post(("data_failed", str(e)))
            finally:
                for item in caught_warnings:
                    self._post(("log", f"⚠ {item.message}"))

        threading.Thread(target=work, daemon=True).start()

    def _on_data_loaded(self, d):
        self._loading = False
        self.df_target = d["df_target"]
        self.df_bg = d["df_bg"]
        self.q_target = d["q_target"]
        self.q_bg = d["q_bg"]
        self.global_ths = d["ths"]
        self.global_samzs = d["samzs"]
        self.bg_cols = d["bg_cols"]
        self.fitter = d["fitter"]
        self.q_min, self.q_max = d["q_min"], d["q_max"]
        self.bg_q_min, self.bg_q_max = d["bg_q_min"], d["bg_q_max"]
        self.loaded_target_path = d["target_path"]
        self.loaded_bg_path = d["bg_path"]
        self._data_inputs_dirty = False

        # New data invalidates every cached fit
        self._data_rev += 1
        self.fit_cache.clear()
        self.fit_failed.clear()
        self.ref_cache.clear()
        self.clicked_q_values.clear()
        self.marker_objects.clear()

        self.update_ui_mode()
        self._set_busy(False)

        target_cols = [c for c in self.df_target.columns if c != "Q"]
        self.log(
            f"✅ Data merged successfully! (Target: {len(target_cols)} / "
            f"BG Pool: {len(self.bg_cols)}; display Q {self.q_min:g}–{self.q_max:g}, "
            f"BG loaded Q {self.q_bg[0]:g}–{self.q_bg[-1]:g})"
        )
        if self.secondary_listbox.size() > 0:
            self.secondary_listbox.selection_set(0)
            self.track_selection()
            self.schedule_update()
        else:
            self.set_status("No selectable scans", STATUS_ERROR)
            self._draw_empty_state(
                "The workbook loaded, but no scan positions are available.",
                "No selectable scans",
            )

    # =====================================================
    # Update pipeline: schedule -> compute (worker) -> render
    # =====================================================
    def schedule_render(self, *args):
        """Visual-only change: redraw from cache, never refit."""
        if self._render_job is not None:
            self.root.after_cancel(self._render_job)
        self._render_job = self.root.after(150, self._render_debounced)

    def _render_debounced(self):
        self._render_job = None
        if self.df_target is not None:
            self.render()

    def schedule_update(self, *args):
        """Data-affecting change: compute missing fits, then redraw."""
        if self._update_job is not None:
            self.root.after_cancel(self._update_job)
        self._update_job = self.root.after(300, self._update_debounced)

    def _update_debounced(self):
        self._update_job = None
        self._do_update()

    def _snapshot_params(self):
        method = self.processing_mode_var.get()
        if method == "arPLS Only":
            anchors, peaks = [], []
        else:
            try:
                anchors = parse_anchor_lines(self.anchor_text.get(1.0, tk.END))
                peaks = parse_peak_lines(self.peak_text.get(1.0, tk.END))
            except ValueError as exc:
                self.log(f"❌ Error: {exc}")
                self.param_hint_label.config(text=str(exc))
                self.set_status("Invalid processing settings", STATUS_ERROR)
                return None

        if method == "Fitting Only":
            lam = 0.0  # Keep the cache key stable; lambda is unused in this mode.
        else:
            try:
                lam = float(self.arpls_lam_var.get())
            except ValueError:
                self.log("❌ Error: arPLS λ must be a number.")
                self.param_hint_label.config(text="arPLS λ must be a number.")
                self.set_status("Invalid processing settings", STATUS_ERROR)
                return None
            if not np.isfinite(lam) or lam <= 0:
                self.log("❌ Error: arPLS λ must be a finite positive number.")
                self.param_hint_label.config(
                    text="arPLS λ must be a finite positive number."
                )
                self.set_status("Invalid processing settings", STATUS_ERROR)
                return None
        self.param_hint_label.config(text="")
        return {
            "anchors": anchors,
            "peaks": peaks,
            "method": method,
            "lam": lam,
        }

    def _cache_key(self, pos, params):
        return (
            self._data_rev,
            pos,
            params["method"],
            params["lam"],
            tuple(params["anchors"]),
            tuple(params["peaks"]),
        )

    def _resolve_positions(self):
        """Return [(column_name, display_label)] for the current selection,
        sorted per the sort mode."""
        if self.df_target is None or not self.current_selection_order:
            return []
        mode = self.view_mode_var.get()
        primary_val = self.primary_combo.get()

        items = self.current_selection_order.copy()
        sort_mode = self.sort_mode_var.get()
        if sort_mode == "Ascending":
            items.sort(key=numeric_sort_key)
        elif sort_mode == "Descending":
            items.sort(key=numeric_sort_key, reverse=True)

        pairs = []
        for sec_val in items:
            if mode == "th":
                pos = f"th:{primary_val}_samz:{sec_val}"
            else:
                pos = f"th:{sec_val}_samz:{primary_val}"
            if pos in self.df_target.columns:
                pairs.append((pos, sec_val))
        return pairs

    def _do_update(self):
        if self.df_target is None:
            self.log("❌ Load target and background data first.")
            self.set_status("No data loaded", STATUS_ERROR)
            return
        if self._computing:
            self._pending_update = True
            return
        if not self.current_selection_order:
            self.log("❌ Please select at least 1 position to compare from the list.")
            self.set_status("No scans selected", STATUS_ERROR)
            return

        params = self._snapshot_params()
        if params is None:
            return
        pairs = self._resolve_positions()
        if not pairs:
            return

        jobs = []
        for pos, _label in pairs:
            key = self._cache_key(pos, params)
            if key not in self.fit_cache and key not in self.fit_failed:
                jobs.append((key, pos))

        if not jobs:
            self.render()
            return

        self._computing = True
        self._set_busy(True, f"Fitting 0/{len(jobs)}…")
        worker_count = min(MAX_FIT_WORKERS, len(jobs))
        self.log(
            f"Calculating fitting for {len(jobs)} profile(s) with "
            f"{worker_count} parallel worker(s)…"
        )

        gen = self._data_rev
        ctx = {
            "df_target": self.df_target,
            "q_target": self.q_target,
            "fitter": self.fitter,
        }

        def work():
            def completed(done, total, key, pos, result, error):
                if error is None:
                    self._post(("fit_result", key, result))
                else:
                    self._post(("fit_failed", key, pos, str(error)))
                self._post(("progress", done, total))

            try:
                compute_fit_jobs(jobs, params, ctx, on_complete=completed)
            except Exception as exc:
                self._post(("log", f"❌ Fitting worker failed: {exc}"))
            finally:
                self._post(("compute_done", gen))

        threading.Thread(target=work, daemon=True).start()

    # =====================================================
    # Reference peaks (cached per file + Q range)
    # =====================================================
    def _find_ref_file(self):
        """Look for ReferencePeaks.xlsx next to the target data file first,
        then next to this script, then in the current working directory."""
        candidates = []
        t_path = self.loaded_target_path
        if t_path:
            candidates.append(
                os.path.join(os.path.dirname(os.path.abspath(t_path)), REF_FILENAME)
            )
        candidates.append(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), REF_FILENAME)
        )
        candidates.append(os.path.abspath(REF_FILENAME))
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    def _get_ref_data(self):
        path = self._find_ref_file()
        key = (path, self.q_min, self.q_max)
        if key not in self.ref_cache:
            if path is None:
                self.log(
                    f"⚠ {REF_FILENAME} not found (looked in the target data folder, "
                    "the script folder, and the current directory) — "
                    "reference panels will be empty."
                )
                ref_data = {}
            else:
                ref_data, err = load_reference_peaks(path, self.q_min, self.q_max)
                if err:
                    self.log(f"⚠ {err}")
                else:
                    self.log(f"📖 Reference peaks loaded: {path}")
            self.ref_cache[key] = ref_data
        return self.ref_cache[key]

    # =====================================================
    # Rendering (main thread, cache-only — no fitting here)
    # =====================================================
    def render(self):
        """Render the current cached results without running any fitting work."""
        if self.df_target is None:
            return
        params = self._snapshot_params()
        if params is None:
            return
        pairs = self._resolve_positions()
        if not pairs:
            self.set_status("No scans selected", STATUS_ERROR)
            self._draw_empty_state(
                "Select one or more scan positions, then click Run analysis.",
                "No scans selected",
            )
            return

        plot_items = []
        missing = False
        relevant_failures = 0
        for position, label in pairs:
            key = self._cache_key(position, params)
            if key in self.fit_cache:
                plot_items.append((position, label, self.fit_cache[key]))
            elif key in self.fit_failed:
                relevant_failures += 1
            else:
                missing = True
        if missing:
            self.schedule_update()
            return
        if not plot_items:
            self.set_status("Analysis failed", STATUS_ERROR)
            self._draw_empty_state(
                "See the Activity log for the fitting error, then adjust settings "
                "and click Run analysis to retry.",
                "No profile could be processed",
            )
            return

        def float_or_default(variable, default):
            try:
                value = float(variable.get())
                return value if np.isfinite(value) else default
            except (ValueError, tk.TclError):
                return default

        show_reference = self.show_ref_var.get()
        reference_data = self._get_ref_data() if show_reference else {}
        options = PlotOptions(
            q_min=self.q_min,
            q_max=self.q_max,
            cmap_name=CMAP_OPTIONS.get(self.cmap_var.get()),
            legend_mode=self.legend_mode_var.get(),
            show_reference=show_reference,
            show_hkl=self.show_hkl_var.get(),
            show_top_xaxis=self.show_top_xaxis_var.get(),
            show_background=self.show_bg_var.get(),
            show_anchors=self.show_anchor_var.get(),
            show_masks=self.show_mask_var.get(),
            anchors=tuple(params["anchors"]),
            masks=tuple(params["peaks"]),
            offset_mode=self.offset_mode_var.get(),
            normalized_offset=float_or_default(self.norm_offset_var, 0.5),
            raw_offset=float_or_default(self.raw_offset_var, 500.0),
            width_ratio=float(self.width_ratio_var.get()),
            height_ratio=float(self.height_ratio_var.get()),
            title_font_size=self._safe_int(self.title_font_var, 12),
            axis_title_font_size=self._safe_int(self.axis_title_font_var, 11),
            axis_tick_font_size=self._safe_int(self.axis_tick_font_var, 10),
            legend_font_size=self._safe_int(self.legend_font_var, 9),
            colorbar_label=(
                "samz position"
                if self.view_mode_var.get() == "th"
                else "th position"
            ),
        )
        try:
            axes = render_dashboard_figure(
                self.fig,
                self.q_target,
                plot_items,
                reference_data,
                options,
            )
        except Exception as exc:
            self.log(f"❌ Plot rendering failed: {exc}")
            self.set_status("Plot error", STATUS_ERROR)
            return

        padded_axes = tuple(axes) + (None,) * (4 - len(axes))
        self.ax1, self.ax2, self.ax3, self.ax4 = padded_axes[:4]
        self.marker_objects.clear()
        for q_value in self.clicked_q_values:
            self.draw_single_marker(q_value)
        self.canvas.draw_idle()
        self.progress.configure(mode="determinate", maximum=1, value=0)
        if relevant_failures:
            self.set_status(
                f"Complete · {relevant_failures} failed",
                STATUS_ERROR,
            )
        else:
            self.set_status(
                f"Complete · {len(plot_items)} profile"
                f"{'s' if len(plot_items) != 1 else ''}",
                COLOR_SUCCESS,
            )

    @staticmethod
    def _safe_int(var, default):
        try:
            return int(var.get())
        except (ValueError, tk.TclError):
            return default

    # Click markers
    # =====================================================
    def draw_single_marker(self, q_val):
        arts = []
        axes_list = [ax for ax in (self.ax1, self.ax2, self.ax3, self.ax4) if ax is not None]
        for ax in axes_list:
            line = ax.axvline(
                q_val,
                color="#6D28D9",
                linestyle="-",
                linewidth=1.5,
                alpha=0.9,
                zorder=200,
            )
            arts.append(line)
            if ax in (self.ax1, self.ax2):
                y_min, y_max = ax.get_ylim()
                t = ax.text(
                    q_val,
                    y_max - (y_max - y_min) * 0.04,
                    f" {q_val:.3f}",
                    color="#4C1D95",
                    rotation=90,
                    verticalalignment="top",
                    horizontalalignment="right",
                    fontsize=9,
                    fontweight="bold",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "#C4B5FD",
                        "boxstyle": "round,pad=0.18",
                        "alpha": 0.92,
                    },
                    zorder=210,
                )
                arts.append(t)
        self.marker_objects[q_val] = arts

    def on_click(self, event):
        if self.toolbar.mode != "":
            return
        axes_list = [ax for ax in (self.ax1, self.ax2, self.ax3, self.ax4) if ax is not None]
        if event.inaxes not in axes_list or event.xdata is None:
            return
        q_clicked = event.xdata
        secondary_click = _is_secondary_click(event)
        if event.button == 1 and not secondary_click:
            self.clicked_q_values.append(q_clicked)
            self.draw_single_marker(q_clicked)
            self.canvas.draw_idle()
        elif secondary_click:
            if not self.clicked_q_values:
                return
            closest_q = min(self.clicked_q_values, key=lambda q: abs(q - q_clicked))
            if abs(closest_q - q_clicked) < 0.03:
                self.clicked_q_values.remove(closest_q)
                for art in self.marker_objects.pop(closest_q, []):
                    art.remove()
                self.canvas.draw_idle()

    def clear_markers(self):
        for arts in self.marker_objects.values():
            for art in arts:
                art.remove()
        self.marker_objects.clear()
        self.clicked_q_values.clear()
        self.canvas.draw_idle()
        self.log("🗑️ Markers cleared.")

    # =====================================================
    # Export (worker thread; reuses the fit cache)
    # =====================================================
    def export_all(self):
        if self.df_target is None:
            self.log("❌ Please click [Apply Data] first.")
            return
        if not self.current_selection_order:
            self.log("❌ Please select at least 1 position to save.")
            return
        if self._exporting or self._computing:
            self.log("⚠ Busy — please wait for the current task to finish.")
            return

        params = self._snapshot_params()
        if params is None:
            return
        pairs = self._resolve_positions()
        if not pairs:
            return

        fmt = self.save_format_var.get()
        smode = self.save_mode_var.get()
        x_axis = self.x_axis_var.get()

        if x_axis == "2theta":
            try:
                wl = float(self.wavelength_var.get())
            except ValueError:
                self.log("❌ Error: Enter a valid Wavelength.")
                return
            if not np.isfinite(wl) or wl <= 0:
                self.log("❌ Error: Wavelength must be a finite positive number.")
                return
            export_x_array = q_to_2theta(self.q_target, wl)
            export_x_label = "2Theta"
            self.log(f"▶ Saving with X-axis converted to 2Theta. (λ = {wl} Å)")
        else:
            export_x_array = self.q_target
            export_x_label = "Q"
            self.log("▶ Saving with X-axis as Q.")

        target_path = self.loaded_target_path or ""
        base_name = safe_filename_component(
            os.path.splitext(os.path.basename(target_path))[0]
            if target_path
            else "Cleaned_Signals"
        )

        # All dialogs happen on the main thread, before the worker starts
        if fmt == "xye" or (fmt == "excel" and smode == "individual"):
            save_target = filedialog.askdirectory(title="Select folder to save files")
            if not save_target:
                self.log("⚠ Folder selection cancelled.")
                return
        else:
            save_target = filedialog.asksaveasfilename(
                title="Save Merged Excel File",
                initialfile=f"{base_name}_Selected.xlsx",
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            )
            if not save_target:
                self.log("⚠ Save cancelled.")
                return

        # Snapshot of cached fits (worker computes only what is missing)
        cached, keys_by_pos = {}, {}
        for pos, _label in pairs:
            key = self._cache_key(pos, params)
            keys_by_pos[pos] = key
            if key in self.fit_cache:
                cached[pos] = self.fit_cache[key]

        self._exporting = True
        self._set_busy(True, "Exporting…")
        ctx = {
            "df_target": self.df_target,
            "q_target": self.q_target,
            "fitter": self.fitter,
        }
        positions = [pos for pos, _label in pairs]

        def work():
            try:
                fit_results = dict(cached)
                fit_errors = {}
                missing_jobs = [
                    (keys_by_pos[pos], pos) for pos in positions if pos not in cached
                ]
                if missing_jobs:
                    self._post(
                        (
                            "log",
                            f"Computing {len(missing_jobs)} uncached profile(s) with "
                            f"{min(MAX_FIT_WORKERS, len(missing_jobs))} parallel worker(s)…",
                        )
                    )

                    def completed(done, total, key, pos, result, error):
                        if error is None:
                            fit_results[pos] = result
                            self._post(("fit_result", key, result))
                        else:
                            fit_errors[pos] = error
                        self._post(("progress", done, total))

                    compute_fit_jobs(
                        missing_jobs, params, ctx, on_complete=completed
                    )

                def report_export_error(position, error):
                    fitting_error = fit_errors.get(position)
                    if fitting_error is not None:
                        self._post(
                            (
                                "log",
                                f"❌ Fitting failed for '{position}': {fitting_error}",
                            )
                        )
                    else:
                        self._post(
                            (
                                "log",
                                f"❌ Export failed for '{position}': {error}",
                            )
                        )

                summary = write_export(
                    save_target=save_target,
                    fmt=fmt,
                    save_mode=smode,
                    x_label=export_x_label,
                    x_values=export_x_array,
                    positions=positions,
                    fit_results=fit_results,
                    base_name=base_name,
                    on_error=report_export_error,
                )
                if summary.succeeded == 0:
                    raise RuntimeError("No selected profile could be exported.")
                output_description = (
                    f"{summary.succeeded} file"
                    f"{'s' if summary.succeeded != 1 else ''}"
                    if summary.save_mode == "individual"
                    else os.path.basename(save_target)
                )
                if summary.failed:
                    message = (
                        f"⚠ Export complete with errors: {summary.succeeded}/"
                        f"{summary.requested} profiles → {output_description}"
                    )
                else:
                    message = (
                        f"✅ Export complete: {summary.succeeded}/"
                        f"{summary.requested} profiles → {output_description}"
                    )
                self._post(("export_done", message))
            except Exception as e:
                self._post(("export_done", f"❌ Export failed: {e}"))

        threading.Thread(target=work, daemon=True).start()


def main():
    enable_high_dpi()
    root = tk.Tk()
    if not ensure_supported_tk(root):
        return
    app = XRDDashboard(root)
    print("🎉 Dashboard launched successfully!")
    root.mainloop()


if __name__ == "__main__":
    main()
