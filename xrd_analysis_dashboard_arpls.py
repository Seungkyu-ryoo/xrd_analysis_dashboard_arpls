"""Tkinter dashboard for interactive XRD background fitting and signal cleanup."""

# =========================================================
# ★ Tkinter-based All-in-One Dashboard (Ultimate Version)
# =========================================================
from operator import pos

import pandas as pd
import numpy as np
import warnings
import os

# --- GUI & Graph Libraries ---
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

# ★ Import separated fitting module (File name must be fitting.py)
from fitting import run_optimization
from fitting_arpls import run_arpls  # 추가된 부분

warnings.filterwarnings("ignore")

# =========================================================
# 1. Global Variables Initialization
# =========================================================
df_target = None
df_bg = None
q_target = None
q_bg = None
global_ths = []
global_samzs = []
target_cols = []
bg_cols = []
q_min, q_max = 1.6, 3.7

current_selection_order = []
ax1, ax2, ax3, ax4 = None, None, None, None
cached_ref_data = (
    None  # ★ Optimization: Cache for ReferencePeaks to avoid repeated disk reads
)


# =========================================================
# 2. Multi-sheet Excel Data Parsing Function
# =========================================================
def read_multisheet(path, q_min_val, q_max_val):
    xls = pd.ExcelFile(path)
    all_series = []
    ths = set()
    samzs = set()

    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        df.columns = df.columns.astype(str).str.strip()

        q_cols = [c for c in df.columns if c.lower() == "q"]
        if not q_cols:
            continue
        q_col = q_cols[0]

        df = df.set_index(q_col)
        df = df[~df.index.duplicated(keep="first")]

        for col in df.columns:
            if "unnamed" in col.lower() or "nan" in col.lower():
                continue
            col_key = f"th:{sheet}_samz:{col}"
            series = df[col].rename(col_key)
            all_series.append(series)
            samzs.add(col)
        ths.add(sheet)

    master_df = pd.concat(all_series, axis=1)
    master_df = master_df.sort_index().interpolate(method="index").ffill().bfill()

    q_arr = master_df.index.values
    master_df.insert(0, "Q", q_arr)
    master_df.reset_index(drop=True, inplace=True)

    master_df = master_df[(master_df["Q"] >= q_min_val) & (master_df["Q"] <= q_max_val)]
    q_arr_filtered = master_df["Q"].values

    try:
        ths_sorted = sorted(list(ths), key=float)
    except ValueError:
        ths_sorted = sorted(list(ths))

    try:
        samzs_sorted = sorted(list(samzs), key=float)
    except ValueError:
        samzs_sorted = sorted(list(samzs))

    return master_df, q_arr_filtered, ths_sorted, samzs_sorted


def process_data():
    global \
        df_target, \
        df_bg, \
        q_target, \
        q_bg, \
        global_ths, \
        global_samzs, \
        target_cols, \
        bg_cols, \
        q_min, \
        q_max

    try:
        q_min = float(q_min_var.get())
        q_max = float(q_max_var.get())
    except ValueError:
        log("❌ Error: Please enter numbers for the Q range.")
        return

    t_path = target_file_var.get().strip()
    b_path = bg_file_var.get().strip()

    if not os.path.exists(t_path) or not os.path.exists(b_path):
        log("⚠ File not found. Please check the paths.")
        return

    log(f"Loading data... (Merging multi-sheets)")
    root.update()

    try:
        df_target, q_target, global_ths, global_samzs = read_multisheet(
            t_path, q_min, q_max
        )
        df_bg, q_bg, _, _ = read_multisheet(b_path, q_min, q_max)

        target_cols = [c for c in df_target.columns if c != "Q"]
        bg_cols = [c for c in df_bg.columns if c != "Q"]

        update_ui_mode()

        log(
            f"✅ Data merged successfully! (Target: {len(target_cols)} / BG Pool: {len(bg_cols)})"
        )
        if secondary_listbox.size() > 0:
            secondary_listbox.selection_set(0)
            track_selection()
            trigger_update_plot()

    except Exception as e:
        log(f"❌ Load failed: {str(e)}")


# =========================================================
# 3. GUI Layout (Modern Flat Design)
# =========================================================
root = tk.Tk()
root.title("XRD Analysis Dashboard - Pro Layout")
root.geometry("1650x950")

BG_APP, BG_PANEL, BG_CARD, FG_TEXT, BORDER_COLOR = (
    "#FFFFFF",
    "#F4F6F9",
    "#FFFFFF",
    "#2B2D42",
    "#DEE2E6",
)
COLOR_PRIMARY, COLOR_HOVER_P = "#4361EE", "#3A0CA3"
COLOR_WARN, COLOR_HOVER_W = "#FF9F1C", "#E85D04"
COLOR_SUCCESS, COLOR_HOVER_S = "#2EC4B6", "#20A498"

FONT_H1, FONT_H2, FONT_BODY, FONT_MONO = (
    ("Segoe UI", 16, "bold"),
    ("Segoe UI", 10, "bold"),
    ("Segoe UI", 9),
    ("Consolas", 9),
)

style = ttk.Style()
if "clam" in style.theme_names():
    style.theme_use("clam")
style.configure(
    "TCombobox",
    fieldbackground=BG_CARD,
    background=BG_CARD,
    borderwidth=1,
    bordercolor=BORDER_COLOR,
)
style.configure(
    "Vertical.TScrollbar",
    background=BG_PANEL,
    bordercolor=BORDER_COLOR,
    arrowcolor=FG_TEXT,
)
style.configure("TPanedwindow", background=BORDER_COLOR)
style.configure("Sash", sashthickness=5, background=BORDER_COLOR)

root.configure(bg=BG_APP)

main_paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
main_paned.pack(fill=tk.BOTH, expand=True)

control_frame = tk.Frame(main_paned, width=380, bg=BG_PANEL)
main_paned.add(control_frame, weight=0)

title_frame = tk.Frame(control_frame, bg=BG_PANEL)
title_frame.pack(side=tk.TOP, fill=tk.X, pady=(20, 10), padx=20)
tk.Label(
    title_frame, text="XRD Analysis Pro", font=FONT_H1, bg=BG_PANEL, fg=COLOR_PRIMARY
).pack(anchor=tk.W)
tk.Label(
    title_frame,
    text="Advanced Background Fitting & Peak Picker",
    font=FONT_BODY,
    bg=BG_PANEL,
    fg="#6C757D",
).pack(anchor=tk.W)
tk.Frame(control_frame, bg=BORDER_COLOR, height=1).pack(
    side=tk.TOP, fill=tk.X, padx=20, pady=(0, 10)
)

btn_frame = tk.Frame(control_frame, bg=BG_PANEL)
btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=(10, 15))


def create_styled_btn(parent, text, color, hover_color, cmd, height=1):
    btn = tk.Button(
        parent,
        text=text,
        font=FONT_H2,
        bg=color,
        fg="white",
        relief=tk.FLAT,
        bd=0,
        cursor="hand2",
        height=height,
        command=cmd,
    )
    btn.bind("<Enter>", lambda e: btn.config(bg=hover_color))
    btn.bind("<Leave>", lambda e: btn.config(bg=color))
    return btn


btn_run = create_styled_btn(
    btn_frame,
    "▶ Run Fitting & Analysis",
    COLOR_PRIMARY,
    COLOR_HOVER_P,
    lambda: trigger_update_plot(),
    height=2,
)
btn_run.pack(fill=tk.X, pady=(0, 8))

sub_btn_frame = tk.Frame(btn_frame, bg=BG_PANEL)
sub_btn_frame.pack(fill=tk.X)
btn_clear = create_styled_btn(
    sub_btn_frame, "🗑 Clear Markers", COLOR_WARN, COLOR_HOVER_W, lambda: clear_markers()
)
btn_clear.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
btn_export = create_styled_btn(
    sub_btn_frame, "💾 Save Data", COLOR_SUCCESS, COLOR_HOVER_S, lambda: export_all()
)
btn_export.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

left_paned = ttk.PanedWindow(control_frame, orient=tk.VERTICAL)
left_paned.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=15)


def create_card_border(parent, title):
    card_border = tk.Frame(parent, bg=BORDER_COLOR, padx=1, pady=1)
    card = tk.Frame(card_border, bg=BG_CARD, padx=12, pady=12)
    card.pack(fill=tk.BOTH, expand=True)
    tk.Label(card, text=title, font=FONT_H2, bg=BG_CARD, fg=FG_TEXT).pack(
        anchor=tk.W, pady=(0, 8)
    )
    return card_border, card


# --- Card 1 ---
card1_border, file_card = create_card_border(left_paned, "1. Data Load & Setup")
left_paned.add(card1_border, weight=0)

target_file_var = tk.StringVar(value="Combi2.xlsx")
bg_file_var = tk.StringVar(value="Combi16.xlsx")
q_min_var = tk.StringVar(value="1.60")
q_max_var = tk.StringVar(value="3.70")


def browse_target(*args):
    filepath = filedialog.askopenfilename(title="Select Target Data File")
    if filepath:
        target_file_var.set(filepath)


def browse_bg(*args):
    filepath = filedialog.askopenfilename(title="Select Background Data File")
    if filepath:
        bg_file_var.set(filepath)


row1 = tk.Frame(file_card, bg=BG_CARD)
row1.pack(fill=tk.X, pady=2)
btn_t = tk.Button(
    row1,
    text="📂 Target Data",
    width=14,
    command=browse_target,
    bg="#E9ECEF",
    relief=tk.FLAT,
    font=FONT_BODY,
    cursor="hand2",
)
btn_t.pack(side=tk.LEFT, padx=(0, 5))
ent_t = tk.Entry(
    row1,
    textvariable=target_file_var,
    font=FONT_BODY,
    relief=tk.SOLID,
    bd=1,
    state="readonly",
)
ent_t.pack(side=tk.LEFT, fill=tk.X, expand=True)
ent_t.bind("<Button-1>", browse_target)

row2 = tk.Frame(file_card, bg=BG_CARD)
row2.pack(fill=tk.X, pady=4)
btn_b = tk.Button(
    row2,
    text="📂 BG Data",
    width=14,
    command=browse_bg,
    bg="#E9ECEF",
    relief=tk.FLAT,
    font=FONT_BODY,
    cursor="hand2",
)
btn_b.pack(side=tk.LEFT, padx=(0, 5))
ent_b = tk.Entry(
    row2,
    textvariable=bg_file_var,
    font=FONT_BODY,
    relief=tk.SOLID,
    bd=1,
    state="readonly",
)
ent_b.pack(side=tk.LEFT, fill=tk.X, expand=True)
ent_b.bind("<Button-1>", browse_bg)

row3 = tk.Frame(file_card, bg=BG_CARD)
row3.pack(fill=tk.X, pady=(4, 8))
tk.Label(row3, text="🔍 Q Range", width=14, bg="#E9ECEF", font=FONT_BODY).pack(
    side=tk.LEFT, padx=(0, 5), ipady=2
)
tk.Entry(
    row3,
    textvariable=q_min_var,
    font=FONT_BODY,
    relief=tk.SOLID,
    bd=1,
    width=8,
    justify="center",
).pack(side=tk.LEFT)
tk.Label(row3, text="~", font=FONT_BODY, bg=BG_CARD).pack(side=tk.LEFT, padx=5)
tk.Entry(
    row3,
    textvariable=q_max_var,
    font=FONT_BODY,
    relief=tk.SOLID,
    bd=1,
    width=8,
    justify="center",
).pack(side=tk.LEFT)
tk.Button(
    file_card,
    text="Apply Data",
    font=("Segoe UI", 9, "bold"),
    bg="#6C757D",
    fg="white",
    relief=tk.FLAT,
    cursor="hand2",
    command=process_data,
).pack(fill=tk.X, pady=(5, 0))

# --- Card 2: UI Mode and Location Selection ---
card2_border, list_card = create_card_border(left_paned, "2. Data Group & Selection")
left_paned.add(card2_border, weight=2)

view_mode_var = tk.StringVar(value="th")


def update_ui_mode(*args):
    mode = view_mode_var.get()
    primary_combo.set("")
    if mode == "th":
        primary_label.config(text="Select th:")
        primary_combo["values"] = global_ths
        if global_ths:
            primary_combo.current(0)
    else:
        primary_label.config(text="Select samz:")
        primary_combo["values"] = global_samzs
        if global_samzs:
            primary_combo.current(0)
    update_secondary_list()


def update_secondary_list(*args):
    global current_selection_order
    secondary_listbox.delete(0, tk.END)
    current_selection_order.clear()
    mode = view_mode_var.get()
    if not primary_combo.get():
        return
    if mode == "th":
        for s in global_samzs:
            secondary_listbox.insert(tk.END, s)
    else:
        for t in global_ths:
            secondary_listbox.insert(tk.END, t)


ctrl_frame = tk.Frame(list_card, bg=BG_CARD)
ctrl_frame.pack(fill=tk.X, pady=(0, 5))

rb_frame = tk.Frame(ctrl_frame, bg=BG_CARD)
rb_frame.pack(fill=tk.X, pady=(0, 2))
tk.Radiobutton(
    rb_frame,
    text="Sort by th",
    variable=view_mode_var,
    value="th",
    command=update_ui_mode,
    bg=BG_CARD,
    cursor="hand2",
).pack(side=tk.LEFT)
tk.Radiobutton(
    rb_frame,
    text="Sort by samz",
    variable=view_mode_var,
    value="samz",
    command=update_ui_mode,
    bg=BG_CARD,
    cursor="hand2",
).pack(side=tk.LEFT, padx=10)

combo_frame = tk.Frame(ctrl_frame, bg=BG_CARD)
combo_frame.pack(fill=tk.X, pady=2)
primary_label = tk.Label(
    combo_frame, text="Select:", bg=BG_CARD, font=FONT_BODY, width=12, anchor="w"
)
primary_label.pack(side=tk.LEFT)
primary_combo = ttk.Combobox(combo_frame, state="readonly")
primary_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
primary_combo.bind("<<ComboboxSelected>>", update_secondary_list)

# Debounce Update System
_update_job = None


def trigger_update_plot(*args):
    global _update_job
    if _update_job is not None:
        root.after_cancel(_update_job)
    _update_job = root.after(
        400, lambda: update_plot() if df_target is not None else None
    )


list_header_frame = tk.Frame(list_card, bg=BG_CARD)
list_header_frame.pack(fill=tk.X, pady=(5, 0))

sort_mode_var = tk.StringVar(value="Ascending")
sort_combo = ttk.Combobox(
    list_header_frame,
    textvariable=sort_mode_var,
    values=["Ascending", "Descending", "Selection Order"],
    state="readonly",
    width=12,
    font=("Segoe UI", 8),
)
sort_combo.pack(side=tk.LEFT, padx=(0, 5))
sort_combo.bind("<<ComboboxSelected>>", trigger_update_plot)


def track_selection(event=None):
    global current_selection_order
    sel_indices = secondary_listbox.curselection()
    sel_items = [secondary_listbox.get(i) for i in sel_indices]
    for item in sel_items:
        if item not in current_selection_order:
            current_selection_order.append(item)
    current_selection_order = [x for x in current_selection_order if x in sel_items]


def clear_list_action():
    secondary_listbox.selection_clear(0, tk.END)
    track_selection()


def select_all_action():
    secondary_listbox.selection_set(0, tk.END)
    track_selection()


btn_clear_list = tk.Button(
    list_header_frame,
    text="⟲ Clear",
    font=("Segoe UI", 8),
    bg="#E9ECEF",
    relief=tk.FLAT,
    cursor="hand2",
    command=clear_list_action,
)
btn_clear_list.pack(side=tk.RIGHT, padx=(2, 0))
btn_select_all = tk.Button(
    list_header_frame,
    text="☑ Select All",
    font=("Segoe UI", 8),
    bg="#E9ECEF",
    relief=tk.FLAT,
    cursor="hand2",
    command=select_all_action,
)
btn_select_all.pack(side=tk.RIGHT)

list_inner = tk.Frame(list_card, bg=BORDER_COLOR, bd=1)
list_inner.pack(fill=tk.BOTH, expand=True)
scrollbar = ttk.Scrollbar(list_inner, orient=tk.VERTICAL)
secondary_listbox = tk.Listbox(
    list_inner,
    selectmode=tk.EXTENDED,
    exportselection=False,
    yscrollcommand=scrollbar.set,
    font=FONT_BODY,
    height=4,
    relief=tk.FLAT,
    bd=0,
    highlightthickness=0,
)
scrollbar.config(command=secondary_listbox.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
secondary_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
secondary_listbox.bind("<<ListboxSelect>>", track_selection)

# --- Card 3: Anchor & Masking Setup ---
card3_border, param_card = create_card_border(left_paned, "3. Anchor & Masking Setup")
left_paned.add(card3_border, weight=2)

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

preset_frame = tk.Frame(param_card, bg=BG_CARD)
preset_frame.pack(fill=tk.X, pady=(0, 5))
tk.Label(
    preset_frame, text="Preset:", bg=BG_CARD, font=FONT_BODY, width=6, anchor="w"
).pack(side=tk.LEFT)
preset_combo = ttk.Combobox(
    preset_frame, state="readonly", values=list(PRESETS.keys()), font=FONT_BODY
)
preset_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)


def on_preset_change(*args):
    sel = preset_combo.get()
    if sel in PRESETS:
        anchor_text.delete(1.0, tk.END)
        anchor_text.insert(tk.END, PRESETS[sel]["anchors"])
        peak_text.delete(1.0, tk.END)
        peak_text.insert(tk.END, PRESETS[sel]["masks"])


preset_combo.bind("<<ComboboxSelected>>", on_preset_change)

tk.Label(
    param_card,
    text="Anchors (Min, Max, Weight)",
    font=("Segoe UI", 8),
    bg=BG_CARD,
    fg="#6C757D",
).pack(anchor=tk.W)
anchor_text = tk.Text(param_card, height=4, font=FONT_MONO, relief=tk.SOLID, bd=1)
anchor_text.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

tk.Label(
    param_card,
    text="Peak Masks (Min, Max)",
    font=("Segoe UI", 8),
    bg=BG_CARD,
    fg="#6C757D",
).pack(anchor=tk.W)
peak_text = tk.Text(param_card, height=3, font=FONT_MONO, relief=tk.SOLID, bd=1)
peak_text.pack(fill=tk.BOTH, expand=True)

tk.Frame(param_card, bg=BORDER_COLOR, height=1).pack(fill=tk.X, pady=5)  # 구분선

processing_mode_var = tk.StringVar(value="Fitting + arPLS")
arpls_lam_var = tk.StringVar(value="100000")

proc_frame1 = tk.Frame(param_card, bg=BG_CARD)
proc_frame1.pack(fill=tk.X, pady=1)
tk.Label(
    proc_frame1, text="Method:", bg=BG_CARD, font=FONT_BODY, width=7, anchor="w"
).pack(side=tk.LEFT)
proc_combo = ttk.Combobox(
    proc_frame1,
    textvariable=processing_mode_var,
    values=["Fitting Only", "arPLS Only", "Fitting + arPLS"],
    state="readonly",
    font=FONT_BODY,
)
proc_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
proc_combo.bind("<<ComboboxSelected>>", trigger_update_plot)

proc_frame2 = tk.Frame(param_card, bg=BG_CARD)
proc_frame2.pack(fill=tk.X, pady=1)
tk.Label(
    proc_frame2, text="arPLS λ:", bg=BG_CARD, font=FONT_BODY, width=7, anchor="w"
).pack(side=tk.LEFT)
ent_lam = tk.Entry(
    proc_frame2,
    textvariable=arpls_lam_var,
    font=FONT_BODY,
    relief=tk.SOLID,
    bd=1,
    width=12,
)
ent_lam.pack(side=tk.LEFT)
ent_lam.bind("<Return>", trigger_update_plot)

preset_combo.current(0)
on_preset_change()

# --- Card 4: Console Log ---
card4_border, log_card = create_card_border(left_paned, "Console Log")
left_paned.add(card4_border, weight=1)
log_inner = tk.Frame(log_card)
log_inner.pack(fill=tk.BOTH, expand=True)
x_scrollbar = ttk.Scrollbar(log_inner, orient=tk.HORIZONTAL)
x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
y_scrollbar = ttk.Scrollbar(log_inner, orient=tk.VERTICAL)
y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
log_text = tk.Text(
    log_inner,
    height=5,
    font=FONT_MONO,
    bg="#1E1E1E",
    fg="#00FF00",
    relief=tk.FLAT,
    bd=0,
    wrap=tk.NONE,
    xscrollcommand=x_scrollbar.set,
    yscrollcommand=y_scrollbar.set,
)
log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
x_scrollbar.config(command=log_text.xview)
y_scrollbar.config(command=log_text.yview)


def log(msg):
    log_text.config(state=tk.NORMAL)
    log_text.insert(tk.END, msg + "\n")
    log_text.see(tk.END)
    log_text.config(state=tk.DISABLED)
    root.update()


# --- Card 5: Save Options ---
card5_border, save_opt_card = create_card_border(left_paned, "4. Save Options")
left_paned.add(card5_border, weight=0)

save_format_var = tk.StringVar(value="excel")
save_mode_var = tk.StringVar(value="merged")
x_axis_var = tk.StringVar(value="q")
wavelength_var = tk.StringVar(value="1.5406")


def _on_format_change(*args):
    if save_format_var.get() == "xye":
        save_mode_var.set("individual")
        rb_merged.config(state=tk.DISABLED)
    else:
        rb_merged.config(state=tk.NORMAL)


def _on_xaxis_change(*args):
    if x_axis_var.get() == "2theta":
        entry_wl.config(state=tk.NORMAL)
    else:
        entry_wl.config(state=tk.DISABLED)


save_format_var.trace("w", _on_format_change)
x_axis_var.trace("w", _on_xaxis_change)

opt_frame1 = tk.Frame(save_opt_card, bg=BG_CARD)
opt_frame1.pack(fill=tk.X, pady=1)
tk.Label(
    opt_frame1, text="Format:", bg=BG_CARD, font=FONT_BODY, width=8, anchor="w"
).pack(side=tk.LEFT)
tk.Radiobutton(
    opt_frame1,
    text="Excel",
    variable=save_format_var,
    value="excel",
    bg=BG_CARD,
    cursor="hand2",
).pack(side=tk.LEFT)
tk.Radiobutton(
    opt_frame1,
    text="XYE",
    variable=save_format_var,
    value="xye",
    bg=BG_CARD,
    cursor="hand2",
).pack(side=tk.LEFT)

opt_frame2 = tk.Frame(save_opt_card, bg=BG_CARD)
opt_frame2.pack(fill=tk.X, pady=1)
tk.Label(
    opt_frame2, text="Mode:", bg=BG_CARD, font=FONT_BODY, width=8, anchor="w"
).pack(side=tk.LEFT)
rb_merged = tk.Radiobutton(
    opt_frame2,
    text="Merged",
    variable=save_mode_var,
    value="merged",
    bg=BG_CARD,
    cursor="hand2",
)
rb_merged.pack(side=tk.LEFT)
tk.Radiobutton(
    opt_frame2,
    text="Individual",
    variable=save_mode_var,
    value="individual",
    bg=BG_CARD,
    cursor="hand2",
).pack(side=tk.LEFT)

tk.Frame(save_opt_card, bg=BORDER_COLOR, height=1).pack(fill=tk.X, pady=4)

opt_frame3 = tk.Frame(save_opt_card, bg=BG_CARD)
opt_frame3.pack(fill=tk.X, pady=1)
tk.Label(
    opt_frame3, text="X-axis:", bg=BG_CARD, font=FONT_BODY, width=8, anchor="w"
).pack(side=tk.LEFT)
tk.Radiobutton(
    opt_frame3, text="Q", variable=x_axis_var, value="q", bg=BG_CARD, cursor="hand2"
).pack(side=tk.LEFT)
tk.Radiobutton(
    opt_frame3,
    text="2Theta",
    variable=x_axis_var,
    value="2theta",
    bg=BG_CARD,
    cursor="hand2",
).pack(side=tk.LEFT)

opt_frame4 = tk.Frame(save_opt_card, bg=BG_CARD)
opt_frame4.pack(fill=tk.X, pady=1)
tk.Label(
    opt_frame4, text="Wave(λ):", bg=BG_CARD, font=FONT_BODY, width=8, anchor="w"
).pack(side=tk.LEFT)
entry_wl = tk.Entry(
    opt_frame4,
    textvariable=wavelength_var,
    font=FONT_BODY,
    relief=tk.SOLID,
    bd=1,
    width=10,
    state=tk.DISABLED,
)
entry_wl.pack(side=tk.LEFT)
tk.Label(opt_frame4, text=" Å", bg=BG_CARD, font=FONT_BODY).pack(side=tk.LEFT)

# --- Right Graph Panel ---
plot_frame = tk.Frame(main_paned, bg=BG_APP)
main_paned.add(plot_frame, weight=3)

# =========================================================
# ★ Top Bar (UI Control Panel)
# =========================================================
top_bar = tk.Frame(plot_frame, bg=BG_APP, pady=10, padx=20)
top_bar.pack(fill=tk.X)

cmap_var = tk.StringVar()
tk.Label(top_bar, text="Theme:", font=FONT_H2, bg=BG_APP, fg=FG_TEXT).pack(side=tk.LEFT)
cmap_combo = ttk.Combobox(
    top_bar,
    textvariable=cmap_var,
    values=[
        "Distinct (tab20b)",
        "Rainbow (gist_rainbow)",
        "Jet (Temperature)",
        "Grayscale",
    ],
    state="readonly",
    font=FONT_BODY,
    width=12,
)
cmap_combo.current(0)
cmap_combo.pack(side=tk.LEFT, padx=(5, 5))

tk.Label(top_bar, text="Legend:", font=FONT_BODY, bg=BG_APP, fg=FG_TEXT).pack(
    side=tk.LEFT, padx=(5, 2)
)
legend_mode_var = tk.StringVar(value="Auto")
legend_combo = ttk.Combobox(
    top_bar,
    textvariable=legend_mode_var,
    values=["Auto", "Legend", "Colorbar", "None"],
    state="readonly",
    font=FONT_BODY,
    width=8,
)
legend_combo.current(0)
legend_combo.pack(side=tk.LEFT, padx=(0, 5))
legend_combo.bind("<<ComboboxSelected>>", trigger_update_plot)

tk.Label(top_bar, text="|", bg=BG_APP, fg="#DEE2E6").pack(side=tk.LEFT, padx=5)

show_top_xaxis_var = tk.BooleanVar(value=False)
tk.Checkbutton(
    top_bar,
    text="Top X-axis",
    variable=show_top_xaxis_var,
    bg=BG_APP,
    fg=FG_TEXT,
    selectcolor=BG_APP,
    activebackground=BG_APP,
    cursor="hand2",
    font=FONT_BODY,
    command=trigger_update_plot,
).pack(side=tk.LEFT, padx=(0, 5))
show_bg_var = tk.BooleanVar(value=True)
tk.Checkbutton(
    top_bar,
    text="BG Line",
    variable=show_bg_var,
    bg=BG_APP,
    fg=FG_TEXT,
    selectcolor=BG_APP,
    activebackground=BG_APP,
    cursor="hand2",
    font=FONT_BODY,
    command=trigger_update_plot,
).pack(side=tk.LEFT, padx=(0, 5))
show_anchor_var = tk.BooleanVar(value=True)
tk.Checkbutton(
    top_bar,
    text="Anchor",
    variable=show_anchor_var,
    bg=BG_APP,
    fg=FG_TEXT,
    selectcolor=BG_APP,
    activebackground=BG_APP,
    cursor="hand2",
    font=FONT_BODY,
    command=trigger_update_plot,
).pack(side=tk.LEFT, padx=(0, 5))
show_mask_var = tk.BooleanVar(value=True)
tk.Checkbutton(
    top_bar,
    text="Mask",
    variable=show_mask_var,
    bg=BG_APP,
    fg=FG_TEXT,
    selectcolor=BG_APP,
    activebackground=BG_APP,
    cursor="hand2",
    font=FONT_BODY,
    command=trigger_update_plot,
).pack(side=tk.LEFT, padx=(0, 5))

tk.Label(top_bar, text="|", bg=BG_APP, fg="#DEE2E6").pack(side=tk.LEFT, padx=5)

tk.Label(top_bar, text="Offset:", font=FONT_BODY, bg=BG_APP, fg=FG_TEXT).pack(
    side=tk.LEFT
)
offset_mode_var = tk.StringVar(value="auto")
tk.Radiobutton(
    top_bar,
    text="Auto",
    variable=offset_mode_var,
    value="auto",
    bg=BG_APP,
    cursor="hand2",
    command=trigger_update_plot,
).pack(side=tk.LEFT)
tk.Radiobutton(
    top_bar,
    text="Manual",
    variable=offset_mode_var,
    value="manual",
    bg=BG_APP,
    cursor="hand2",
    command=trigger_update_plot,
).pack(side=tk.LEFT)

tk.Label(top_bar, text="Norm:", font=FONT_BODY, bg=BG_APP, fg=FG_TEXT).pack(
    side=tk.LEFT, padx=(5, 0)
)
norm_offset_var = tk.StringVar(value="0.5")
ent_norm_off = tk.Entry(top_bar, textvariable=norm_offset_var, width=4, font=FONT_BODY)
ent_norm_off.pack(side=tk.LEFT)
ent_norm_off.bind("<Return>", trigger_update_plot)

tk.Label(top_bar, text="Raw:", font=FONT_BODY, bg=BG_APP, fg=FG_TEXT).pack(
    side=tk.LEFT, padx=(5, 0)
)
raw_offset_var = tk.StringVar(value="500")
ent_raw_off = tk.Entry(top_bar, textvariable=raw_offset_var, width=5, font=FONT_BODY)
ent_raw_off.pack(side=tk.LEFT)
ent_raw_off.bind("<Return>", trigger_update_plot)


# =========================================================
# ★ Layout Bar (Ref Peaks, Labels, Ratios)
# =========================================================
layout_bar = tk.Frame(plot_frame, bg=BG_APP, pady=0, padx=20)
layout_bar.pack(fill=tk.X, pady=(0, 10))

show_ref_var = tk.BooleanVar(value=True)
tk.Checkbutton(
    layout_bar,
    text="Show Ref Peaks",
    variable=show_ref_var,
    bg=BG_APP,
    fg=FG_TEXT,
    selectcolor=BG_APP,
    activebackground=BG_APP,
    cursor="hand2",
    font=FONT_BODY,
    command=trigger_update_plot,
).pack(side=tk.LEFT, padx=(0, 5))

show_hkl_var = tk.BooleanVar(value=True)
tk.Checkbutton(
    layout_bar,
    text="Show HKL Labels",
    variable=show_hkl_var,
    bg=BG_APP,
    fg=FG_TEXT,
    selectcolor=BG_APP,
    activebackground=BG_APP,
    cursor="hand2",
    font=FONT_BODY,
    command=trigger_update_plot,
).pack(side=tk.LEFT, padx=(0, 10))

tk.Label(layout_bar, text="|", bg=BG_APP, fg="#DEE2E6").pack(side=tk.LEFT, padx=5)

tk.Label(layout_bar, text="Width Ratio (Eval:Sub):", bg=BG_APP, font=FONT_BODY).pack(
    side=tk.LEFT, padx=(5, 2)
)
width_ratio_var = tk.DoubleVar(value=1.0)
tk.Scale(
    layout_bar,
    variable=width_ratio_var,
    from_=0.3,
    to=3.0,
    resolution=0.1,
    orient=tk.HORIZONTAL,
    length=120,
    showvalue=False,
    bg=BG_APP,
    highlightthickness=0,
    command=trigger_update_plot,
).pack(side=tk.LEFT, padx=5)

tk.Label(layout_bar, text="Height Ratio (Main:Ref):", bg=BG_APP, font=FONT_BODY).pack(
    side=tk.LEFT, padx=(15, 2)
)
height_ratio_var = tk.DoubleVar(value=2.5)
tk.Scale(
    layout_bar,
    variable=height_ratio_var,
    from_=0.5,
    to=5.0,
    resolution=0.1,
    orient=tk.HORIZONTAL,
    length=120,
    showvalue=False,
    bg=BG_APP,
    highlightthickness=0,
    command=trigger_update_plot,
).pack(side=tk.LEFT, padx=5)

# =========================================================
# ★ Font Size Control Bar (새로 추가됨)
# =========================================================
font_bar = tk.Frame(plot_frame, bg=BG_APP, pady=0, padx=20)
font_bar.pack(fill=tk.X, pady=(0, 10))

title_font_var = tk.IntVar(value=12)
axis_title_font_var = tk.IntVar(value=11)
axis_tick_font_var = tk.IntVar(value=10)
legend_font_var = tk.IntVar(value=8)


def create_font_control(parent, label_text, var):
    tk.Label(parent, text=label_text, bg=BG_APP, font=FONT_BODY).pack(
        side=tk.LEFT, padx=(15 if parent.winfo_children() else 0, 2)
    )
    spin = tk.Spinbox(
        parent,
        from_=6,
        to=30,
        textvariable=var,
        width=3,
        font=FONT_BODY,
        command=trigger_update_plot,
    )
    spin.pack(side=tk.LEFT)
    spin.bind("<Return>", trigger_update_plot)
    # ★ 숫자가 변경(타이핑)될 때마다 즉각적으로 업데이트 트리거 (실시간 반영)
    var.trace_add("write", lambda *args: trigger_update_plot())


create_font_control(font_bar, "Title Size:", title_font_var)
create_font_control(font_bar, "Axis Title:", axis_title_font_var)
create_font_control(font_bar, "Axis Tick:", axis_tick_font_var)
create_font_control(font_bar, "Legend Size:", legend_font_var)

# Canvas Settings
fig = Figure(figsize=(11, 7), dpi=100)
fig.patch.set_facecolor(BG_APP)
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().pack(
    side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10)
)
toolbar = NavigationToolbar2Tk(canvas, plot_frame)
toolbar.config(background=BG_APP)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

clicked_q_values = []
marker_objects = {}


def draw_single_marker(q_val):
    arts = []
    axes_list = [ax for ax in [ax1, ax2, ax3, ax4] if ax is not None]
    for ax in axes_list:
        l = ax.axvline(
            q_val, color="magenta", linestyle="-", linewidth=1.5, alpha=0.8, zorder=200
        )
        arts.append(l)
        if ax in [ax1, ax2]:
            y_min, y_max = ax.get_ylim()
            t = ax.text(
                q_val,
                y_max * 0.96,
                f" {q_val:.3f}",
                color="magenta",
                rotation=90,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=9,
                fontweight="bold",
                zorder=210,
            )
            arts.append(t)
    marker_objects[q_val] = arts


def on_click(event):
    if toolbar.mode != "":
        return
    axes_list = [ax for ax in [ax1, ax2, ax3, ax4] if ax is not None]
    if event.inaxes not in axes_list:
        return
    q_clicked = event.xdata
    if event.button == 1:
        clicked_q_values.append(q_clicked)
        draw_single_marker(q_clicked)
        canvas.draw_idle()
    elif event.button == 3:
        if not clicked_q_values:
            return
        closest_q = min(clicked_q_values, key=lambda q: abs(q - q_clicked))
        if abs(closest_q - q_clicked) < 0.03:
            clicked_q_values.remove(closest_q)
            if closest_q in marker_objects:
                for art in marker_objects[closest_q]:
                    art.remove()
                del marker_objects[closest_q]
            canvas.draw_idle()


canvas.mpl_connect("button_press_event", on_click)


def parse_anchors():
    return [
        (float(p.split(",")[0]), float(p.split(",")[1]), float(p.split(",")[2]))
        for p in anchor_text.get(1.0, tk.END).strip().split("\n")
        if len(p.split(",")) == 3
    ]


def parse_peaks():
    return [
        (float(p.split(",")[0]), float(p.split(",")[1]))
        for p in peak_text.get(1.0, tk.END).strip().split("\n")
        if len(p.split(",")) == 2
    ]


def get_selected_cmap():
    cmap_str = cmap_var.get()
    if "Rainbow" in cmap_str:
        return cm.get_cmap("gist_rainbow")
    elif "Jet" in cmap_str:
        return cm.get_cmap("jet")
    elif "Grayscale" in cmap_str:
        return cm.get_cmap("gray")
    return cm.get_cmap("tab20b")


def draw_reference_peaks_separate(ax, ref_data, show_hkl):
    if not ref_data or ax is None:
        return
    phases, ref_colors = (
        ["m-HfO2", "t-ZrO2", "o-HfO2", "HfTiO2", "TiN"],
        ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4"],
    )
    yticks, yticklabels = [], []
    ax.set_ylim(0, len(phases) * 2)

    for i, phase in enumerate(phases):
        y_base = i * 2
        bg_color = "#f8f9fa" if i % 2 != 0 else "#ffffff"
        ax.axhspan(y_base, y_base + 2, facecolor=bg_color, edgecolor="none", zorder=0)
        ax.axhline(y_base, color="#E9ECEF", linewidth=1, zorder=1)
        yticks.append(y_base + 1.0)
        yticklabels.append(phase)

        if phase not in ref_data or ref_data[phase].empty:
            continue
        valid_ref = ref_data[phase]
        color = ref_colors[i]

        max_stem_height = 0.9
        y_tops = valid_ref["Intensity_Norm"] * max_stem_height + y_base
        ax.vlines(
            valid_ref["Q"],
            ymin=y_base,
            ymax=y_tops,
            color=color,
            linewidth=2.0,
            zorder=10,
        )

        placed_rects = []
        text_w_q, text_h_y = 0.03, 0.45
        max_y_boundary = y_base + 2.0 - 0.05

        for _, row in valid_ref.iterrows():
            q_val = row["Q"]
            stem_height = row["Intensity_Norm"] * max_stem_height
            peak_y_top = y_base + stem_height
            best_x, best_y = q_val, peak_y_top + 0.05

            candidate_offsets = [
                (0.05, 0.0),
                (0.05, 0.018),
                (0.05, -0.018),
                (0.05, 0.035),
                (0.05, -0.035),
                (0.25, 0.0),
                (0.25, 0.02),
                (0.25, -0.02),
                (0.45, 0.0),
                (0.45, 0.02),
                (0.45, -0.02),
            ]
            found_place = False
            for dy, dx in candidate_offsets:
                cand_x, cand_y = q_val + dx, peak_y_top + dy
                if cand_x < q_min + 0.02 or cand_x > q_max - 0.02:
                    continue
                if cand_y + text_h_y > max_y_boundary:
                    continue
                cand_rect = (
                    cand_x - text_w_q / 2,
                    cand_x + text_w_q / 2,
                    cand_y,
                    cand_y + text_h_y,
                )
                overlap = False
                for pr in placed_rects:
                    if (
                        cand_rect[0] < pr[1]
                        and cand_rect[1] > pr[0]
                        and cand_rect[2] < pr[3]
                        and cand_rect[3] > pr[2]
                    ):
                        overlap = True
                        break
                if not overlap:
                    best_x, best_y = cand_x, cand_y
                    placed_rects.append(cand_rect)
                    found_place = True
                    break

            if not found_place:
                best_y = min(best_y, max_y_boundary - text_h_y)
                placed_rects.append(
                    (
                        best_x - text_w_q / 2,
                        best_x + text_w_q / 2,
                        best_y,
                        best_y + text_h_y,
                    )
                )

            if show_hkl:
                ax.text(
                    best_x,
                    best_y,
                    row["hkl"],
                    color=color,
                    fontsize=8.5,
                    rotation=90,
                    verticalalignment="bottom",
                    horizontalalignment="center",
                    zorder=20,
                )
                if abs(best_x - q_val) > 0.01 or (best_y - peak_y_top) > 0.15:
                    ax.plot(
                        [q_val, best_x],
                        [peak_y_top + 0.02, best_y - 0.01],
                        color=color,
                        linestyle="-",
                        linewidth=0.6,
                        alpha=0.5,
                        zorder=5,
                    )

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=10, fontweight="bold", color=FG_TEXT)
    ax.set_xlim(q_min, q_max)


def execute_pipeline(pos, current_anchors, current_peaks):
    mode = processing_mode_var.get()
    try:
        lam_val = float(arpls_lam_var.get())
    except ValueError:
        lam_val = 1e5

    tin_raw = df_target[pos].values
    tin_max = np.max(tin_raw)

    if mode == "Fitting Only":
        return run_optimization(
            pos,
            current_anchors,
            current_peaks,
            df_target,
            df_bg,
            q_target,
            q_bg,
            bg_cols,
        )

    elif mode == "arPLS Only":
        arpls_res = run_arpls(q_target, tin_raw, lam=lam_val)
        return {
            "target_norm": tin_raw / tin_max,
            "best_bg_fit": arpls_res["bkg_arpls"] / tin_max,
            "clean_target_raw": arpls_res["cleaned_y"],
            "best_ref_name": "arPLS Only",
            "best_rigid_params": [0, 0, 0, 0],
            "best_flex_params": [0, 0, 0, 0, 0, 0],
        }

    elif mode == "Fitting + arPLS":
        fit_data = run_optimization(
            pos,
            current_anchors,
            current_peaks,
            df_target,
            df_bg,
            q_target,
            q_bg,
            bg_cols,
        )
        arpls_res = run_arpls(q_target, fit_data["clean_target_raw"], lam=lam_val)
        total_bg_raw = (fit_data["best_bg_fit"] * tin_max) + arpls_res["bkg_arpls"]

        fit_data["clean_target_raw"] = arpls_res["cleaned_y"]
        fit_data["best_bg_fit"] = total_bg_raw / tin_max
        fit_data["best_ref_name"] = str(fit_data["best_ref_name"]) + " + arPLS"
        return fit_data


def update_plot(*args):
    global ax1, ax2, ax3, ax4, marker_objects, cached_ref_data

    if df_target is None or df_bg is None:
        log("❌ Please click [Apply Data] first.")
        return

    if not current_selection_order:
        log("❌ Please select at least 1 position to compare from the list.")
        return

    mode = view_mode_var.get()
    primary_val = primary_combo.get()

    items_to_plot = current_selection_order.copy()
    sort_mode = sort_mode_var.get()

    def try_float(val):
        try:
            return float(val)
        except:
            return str(val)

    if sort_mode == "Ascending":
        items_to_plot.sort(key=try_float)
    elif sort_mode == "Descending":
        items_to_plot.sort(key=try_float, reverse=True)

    selected_positions = []
    for sec_val in items_to_plot:
        if mode == "th":
            pos = f"th:{primary_val}_samz:{sec_val}"
        else:
            pos = f"th:{sec_val}_samz:{primary_val}"
        if pos in df_target.columns:
            selected_positions.append(pos)

    if not selected_positions:
        return

    log_text.config(state=tk.NORMAL)
    log_text.delete(1.0, tk.END)
    log_text.config(state=tk.DISABLED)

    try:
        current_anchors, current_peaks = parse_anchors(), parse_peaks()
    except Exception:
        return log("❌ Error: Incorrect anchor/peak format.")

    # ★ Optimization 1: Cache Reference Data
    if cached_ref_data is None:
        ref_filename = "ReferencePeaks.xlsx"
        ref_data = {}
        if os.path.exists(ref_filename):
            try:
                df_raw = pd.read_excel(ref_filename)
                num_phases = len(df_raw.columns) // 5
                for i in range(num_phases):
                    q_col, h_col, k_col, l_col, int_col = df_raw.columns[
                        i * 5 : i * 5 + 5
                    ]
                    phase_name = str(int_col).strip()
                    for p in ["HfTiO2", "o-HfO2", "m-HfO2", "t-ZrO2", "TiN"]:
                        if p in phase_name:
                            phase_name = p
                    temp_df = df_raw[[q_col, h_col, k_col, l_col, int_col]].dropna()
                    temp_df.columns = ["Q", "h", "k", "l", "Intensity"]
                    temp_df = temp_df[
                        (temp_df["Q"] >= q_min) & (temp_df["Q"] <= q_max)
                    ].copy()
                    temp_df = temp_df[temp_df["Intensity"] > 1]
                    if not temp_df.empty:
                        temp_df = temp_df.sort_values(by="Intensity", ascending=False)
                        temp_df["Q_round"] = temp_df["Q"].round(2)
                        temp_df = temp_df.drop_duplicates(
                            subset=["Q_round"], keep="first"
                        )
                        temp_df = temp_df.sort_values(by="Q")
                        temp_df["Intensity_Norm"] = (
                            temp_df["Intensity"] / temp_df["Intensity"].max()
                        )

                        def format_hkl(r):
                            try:
                                h, k, l = int(r["h"]), int(r["k"]), int(r["l"])
                                h_str = f"\\bar{{{abs(h)}}}" if h < 0 else str(h)
                                k_str = f"\\bar{{{abs(k)}}}" if k < 0 else str(k)
                                l_str = f"\\bar{{{abs(l)}}}" if l < 0 else str(l)
                                return f"$({h_str}{k_str}{l_str})$"
                            except:
                                return ""

                        temp_df["hkl"] = temp_df.apply(format_hkl, axis=1)
                        ref_data[phase_name] = temp_df
            except:
                ref_data = {}
        cached_ref_data = ref_data
    ref_data = cached_ref_data

    # ★ Optimization 2: Cache fit calculations
    log(f"Calculating fitting for {len(selected_positions)} profiles...")
    root.update()

    cached_fit_results = []
    max_peak_gap_raw = 0.0
    last_data = None
    for pos in selected_positions:
        data = execute_pipeline(pos, current_anchors, current_peaks)  # 변경됨
        cached_fit_results.append(data)

        gap_raw = np.max(data["clean_target_raw"]) - np.min(data["clean_target_raw"])
        if gap_raw > max_peak_gap_raw:
            max_peak_gap_raw = gap_raw
        last_data = data

    selected_cmap_obj = get_selected_cmap()
    colors = selected_cmap_obj(np.linspace(0.0, 0.9, len(selected_positions)))

    fig.clear()
    marker_objects.clear()

    w_ratio = width_ratio_var.get()
    h_ratio = height_ratio_var.get()
    show_ref = show_ref_var.get()
    show_hkl = show_hkl_var.get()

    if show_ref:
        gs = gridspec.GridSpec(
            2,
            2,
            height_ratios=[h_ratio, 1.0],
            width_ratios=[w_ratio, 1.0],
            hspace=0.1,
            wspace=0.25,
        )
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax4 = fig.add_subplot(gs[1, 1], sharex=ax2)
    else:
        gs = gridspec.GridSpec(1, 2, width_ratios=[w_ratio, 1.0], wspace=0.25)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3, ax4 = None, None

    if offset_mode_var.get() == "manual":
        try:
            norm_offset_step = float(norm_offset_var.get())
        except ValueError:
            norm_offset_step = 0.5
        try:
            raw_offset_step = float(raw_offset_var.get())
        except ValueError:
            raw_offset_step = 500.0
    else:
        if last_data is not None:
            norm_offset_step = (
                np.max(last_data["best_bg_fit"]) - np.min(last_data["best_bg_fit"])
            ) * 0.65
        else:
            norm_offset_step = 0.5
        raw_offset_step = max_peak_gap_raw * 1.1

    raw_cumulative_offset = 0.0
    is_raw_overlay = np.isclose(raw_offset_step, 0.0)

    raw_y_min = np.inf
    raw_y_max = -np.inf

    for idx, target_pos in enumerate(selected_positions):
        data = cached_fit_results[idx]
        color = colors[idx]
        display_name = target_pos.replace("th:", "").replace("samz:", "")

        current_norm_offset = idx * norm_offset_step
        ax1.plot(
            q_target,
            data["target_norm"] + current_norm_offset,
            label=f"{display_name}",
            color=color,
            alpha=0.9,
            linewidth=1.5,
            zorder=10,
        )
        if show_bg_var.get():
            ax1.plot(
                q_target,
                data["best_bg_fit"] + current_norm_offset,
                color="#E63946",
                linestyle="--",
                alpha=0.8,
                linewidth=1.2,
                zorder=5,
            )

        y_raw = (
            data["clean_target_raw"]
            if is_raw_overlay
            else data["clean_target_raw"] + raw_cumulative_offset
        )

        ax2.plot(
            q_target,
            y_raw,
            color=color,
            linewidth=1.8,
            zorder=10,
        )

        raw_y_min = min(raw_y_min, np.nanmin(y_raw))
        raw_y_max = max(raw_y_max, np.nanmax(y_raw))

        if not is_raw_overlay:
            raw_cumulative_offset += raw_offset_step

    y1_min, y1_max = ax1.get_ylim()
    ax1_range = y1_max - y1_min
    if np.isclose(ax1_range, 0.0):
        ax1_range = max(abs(y1_max), 1.0) * 0.1
    ax1.set_ylim(y1_min, y1_max + ax1_range * 0.15)

    if np.isfinite(raw_y_min) and np.isfinite(raw_y_max):
        raw_range = raw_y_max - raw_y_min
        if np.isclose(raw_range, 0.0):
            raw_pad = max(abs(raw_y_max), 1.0) * 0.1
        else:
            raw_pad = raw_range * 0.15
        ax2.set_ylim(raw_y_min - raw_pad, raw_y_max + raw_pad)

    if show_ref:
        draw_reference_peaks_separate(ax3, ref_data, show_hkl)
        draw_reference_peaks_separate(ax4, ref_data, show_hkl)

    axes_list = [ax for ax in [ax1, ax2, ax3, ax4] if ax is not None]
    for ax in axes_list:
        if show_mask_var.get():
            for start, end in current_peaks:
                ax.axvspan(start, end, color="#ADB5BD", alpha=0.2, zorder=0)
        if show_anchor_var.get():
            for start, end, w in current_anchors:
                ax.axvspan(start, end, color="#20C997", alpha=0.15, zorder=0)
        for spine in ax.spines.values():
            spine.set_color(BORDER_COLOR)
            spine.set_linewidth(1.5)

    t_font = title_font_var.get()
    ax1.set_title(
        "Stacked Evaluation", fontsize=t_font, fontweight="bold", color=FG_TEXT, pad=10
    )
    ax2.set_title(
        "Stacked Subtracted Signal",
        fontsize=t_font,
        fontweight="bold",
        color=FG_TEXT,
        pad=10,
    )

    # Label axis alignment logic
    ax_t_font = axis_title_font_var.get()
    ax_tick_font = axis_tick_font_var.get()

    if show_ref:
        ax3.set_xlabel(
            r"Q ($\AA^{-1}$)", fontsize=ax_t_font, fontweight="bold", color=FG_TEXT
        )
        ax4.set_xlabel(
            r"Q ($\AA^{-1}$)", fontsize=ax_t_font, fontweight="bold", color=FG_TEXT
        )
        ax1.tick_params(
            labelbottom=show_top_xaxis_var.get(), colors=FG_TEXT, labelsize=ax_tick_font
        )
        ax2.tick_params(
            labelbottom=show_top_xaxis_var.get(), colors=FG_TEXT, labelsize=ax_tick_font
        )
        if show_top_xaxis_var.get():
            ax1.set_xlabel(
                r"Q ($\AA^{-1}$)", fontsize=ax_t_font, fontweight="bold", color=FG_TEXT
            )
            ax2.set_xlabel(
                r"Q ($\AA^{-1}$)", fontsize=ax_t_font, fontweight="bold", color=FG_TEXT
            )
        else:
            ax1.set_xlabel("")
            ax2.set_xlabel("")
    else:
        ax1.set_xlabel(
            r"Q ($\AA^{-1}$)", fontsize=ax_t_font, fontweight="bold", color=FG_TEXT
        )
        ax2.set_xlabel(
            r"Q ($\AA^{-1}$)", fontsize=ax_t_font, fontweight="bold", color=FG_TEXT
        )
        ax1.tick_params(labelbottom=True, colors=FG_TEXT, labelsize=ax_tick_font)
        ax2.tick_params(labelbottom=True, colors=FG_TEXT, labelsize=ax_tick_font)

    # Y축 및 Reference Peak tick 폰트 크기 일괄 적용
    for ax in axes_list:
        ax.tick_params(axis="x", labelsize=ax_tick_font)
        if show_ref and ax in [ax3, ax4]:
            for label in ax.get_yticklabels():
                label.set_fontsize(ax_tick_font)

    l_mode = legend_mode_var.get()
    use_legend, use_cbar = False, False
    if l_mode == "Auto":
        if len(selected_positions) <= 12:
            use_legend = True
        else:
            use_cbar = True
    elif l_mode == "Legend":
        use_legend = True
    elif l_mode == "Colorbar":
        use_cbar = True

    if use_legend:
        # 그래프에 그려진 선(handles)과 이름(labels)을 가져와서 순서를 거꾸로 뒤집음
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(
            handles[::-1],
            labels[::-1],
            loc="upper right",
            fontsize=legend_font_var.get(),
        )

    if use_cbar:
        norm = mcolors.Normalize(vmin=0, vmax=max(1, len(selected_positions) - 1))
        sm = cm.ScalarMappable(cmap=selected_cmap_obj, norm=norm)
        sm.set_array([])

        if len(selected_positions) > 1:
            tick_indices = [0, len(selected_positions) - 1]
        else:
            tick_indices = [0]
        cbar_labels = [items_to_plot[i] for i in tick_indices]

        cb1 = fig.colorbar(sm, ax=ax1, shrink=0.8, aspect=25, pad=0.03)
        cb1.set_ticks(tick_indices)
        cb1.set_ticklabels(cbar_labels)
        cb1.ax.tick_params(labelsize=axis_tick_font_var.get())  # 컬러바 1 틱 크기 적용

        cb2 = fig.colorbar(sm, ax=ax2, shrink=0.8, aspect=25, pad=0.03)
        cb2.set_ticks(tick_indices)
        cb2.set_ticklabels(cbar_labels)
        cb2.set_label(
            f"Variable: {'samz' if mode == 'th' else 'th'}",
            fontsize=axis_title_font_var.get(),
        )  # 컬러바 2 제목 크기 적용
        cb2.ax.tick_params(labelsize=axis_tick_font_var.get())  # 컬러바 2 틱 크기 적용

    for q_val in clicked_q_values:
        draw_single_marker(q_val)

    # ★ Primary spacing adjustment
    fig.tight_layout(pad=2.0)
    if show_top_xaxis_var.get() and show_ref:
        fig.subplots_adjust(hspace=0.4)
    else:
        fig.subplots_adjust(hspace=0.15)

    # ★ Force alignment for lower plots if upper plots are squeezed by colorbar
    if show_ref and use_cbar:
        pos1 = ax1.get_position()
        pos3 = ax3.get_position()
        ax3.set_position([pos1.x0, pos3.y0, pos1.width, pos3.height])

        pos2 = ax2.get_position()
        pos4 = ax4.get_position()
        ax4.set_position([pos2.x0, pos4.y0, pos2.width, pos4.height])

    canvas.draw()
    log("✅ Graph rendering complete!")


def clear_markers():
    for arts in marker_objects.values():
        for art in arts:
            art.remove()
    marker_objects.clear()
    clicked_q_values.clear()
    canvas.draw_idle()
    log("🗑️ Markers cleared.")


def q_to_2theta(q_array, wavelength):
    sin_theta = (q_array * wavelength) / (4.0 * np.pi)
    sin_theta = np.clip(sin_theta, -1.0, 1.0)
    theta_rad = np.arcsin(sin_theta)
    return 2.0 * np.degrees(theta_rad)


def export_all():
    if df_target is None or df_bg is None:
        log("❌ Please click [Apply Data] first.")
        return

    if not current_selection_order:
        log("❌ Please select at least 1 position to save.")
        return

    mode = view_mode_var.get()
    primary_val = primary_combo.get()

    items_to_export = current_selection_order.copy()
    sort_mode = sort_mode_var.get()

    def try_float(val):
        try:
            return float(val)
        except:
            return str(val)

    if sort_mode == "Ascending":
        items_to_export.sort(key=try_float)
    elif sort_mode == "Descending":
        items_to_export.sort(key=try_float, reverse=True)

    selected_positions = []
    for sec_val in items_to_export:
        if mode == "th":
            pos = f"th:{primary_val}_samz:{sec_val}"
        else:
            pos = f"th:{sec_val}_samz:{primary_val}"
        if pos in df_target.columns:
            selected_positions.append(pos)

    if not selected_positions:
        return

    target_path = target_file_var.get().strip()
    base_name = (
        os.path.splitext(os.path.basename(target_path))[0]
        if target_path
        else "Cleaned_Signals"
    )

    try:
        current_anchors, current_peaks = parse_anchors(), parse_peaks()
    except Exception:
        return log("❌ Error: Please check the format.")

    fmt = save_format_var.get()
    smode = save_mode_var.get()
    x_axis = x_axis_var.get()

    if x_axis == "2theta":
        try:
            wl = float(wavelength_var.get())
        except ValueError:
            log("❌ Error: Enter a valid Wavelength.")
            return
        export_x_array = q_to_2theta(q_target, wl)
        export_x_label = "2Theta"
        log(f"▶ Saving with X-axis converted to 2Theta. (λ = {wl} Å)")
    else:
        export_x_array = q_target
        export_x_label = "Q"
        log("▶ Saving with X-axis as Q.")

    if fmt == "xye":
        save_dir = filedialog.askdirectory(title="Select folder to save XYE files")
        if not save_dir:
            return log("⚠ Folder selection cancelled.")
        root.update()
        success_count = 0
        for pos in selected_positions:
            data = execute_pipeline(pos, current_anchors, current_peaks)
            parts = pos.replace("th:", "").replace("samz:", "").split("_")
            th_val, samz_val = (
                parts[0] if len(parts) > 0 else "",
                parts[1] if len(parts) > 1 else "",
            )
            indiv_filename = f"{base_name}_th_{th_val}_samz_{samz_val}.xye"
            indiv_filepath = os.path.join(save_dir, indiv_filename)
            df_xye = pd.DataFrame(
                {export_x_label: export_x_array, "Intensity": data["clean_target_raw"]}
            )
            try:
                df_xye.to_csv(indiv_filepath, sep="\t", header=False, index=False)
                success_count += 1
            except Exception as e:
                log(f"❌ Failed to save XYE '{indiv_filename}': {str(e)}")
        log(
            f"🎉 Total {success_count}/{len(selected_positions)} XYE files successfully saved!"
        )

    elif fmt == "excel":
        if smode == "individual":
            save_dir = filedialog.askdirectory(
                title="Select folder to save individual Excel files"
            )
            if not save_dir:
                return log("⚠ Folder selection cancelled.")
            root.update()
            success_count = 0
            for pos in selected_positions:
                data = execute_pipeline(pos, current_anchors, current_peaks)
                parts = pos.replace("th:", "").replace("samz:", "").split("_")
                th_val, samz_val = (
                    parts[0] if len(parts) > 0 else "",
                    parts[1] if len(parts) > 1 else "",
                )
                indiv_filename = f"{base_name}_th_{th_val}_samz_{samz_val}.xlsx"
                indiv_filepath = os.path.join(save_dir, indiv_filename)
                df_signal = pd.DataFrame(
                    {export_x_label: export_x_array, pos: data["clean_target_raw"]}
                )
                p_r, p_f = data["best_rigid_params"], data["best_flex_params"]
                param_record = [
                    {
                        "Sheet(th)": th_val,
                        "Position(samz)": samz_val,
                        "Full_Position": pos,
                        "Best_BG_Used": data["best_ref_name"],
                        "Rigid_Scale": p_r[0],
                        "Rigid_Q_Scale": p_r[1],
                        "Rigid_dq": p_r[2],
                        "Rigid_c0": p_r[3],
                        "Flex_Scale": p_f[0],
                        "Flex_Q_Scale": p_f[1],
                        "Flex_dq": p_f[2],
                        "Flex_c0": p_f[3],
                        "Flex_c1": p_f[4],
                        "Flex_c2": p_f[5],
                    }
                ]
                try:
                    with pd.ExcelWriter(indiv_filepath) as writer:
                        df_signal.to_excel(
                            writer, sheet_name="Cleaned_Signal", index=False
                        )
                        pd.DataFrame(param_record).to_excel(
                            writer, sheet_name="Fitting_Parameters", index=False
                        )
                    success_count += 1
                except Exception as e:
                    log(f"❌ Failed to save '{indiv_filename}': {str(e)}")
            log(
                f"🎉 Total {success_count}/{len(selected_positions)} individual Excel files saved!"
            )

        else:
            initial_filename = f"{base_name}_Selected.xlsx"
            save_path = filedialog.asksaveasfilename(
                title="Save Merged Excel File",
                initialfile=initial_filename,
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            )
            if not save_path:
                return log("⚠ Save cancelled.")
            root.update()
            df_signals = pd.DataFrame({export_x_label: export_x_array})
            param_records = []
            for pos in selected_positions:
                data = execute_pipeline(pos, current_anchors, current_peaks)
                df_signals[pos] = data["clean_target_raw"]
                parts = pos.replace("th:", "").replace("samz:", "").split("_")
                th_val, samz_val = (
                    parts[0] if len(parts) > 0 else "",
                    parts[1] if len(parts) > 1 else "",
                )
                p_r, p_f = data["best_rigid_params"], data["best_flex_params"]
                param_records.append(
                    {
                        "Sheet(th)": th_val,
                        "Position(samz)": samz_val,
                        "Full_Position": pos,
                        "Best_BG_Used": data["best_ref_name"],
                        "Rigid_Scale": p_r[0],
                        "Rigid_Q_Scale": p_r[1],
                        "Rigid_dq": p_r[2],
                        "Rigid_c0": p_r[3],
                        "Flex_Scale": p_f[0],
                        "Flex_Q_Scale": p_f[1],
                        "Flex_dq": p_f[2],
                        "Flex_c0": p_f[3],
                        "Flex_c1": p_f[4],
                        "Flex_c2": p_f[5],
                    }
                )
            try:
                with pd.ExcelWriter(save_path) as writer:
                    df_signals.to_excel(
                        writer, sheet_name="Cleaned_Signals", index=False
                    )
                    pd.DataFrame(param_records).to_excel(
                        writer, sheet_name="Fitting_Parameters", index=False
                    )
                log(f"🎉 '{os.path.basename(save_path)}' merged save complete!")
            except Exception as e:
                log(f"❌ Excel save failed: {str(e)}")


if os.path.exists("Combi2.xlsx") and os.path.exists("Combi16.xlsx"):
    process_data()

print("🎉 Dashboard launched successfully!")
root.mainloop()
