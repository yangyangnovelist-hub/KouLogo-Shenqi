import os
import platform
import queue
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from sklearn.cluster import MiniBatchKMeans
from tkinter import filedialog, messagebox, ttk
from tkinterdnd2 import DND_FILES, TkinterDnD
import vtracer


@dataclass
class TaskResult:
    ok: bool
    message: str = ""
    image: np.ndarray | None = None
    extra: dict | None = None


@dataclass
class EditorState:
    image: np.ndarray
    selection_mask: np.ndarray | None
    locked_selection_mask: np.ndarray | None


class ProLogoEditorV10(TkinterDnD.Tk):
    # Overlay is always auto-computed — no user picker needed
    # (kept as method for compatibility)

    def __init__(self):
        super().__init__()
        self.title("Logo 精修台 v14")
        self.geometry("1540x1000")
        self.minsize(1240, 780)

        self.filepath = ""
        self.cv_image_rgba: np.ndarray | None = None
        self.history: list[EditorState] = []
        self.max_history = 30
        self.zoom_level = 1.0

        self.p_image_tk = None
        self._checker_cache = None
        self._checker_cache_size = (0, 0)
        self.canvas_img_id = None
        self.canvas_busy_text_id = None
        self.placeholder_id = None

        self.last_mouse_x = -1
        self.last_mouse_y = -1
        self._hover_after_id = None
        self.current_hover_mask: np.ndarray | None = None
        self.selection_mask: np.ndarray | None = None
        self.locked_selection_mask: np.ndarray | None = None
        self._compare_snapshot = None

        self._preview_src_cache: np.ndarray | None = None
        self._preview_cache_key = None
        self._last_hover_key = None
        self._display_cache_key = None

        self.worker_thread = None
        self.worker_queue = queue.Queue()
        self.task_running = False

        self._tol_widgets = []
        self._brush_widgets = []
        self._tool_btns: dict = {}  # populated in setup_ui
        self.brush_size_var = tk.IntVar(value=18)  # 笔刷大小，在左侧面板控制
        self.tol_var = tk.IntVar(value=20)          # 容差，在工具栏控制

        self._mouse_dragging = False
        self._pan_active = False
        self._last_brush_point = None
        self._original_rgba: np.ndarray | None = None  # 原始像素快照，供「画笔补」还原真实颜色

        self.defringe_radius_var = tk.IntVar(value=3)
        self.rembg_model_var = tk.StringVar(value="u2net")

        self.setup_ui()
        self.bind_shortcuts()
        self.after(80, self._poll_worker_queue)
        self.after(100, self._on_tool_mode_changed)  # set initial context-sensitive state

    # UI -----------------------------------------------------------------
    def setup_ui(self):
        toolbar = ttk.Frame(self, padding=(8, 4))
        toolbar.pack(fill=tk.X)

        # ── RIGHT 侧导出按钮必须最先声明，否则被 LEFT 内容挤出可视区 ──
        self.btn_save_png = ttk.Button(toolbar, text="💾 PNG", command=self.save_as_png, state=tk.DISABLED)
        self.btn_save_png.pack(side=tk.RIGHT, padx=4)
        self.btn_save_svg = ttk.Button(toolbar, text="💾 SVG", command=self.save_as_vector, state=tk.DISABLED)
        self.btn_save_svg.pack(side=tk.RIGHT, padx=4)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.RIGHT, fill=tk.Y, padx=6, pady=2)

        # ── LEFT 侧工具区 ──────────────────────────────────────────────
        ttk.Button(toolbar, text="📁 打开", command=self.load_image_dialog).pack(side=tk.LEFT, padx=4)
        self.btn_undo = ttk.Button(toolbar, text="↩ 撤销", command=self.undo, state=tk.DISABLED)
        self.btn_undo.pack(side=tk.LEFT, padx=4)

        # ── 工具组 ──────────────────────────────────────────────────────
        ttk.Label(toolbar, text="工具").pack(side=tk.LEFT, padx=(14, 2))
        self.interaction_mode_var = tk.StringVar(value="魔棒")
        tool_frame = ttk.Frame(toolbar)
        tool_frame.pack(side=tk.LEFT, padx=2)
        self._tool_hints = {
            "魔棒":   "🪄 魔棒 — 移动鼠标预览，左键点击擦除该颜色区域",
            "画笔擦": "🖌 画笔擦 — 拖动擦除像素  [ ] 调整笔刷大小",
            "画笔补": "💧 画笔补 — 拖动恢复像素  [ ] 调整笔刷大小",
            "护棒":   "🛡 护棒 — 移动鼠标预览，左键点击将该颜色区域加入保护（浅灰色显示）",
            "护笔":   "✏ 护笔 — 拖动涂抹，涂到的像素加入保护区域  [ ] 调整笔刷大小",
        }
        self._tool_btns: dict[str, ttk.Button] = {}
        for mode, hint in self._tool_hints.items():
            b = ttk.Button(tool_frame, text=mode, width=6,
                           command=lambda m=mode: self._select_tool(m))
            b.pack(side=tk.LEFT, padx=1)
            self._tool_btns[mode] = b

        self._tol_widgets = []
        self._brush_widgets = []

        ttk.Label(toolbar, text="容差").pack(side=tk.LEFT, padx=(12, 2))
        tol_scale = ttk.Scale(toolbar, from_=1, to=150, variable=self.tol_var, command=self._on_tol_change)
        tol_scale.pack(side=tk.LEFT, ipadx=48)
        tol_spin = ttk.Spinbox(toolbar, from_=1, to=150, textvariable=self.tol_var, width=5, command=self._debounce_preview)
        tol_spin.pack(side=tk.LEFT, padx=4)
        tol_spin.bind("<Return>", lambda _e: self._debounce_preview())
        self._tol_widgets = [tol_scale, tol_spin]

        self.contiguous_mode = tk.BooleanVar(value=False)
        _contiguous_cb = ttk.Checkbutton(toolbar, text="仅相连区域", variable=self.contiguous_mode, command=self._debounce_preview)
        _contiguous_cb.pack(side=tk.LEFT, padx=8)
        self._tol_widgets.append(_contiguous_cb)

        # 查看修改前（原在左侧面板底部，移到工具栏更顺手）
        self.btn_compare = ttk.Button(toolbar, text="👁 按住对比", state=tk.NORMAL)
        self.btn_compare.pack(side=tk.LEFT, padx=(10, 4))
        # 注意：compare 按钮始终保持 NORMAL，不进入 DISABLED 状态
        # 有没有历史记录的判断在 _compare_press 内部处理
        self.btn_compare.bind("<ButtonPress-1>",   self._compare_press)
        self.btn_compare.bind("<ButtonRelease-1>", self._compare_release)

        self.main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        # ── 左侧可滚动面板 ──────────────────────────────────────────
        _left_wrapper = ttk.Frame(self.main_pane)
        self.main_pane.add(_left_wrapper, weight=0)

        _scroll_canvas = tk.Canvas(_left_wrapper, width=240, highlightthickness=0, bd=0, takefocus=False)
        _scrollbar = ttk.Scrollbar(_left_wrapper, orient="vertical", command=_scroll_canvas.yview)
        _scroll_canvas.configure(yscrollcommand=_scrollbar.set)
        _scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        _scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.left_panel = ttk.Frame(_scroll_canvas, padding=6)
        _win_id = _scroll_canvas.create_window((0, 0), window=self.left_panel, anchor="nw")

        def _on_lp_configure(_e):
            _scroll_canvas.configure(scrollregion=_scroll_canvas.bbox("all"))
        def _on_sc_configure(e):
            _scroll_canvas.itemconfig(_win_id, width=e.width)
        self.left_panel.bind("<Configure>", _on_lp_configure)
        _scroll_canvas.bind("<Configure>", _on_sc_configure)

        if platform.system() == "Darwin":
            def _lp_wheel(e): _scroll_canvas.yview_scroll(int(-1 * e.delta), "units")
        else:
            def _lp_wheel(e): _scroll_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

        def _enter_left(_e): self.bind_all("<MouseWheel>", _lp_wheel)
        def _leave_left(_e): self.unbind_all("<MouseWheel>"); self._rebind_canvas_wheel()
        _scroll_canvas.bind("<Enter>", _enter_left)
        _scroll_canvas.bind("<Leave>", _leave_left)

        right_panel = ttk.Frame(self.main_pane)
        self.main_pane.add(right_panel, weight=1)

        self._build_left_panel()

        self.canvas = tk.Canvas(right_panel, bg="#2b2b2b", cursor="crosshair", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.drop_target_register(DND_FILES)
        self.canvas.dnd_bind("<<Drop>>", self.on_drop_file)

        self.canvas.bind("<Button-1>", self.on_left_down)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_up)
        self.canvas.bind("<Motion>", self.on_mouse_hover)
        self.canvas.bind("<Leave>", self.on_mouse_leave)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        # Ensure canvas grabs focus on first click so subsequent events fire immediately
        self.canvas.bind("<Enter>", lambda _e: self.canvas.focus_set())
        self.canvas.bind("<ButtonPress-3>", self.on_pan_start)
        self.canvas.bind("<B3-Motion>", self.on_pan_move)
        self.canvas.bind("<ButtonRelease-3>", self.on_pan_end)
        self.canvas.bind("<ButtonPress-2>", self.on_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_pan_move)
        self.canvas.bind("<ButtonRelease-2>", self.on_pan_end)

        if platform.system() == "Darwin":
            self.canvas.bind("<MouseWheel>", self._wheel_mac)
        else:
            self.canvas.bind("<MouseWheel>", self._wheel_win)

        self._rebind_canvas_wheel()  # ensure initial state

        self._draw_placeholder()

        status_frame = ttk.Frame(self, padding=(10, 3))
        status_frame.pack(fill=tk.X)
        self.status_var = tk.StringVar(value="就绪 — 拖入或打开一张图片")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, anchor="w")
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.zoom_label_var = tk.StringVar(value="")
        ttk.Label(status_frame, textvariable=self.zoom_label_var, anchor="e", width=16).pack(side=tk.RIGHT, padx=(0, 8))
        self.progress = ttk.Progressbar(status_frame, mode="indeterminate", length=120)
        self.progress.pack(side=tk.RIGHT)
        self._status_reset_id = None

    def _build_left_panel(self):
        for i in range(3):
            self.left_panel.columnconfigure(i, weight=1 if i == 1 else 0)

        row = 0

        # ── 使用说明卡片 ─────────────────────────────────────────────
        help_box = ttk.LabelFrame(self.left_panel, text="快速上手", padding=8)
        help_box.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        ttk.Label(help_box, text=(
            "① 选「魔棒」，移动鼠标预览，点击擦除\n"
            "② 选「护棒」，点击将整片同色区域设为保护\n"
            "   选「护笔」，直接涂抹要保护的区域\n"
            "   保护区 = 浅灰色，任何工具都不会影响它\n"
            "③ 擦多了？选「画笔补」涂回来\n"
            "④ Ctrl+Z 随时撤销   F 键适应窗口"
        ), justify=tk.LEFT, foreground="#555").pack(anchor="w")

        row += 1

        # ── 保护区域状态 ──────────────────────────────────────────────
        scope_box = ttk.LabelFrame(self.left_panel, text="保护区域", padding=8)
        scope_box.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        scope_box.columnconfigure(1, weight=1)
        self.scope_var = tk.StringVar(value="全图")
        self.overlay_color_var = tk.StringVar(value="自动")
        self.lock_indicator_var = tk.StringVar(value="⬜ 无保护区域（全图操作）")
        self.lock_indicator_label = ttk.Label(scope_box, textvariable=self.lock_indicator_var, foreground="#888888")
        self.lock_indicator_label.grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(scope_box, text="用「护棒」或「护笔」工具在图上涂抹\n即可添加保护区域（浅灰色显示）",
                  foreground="#777", justify=tk.LEFT).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Button(scope_box, text="🗑 清除全部保护区域",
                   command=self.clear_selection).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(scope_box, text="处理范围", foreground="#777").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(scope_box, textvariable=self.scope_var, values=["全图", "仅保护区域内"], state="readonly", width=12).grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))

        row += 1
        purify_box = ttk.LabelFrame(self.left_panel, text="纯化颜色", padding=8)
        purify_box.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        purify_box.columnconfigure(1, weight=1)
        self.k_colors_var = tk.IntVar(value=8)
        ttk.Label(purify_box, text="颜色数").grid(row=0, column=0, sticky="w")
        ttk.Scale(purify_box, from_=2, to=32, variable=self.k_colors_var, orient=tk.HORIZONTAL).grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Spinbox(purify_box, from_=2, to=32, textvariable=self.k_colors_var, width=5).grid(row=0, column=2, sticky="e")
        self.btn_purify_color = ttk.Button(purify_box, text="🎨 执行纯化颜色", command=self.apply_color_purify, state=tk.DISABLED)
        self.btn_purify_color.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        row += 1
        denoise_box = ttk.LabelFrame(self.left_panel, text="保边去噪", padding=8)
        denoise_box.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        denoise_box.columnconfigure(1, weight=1)
        self.denoise_strength_var = tk.IntVar(value=1)
        ttk.Label(denoise_box, text="强度").grid(row=0, column=0, sticky="w")
        ttk.Scale(denoise_box, from_=0, to=3, variable=self.denoise_strength_var, orient=tk.HORIZONTAL).grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Spinbox(denoise_box, from_=0, to=3, textvariable=self.denoise_strength_var, width=5).grid(row=0, column=2, sticky="e")
        self.btn_denoise = ttk.Button(denoise_box, text="🔇 执行保边去噪", command=self.apply_edge_preserving_denoise, state=tk.DISABLED)
        self.btn_denoise.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        row += 1
        black_box = ttk.LabelFrame(self.left_panel, text="黑线净化", padding=8)
        black_box.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        black_box.columnconfigure(1, weight=1)
        self.black_thresh_var = tk.IntVar(value=80)
        self.black_min_area_var = tk.IntVar(value=8)
        self.black_close_iter_var = tk.IntVar(value=1)
        ttk.Label(black_box, text="黑线阈值").grid(row=0, column=0, sticky="w")
        ttk.Scale(black_box, from_=20, to=180, variable=self.black_thresh_var, orient=tk.HORIZONTAL).grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Spinbox(black_box, from_=20, to=180, textvariable=self.black_thresh_var, width=5).grid(row=0, column=2, sticky="e")
        ttk.Label(black_box, text="最小连通面积").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Scale(black_box, from_=1, to=120, variable=self.black_min_area_var, orient=tk.HORIZONTAL).grid(row=1, column=1, sticky="ew", padx=(8, 8), pady=(6, 0))
        ttk.Spinbox(black_box, from_=1, to=120, textvariable=self.black_min_area_var, width=5).grid(row=1, column=2, sticky="e", pady=(6, 0))
        ttk.Label(black_box, text="补小断线").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Scale(black_box, from_=0, to=2, variable=self.black_close_iter_var, orient=tk.HORIZONTAL).grid(row=2, column=1, sticky="ew", padx=(8, 8), pady=(6, 0))
        ttk.Spinbox(black_box, from_=0, to=2, textvariable=self.black_close_iter_var, width=5).grid(row=2, column=2, sticky="e", pady=(6, 0))
        self.btn_black_cleanup = ttk.Button(black_box, text="🖊 执行黑线净化", command=self.apply_black_cleanup, state=tk.DISABLED)
        self.btn_black_cleanup.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        row += 1
        edge_box = ttk.LabelFrame(self.left_panel, text="边缘平滑", padding=8)
        edge_box.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        edge_box.columnconfigure(1, weight=1)
        self.edge_mode_var = tk.StringVar(value="柔和曲线")
        self.edge_strength_var = tk.DoubleVar(value=1.0)
        ttk.Label(edge_box, text="模式").grid(row=0, column=0, sticky="w")
        ttk.Combobox(edge_box, textvariable=self.edge_mode_var, values=["柔和曲线", "直线规整"], state="readonly", width=12).grid(row=0, column=1, columnspan=2, sticky="ew", padx=(8, 0))
        ttk.Label(edge_box, text="强度").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Scale(edge_box, from_=0.2, to=4.0, variable=self.edge_strength_var, orient=tk.HORIZONTAL).grid(row=1, column=1, sticky="ew", padx=(8, 8), pady=(6, 0))
        ttk.Spinbox(edge_box, from_=0.2, to=4.0, increment=0.1, textvariable=self.edge_strength_var, width=5, format="%.1f").grid(row=1, column=2, sticky="e", pady=(6, 0))
        self.btn_edge_smooth = ttk.Button(edge_box, text="✨ 执行边缘平滑", command=self.apply_edge_smoothing, state=tk.DISABLED)
        self.btn_edge_smooth.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        row += 1
        export_box = ttk.LabelFrame(self.left_panel, text="导出优化", padding=8)
        export_box.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        export_box.columnconfigure(1, weight=1)
        export_box.columnconfigure(2, weight=0)

        self.export_scale_var = tk.IntVar(value=2)
        self.export_padding_var = tk.IntVar(value=24)
        self.export_alpha_mode_var = tk.StringVar(value="柔和")
        self.export_dpi_var = tk.IntVar(value=300)
        self.export_filter_speckle_var = tk.IntVar(value=6)
        self.export_corner_threshold_var = tk.IntVar(value=70)
        self.export_layer_difference_var = tk.IntVar(value=12)
        self.svg_engine_var = tk.StringVar(value="自动")

        # ── PNG 部分 ──────────────────────────────────────────────────
        ttk.Label(export_box, text="PNG 放大倍数", foreground="#555").grid(row=0, column=0, sticky="w")
        ttk.Combobox(export_box, textvariable=self.export_scale_var, values=[1, 2, 4, 6], state="readonly", width=6).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ttk.Label(export_box, text="输出 DPI", foreground="#555").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(export_box, textvariable=self.export_dpi_var, values=[72, 96, 150, 300, 600], state="readonly", width=6).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        ttk.Label(export_box, text="透明边缘", foreground="#555").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(export_box, textvariable=self.export_alpha_mode_var, values=["柔和", "锐利"], state="readonly", width=6).grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        ttk.Label(export_box, text="留白 px", foreground="#555").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(export_box, from_=0, to=200, textvariable=self.export_padding_var, width=6).grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))

        ttk.Separator(export_box, orient=tk.HORIZONTAL).grid(row=4, column=0, columnspan=3, sticky="ew", pady=10)

        # ── SVG 部分 ──────────────────────────────────────────────────
        ttk.Label(export_box, text="SVG 引擎", foreground="#555").grid(row=5, column=0, sticky="w")
        svg_engine_combo = ttk.Combobox(
            export_box, textvariable=self.svg_engine_var,
            values=["自动", "vtracer", "Potrace（需安装）", "Inkscape（需安装）"],
            state="readonly", width=14,
        )
        svg_engine_combo.grid(row=5, column=1, columnspan=2, sticky="ew", padx=(8, 0))

        ttk.Label(export_box, text="去小噪点", foreground="#555").grid(row=6, column=0, sticky="w", pady=(6, 0))
        ttk.Scale(export_box, from_=0, to=20, variable=self.export_filter_speckle_var, orient=tk.HORIZONTAL).grid(row=6, column=1, sticky="ew", padx=(8, 4), pady=(6, 0))
        ttk.Spinbox(export_box, from_=0, to=20, textvariable=self.export_filter_speckle_var, width=4).grid(row=6, column=2, sticky="e", pady=(6, 0))
        ttk.Label(export_box, text="角点保留", foreground="#555").grid(row=7, column=0, sticky="w", pady=(6, 0))
        ttk.Scale(export_box, from_=0, to=100, variable=self.export_corner_threshold_var, orient=tk.HORIZONTAL).grid(row=7, column=1, sticky="ew", padx=(8, 4), pady=(6, 0))
        ttk.Spinbox(export_box, from_=0, to=100, textvariable=self.export_corner_threshold_var, width=4).grid(row=7, column=2, sticky="e", pady=(6, 0))
        ttk.Label(export_box, text="图层差异", foreground="#555").grid(row=8, column=0, sticky="w", pady=(6, 0))
        ttk.Scale(export_box, from_=1, to=40, variable=self.export_layer_difference_var, orient=tk.HORIZONTAL).grid(row=8, column=1, sticky="ew", padx=(8, 4), pady=(6, 0))
        ttk.Spinbox(export_box, from_=1, to=40, textvariable=self.export_layer_difference_var, width=4).grid(row=8, column=2, sticky="e", pady=(6, 0))

        row += 1
        defringe_box = ttk.LabelFrame(self.left_panel, text="去边渗色", padding=8)
        defringe_box.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        defringe_box.columnconfigure(1, weight=1)
        ttk.Label(defringe_box, text="消除边缘残留的背景色像素", foreground="#777",
                  font=("", 9)).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))
        ttk.Label(defringe_box, text="半径").grid(row=1, column=0, sticky="w")
        ttk.Scale(defringe_box, from_=1, to=12, variable=self.defringe_radius_var,
                  orient=tk.HORIZONTAL).grid(row=1, column=1, sticky="ew", padx=(8, 8))
        ttk.Spinbox(defringe_box, from_=1, to=12, textvariable=self.defringe_radius_var,
                    width=5).grid(row=1, column=2, sticky="e")
        self.btn_defringe = ttk.Button(defringe_box, text="🧹 执行去边渗色",
                                       command=self.apply_defringe, state=tk.DISABLED)
        self.btn_defringe.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        row += 1
        rembg_box = ttk.LabelFrame(self.left_panel, text="AI 重新抠图（Rembg）", padding=8)
        rembg_box.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        rembg_box.columnconfigure(1, weight=1)
        ttk.Label(rembg_box, text="用深度学习重新生成 alpha 遮罩\n首次运行自动下载模型（~170MB）",
                  foreground="#777", font=("", 9), justify=tk.LEFT).grid(
                  row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))
        ttk.Label(rembg_box, text="模型").grid(row=1, column=0, sticky="w")
        ttk.Combobox(rembg_box, textvariable=self.rembg_model_var,
                     values=["u2net", "u2net_human_seg", "isnet-general-use", "silueta"],
                     state="readonly", width=16).grid(row=1, column=1, columnspan=2,
                     sticky="ew", padx=(8, 0))
        self.btn_rembg = ttk.Button(rembg_box, text="🤖 执行 AI 抠图",
                                    command=self.apply_rembg, state=tk.DISABLED)
        self.btn_rembg.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        row += 1
        brush_box = ttk.LabelFrame(self.left_panel, text="画笔大小", padding=8)
        brush_box.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        brush_box.columnconfigure(1, weight=1)
        ttk.Label(brush_box, text="大小").grid(row=0, column=0, sticky="w")
        ttk.Scale(brush_box, from_=2, to=120, variable=self.brush_size_var,
                  orient=tk.HORIZONTAL).grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Spinbox(brush_box, from_=2, to=120, textvariable=self.brush_size_var,
                    width=5).grid(row=0, column=2, sticky="e")
        ttk.Label(brush_box, text="[ ] 快捷键增减", foreground="#777",
                  font=("", 9)).grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))

        self.left_panel.rowconfigure(row + 1, weight=1)

    def bind_shortcuts(self):
        for key in ("<Command-z>", "<Control-z>"):
            self.bind(key, lambda _e: self.undo())
        for key in ("<Command-s>", "<Control-s>"):
            self.bind(key, lambda _e: self.save_as_png() if self.cv_image_rgba is not None else None)
        for key in ("<Command-e>", "<Control-e>"):
            self.bind(key, lambda _e: self.save_as_vector() if self.cv_image_rgba is not None else None)
        # [ and ] to decrease/increase brush size
        self.bind("[", lambda _e: self.brush_size_var.set(max(2, self.brush_size_var.get() - 4)))
        self.bind("]", lambda _e: self.brush_size_var.set(min(120, self.brush_size_var.get() + 4)))
        # F to fit image to window
        self.bind("<f>", lambda _e: self._zoom_fit())
        self.bind("<F>", lambda _e: self._zoom_fit())

    def _zoom_fit(self):
        if self.cv_image_rgba is None:
            return
        h, w = self.cv_image_rgba.shape[:2]
        cw = self.canvas.winfo_width() or 800
        ch = self.canvas.winfo_height() or 600
        self.zoom_level = min(cw / max(1, w), ch / max(1, h)) * 0.95
        self._display_cache_key = None
        self._update_zoom_label()
        self.update_display(self.current_hover_mask)

    def _set_status_timed(self, msg, delay=4000):
        """Set status bar message, then auto-reset to image info after delay ms."""
        if self._status_reset_id is not None:
            self.after_cancel(self._status_reset_id)
        self.status_var.set(msg)
        def _reset():
            self._status_reset_id = None
            if self.cv_image_rgba is not None:
                h, w = self.cv_image_rgba.shape[:2]
                self.status_var.set(f"就绪  [{w}×{h}]  Ctrl+Z 撤销 · F 适应窗口 · [ ] 调笔刷")
            else:
                self.status_var.set("就绪 — 拖入或打开一张图片")
        self._status_reset_id = self.after(delay, _reset)

    def _update_lock_indicator(self):
        if self.locked_selection_mask is not None and np.count_nonzero(self.locked_selection_mask) > 0:
            px = int(np.count_nonzero(self.locked_selection_mask))
            self.lock_indicator_var.set(f"🟢 已保护区域（{px:,} px）")
            self.lock_indicator_label.config(foreground="#2e7d32")
        else:
            self.lock_indicator_var.set("⬜ 无保护区域（全图操作）")
            self.lock_indicator_label.config(foreground="#888888")

    # Worker ---------------------------------------------------------------
    def _action_buttons(self):
        # btn_compare is intentionally excluded — it's a read-only preview,
        # disabling it during worker tasks causes the "can't click" symptom
        return [
            self.btn_undo,
            self.btn_save_png, self.btn_save_svg, self.btn_purify_color,
            self.btn_denoise, self.btn_black_cleanup, self.btn_edge_smooth,
            self.btn_defringe, self.btn_rembg,
        ]

    def _set_busy(self, text):
        self.task_running = True
        self.status_var.set(text)
        self.progress.start(12)
        for btn in self._action_buttons():
            try:
                btn.config(state=tk.DISABLED)
            except Exception:
                pass
        self._show_canvas_busy(text)
        # Do NOT call update_idletasks() here — it re-enters the event loop
        # and causes every button click to feel laggy before work even starts.

    def _restore_enabled_state(self):
        self.progress.stop()
        self.task_running = False
        self._hide_canvas_busy()

        has_image = self.cv_image_rgba is not None
        if has_image:
            for btn in [
                self.btn_save_png, self.btn_save_svg, self.btn_purify_color,
                self.btn_denoise, self.btn_black_cleanup, self.btn_edge_smooth,
                self.btn_defringe, self.btn_rembg,
            ]:
                btn.config(state=tk.NORMAL)
        if self.history:
            self.btn_undo.config(state=tk.NORMAL)
        else:
            self.btn_undo.config(state=tk.DISABLED)

        # Only reset to hint message when no timed message is pending
        if self._status_reset_id is None:
            if self.cv_image_rgba is not None:
                h, w = self.cv_image_rgba.shape[:2]
                self.status_var.set(f"就绪  [{w}×{h}]  Ctrl+Z 撤销 · F 适应窗口 · [ ] 调笔刷")
            else:
                self.status_var.set("就绪 — 拖入或打开一张图片")

    def _run_in_worker(self, title, func, on_success=None):
        if self.task_running or self.cv_image_rgba is None:
            return
        self._set_busy(title)

        def _job():
            try:
                result = func()
                self.worker_queue.put(("ok", result, on_success))
            except Exception as exc:
                self.worker_queue.put(("err", str(exc), None))

        self.worker_thread = threading.Thread(target=_job, daemon=True)
        self.worker_thread.start()

    def _poll_worker_queue(self):
        try:
            while True:
                kind, payload, on_success = self.worker_queue.get_nowait()
                if kind == "ok":
                    if on_success is not None:
                        on_success(payload)
                else:
                    messagebox.showerror("处理失败", str(payload))
                self._restore_enabled_state()
        except queue.Empty:
            pass
        # Poll faster while a task is in flight so completion feels instant
        interval = 30 if self.task_running else 120
        self.after(interval, self._poll_worker_queue)

    def _show_canvas_busy(self, text):
        self._hide_canvas_busy()
        self.canvas_busy_text_id = self.canvas.create_text(
            18, 18,
            anchor=tk.NW,
            text=f"⏳ {text}",
            fill="#ffffff",
            font=("Arial", 13, "bold"),
            tags="busy_label",
        )

    def _hide_canvas_busy(self):
        if self.canvas_busy_text_id is not None:
            self.canvas.delete(self.canvas_busy_text_id)
            self.canvas_busy_text_id = None

    # Compare --------------------------------------------------------------
    def _compare_press(self, _event=None):
        if not self.history or self.cv_image_rgba is None:
            self._set_status_timed("暂无历史记录可对比，先进行一次编辑操作")
            return
        self._compare_snapshot = self.cv_image_rgba.copy()
        self.cv_image_rgba = self.history[-1].image.copy()
        # 必须清空 cache key，否则 shape/_image_version 相同会命中缓存、跳过重绘
        self._display_cache_key = None
        self.update_display()
        self.canvas.update_idletasks()
        self.btn_compare.config(relief=tk.SUNKEN, text="👁 松开返回")

    def _compare_release(self, _event=None):
        if self._compare_snapshot is not None:
            self.cv_image_rgba = self._compare_snapshot
            self._compare_snapshot = None
            self._display_cache_key = None
            self.update_display(self.current_hover_mask)
            self.canvas.update_idletasks()
        self.btn_compare.config(relief=tk.RAISED, text="👁 按住对比")

    # Files ----------------------------------------------------------------
    def _draw_placeholder(self):
        self.canvas.delete("all")
        self.canvas_img_id = None
        self.placeholder_id = self.canvas.create_text(
            max(50, self.canvas.winfo_width() // 2),
            max(50, self.canvas.winfo_height() // 2),
            text="拖拽图片到这里，或点击『打开图片』",
            fill="#8e8e8e",
            font=("Arial", 24, "bold"),
        )

    def _on_canvas_configure(self, _event=None):
        if self.cv_image_rgba is None:
            self._draw_placeholder()
        else:
            self._display_cache_key = None
            self.update_display(self.current_hover_mask)

    def on_drop_file(self, event):
        self.process_loaded_file(event.data.strip("{}"))

    def load_image_dialog(self):
        fp = filedialog.askopenfilename(filetypes=[("图片", "*.jpg *.jpeg *.png *.webp")])
        if fp:
            self.process_loaded_file(fp)

    def process_loaded_file(self, filepath):
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            messagebox.showerror("加载错误", "图片格式不支持或路径错误")
            return

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            img[:, :, 3] = 255
        elif img.shape[2] != 4:
            messagebox.showerror("加载错误", "暂不支持该图片通道格式")
            return

        self.filepath = filepath
        self.cv_image_rgba = img.copy()
        self._original_rgba = img.copy()  # 保存原始像素，用于「画笔补」精确还原颜色
        self.history.clear()
        self.zoom_level = 1.0
        self.current_hover_mask = None
        self.selection_mask = np.zeros(self.cv_image_rgba.shape[:2], dtype=np.uint8)
        self.locked_selection_mask = None
        self._compare_snapshot = None
        self._last_brush_point = None
        self._update_lock_indicator()
        self._invalidate_caches(full=True)   # also bumps _image_version
        self._update_zoom_label()
        self.status_var.set(f"已加载：{os.path.basename(filepath)}")
        self.update_display()
        self._restore_enabled_state()

    # Display --------------------------------------------------------------
    # Internal counter bumped whenever image data actually changes
    _image_version: int = 0

    def _bump_image_version(self):
        self._image_version += 1

    def _invalidate_caches(self, full=False):
        self._display_cache_key = None
        self._last_hover_key = None
        if full:
            self._preview_src_cache = None
            self._preview_cache_key = None
            self._bump_image_version()

    def _get_checker_pil(self, w, h):
        if self._checker_cache_size == (w, h) and self._checker_cache is not None:
            return self._checker_cache.copy()
        xs = np.arange(w) // 18
        ys = np.arange(h) // 18
        pattern = (xs[np.newaxis, :] + ys[:, np.newaxis]) % 2
        light = np.array([224, 224, 224], dtype=np.uint8)
        dark = np.array([186, 186, 186], dtype=np.uint8)
        rgb = np.where(pattern[:, :, np.newaxis], light, dark).astype(np.uint8)
        self._checker_cache = Image.fromarray(rgb, "RGB").convert("RGBA")
        self._checker_cache_size = (w, h)
        return self._checker_cache.copy()

    def _resolve_overlay_color(self, display_bgr, mask):
        """Auto-compute maximum-contrast overlay color from image content."""
        if mask is None or np.count_nonzero(mask) == 0:
            return (0, 220, 255)
        pix = display_bgr[mask > 0]
        if len(pix) == 0:
            return (0, 220, 255)
        mean_bgr = pix.mean(axis=0)
        b, g, r = mean_bgr.tolist()
        # Pick the channel furthest from the mean — ensures visibility on any background
        if r > g + 35 and r > b + 35:
            return (0, 220, 255)   # red-dominant image → cyan overlay
        if (b + g) / 2 > r + 20:
            return (255, 208, 0)   # blue/green-dominant → yellow overlay
        return (255, 48, 48)       # default → red overlay

    def update_display(self, preview_mask=None):
        if self.cv_image_rgba is None:
            return

        h, w = self.cv_image_rgba.shape[:2]
        sel_sum = int(np.count_nonzero(self.selection_mask)) if self.selection_mask is not None else 0
        locked_sum = int(np.count_nonzero(self.locked_selection_mask)) if self.locked_selection_mask is not None else 0
        prev_sum = int(np.count_nonzero(preview_mask)) if preview_mask is not None else 0
        key = (
            self.cv_image_rgba.shape,
            self._image_version,
            self.zoom_level,
            self.overlay_color_var.get(),
            sel_sum,
            locked_sum,
            prev_sum,
            self.canvas.winfo_width(),
            self.canvas.winfo_height(),
        )
        if key == self._display_cache_key and self.p_image_tk is not None:
            if self.canvas_img_id is not None:
                self.canvas.itemconfig(self.canvas_img_id, image=self.p_image_tk)
            return

        display = self.cv_image_rgba.copy()

        if self.selection_mask is not None and np.count_nonzero(self.selection_mask) > 0:
            sel = (self.selection_mask > 0) & (display[:, :, 3] > 0)
            display[sel, 1] = np.clip(display[sel, 1].astype(np.int16) + 40, 0, 255).astype(np.uint8)

        if self.locked_selection_mask is not None and np.count_nonzero(self.locked_selection_mask) > 0:
            # 保护区域：浅灰色半透明叠加，让用户一眼看清哪里被保护
            lk = (self.locked_selection_mask > 0) & (display[:, :, 3] > 0)
            gray_val = np.array([200, 200, 200], dtype=np.float32)
            alpha_guard = 0.55
            for c in range(3):
                ch = display[:, :, c].astype(np.float32)
                ch[lk] = ch[lk] * (1.0 - alpha_guard) + gray_val[c] * alpha_guard
                display[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)

        if preview_mask is not None and np.count_nonzero(preview_mask) > 0:
            valid = (preview_mask > 0) & (display[:, :, 3] > 0)
            r, g, b = self._resolve_overlay_color(display[:, :, :3], preview_mask)
            alpha = 0.52
            for c, ov in zip((2, 1, 0), (r, g, b)):
                base = display[:, :, c].astype(np.float32)
                base[valid] = base[valid] * (1.0 - alpha) + ov * alpha
                display[:, :, c] = np.clip(base, 0, 255).astype(np.uint8)

        new_w = max(1, int(w * self.zoom_level))
        new_h = max(1, int(h * self.zoom_level))

        pil_fg = Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGRA2RGBA))
        resample = Image.Resampling.NEAREST if self.zoom_level >= 1 else Image.Resampling.BILINEAR
        pil_fg = pil_fg.resize((new_w, new_h), resample)
        bg = self._get_checker_pil(new_w, new_h)
        bg.paste(pil_fg, (0, 0), pil_fg)
        self.p_image_tk = ImageTk.PhotoImage(bg)

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        x_off = max(0, (cw - new_w) // 2)
        y_off = max(0, (ch - new_h) // 2)

        if self.placeholder_id is not None:
            self.canvas.delete(self.placeholder_id)
            self.placeholder_id = None

        if self.canvas_img_id is None:
            self.canvas.delete("all")
            self.canvas_img_id = self.canvas.create_image(x_off, y_off, image=self.p_image_tk, anchor=tk.NW)
        else:
            self.canvas.coords(self.canvas_img_id, x_off, y_off)
            self.canvas.itemconfig(self.canvas_img_id, image=self.p_image_tk)

        if self.canvas_busy_text_id is not None:
            self.canvas.tag_raise(self.canvas_busy_text_id)

        self.canvas.config(scrollregion=(0, 0, max(cw, new_w + x_off), max(ch, new_h + y_off)))
        self._display_cache_key = key

    def _rebind_canvas_wheel(self):
        """Restore per-canvas wheel bindings (called when mouse leaves left panel)."""
        if platform.system() == "Darwin":
            self.canvas.bind("<MouseWheel>", self._wheel_mac)
        else:
            self.canvas.bind("<MouseWheel>", self._wheel_win)

    def _update_zoom_label(self):
        if self.cv_image_rgba is not None:
            h, w = self.cv_image_rgba.shape[:2]
            self.zoom_label_var.set(f"{w}×{h}  {self.zoom_level*100:.0f}%")
        else:
            self.zoom_label_var.set("")

    def _wheel_win(self, event):
        old = self.zoom_level
        self.zoom_level = min(16.0, self.zoom_level * 1.15) if event.delta > 0 else max(0.15, self.zoom_level / 1.15)
        if old != self.zoom_level:
            self._display_cache_key = None
            self._update_zoom_label()
            self.update_display(self.current_hover_mask)

    def _wheel_mac(self, event):
        old = self.zoom_level
        self.zoom_level = min(16.0, self.zoom_level * 1.1) if event.delta > 0 else max(0.15, self.zoom_level / 1.1)
        if old != self.zoom_level:
            self._display_cache_key = None
            self._update_zoom_label()
            self.update_display(self.current_hover_mask)

    def on_pan_start(self, event):
        self._pan_active = True
        self.canvas.scan_mark(event.x, event.y)
        self.canvas.config(cursor="fleur")

    def on_pan_move(self, event):
        if self._pan_active:
            self.canvas.scan_dragto(event.x, event.y, gain=1)

    def on_pan_end(self, _event):
        self._pan_active = False
        self.canvas.config(cursor="crosshair")

    def _canvas_to_image_coords(self, cx, cy):
        if self.cv_image_rgba is None:
            return -1, -1
        h, w = self.cv_image_rgba.shape[:2]
        nw, nh = int(w * self.zoom_level), int(h * self.zoom_level)
        x_off = max(0, (self.canvas.winfo_width() - nw) // 2)
        y_off = max(0, (self.canvas.winfo_height() - nh) // 2)
        return (
            int((self.canvas.canvasx(cx) - x_off) / self.zoom_level),
            int((self.canvas.canvasy(cy) - y_off) / self.zoom_level),
        )

    def _select_tool(self, mode: str):
        """Switch active tool, update button highlight and status hint."""
        self.interaction_mode_var.set(mode)
        self._last_brush_point = None
        is_wand = mode in ("魔棒", "护棒")        # 两种魔棒都启用容差控件
        is_brush = mode in ("画笔擦", "画笔补", "护笔")  # 三种画笔都启用笔刷控件
        tol_state = tk.NORMAL if is_wand else tk.DISABLED
        brush_state = tk.NORMAL if is_brush else tk.DISABLED
        for w in self._tol_widgets:
            try:
                w.config(state=tol_state)
            except Exception:
                pass
        for w in self._brush_widgets:
            try:
                w.config(state=brush_state)
            except Exception:
                pass
        # Highlight active button with sunken relief
        for m, b in self._tool_btns.items():
            b.config(relief=tk.SUNKEN if m == mode else tk.RAISED)
        hint = self._tool_hints.get(mode, "")
        self.status_var.set(hint)
        self._refresh_overlay_now()

    def _on_tool_mode_changed(self):
        """Compatibility shim — called from after(100,...) on startup."""
        self._select_tool(self.interaction_mode_var.get())

    def _on_tol_change(self, _value=None):
        self._debounce_preview()

    def _debounce_preview(self, *_args):
        if self._hover_after_id:
            self.after_cancel(self._hover_after_id)
        self._hover_after_id = self.after(55, self._do_refresh_preview)

    def _refresh_overlay_now(self):
        self._hover_after_id = None
        if self.interaction_mode_var.get() in ("魔棒", "护棒"):
            self._do_refresh_preview()
        else:
            self.update_display(self.current_hover_mask)

    def on_mouse_hover(self, event):
        self.last_mouse_x, self.last_mouse_y = event.x, event.y
        if self.interaction_mode_var.get() in ("魔棒", "护棒") and not self._mouse_dragging:
            self._debounce_preview()

    def on_mouse_leave(self, _event):
        self.last_mouse_x, self.last_mouse_y = -1, -1
        self._last_brush_point = None
        if self.interaction_mode_var.get() in ("魔棒", "护棒"):
            self.current_hover_mask = None
            self.update_display(None)

    def _get_preview_src(self):
        if self.cv_image_rgba is None:
            return None
        cache_key = (self.cv_image_rgba.shape, self._image_version)
        if self._preview_src_cache is not None and self._preview_cache_key == cache_key:
            return self._preview_src_cache
        self._preview_src_cache = cv2.GaussianBlur(self.cv_image_rgba[:, :, :3], (3, 3), 0)
        self._preview_cache_key = cache_key
        return self._preview_src_cache

    def _do_refresh_preview(self):
        self._hover_after_id = None
        if self.cv_image_rgba is None or self.last_mouse_x == -1:
            return
        if self.interaction_mode_var.get() not in ("魔棒", "护棒"):
            return
        rx, ry = self._canvas_to_image_coords(self.last_mouse_x, self.last_mouse_y)
        h, w = self.cv_image_rgba.shape[:2]
        seed = self._find_opaque_seed(rx, ry)
        if seed is None:
            self.current_hover_mask = None
            self.update_display(None)
            return
        rx, ry = seed
        key = (
            rx, ry, int(self.tol_var.get()), bool(self.contiguous_mode.get()),
            self.cv_image_rgba.shape, self._image_version
        )
        if key == self._last_hover_key and self.current_hover_mask is not None:
            self.update_display(self.current_hover_mask)
            return
        self.current_hover_mask = self._compute_selection_mask(rx, ry)
        self._last_hover_key = key
        self.update_display(self.current_hover_mask)

    def _compute_selection_mask(self, rx, ry):
        if self.cv_image_rgba is None:
            return None
        h, w = self.cv_image_rgba.shape[:2]
        tol = int(self.tol_var.get())
        blr = self._get_preview_src().copy()
        if self.contiguous_mode.get():
            fm = np.zeros((h + 2, w + 2), np.uint8)
            flags = cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
            cv2.floodFill(blr, fm, (rx, ry), (0, 0, 0), (tol,) * 3, (tol,) * 3, flags)
            mask = fm[1:-1, 1:-1]
        else:
            tc = blr[ry, rx].astype(np.int16)
            lo = np.clip(tc - tol, 0, 255).astype(np.uint8)
            hi = np.clip(tc + tol, 0, 255).astype(np.uint8)
            mask = cv2.inRange(blr, lo, hi)
        alpha = self.cv_image_rgba[:, :, 3]
        mask[alpha == 0] = 0
        return mask

    def _ensure_selection_mask(self):
        if self.cv_image_rgba is None:
            return
        if self.selection_mask is None or self.selection_mask.shape != self.cv_image_rgba.shape[:2]:
            self.selection_mask = np.zeros(self.cv_image_rgba.shape[:2], dtype=np.uint8)

    def add_current_hover_to_selection(self):
        if self.current_hover_mask is None:
            messagebox.showinfo("提示", "先把鼠标移动到要选择的位置，看到遮罩后再加入选区。")
            return
        self._push_history_state()
        self._ensure_selection_mask()
        self.selection_mask = cv2.bitwise_or(self.selection_mask, self.current_hover_mask)
        self.update_display(self.current_hover_mask)
        self._restore_enabled_state()

    def subtract_current_hover_from_selection(self):
        if self.current_hover_mask is None:
            messagebox.showinfo("提示", "先把鼠标移动到要选择的位置，看到遮罩后再减选。")
            return
        self._push_history_state()
        self._ensure_selection_mask()
        inv = cv2.bitwise_not(self.current_hover_mask)
        self.selection_mask = cv2.bitwise_and(self.selection_mask, inv)
        self.update_display(self.current_hover_mask)
        self._restore_enabled_state()

    def clear_selection(self):
        if self.cv_image_rgba is None:
            return
        if self.locked_selection_mask is not None:
            self._push_history_state()
        self.locked_selection_mask = None
        self.current_hover_mask = None
        self._update_lock_indicator()
        self._invalidate_caches(full=True)
        self.update_display(None)
        self._restore_enabled_state()
        self.status_var.set("🗑 保护区域已清除，现在操作作用于全图")

    def _get_scope_mask_from_state(self, image_rgba, scope_name, locked_mask):
        alpha = image_rgba[:, :, 3]
        if scope_name == "仅保护区域内":
            # 只在保护区域内部做处理（对选中区局部应用效果）
            if locked_mask is None or np.count_nonzero(locked_mask) == 0:
                raise ValueError("当前作用范围为「仅保护区域内」，但还没有设置保护区域。\n请先用护棒/护笔标记区域。")
            return ((locked_mask > 0) & (alpha > 0)).astype(np.uint8) * 255
        # "全图" —— 全图可见像素，但保护区域完全排除在外
        scope = (alpha > 0).astype(np.uint8) * 255
        if locked_mask is not None and np.count_nonzero(locked_mask) > 0:
            scope[locked_mask > 0] = 0   # 保护区域像素不参与任何批量处理
        return scope

    # Mouse tools ----------------------------------------------------------
    def on_left_down(self, event):
        if self.cv_image_rgba is None or self.task_running:
            return
        self._mouse_dragging = True
        self.last_mouse_x, self.last_mouse_y = event.x, event.y
        self._last_brush_point = (event.x, event.y)
        mode = self.interaction_mode_var.get()
        if mode == "魔棒":
            if self.current_hover_mask is None:
                rx, ry = self._canvas_to_image_coords(event.x, event.y)
                seed = self._find_opaque_seed(rx, ry)
                if seed:
                    self.current_hover_mask = self._compute_selection_mask(*seed)
            self._apply_magic_wand_erase(event.x, event.y)
        elif mode == "护棒":
            self._apply_guard_wand(event.x, event.y)
        else:
            self._push_history_state()
            self._apply_brush_tool(event.x, event.y)

    def on_left_drag(self, event):
        if self.cv_image_rgba is None or self.task_running:
            return
        self.last_mouse_x, self.last_mouse_y = event.x, event.y
        mode = self.interaction_mode_var.get()
        if mode not in ("魔棒", "护棒"):
            self._apply_brush_tool(event.x, event.y)
            self._last_brush_point = (event.x, event.y)

    def on_left_up(self, _event):
        self._mouse_dragging = False
        self._last_brush_point = None
        self._invalidate_caches(full=True)
        self.update_display(self.current_hover_mask)
        self._restore_enabled_state()

    def _find_opaque_seed(self, rx, ry) -> tuple[int, int] | None:
        """返回 (rx,ry) 附近最近的不透明像素坐标；若本身不透明直接返回自身。
        用于修复魔棒/护棒在半透明边缘无反应的问题。"""
        h, w = self.cv_image_rgba.shape[:2]
        if not (0 <= rx < w and 0 <= ry < h):
            return None
        if self.cv_image_rgba[ry, rx, 3] >= 10:
            return rx, ry
        # 向外螺旋搜索，最多扩展 8px
        for r in range(1, 9):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if abs(dx) != r and abs(dy) != r:
                        continue
                    nx, ny = rx + dx, ry + dy
                    if 0 <= nx < w and 0 <= ny < h and self.cv_image_rgba[ny, nx, 3] >= 10:
                        return nx, ny
        return None

    def _apply_magic_wand_erase(self, cx, cy):
        rx, ry = self._canvas_to_image_coords(cx, cy)
        seed = self._find_opaque_seed(rx, ry)
        if seed is None:
            return
        rx, ry = seed
        mask = self._compute_selection_mask(rx, ry)
        if mask is None or cv2.countNonZero(mask) == 0:
            return
        mask = self._get_effective_stroke(mask)
        if cv2.countNonZero(mask) == 0:
            self.status_var.set("🔒 该区域已被保护，无法擦除")
            return
        self._push_history_state()
        self.cv_image_rgba[mask > 0, 3] = 0
        self.current_hover_mask = mask
        self._invalidate_caches(full=True)
        self.update_display(mask)
        self.status_var.set("已执行魔棒擦除")

    def _apply_guard_wand(self, cx, cy):
        """护棒：点击将该颜色区域累加进 locked_selection_mask。"""
        rx, ry = self._canvas_to_image_coords(cx, cy)
        seed = self._find_opaque_seed(rx, ry)
        if seed is None:
            return
        rx, ry = seed
        h, w = self.cv_image_rgba.shape[:2]
        mask = self._compute_selection_mask(rx, ry)
        if mask is None or cv2.countNonZero(mask) == 0:
            return
        self._push_history_state()
        if self.locked_selection_mask is None:
            self.locked_selection_mask = np.zeros((h, w), dtype=np.uint8)
        self.locked_selection_mask = cv2.bitwise_or(self.locked_selection_mask, mask)
        self.current_hover_mask = None
        self._update_lock_indicator()
        self._invalidate_caches(full=True)
        self.update_display(None)
        px = int(np.count_nonzero(self.locked_selection_mask))
        self._set_status_timed(f"🛡 已保护 {px:,} 个像素")

    def _paint_line_mask(self, start, end, radius):
        """生成从 start 到 end 的笔刷遮罩（线段 + 端点圆形）。"""
        h, w = self.cv_image_rgba.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.line(mask, start, end, 255, thickness=max(1, radius * 2), lineType=cv2.LINE_AA)
        cv2.circle(mask, end, radius, 255, thickness=-1, lineType=cv2.LINE_AA)
        return mask

    def _get_effective_stroke(self, stroke: np.ndarray) -> np.ndarray:
        """从 stroke mask 中剔除被 locked_selection_mask 保护的像素。
        保护区域内的像素永远不会被擦除或修改。"""
        if self.locked_selection_mask is not None and np.count_nonzero(self.locked_selection_mask) > 0:
            eff = stroke.copy()
            eff[self.locked_selection_mask > 0] = 0
            return eff
        return stroke

    def _apply_brush_tool(self, cx, cy):
        rx, ry = self._canvas_to_image_coords(cx, cy)
        h, w = self.cv_image_rgba.shape[:2]
        if not (0 <= rx < w and 0 <= ry < h):
            return

        radius = max(1, int(self.brush_size_var.get() / 2))
        if self._last_brush_point is not None:
            px, py = self._canvas_to_image_coords(*self._last_brush_point)
            if 0 <= px < w and 0 <= py < h:
                stroke = self._paint_line_mask((px, py), (rx, ry), radius)
            else:
                stroke = self._paint_line_mask((rx, ry), (rx, ry), radius)
        else:
            stroke = self._paint_line_mask((rx, ry), (rx, ry), radius)

        mode = self.interaction_mode_var.get()
        self._ensure_selection_mask()

        if mode == "画笔擦":
            eff = self._get_effective_stroke(stroke)
            self.cv_image_rgba[eff > 0, 3] = 0
            self.current_hover_mask = eff
            self.status_var.set(f"🖌 画笔擦  笔刷 {self.brush_size_var.get()}px")
        elif mode == "画笔补":
            eff = self._get_effective_stroke(stroke)
            if self._original_rgba is not None:
                self.cv_image_rgba[eff > 0, :3] = self._original_rgba[eff > 0, :3]
            self.cv_image_rgba[eff > 0, 3] = 255
            self.current_hover_mask = eff
            self.status_var.set(f"💧 画笔补  笔刷 {self.brush_size_var.get()}px")
        elif mode == "护笔":
            h, w = self.cv_image_rgba.shape[:2]
            if self.locked_selection_mask is None:
                self.locked_selection_mask = np.zeros((h, w), dtype=np.uint8)
            self.locked_selection_mask = cv2.bitwise_or(self.locked_selection_mask, stroke)
            self.current_hover_mask = stroke
            self._update_lock_indicator()
            self.status_var.set(f"✏ 护笔  笔刷 {self.brush_size_var.get()}px")
        else:
            return

        # 每次笔划都 bump 版本 + 强制刷新，确保画面立即更新
        self._bump_image_version()
        self._display_cache_key = None
        self._last_hover_key = None
        self.update_display(self.current_hover_mask)
        self.canvas.update_idletasks()

    # History --------------------------------------------------------------
    def _snapshot_state(self):
        return EditorState(
            image=self.cv_image_rgba.copy(),
            selection_mask=None if self.selection_mask is None else self.selection_mask.copy(),
            locked_selection_mask=None if self.locked_selection_mask is None else self.locked_selection_mask.copy(),
        )

    def _push_history_state(self):
        if self.cv_image_rgba is None:
            return
        self.history.append(self._snapshot_state())
        if len(self.history) > self.max_history:
            self.history.pop(0)
        self.btn_undo.config(state=tk.NORMAL)

    def undo(self):
        if not self.history or self.task_running:
            return
        state = self.history.pop()
        self.cv_image_rgba = state.image.copy()
        self.selection_mask = None if state.selection_mask is None else state.selection_mask.copy()
        self.locked_selection_mask = None if state.locked_selection_mask is None else state.locked_selection_mask.copy()
        self.current_hover_mask = None
        self._update_lock_indicator()
        self._invalidate_caches(full=True)
        self.update_display(None)
        self._restore_enabled_state()

    # Utils ----------------------------------------------------------------
    @staticmethod
    def _bbox_from_mask(mask):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        x1, x2 = xs.min(), xs.max() + 1
        y1, y2 = ys.min(), ys.max() + 1
        return x1, y1, x2, y2

    @staticmethod
    def _apply_bilateral_strength(bgr, strength):
        s = int(strength)
        if s <= 0:
            return bgr.copy()
        if s == 1:
            return cv2.bilateralFilter(bgr, d=5, sigmaColor=30, sigmaSpace=30)
        if s == 2:
            return cv2.bilateralFilter(bgr, d=7, sigmaColor=45, sigmaSpace=45)
        return cv2.bilateralFilter(bgr, d=9, sigmaColor=60, sigmaSpace=60)

    @staticmethod
    def _blend_region(original, processed, region_mask):
        out = original.copy()
        idx = region_mask > 0
        out[idx] = processed[idx]
        return out

    @staticmethod
    def _remove_small_components(bin_mask, min_area):
        num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
        out = np.zeros_like(bin_mask)
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                out[labels == i] = 255
        return out

    def _prepare_region(self, rgba_state, scope_mask):
        bbox = self._bbox_from_mask(scope_mask)
        if bbox is None:
            raise ValueError("没有可处理的区域。")
        x1, y1, x2, y2 = bbox
        rgba = rgba_state.copy()
        crop = rgba[y1:y2, x1:x2].copy()
        region_mask = scope_mask[y1:y2, x1:x2].copy()
        return rgba, crop, region_mask, bbox

    # Operations -----------------------------------------------------------
    def apply_color_purify(self):
        rgba_state = self.cv_image_rgba.copy()
        scope_name = self.scope_var.get()
        locked_mask = None if self.locked_selection_mask is None else self.locked_selection_mask.copy()
        n_colors = int(self.k_colors_var.get())

        def work():
            scope_mask = self._get_scope_mask_from_state(rgba_state, scope_name, locked_mask)
            rgba, crop, region_mask, bbox = self._prepare_region(rgba_state, scope_mask)
            bgr = crop[:, :, :3]
            alpha = crop[:, :, 3]
            visible = (region_mask > 0) & (alpha > 0)
            pixels = bgr[visible].astype(np.float32)
            if len(pixels) < 2:
                raise ValueError("选区太小，无法纯化颜色。")
            n = max(1, min(n_colors, len(pixels)))
            model = MiniBatchKMeans(n_clusters=n, random_state=42, batch_size=2048, n_init=5)
            labels = model.fit_predict(pixels)
            centers = np.clip(model.cluster_centers_, 0, 255).astype(np.uint8)
            out_bgr = bgr.copy()
            out_bgr[visible] = centers[labels]
            crop[:, :, :3] = out_bgr
            x1, y1, x2, y2 = bbox
            rgba[y1:y2, x1:x2] = crop
            return TaskResult(True, image=rgba, extra={"n": len(centers)})

        def done(result):
            self._push_history_state()
            self.cv_image_rgba = result.image
            self._invalidate_caches(full=True)
            self.update_display(self.current_hover_mask)
            self._set_status_timed(f"✅ 纯化颜色完成：{result.extra['n']} 色")

        self._run_in_worker("正在纯化颜色…", work, done)

    def apply_edge_preserving_denoise(self):
        rgba_state = self.cv_image_rgba.copy()
        scope_name = self.scope_var.get()
        locked_mask = None if self.locked_selection_mask is None else self.locked_selection_mask.copy()
        strength = int(self.denoise_strength_var.get())

        def work():
            scope_mask = self._get_scope_mask_from_state(rgba_state, scope_name, locked_mask)
            rgba, crop, region_mask, bbox = self._prepare_region(rgba_state, scope_mask)
            bgr = crop[:, :, :3]
            denoised = self._apply_bilateral_strength(bgr, strength)
            out_bgr = self._blend_region(bgr, denoised, region_mask)
            crop[:, :, :3] = out_bgr
            x1, y1, x2, y2 = bbox
            rgba[y1:y2, x1:x2] = crop
            return TaskResult(True, image=rgba)

        def done(result):
            self._push_history_state()
            self.cv_image_rgba = result.image
            self._invalidate_caches(full=True)
            self.update_display(self.current_hover_mask)
            self._set_status_timed("✅ 保边去噪完成")

        self._run_in_worker("正在保边去噪…", work, done)

    def apply_black_cleanup(self):
        rgba_state = self.cv_image_rgba.copy()
        scope_name = self.scope_var.get()
        locked_mask = None if self.locked_selection_mask is None else self.locked_selection_mask.copy()
        black_thresh = int(self.black_thresh_var.get())
        black_min_area = max(1, int(self.black_min_area_var.get()))
        close_iter = int(self.black_close_iter_var.get())

        def work():
            scope_mask = self._get_scope_mask_from_state(rgba_state, scope_name, locked_mask)
            rgba, crop, region_mask, bbox = self._prepare_region(rgba_state, scope_mask)
            bgr = crop[:, :, :3]
            alpha = crop[:, :, 3]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            dark_mask = (((gray < black_thresh) & (region_mask > 0) & (alpha > 0))).astype(np.uint8) * 255
            dark_mask = self._remove_small_components(dark_mask, black_min_area)
            if close_iter > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, k, iterations=close_iter)
                dark_mask = self._remove_small_components(dark_mask, black_min_area)
            out_bgr = bgr.copy()
            out_bgr[dark_mask > 0] = (0, 0, 0)
            crop[:, :, :3] = out_bgr
            x1, y1, x2, y2 = bbox
            rgba[y1:y2, x1:x2] = crop
            return TaskResult(True, image=rgba, extra={"pixels": int(np.count_nonzero(dark_mask))})

        def done(result):
            self._push_history_state()
            self.cv_image_rgba = result.image
            self._invalidate_caches(full=True)
            self.update_display(self.current_hover_mask)
            self._set_status_timed(f"✅ 黑线净化完成：处理像素 {result.extra['pixels']}")

        self._run_in_worker("正在净化黑线…", work, done)

    def apply_edge_smoothing(self):
        rgba_state = self.cv_image_rgba.copy()
        scope_name = self.scope_var.get()
        locked_mask = None if self.locked_selection_mask is None else self.locked_selection_mask.copy()
        mode = self.edge_mode_var.get()
        strength = float(self.edge_strength_var.get())

        def work():
            scope_mask = self._get_scope_mask_from_state(rgba_state, scope_name, locked_mask)
            rgba = rgba_state.copy()
            alpha = rgba[:, :, 3].copy()
            obj_mask = ((alpha > 0) & (scope_mask > 0)).astype(np.uint8) * 255
            if np.count_nonzero(obj_mask) == 0:
                raise ValueError("当前区域没有可平滑的边缘。")

            if mode == "柔和曲线":
                # 先轻微膨胀 1px，补偿高斯模糊后二值化时的边缘收缩
                k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                dilated = cv2.dilate(obj_mask, k1, iterations=1)
                ksize = max(3, int(round(strength * 2)) * 2 + 1)
                blurred = cv2.GaussianBlur(dilated, (ksize, ksize), sigmaX=max(0.8, strength))
                # threshold 降至 64：高斯模糊后边缘均值远低于 127，用 127 会让轮廓向内收缩
                smoothed = cv2.threshold(blurred, 64, 255, cv2.THRESH_BINARY)[1]
            else:
                contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                smoothed = np.zeros_like(obj_mask)
                eps = max(0.5, 1.2 * strength)
                for cnt in contours:
                    if cv2.contourArea(cnt) < 4:
                        continue
                    approx = cv2.approxPolyDP(cnt, epsilon=eps, closed=True)
                    cv2.drawContours(smoothed, [approx], -1, 255, thickness=cv2.FILLED)

            alpha[(scope_mask > 0)] = 0
            alpha[smoothed > 0] = 255
            rgba[:, :, 3] = alpha
            return TaskResult(True, image=rgba)

        def done(result):
            self._push_history_state()
            self.cv_image_rgba = result.image
            self._invalidate_caches(full=True)
            self.update_display(self.current_hover_mask)
            self._set_status_timed(f"✅ 边缘平滑完成：{mode}")

        self._run_in_worker("正在平滑边缘…", work, done)

    def apply_defringe(self):
        """去边渗色：把边缘像素的 RGB 替换成内部纯色区域扩散来的颜色，消除背景色污染。"""
        if self.cv_image_rgba is None or self.task_running:
            return
        rgba_state = self.cv_image_rgba.copy()
        scope_name = self.scope_var.get()
        locked_mask = None if self.locked_selection_mask is None else self.locked_selection_mask.copy()
        radius = int(self.defringe_radius_var.get())

        def work():
            scope_mask = self._get_scope_mask_from_state(rgba_state, scope_name, locked_mask)
            result = self._defringe_rgba(rgba_state, radius, scope_mask)
            return TaskResult(True, image=result)

        def done(result):
            self._push_history_state()
            self.cv_image_rgba = result.image
            self._invalidate_caches(full=True)
            self.update_display(self.current_hover_mask)
            self._set_status_timed("✅ 去边渗色完成")

        self._run_in_worker("正在去边渗色…", work, done)

    @staticmethod
    def _defringe_rgba(rgba, radius, scope_mask=None):
        """
        去边渗色核心算法：
        1. 用腐蚀找到"纯内部"像素（远离边缘、背景色未渗入）
        2. 把内部颜色向外高斯扩散
        3. 用扩散色替换边缘区域（alpha>0 但非内部）的 RGB
        同时对 alpha=0 的透明像素也做颜色清理（消除 PNG 边缘光晕）
        """
        alpha = rgba[:, :, 3]
        obj_mask = (alpha > 0).astype(np.uint8) * 255
        if scope_mask is not None:
            # 只处理 scope 范围内的边缘
            obj_mask = cv2.bitwise_and(obj_mask, scope_mask)

        k_size = radius * 2 + 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        interior = cv2.erode(obj_mask, k, iterations=1)

        if np.count_nonzero(interior) == 0:
            # 图像太小腐蚀后全没了，退化为不腐蚀
            interior = obj_mask.copy()

        edge_zone = (obj_mask > 0) & (interior == 0)

        out = rgba.copy()
        bgr = rgba[:, :, :3].astype(np.float32)

        # 只用内部像素的颜色作为"干净色"来源
        interior_color = bgr.copy()
        interior_color[interior == 0] = 0
        interior_weight = (interior > 0).astype(np.float32)

        # 高斯扩散：把内部颜色向外铺开
        blur_k = max(3, radius * 6 + 1) | 1  # 保证奇数
        sigma = radius * 2.0
        spread_color = cv2.GaussianBlur(interior_color, (blur_k, blur_k), sigma)
        spread_weight = cv2.GaussianBlur(interior_weight, (blur_k, blur_k), sigma)

        safe_w = np.where(spread_weight > 1e-6, spread_weight, 1.0)
        clean_rgb = np.clip(spread_color / safe_w[:, :, None], 0, 255)

        # 替换边缘区域 RGB
        for c in range(3):
            ch = out[:, :, c].astype(np.float32)
            ch[edge_zone] = clean_rgb[:, :, c][edge_zone]
            out[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)

        # 顺带清理透明像素的 RGB（消除半透明边缘光晕 artifact）
        transparent_zone = (alpha == 0)
        for c in range(3):
            ch = out[:, :, c].astype(np.float32)
            ch[transparent_zone] = clean_rgb[:, :, c][transparent_zone]
            out[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)

        return out

    def apply_rembg(self):
        """AI 重新抠图：用 rembg/U2Net 生成高质量 alpha 遮罩，替换当前 alpha 通道。"""
        try:
            import rembg  # noqa: F401
        except ImportError:
            messagebox.showerror(
                "缺少依赖 rembg",
                "请先在终端运行：\n\npip install rembg\n\n安装后重新打开工具即可使用。"
            )
            return
        if self.cv_image_rgba is None or self.task_running:
            return
        rgba_state = self.cv_image_rgba.copy()
        model_name = self.rembg_model_var.get()

        def work():
            from rembg import remove, new_session
            pil_in = Image.fromarray(cv2.cvtColor(rgba_state, cv2.COLOR_BGRA2RGBA))
            session = new_session(model_name)
            pil_out = remove(pil_in, session=session)
            new_alpha = np.array(pil_out)[:, :, 3]  # 只取 AI 生成的 alpha
            out = rgba_state.copy()
            out[:, :, 3] = new_alpha
            return TaskResult(True, image=out)

        def done(result):
            self._push_history_state()
            self.cv_image_rgba = result.image
            self._invalidate_caches(full=True)
            self.update_display(self.current_hover_mask)
            self._set_status_timed(f"✅ AI 抠图完成（{model_name}）")

        self._run_in_worker(
            f"AI 正在重新抠图（{model_name}，首次运行需下载模型）…", work, done
        )

    # Export ---------------------------------------------------------------
    @staticmethod
    def _crop_to_content(rgba, padding=0):
        alpha = rgba[:, :, 3]
        ys, xs = np.where(alpha > 0)
        if len(xs) == 0:
            return rgba.copy()
        x1 = max(0, int(xs.min()) - padding)
        y1 = max(0, int(ys.min()) - padding)
        x2 = min(rgba.shape[1], int(xs.max()) + 1 + padding)
        y2 = min(rgba.shape[0], int(ys.max()) + 1 + padding)
        return rgba[y1:y2, x1:x2].copy()

    @staticmethod
    def _resize_rgba_premultiplied(rgba, scale):
        if scale == 1:
            return rgba.copy()
        h, w = rgba.shape[:2]
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        rgb = rgba[:, :, :3].astype(np.float32)
        a = rgba[:, :, 3:4].astype(np.float32) / 255.0
        premul = rgb * a
        premul_up = cv2.resize(premul, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        alpha_up = cv2.resize(rgba[:, :, 3], (new_w, new_h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        alpha_up = np.clip(alpha_up, 0, 255)
        a_up = alpha_up[:, :, None] / 255.0
        rgb_up = np.zeros_like(premul_up)
        nz = a_up[:, :, 0] > 1e-6
        rgb_up[nz] = premul_up[nz] / a_up[nz]
        rgb_up = np.clip(rgb_up, 0, 255).astype(np.uint8)
        return np.dstack([rgb_up, alpha_up.astype(np.uint8)])

    def _build_export_rgba_from_state(self, rgba_state, scale, padding, alpha_mode):
        rgba = rgba_state.copy()
        rgba = self._crop_to_content(rgba, padding=padding)
        rgba = self._resize_rgba_premultiplied(rgba, scale)
        # 支持中文标签和旧英文标签
        if alpha_mode in ("crisp", "锐利"):
            alpha = rgba[:, :, 3]
            _, alpha = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
            rgba[:, :, 3] = alpha
        return rgba

    def save_as_png(self):
        if self.cv_image_rgba is None or self.task_running:
            return
        filepath = self.filepath
        d, fname = os.path.split(filepath)
        name, _ = os.path.splitext(fname)
        default_name = f"{name}_导出.png"
        out_path = filedialog.asksaveasfilename(
            initialdir=d,
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[("PNG 图片", "*.png")],
            title="导出 PNG",
        )
        if not out_path:
            return
        rgba_state = self.cv_image_rgba.copy()
        alpha_mode = self.export_alpha_mode_var.get()
        scale = int(self.export_scale_var.get())
        padding = int(self.export_padding_var.get())
        dpi = int(self.export_dpi_var.get())

        def work():
            rgba = self._build_export_rgba_from_state(rgba_state, scale, padding, alpha_mode)
            pil_img = Image.fromarray(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA))
            # Embed DPI metadata so the file opens at correct physical size in Illustrator / print
            pil_img.save(out_path, format="PNG", dpi=(dpi, dpi), optimize=False)
            return TaskResult(True, message=out_path, extra={"dpi": dpi, "size": pil_img.size})

        def done(result):
            w, h = result.extra["size"]
            self._set_status_timed(
                f"✅ PNG 已导出：{os.path.basename(result.message)}  ({w}×{h} px, {result.extra['dpi']} DPI)"
            )

        self._run_in_worker("正在导出 PNG…", work, done)

    def save_as_vector(self):
        if self.cv_image_rgba is None or self.task_running:
            return
        filepath = self.filepath
        d, fname = os.path.split(filepath)
        name, _ = os.path.splitext(fname)
        default_name = f"{name}_导出.svg"
        out_path = filedialog.asksaveasfilename(
            initialdir=d,
            initialfile=default_name,
            defaultextension=".svg",
            filetypes=[("SVG 矢量图", "*.svg")],
            title="导出 SVG",
        )
        if not out_path:
            return
        rgba_state = self.cv_image_rgba.copy()
        alpha_mode = self.export_alpha_mode_var.get()
        scale = int(self.export_scale_var.get())
        padding = int(self.export_padding_var.get())
        filter_speckle = int(self.export_filter_speckle_var.get())
        corner_threshold = int(self.export_corner_threshold_var.get())
        layer_difference = int(self.export_layer_difference_var.get())
        engine_choice = self.svg_engine_var.get()
        inkscape = self._find_inkscape()
        potrace = self._find_potrace()

        def work():
            rgba = self._build_export_rgba_from_state(rgba_state, scale, padding, alpha_mode)
            if engine_choice == "Potrace（需安装）":
                if not potrace:
                    raise RuntimeError("未找到 Potrace。请先安装：\nmacOS: brew install potrace\nWindows: https://potrace.sourceforge.net")
                svg_str = self._potrace_direct(rgba, potrace)
            else:
                svg_mode = "单色(黑线)" if engine_choice == "Inkscape（需安装）" else "彩色"
                svg_str = self._convert_to_svg_pro(
                    rgba, svg_mode, filter_speckle, corner_threshold, layer_difference,
                    inkscape_override=inkscape if engine_choice != "vtracer" else None,
                )
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(svg_str)
            h2, w2 = rgba.shape[:2]
            path_count = svg_str.count("<path")
            return TaskResult(True, message=out_path, extra={"paths": path_count, "size": (w2, h2)})

        def done(result):
            w2, h2 = result.extra["size"]
            self._set_status_timed(
                f"✅ SVG 已导出：{os.path.basename(result.message)}  ({result.extra['paths']} 路径, {w2}×{h2})"
            )

        self._run_in_worker("正在导出 SVG（专业模式）…", work, done)

    # ── Professional SVG conversion pipeline ──────────────────────────────
    @staticmethod
    def _svg_clean_paths(svg_str):
        """Post-process: remove redundant whitespace in path data, add viewBox if missing."""
        import re
        # Collapse runs of spaces in d="..." attributes
        def _clean_d(m):
            d = m.group(1)
            d = re.sub(r'  +', ' ', d).strip()
            return f'd="{d}"'
        svg_str = re.sub(r'd="([^"]+)"', _clean_d, svg_str)
        return svg_str

    @staticmethod
    def _build_svg_wrapper(w, h, body):
        """Wrap path body in a clean SVG root with correct viewBox and metadata."""
        return (
            f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<!-- Generated by Logo Editor v11 -->\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
            f'{body}\n'
            f'</svg>\n'
        )

    def _convert_to_svg_pro(self, rgba, svg_mode, filter_speckle, corner_threshold, layer_difference, inkscape_override=None):
        """
        Multi-strategy SVG conversion:
        1. Try Potrace (inkscape CLI) for single-color / line-art mode — best quality
        2. Fall back to vtracer for color mode
        3. For color mode: pre-quantize colors before tracing to reduce path bloat
        4. Post-process: inject correct viewBox, clean path data, add alpha mask group
        """
        import re, subprocess, tempfile

        h, w = rgba.shape[:2]
        pil_rgba = Image.fromarray(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA))

        # ── Strategy A: single-color / Potrace via Inkscape CLI ───────────
        use_potrace = svg_mode == "单色(黑线)"
        if use_potrace:
            inkscape_path = inkscape_override or self._find_inkscape()
            if inkscape_path:
                try:
                    return self._potrace_via_inkscape(pil_rgba, inkscape_path, w, h)
                except Exception:
                    pass  # fall through to vtracer

        # ── Strategy B: color mode with vtracer ───────────────────────────
        # Pre-quantize 改为 KMeans（更平滑的颜色分桶，避免 MEDIANCUT 产生的色带/色阶崩溃）
        # 同时先做轻微高斯模糊，消除噪点对量化结果的影响
        n_colors = max(2, min(24, 32 - layer_difference))
        from PIL import ImageFilter
        pil_blurred = pil_rgba.filter(ImageFilter.GaussianBlur(radius=1.0))
        rgb_arr = np.array(pil_blurred)[:, :, :3].reshape(-1, 3).astype(np.float32)
        km = MiniBatchKMeans(n_clusters=n_colors, random_state=42, batch_size=4096, n_init=5)
        labels = km.fit_predict(rgb_arr)
        centers = np.clip(km.cluster_centers_, 0, 255).astype(np.uint8)
        q_rgb = centers[labels].reshape(h, w, 3)
        orig_alpha = pil_rgba.split()[3]
        pil_quantized = Image.fromarray(
            np.dstack([q_rgb, np.array(orig_alpha)]), "RGBA"
        )

        svg_str = vtracer.convert_pixels_to_svg(
            list(pil_quantized.getdata()),
            size=(w, h),
            colormode="color",
            hierarchical="stacked",
            mode="spline",
            filter_speckle=filter_speckle,
            color_precision=8,                        # 从 6 → 8，保留更多颜色细节
            layer_difference=max(1, min(layer_difference, 8)),  # 上限收窄，避免色阶跳变
            corner_threshold=corner_threshold,
        )

        # ── Post-process ──────────────────────────────────────────────────
        # 1. Ensure correct viewBox
        if 'viewBox' not in svg_str:
            svg_str = svg_str.replace('<svg ', f'<svg viewBox="0 0 {w} {h}" ', 1)

        # 2. Inject transparency: wrap all paths in a <g> with clip-path from alpha mask
        # Extract the alpha channel as a 1-bit mask and embed as SVG clipPath
        alpha_np = np.array(orig_alpha)
        _, alpha_bin = cv2.threshold(alpha_np, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(alpha_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

        clip_paths = []
        for cnt in contours:
            if cv2.contourArea(cnt) < filter_speckle * filter_speckle:
                continue
            pts = cnt.reshape(-1, 2)
            if len(pts) < 3:
                continue
            d = "M " + " L ".join(f"{x},{y}" for x, y in pts) + " Z"
            clip_paths.append(f'  <path d="{d}"/>')

        if clip_paths:
            clip_id = "alpha_clip"
            clip_def = (
                f'<defs><clipPath id="{clip_id}">\n'
                + "\n".join(clip_paths)
                + f'\n</clipPath></defs>'
            )
            # Inject clipPath defs after <svg ...> opening tag
            svg_str = re.sub(r'(<svg[^>]*>)', r'\1\n' + clip_def, svg_str, count=1)
            # Wrap body paths in a clipped group
            svg_str = re.sub(
                r'(<svg[^>]*>)(.*?)(</svg>)',
                lambda m: m.group(1) + m.group(2).replace(
                    '<g ', f'<g clip-path="url(#{clip_id})" ', 1
                ) + m.group(3),
                svg_str, flags=re.DOTALL
            )

        # 3. Clean up path data whitespace
        svg_str = self._svg_clean_paths(svg_str)

        # 4. Add generator comment if missing
        if '<?xml' not in svg_str:
            svg_str = '<?xml version="1.0" encoding="UTF-8"?>\n<!-- Generated by Logo Editor v11 -->\n' + svg_str

        return svg_str

    @staticmethod
    def _find_inkscape():
        """Return Inkscape executable path if available, else None."""
        import shutil
        candidates = [
            "inkscape",
            "/Applications/Inkscape.app/Contents/MacOS/inkscape",
            r"C:\Program Files\Inkscape\bin\inkscape.exe",
            r"C:\Program Files (x86)\Inkscape\bin\inkscape.exe",
        ]
        for c in candidates:
            if shutil.which(c) or os.path.isfile(c):
                return c
        return None

    @staticmethod
    def _find_potrace():
        """Return potrace executable path if available, else None."""
        import shutil
        candidates = [
            "potrace",
            "/usr/local/bin/potrace",
            "/opt/homebrew/bin/potrace",
            r"C:\Program Files\potrace\potrace.exe",
        ]
        for c in candidates:
            if shutil.which(c) or os.path.isfile(c):
                return c
        return None

    @staticmethod
    def _potrace_direct(rgba, potrace_path):
        """
        用 Potrace CLI 直接矢量化（黑线图最佳质量）：
        1. RGBA → 1-bit PBM（黑=前景）
        2. potrace 输出 SVG
        3. 注入正确的 viewBox
        """
        import subprocess, tempfile, re
        h, w = rgba.shape[:2]
        gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_BGR2GRAY)
        alpha = rgba[:, :, 3]
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        bw[alpha < 128] = 0  # 透明区域强制白（背景）

        with tempfile.TemporaryDirectory() as tmp:
            pbm_path = os.path.join(tmp, "input.pbm")
            svg_path = os.path.join(tmp, "output.svg")
            Image.fromarray(bw).convert("1").save(pbm_path)
            result = subprocess.run(
                [potrace_path, pbm_path, "-s", "-o", svg_path,
                 "--tight", "--opttolerance", "0.2"],
                capture_output=True, timeout=30,
            )
            if not os.path.exists(svg_path):
                raise RuntimeError(f"Potrace 失败：{result.stderr.decode(errors='replace')}")
            with open(svg_path, encoding="utf-8") as f:
                svg_str = f.read()

        # 修正 viewBox（Potrace 有时输出 pt 单位）
        if f'viewBox="0 0 {w} {h}"' not in svg_str:
            svg_str = re.sub(
                r'<svg[^>]*>',
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
                svg_str, count=1,
            )
        return svg_str

    @staticmethod
    def _potrace_via_inkscape(pil_rgba, inkscape_path, w, h):
        """
        Use Inkscape's built-in Potrace tracer for single-color / line-art SVG.
        Produces the cleanest possible Bezier curves for logos.
        """
        import subprocess, tempfile
        # Flatten to grayscale then threshold
        gray = np.array(pil_rgba.convert("L"))
        alpha = np.array(pil_rgba.split()[3])
        _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        bw[alpha < 128] = 0
        bw_img = Image.fromarray(bw)

        with tempfile.TemporaryDirectory() as tmp:
            png_path = os.path.join(tmp, "input.png")
            svg_path = os.path.join(tmp, "output.svg")
            bw_img.save(png_path)
            result = subprocess.run(
                [
                    inkscape_path,
                    png_path,
                    "--actions=select-all;object-to-path;trace-bitmap:select=all;export-filename:" + svg_path + ";export-do",
                ],
                capture_output=True, timeout=30,
            )
            if os.path.exists(svg_path):
                with open(svg_path, encoding="utf-8") as f:
                    return f.read()
        raise RuntimeError("Inkscape trace failed")


if __name__ == "__main__":
    app = ProLogoEditorV10()
    app.mainloop()
