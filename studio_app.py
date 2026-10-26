"""
VHR Maths Engine Pro - Studio (real-time equation <-> image workstation).
Created & Published by Vihanga Himantha

  Tab 1  Equation -> Image : type an equation, see it render live (debounced, progressive, animatable with `t`,
                             auto sliders for  `name = value  # [lo, hi]`  lines, typeset LaTeX preview).
  Tab 2  Image -> Equation : photo/clipboard -> closed-form cosine/primitive equation with a LIVE term-count
                             slider (milliseconds per update), rate/distortion plot and measured PSNR/SSIM.
  Tab 3  Catalog           : Cosmic reference plates (Galaxy Cluster, Colliding Planets).
  Tab 4  Grid & Fractions  : coordinate grid presets and LaTeX fraction settings.

All numeric work runs on background threads (latest request wins, stale work is cancelled cooperatively);
Tk is only ever touched from the UI thread.
"""
import os
import queue
import sys
import threading
import time
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageGrab, ImageTk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.mathtext import MathTextParser
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from models import GridConfig, VHRModel
from algebraic_core import AlgebraicCore
from exact_catalog import VHRExactCatalog, RenderCancelled
from precision_decomposer import PrecisionImageDecomposer, DecompositionCancelled, ImageAnalysis
from exporters import VHRExporterSuite, ExportError
from expression_engine import (
    ExpressionError, EvaluationCancelled, RenderContext, compile_program, find_sliders, set_slider_value,
    to_latex, Program,
)

APP_TITLE = "VHR Maths Engine Pro v4.0 - Published by Vihanga Himantha"

EXAMPLES: Dict[str, str] = {
    "Plasma (animated)": """# Animated plasma - tick "Animate", drag the sliders
speed = 1.0   # [0, 4]
zoom = 3      # [0.5, 10]
v = sin(zoom*x + t*speed) + sin(zoom*(y*cos(t*0.3) + x*sin(t*0.3)) + t*0.7) + sin(zoom*sqrt(x^2 + y^2) - t)
H(x, y, s) = 0.5 + 0.5*sin(pi*v + 2*pi*s/3)
""",
    "Neon rose (polar)": """# Polar rose with glow, RGB written out separately
petals = 5    # [2, 12]
glow = 18     # [4, 60]
rr = sqrt(x^2 + y^2)
ang = atan2(y, x)
curve = 0.75*abs(cos(petals*ang/2 + t*0.5))
I = exp(-glow*abs(rr - curve))
R = I
G = 0.55*I + 0.25*exp(-5*rr^2)
B = 0.35 + 0.65*I
""",
    "Wave interference": """# Two point sources
freq = 40     # [5, 90]
d1 = sqrt((x + 0.5)^2 + y^2)
d2 = sqrt((x - 0.5)^2 + y^2)
w = sin(freq*d1 - 6*t) + sin(freq*d2 - 6*t)
H(x, y, s) = 0.5 + 0.25*w*(1 + 0.5*cos(2*pi*s/3 + 3*d1))
""",
    "Spiral galaxy (sum / prod)": """# Logarithmic spiral arms + a field of stars using sum(expr, k, lo, hi)
arms = 3      # [1, 6]
twist = 4     # [0, 10]
rr = sqrt(x^2 + y^2) + 0.000001
ang = atan2(y, x)
arm = 0.5 + 0.5*cos(arms*(ang - twist*ln(rr + 0.15) + 0.4*t))
core = exp(-10*rr^2)
disk = 0.9*exp(-3*rr)*arm
stars = sum(exp(-4000*((x - 0.9*cos(7*k^2))^2 + (y - 0.6*sin(5*k^3))^2)), k, 1, 60)
R = 1.2*core + 0.55*disk + stars
G = 0.9*core + 0.65*disk + stars
B = 0.7*core + 1.0*disk + stars
""",
    "VHR-style discs (prod)": """# 1 - prod(1 - A_k): the double-exponential coverage trick in VHR plates
n_discs = 14  # [1, 40]
sharp = 30    # [4, 80]
A(k) = exp(-exp(sharp*((x - 0.7*cos(k*2.4))^2 + (y - 0.45*sin(k*1.9 + t))^2 - 0.012)))
cover = 1 - prod(1 - A(k), k, 1, n_discs)
H(x, y, s) = cover*(0.35 + 0.65*sin(pi*(x + y)/2 + s)^2)
""",
    "Heart (implicit curve)": """# Implicit curve (x^2+y^2-1)^3 - x^2 y^3 <= 0 with a soft edge
soft = 0.03   # [0.005, 0.2]
px = 1.15*x
py = 1.15*y + 0.1
h = (px^2 + py^2 - 1)^3 - px^2*py^3
inside = 1 - smoothstep(-soft, soft, h)
R = 0.06 + inside*(0.93 + 0.07*sin(3*t))
G = 0.04 + inside*0.10
B = 0.12 + inside*0.28
""",
}

SYNTAX_HELP = """EQUATION SYNTAX

Variables   x, y          grid coordinates (default window: y in [-1, 1])
            m, n          1-based pixel column / row
            r, th         polar radius / angle      s   colour channel (0, 1, 2)
            t             time in seconds (tick 'Animate')      Wpx, Hpx   image size
Constants   pi  e  tau  phi
Operators   + - * / ^ (or **)  %  comparisons < <= > >= == !=   a if cond else b
            implicit multiplication: 2x, 3(x+1), 2sin(x), (a)(b)
Functions   sin cos tan asin acos atan atan2 sinh cosh tanh exp ln log log10 log2 sqrt cbrt abs
            floor ceil round sign fract min max clip mod pow hypot smoothstep mix where step
Sums        sum(expr, k, lo, hi)   prod(expr, k, lo, hi)   (optional 5th argument = step)
Lists       A = [1, 2, 3]   then A[k] inside sum/prod
Functions   f(a, b) = a^2 + b      (you may use x, y, s, t inside)
Output      define  H(x, y, s) = ...  or  H = ...  or  R = ...  G = ...  B = ...
            (a bare last line is used as H)
Sliders     scale = 3   # [0.5, 10]   <- becomes a live slider
Grid        #! grid 640 360 320 180 180     (W H m0 n0 S)  optional, set by exported programs

The result is quantised with VHR's  F(x) = floor(255 e^{-e^{-1000x}} |x|^{e^{-e^{1000(x-1)}}}).
"""


# ======================================================================================
# Infrastructure
# ======================================================================================
class JobRunner:
    """Runs jobs on one daemon thread. A new submit() cancels the running job cooperatively and replaces any
    queued one (latest wins). Results are delivered through `events` and consumed on the Tk thread."""

    def __init__(self, name: str):
        self._cv = threading.Condition()
        self._pending = None
        self._current_cancel = threading.Event()
        self.events: "queue.Queue" = queue.Queue()
        self.busy = False
        threading.Thread(target=self._loop, daemon=True, name=name).start()

    def submit(self, fn: Callable, tag: str) -> threading.Event:
        with self._cv:
            self._current_cancel.set()
            token = threading.Event()
            self._current_cancel = token
            self._pending = (fn, tag, token)
            self._cv.notify()
        return token

    def cancel_all(self) -> None:
        with self._cv:
            self._current_cancel.set()
            self._pending = None

    def _loop(self) -> None:
        while True:
            with self._cv:
                while self._pending is None:
                    self._cv.wait()
                fn, tag, token = self._pending
                self._pending = None
                self.busy = True

            def emit(kind: str, payload=None, _tag=tag, _token=token):
                self.events.put((_tag, kind, payload, _token))

            try:
                fn(token.is_set, emit)
                emit("done")
            except (EvaluationCancelled, DecompositionCancelled, RenderCancelled):
                pass
            except Exception as exc:  # delivered to the UI thread; never swallowed silently
                self.events.put((tag, "error", (exc, traceback.format_exc()), token))
            finally:
                self.busy = False


def draw_equation_lines(ax, lines: List[str], title: str = "", base_fs: float = 10.0) -> None:
    """Typesets LaTeX lines onto `ax`, shrinking each line to fit the axes width and pre-validating with
    matplotlib's parser so a bad line degrades to monospace text instead of raising in a Tk callback."""
    ax.clear()
    ax.axis("off")
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    parser = MathTextParser("agg")
    width_px, height_px = max(ax.bbox.width, 50.0), max(ax.bbox.height, 50.0)
    y_px = height_px - 4.0
    items = ([("title", title)] if title else []) + [("eq", ln) for ln in lines]
    for kind, raw in items:
        if y_px < 6:
            break
        clean = AlgebraicCore.sanitize_for_matplotlib(raw)
        if kind == "title":
            txt = ax.text(0.5, y_px / height_px, clean, transform=ax.transAxes, ha="center", va="top",
                          fontsize=base_fs + 1.5, fontweight="bold")
        else:
            expr = f"${clean}$"
            try:
                parser.parse(expr, dpi=100)
                txt = ax.text(0.5, y_px / height_px, expr, transform=ax.transAxes, ha="center", va="top", fontsize=base_fs)
            except Exception:
                txt = ax.text(0.5, y_px / height_px, clean[:260], transform=ax.transAxes, ha="center", va="top",
                              fontsize=base_fs - 2.5, family="monospace", color="#8a1c1c")
        try:
            bb = txt.get_window_extent(renderer)
            if bb.width > width_px - 12:
                txt.set_fontsize(max(3.0, txt.get_fontsize() * (width_px - 12) / bb.width))
                bb = txt.get_window_extent(renderer)
            y_px -= bb.height + 7.0
        except Exception:
            y_px -= 24.0


_INFERNO_LUT: Optional[np.ndarray] = None


def _inferno_lut() -> np.ndarray:
    global _INFERNO_LUT
    if _INFERNO_LUT is None:
        cmap = matplotlib.colormaps["inferno"]
        _INFERNO_LUT = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    return _INFERNO_LUT


# ======================================================================================
# Viewport
# ======================================================================================
class ImageViewport(tk.Frame):
    """Canvas-based viewer: fit/zoom/pan, wipe comparison, x10 error heat-map, layer split and a pixel probe."""

    def __init__(self, master, on_probe: Callable[[str], None]):
        super().__init__(master, bg="#14161a")
        self.on_probe = on_probe
        self.canvas = tk.Canvas(self, bg="#14161a", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.result: Optional[np.ndarray] = None
        self.original: Optional[np.ndarray] = None
        self.layers: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.grid: Optional[GridConfig] = None
        self.mode = "result"
        self.zoom = 1.0
        self.pan = [0.0, 0.0]
        self.wipe = 0.5
        self._photo = None
        self._tf: Optional[Tuple[float, float, float, int, int]] = None
        self._drag: Optional[Tuple[int, int, float, float]] = None
        self._redraw_job: Optional[str] = None
        c = self.canvas
        c.bind("<Configure>", lambda e: self.request_redraw())
        c.bind("<MouseWheel>", self._on_wheel)
        c.bind("<ButtonPress-1>", self._on_press)
        c.bind("<B1-Motion>", self._on_drag)
        c.bind("<ButtonRelease-1>", lambda e: setattr(self, "_drag", None))
        c.bind("<Motion>", self._on_motion)
        c.bind("<Double-Button-1>", lambda e: self.reset_view())
        c.bind("<Leave>", lambda e: self.on_probe(""))

    # ---- state ----------------------------------------------------------------------
    def set_frames(self, result: Optional[np.ndarray], original: Optional[np.ndarray] = None,
                   layers: Optional[Tuple[np.ndarray, np.ndarray]] = None, grid: Optional[GridConfig] = None) -> None:
        shape_changed = (self.result is None or result is None or self.result.shape != result.shape)
        self.result, self.original, self.layers, self.grid = result, original, layers, grid
        if shape_changed:
            self.reset_view(redraw=False)
        self.request_redraw()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.request_redraw()

    def reset_view(self, redraw: bool = True) -> None:
        self.zoom, self.pan = 1.0, [0.0, 0.0]
        if redraw:
            self.request_redraw()

    def set_zoom_actual(self) -> None:
        if self._tf:
            _, _, sc, _, _ = self._tf
            fit = sc / self.zoom
            self.zoom = 1.0 / fit if fit > 0 else 1.0
            self.pan = [0.0, 0.0]
            self.request_redraw()

    # ---- composition ------------------------------------------------------------------
    def _compose(self) -> Optional[np.ndarray]:
        res = self.result
        if res is None:
            return None
        orig = self.original if (self.original is not None and self.original.shape == res.shape) else None
        if self.mode == "original" and orig is not None:
            return orig
        if self.mode == "wipe" and orig is not None:
            out = res.copy()
            col = int(round(res.shape[1] * self.wipe))
            out[:, :col] = orig[:, :col]
            return out
        if self.mode == "error" and orig is not None:
            err = np.abs(orig.astype(np.int16) - res.astype(np.int16)).mean(axis=2) * 10.0
            return _inferno_lut()[np.clip(err, 0, 255).astype(np.uint8)]
        if self.mode == "layers" and self.layers is not None and self.layers[0].shape == res.shape:
            return np.hstack(self.layers)
        return res

    def request_redraw(self) -> None:
        if self._redraw_job is None:
            self._redraw_job = self.after_idle(self._redraw)

    def _redraw(self) -> None:
        self._redraw_job = None
        c = self.canvas
        c.delete("all")
        arr = self._compose()
        cw, ch = max(c.winfo_width(), 2), max(c.winfo_height(), 2)
        if arr is None:
            c.create_text(cw / 2, ch / 2, text="Render an equation or convert an image to begin",
                          fill="#8b93a1", font=("Segoe UI", 13))
            self._tf = None
            return
        ih, iw = arr.shape[:2]
        fit = min(cw / iw, ch / ih)
        sc = fit * self.zoom
        ox = (cw - iw * sc) / 2 + self.pan[0]
        oy = (ch - ih * sc) / 2 + self.pan[1]
        self._tf = (ox, oy, sc, iw, ih)
        fx0, fy0 = max(0.0, -ox / sc), max(0.0, -oy / sc)
        fx1, fy1 = min(float(iw), (cw - ox) / sc), min(float(ih), (ch - oy) / sc)
        if fx1 - fx0 >= 1e-6 and fy1 - fy0 >= 1e-6:
            x0, y0 = int(np.floor(fx0)), int(np.floor(fy0))
            x1, y1 = min(iw, int(np.ceil(fx1))), min(ih, int(np.ceil(fy1)))
            crop = Image.fromarray(np.ascontiguousarray(arr[y0:y1, x0:x1]))
            size = (max(1, int(round((x1 - x0) * sc))), max(1, int(round((y1 - y0) * sc))))
            method = Image.Resampling.NEAREST if sc >= 3 else Image.Resampling.BILINEAR
            self._photo = ImageTk.PhotoImage(crop.resize(size, method))
            c.create_image(ox + x0 * sc, oy + y0 * sc, image=self._photo, anchor="nw")
        if self.mode == "wipe" and self.original is not None:
            wx = ox + self.wipe * iw * sc
            c.create_line(wx, oy, wx, oy + ih * sc, fill="#ffffff", width=2)
            c.create_oval(wx - 7, oy + ih * sc / 2 - 7, wx + 7, oy + ih * sc / 2 + 7, fill="#ffffff", outline="#444")
            c.create_text(ox + 8, oy + 8, text="ORIGINAL", anchor="nw", fill="#ffffff", font=("Segoe UI", 9, "bold"))
            c.create_text(ox + iw * sc - 8, oy + 8, text="EQUATION", anchor="ne", fill="#ffffff", font=("Segoe UI", 9, "bold"))
        c.create_text(cw - 8, ch - 6, text=f"{iw}x{ih}  {sc * 100:.0f}%", anchor="se", fill="#6f7785", font=("Consolas", 9))

    # ---- interaction ------------------------------------------------------------------
    def _to_image(self, cx: float, cy: float) -> Optional[Tuple[float, float]]:
        if not self._tf:
            return None
        ox, oy, sc, iw, ih = self._tf
        ix, iy = (cx - ox) / sc, (cy - oy) / sc
        return (ix, iy) if (0 <= ix < iw and 0 <= iy < ih) else None

    def _on_wheel(self, e) -> None:
        if not self._tf:
            return
        ox, oy, sc, iw, ih = self._tf
        factor = 1.18 if e.delta > 0 else 1 / 1.18
        new_zoom = float(np.clip(self.zoom * factor, 0.2, 80.0))
        factor = new_zoom / self.zoom
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        ix, iy = (e.x - ox) / sc, (e.y - oy) / sc
        sc2 = sc * factor
        self.pan[0] = e.x - ix * sc2 - (cw - iw * sc2) / 2
        self.pan[1] = e.y - iy * sc2 - (ch - ih * sc2) / 2
        self.zoom = new_zoom
        self.request_redraw()

    def _on_press(self, e) -> None:
        self._drag = (e.x, e.y, self.pan[0], self.pan[1])
        if self.mode == "wipe":
            self._move_wipe(e.x)

    def _move_wipe(self, cx: float) -> None:
        if self._tf:
            ox, _, sc, iw, _ = self._tf
            self.wipe = float(np.clip((cx - ox) / (iw * sc), 0.0, 1.0))
            self.request_redraw()

    def _on_drag(self, e) -> None:
        if self._drag is None:
            return
        if self.mode == "wipe":
            self._move_wipe(e.x)
            return
        x0, y0, px, py = self._drag
        self.pan = [px + e.x - x0, py + e.y - y0]
        self.request_redraw()

    def _on_motion(self, e) -> None:
        pos = self._to_image(e.x, e.y)
        if pos is None or self.result is None:
            self.on_probe("")
            return
        if self.mode == "layers":
            self.on_probe("")
            return
        ix, iy = int(pos[0]), int(pos[1])
        h, w = self.result.shape[:2]
        if not (0 <= ix < w and 0 <= iy < h):
            return
        m, n = ix + 1, iy + 1
        text = f"pixel (m={m}, n={n})"
        if self.grid is not None:
            x = (m - self.grid.m0) / self.grid.scale
            y = (self.grid.n0 - n) / self.grid.scale
            text += f"   (x={x:+.4f}, y={y:+.4f})"
        r, g, b = (int(v) for v in self.result[iy, ix])
        text += f"   rgb=({r},{g},{b})"
        if self.original is not None and self.original.shape == self.result.shape:
            o = self.original[iy, ix]
            text += f"   target=({int(o[0])},{int(o[1])},{int(o[2])})   |err|={abs(int(o[0]) - r) + abs(int(o[1]) - g) + abs(int(o[2]) - b)}"
        self.on_probe(text)


# ======================================================================================
# Application
# ======================================================================================
class VHRProStudioApp:
    DEBOUNCE_MS = 160
    ASPECTS = {"16:9": 16 / 9, "4:3": 4 / 3, "3:2": 3 / 2, "1:1": 1.0, "21:9": 21 / 9, "9:16": 9 / 16}
    QUALITY = {"Draft (180 px)": 180, "Standard (360 px)": 360, "Fine (540 px)": 540, "High (720 px)": 720}

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1620x980")
        self.root.minsize(1280, 800)

        # Set native application window and taskbar icon
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icon.ico")
        if os.path.exists(icon_path):
            try:
                self.root.iconbitmap(icon_path)
            except Exception:
                pass

        self.eq_runner = JobRunner("equation-renderer")
        self.img_runner = JobRunner("image-engine")

        self.context = "equation"           # which pipeline produced the current picture
        self.current_model: Optional[VHRModel] = None
        self.current_rgb: Optional[np.ndarray] = None
        self.current_latex: List[str] = []
        self.current_title: str = ""
        self.program: Optional[Program] = None
        self.analysis: Optional[ImageAnalysis] = None
        self.decomposer: Optional[PrecisionImageDecomposer] = None
        self.source_path: Optional[str] = None
        self.source_array: Optional[np.ndarray] = None
        self.source_preview: Optional[np.ndarray] = None

        self._eq_debounce: Optional[str] = None
        self._latex_debounce: Optional[str] = None
        self._k_debounce: Optional[str] = None
        self._slider_specs: List = []
        self._slider_sig: Tuple = ()
        self._animating = False
        self._anim_t0 = 0.0
        self._anim_factor = 1.0
        self._anim_frames: List[float] = []
        self._last_full_time = 0.0
        self._eq_inflight = False
        self._updating_slider_text = False
        self._eq_error_line: Optional[int] = None

        self._setup_theme()
        self._build_menu()
        self._build_layout()
        self._load_example("Plasma (animated)")
        self.root.after(20, self._pump)

    # ==================================================================================
    # UI construction
    # ==================================================================================
    def _setup_theme(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Title.TLabel", font=("Segoe UI", 13, "bold"))
        style.configure("Section.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Metric.TLabel", font=("Consolas", 10, "bold"), foreground="#0b5394")
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=6)
        style.configure("Err.TLabel", foreground="#b3261e", font=("Segoe UI", 9))
        style.configure("Ok.TLabel", foreground="#1b7a3d", font=("Segoe UI", 9))

    def _build_menu(self) -> None:
        bar = tk.Menu(self.root)
        file_menu = tk.Menu(bar, tearoff=0)
        file_menu.add_command(label="Open image...", command=self.browse_image)
        file_menu.add_command(label="Paste image from clipboard", command=self.paste_image)
        file_menu.add_command(label="Open equation program (.vhr / .txt)...", command=self.open_program)
        file_menu.add_command(label="Open model (.json)...", command=self.open_model_json)
        file_menu.add_separator()
        file_menu.add_command(label="Save picture (.png)...", command=self.export_png, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy)
        bar.add_cascade(label="File", menu=file_menu)
        help_menu = tk.Menu(bar, tearoff=0)
        help_menu.add_command(label="Equation syntax guide", command=self.show_syntax_help)
        bar.add_cascade(label="Help", menu=help_menu)
        self.root.config(menu=bar)
        self.root.bind("<Control-s>", lambda e: self.export_png())
        self.root.bind("<Control-Return>", lambda e: self.render_equation_now())
        self.root.bind("<F5>", lambda e: self.render_equation_now())

    def _build_layout(self) -> None:
        outer = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        outer.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        left = ttk.Frame(outer, width=470)
        outer.add(left, weight=0)
        hdr = ttk.Frame(left)
        hdr.pack(fill=tk.X, padx=8, pady=(2, 4))
        ttk.Label(hdr, text="VHR Maths Engine Pro", style="Title.TLabel").pack(anchor="w")
        ttk.Label(hdr, text="Created & Published by Vihanga Himantha", font=("Segoe UI", 8, "italic"), foreground="#444").pack(anchor="w")
        self.tabs = ttk.Notebook(left)
        self.tabs.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)
        self._build_tab_equation()
        self._build_tab_image()
        self._build_tab_catalog()
        self._build_tab_grid()
        self.tabs.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        card = ttk.LabelFrame(left, text="Precision & Performance Telemetry", padding=6)
        card.pack(fill=tk.X, padx=4, pady=3)
        self.telemetry_var = tk.StringVar(value="--")
        ttk.Label(card, textvariable=self.telemetry_var, style="Metric.TLabel", justify="left").pack(anchor="w")
        status = ttk.LabelFrame(left, text="Status", padding=6)
        status.pack(fill=tk.X, padx=4, pady=3)
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(status, textvariable=self.status_var, wraplength=430, justify="left").pack(anchor="w", pady=(0, 4))
        self.progress_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(status, variable=self.progress_var, maximum=100.0).pack(fill=tk.X)

        right = ttk.Frame(outer)
        outer.add(right, weight=1)
        bar = ttk.Frame(right, padding=(2, 2))
        bar.pack(fill=tk.X)
        ttk.Label(bar, text="View:", style="Section.TLabel").pack(side=tk.LEFT, padx=(2, 4))
        self.view_mode_var = tk.StringVar(value="result")
        for label, val in [("Result", "result"), ("Original", "original"), ("Wipe", "wipe"),
                           ("Error x10", "error"), ("Layers", "layers")]:
            ttk.Radiobutton(bar, text=label, variable=self.view_mode_var, value=val,
                            command=lambda: self.viewport.set_mode(self.view_mode_var.get())).pack(side=tk.LEFT, padx=3)
        ttk.Button(bar, text="Fit", width=4, command=lambda: self.viewport.reset_view()).pack(side=tk.LEFT, padx=(12, 2))
        ttk.Button(bar, text="1:1", width=4, command=lambda: self.viewport.set_zoom_actual()).pack(side=tk.LEFT, padx=2)
        self.probe_var = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.probe_var, font=("Consolas", 9)).pack(side=tk.RIGHT, padx=6)

        exp = ttk.Frame(right, padding=(2, 2))
        exp.pack(fill=tk.X)
        for label, cmd in [("Save PNG", self.export_png), ("Plate PNG/PDF", self.export_plate),
                           ("LaTeX .tex", self.export_latex), ("Python .py", self.export_python_script),
                           ("Program .vhr", self.export_program), ("Model .json", self.export_json)]:
            ttk.Button(exp, text=label, command=cmd).pack(side=tk.LEFT, padx=2)

        vsplit = ttk.PanedWindow(right, orient=tk.VERTICAL)
        vsplit.pack(fill=tk.BOTH, expand=True, pady=2)
        vp_frame = ttk.LabelFrame(vsplit, text="Interactive Mathematical Viewport (wheel = zoom, drag = pan, double-click = fit)", padding=2)
        self.viewport = ImageViewport(vp_frame, on_probe=self.probe_var.set)
        self.viewport.pack(fill=tk.BOTH, expand=True)
        vsplit.add(vp_frame, weight=5)

        eq_frame = ttk.LabelFrame(vsplit, text="Equation", padding=2)
        vsplit.add(eq_frame, weight=2)
        eq_tabs = ttk.Notebook(eq_frame)
        eq_tabs.pack(fill=tk.BOTH, expand=True)
        typeset = ttk.Frame(eq_tabs)
        eq_tabs.add(typeset, text="Typeset")
        self.fig = Figure(figsize=(10, 2.4), dpi=100, facecolor="white")
        self.ax_eq = self.fig.add_axes([0, 0, 1, 1])
        self.ax_eq.axis("off")
        self.eq_canvas = FigureCanvasTkAgg(self.fig, master=typeset)
        self.eq_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.eq_canvas.get_tk_widget().bind("<Configure>", lambda e: self._schedule_latex_refresh(60))
        src = ttk.Frame(eq_tabs)
        eq_tabs.add(src, text="LaTeX source")
        self.latex_box = tk.Text(src, height=6, font=("Consolas", 9), wrap=tk.NONE)
        self.latex_box.pack(fill=tk.BOTH, expand=True)

    # ---- tab 1: equation -> image -----------------------------------------------------
    def _build_tab_equation(self) -> None:
        tab = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(tab, text="1. Equation -> Image (live)")

        row = ttk.Frame(tab)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Example:", style="Section.TLabel").pack(side=tk.LEFT)
        self.example_var = tk.StringVar(value="Plasma (animated)")
        combo = ttk.Combobox(row, textvariable=self.example_var, values=list(EXAMPLES), state="readonly", width=30)
        combo.pack(side=tk.LEFT, padx=6)
        combo.bind("<<ComboboxSelected>>", lambda e: self._load_example(self.example_var.get()))
        ttk.Button(row, text="Syntax", width=7, command=self.show_syntax_help).pack(side=tk.RIGHT)

        ed = ttk.Frame(tab)
        ed.pack(fill=tk.BOTH, expand=True, pady=(6, 2))
        self.editor = tk.Text(ed, height=14, font=("Consolas", 11), wrap=tk.NONE, undo=True, background="#fbfbfd",
                              insertbackground="#222", padx=6, pady=4)
        ys = ttk.Scrollbar(ed, orient=tk.VERTICAL, command=self.editor.yview)
        xs = ttk.Scrollbar(ed, orient=tk.HORIZONTAL, command=self.editor.xview)
        self.editor.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        ys.pack(side=tk.RIGHT, fill=tk.Y)
        xs.pack(side=tk.BOTTOM, fill=tk.X)
        self.editor.pack(fill=tk.BOTH, expand=True)
        self.editor.tag_configure("err", background="#ffd9d6")
        for seq in ("<KeyRelease>", "<<Paste>>", "<<Cut>>", "<<Undo>>", "<<Redo>>"):
            self.editor.bind(seq, lambda e: self._on_source_edit())

        self.eq_msg_var = tk.StringVar(value="")
        self.eq_msg = ttk.Label(tab, textvariable=self.eq_msg_var, style="Ok.TLabel", wraplength=430, justify="left")
        self.eq_msg.pack(anchor="w", pady=2)

        self.slider_frame = ttk.LabelFrame(tab, text="Live parameters  (add  `a = 3  # [0, 10]`  to create more)", padding=4)
        self.slider_frame.pack(fill=tk.X, pady=3)

        opts = ttk.Frame(tab)
        opts.pack(fill=tk.X, pady=2)
        ttk.Label(opts, text="Quality:").grid(row=0, column=0, sticky="w")
        self.quality_var = tk.StringVar(value="Standard (360 px)")
        qc = ttk.Combobox(opts, textvariable=self.quality_var, values=list(self.QUALITY), state="readonly", width=16)
        qc.grid(row=0, column=1, padx=4)
        qc.bind("<<ComboboxSelected>>", lambda e: self.render_equation_now())
        ttk.Label(opts, text="Aspect:").grid(row=0, column=2, padx=(8, 0))
        self.aspect_var = tk.StringVar(value="16:9")
        ac = ttk.Combobox(opts, textvariable=self.aspect_var, values=list(self.ASPECTS), state="readonly", width=6)
        ac.grid(row=0, column=3, padx=4)
        ac.bind("<<ComboboxSelected>>", lambda e: self.render_equation_now())

        anim = ttk.Frame(tab)
        anim.pack(fill=tk.X, pady=2)
        self.anim_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(anim, text="Animate  (variable t, adaptive resolution)", variable=self.anim_var,
                        command=self._toggle_animation).pack(side=tk.LEFT)
        self.fps_var = tk.StringVar(value="")
        ttk.Label(anim, textvariable=self.fps_var, font=("Consolas", 9)).pack(side=tk.RIGHT)
        sp = ttk.Frame(tab)
        sp.pack(fill=tk.X)
        ttk.Label(sp, text="Speed:").pack(side=tk.LEFT)
        self.anim_speed_var = tk.DoubleVar(value=1.0)
        ttk.Scale(sp, from_=0.1, to=4.0, variable=self.anim_speed_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Label(sp, text="t =").pack(side=tk.LEFT)
        self.t_var = tk.DoubleVar(value=0.0)
        ttk.Scale(sp, from_=0.0, to=20.0, variable=self.t_var, command=lambda v: self._on_t_slider()).pack(side=tk.LEFT, padx=4)

        btns = ttk.Frame(tab)
        btns.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(btns, text="Render now  (F5)", style="Primary.TButton", command=self.render_equation_now).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(btns, text="Open program...", command=self.open_program).pack(side=tk.LEFT, padx=3)
        ttk.Button(btns, text="Save high-res...", command=self.export_hires).pack(side=tk.LEFT)

    # ---- tab 2: image -> equation -----------------------------------------------------
    def _build_tab_image(self) -> None:
        tab = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(tab, text="2. Image -> Equation")
        r = ttk.Frame(tab)
        r.pack(fill=tk.X)
        ttk.Button(r, text="Load image...", command=self.browse_image).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(r, text="Paste from clipboard", command=self.paste_image).pack(side=tk.LEFT, padx=(4, 0))
        self.path_label = ttk.Label(tab, text="No image loaded.", foreground="gray", wraplength=430)
        self.path_label.pack(anchor="w", pady=(4, 4))

        grid_row = ttk.Frame(tab)
        grid_row.pack(fill=tk.X)
        ttk.Label(grid_row, text="Working size (longest side):").pack(side=tk.LEFT)
        self.max_side_var = tk.StringVar(value="480")
        ttk.Combobox(grid_row, textvariable=self.max_side_var, values=["160", "240", "320", "480", "640", "800", "1024", "1600", "2000"],
                     width=7).pack(side=tk.LEFT, padx=6)
        self.manual_grid_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tab, text="Use the Grid tab values instead (may crop to its aspect ratio)",
                        variable=self.manual_grid_var).pack(anchor="w", pady=2)
        ttk.Separator(tab).pack(fill=tk.X, pady=6)

        self.target_psnr_var = tk.DoubleVar(value=38.0)
        self._add_slider(tab, "Target PSNR (dB):", self.target_psnr_var, 20.0, 60.0)
        self.max_trig_var = tk.IntVar(value=4000)
        self._add_slider(tab, "Max cosine terms (K cap):", self.max_trig_var, 50, 40000, is_int=True)
        self.prim_cnt_var = tk.IntVar(value=8)
        self._add_slider(tab, "Primitives (P, kept only if they help):", self.prim_cnt_var, 0, 40, is_int=True)
        self.refine_var = tk.IntVar(value=3)
        self._add_slider(tab, "Saturation-aware refinement passes:", self.refine_var, 0, 6, is_int=True)
        self.opt_lsq_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tab, text="Sub-pixel Levenberg-Marquardt primitive refinement", variable=self.opt_lsq_var).pack(anchor="w", pady=2)

        ttk.Button(tab, text="Convert image to equation", style="Primary.TButton", command=self.run_conversion).pack(fill=tk.X, pady=(8, 4))

        live = ttk.LabelFrame(tab, text="Live precision control (updates in milliseconds)", padding=6)
        live.pack(fill=tk.X, pady=4)
        self.k_label_var = tk.StringVar(value="Convert an image first.")
        ttk.Label(live, textvariable=self.k_label_var, font=("Consolas", 10, "bold")).pack(anchor="w")
        self.k_pos_var = tk.DoubleVar(value=0.0)
        self.k_scale = ttk.Scale(live, from_=0.0, to=1000.0, variable=self.k_pos_var, command=self._on_k_slider, state="disabled")
        self.k_scale.pack(fill=tk.X, pady=4)
        self.rd_canvas = tk.Canvas(live, height=120, bg="#fbfbfd", highlightthickness=1, highlightbackground="#c5c9d3")
        self.rd_canvas.pack(fill=tk.X)
        self.rd_canvas.bind("<Configure>", lambda e: self._draw_rd_curve())
        ttk.Button(live, text="Send equation to the live editor (tab 1)", command=self.send_to_editor).pack(fill=tk.X, pady=(6, 0))
        ttk.Button(live, text="Verify: render exported equation & compare", command=self.verify_roundtrip).pack(fill=tk.X, pady=(3, 0))

    # ---- tab 3: catalog ---------------------------------------------------------------
    def _build_tab_catalog(self) -> None:
        tab = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(tab, text="3. Cosmic Catalog")
        ttk.Label(tab, text="Cosmic reference plates (procedural, not fitted):", style="Section.TLabel").pack(anchor="w", pady=(0, 6))
        self.catalog_var = tk.StringVar(value="colliding_planets")
        ttk.Radiobutton(tab, text="Colliding Planets (H_v, U, K_v, R_s, Z_s, M, N)", variable=self.catalog_var,
                        value="colliding_planets").pack(anchor="w", pady=2)
        ttk.Radiobutton(tab, text="Galaxy Cluster (H_s, A_{r,p}, U_s, V_s, C_s, E_s)", variable=self.catalog_var,
                        value="galaxy_cluster").pack(anchor="w", pady=2)
        ttk.Separator(tab).pack(fill=tk.X, pady=10)
        ttk.Label(tab, text="Summation / Product Upper Bounds:", style="Section.TLabel").pack(anchor="w", pady=(0, 4))
        self.gal_p_var = tk.IntVar(value=60)
        self._add_slider(tab, "Galaxy Product Bound (p <= 300):", self.gal_p_var, 10, 300, is_int=True)
        self.star_k_var = tk.IntVar(value=80)
        self._add_slider(tab, "Starfield Sum Bound (E_s, k <= 300):", self.star_k_var, 20, 300, is_int=True)
        self.planet_s_var = tk.IntVar(value=30)
        self._add_slider(tab, "Planet Turbulence & Wave Bound (s <= 50):", self.planet_s_var, 10, 50, is_int=True)
        ttk.Button(tab, text="Evaluate formula", style="Primary.TButton", command=self.run_exact_catalog).pack(fill=tk.X, pady=(16, 4))
        ttk.Label(tab, text="Catalog plates report no PSNR (there is no reference image to compare against).",
                  foreground="gray", wraplength=420).pack(anchor="w", pady=6)

    # ---- tab 4: grid -------------------------------------------------------------------
    def _build_tab_grid(self) -> None:
        tab = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(tab, text="4. Grid & Fractions")
        ttk.Label(tab, text="Coordinate Grid Geometry Presets:", style="Section.TLabel").pack(anchor="w", pady=(0, 6))
        self.grid_preset_var = tk.StringVar(value="preview")
        for txt, val in [
            ("Interactive Preview (fast, formula-specific aspect)", "preview"),
            ("Exact Galaxy Cluster Grid (2000 x 1200, m0=1080, n0=401, S=680)", "galaxy"),
            ("Exact Colliding Planets Grid (2100 x 1200, m0=970, n0=601, S=652)", "planets"),
        ]:
            ttk.Radiobutton(tab, text=txt, variable=self.grid_preset_var, value=val, command=self._apply_grid_preset).pack(anchor="w", pady=3)
        ttk.Separator(tab).pack(fill=tk.X, pady=10)
        g = GridConfig.preset_fast_preview()
        self.w_var, self.h_var = tk.IntVar(value=g.width), tk.IntVar(value=g.height)
        self.m0_var, self.n0_var = tk.IntVar(value=g.m0), tk.IntVar(value=g.n0)
        self.scale_var = tk.DoubleVar(value=g.scale)
        self.max_den_var = tk.IntVar(value=1000)
        for lbl, var in [("Width (M):", self.w_var), ("Height (N):", self.h_var), ("Origin Column (m0):", self.m0_var),
                         ("Origin Row (n0):", self.n0_var), ("Scale Divisor (S):", self.scale_var),
                         ("Max LaTeX Fraction Denominator (q):", self.max_den_var)]:
            row = ttk.Frame(tab)
            row.pack(fill=tk.X, pady=3)
            ttk.Label(row, text=lbl).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var, width=10).pack(side=tk.RIGHT)

    def _add_slider(self, parent, text, var, low, high, is_int=False) -> None:
        hdr = ttk.Frame(parent)
        hdr.pack(fill=tk.X, pady=1)
        val_lbl = ttk.Label(hdr, text=f"{int(var.get())}" if is_int else f"{var.get():.1f}", width=7, anchor="e")
        val_lbl.pack(side=tk.RIGHT)
        ttk.Label(hdr, text=text).pack(side=tk.LEFT)

        def _on_slide(v):
            f = float(v)
            if is_int:
                var.set(int(round(f)))
                val_lbl.config(text=f"{int(round(f))}")
            else:
                var.set(round(f, 1))
                val_lbl.config(text=f"{f:.1f}")

        ttk.Scale(parent, from_=low, to=high, variable=var, command=_on_slide).pack(fill=tk.X, pady=(0, 4))

    # ==================================================================================
    # Event pump (UI thread)
    # ==================================================================================
    def _pump(self) -> None:
        try:
            for runner in (self.eq_runner, self.img_runner):
                while True:
                    try:
                        tag, kind, payload, token = runner.events.get_nowait()
                    except queue.Empty:
                        break
                    if token.is_set() and kind != "error":
                        continue                     # superseded by a newer request
                    self._handle_event(tag, kind, payload)
        except Exception:
            traceback.print_exc()
        finally:
            self.root.after(15, self._pump)

    def _handle_event(self, tag: str, kind: str, payload) -> None:
        if kind == "progress":
            frac, msg = payload
            self.progress_var.set(frac * 100.0)
            self.status_var.set(msg)
        elif kind == "error":
            exc, tb = payload
            if tag.startswith("eq"):
                self._eq_inflight = False
                self._set_eq_error(str(exc) if isinstance(exc, ExpressionError) else f"{type(exc).__name__}: {exc}",
                                   getattr(exc, "line", None))
                self._animating = False
                self.anim_var.set(False)
            else:
                print(tb, file=sys.stderr)
                self.status_var.set(f"Error: {exc}")
                messagebox.showerror("Engine Error", f"{type(exc).__name__}: {exc}")
        elif tag == "eq" and kind == "frame":
            self._on_eq_frame(payload)
        elif tag == "eq" and kind == "done":
            self._eq_inflight = False
            if self._animating:
                self.root.after(1, self._animate_step)
        elif tag == "eq-hires" and kind == "saved":
            self.status_var.set(f"Saved high-resolution render: {payload}")
            self.progress_var.set(100.0)
        elif tag == "convert" and kind == "analysis":
            self._on_analysis(payload)
        elif tag in ("convert", "k") and kind == "model":
            self._on_model(*payload)
        elif tag == "catalog" and kind == "plate":
            model, rgb = payload
            self.context = "catalog"
            self.analysis = None
            self.current_model, self.current_rgb = model, rgb
            self.current_latex, self.current_title = model.latex_lines, model.title
            self.viewport.set_frames(rgb, None, None, model.grid)
            self.telemetry_var.set(f"{model.title}\nTerms: {model.metrics.total_terms}   Time: {model.metrics.render_time_sec:.2f} s\nPSNR/SSIM: n/a (procedural plate)")
            self._refresh_latex_panel()
            self.status_var.set("Catalog plate rendered.")
            self.progress_var.set(100.0)
        elif tag == "verify" and kind == "report":
            self.status_var.set(payload)
            messagebox.showinfo("Round-trip verification", payload)

    def _on_tab_changed(self, _e=None) -> None:
        idx = self.tabs.index(self.tabs.select())
        if idx == 0:
            self.context = "equation"
            self.render_equation_now()
        else:
            self._stop_animation()

    # ==================================================================================
    # Tab 1 - live equation
    # ==================================================================================
    def _load_example(self, name: str) -> None:
        self.editor.delete("1.0", tk.END)
        self.editor.insert("1.0", EXAMPLES[name])
        self.t_var.set(0.0)
        self._on_source_edit(immediate=True)

    def _on_source_edit(self, immediate: bool = False) -> None:
        if self._eq_debounce:
            self.root.after_cancel(self._eq_debounce)
        self._eq_debounce = self.root.after(1 if immediate else self.DEBOUNCE_MS, self.render_equation_now)

    def _set_eq_error(self, message: str, line: Optional[int]) -> None:
        self.eq_msg.configure(style="Err.TLabel")
        self.eq_msg_var.set(("Line %d: " % line if line else "") + message.split(": ", 1)[-1] if message.startswith("line ") else message)
        self.editor.tag_remove("err", "1.0", tk.END)
        if line:
            self.editor.tag_add("err", f"{line}.0", f"{line}.end+1c")

    def _clear_eq_error(self, text: str) -> None:
        self.eq_msg.configure(style="Ok.TLabel")
        self.eq_msg_var.set(text)
        self.editor.tag_remove("err", "1.0", tk.END)

    def _eq_base_grid(self, prog: Program) -> GridConfig:
        height = self.QUALITY[self.quality_var.get()]
        if prog.grid_hint:
            W, H, m0, n0, S = prog.grid_hint
            return GridConfig(width=W, height=H, m0=m0, n0=n0, scale=S).scaled(height / H)
        h = height
        w = max(2, int(round(h * self.ASPECTS[self.aspect_var.get()])))
        return GridConfig(width=w, height=h, m0=w / 2.0, n0=h / 2.0, scale=h / 2.0)

    def render_equation_now(self) -> None:
        if self._eq_debounce:
            try:
                self.root.after_cancel(self._eq_debounce)
            except Exception:
                pass
            self._eq_debounce = None
        if self.tabs.index(self.tabs.select()) != 0 and not self._animating:
            return
        source = self.editor.get("1.0", "end-1c")
        try:
            prog = compile_program(source)
        except ExpressionError as exc:
            self._set_eq_error(exc.message, exc.line)
            return
        self.program = prog
        self.context = "equation"
        self._clear_eq_error("Equation OK" + (" - uses time `t`: tick Animate" if prog.uses_time else ""))
        self._sync_sliders(source)
        self._schedule_latex_refresh(300)
        if self._animating:
            return                                  # the animation loop drives rendering
        self._submit_eq_render(prog, t=float(self.t_var.get()), progressive=True)

    def _submit_eq_render(self, prog: Program, t: float, progressive: bool, factor: float = 1.0) -> None:
        base = self._eq_base_grid(prog)
        if progressive and self._last_full_time > 0.08:
            stages = (0.25, 1.0)
        else:
            stages = (factor,)

        def job(cancel, emit):
            for f in stages:
                g = base.scaled(f)
                ctx = RenderContext(g.width, g.height, g.m0, g.n0, g.scale, t)
                t0 = time.perf_counter()
                rgb = prog.render(ctx, cancel=cancel)
                emit("frame", (rgb, g, time.perf_counter() - t0, f, t))

        self._eq_inflight = True
        self.eq_runner.submit(job, "eq")

    def _on_eq_frame(self, payload) -> None:
        rgb, g, dt, factor, t = payload
        self.current_rgb = rgb
        self.viewport.set_frames(rgb, None, None, g)
        if factor >= 0.99:
            self._last_full_time = dt
        self.telemetry_var.set(
            f"Equation render: {g.width}x{g.height}   {dt * 1000:6.1f} ms  ({1.0 / max(dt, 1e-6):5.1f} fps)\n"
            f"t = {t:6.2f} s    program lines: {len(self.program.statements) if self.program else 0}\n"
            f"PSNR/SSIM: n/a (no reference image)")
        self.current_title = "Equation program"
        if self._animating:
            now = time.perf_counter()
            self._anim_frames = [x for x in self._anim_frames if now - x < 1.0] + [now]
            self.fps_var.set(f"{len(self._anim_frames)} fps  @ {int(self._anim_factor * 100)}% res")
            if dt > 0.06:
                self._anim_factor = max(0.25, self._anim_factor * 0.85)
            elif dt < 0.025:
                self._anim_factor = min(1.0, self._anim_factor * 1.12)

    def _sync_sliders(self, source: str) -> None:
        specs = find_sliders(source)
        sig = tuple((s.name, s.lo, s.hi, s.line_index) for s in specs)
        self._slider_specs = specs
        if sig == self._slider_sig:
            return
        self._slider_sig = sig
        for child in self.slider_frame.winfo_children():
            child.destroy()
        if not specs:
            ttk.Label(self.slider_frame, text="(none)", foreground="gray").pack(anchor="w")
            return
        for spec in specs[:12]:
            row = ttk.Frame(self.slider_frame)
            row.pack(fill=tk.X)
            ttk.Label(row, text=spec.name, width=10).pack(side=tk.LEFT)
            val_lbl = ttk.Label(row, text=f"{spec.value:g}", width=8, anchor="e")
            val_lbl.pack(side=tk.RIGHT)
            var = tk.DoubleVar(value=spec.value)

            def on_move(v, spec=spec, lbl=val_lbl):
                value = float(v)
                lbl.config(text=f"{value:.4g}")
                self._write_slider_value(spec.line_index, value)

            ttk.Scale(row, from_=spec.lo, to=spec.hi, variable=var, command=on_move).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

    def _write_slider_value(self, line_index: int, value: float) -> None:
        lineno = line_index + 1
        current = self.editor.get(f"{lineno}.0", f"{lineno}.end")
        updated = set_slider_value(current, 0, value)
        if updated != current:
            self.editor.delete(f"{lineno}.0", f"{lineno}.end")
            self.editor.insert(f"{lineno}.0", updated)
        self._on_source_edit(immediate=not self._animating)

    # ---- animation ----------------------------------------------------------------------
    def _toggle_animation(self) -> None:
        if self.anim_var.get():
            if self.program is None:
                self.render_equation_now()
            self._animating = True
            self._anim_t0 = time.perf_counter() - self.t_var.get() / max(self.anim_speed_var.get(), 1e-3)
            self._anim_factor = 1.0
            self._anim_frames = []
            self._animate_step()
        else:
            self._stop_animation()
            self.render_equation_now()

    def _stop_animation(self) -> None:
        self._animating = False
        self.anim_var.set(False)
        self.fps_var.set("")

    def _animate_step(self) -> None:
        if not self._animating or self._eq_inflight:
            return
        source = self.editor.get("1.0", "end-1c")
        try:
            prog = compile_program(source)
        except ExpressionError as exc:
            self._set_eq_error(exc.message, exc.line)
            self.root.after(120, self._animate_step)
            return
        self.program = prog
        self._clear_eq_error("Animating...")
        t = (time.perf_counter() - self._anim_t0) * self.anim_speed_var.get()
        self.t_var.set(t % 20.0)
        self._submit_eq_render(prog, t=t, progressive=False, factor=self._anim_factor)

    def _on_t_slider(self) -> None:
        if not self._animating:
            self._on_source_edit(immediate=True)

    # ---- typeset ------------------------------------------------------------------------
    def _schedule_latex_refresh(self, delay: int = 250) -> None:
        if self._latex_debounce:
            self.root.after_cancel(self._latex_debounce)
        self._latex_debounce = self.root.after(delay, self._refresh_latex_panel)

    def _refresh_latex_panel(self) -> None:
        self._latex_debounce = None
        if self.context == "equation" and self.program is not None:
            try:
                self.current_latex = to_latex(self.program)
            except Exception:
                self.current_latex = []
            self.current_title = "Equation program"
        lines, title = self.current_latex, self.current_title
        self.latex_box.delete("1.0", tk.END)
        for line in lines:
            self.latex_box.insert(tk.END, line + "\n\n")
        try:
            self.eq_canvas.draw()                        # make sure a renderer / real size exists
            draw_equation_lines(self.ax_eq, lines, title if self.context != "equation" else "", base_fs=10.5)
            self.eq_canvas.draw_idle()
        except Exception:
            traceback.print_exc()

    # ==================================================================================
    # Tab 2 - image -> equation
    # ==================================================================================
    def browse_image(self) -> None:
        path = filedialog.askopenfilename(title="Select image to convert to an equation",
                                          filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff *.gif")])
        if path:
            self._set_source(path=path)

    def paste_image(self) -> None:
        try:
            data = ImageGrab.grabclipboard()
        except Exception as exc:
            messagebox.showwarning("Clipboard", f"Could not read the clipboard: {exc}")
            return
        if isinstance(data, Image.Image):
            self._set_source(array=np.asarray(data.convert("RGB")), label="clipboard image")
        elif isinstance(data, list) and data:
            self._set_source(path=str(data[0]))
        else:
            messagebox.showinfo("Clipboard", "There is no image on the clipboard.")

    def _set_source(self, path: Optional[str] = None, array: Optional[np.ndarray] = None, label: Optional[str] = None) -> None:
        try:
            probe = PrecisionImageDecomposer(image_path=path, image=array)
            img = probe._open_source()
        except Exception as exc:
            messagebox.showerror("Cannot open image", f"{type(exc).__name__}: {exc}")
            return
        self.source_path, self.source_array = path, array
        self.path_label.config(text=f"{label or path}   ({img.size[0]} x {img.size[1]})", foreground="black")
        preview = img if max(img.size) <= 1800 else img.resize(
            (int(img.size[0] * 1800 / max(img.size)), int(img.size[1] * 1800 / max(img.size))), Image.Resampling.LANCZOS)
        self.source_preview = np.asarray(preview.convert("RGB"))
        self.tabs.select(1)
        self.context = "image"
        self.viewport.set_frames(self.source_preview, None, None, None)
        self.view_mode_var.set("result")
        self.viewport.set_mode("result")
        self.status_var.set("Image loaded. Press 'Convert image to equation'.")

    def _read_grid(self) -> GridConfig:
        return GridConfig(width=max(2, int(self.w_var.get())), height=max(2, int(self.h_var.get())),
                          m0=int(self.m0_var.get()), n0=int(self.n0_var.get()),
                          scale=max(1.0, float(self.scale_var.get())), chunk_rows=120).validate()

    def run_conversion(self) -> None:
        if not self.source_path and self.source_array is None:
            messagebox.showwarning("Image required", "Load or paste an image first.")
            return
        try:
            dec = PrecisionImageDecomposer(
                image_path=self.source_path, image=self.source_array,
                num_primitives=int(self.prim_cnt_var.get()), max_trig_terms=int(self.max_trig_var.get()),
                target_psnr_db=float(self.target_psnr_var.get()), optimize_primitives=bool(self.opt_lsq_var.get()),
                max_fraction_den=int(self.max_den_var.get()), refine_iters=int(self.refine_var.get()))
            grid = self._read_grid() if self.manual_grid_var.get() else dec.default_grid(int(self.max_side_var.get()))
        except Exception as exc:
            messagebox.showerror("Invalid settings", f"{type(exc).__name__}: {exc}")
            return
        self._stop_animation()
        self.decomposer = dec
        self.analysis = None
        self.k_scale.state(["disabled"])
        self.status_var.set("Converting...")

        def job(cancel, emit):
            t0 = time.perf_counter()
            analysis = dec.analyze(grid, progress_cb=lambda f, m: emit("progress", (f, m)), cancel=cancel)
            emit("analysis", analysis)
            model, recon = dec.build_model(analysis)
            model.metrics.render_time_sec = round(time.perf_counter() - t0, 2)
            emit("model", (model, recon, analysis.k_default))
            emit("progress", (1.0, f"Done. PSNR {model.metrics.psnr_db} dB | SSIM {model.metrics.ssim} | {model.metrics.total_terms} terms"))

        self.img_runner.submit(job, "convert")

    def _k_from_pos(self, pos: float) -> int:
        n = max(2, self.analysis.total_coefficients if self.analysis else 2)
        return int(np.clip(round(np.exp(np.log(1.0) + (np.log(n) - np.log(1.0)) * pos / 1000.0)), 1, n))

    def _pos_from_k(self, K: int) -> float:
        n = max(2, self.analysis.total_coefficients if self.analysis else 2)
        return float(1000.0 * np.log(max(K, 1)) / np.log(n))

    def _on_analysis(self, analysis: ImageAnalysis) -> None:
        self.analysis = analysis
        self.k_scale.state(["!disabled"])
        self.k_pos_var.set(self._pos_from_k(analysis.k_default))
        self._update_k_label(analysis.k_default)
        self._draw_rd_curve()

    def _update_k_label(self, K: int) -> None:
        if self.analysis:
            n = self.analysis.total_coefficients
            self.k_label_var.set(f"K = {K:,} of {n:,} terms ({100.0 * K / n:.2f}%)   est. {self.analysis.est_psnr(K):.1f} dB")

    def _on_k_slider(self, _v=None) -> None:
        if not self.analysis:
            return
        K = self._k_from_pos(self.k_pos_var.get())
        self._update_k_label(K)
        self._draw_rd_curve(K)
        if self._k_debounce:
            self.root.after_cancel(self._k_debounce)
        self._k_debounce = self.root.after(70, lambda: self._submit_k(K))

    def _submit_k(self, K: int) -> None:
        self._k_debounce = None
        analysis, dec = self.analysis, self.decomposer
        if analysis is None or dec is None:
            return

        def job(cancel, emit):
            model, recon = dec.build_model(analysis, K)
            emit("model", (model, recon, K))

        self.img_runner.submit(job, "k")

    def _on_model(self, model: VHRModel, recon: np.ndarray, K: int) -> None:
        self.context = "image"
        self.current_model, self.current_rgb = model, recon
        self.current_latex, self.current_title = model.latex_lines, model.title
        a = self.analysis
        layers = None
        if a is not None:
            prim_u8 = np.clip(a.prim_layer * 255.0, 0, 255).astype(np.uint8)
            diff = np.clip(recon.astype(np.int16) - prim_u8.astype(np.int16) + 128, 0, 255).astype(np.uint8)
            layers = (prim_u8, diff)
        self.viewport.set_frames(recon, a.target_rgb if a else None, layers, model.grid)
        m = model.metrics
        self.telemetry_var.set(
            f"PSNR:  {m.psnr_db:6.2f} dB | SSIM:   {m.ssim:.4f}\n"
            f"MAE:   {m.mae_255:6.2f} /255 | MaxErr: {m.max_err_255:5.1f} /255\n"
            f"Terms: {m.total_terms:6d}    | Time:   {m.render_time_sec:5.2f} s   Grid {model.grid.width}x{model.grid.height}")
        self._refresh_latex_panel()
        if a is not None:
            self._draw_rd_curve(K, measured=m.psnr_db)
        for note in model.extra_params.get("notes", []):
            self.status_var.set(note)

    def _draw_rd_curve(self, K: Optional[int] = None, measured: Optional[float] = None) -> None:
        c = self.rd_canvas
        c.delete("all")
        w, h = max(c.winfo_width(), 50), max(c.winfo_height(), 40)
        a = self.analysis
        if a is None:
            c.create_text(w / 2, h / 2, text="rate / distortion curve appears here", fill="#8b93a1")
            return
        ks, ps = a.curve(100)
        pad_l, pad_b, pad_t, pad_r = 34, 18, 8, 8
        lo, hi = max(0.0, float(ps.min()) - 2), min(100.0, float(ps.max()) + 2)
        lk0, lk1 = np.log(ks[0]), np.log(ks[-1]) if ks[-1] > ks[0] else np.log(ks[0]) + 1
        px = lambda k: pad_l + (np.log(k) - lk0) / (lk1 - lk0) * (w - pad_l - pad_r)
        py = lambda p: pad_t + (1 - (p - lo) / max(hi - lo, 1e-9)) * (h - pad_t - pad_b)
        c.create_line(pad_l, pad_t, pad_l, h - pad_b, fill="#9aa1ad")
        c.create_line(pad_l, h - pad_b, w - pad_r, h - pad_b, fill="#9aa1ad")
        c.create_text(pad_l - 4, pad_t, text=f"{hi:.0f}", anchor="e", font=("Consolas", 8), fill="#555")
        c.create_text(pad_l - 4, h - pad_b, text=f"{lo:.0f}", anchor="e", font=("Consolas", 8), fill="#555")
        c.create_text(w - pad_r, h - 4, text="terms K (log)  ->", anchor="se", font=("Consolas", 8), fill="#555")
        c.create_text(pad_l + 3, pad_t, text="dB", anchor="nw", font=("Consolas", 8), fill="#555")
        target = float(self.target_psnr_var.get())
        if lo < target < hi:
            c.create_line(pad_l, py(target), w - pad_r, py(target), fill="#c9a227", dash=(3, 3))
        pts = [coord for k, p in zip(ks, ps) for coord in (px(k), py(p))]
        if len(pts) >= 4:
            c.create_line(*pts, fill="#0b5394", width=2, smooth=True)
        kk = K if K is not None else a.k_default
        c.create_line(px(kk), pad_t, px(kk), h - pad_b, fill="#b3261e")
        c.create_oval(px(kk) - 3, py(a.est_psnr(kk)) - 3, px(kk) + 3, py(a.est_psnr(kk)) + 3, fill="#b3261e", outline="")
        if measured is not None:
            c.create_oval(px(kk) - 3, py(min(measured, hi)) - 3, px(kk) + 3, py(min(measured, hi)) + 3, outline="#1b7a3d", width=2)

    def send_to_editor(self) -> None:
        if self.current_model is None or self.context != "image":
            messagebox.showinfo("Nothing to send", "Convert an image first.")
            return
        try:
            text = VHRExporterSuite.build_vhr_program(self.current_model)
        except ExportError as exc:
            messagebox.showwarning("Cannot send", str(exc))
            return
        self.tabs.select(0)
        self.editor.delete("1.0", tk.END)
        self.editor.insert("1.0", text)
        self.status_var.set("Equation loaded into the live editor - edit it and watch the image change.")
        self._on_source_edit(immediate=True)

    def verify_roundtrip(self) -> None:
        model = self.current_model
        if model is None or self.context != "image":
            messagebox.showinfo("Nothing to verify", "Convert an image first.")
            return
        recon = self.current_rgb

        def job(cancel, emit):
            emit("progress", (0.1, "Verifying: rendering the exported equation with the independent expression engine..."))
            prog = compile_program(VHRExporterSuite.build_vhr_program(model))
            g = model.grid
            ctx = RenderContext(g.width, g.height, g.m0, g.n0, g.scale)
            out = prog.render(ctx, cancel=cancel)
            diff = np.abs(out.astype(np.int16) - recon.astype(np.int16))
            emit("progress", (1.0, "Verification complete."))
            emit("report", f"Exported equation vs. in-app reconstruction ({g.width}x{g.height}):\n"
                           f"  max difference = {int(diff.max())} / 255\n  mean difference = {float(diff.mean()):.6f} / 255\n"
                           + ("  -> identical: the equation text fully reproduces the image." if diff.max() <= 1 else
                              "  -> small deviations (rounding) - see max difference."))

        self.img_runner.submit(job, "verify")

    # ==================================================================================
    # Tab 3/4 - catalog and grid
    # ==================================================================================
    def _apply_grid_preset(self) -> None:
        p = self.grid_preset_var.get()
        g = (GridConfig.preset_fast_preview() if p == "preview" else
             GridConfig.preset_galaxy_cluster() if p == "galaxy" else GridConfig.preset_colliding_planets())
        self.w_var.set(g.width)
        self.h_var.set(g.height)
        self.m0_var.set(g.m0)
        self.n0_var.set(g.n0)
        self.scale_var.set(g.scale)

    def run_exact_catalog(self) -> None:
        mode = self.catalog_var.get()
        try:
            grid = self._read_grid()
        except Exception as exc:
            messagebox.showerror("Invalid grid", str(exc))
            return
        if self.grid_preset_var.get() == "preview":
            grid = (GridConfig(width=600, height=360, m0=324, n0=120, scale=204.0, chunk_rows=120) if mode == "galaxy_cluster"
                    else GridConfig(width=630, height=360, m0=291, n0=180, scale=195.6, chunk_rows=120))
        self._stop_animation()
        gp, sk, ps = int(self.gal_p_var.get()), int(self.star_k_var.get()), int(self.planet_s_var.get())

        def job(cancel, emit):
            cb = lambda f, m: emit("progress", (f, m))
            if mode == "galaxy_cluster":
                model = VHRExactCatalog.build_galaxy_cluster_model(grid, galaxy_count=gp, star_count=sk, wave_count=50)
                rgb = VHRExactCatalog.render_galaxy_cluster(model, progress_cb=cb, cancel=cancel)
            else:
                model = VHRExactCatalog.build_colliding_planets_model(grid, Z_terms=ps, K_terms=ps)
                rgb = VHRExactCatalog.render_colliding_planets(model, progress_cb=cb, cancel=cancel)
            emit("plate", (model, rgb))

        self.img_runner.submit(job, "catalog")

    # ==================================================================================
    # Files / exports
    # ==================================================================================
    def open_program(self) -> None:
        path = filedialog.askopenfilename(title="Open equation program",
                                          filetypes=[("Equation program", "*.vhr *.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as exc:
            messagebox.showerror("Cannot open", str(exc))
            return
        self.tabs.select(0)
        self.editor.delete("1.0", tk.END)
        self.editor.insert("1.0", text)
        self._on_source_edit(immediate=True)

    def open_model_json(self) -> None:
        path = filedialog.askopenfilename(title="Open model", filetypes=[("JSON model", "*.json")])
        if not path:
            return
        try:
            model = VHRExporterSuite.load_json(path)
            recon = PrecisionImageDecomposer.reconstruct_from_model(model) if (model.trig_terms or model.primitives) else None
        except Exception as exc:
            messagebox.showerror("Cannot open model", f"{type(exc).__name__}: {exc}")
            return
        if recon is None:
            messagebox.showinfo("Model", "This model has no fitted terms to render.")
            return
        self.analysis = None
        self.k_scale.state(["disabled"])
        self._on_model(model, recon, len(model.trig_terms))
        self.status_var.set(f"Loaded model {os.path.basename(path)}")

    def show_syntax_help(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Equation syntax guide")
        win.geometry("760x560")
        txt = tk.Text(win, font=("Consolas", 10), wrap=tk.WORD, padx=10, pady=8)
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert("1.0", SYNTAX_HELP)
        txt.configure(state="disabled")

    def _ask_save(self, ext: str, kinds) -> Optional[str]:
        return filedialog.asksaveasfilename(defaultextension=ext, filetypes=kinds) or None

    def export_png(self) -> None:
        if self.current_rgb is None:
            messagebox.showinfo("Nothing to save", "Render something first.")
            return
        path = self._ask_save(".png", [("PNG image", "*.png")])
        if path:
            Image.fromarray(self.current_rgb).save(path)
            self.status_var.set(f"Saved {path}")

    def export_hires(self) -> None:
        if self.program is None:
            messagebox.showinfo("No equation", "Type a valid equation first.")
            return
        path = self._ask_save(".png", [("PNG image", "*.png")])
        if not path:
            return
        prog, t = self.program, float(self.t_var.get())
        height = 1080
        w = int(round(height * (prog.grid_hint[0] / prog.grid_hint[1] if prog.grid_hint else self.ASPECTS[self.aspect_var.get()])))
        base = self._eq_base_grid(prog)
        g = base.scaled(height / base.height)

        def job(cancel, emit):
            emit("progress", (0.05, f"Rendering {g.width}x{g.height}..."))
            rgb = prog.render(RenderContext(g.width, g.height, g.m0, g.n0, g.scale, t), cancel=cancel)
            Image.fromarray(rgb).save(path)
            emit("saved", path)

        self.eq_runner.submit(job, "eq-hires")

    def _model_for_tex(self) -> Optional[VHRModel]:
        if self.context == "equation" and self.program is not None:
            self._refresh_latex_panel()
            g = self._eq_base_grid(self.program)
            return VHRModel(title="Equation program", engine_mode="program", grid=g, latex_lines=list(self.current_latex))
        return self.current_model

    def export_latex(self) -> None:
        model = self._model_for_tex()
        if model is None:
            messagebox.showinfo("Nothing to export", "Create an equation first.")
            return
        path = self._ask_save(".tex", [("LaTeX document", "*.tex")])
        if path:
            VHRExporterSuite.export_compilable_latex(model, path, max_den=int(self.max_den_var.get()))
            self.status_var.set(f"Saved LaTeX source: {path}")

    def export_python_script(self) -> None:
        if self.current_model is None or self.context == "equation":
            messagebox.showinfo("Python export", "Python export is available after an Image -> Equation conversion.\n"
                                                 "For typed equations use 'Program .vhr' (the equation text itself).")
            return
        path = self._ask_save(".py", [("Python script", "*.py")])
        if path:
            try:
                VHRExporterSuite.export_standalone_python_generator(self.current_model, path)
                self.status_var.set(f"Saved standalone script: {path}")
            except ExportError as exc:
                messagebox.showwarning("Cannot export", str(exc))

    def export_program(self) -> None:
        if self.context == "equation":
            text = self.editor.get("1.0", "end-1c")
        elif self.current_model is not None:
            try:
                text = VHRExporterSuite.build_vhr_program(self.current_model)
            except ExportError as exc:
                messagebox.showwarning("Cannot export", str(exc))
                return
        else:
            messagebox.showinfo("Nothing to export", "Create an equation first.")
            return
        path = self._ask_save(".vhr", [("Equation program", "*.vhr"), ("Text", "*.txt")])
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            self.status_var.set(f"Saved equation program: {path}")

    def export_json(self) -> None:
        if self.current_model is None or self.context == "equation":
            messagebox.showinfo("JSON export", "JSON models are produced by Image -> Equation conversions and catalog plates.")
            return
        path = self._ask_save(".json", [("JSON model", "*.json")])
        if path:
            VHRExporterSuite.export_json(self.current_model, path)
            self.status_var.set(f"Saved model: {path}")

    def export_plate(self) -> None:
        if self.current_rgb is None:
            messagebox.showinfo("Nothing to save", "Render something first.")
            return
        path = self._ask_save(".png", [("PNG plate", "*.png"), ("PDF plate", "*.pdf")])
        if not path:
            return
        self._refresh_latex_panel()
        fig = Figure(figsize=(13, 9), dpi=150, facecolor="white")
        FigureCanvasAgg(fig)
        ax_img = fig.add_axes([0.02, 0.36, 0.96, 0.62])
        ax_img.imshow(self.current_rgb)
        ax_img.axis("off")
        ax_eq = fig.add_axes([0.02, 0.01, 0.96, 0.33])
        draw_equation_lines(ax_eq, self.current_latex, self.current_title if self.context != "equation" else "Equation program", base_fs=10.5)
        fig.savefig(path, dpi=150)
        self.status_var.set(f"Saved plate: {path}")


def main() -> None:
    try:                                    # crisp text on high-DPI Windows displays
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    VHRProStudioApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
