import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageOps
from scipy import fft as sfft
from scipy.ndimage import label, gaussian_filter, maximum_filter, uniform_filter
from scipy.optimize import least_squares

from models import GridConfig, PrimitiveTerm, TrigBasisTerm, VHRModel
from algebraic_core import AlgebraicCore

PARAM_DECIMALS = 7                     # model constants are rounded so exports reproduce the app bit-for-bit
QUANT_MSE_NORM = (1.0 / 12.0) / (255.0 ** 2)   # unavoidable uniform 8-bit quantisation noise (normalised units)
ProgressCb = Optional[Callable[[float, str], None]]
CancelCb = Optional[Callable[[], bool]]


class DecompositionCancelled(Exception):
    """Raised when the caller's cancel callback reports True (a newer job superseded this one)."""


def _check(cancel: CancelCb) -> None:
    if cancel is not None and cancel():
        raise DecompositionCancelled()


def _dct(plane: np.ndarray) -> np.ndarray:
    return sfft.dctn(plane, type=2, norm="ortho", workers=-1)


def _idct(coeff: np.ndarray) -> np.ndarray:
    return sfft.idctn(coeff, type=2, norm="ortho", workers=-1)


def _basis_norm(u: np.ndarray, v: np.ndarray, H: int, W: int) -> np.ndarray:
    au = np.where(u == 0, np.sqrt(1.0 / H), np.sqrt(2.0 / H))
    av = np.where(v == 0, np.sqrt(1.0 / W), np.sqrt(2.0 / W))
    return au * av


@dataclass
class ImageAnalysis:
    """
    Result of one analysis pass. Holds the full DCT spectrum ranked by energy so the
    reconstruction for *any* term count K can be produced in milliseconds (live slider),
    and an estimated PSNR(K) curve (Parseval) for the rate/distortion plot.
    """
    grid: GridConfig
    target_rgb: np.ndarray
    prim_layer: np.ndarray
    primitives: List[PrimitiveTerm]
    coeffs: np.ndarray                 # (3, H, W) orthonormal DCT of the (clip-adjusted) signed residual
    order: np.ndarray                  # flat indices sorted by descending energy
    energy_sorted: np.ndarray
    k_default: int
    notes: List[str] = field(default_factory=list)

    @property
    def total_coefficients(self) -> int:
        return int(self.order.size)

    def est_psnr(self, K: int) -> float:
        K = int(np.clip(K, 0, self.order.size))
        total = float(self.energy_sorted.sum())
        remaining = total - float(self.energy_sorted[:K].sum())
        n = 3.0 * self.grid.width * self.grid.height
        mse = max(remaining, 0.0) / n + QUANT_MSE_NORM
        return AlgebraicCore.psnr_from_mse(mse * 255.0 ** 2)

    def curve(self, points: int = 80) -> Tuple[np.ndarray, np.ndarray]:
        """(K values, estimated PSNR) on a log-spaced grid, computed with a single cumulative sum."""
        n = self.order.size
        ks = np.unique(np.clip(np.round(np.geomspace(1, n, points)).astype(np.int64), 1, n))
        cum = np.cumsum(self.energy_sorted)
        total = cum[-1]
        mse = np.maximum(total - cum[ks - 1], 0.0) / (3.0 * self.grid.width * self.grid.height) + QUANT_MSE_NORM
        psnr = 10.0 * np.log10(1.0 / mse)
        return ks.astype(np.float64), np.minimum(psnr, 99.99)

    def float_reconstruction(self, K: int) -> np.ndarray:
        """Unquantised H_s(x,y) planes using the K strongest cosine terms (masked inverse DCT)."""
        K = int(np.clip(K, 0, self.order.size))
        H, W = self.grid.height, self.grid.width
        mask = np.zeros(H * W, dtype=bool)
        mask[self.order[:K]] = True
        mask = mask.reshape(H, W)
        out = self.prim_layer.copy()
        for s in range(3):
            out[:, :, s] += _idct(self.coeffs[s] * mask)
        return out

    def reconstruct(self, K: int) -> np.ndarray:
        planes = self.float_reconstruction(K)
        return np.stack([AlgebraicCore.F_quantize(planes[:, :, s]) for s in range(3)], axis=-1)


class PrecisionImageDecomposer:
    """
    High-Precision Hybrid Image -> Equation Engine:
      Stage 1: Multi-peak & connected-component detection + Levenberg-Marquardt
               refinement of double-exponential primitives A_p(x, y); every primitive is
               only kept if it measurably lowers the final error (greedy A/B test).
      Stage 2: Unclipped Signed Residual calculation R_s(x, y) = I_s(x, y) - P_s(x, y).
      Stage 3: Adaptive 2D Orthonormal Cosine Series selection by Target PSNR
               (Parseval-exact term count, floor-quantiser compensation and
               saturation-aware refinement).
      Stage 4: Continued-Fraction Rationalization with 100% Matplotlib-safe LaTeX.
    """

    def __init__(
        self,
        image_path: Optional[str] = None,
        num_primitives: int = 12,
        max_trig_terms: int = 800,
        target_psnr_db: float = 42.0,
        optimize_primitives: bool = True,
        max_fraction_den: int = 1000,
        image: Optional[Union[np.ndarray, Image.Image]] = None,
        refine_iters: int = 3,
        fixed_terms: Optional[int] = None,
    ):
        if image_path is None and image is None:
            raise ValueError("Provide either image_path or image.")
        self.image_path = image_path
        self.image = image
        self.num_primitives = num_primitives
        self.max_trig_terms = max_trig_terms
        self.target_psnr_db = target_psnr_db
        self.optimize_primitives = optimize_primitives
        self.max_fraction_den = max_fraction_den
        self.refine_iters = max(0, int(refine_iters))
        self.fixed_terms = fixed_terms
        self.last_target_rgb: Optional[np.ndarray] = None
        self.analysis: Optional[ImageAnalysis] = None

    # ------------------------------------------------------------------ image loading
    def source_size(self) -> Tuple[int, int]:
        img = self._open_source()
        return img.size

    def _open_source(self) -> Image.Image:
        if self.image is not None:
            img = Image.fromarray(self.image) if isinstance(self.image, np.ndarray) else self.image
        else:
            img = Image.open(self.image_path)
            img.load()
        img = ImageOps.exif_transpose(img)           # phone photos are stored rotated + an EXIF flag
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            canvas.alpha_composite(rgba)
            return canvas.convert("RGB")
        return img.convert("RGB")

    def default_grid(self, max_side: int = 640) -> GridConfig:
        w, h = self.source_size()
        return GridConfig.for_image(w, h, max_side=max_side)

    def load_target_image(self, grid: GridConfig) -> np.ndarray:
        img = self._open_source()
        iw, ih = img.size
        target_ar = grid.width / grid.height
        if abs(iw / ih - target_ar) / target_ar > 0.01:
            # never stretch: centre-crop to the grid's aspect ratio instead
            if iw / ih > target_ar:
                new_w = int(round(ih * target_ar))
                left = (iw - new_w) // 2
                img = img.crop((left, 0, left + new_w, ih))
            else:
                new_h = int(round(iw / target_ar))
                top = (ih - new_h) // 2
                img = img.crop((0, top, iw, top + new_h))
        if img.size != (grid.width, grid.height):
            resample = Image.Resampling.LANCZOS if img.size[0] > grid.width else Image.Resampling.BICUBIC
            img = img.resize((grid.width, grid.height), resample)
        rgb = np.asarray(img, dtype=np.uint8)
        self.last_target_rgb = rgb
        return rgb

    # ------------------------------------------------------------------ primitives
    @staticmethod
    def _grid_axes(grid: GridConfig) -> Tuple[np.ndarray, np.ndarray]:
        x_row = (np.arange(1, grid.width + 1, dtype=np.float64) - grid.m0) / grid.scale
        y_col = (grid.n0 - np.arange(1, grid.height + 1, dtype=np.float64)) / grid.scale
        return x_row, y_col

    @staticmethod
    def _primitive_window(p: PrimitiveTerm, grid: GridConfig) -> Tuple[slice, slice]:
        """Pixel window outside which A_p < 1e-6 (double exponential decays super-exponentially)."""
        q_max = 1.0 + 2.7 / max(p.k, 1e-3)
        R = max(p.a, p.b) * np.sqrt(q_max)
        j0 = int(np.floor(grid.m0 - 1 + grid.scale * (p.x0 - R)))
        j1 = int(np.ceil(grid.m0 - 1 + grid.scale * (p.x0 + R)))
        i0 = int(np.floor(grid.n0 - 1 - grid.scale * (p.y0 + R)))
        i1 = int(np.ceil(grid.n0 - 1 - grid.scale * (p.y0 - R)))
        return (slice(max(i0, 0), min(i1 + 1, grid.height)), slice(max(j0, 0), min(j1 + 1, grid.width)))

    @staticmethod
    def _primitive_field(p: PrimitiveTerm, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ct, st = np.cos(p.theta), np.sin(p.theta)
        dx, dy = x - p.x0, y - p.y0
        u = (dx * ct + dy * st) / p.a
        v = (-dx * st + dy * ct) / p.b
        return np.exp(-np.exp(np.clip(p.k * (u * u + v * v - 1.0), -40.0, 40.0)))

    @staticmethod
    def evaluate_primitives(primitives: List[PrimitiveTerm], grid: GridConfig) -> np.ndarray:
        """P_s = 1 - prod_p (1 - W_{p,s} A_p) with signed weights W_{p,s} (unclipped)."""
        out = np.zeros((grid.height, grid.width, 3), dtype=np.float64)
        if not primitives:
            return out
        x_row, y_col = PrecisionImageDecomposer._grid_axes(grid)
        prod_acc = np.ones((grid.height, grid.width, 3), dtype=np.float64)
        for p in primitives:
            rows, cols = PrecisionImageDecomposer._primitive_window(p, grid)
            if rows.start >= rows.stop or cols.start >= cols.stop:
                continue
            A = PrecisionImageDecomposer._primitive_field(p, x_row[None, cols], y_col[rows, None])
            for s in range(3):
                w_s = AlgebraicCore.eval_lagrange(p.c0, p.c1, p.c2, s)
                prod_acc[rows, cols, s] *= (1.0 - w_s * A)
        return 1.0 - prod_acc

    def _candidate_primitives(self, target: np.ndarray, grid: GridConfig, limit: int,
                              cancel: CancelCb) -> List[PrimitiveTerm]:
        H, W = grid.height, grid.width
        luma = 0.299 * target[:, :, 0] + 0.587 * target[:, :, 1] + 0.114 * target[:, :, 2]
        short = min(H, W)
        smooth = gaussian_filter(luma, sigma=max(1.2, short / 220.0))
        background = gaussian_filter(luma, sigma=max(4.0, short / 10.0))
        prominence = smooth - background
        nb = max(7, int(short / 12)) | 1
        peaks = (smooth == maximum_filter(smooth, size=nb)) & (prominence > max(0.04, 1.2 * float(np.std(prominence))))
        py, px = np.nonzero(peaks)
        order = np.argsort(-prominence[py, px])[:limit]
        x_row, y_col = self._grid_axes(grid)
        found: List[PrimitiveTerm] = []

        for idx in order:
            _check(cancel)
            r, c = int(py[idx]), int(px[idx])
            half_win = nb * 2
            r0, r1 = max(0, r - half_win), min(H, r + half_win + 1)
            c0, c1 = max(0, c - half_win), min(W, c + half_win + 1)
            sm_w = smooth[r0:r1, c0:c1]
            bg_val = float(background[r, c])
            half = bg_val + 0.5 * (float(smooth[r, c]) - bg_val)
            lab, _ = label(sm_w > half)
            comp = lab == lab[r - r0, c - c0]
            area = int(comp.sum())
            if area < 6 or area > 0.45 * H * W:
                continue
            xs = np.broadcast_to(x_row[None, c0:c1], comp.shape)[comp]
            ys = np.broadcast_to(y_col[r0:r1, None], comp.shape)[comp]
            x0, y0 = float(xs.mean()), float(ys.mean())
            dx, dy = xs - x0, ys - y0
            cxx, cyy = float(np.mean(dx * dx)) + 1e-8, float(np.mean(dy * dy)) + 1e-8
            cxy = float(np.mean(dx * dy))
            theta = 0.5 * float(np.arctan2(2.0 * cxy, cxx - cyy))
            ct, st = np.cos(theta), np.sin(theta)
            a = float(np.sqrt(max(cxx * ct ** 2 + 2 * cxy * ct * st + cyy * st ** 2, 1e-9))) * 2.0
            b = float(np.sqrt(max(cxx * st ** 2 - 2 * cxy * ct * st + cyy * ct ** 2, 1e-9))) * 2.0
            k = 4.0
            amp_luma = float(smooth[r, c]) - bg_val

            if self.optimize_primitives:
                # fit window = blob extent + margin (never the whole neighbourhood), sub-sampled to <= ~3000 px
                rad = int(min(3 * half_win, 2.4 * max(a, b) * grid.scale + 6))
                wr0, wr1 = max(0, r - rad), min(H, r + rad + 1)
                wc0, wc1 = max(0, c - rad), min(W, c + rad + 1)
                stride = max(1, int(np.ceil(np.sqrt((wr1 - wr0) * (wc1 - wc0) / 3000.0))))
                xw = x_row[None, wc0:wc1:stride]
                yw = y_col[wr0:wr1:stride, None]
                lw = luma[wr0:wr1:stride, wc0:wc1:stride]
                pad = 1.0 / grid.scale

                def residual(params):
                    px0, py0, pth, pa, pb, pk, pamp = params
                    tmp = PrimitiveTerm(px0, py0, pth, max(pa, 1e-4), max(pb, 1e-4), pk, 0.0, 0.0, 0.0, 0.0)
                    return (bg_val + pamp * self._primitive_field(tmp, xw, yw) - lw).ravel()

                try:
                    sol = least_squares(
                        residual, x0=[x0, y0, theta, a, b, k, amp_luma],
                        bounds=([x0 - 5 * pad, y0 - 5 * pad, -np.pi, 2 * pad, 2 * pad, 1.0, -1.5],
                                [x0 + 5 * pad, y0 + 5 * pad, np.pi, 4 * max(a, b), 4 * max(a, b), 24.0, 1.5]),
                        max_nfev=20,
                    )
                    x0, y0, theta, a, b, k = (float(v) for v in sol.x[:6])
                except Exception:
                    pass

            prim = PrimitiveTerm(x0=x0, y0=y0, theta=theta, a=a, b=b, k=k, spiral_pitch=0.0,
                                 c0=0.0, c1=0.0, c2=0.0, energy=float(area))
            rows, cols = self._primitive_window(prim, grid)
            if rows.start >= rows.stop or cols.start >= cols.stop:
                continue
            A = self._primitive_field(prim, x_row[None, cols], y_col[rows, None])
            core = A > 0.5
            ring = A < 0.02
            if core.sum() < 3 or ring.sum() < 3:
                continue
            tw = target[rows, cols]
            delta = [float(np.mean(tw[:, :, s][core]) - np.median(tw[:, :, s][ring])) for s in range(3)]
            prim.c0, prim.c1, prim.c2 = AlgebraicCore.rgb_to_lagrange(*delta)
            found.append(prim)
        return found

    @staticmethod
    def _topk_remaining_energy(residual: np.ndarray, K: int) -> float:
        energy = sum(_dct(residual[:, :, s]) ** 2 for s in range(3)).ravel()
        K = int(np.clip(K, 1, energy.size))
        top = np.partition(energy, energy.size - K)[energy.size - K:]
        return float(energy.sum() - top.sum())

    def _select_primitives(self, target: np.ndarray, grid: GridConfig, k_ref: int,
                           cancel: CancelCb, notes: List[str]) -> List[PrimitiveTerm]:
        if self.num_primitives <= 0:
            return []
        cands = self._candidate_primitives(target, grid, limit=self.num_primitives * 2, cancel=cancel)
        if not cands:
            notes.append("No bright compact features detected - using the pure cosine series.")
            return []
        chosen: List[PrimitiveTerm] = []
        best = self._topk_remaining_energy(target, k_ref)
        for cand in cands:
            if len(chosen) >= self.num_primitives:
                break
            _check(cancel)
            layer = self.evaluate_primitives(chosen + [cand], grid)
            err = self._topk_remaining_energy(target - layer, k_ref)
            if err < best * 0.995:
                chosen.append(cand)
                best = err
        notes.append(f"Primitives: {len(chosen)} of {len(cands)} candidates improved the fit and were kept.")
        return chosen

    # ------------------------------------------------------------------ analysis
    def _auto_k(self, energy: np.ndarray, n_pix3: float) -> int:
        n = energy.size
        if self.fixed_terms is not None:
            return int(np.clip(self.fixed_terms, 1, min(n, max(1, self.max_trig_terms))))
        cap = int(np.clip(self.max_trig_terms, 1, n))
        allowed_mse = 10.0 ** (-self.target_psnr_db / 10.0) - QUANT_MSE_NORM
        if allowed_mse <= 0:
            return cap
        sorted_e = np.sort(energy)[::-1]
        remaining = sorted_e.sum() - np.cumsum(sorted_e)
        ok = remaining <= allowed_mse * n_pix3
        if not ok.any():
            return cap
        return int(min(int(np.argmax(ok)) + 1, cap))

    def analyze(self, grid: Optional[GridConfig] = None, progress_cb: ProgressCb = None,
                cancel: CancelCb = None) -> ImageAnalysis:
        def say(frac: float, msg: str) -> None:
            if progress_cb:
                progress_cb(frac, msg)

        if grid is None:
            grid = self.default_grid()
        grid.validate()
        say(0.05, "Loading target image (EXIF-aware, aspect preserved)...")
        target_rgb = self.load_target_image(grid)
        H, W = grid.height, grid.width
        n_pix3 = 3.0 * H * W
        notes: List[str] = []
        # F() floors, so aim at (value + 1/2): turns floor into round-to-nearest (removes a 0.5-level bias)
        target0 = (target_rgb.astype(np.float64) + 0.5) / 255.0
        hi, lo = target_rgb == 255, target_rgb == 0

        say(0.15, "Measuring cosine spectrum...")
        _check(cancel)
        base_energy = sum(_dct(target0[:, :, s]) ** 2 for s in range(3)).ravel()
        k_ref = self._auto_k(base_energy, n_pix3)

        say(0.25, "Detecting and A/B-testing double-exponential primitives...")
        primitives = self._select_primitives(target0, grid, k_ref, cancel, notes)
        prim_layer = self.evaluate_primitives(primitives, grid)

        target = target0.copy()
        coeffs = np.zeros((3, H, W))
        energy = np.zeros(H * W)
        K = k_ref
        for it in range(self.refine_iters + 1):
            _check(cancel)
            say(0.45 + 0.4 * it / (self.refine_iters + 1), f"Cosine fit, saturation-aware pass {it + 1}/{self.refine_iters + 1}...")
            residual = target - prim_layer
            coeffs = np.stack([_dct(residual[:, :, s]) for s in range(3)])
            energy = (coeffs ** 2).sum(axis=0).ravel()
            K = self._auto_k(energy, n_pix3)
            if it == self.refine_iters:
                break
            pick = np.argpartition(energy, energy.size - K)[energy.size - K:]
            mask = np.zeros(energy.size, dtype=bool)
            mask[pick] = True
            mask = mask.reshape(H, W)
            recon = prim_layer + np.stack([_idct(coeffs[s] * mask) for s in range(3)], axis=-1)
            target = target0.copy()
            target = np.where(hi & (recon >= 1.0), recon, target)          # F saturates: no need to match beyond 1
            target = np.where(lo & (recon < 1.0 / 255.0), recon, target)   # ... or below 1/255

        say(0.9, "Ranking spectrum...")
        order = np.argsort(-energy, kind="stable")
        analysis = ImageAnalysis(
            grid=grid, target_rgb=target_rgb, prim_layer=prim_layer, primitives=primitives,
            coeffs=coeffs, order=order, energy_sorted=energy[order], k_default=K, notes=notes,
        )
        if self.fixed_terms is None and K >= self.max_trig_terms and analysis.est_psnr(K) < self.target_psnr_db:
            notes.append(f"Term cap ({self.max_trig_terms}) reached before {self.target_psnr_db:.1f} dB; "
                         f"estimated PSNR {analysis.est_psnr(K):.1f} dB. Raise the cap for more accuracy.")
        self.analysis = analysis
        return analysis

    # ------------------------------------------------------------------ model building
    def build_model(self, analysis: ImageAnalysis, K: Optional[int] = None) -> Tuple[VHRModel, np.ndarray]:
        """Turns the K strongest terms into a closed-form model and re-renders it *from the rounded
        constants* so the reported metrics describe exactly what an exported equation produces."""
        t0 = time.perf_counter()
        grid = analysis.grid
        K = analysis.k_default if K is None else int(np.clip(K, 1, analysis.total_coefficients))
        flat = analysis.order[:K]
        u, v = np.divmod(flat, grid.width)
        norm = _basis_norm(u, v, grid.height, grid.width)
        sel = analysis.coeffs[:, u, v] * norm            # (3, K) amplitude of cos*cos for s = 0, 1, 2
        c0 = sel[0]
        c1 = -1.5 * sel[0] + 2.0 * sel[1] - 0.5 * sel[2]
        c2 = 0.5 * sel[0] - 1.0 * sel[1] + 0.5 * sel[2]
        trig_terms = [
            TrigBasisTerm(u=int(a), v=int(b), c0=float(np.round(x, PARAM_DECIMALS)),
                          c1=float(np.round(y, PARAM_DECIMALS)), c2=float(np.round(z, PARAM_DECIMALS)))
            for a, b, x, y, z in zip(u, v, c0, c1, c2)
        ]
        primitives = [self._round_primitive(p) for p in analysis.primitives]
        model = VHRModel(
            title="High-Precision VHR Algebraic Reconstruction",
            engine_mode="precision_hybrid",
            grid=grid, primitives=primitives, trig_terms=trig_terms,
            extra_params={"notes": list(analysis.notes)},
        )
        recon = self.reconstruct_from_model(model)
        elapsed = time.perf_counter() - t0
        model.metrics = AlgebraicCore.compute_metrics(analysis.target_rgb, recon, len(primitives) + len(trig_terms), elapsed)
        model.title = (f"High-Precision VHR Algebraic Reconstruction "
                       f"(PSNR: {model.metrics.psnr_db} dB, SSIM: {model.metrics.ssim})")
        model.latex_lines = self._compile_rational_latex(grid, primitives, trig_terms)
        return model, recon

    @staticmethod
    def _round_primitive(p: PrimitiveTerm) -> PrimitiveTerm:
        r = lambda val: float(np.round(val, PARAM_DECIMALS))
        return PrimitiveTerm(x0=r(p.x0), y0=r(p.y0), theta=r(p.theta), a=r(p.a), b=r(p.b), k=r(p.k),
                             spiral_pitch=0.0, c0=r(p.c0), c1=r(p.c1), c2=r(p.c2), energy=float(p.energy))

    @staticmethod
    def reconstruct_from_model(model: VHRModel) -> np.ndarray:
        """Evaluates the closed form with the model's stored constants (fast sparse inverse DCT).
        Mathematically identical to  sum_k (c0+c1 s+c2 s^2) cos(pi v (m-1/2)/W) cos(pi u (n-1/2)/H)."""
        grid = model.grid
        H, W = grid.height, grid.width
        planes = PrecisionImageDecomposer.evaluate_primitives(model.primitives, grid)
        if model.trig_terms:
            u = np.array([t.u for t in model.trig_terms])
            v = np.array([t.v for t in model.trig_terms])
            c0 = np.array([t.c0 for t in model.trig_terms])
            c1 = np.array([t.c1 for t in model.trig_terms])
            c2 = np.array([t.c2 for t in model.trig_terms])
            norm = _basis_norm(u, v, H, W)
            for s in range(3):
                spec = np.zeros((H, W))
                np.add.at(spec, (u, v), (c0 + c1 * s + c2 * s * s) / norm)
                planes[:, :, s] += _idct(spec)
        return np.stack([AlgebraicCore.F_quantize(planes[:, :, s]) for s in range(3)], axis=-1)

    @staticmethod
    def evaluate_closed_form(model: VHRModel) -> np.ndarray:
        """Slow, completely independent evaluation (explicit cosines) used to verify the fast path."""
        grid = model.grid
        H, W = grid.height, grid.width
        planes = PrecisionImageDecomposer.evaluate_primitives(model.primitives, grid)
        if model.trig_terms:
            m = np.arange(1, W + 1, dtype=np.float64)
            n = np.arange(1, H + 1, dtype=np.float64)
            u = np.array([t.u for t in model.trig_terms], dtype=np.float64)
            v = np.array([t.v for t in model.trig_terms], dtype=np.float64)
            cos_y = np.cos(np.pi * (n[:, None] - 0.5) * u[None, :] / H)
            cos_x = np.cos(np.pi * v[:, None] * (m[None, :] - 0.5) / W)
            for s in range(3):
                w = np.array([t.c0 + t.c1 * s + t.c2 * s * s for t in model.trig_terms])
                planes[:, :, s] += (cos_y * w[None, :]) @ cos_x
        return np.stack([AlgebraicCore.F_quantize(planes[:, :, s]) for s in range(3)], axis=-1)

    # ------------------------------------------------------------------ one-shot API (kept for compatibility)
    def decompose_and_render(
        self,
        grid: Optional[GridConfig] = None,
        progress_cb: ProgressCb = None,
        cancel: CancelCb = None,
    ) -> Tuple[VHRModel, np.ndarray, np.ndarray]:
        t0 = time.perf_counter()
        analysis = self.analyze(grid, progress_cb, cancel)
        if progress_cb:
            progress_cb(0.95, f"Compiling closed-form equation with {analysis.k_default} basis waves...")
        model, recon = self.build_model(analysis)
        model.metrics.render_time_sec = round(time.perf_counter() - t0, 2)
        if progress_cb:
            progress_cb(1.0, f"Done! PSNR: {model.metrics.psnr_db} dB | SSIM: {model.metrics.ssim} | Terms: {model.metrics.total_terms}")
        return model, recon, analysis.prim_layer

    def _compile_rational_latex(
        self,
        grid: GridConfig,
        primitives: List[PrimitiveTerm],
        trig_terms: List[TrigBasisTerm]
    ) -> List[str]:
        W, H, m0, n0 = grid.width, grid.height, int(round(grid.m0)), int(round(grid.n0))
        S = f"{grid.scale:g}"
        P_cnt, K_cnt = len(primitives), len(trig_terms)
        D = self.max_fraction_den

        sample_trig = []
        for t in trig_terms[:2]:
            r0 = AlgebraicCore.to_latex_rational(t.c0, D, signed=False)
            r1 = AlgebraicCore.to_latex_rational(t.c1, D, signed=True)
            r2 = AlgebraicCore.to_latex_rational(t.c2, D, signed=True)
            sample_trig.append(
                rf"\left({r0}{r1}s{r2}s^2\right)\cos\left(\frac{{{t.v}\pi(m-1/2)}}{{{W}}}\right)\cos\left(\frac{{{t.u}\pi(n-1/2)}}{{{H}}}\right)"
            )
        sample_str = " + ".join(sample_trig) + rf" + \dots \quad ({K_cnt}\ \mathrm{{terms}})"
        head = rf"\mathrm{{The\ above\ picture\ is\ a\ converted\ version\ of\ a\ }} {W}\times {H} \mathrm{{\ image.\ For\ }} m=1,2,\dots,{W} \mathrm{{\ and\ }} n=1,2,\dots,{H}:"
        rgb = (rf"\mathrm{{rgb}}\left(F\left(H_0\left(\frac{{m-{m0}}}{{{S}}},\frac{{{n0}-n}}{{{S}}}\right)\right), F\left(H_1\left(\frac{{m-{m0}}}{{{S}}},\frac{{{n0}-n}}{{{S}}}\right)\right), "
               rf"F\left(H_2\left(\frac{{m-{m0}}}{{{S}}},\frac{{{n0}-n}}{{{S}}}\right)\right)\right), \quad F(x)=\left\lfloor 255e^{{-e^{{-1000x}}}}|x|^{{e^{{-e^{{1000(x-1)}}}}}}\right\rfloor")

        if P_cnt > 0:
            hs_line = rf"H_s(x,y) = \left(1 - \prod_{{p=1}}^{{{P_cnt}}}\left(1 - \left(c_{{0,p}} + c_{{1,p}}s + c_{{2,p}}s^2\right)e^{{-e^{{k_p(U_p(x,y)^2 + V_p(x,y)^2 - 1)}}}}\right)\right) + C_s(x,y)"
            uv_line = rf"U_p(x,y) = \frac{{(x-x_p)\cos\theta_p + (y-y_p)\sin\theta_p}}{{a_p}}, \qquad V_p(x,y) = \frac{{-(x-x_p)\sin\theta_p + (y-y_p)\cos\theta_p}}{{b_p}}"
            cs_head = rf"C_s(x,y) = \sum_{{k=1}}^{{{K_cnt}}} \left(\alpha_{{0,k}} + \alpha_{{1,k}}s + \alpha_{{2,k}}s^2\right)\cos\left(\frac{{\pi v_k(m-1/2)}}{{{W}}}\right)\cos\left(\frac{{\pi u_k(n-1/2)}}{{{H}}}\right)"
            return [head, rgb, hs_line, uv_line, cs_head, rf"C_s(x,y) = {sample_str}"]
        return [
            head, rgb,
            rf"H_s(x,y) = C_s(x,y) = \sum_{{k=1}}^{{{K_cnt}}} \left(\alpha_{{0,k}} + \alpha_{{1,k}}s + \alpha_{{2,k}}s^2\right)\cos\left(\frac{{\pi v_k(m-1/2)}}{{{W}}}\right)\cos\left(\frac{{\pi u_k(n-1/2)}}{{{H}}}\right)",
            rf"C_s(x,y) = {sample_str}",
        ]
