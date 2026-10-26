import re
from fractions import Fraction
from typing import Tuple
import numpy as np
from scipy.ndimage import uniform_filter


class AlgebraicCore:
    """
    High-precision 64-bit algebraic core providing:
      - Exact branchless quantization F(x)
      - Exact Lagrange quadratic channel multiplexing
      - Continued-fraction Rationalizer (converts floats to exact LaTeX fractions)
      - Matplotlib _mathtext sanitizer & validator
      - Quantitative accuracy metrics (PSNR, SSIM, MAE, Max Error)
    """

    @staticmethod
    def F_quantize(x: np.ndarray) -> np.ndarray:
        """
        Exact branchless 8-bit clamping operator:
        F(x) = floor(255 * exp(-exp(-1000*x)) * |x|^(exp(-exp(1000*(x-1)))))
        """
        x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=1e30, neginf=-1e30)
        t_low = np.clip(-1000.0 * x, -500.0, 500.0)
        t_high = np.clip(1000.0 * (x - 1.0), -500.0, 500.0)

        step_zero = np.exp(-np.exp(t_low))
        exp_one = np.exp(-np.exp(t_high))

        base = np.maximum(np.abs(x), 1e-15)
        val = 255.0 * step_zero * (base ** exp_one)
        return np.floor(np.clip(val + 1e-9, 0.0, 255.0)).astype(np.uint8)

    @staticmethod
    def rgb_to_lagrange(r: float, g: float, b: float) -> Tuple[float, float, float]:
        """
        Solves for exact quadratic coefficients (c0, c1, c2) such that
        P(s) = c0 + c1*s + c2*s^2 evaluates to (r, g, b) at s = 0, 1, 2.
        """
        c0 = float(r)
        c1 = float(-1.5 * r + 2.0 * g - 0.5 * b)
        c2 = float(0.5 * r - 1.0 * g + 0.5 * b)
        return c0, c1, c2

    @staticmethod
    def lagrange_to_rgb(c0: float, c1: float, c2: float) -> Tuple[float, float, float]:
        """Inverse of rgb_to_lagrange: evaluates the polynomial at s = 0, 1, 2."""
        return c0, c0 + c1 + c2, c0 + 2.0 * c1 + 4.0 * c2

    @staticmethod
    def eval_lagrange(c0: float, c1: float, c2: float, s: int) -> float:
        return c0 + c1 * s + c2 * (s ** 2)

    @staticmethod
    def to_latex_rational(value: float, max_denominator: int = 1000, signed: bool = False) -> str:
        """
        Uses continued fractions to convert a floating-point constant into an
        authentic VHR exact rational fraction \\frac{p}{q}.
        If no fraction with denominator <= max_denominator reproduces the value to 1e-6
        a plain decimal is emitted instead (never a silently wrong fraction).
        """
        if abs(value) < 1e-7:
            return "+0" if signed else "0"

        frac = Fraction(float(value)).limit_denominator(max_denominator)
        num, den = frac.numerator, frac.denominator
        if num == 0 or abs(float(frac) - value) > 1e-6 * max(1.0, abs(value)):
            text = f"{value:.6g}"
            if "e" in text:
                mant, exp = text.split("e")
                text = rf"{mant}\times 10^{{{int(exp)}}}"
            return f"+{text}" if (signed and value > 0) else text

        sign_str = "+" if (num > 0 and signed) else ("-" if num < 0 else "")
        abs_num = abs(num)

        if den == 1:
            return f"{sign_str}{abs_num}"
        return rf"{sign_str}\frac{{{abs_num}}}{{{den}}}"

    @staticmethod
    def sanitize_for_matplotlib(latex_str: str) -> str:
        """
        Replaces full-LaTeX sizing macros (\\big, \\Big, \\bigg, \\Bigg, \\!)
        that cause ParseSyntaxException in matplotlib._mathtext.
        """
        s = latex_str
        s = re.sub(r"\\(?:big|Big|bigg|Bigg)[lrm]?\s*\(", r"\\left(", s)
        s = re.sub(r"\\(?:big|Big|bigg|Bigg)[lrm]?\s*\)", r"\\right)", s)
        s = re.sub(r"\\(?:big|Big|bigg|Bigg)[lrm]?\s*\[", r"\\left[", s)
        s = re.sub(r"\\(?:big|Big|bigg|Bigg)[lrm]?\s*\]", r"\\right]", s)
        s = s.replace(r"\!", "")
        return s

    @staticmethod
    def psnr_from_mse(mse: float, peak: float = 255.0) -> float:
        return 99.99 if mse < 1e-10 else float(10.0 * np.log10((peak ** 2) / mse))

    @staticmethod
    def compute_metrics(target_rgb: np.ndarray, recon_rgb: np.ndarray, total_terms: int, elapsed: float):
        """Computes PSNR (dB), SSIM, MAE, and Max Error between target and reconstructed images."""
        from models import QualityMetrics

        t_f = target_rgb.astype(np.float64)
        r_f = recon_rgb.astype(np.float64)
        diff = np.abs(t_f - r_f)

        mse = float(np.mean(diff ** 2))
        psnr = AlgebraicCore.psnr_from_mse(mse)
        mae = float(np.mean(diff))
        max_err = float(np.max(diff))

        y_t = 0.299 * t_f[:, :, 0] + 0.587 * t_f[:, :, 1] + 0.114 * t_f[:, :, 2]
        y_r = 0.299 * r_f[:, :, 0] + 0.587 * r_f[:, :, 1] + 0.114 * r_f[:, :, 2]

        c1 = (0.01 * 255.0) ** 2
        c2 = (0.03 * 255.0) ** 2

        # SSIM window cannot exceed the image; tiny images previously produced garbage.
        win = int(max(1, min(7, y_t.shape[0], y_t.shape[1])))
        mu_t = uniform_filter(y_t, size=win)
        mu_r = uniform_filter(y_r, size=win)
        mu_t_sq = mu_t ** 2
        mu_r_sq = mu_r ** 2
        mu_tr = mu_t * mu_r

        sigma_t_sq = np.maximum(uniform_filter(y_t ** 2, size=win) - mu_t_sq, 0.0)
        sigma_r_sq = np.maximum(uniform_filter(y_r ** 2, size=win) - mu_r_sq, 0.0)
        sigma_tr = uniform_filter(y_t * y_r, size=win) - mu_tr

        ssim_map = ((2.0 * mu_tr + c1) * (2.0 * sigma_tr + c2)) / (
            (mu_t_sq + mu_r_sq + c1) * (sigma_t_sq + sigma_r_sq + c2)
        )
        ssim_val = float(np.clip(np.mean(ssim_map), -1.0, 1.0))

        return QualityMetrics(
            psnr_db=round(psnr, 2),
            ssim=round(ssim_val, 4),
            mae_255=round(mae, 2),
            max_err_255=round(max_err, 1),
            total_terms=total_terms,
            render_time_sec=round(elapsed, 2)
        )
