from dataclasses import dataclass, field, asdict, fields
from typing import List, Optional, Dict, Any


@dataclass
class GridConfig:
    """Coordinate grid configuration matching VHR's (m, n) -> (x, y) mapping:
    x = (m - m0) / scale,  y = (n0 - n) / scale,  m = 1..width, n = 1..height."""
    width: int = 2000
    height: int = 1200
    m0: int = 1080
    n0: int = 401
    scale: float = 680.0
    chunk_rows: int = 150

    def validate(self) -> "GridConfig":
        if self.width < 2 or self.height < 2:
            raise ValueError("Grid width and height must be at least 2 pixels.")
        if self.width * self.height > 40_000_000:
            raise ValueError("Grid is larger than 40 megapixels; choose a smaller size.")
        if not self.scale > 0:
            raise ValueError("Grid scale must be positive.")
        return self

    @classmethod
    def for_image(cls, img_w: int, img_h: int, max_side: int = 640) -> "GridConfig":
        """Aspect-preserving grid centred on the origin with y spanning [-1, 1].
        (The previous fixed 16:9 grids stretched every non-16:9 image.)"""
        longest = max(img_w, img_h)
        f = min(1.0, max_side / float(longest))
        w, h = max(2, int(round(img_w * f))), max(2, int(round(img_h * f)))
        return cls(width=w, height=h, m0=int(round(w / 2)), n0=int(round(h / 2)),
                   scale=h / 2.0, chunk_rows=120)

    def scaled(self, factor: float) -> "GridConfig":
        """Same mathematical window sampled at `factor` times the pixel density."""
        w, h = max(2, int(round(self.width * factor))), max(2, int(round(self.height * factor)))
        return GridConfig(width=w, height=h, m0=self.m0 * w / self.width, n0=self.n0 * h / self.height,
                          scale=self.scale * h / self.height, chunk_rows=self.chunk_rows)

    @classmethod
    def preset_fast_preview(cls) -> "GridConfig":
        # Exact aspect ratio & normalized coordinate bounds scaled for interactive speed
        return cls(width=600, height=360, m0=324, n0=120, scale=204.0, chunk_rows=120)

    @classmethod
    def preset_galaxy_cluster(cls) -> "GridConfig":
        # Exact parameters from "Galaxy Cluster" (2000 x 1200)
        return cls(width=2000, height=1200, m0=1080, n0=401, scale=680.0, chunk_rows=100)

    @classmethod
    def preset_colliding_planets(cls) -> "GridConfig":
        # Exact parameters from "Colliding Planets" (2100 x 1200)
        return cls(width=2100, height=1200, m0=970, n0=601, scale=652.0, chunk_rows=100)


@dataclass
class PrimitiveTerm:
    """Anisotropic double-exponential / spiral primitive optimized via Least Squares.
    A_p = exp(-exp(k(U^2+V^2-1))) with a signed per-channel weight W_s = c0 + c1*s + c2*s^2."""
    x0: float
    y0: float
    theta: float
    a: float
    b: float
    k: float
    spiral_pitch: float
    c0: float
    c1: float
    c2: float
    energy: float = 1.0


@dataclass
class TrigBasisTerm:
    """2D Cosine basis term with quadratic Lagrange polynomial color weights in s."""
    u: int
    v: int
    c0: float
    c1: float
    c2: float


@dataclass
class QualityMetrics:
    """Quantitative accuracy metrics comparing target image vs. equation reconstruction.
    `measured` is False for models that have no reference image (e.g. catalog formulas),
    so the UI never reports a fake perfect score."""
    psnr_db: float = 0.0
    ssim: float = 1.0
    mae_255: float = 0.0
    max_err_255: float = 0.0
    total_terms: int = 0
    render_time_sec: float = 0.0
    measured: bool = True


@dataclass
class VHRModel:
    """Complete state of a synthesized or decomposed VHR mathematical image."""
    title: str
    engine_mode: str
    grid: GridConfig
    primitives: List[PrimitiveTerm] = field(default_factory=list)
    trig_terms: List[TrigBasisTerm] = field(default_factory=list)
    extra_params: Dict[str, Any] = field(default_factory=dict)
    latex_lines: List[str] = field(default_factory=list)
    metrics: QualityMetrics = field(default_factory=QualityMetrics)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VHRModel":
        """Inverse of to_dict (used to reload a saved .json model)."""
        def build(klass, payload):
            names = {f.name for f in fields(klass)}
            return klass(**{k: v for k, v in payload.items() if k in names})

        return cls(
            title=data.get("title", "Untitled"),
            engine_mode=data.get("engine_mode", "precision_hybrid"),
            grid=build(GridConfig, data["grid"]),
            primitives=[build(PrimitiveTerm, p) for p in data.get("primitives", [])],
            trig_terms=[build(TrigBasisTerm, t) for t in data.get("trig_terms", [])],
            extra_params=dict(data.get("extra_params", {})),
            latex_lines=list(data.get("latex_lines", [])),
            metrics=build(QualityMetrics, data.get("metrics", {})),
        )
