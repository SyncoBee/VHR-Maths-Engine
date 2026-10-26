import json
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from algebraic_core import AlgebraicCore
from exporters import VHRExporterSuite, ExportError
from expression_engine import compile_program, RenderContext
from models import GridConfig, VHRModel
from precision_decomposer import PrecisionImageDecomposer, DecompositionCancelled


def decompose(path, **kw):
    d = PrecisionImageDecomposer(path, **kw)
    a = d.analyze(d.default_grid(240))
    model, recon = d.build_model(a)
    return d, a, model, recon


def test_aspect_ratio_is_preserved(photo_4x3):
    path, arr = photo_4x3
    d = PrecisionImageDecomposer(path)
    g = d.default_grid(160)
    assert abs(g.width / g.height - 4 / 3) < 0.02
    target = d.load_target_image(g)
    assert target.shape == (g.height, g.width, 3)


def test_wide_image_into_square_grid_is_cropped_not_stretched(tmp_path):
    arr = np.zeros((100, 400, 3), np.uint8)
    arr[:, 150:250] = 255                          # centred white band, 100 px wide
    p = tmp_path / "wide.png"
    Image.fromarray(arr).save(p)
    d = PrecisionImageDecomposer(str(p))
    t = d.load_target_image(GridConfig(width=100, height=100, m0=50, n0=50, scale=50.0))
    assert (t[:, :, 0] == 255).all()               # centre crop is entirely the band (a stretch would not be)


def test_rgba_image_is_composited_on_white(tmp_path):
    arr = np.zeros((20, 20, 4), np.uint8)          # fully transparent black
    p = tmp_path / "t.png"
    Image.fromarray(arr, "RGBA").save(p)
    d = PrecisionImageDecomposer(str(p))
    assert (d.load_target_image(GridConfig(width=20, height=20, m0=10, n0=10, scale=10.0)) == 255).all()


def test_fast_reconstruction_equals_explicit_closed_form(photo_4x3):
    _, _, model, recon = decompose(photo_4x3[0], num_primitives=4, max_trig_terms=600, target_psnr_db=35)
    explicit = PrecisionImageDecomposer.evaluate_closed_form(model)
    assert np.array_equal(explicit, recon)


def test_quality_reaches_target_and_estimate_is_accurate(photo_4x3):
    _, a, model, _ = decompose(photo_4x3[0], num_primitives=0, max_trig_terms=20000, target_psnr_db=36)
    assert model.metrics.psnr_db >= 35.5
    assert abs(a.est_psnr(a.k_default) - model.metrics.psnr_db) < 1.5       # Parseval estimate is trustworthy


def test_psnr_is_monotonic_in_terms(photo_4x3):
    d, a, _, _ = decompose(photo_4x3[0], num_primitives=0, max_trig_terms=20000, target_psnr_db=40)
    vals = [AlgebraicCore.compute_metrics(a.target_rgb, a.reconstruct(k), k, 0).psnr_db for k in (10, 100, 1000, 8000)]
    assert vals == sorted(vals) and vals[-1] > vals[0] + 10


def test_primitives_are_only_kept_when_they_help(photo_4x3):
    path = photo_4x3[0]
    _, _, with_p, _ = decompose(path, num_primitives=8, max_trig_terms=500, fixed_terms=500)
    _, _, without, _ = decompose(path, num_primitives=0, max_trig_terms=500, fixed_terms=500)
    assert with_p.metrics.psnr_db >= without.metrics.psnr_db - 0.05


def test_saturation_refinement_never_hurts(photo_4x3):
    path = photo_4x3[0]
    _, _, r0, _ = decompose(path, num_primitives=0, max_trig_terms=400, fixed_terms=400, refine_iters=0)
    _, _, r3, _ = decompose(path, num_primitives=0, max_trig_terms=400, fixed_terms=400, refine_iters=3)
    assert r3.metrics.psnr_db >= r0.metrics.psnr_db - 0.05


def test_floor_compensation_removes_bias(photo_4x3):
    _, a, model, recon = decompose(photo_4x3[0], num_primitives=0, max_trig_terms=20000, target_psnr_db=45)
    signed = recon.astype(float) - a.target_rgb.astype(float)
    assert abs(signed.mean()) < 0.15                # a plain floor() quantiser would sit at about -0.5


def test_exported_program_reproduces_image_bit_exactly(photo_4x3):
    _, _, model, recon = decompose(photo_4x3[0], num_primitives=5, max_trig_terms=300, target_psnr_db=33)
    prog = compile_program(VHRExporterSuite.build_vhr_program(model))
    W, H, m0, n0, S = prog.grid_hint
    out = prog.render(RenderContext(W, H, m0, n0, S))
    assert np.array_equal(out, recon)


def test_exported_python_script_reproduces_image(photo_4x3, tmp_path):
    _, _, model, recon = decompose(photo_4x3[0], num_primitives=3, max_trig_terms=300, target_psnr_db=33)
    VHRExporterSuite.export_standalone_python_generator(model, str(tmp_path / "gen.py"))
    subprocess.run([sys.executable, "gen.py"], cwd=tmp_path, check=True, capture_output=True)
    got = np.asarray(Image.open(tmp_path / "reconstructed_from_equation.png"))
    assert np.array_equal(got, recon)


def test_json_roundtrip(photo_4x3, tmp_path):
    _, _, model, recon = decompose(photo_4x3[0], num_primitives=3, max_trig_terms=300, target_psnr_db=33)
    VHRExporterSuite.export_json(model, str(tmp_path / "m.json"))
    json.load(open(tmp_path / "m.json"))
    again = VHRExporterSuite.load_json(str(tmp_path / "m.json"))
    assert np.array_equal(PrecisionImageDecomposer.reconstruct_from_model(again), recon)


def test_latex_export_has_valid_structure_and_escapes_title(photo_4x3, tmp_path):
    _, _, model, _ = decompose(photo_4x3[0], num_primitives=2, max_trig_terms=200, target_psnr_db=30)
    model.title = "50% of A_B & C"
    VHRExporterSuite.export_compilable_latex(model, str(tmp_path / "x.tex"))
    tex = (tmp_path / "x.tex").read_text(encoding="utf-8")
    assert r"50\% of A\_B \& C" in tex and tex.rstrip().endswith(r"\end{document}")
    assert tex.count(r"\begin{longtable}") == tex.count(r"\end{longtable}") == 1
    assert tex.count(r"\begin{center}") == tex.count(r"\end{center}")


def test_catalog_models_cannot_export_blank_scripts(tmp_path):
    empty = VHRModel(title="Procedural", engine_mode="exact_galaxy_cluster", grid=GridConfig(width=8, height=8, m0=4, n0=4, scale=4.0))
    with pytest.raises(ExportError):
        VHRExporterSuite.export_standalone_python_generator(empty, str(tmp_path / "a.py"))
    with pytest.raises(ExportError):
        VHRExporterSuite.build_vhr_program(empty)


def test_latex_rational_never_lies():
    f = AlgebraicCore.to_latex_rational
    assert f(0.5) == r"\frac{1}{2}" and f(-3.0) == "-3" and f(0.25, signed=True) == r"+\frac{1}{4}"
    assert f(0.0003) != "0"                        # previously collapsed small coefficients to 0
    assert "frac" not in f(0.0123456) or abs(eval(f(0.0123456).replace(r"\frac{", "(").replace("}{", ")/(").replace("}", ")")) - 0.0123456) < 1e-6


def test_quantiser_edge_cases():
    x = np.array([-5.0, -1e-9, 0.0, 0.5, 1.0, 1.0000001, 7.0, np.nan, np.inf, -np.inf])
    out = AlgebraicCore.F_quantize(x)
    assert out.tolist() == [0, 0, 0, 127, 255, 255, 255, 0, 255, 0]


def test_cancel_stops_analysis(photo_4x3):
    d = PrecisionImageDecomposer(photo_4x3[0])
    with pytest.raises(DecompositionCancelled):
        d.analyze(d.default_grid(160), cancel=lambda: True)


def test_numpy_array_input_and_metrics_tiny_image():
    arr = (np.random.default_rng(0).random((5, 7, 3)) * 255).astype(np.uint8)
    d = PrecisionImageDecomposer(image=arr, num_primitives=0, max_trig_terms=100000, target_psnr_db=60)
    a = d.analyze(GridConfig(width=7, height=5, m0=3, n0=2, scale=2.5))
    model, recon = d.build_model(a)
    assert model.metrics.psnr_db > 55 and np.isfinite(model.metrics.ssim)
