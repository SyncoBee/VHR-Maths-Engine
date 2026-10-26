#!/usr/bin/env python3
"""
Command-line interface for the VHR Maths Engine Pro.
Created & Published by Vihanga Himantha.

Commands:
  gui                    Launch the interactive GUI Studio (default if no command given)
  render-eq              Render an equation string or .vhr file to an image
  img2eq                 Convert an input image into a closed-form equation model
  catalog                Render cosmic reference catalog plates
"""

import argparse
import os
import sys
import time
import numpy as np
from PIL import Image

from models import GridConfig
from expression_engine import compile_program, RenderContext, render_source
from precision_decomposer import PrecisionImageDecomposer
from exact_catalog import VHRExactCatalog
from exporters import VHRExporterSuite


def cmd_gui(args):
    """Launch the flagship GUI application."""
    import studio_app
    studio_app.main()


def cmd_render_eq(args):
    """Render an equation string or .vhr program file to an image."""
    t0 = time.perf_counter()
    if os.path.isfile(args.source):
        with open(args.source, "r", encoding="utf-8") as f:
            source_text = f.read()
    else:
        source_text = args.source

    prog = compile_program(source_text)
    w = args.width
    h = args.height

    if prog.grid_hint and args.respect_hint:
        W, H, m0, n0, S = prog.grid_hint
        if args.width == 640 and args.height == 360:  # defaults, override with hint
            w, h = W, H
        ctx = RenderContext(width=w, height=h, m0=m0 * (w / W), n0=n0 * (h / H), scale=S * (h / H), t=args.time)
    else:
        ctx = RenderContext(
            width=w,
            height=h,
            m0=w / 2.0 if args.m0 is None else args.m0,
            n0=h / 2.0 if args.n0 is None else args.n0,
            scale=h / 2.0 if args.scale is None else args.scale,
            t=args.time
        )

    print(f"[*] Rendering equation to {w}x{h} (t={args.time:.2f})...")
    rgb = prog.render(ctx)
    elapsed = time.perf_counter() - t0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    Image.fromarray(rgb).save(args.output)
    print(f"[+] Output saved to {args.output} in {elapsed:.2f}s.")


def cmd_img2eq(args):
    """Decompose an image into closed-form mathematical equations."""
    t0 = time.perf_counter()
    if not os.path.isfile(args.input):
        print(f"[-] Error: Image file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Analyzing '{args.input}'...")
    decomposer = PrecisionImageDecomposer(
        image_path=args.input,
        num_primitives=args.primitives,
        max_trig_terms=args.max_terms,
        target_psnr_db=args.psnr,
        optimize_primitives=not args.no_lsq,
        max_fraction_den=args.max_den,
        refine_iters=args.refine
    )

    grid = decomposer.default_grid(max_side=args.max_side)
    print(f"[*] Working grid: {grid.width}x{grid.height} (scale={grid.scale:.1f})")

    def progress(pct, msg):
        sys.stdout.write(f"\r[{int(pct * 100):3d}%] {msg}")
        sys.stdout.flush()

    analysis = decomposer.analyze(grid=grid, progress_cb=progress)
    print()

    model, recon = decomposer.build_model(analysis)
    m = model.metrics
    print(f"[+] Reconstruction metrics:")
    print(f"    PSNR:        {m.psnr_db:.2f} dB")
    print(f"    SSIM:        {m.ssim:.4f}")
    print(f"    MAE:         {m.mae_255:.2f} / 255")
    print(f"    Max Err:     {m.max_err_255:.1f} / 255")
    print(f"    Total terms: {m.total_terms} (primitives={len(model.primitives)}, cosine={len(model.trig_terms)})")

    # Outputs
    if args.output_img:
        Image.fromarray(recon).save(args.output_img)
        print(f"[+] Reconstructed image saved to: {args.output_img}")

    if args.output_vhr:
        VHRExporterSuite.export_vhr_program(model, args.output_vhr)
        print(f"[+] VHR Equation Program saved to: {args.output_vhr}")

    if args.output_py:
        VHRExporterSuite.export_standalone_python_generator(model, args.output_py)
        print(f"[+] Standalone Python generator saved to: {args.output_py}")

    if args.output_tex:
        VHRExporterSuite.export_compilable_latex(model, args.output_tex, max_den=args.max_den)
        print(f"[+] Compilable LaTeX saved to: {args.output_tex}")

    if args.output_json:
        VHRExporterSuite.export_json(model, args.output_json)
        print(f"[+] JSON model saved to: {args.output_json}")

    print(f"[+] Entire conversion completed in {time.perf_counter() - t0:.2f}s.")


def cmd_catalog(args):
    """Render VHR reference catalog plates."""
    t0 = time.perf_counter()
    mode = args.plate.lower()
    w, h = args.width, args.height

    if "galaxy" in mode:
        grid = GridConfig(width=w, height=h, m0=int(w * 0.54), n0=int(h * 0.334), scale=h * 0.567)
        model = VHRExactCatalog.build_galaxy_cluster_model(
            grid,
            galaxy_count=args.galaxies,
            star_count=args.stars,
            wave_count=args.waves
        )
        print(f"[*] Rendering Galaxy Cluster ({w}x{h})...")
        rgb = VHRExactCatalog.render_galaxy_cluster(
            model,
            progress_cb=lambda pct, msg: (sys.stdout.write(f"\r[{int(pct * 100):3d}%] {msg}"), sys.stdout.flush())
        )
    else:
        grid = GridConfig(width=w, height=h, m0=int(w * 0.462), n0=int(h * 0.5), scale=h * 0.543)
        model = VHRExactCatalog.build_colliding_planets_model(
            grid,
            Z_terms=args.turbulence,
            K_terms=args.turbulence
        )
        print(f"[*] Rendering Colliding Planets ({w}x{h})...")
        rgb = VHRExactCatalog.render_colliding_planets(
            model,
            progress_cb=lambda pct, msg: (sys.stdout.write(f"\r[{int(pct * 100):3d}%] {msg}"), sys.stdout.flush())
        )
    print()

    Image.fromarray(rgb).save(args.output)
    print(f"[+] Saved {model.title} to {args.output} in {time.perf_counter() - t0:.2f}s.")


def main():
    parser = argparse.ArgumentParser(
        description="VHR Maths Engine Pro — Created & Published by Vihanga Himantha"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # GUI
    sub_gui = subparsers.add_parser("gui", help="Launch the interactive GUI Studio")
    sub_gui.set_defaults(func=cmd_gui)

    # Render Equation
    sub_eq = subparsers.add_parser("render-eq", help="Render an equation string or .vhr file to an image")
    sub_eq.add_argument("source", help="Equation code string or path to a .vhr / .txt file")
    sub_eq.add_argument("-o", "--output", default="equation_render.png", help="Output PNG path (default: equation_render.png)")
    sub_eq.add_argument("--width", type=int, default=640, help="Width in pixels (default: 640)")
    sub_eq.add_argument("--height", type=int, default=360, help="Height in pixels (default: 360)")
    sub_eq.add_argument("-t", "--time", type=float, default=0.0, help="Time variable t (default: 0.0)")
    sub_eq.add_argument("--m0", type=float, default=None, help="Origin column m0")
    sub_eq.add_argument("--n0", type=float, default=None, help="Origin row n0")
    sub_eq.add_argument("--scale", type=float, default=None, help="Scale divisor S")
    sub_eq.add_argument("--no-hint", dest="respect_hint", action="store_false", help="Ignore #! grid directive hints in file")
    sub_eq.set_defaults(func=cmd_render_eq)

    # Image to Equation
    sub_img = subparsers.add_parser("img2eq", help="Decompose an image into equations")
    sub_img.add_argument("input", help="Path to input image (PNG, JPG, BMP, etc.)")
    sub_img.add_argument("--psnr", type=float, default=38.0, help="Target reconstruction PSNR in dB (default: 38.0)")
    sub_img.add_argument("--max-terms", type=int, default=4000, help="Maximum cosine terms K (default: 4000)")
    sub_img.add_argument("--primitives", type=int, default=8, help="Number of double-exponential primitives (default: 8)")
    sub_img.add_argument("--refine", type=int, default=3, help="Saturation-aware refinement passes (default: 3)")
    sub_img.add_argument("--max-side", type=int, default=480, help="Working resolution longest side (default: 480)")
    sub_img.add_argument("--max-den", type=int, default=1000, help="Max fraction denominator for LaTeX (default: 1000)")
    sub_img.add_argument("--no-lsq", action="store_true", help="Disable sub-pixel Levenberg-Marquardt optimization")
    sub_img.add_argument("-o", "--output-img", default="reconstruction.png", help="Output reconstructed image file")
    sub_img.add_argument("--output-vhr", default="equation.vhr", help="Output VHR program file")
    sub_img.add_argument("--output-py", default="render.py", help="Output standalone Python script")
    sub_img.add_argument("--output-tex", default=None, help="Output compilable LaTeX document")
    sub_img.add_argument("--output-json", default=None, help="Output JSON model")
    sub_img.set_defaults(func=cmd_img2eq)

    # Catalog
    sub_cat = subparsers.add_parser("catalog", help="Render cosmic reference catalog plates")
    sub_cat.add_argument("plate", choices=["galaxy", "planets"], help="Catalog plate to render")
    sub_cat.add_argument("-o", "--output", default="catalog_plate.png", help="Output PNG path")
    sub_cat.add_argument("--width", type=int, default=600, help="Width in pixels (default: 600)")
    sub_cat.add_argument("--height", type=int, default=360, help="Height in pixels (default: 360)")
    sub_cat.add_argument("--galaxies", type=int, default=50, help="Galaxy count bound for Galaxy Cluster (default: 50)")
    sub_cat.add_argument("--stars", type=int, default=80, help="Starfield sum bound for Galaxy Cluster (default: 80)")
    sub_cat.add_argument("--waves", type=int, default=40, help="Wave count for Galaxy Cluster (default: 40)")
    sub_cat.add_argument("--turbulence", type=int, default=30, help="Turbulence sum bound for Colliding Planets (default: 30)")
    sub_cat.set_defaults(func=cmd_catalog)

    args = parser.parse_args()
    if args.command is None:
        cmd_gui(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
