import numpy as np, time, subprocess, sys, os
from PIL import Image
from models import GridConfig
from precision_decomposer import PrecisionImageDecomposer
from exporters import VHRExporterSuite
from expression_engine import compile_program, RenderContext

d = PrecisionImageDecomposer("scratch/test_4x3.png", num_primitives=6, max_trig_terms=2000, target_psnr_db=40)
grid = d.default_grid(320)
print("grid", grid)
t=time.time(); a = d.analyze(grid); print("analyze %.2fs"%(time.time()-t), "K=",a.k_default, "prims=",len(a.primitives), a.notes)
t=time.time(); model, recon = d.build_model(a); print("build %.2fs"%(time.time()-t), model.metrics)
print("est psnr at K:", a.est_psnr(a.k_default))
# 1. fast vs independent explicit closed form
slow = PrecisionImageDecomposer.evaluate_closed_form(model)
print("fast-vs-explicit max diff:", int(np.abs(slow.astype(int)-recon.astype(int)).max()))
# 2. exported equation program rendered by expression engine
src = VHRExporterSuite.build_vhr_program(model); print("program chars", len(src))
prog = compile_program(src); gh = prog.grid_hint; print("grid_hint", gh)
ctx = RenderContext(width=gh[0], height=gh[1], m0=gh[2], n0=gh[3], scale=gh[4])
t=time.time(); eng = prog.render(ctx); print("engine render %.2fs"%(time.time()-t))
print("engine-vs-fast max diff:", int(np.abs(eng.astype(int)-recon.astype(int)).max()), "mean", np.abs(eng.astype(int)-recon.astype(int)).mean())
# 3. standalone python script
VHRExporterSuite.export_standalone_python_generator(model, "scratch/export_script.py")
r = subprocess.run([sys.executable, "export_script.py"], cwd="scratch", capture_output=True, text=True); print(r.stdout.strip(), r.stderr.strip()[-300:])
sc = np.asarray(Image.open("scratch/reconstructed_from_equation.png"))
print("script-vs-fast max diff:", int(np.abs(sc.astype(int)-recon.astype(int)).max()))
# 4. live K behaviour
for K in (50, 200, 800, a.k_default):
    t=time.time(); r=a.reconstruct(K); dt=time.time()-t
    from algebraic_core import AlgebraicCore
    m=AlgebraicCore.compute_metrics(a.target_rgb, r, K, 0); print("K=%5d  est %.1f  real %.2f dB  %.0f ms"%(K, a.est_psnr(K), m.psnr_db, dt*1000))
# 5. JSON round trip
VHRExporterSuite.export_json(model,"scratch/m.json"); m2=VHRExporterSuite.load_json("scratch/m.json")
print("json roundtrip diff:", int(np.abs(PrecisionImageDecomposer.reconstruct_from_model(m2).astype(int)-recon.astype(int)).max()))
