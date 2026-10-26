import numpy as np, time
from expression_engine import *
src = """
# demo
a = 3 # [1, 12]
H(x, y, s) = 0.5 + 0.5*sin(a*sqrt(x^2+y^2)*10 - 4*t + 2*pi*s/3) + 2x*0
"""
p = compile_program(src); print(p.output_mode, p.uses_time, find_sliders(src))
t=time.time(); img = render_source(src, 640, 360, t=1.0); print(img.shape, img.dtype, img.mean(), time.time()-t)
for bad in ["H = foo(x)", "H = x +", "H = __import__('os')", "H = x.real", "H = sin(x, y)", "H = zz+1", "H = 9**9**9**9", "pi = 3"]:
    try: compile_program(bad); print("NO ERROR for", bad)
    except ExpressionError as e: print("ok:", bad, "->", e)
# sums
img = render_source("H = sum(sin(k*x)/k, k, 1, 20)*0.3+0.5", 200,100); print(img.mean())
for l in to_latex(compile_program(src)): print(l)
from matplotlib.mathtext import MathTextParser
mp=MathTextParser("agg")
tests=["H = 1 - prod(1 - exp(-exp(8*((x-0.1k)^2+y^2-0.1))), k, 1, 5)", "H = (sin(x)^2 + 2*cos(3x)/(1+y^2))*sqrt(abs(x)) + floor(3x) + atan2(y,x)/pi", "A=[1,2,3,4,5,6]\nH = sum(A[k]*cos(k*x), k, 0, 5)/10"]
for s in tests:
    for l in to_latex(compile_program(s)):
        try: mp.parse("$"+l+"$", dpi=100); print("mathtext ok:", l[:90])
        except Exception as e: print("mathtext FAIL", l, str(e)[:150])
