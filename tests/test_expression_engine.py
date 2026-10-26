import numpy as np
import pytest

from expression_engine import (
    ExpressionError, RenderContext, compile_program, find_sliders, set_slider_value, to_latex, render_source,
)


def planes(src, w=40, h=24, t=0.0):
    prog = compile_program(src)
    ctx = RenderContext(w, h, w / 2, h / 2, h / 2, t)
    return prog.evaluate_planes(ctx)


def test_basic_gradient_matches_numpy():
    p = planes("H = x")
    x = (np.arange(1, 41) - 20) / 12.0
    assert np.allclose(p[0, :, 0], x)
    assert np.allclose(p[:, 0, 0], p[0, 0, 0])


def test_caret_power_and_implicit_multiplication():
    a = planes("H = 2x^2 + 3(x+1) + 2sin(x)")
    x = (np.arange(1, 41) - 20) / 12.0
    assert np.allclose(a[0, :, 0], 2 * x ** 2 + 3 * (x + 1) + 2 * np.sin(x))


def test_scientific_notation_is_not_split():
    p = planes("H = 1e-3 + 2.5E+1*0 + 8.3e-07")
    assert np.allclose(p, 1e-3 + 8.3e-07)


def test_unicode_symbols():
    p = planes("H = π·x² − 1")
    x = (np.arange(1, 41) - 20) / 12.0
    assert np.allclose(p[0, :, 0], np.pi * x ** 2 - 1)


def test_channel_variable_and_function_output():
    p = planes("H(x, y, s) = s")
    assert np.allclose(p[..., 0], 0) and np.allclose(p[..., 1], 1) and np.allclose(p[..., 2], 2)


def test_rgb_outputs():
    p = planes("R = 0.1\nG = 0.2\nB = 0.3")
    assert np.allclose(p[..., 0], 0.1) and np.allclose(p[..., 1], 0.2) and np.allclose(p[..., 2], 0.3)


def test_sum_prod_and_lists():
    p = planes("A = [1, 2, 3]\nH = sum(A[k]*k, k, 0, 2) + prod(k, k, 1, 4)")
    assert np.allclose(p, (0 * 1 + 1 * 2 + 2 * 3) + 24)


def test_user_functions_and_where():
    p = planes("sq(a) = a*a\nH = where(x > 0, sq(x), 0 - sq(x))")
    x = (np.arange(1, 41) - 20) / 12.0
    assert np.allclose(p[0, :, 0], np.where(x > 0, x * x, -x * x))


def test_last_bare_expression_is_output():
    assert np.allclose(planes("u = 2\nu*3"), 6)


def test_time_variable():
    assert np.allclose(planes("H = t", t=2.5), 2.5)
    assert compile_program("H = sin(t)").uses_time and not compile_program("H = x").uses_time


@pytest.mark.parametrize("bad", [
    "H = foo(x)", "H = x +", "H = __import__('os')", "H = x.real", "H = sin(x, y)", "H = zz + 1",
    "pi = 3", "H = H + 1", "H = 'abc'", "H = [x for x in y]", "H = lambda: 1", "H = x[1:2]", "",
    "H = sum(x, k, 1)", "a = 1 < 2 < 3\nH = a",
])
def test_rejects_unsafe_or_invalid(bad):
    with pytest.raises(ExpressionError):
        compile_program(bad)


def test_error_carries_line_number():
    with pytest.raises(ExpressionError) as ei:
        compile_program("a = 1\nb = 2\nH = a + nonsense")
    assert ei.value.line == 3


def test_huge_power_cannot_hang():
    out = render_source("H = 9**9**9**9", 8, 8)
    assert out.shape == (8, 8, 3) and (out == 255).all()


def test_evaluation_budget_is_enforced():
    with pytest.raises(ExpressionError):
        planes("H = sum(sum(x, j, 1, 5000), k, 1, 5000)")


def test_list_index_out_of_range_is_reported():
    with pytest.raises(ExpressionError):
        planes("A = [1, 2]\nH = A[5]")


def test_nan_and_inf_are_sanitised():
    out = render_source("H = sqrt(x) + 1/0", 16, 16)
    assert out.dtype == np.uint8


def test_cancellation():
    from expression_engine import EvaluationCancelled
    prog = compile_program("H = sum(sin(k*x), k, 1, 500)")
    with pytest.raises(EvaluationCancelled):
        prog.render(RenderContext(20, 20, 10, 10, 10), cancel=lambda: True)


def test_quantiser_matches_definition():
    out = render_source("H(x,y,s) = 0.5", 4, 4)
    assert (out == 127).all()          # floor(255 * 0.5)


def test_sliders_roundtrip():
    src = "a = 3   # [1, 12]\nb = 0.5 # [0, 1]\nH = a*b"
    sl = find_sliders(src)
    assert [(s.name, s.value, s.lo, s.hi) for s in sl] == [("a", 3.0, 1.0, 12.0), ("b", 0.5, 0.0, 1.0)]
    new = set_slider_value(src, sl[0].line_index, 7.25)
    assert "a = 7.25" in new and "# [1, 12]" in new


def test_grid_directive():
    prog = compile_program("#! grid 64 48 32 24 24.0\nH = x")
    assert prog.grid_hint == (64, 48, 32, 24, 24.0)


def test_latex_is_matplotlib_safe():
    from matplotlib.mathtext import MathTextParser
    parser = MathTextParser("agg")
    srcs = [
        "H = 1 - prod(1 - exp(-exp(8*((x-0.1k)^2+y^2-0.1))), k, 1, 5)",
        "H = (sin(x)^2 + 2*cos(3x)/(1+y^2))*sqrt(abs(x)) + floor(3x) + atan2(y,x)/pi",
        "A=[1,2,3,4,5,6]\nH = sum(A[k]*cos(k*x), k, 0, 5)/10",
        "f(a,b) = a^2/b\nH(x,y,s) = f(x, 1+y^2) if x > 0 else -mod(x, 0.3)",
    ]
    for s in srcs:
        for line in to_latex(compile_program(s)):
            parser.parse("$" + line + "$", dpi=100)


def test_big_numeric_table_is_not_rejected():
    table = ", ".join(str(i * 0.001) for i in range(20000))
    p = planes(f"A = [{table}]\nH = sum(A[k], k, 0, 5)", w=8, h=8)
    assert np.allclose(p, sum(i * 0.001 for i in range(6)))
