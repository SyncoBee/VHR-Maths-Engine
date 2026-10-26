"""
Safe, vectorised mathematical-expression engine for the VHR Maths Engine.

An *equation program* is plain text such as::

    # grid: 640 x 360, centred, S = 180
    a = 3            # [1, 12]         <- becomes a live slider in the Studio
    r2 = x^2 + y^2
    H(x, y, s) = 0.5 + 0.5*sin(a*sqrt(r2)*10 - 4*t + 2*pi*s/3)

and is turned into an RGB image with VHR Maths Engine's branchless quantiser
F(x) = floor(255 e^{-e^{-1000x}} |x|^{e^{-e^{1000(x-1)}}}).

Highlights
  * Only a whitelisted subset of Python's AST is evaluated - no ``eval``/``exec``,
    no attribute access, no imports, every literal becomes a float64 so
    ``9**9**9**9`` can never hang the process.
  * Variables: x, y (grid coordinates), m, n (1-based pixel indices), r, th (polar),
    s (colour channel 0/1/2), t (time, for animation), Wpx, Hpx, pi, e, tau.
  * ``sum(expr, k, lo, hi)`` and ``prod(expr, k, lo, hi)`` (the sigma / pi notation),
    list constants ``[1, 2, 3]`` with indexing ``A[k]``, user functions
    ``f(a, b) = ...``, ``where(cond, a, b)``, comparisons, ``a if c else b``.
  * Friendly input: ``^`` for powers, implicit multiplication (``2x``, ``3(x+1)``),
    unicode symbols (π × · − ÷ ² ³), ``#`` comments.
  * Cooperative cancellation + evaluation budget so a live editor stays responsive.
  * ``to_latex`` converts a program back into typeset LaTeX (matplotlib-mathtext safe).
"""
from __future__ import annotations

import ast
import difflib
import math
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from algebraic_core import AlgebraicCore


class ExpressionError(ValueError):
    """Raised for syntax / evaluation problems; carries a 1-based line number when known."""

    def __init__(self, message: str, line: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.line = line

    def __str__(self) -> str:
        return f"line {self.line}: {self.message}" if self.line else self.message


class EvaluationCancelled(Exception):
    """Raised internally when the cancel callback reports True."""


# --------------------------------------------------------------------------------------
# Function / constant tables
# --------------------------------------------------------------------------------------
def _safe_div(a, b):
    return np.divide(a, b)


def _clip(a, lo, hi):
    return np.minimum(np.maximum(a, lo), hi)


def _smoothstep(e0, e1, v):
    t = _clip((v - e0) / (e1 - e0 if np.all(e1 != e0) else 1.0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _fract(a):
    return a - np.floor(a)


FUNCTIONS: Dict[str, Tuple[Callable, int, int]] = {
    # name: (callable, min_args, max_args)
    "sin": (np.sin, 1, 1), "cos": (np.cos, 1, 1), "tan": (np.tan, 1, 1),
    "asin": (np.arcsin, 1, 1), "arcsin": (np.arcsin, 1, 1),
    "acos": (np.arccos, 1, 1), "arccos": (np.arccos, 1, 1),
    "atan": (np.arctan, 1, 1), "arctan": (np.arctan, 1, 1),
    "atan2": (np.arctan2, 2, 2), "arctan2": (np.arctan2, 2, 2),
    "sinh": (np.sinh, 1, 1), "cosh": (np.cosh, 1, 1), "tanh": (np.tanh, 1, 1),
    "exp": (np.exp, 1, 1), "ln": (np.log, 1, 1), "log": (np.log, 1, 2),
    "log10": (np.log10, 1, 1), "log2": (np.log2, 1, 1),
    "sqrt": (np.sqrt, 1, 1), "cbrt": (np.cbrt, 1, 1), "abs": (np.abs, 1, 1),
    "floor": (np.floor, 1, 1), "ceil": (np.ceil, 1, 1), "round": (np.round, 1, 1),
    "sign": (np.sign, 1, 1), "fract": (_fract, 1, 1),
    "min": (np.minimum, 2, 2), "max": (np.maximum, 2, 2),
    "clip": (_clip, 3, 3), "clamp": (_clip, 3, 3),
    "mod": (np.mod, 2, 2), "pow": (np.power, 2, 2), "hypot": (np.hypot, 2, 2),
    "smoothstep": (_smoothstep, 3, 3),
    "mix": (lambda a, b, w: a + (b - a) * w, 3, 3),
    "where": (lambda c, a, b: np.where(np.asarray(c) != 0, a, b), 3, 3),
    "step": (lambda edge, v: (np.asarray(v) >= edge).astype(np.float64), 2, 2),
    "arccos_cos": (lambda a: np.arccos(np.clip(np.cos(a), -1.0, 1.0)), 1, 1),
}
SPECIAL_FORMS = {"sum", "prod"}
CONSTANTS = {"pi": math.pi, "e": math.e, "tau": 2.0 * math.pi, "phi": (1.0 + 5.0 ** 0.5) / 2.0}
CONTEXT_NAMES = {"x", "y", "m", "n", "r", "th", "s", "t", "Wpx", "Hpx"}
OUTPUT_NAMES = {"H", "R", "G", "B"}

MAX_AST_NODES = 6000
MAX_SUM_TERMS = 20000
DEFAULT_BUDGET = 400_000          # total sum/prod iterations per frame
MAX_CALL_DEPTH = 24


# --------------------------------------------------------------------------------------
# Source pre-processing
# --------------------------------------------------------------------------------------
_UNICODE_MAP = {
    "π": "pi", "τ": "tau", "θ": "th", "×": "*", "·": "*", "⋅": "*", "−": "-", "÷": "/",
    "²": "**2", "³": "**3", "√": "sqrt", "≤": "<=", "≥": ">=", "≠": "!=", "\u00a0": " ",
}
_NUMBER_TOKEN = re.compile(r"(?<![\w.])(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_KEYWORDS = r"(?:if|else|and|or|not|is|in)\b"
_CLOSE_BEFORE_ATOM = re.compile(r"\)\s*(?!" + _KEYWORDS + r")(?=[A-Za-z_\d(])")
_KEYWORD_AT_START = re.compile(r"\s*" + _KEYWORDS)
_FUNC_DEF = re.compile(r"^\s*([A-Za-z_]\w*)\s*\(\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)?\s*\)\s*=(?!=)\s*(.+)$")
_ASSIGN = re.compile(r"^\s*([A-Za-z_]\w*)\s*=(?!=)\s*(.+)$")
_DIRECTIVE_GRID = re.compile(r"^#!\s*grid\s+(\d+)\s+(\d+)\s+(-?\d+)\s+(-?\d+)\s+([0-9.eE+-]+)\s*$", re.I)
_SLIDER = re.compile(
    r"^(?P<head>\s*(?P<name>[A-Za-z_]\w*)\s*=\s*)(?P<val>-?\d+\.?\d*(?:[eE][+-]?\d+)?)"
    r"(?P<tail>\s*#\s*\[\s*(?P<lo>-?[\d.eE+-]+)\s*,\s*(?P<hi>-?[\d.eE+-]+)\s*\].*)$"
)


def _insert_number_multiplication(text: str) -> str:
    """2x -> 2*x, 3(x) -> 3*(x), 2exp(x) -> 2*exp(x) while leaving 1e-5 / 8.3e+2 untouched.
    (A look-ahead regex backtracks 8.3e-07 into 8.3*e-07, so numbers are matched whole first.)"""
    out, pos = [], 0
    for mo in _NUMBER_TOKEN.finditer(text):
        out.append(text[pos:mo.end()])
        pos = mo.end()
        rest = text[pos:]
        nxt = rest.lstrip()[:1]
        if nxt and (nxt.isalpha() or nxt in "_(") and not _KEYWORD_AT_START.match(rest):
            out.append("*")
    out.append(text[pos:])
    return "".join(out)


def _normalise(text: str) -> str:
    for src, dst in _UNICODE_MAP.items():
        text = text.replace(src, dst)
    text = text.replace("^", "**")
    # implicit multiplication:  2x -> 2*x, 3(x) -> 3*(x), (a)(b) -> (a)*(b), (a)x -> (a)*x
    text = _insert_number_multiplication(text)
    text = _CLOSE_BEFORE_ATOM.sub(")*", text)
    return text


def _strip_comment(line: str) -> str:
    cut = line.find("#")
    return line if cut < 0 else line[:cut]


def _logical_lines(source: str) -> List[Tuple[int, str]]:
    """Joins physical lines while brackets are open; splits on ';'. Returns (first_lineno, text)."""
    out: List[Tuple[int, str]] = []
    buf, start, depth = "", 0, 0
    for lineno, raw in enumerate(source.splitlines(), start=1):
        code = _strip_comment(raw)
        if not code.strip() and depth == 0:
            continue
        if not buf:
            start = lineno
        buf += " " + code
        depth += sum(code.count(c) for c in "([") - sum(code.count(c) for c in ")]")
        if depth <= 0:
            depth = 0
            for piece in buf.split(";"):
                if piece.strip():
                    out.append((start, piece.strip()))
            buf = ""
    if buf.strip():
        raise ExpressionError("Unbalanced brackets - a '(' or '[' is never closed.", start)
    return out


# --------------------------------------------------------------------------------------
# Program representation
# --------------------------------------------------------------------------------------
@dataclass
class Statement:
    name: Optional[str]
    expr: ast.expr
    line: int
    params: Optional[Tuple[str, ...]] = None     # not None -> user function definition
    s_dependent: bool = True


@dataclass
class RenderContext:
    width: int
    height: int
    m0: float
    n0: float
    scale: float
    t: float = 0.0


@dataclass
class Program:
    source: str
    statements: List[Statement] = field(default_factory=list)
    grid_hint: Optional[Tuple[int, int, int, int, float]] = None
    output_mode: str = "H"            # "H" | "RGB" | "HFUNC"
    output_func: Optional[str] = None
    list_cache: Dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    # ---- introspection helpers ----------------------------------------------------
    @property
    def uses_time(self) -> bool:
        return any(isinstance(n, ast.Name) and n.id == "t" for st in self.statements for n in ast.walk(st.expr))

    @property
    def function_defs(self) -> Dict[str, Statement]:
        return {st.name: st for st in self.statements if st.params is not None}

    # ---- rendering ------------------------------------------------------------------
    def evaluate_planes(
        self,
        ctx: RenderContext,
        cancel: Optional[Callable[[], bool]] = None,
        budget: int = DEFAULT_BUDGET,
    ) -> np.ndarray:
        """Returns float64 planes (H, W, 3) of H_0, H_1, H_2 *before* quantisation."""
        ev = _Evaluator(self, ctx, cancel, budget)
        return ev.run()

    def render(
        self,
        ctx: RenderContext,
        cancel: Optional[Callable[[], bool]] = None,
        budget: int = DEFAULT_BUDGET,
    ) -> np.ndarray:
        planes = self.evaluate_planes(ctx, cancel, budget)
        return np.stack([AlgebraicCore.F_quantize(planes[:, :, c]) for c in range(3)], axis=-1)


def _is_numeric_list(node: ast.AST) -> bool:
    return isinstance(node, ast.List) and all(
        isinstance(e, ast.Constant) or (isinstance(e, ast.UnaryOp) and isinstance(e.operand, ast.Constant))
        for e in node.elts
    )


def _count_nodes(tree: ast.AST) -> int:
    """Syntax-node count; a pure numeric table such as [0.1, 0.2, ...] counts as a single node so that
    image -> equation programs with thousands of coefficients are not rejected as 'too large'."""
    total, stack = 0, [tree]
    while stack:
        node = stack.pop()
        total += 1
        if _is_numeric_list(node):
            continue
        stack.extend(ast.iter_child_nodes(node))
    return total


def _free_names(expr: ast.expr, bound: Sequence[str] = ()) -> set:
    bound_set = set(bound)
    found = set()
    for node in ast.walk(expr):
        if isinstance(node, ast.Name) and node.id not in bound_set:
            found.add(node.id)
    return found


def compile_program(source: str) -> Program:
    """Parses and validates `source`. Raises ExpressionError with a line number on any problem."""
    prog = Program(source=source)
    for raw in source.splitlines():
        mo = _DIRECTIVE_GRID.match(raw.strip())
        if mo:
            prog.grid_hint = (int(mo.group(1)), int(mo.group(2)), int(mo.group(3)), int(mo.group(4)), float(mo.group(5)))

    total_nodes = 0
    for lineno, text in _logical_lines(source):
        name: Optional[str] = None
        params: Optional[Tuple[str, ...]] = None
        body = text

        mo = _FUNC_DEF.match(text)
        if mo and mo.group(1) not in FUNCTIONS and mo.group(1) not in SPECIAL_FORMS:
            name = mo.group(1)
            params = tuple(p.strip() for p in (mo.group(2) or "").split(",") if p.strip())
            body = mo.group(3)
        else:
            mo = _ASSIGN.match(text)
            if mo:
                name, body = mo.group(1), mo.group(2)

        if name is not None:
            if name in CONSTANTS or name in FUNCTIONS or name in SPECIAL_FORMS or name in CONTEXT_NAMES:
                raise ExpressionError(f"'{name}' is a reserved name and cannot be redefined.", lineno)

        try:
            tree = ast.parse(_normalise(body), mode="eval")
        except SyntaxError as exc:
            raise ExpressionError(f"Syntax error: {exc.msg}", lineno) from None
        total_nodes += _count_nodes(tree)
        if total_nodes > MAX_AST_NODES:
            raise ExpressionError(f"Program is too large (more than {MAX_AST_NODES} syntax nodes).", lineno)
        _validate(tree.body, lineno)
        prog.statements.append(Statement(name=name, expr=tree.body, line=lineno, params=params))

    if not prog.statements:
        raise ExpressionError("The program is empty - type an equation such as  H = 0.5 + 0.5*sin(10*x)")

    _resolve_output(prog)
    _check_names(prog)
    _mark_s_dependency(prog)
    return prog


def _validate(node: ast.AST, line: int) -> None:
    allowed = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Call, ast.Load, ast.Compare,
        ast.IfExp, ast.List, ast.Subscript, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
        ast.USub, ast.UAdd, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
    )
    for sub in ast.walk(node):
        if not isinstance(sub, allowed):
            raise ExpressionError(f"'{type(sub).__name__}' syntax is not allowed in equations.", line)
        if isinstance(sub, ast.Constant) and not isinstance(sub.value, (int, float)) or \
                isinstance(sub, ast.Constant) and isinstance(sub.value, bool):
            raise ExpressionError(f"Only numeric constants are allowed (found {sub.value!r}).", line)
        if isinstance(sub, ast.Call):
            if not isinstance(sub.func, ast.Name):
                raise ExpressionError("Only plain function calls like sin(x) are allowed.", line)
            if sub.keywords:
                raise ExpressionError("Keyword arguments are not supported.", line)
        if isinstance(sub, ast.Subscript) and isinstance(sub.slice, ast.Slice):
            raise ExpressionError("Slices are not supported - use A[k].", line)
        if isinstance(sub, ast.Compare) and len(sub.ops) != 1:
            raise ExpressionError("Chained comparisons are not supported; use (a<b)*(b<c).", line)


def _resolve_output(prog: Program) -> None:
    names = {st.name: st for st in prog.statements if st.name}
    if all(c in names and names[c].params in (None, ()) for c in "RGB"):
        prog.output_mode = "RGB"
        return
    if "H" in names and names["H"].params:
        prog.output_mode, prog.output_func = "HFUNC", "H"
        return
    if "H" in names:
        prog.output_mode = "H"
        return
    last = prog.statements[-1]
    if last.params is not None:
        raise ExpressionError("The last line defines a function; add a line such as  H = f(x, y, s)", last.line)
    if last.name is None:
        prog.statements[-1] = Statement(name="H", expr=last.expr, line=last.line)
    else:
        prog.statements.append(Statement(name="H", expr=ast.Name(id=last.name, ctx=ast.Load()), line=last.line))
    prog.output_mode = "H"


def _check_names(prog: Program) -> None:
    defined: set = set()
    fn_params = {st.name: st.params for st in prog.statements if st.params is not None}
    known_funcs = set(FUNCTIONS) | SPECIAL_FORMS | set(fn_params)
    for st in prog.statements:
        local = set(st.params or ())
        for node in ast.walk(st.expr):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                fid = node.func.id
                if fid in FUNCTIONS:
                    lo, hi = FUNCTIONS[fid][1], FUNCTIONS[fid][2]
                    if not lo <= len(node.args) <= hi:
                        raise ExpressionError(f"{fid}() takes {lo if lo == hi else f'{lo}-{hi}'} argument(s), got {len(node.args)}.", st.line)
                elif fid in fn_params:
                    if len(node.args) != len(fn_params[fid]):
                        raise ExpressionError(f"{fid}() takes {len(fn_params[fid])} argument(s), got {len(node.args)}.", st.line)
                elif fid in SPECIAL_FORMS:
                    if len(node.args) not in (4, 5) or not isinstance(node.args[1], ast.Name):
                        raise ExpressionError(f"{fid}(expr, k, lo, hi) needs an expression, a counter name, lower and upper bound.", st.line)
                elif fid in defined or fid in CONTEXT_NAMES or fid in CONSTANTS or fid in local:
                    pass     # implicit multiplication such as  a(x+1)
                else:
                    close = difflib.get_close_matches(fid, sorted(known_funcs), n=1)
                    hint = f" Did you mean '{close[0]}'?" if close else ""
                    raise ExpressionError(f"Unknown function '{fid}'.{hint}", st.line)
        counters = {n.args[1].id for n in ast.walk(st.expr)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in SPECIAL_FORMS
                    and len(n.args) >= 2 and isinstance(n.args[1], ast.Name)}
        for node in ast.walk(st.expr):
            if isinstance(node, ast.Name):
                nid = node.id
                if nid in fn_params or nid in FUNCTIONS or nid in SPECIAL_FORMS:
                    continue
                if nid in defined or nid in local or nid in counters or nid in CONSTANTS or nid in CONTEXT_NAMES:
                    continue
                if st.name == nid and st.params is None:
                    raise ExpressionError(f"'{nid}' refers to itself - define it in terms of other names.", st.line)
                pool = sorted(defined | local | counters | set(CONSTANTS) | CONTEXT_NAMES | set(fn_params))
                close = difflib.get_close_matches(nid, pool, n=1)
                hint = f" Did you mean '{close[0]}'?" if close else ""
                raise ExpressionError(f"Unknown name '{nid}'.{hint}", st.line)
        if st.name and st.params is None:
            defined.add(st.name)


def _mark_s_dependency(prog: Program) -> None:
    """Statements that never touch the colour channel `s` are evaluated once instead of three times."""
    fn_dep: Dict[str, bool] = {}
    fn_stmts = prog.function_defs
    for _ in range(len(fn_stmts) + 1):
        for name, st in fn_stmts.items():
            free = _free_names(st.expr, st.params or ())
            fn_dep[name] = ("s" in free) or any(fn_dep.get(f, True) for f in free if f in fn_stmts)
    var_dep: Dict[str, bool] = {}
    for st in prog.statements:
        if st.params is not None:
            st.s_dependent = fn_dep.get(st.name, True)
            continue
        free = _free_names(st.expr)
        dep = ("s" in free) or any(var_dep.get(f, False) for f in free) or any(fn_dep.get(f, False) for f in free)
        st.s_dependent = dep
        if st.name:
            var_dep[st.name] = dep


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
class _Evaluator:
    def __init__(self, prog: Program, ctx: RenderContext, cancel, budget: int):
        self.prog, self.ctx, self.cancel = prog, ctx, cancel
        self.budget = budget
        self.depth = 0
        self.globals: Dict[str, np.ndarray] = {}
        self.funcs = prog.function_defs
        W, H = ctx.width, ctx.height
        m = np.arange(1, W + 1, dtype=np.float64)[None, :]
        n = np.arange(1, H + 1, dtype=np.float64)[:, None]
        x = (m - ctx.m0) / ctx.scale
        y = (ctx.n0 - n) / ctx.scale
        used = {nd.id for st in prog.statements for nd in ast.walk(st.expr) if isinstance(nd, ast.Name)}
        self.base: Dict[str, np.ndarray] = {
            "x": x, "y": y, "m": m, "n": n, "t": np.float64(ctx.t),
            "Wpx": np.float64(W), "Hpx": np.float64(H),
        }
        if "r" in used:
            self.base["r"] = np.sqrt(x * x + y * y)
        if "th" in used:
            self.base["th"] = np.arctan2(y, x)

    # -- control -----------------------------------------------------------------------
    def _tick(self, amount: int = 1) -> None:
        self.budget -= amount
        if self.budget < 0:
            raise ExpressionError("Equation is too expensive to evaluate in real time (iteration budget exceeded). "
                                  "Lower the sum/prod bounds.")
        if self.cancel is not None and self.cancel():
            raise EvaluationCancelled()

    # -- main --------------------------------------------------------------------------
    def run(self) -> np.ndarray:
        H, W = self.ctx.height, self.ctx.width
        planes = np.zeros((H, W, 3), dtype=np.float64)
        prog = self.prog
        invariant: Dict[str, np.ndarray] = {}
        with np.errstate(all="ignore"):
            # pass 1: s-independent statements, evaluated once
            self.globals = dict(self.base)
            self.globals["s"] = np.float64(0.0)
            for st in prog.statements:
                if st.params is None and st.name and not st.s_dependent:
                    try:
                        self.globals[st.name] = self._eval(st.expr, {})
                    except ExpressionError as exc:
                        raise ExpressionError(exc.message, exc.line or st.line) from None
                    invariant[st.name] = self.globals[st.name]
            for s in range(3):
                self.globals = dict(self.base)
                self.globals.update(invariant)
                self.globals["s"] = np.float64(s)
                channel_vals: Dict[str, np.ndarray] = {}
                for st in prog.statements:
                    if st.params is not None or not st.name:
                        continue
                    if not st.s_dependent and st.name in invariant:
                        continue
                    if prog.output_mode == "HFUNC" and st.name == "H":
                        continue
                    try:
                        val = self._eval(st.expr, {})
                    except ExpressionError as exc:
                        raise ExpressionError(exc.message, exc.line or st.line) from None
                    self.globals[st.name] = val
                    channel_vals[st.name] = val
                try:
                    if prog.output_mode == "RGB":
                        out = self.globals["RGB"[s]]
                    elif prog.output_mode == "HFUNC":
                        st = self.funcs["H"]
                        argv = [self.globals["x"], self.globals["y"], self.globals["s"]][:len(st.params)]
                        while len(argv) < len(st.params):
                            argv.append(np.float64(0.0))
                        out = self._call_user("H", argv)
                    else:
                        out = self.globals["H"]
                except KeyError as exc:
                    raise ExpressionError(f"Output '{exc.args[0]}' is not defined.") from None
                planes[:, :, s] = np.nan_to_num(np.broadcast_to(out, (H, W)), nan=0.0, posinf=1e30, neginf=-1e30)
        return planes

    # -- node evaluation -----------------------------------------------------------------
    def _eval(self, node: ast.expr, scope: Dict[str, np.ndarray]):
        if isinstance(node, ast.Constant):
            return np.float64(node.value)
        if isinstance(node, ast.Name):
            nid = node.id
            if nid in scope:
                return scope[nid]
            if nid in self.globals:
                return self.globals[nid]
            if nid in CONSTANTS:
                return np.float64(CONSTANTS[nid])
            if nid in self.base:
                return self.base[nid]
            raise ExpressionError(f"Unknown name '{nid}'.")
        if isinstance(node, ast.BinOp):
            a, b = self._eval(node.left, scope), self._eval(node.right, scope)
            op = node.op
            if isinstance(op, ast.Add):
                return a + b
            if isinstance(op, ast.Sub):
                return a - b
            if isinstance(op, ast.Mult):
                return a * b
            if isinstance(op, ast.Div):
                return np.divide(a, b)
            if isinstance(op, ast.Pow):
                return np.power(a, b)
            if isinstance(op, ast.Mod):
                return np.mod(a, b)
            if isinstance(op, ast.FloorDiv):
                return np.floor(np.divide(a, b))
        if isinstance(node, ast.UnaryOp):
            v = self._eval(node.operand, scope)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.Compare):
            a, b = self._eval(node.left, scope), self._eval(node.comparators[0], scope)
            op = node.ops[0]
            res = (a < b if isinstance(op, ast.Lt) else a <= b if isinstance(op, ast.LtE) else
                   a > b if isinstance(op, ast.Gt) else a >= b if isinstance(op, ast.GtE) else
                   a == b if isinstance(op, ast.Eq) else a != b)
            return np.asarray(res, dtype=np.float64)
        if isinstance(node, ast.IfExp):
            c = self._eval(node.test, scope)
            return np.where(np.asarray(c) != 0, self._eval(node.body, scope), self._eval(node.orelse, scope))
        if isinstance(node, ast.List):
            cached = self.prog.list_cache.get(id(node))
            if cached is not None:
                return cached
            try:
                arr = np.array([float(self._eval(e, scope)) for e in node.elts], dtype=np.float64)
            except TypeError:
                raise ExpressionError("List constants may only contain plain numbers.") from None
            if _is_numeric_list(node):
                self.prog.list_cache[id(node)] = arr
            return arr
        if isinstance(node, ast.Subscript):
            arr = np.asarray(self._eval(node.value, scope))
            idx = np.asarray(self._eval(node.slice, scope))
            if arr.ndim != 1:
                raise ExpressionError("Only list constants can be indexed.")
            if idx.size != 1:
                raise ExpressionError("List index must be a single number (use it inside sum/prod).")
            i = int(round(float(idx.reshape(-1)[0])))
            if not 0 <= i < arr.shape[0]:
                raise ExpressionError(f"List index {i} out of range 0..{arr.shape[0] - 1}.")
            return arr[i]
        if isinstance(node, ast.Call):
            return self._eval_call(node, scope)
        raise ExpressionError(f"Cannot evaluate '{type(node).__name__}'.")

    def _scalar_int(self, node: ast.expr, scope, what: str) -> int:
        val = np.asarray(self._eval(node, scope))
        if val.size != 1 or not np.isfinite(val.reshape(-1)[0]):
            raise ExpressionError(f"The {what} of sum/prod must be a single finite number.")
        return int(round(float(val.reshape(-1)[0])))

    def _eval_call(self, node: ast.Call, scope):
        fid = node.func.id
        if fid in SPECIAL_FORMS:
            body, counter = node.args[0], node.args[1].id
            lo = self._scalar_int(node.args[2], scope, "lower bound")
            hi = self._scalar_int(node.args[3], scope, "upper bound")
            step = self._scalar_int(node.args[4], scope, "step") if len(node.args) == 5 else 1
            if step == 0:
                raise ExpressionError("sum/prod step cannot be 0.")
            count = max(0, (hi - lo) // step + 1) if step > 0 else max(0, (lo - hi) // (-step) + 1)
            if count > MAX_SUM_TERMS:
                raise ExpressionError(f"{fid}() has {count} terms; the maximum is {MAX_SUM_TERMS}.")
            acc = np.float64(0.0 if fid == "sum" else 1.0)
            inner = dict(scope)
            for k in range(lo, hi + (1 if step > 0 else -1), step):
                self._tick()
                inner[counter] = np.float64(k)
                term = self._eval(body, inner)
                if fid == "sum":
                    if isinstance(acc, np.ndarray) and acc.shape == np.shape(term) and acc.ndim:
                        acc += term
                    else:
                        acc = acc + term
                else:
                    if isinstance(acc, np.ndarray) and acc.shape == np.shape(term) and acc.ndim:
                        acc *= term
                    else:
                        acc = acc * term
            return acc
        args = [self._eval(a, scope) for a in node.args]
        if fid in FUNCTIONS:
            fn = FUNCTIONS[fid][0]
            if fid == "log" and len(args) == 2:
                return np.log(args[0]) / np.log(args[1])
            return fn(*args)
        if fid in self.funcs:
            return self._call_user(fid, args)
        # implicit multiplication: a(x+1) where `a` is a variable
        if fid in scope or fid in self.globals or fid in CONSTANTS or fid in self.base:
            base = self._eval(ast.Name(id=fid, ctx=ast.Load()), scope)
            prod = base
            for a in args:
                prod = prod * a
            return prod
        raise ExpressionError(f"Unknown function '{fid}'.")

    def _call_user(self, fid: str, args: list):
        self.depth += 1
        if self.depth > MAX_CALL_DEPTH:
            raise ExpressionError("Function calls are nested too deeply (recursion is not allowed).")
        st = self.funcs[fid]
        local = dict(zip(st.params, args))
        try:
            return self._eval(st.expr, local)
        finally:
            self.depth -= 1


# --------------------------------------------------------------------------------------
# Convenience helpers used by the GUI / CLI / tests
# --------------------------------------------------------------------------------------
def render_source(
    source: str,
    width: int,
    height: int,
    t: float = 0.0,
    m0: Optional[float] = None,
    n0: Optional[float] = None,
    scale: Optional[float] = None,
    cancel: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    """One-shot: compile + render. Defaults to a centred grid with y spanning [-1, 1]."""
    prog = compile_program(source)
    ctx = RenderContext(
        width=width, height=height,
        m0=(width / 2.0 if m0 is None else m0),
        n0=(height / 2.0 if n0 is None else n0),
        scale=(height / 2.0 if scale is None else scale), t=t,
    )
    return prog.render(ctx, cancel=cancel)


@dataclass
class SliderSpec:
    name: str
    value: float
    lo: float
    hi: float
    line_index: int       # 0-based physical line in the source


def find_sliders(source: str) -> List[SliderSpec]:
    """Detects ``name = 3   # [lo, hi]`` lines that should be exposed as live sliders."""
    out: List[SliderSpec] = []
    for i, line in enumerate(source.splitlines()):
        mo = _SLIDER.match(line)
        if mo:
            try:
                out.append(SliderSpec(mo.group("name"), float(mo.group("val")), float(mo.group("lo")),
                                      float(mo.group("hi")), i))
            except ValueError:
                continue
    return out


def set_slider_value(source: str, line_index: int, value: float) -> str:
    lines = source.split("\n")
    mo = _SLIDER.match(lines[line_index].rstrip("\r"))
    if not mo:
        return source
    text = f"{value:.6g}"
    lines[line_index] = f"{mo.group('head')}{text}{mo.group('tail')}"
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# AST -> LaTeX
# --------------------------------------------------------------------------------------
_PREC_ADD, _PREC_MUL, _PREC_NEG, _PREC_POW, _PREC_ATOM = 1, 2, 3, 4, 5
_GREEK = {"pi": r"\pi", "tau": r"\tau", "phi": r"\varphi", "th": r"\theta"}
_FUNC_LATEX = {
    "sin": r"\sin", "cos": r"\cos", "tan": r"\tan", "asin": r"\arcsin", "arcsin": r"\arcsin",
    "acos": r"\arccos", "arccos": r"\arccos", "atan": r"\arctan", "arctan": r"\arctan",
    "sinh": r"\sinh", "cosh": r"\cosh", "tanh": r"\tanh", "ln": r"\ln", "log10": r"\log_{10}", "log2": r"\log_{2}",
    "min": r"\min", "max": r"\max",
}


def _num_latex(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    frac = Fraction(value).limit_denominator(1000)
    if frac.denominator > 1 and abs(float(frac) - value) <= 1e-9 * max(1.0, abs(value)):
        sign = "-" if frac.numerator < 0 else ""
        return rf"{sign}\frac{{{abs(frac.numerator)}}}{{{frac.denominator}}}"
    text = f"{value:.6g}"
    if "e" in text:
        mant, exp = text.split("e")
        return rf"{mant}\times 10^{{{int(exp)}}}"
    return text


def _name_latex(name: str) -> str:
    if name in _GREEK:
        return _GREEK[name]
    if len(name) == 1:
        return name
    if "_" in name:
        head, tail = name.split("_", 1)
        return f"{_name_latex(head)}_{{{tail}}}"
    mo = re.fullmatch(r"([A-Za-z])(\d+)", name)
    if mo:
        return f"{mo.group(1)}_{{{mo.group(2)}}}"
    return rf"\mathrm{{{name}}}"


class _LatexPrinter:
    def __init__(self, user_funcs: Dict[str, Statement]):
        self.user_funcs = user_funcs

    def p(self, node: ast.expr, min_prec: int = 0) -> str:
        text, prec = self._pr(node)
        return rf"\left({text}\right)" if prec < min_prec else text

    def _pr(self, node: ast.expr) -> Tuple[str, int]:
        if isinstance(node, ast.Constant):
            v = float(node.value)
            s = _num_latex(abs(v))
            return (("-" + s), _PREC_NEG) if v < 0 else (s, _PREC_ATOM)
        if isinstance(node, ast.Name):
            return _name_latex(node.id), _PREC_ATOM
        if isinstance(node, ast.UnaryOp):
            inner = self.p(node.operand, _PREC_NEG)
            return (("-" if isinstance(node.op, ast.USub) else "+") + inner), _PREC_NEG
        if isinstance(node, ast.BinOp):
            op = node.op
            if isinstance(op, ast.Add):
                return f"{self.p(node.left, _PREC_ADD)}+{self.p(node.right, _PREC_ADD + 1)}", _PREC_ADD
            if isinstance(op, ast.Sub):
                return f"{self.p(node.left, _PREC_ADD)}-{self.p(node.right, _PREC_ADD + 1)}", _PREC_ADD
            if isinstance(op, ast.Mult):
                left, right = self.p(node.left, _PREC_MUL), self.p(node.right, _PREC_MUL + 1)
                number_left = isinstance(node.left, ast.Constant) or (isinstance(node.left, ast.UnaryOp) and isinstance(node.left.operand, ast.Constant))
                right_starts_digit = right[:1].isdigit() or right.startswith("-")
                joiner = r"\cdot " if (right_starts_digit and (number_left or left[-1:].isdigit())) else r"\,"
                if right.startswith(r"\frac") or left.endswith("}") and right[:1].isdigit():
                    joiner = r"\cdot "
                return f"{left}{joiner}{right}", _PREC_MUL
            if isinstance(op, ast.Div):
                return rf"\frac{{{self.p(node.left)}}}{{{self.p(node.right)}}}", _PREC_ATOM
            if isinstance(op, ast.FloorDiv):
                return rf"\left\lfloor\frac{{{self.p(node.left)}}}{{{self.p(node.right)}}}\right\rfloor", _PREC_ATOM
            if isinstance(op, ast.Mod):
                return rf"{self.p(node.left, _PREC_MUL)}\ \mathrm{{mod}}\ {self.p(node.right, _PREC_MUL + 1)}", _PREC_MUL
            if isinstance(op, ast.Pow):
                base = node.left
                if isinstance(base, ast.Call) and isinstance(base.func, ast.Name) and base.func.id in _FUNC_LATEX and len(base.args) == 1:
                    return rf"{_FUNC_LATEX[base.func.id]}^{{{self.p(node.right)}}}\left({self.p(base.args[0])}\right)", _PREC_ATOM
                return f"{self.p(base, _PREC_ATOM)}^{{{self.p(node.right)}}}", _PREC_POW
        if isinstance(node, ast.Compare):
            sym = {ast.Lt: "<", ast.LtE: r"\leq ", ast.Gt: ">", ast.GtE: r"\geq ", ast.Eq: "=", ast.NotEq: r"\neq "}[type(node.ops[0])]
            return f"{self.p(node.left, _PREC_ADD)}{sym}{self.p(node.comparators[0], _PREC_ADD)}", _PREC_ADD - 1
        if isinstance(node, ast.IfExp):
            return (rf"\mathrm{{if}}\ {self.p(node.test)}\ \mathrm{{then}}\ {self.p(node.body)}"
                    rf"\ \mathrm{{else}}\ {self.p(node.orelse)}"), 0
        if isinstance(node, ast.List):
            vals = [self.p(e) for e in node.elts]
            if len(vals) > 4:
                vals = vals[:3] + [r"\ldots", vals[-1]]
            return r"\left(" + ",".join(vals) + r"\right)", _PREC_ATOM
        if isinstance(node, ast.Subscript):
            return f"{self.p(node.value, _PREC_ATOM)}_{{{self.p(node.slice)}}}", _PREC_ATOM
        if isinstance(node, ast.Call):
            return self._call(node)
        return "?", _PREC_ATOM

    def _call(self, node: ast.Call) -> Tuple[str, int]:
        fid = node.func.id
        args = node.args
        if fid in SPECIAL_FORMS:
            sym = r"\sum" if fid == "sum" else r"\prod"
            counter = args[1].id
            lo, hi = self.p(args[2]), self.p(args[3])
            return rf"{sym}_{{{_name_latex(counter)}={lo}}}^{{{hi}}}{self.p(args[0], _PREC_MUL)}", _PREC_MUL
        a = [self.p(x) for x in args]
        if fid == "sqrt":
            return rf"\sqrt{{{a[0]}}}", _PREC_ATOM
        if fid == "cbrt":
            return rf"\sqrt[3]{{{a[0]}}}", _PREC_ATOM
        if fid == "exp":
            return f"e^{{{a[0]}}}", _PREC_POW
        if fid == "abs":
            return rf"\left|{a[0]}\right|", _PREC_ATOM
        if fid == "floor":
            return rf"\left\lfloor {a[0]}\right\rfloor", _PREC_ATOM
        if fid == "ceil":
            return rf"\left\lceil {a[0]}\right\rceil", _PREC_ATOM
        if fid == "log" and len(a) == 2:
            return rf"\log_{{{a[1]}}}\left({a[0]}\right)", _PREC_ATOM
        if fid == "log":
            return rf"\ln\left({a[0]}\right)", _PREC_ATOM
        if fid == "pow":
            return f"{self.p(args[0], _PREC_ATOM)}^{{{a[1]}}}", _PREC_POW
        if fid == "mod":
            return rf"{self.p(args[0], _PREC_MUL)}\ \mathrm{{mod}}\ {self.p(args[1], _PREC_MUL + 1)}", _PREC_MUL
        if fid in ("atan2", "arctan2"):
            return rf"\mathrm{{atan2}}\left({a[0]},{a[1]}\right)", _PREC_ATOM
        if fid in _FUNC_LATEX:
            return rf"{_FUNC_LATEX[fid]}\left({','.join(a)}\right)", _PREC_ATOM
        name = _name_latex(fid) if fid in self.user_funcs else rf"\mathrm{{{fid}}}"
        return rf"{name}\left({','.join(a)}\right)", _PREC_ATOM


def to_latex(prog: Program, max_len: int = 1800) -> List[str]:
    """LaTeX lines (one per statement) plus the quantiser definition."""
    printer = _LatexPrinter(prog.function_defs)
    lines: List[str] = []
    for st in prog.statements:
        if not st.name:
            continue
        lhs = _name_latex(st.name)
        if st.params is not None:
            lhs += r"\left(" + ",".join(_name_latex(p) for p in st.params) + r"\right)"
        rhs = printer.p(st.expr)
        if len(rhs) > max_len:
            rhs = rhs[:max_len].rsplit("\\", 1)[0] + r"\ldots"
        lines.append(f"{lhs}={rhs}")
    if prog.output_mode == "RGB":
        out = r"\mathrm{rgb}\left(F(R),F(G),F(B)\right)"
    else:
        out = (r"\mathrm{rgb}\left(F\left(H_{0}\right),F\left(H_{1}\right),F\left(H_{2}\right)\right),"
               r"\ \ H_{s}=H(x,y,s)")
    lines.append(out + r",\quad F(x)=\left\lfloor 255\,e^{-e^{-1000x}}\left|x\right|^{e^{-e^{1000(x-1)}}}\right\rfloor")
    return lines
