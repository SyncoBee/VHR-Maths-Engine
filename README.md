# VHR Maths Engine Pro v4.0
### Bidirectional High-Precision Algebraic Image Engine & Real-Time Mathematical Studio
**Created & Published by Vihanga Himantha**

The **VHR Maths Engine Pro** is an advanced high-precision, real-time mathematical framework and GUI/CLI studio created and published by **Vihanga Himantha**. It bridges the gap between **continuous analytical mathematical equations** and **digital discrete raster images**, providing:

1. **Equation $\rightarrow$ Image (Live Synthesis)**: An interactive, real-time formula workstation. Type mathematical expressions, define functions, compute sums and products, animate over time $t$, and adjust parameters live with auto-detected sliders.
2. **Image $\rightarrow$ Equation (Precision Decomposition)**: Convert any photograph, sketch, or texture into a closed-form analytical mathematical equation with guaranteed PSNR/SSIM quality bounds, sub-pixel Levenberg-Marquardt primitive fitting, and live millisecond coefficient tuning.
3. **Cosmic Reference Catalog**: Verbatim mathematical implementations of famous cosmic plates (*Galaxy Cluster* and *Colliding Planets*).
4. **Multi-Target Exporters**: Export closed-form equations directly to standalone Python renderers, compilable LaTeX documents with exact continued-fraction tables, portable `.vhr` programs, and JSON models.

---

## 1. Mathematical Architecture

### Branchless Quantization Operator $F(x)$
Digital images require discrete 8-bit color channels ($0 \dots 255$). Standard rounding introduces discontinuities and conditionals. The VHR Maths Engine uses the branchless, infinitely differentiable clamping operator:

$$F(x) = \left\lfloor 255 \cdot e^{-e^{-1000x}} \cdot |x|^{e^{-e^{1000(x-1)}}} \right\rfloor$$

- When $x \le 0$, $e^{-e^{-1000x}} \to 0$, clamping the output smoothly to $0$.
- When $0 < x < 1$, both double-exponential factors evaluate to $1$, yielding $\lfloor 255 \cdot x \rfloor$.
- When $x \ge 1$, the exponent $e^{-e^{1000(x-1)}} \to 0$, collapsing $|x|^0 = 1$ and clamping the output smoothly to $255$.

### Exact Lagrange Quadratic Color Multiplexing
Rather than storing three separate formulas for Red, Green, and Blue, the engine multiplexes the three color channels into a single polynomial in channel variable $s \in \{0, 1, 2\}$:

$$P(s) = c_0 + c_1 s + c_2 s^2$$

Given target channel intensities $(R, G, B)$ at $s = (0, 1, 2)$, the exact algebraic coefficients are computed in closed form:
$$c_0 = R$$
$$c_1 = -1.5 R + 2.0 G - 0.5 B$$
$$c_2 = 0.5 R - 1.0 G + 0.5 B$$

### Anisotropic Double-Exponential Primitives $A_p(x, y)$
High-contrast focal features, stars, cores, and discs are modeled as rotated anisotropic double-exponential primitives:

$$A_p(x, y) = \exp\left(-\exp\left(\operatorname{clip}\left(k_p \cdot (U_p^2 + V_p^2 - 1), -40, 40\right)\right)\right)$$
$$\begin{pmatrix} U_p \\ V_p \end{pmatrix} = \frac{1}{\begin{pmatrix} a_p \\ b_p \end{pmatrix}} \begin{pmatrix} \cos\theta_p & \sin\theta_p \\ -\sin\theta_p & \cos\theta_p \end{pmatrix} \begin{pmatrix} x - x_{0,p} \\ y - y_{0,p} \end{pmatrix}$$

Coverage over $P$ primitives is combined using the product-subtraction rule:
$$P_s(x, y) = 1 - \prod_{p=1}^P \left(1 - W_{p,s} \cdot A_p(x, y)\right)$$
where $W_{p,s} = c_{0,p} + c_{1,p} s + c_{2,p} s^2$ are signed, unclipped channel weights. Each candidate primitive is evaluated with sub-pixel Levenberg-Marquardt refinement and accepted **only if** it measurably decreases global residual energy (greedy A/B test).

### Adaptive Orthonormal 2D Cosine Series $C_s(x, y)$
The residual image $R_s(x, y) = I_s(x, y) - P_s(x, y)$ is decomposed into a complete 2D orthonormal Discrete Cosine Transform:

$$C_s(x, y) = \sum_{k=1}^K \left(\alpha_{0,k} + \alpha_{1,k} s + \alpha_{2,k} s^2\right) \cos\left(\frac{v_k \pi (m - 1/2)}{W}\right) \cos\left(\frac{u_k \pi (n - 1/2)}{H}\right)$$

- **Parseval Energy Budgeting**: Coefficients are sorted by total energy across RGB. By Parseval's theorem, the exact PSNR achievable for any term count $K$ is known prior to reconstruction.
- **Floor Quantizer Compensation**: Targeting $(I + 0.5) / 255$ shifts the effective quantization interval, removing the systematic $-0.5$ intensity bias of floor operations.
- **Saturation-Aware Multi-Pass Refinement**: Pixels that clip to $0$ or $255$ under $F(x)$ do not need exact linear matching outside $[0, 1]$. Iterative refinement redirects basis energy toward unsaturated gradients, improving effective dynamic range.

### Continued-Fraction Rationalization
To produce exact, elegant mathematical formulas, floating-point coefficients are rationalized into fractions $\frac{p}{q}$ using continued fractions (`limit_denominator(max_den)`). A built-in validation check ensures that if a denominator cap would compromise fidelity, an honest decimal expansion is emitted rather than a truncated fraction.

---

## 2. Key Improvements & Real-Time Advancements

| Area | Status in v4.0 |
| :--- | :--- |
| **Equation $\rightarrow$ Image** | **Full Expression Engine** supporting safe AST evaluation, user functions, sums, prods, lists, custom variables ($x, y, r, \theta, s, t, m, n$), and automatic LaTeX generation. |
| **Real-Time Responsiveness** | **Background JobRunner with cooperative cancellation** and debounced live progressive rendering ($<15$ ms updates). |
| **Image Aspect Ratio** | **EXIF-aware adaptive grid mapping** (`GridConfig.for_image`) and aspect-ratio preservation. |
| **Primitive Optimization** | **Bounded Levenberg-Marquardt solver** with signed channel weights and greedy A/B acceptance filter. |
| **Formula LaTeX Formatting** | **Sanitized Mathtext Printer** with syntax-safe substitutions (`\mathrm{mod}`, `\resizebox`, math fences) and pre-flight validation. |
| **Export Fidelity** | **Bit-exact round-trip**: independent expression engine and exported Python script produce identical uint8 arrays (`Max diff = 0`). |
| **Accurate Telemetry** | Measured metrics are reported only when a ground-truth image exists; catalog formulas report term counts and render time. |

---

## 3. Quick Start

### Installation
Clone or navigate to the repository directory and install dependencies:

```bash
# Using the virtual environment (recommended)
pip install -r requirements.txt
```

### Launch the GUI Studio
Double-click `run.bat` (Windows) or execute:

```bash
python main.py
```

### Command-Line Interface (CLI)

#### 1. Render an Equation to an Image
```bash
python cli.py render-eq "H(x,y,s) = 0.5 + 0.5*sin(12*r - 2*pi*s/3)" -o output.png --width 800 --height 600
```
Render with animation time $t$:
```bash
python cli.py render-eq "H(x,y,s) = 0.5 + 0.5*sin(8*x + 3*t + 2*pi*s/3)" -t 1.5 -o frame_t1.5.png
```

#### 2. Convert an Image into Closed-Form Equations
```bash
python cli.py img2eq input.png --psnr 40.0 --max-terms 2000 -o recon.png --output-vhr equation.vhr --output-py render.py --output-tex equation.tex
```

#### 3. Render a Catalog Plate
```bash
python cli.py catalog galaxy -o galaxy.png --width 800 --height 480
python cli.py catalog planets -o planets.png --width 800 --height 480
```

---

## 4. Equation Syntax Guide

The expression engine supports plain-text mathematical programming:

```python
# Grid configuration (optional)
#! grid 640 360 320 180 180

# Sliders: Any variable with a # [min, max] comment generates a live UI slider
freq = 15.0     # [2, 50]
arms = 3        # [1, 8]

# Coordinates available: x, y, r (radius), th (theta), m, n (pixel index), s (channel 0,1,2), t (time)
r_core = exp(-8 * r^2)
spiral = 0.5 + 0.5 * cos(arms * (th - 2 * ln(r + 0.1)) - 4 * t)

# User-defined functions
disc(cx, cy, rad) = exp(-exp(40 * ((x - cx)^2 + (y - cy)^2 - rad^2)))

# Sums and products: sum(expression, index_var, start, end)
stars = sum(exp(-2000 * ((x - 0.7*cos(k*1.7))^2 + (y - 0.4*sin(k*2.1))^2)), k, 1, 40)

# Output definitions: Define H(x, y, s) or separate R, G, B
H(x, y, s) = (0.7*r_core + 0.3*spiral + stars) * (0.4 + 0.6*sin(2*pi*s/3)^2)
```

### Supported Functions
- **Trigonometric**: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`
- **Exponentials & Logarithms**: `exp`, `ln`, `log`, `log10`, `log2`, `sqrt`, `cbrt`, `pow` (`^` or `**`)
- **Rounding & Shaping**: `abs`, `floor`, `ceil`, `round`, `sign`, `fract`, `min`, `max`, `clip`, `smoothstep`, `step`, `mix`, `where`
- **Iterators**: `sum(expr, k, lo, hi)`, `prod(expr, k, lo, hi)`
- **Arrays**: `A = [1.2, 3.4, 5.6]` with indexing `A[k]` inside sums/products.

---

## 5. Verification & Testing

A complete automated test suite is included in `tests/`:

```bash
pytest tests/ -v
```

Tests cover:
- Mathematical correctness of $F(x)$, Lagrange interpolation, and coordinate transformations.
- AST security validation, budget enforcement, and recursion protection.
- Monotonic rate-distortion behavior and Parseval quality estimation.
- Bit-exact reconstruction across internal DCT, standalone Python script, and expression engine evaluator.
- Compilable LaTeX document structure and continued-fraction honesty.

---

## 6. Publication & Ownership

**VHR Maths Engine Pro** is created, authored, and published by **Vihanga Himantha**.  
Copyright © 2026 Vihanga Himantha. All rights reserved.
