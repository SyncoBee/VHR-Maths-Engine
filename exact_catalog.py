import time
from typing import Optional, Callable
import numpy as np
from models import GridConfig, VHRModel, QualityMetrics
from algebraic_core import AlgebraicCore


class RenderCancelled(Exception):
    """Raised when the caller's cancel callback reports True (a newer render request superseded this one)."""


class VHRExactCatalog:
    """
    Verbatim mathematical formulations of cosmic plates in the VHR Maths Engine
    ('Galaxy Cluster' and 'Colliding Planets'), created and published by Vihanga Himantha.
    """

    @staticmethod
    def build_galaxy_cluster_model(grid: GridConfig, galaxy_count: int = 300, star_count: int = 300, wave_count: int = 50) -> VHRModel:
        W, H, m0, n0, S = grid.width, grid.height, grid.m0, grid.n0, int(grid.scale)
        latex = [
            rf"\mathrm{{Galaxy\ Cluster\ by\ Vihanga\ Himantha\ }} ({W}\times {H}).\ \mathrm{{For\ }} m=1,\dots,{W} \mathrm{{\ and\ }} n=1,\dots,{H}:",
            rf"\mathrm{{rgb}}\left(F\left(R\left(\frac{{m-{m0}}}{{{S}}},\frac{{{n0}-n}}{{{S}}}\right)\right), F\left(G\left(\frac{{m-{m0}}}{{{S}}},\frac{{{n0}-n}}{{{S}}}\right)\right), F\left(B\left(\frac{{m-{m0}}}{{{S}}},\frac{{{n0}-n}}{{{S}}}\right)\right)\right), \quad F(x)=\left\lfloor 255e^{{-e^{{-1000x}}}}|x|^{{e^{{-e^{{1000(x-1)}}}}}}\right\rfloor, \quad R=H_0,\ G=H_1,\ B=H_2",
            rf"H_s(x,y) = \left(1 + \frac{{E_{{s+1}}(x,y)}}{{30}} - \prod_{{p=1}}^{{{galaxy_count}}}\prod_{{r=1}}^{{10+5(-1)^p}}\left(1 - A_{{r,p}}(U_p(x,y),V_p(x,y))\right)\right)\left(\frac{{1}}{{5}} + \left(\frac{{1}}{{5}}-\frac{{s^2}}{{40}}\right)C_1 + \frac{{4+(2-s)^2}}{{40}}C_2 + \frac{{20+3(2-s)^2}}{{200}}C_3 + \left(\frac{{1}}{{2}}-\frac{{1}}{{10}}\left(s^2+\frac{{(-1)^s}}{{2}}\right)\right)E_1 + \frac{{s^2}}{{10}}E_2 + \frac{{9}}{{100}}\left((2-s)^2+\frac{{(-1)^s}}{{2}}\right)E_3\right)",
            rf"U_s(x,y)=\frac{{300}}{{121}}\left(\frac{{253}}{{250}}\right)^s(1+8|\sin(s^4)|)\left(x+y\sin\left(s^6-\frac{{1}}{{4}}\right)+\frac{{121}}{{100}}\cos(s^2)\right), \quad V_s(x,y)=\frac{{300}}{{121}}\left(\frac{{253}}{{250}}\right)^s(1+6|\sin(s^5)|)\left(x\sin\left(s^6-\frac{{1}}{{4}}\right)-y+\frac{{121}}{{100}}\cos(s^3)\right)",
            rf"A_{{r,s}}(x,y)=e^{{-e^{{(8+5\sin(s^4))\left(\left(\sqrt{{x^2+y^2}}-\frac{{13}}{{10}}\cos^2(r^4)\right)^2-\frac{{1}}{{10}}\right)}} - 7(1+\sin(s^5))\left(\left(x-\sqrt{{x^2+y^2}}\cos\left(\frac{{2r\pi}}{{8}}-\frac{{5+2\sin(s^4)}}{{4}}\ln\left(x^2+y^2+\frac{{1}}{{10}}|\cos(15\sqrt{{x^2+y^2}})|\right)\right)\right)^2 + \left(y-\sqrt{{x^2+y^2}}\sin\left(\frac{{2r\pi}}{{8}}-\frac{{5+2\sin(s^4)}}{{4}}\ln\left(x^2+y^2+\frac{{1}}{{10}}|\cos(15\sqrt{{x^2+y^2}})|\right)\right)\right)^2\right)^2}}"
        ]
        return VHRModel(
            title="Galaxy Cluster (VHR reference-style formula)",
            engine_mode="exact_galaxy_cluster",
            grid=grid,
            extra_params={"galaxy_count": galaxy_count, "star_count": star_count, "wave_count": wave_count},
            latex_lines=latex
        )

    @staticmethod
    def render_galaxy_cluster(model: VHRModel, progress_cb: Optional[Callable[[float, str], None]] = None,
                              cancel: Optional[Callable[[], bool]] = None) -> np.ndarray:
        t0 = time.perf_counter()
        grid = model.grid
        P_max = int(model.extra_params.get("galaxy_count", 120))
        K_star = int(model.extra_params.get("star_count", 150))
        K_wave = int(model.extra_params.get("wave_count", 50))

        out_img = np.zeros((grid.height, grid.width, 3), dtype=np.uint8)
        m_row = np.arange(1, grid.width + 1, dtype=np.float64)
        x_row = (m_row - grid.m0) / grid.scale
        total_chunks = max(1, (grid.height + grid.chunk_rows - 1) // grid.chunk_rows)

        for idx, r_start in enumerate(range(0, grid.height, grid.chunk_rows)):
            if cancel is not None and cancel():
                raise RenderCancelled()
            r_end = min(grid.height, r_start + grid.chunk_rows)
            n_col = np.arange(r_start + 1, r_end + 1, dtype=np.float64)[:, None]
            y = (grid.n0 - n_col) / grid.scale
            x = np.broadcast_to(x_row, (r_end - r_start, grid.width))

            C = {}
            for s_idx in (1, 2, 3):
                c_acc = np.zeros_like(x)
                sm1 = s_idx - 1
                for k in range(1, K_wave + 1):
                    arg_x = (7.0 - 2.0 * sm1) * x + (2.0 * k * x) / (2.0 + sm1) + (float(k) ** (3 + sm1))
                    arg_y = (7.0 - 3.0 * sm1) * y + (2.0 * k * y) / (2.0 + sm1) + (float(k) ** (4 + sm1))
                    c_acc += 4.0 * ((np.cos(arg_x) * np.cos(arg_y)) ** (50 + 8 * k))
                C[s_idx] = c_acc

            E = {}
            for s_idx in (1, 2, 3):
                e_acc = np.zeros_like(x)
                sm1 = s_idx - 1
                for k in range(1, K_star + 1):
                    fk = float(k)
                    weight = 0.5 * (1.0 + np.sin(20.0 * (fk ** (3 + sm1))))
                    if weight < 1e-5:
                        continue
                    sin_k6 = np.sin((fk ** 6) - 0.25)
                    dx = (100.0 * x) / 121.0 + (100.0 * y) / 121.0 * sin_k6 + np.cos(fk ** 2)
                    dy = (100.0 * x) / 121.0 * sin_k6 - (100.0 * y) / 121.0 + np.cos(fk ** 3)
                    dist4 = (dx ** 2 + dy ** 2) ** 2
                    decay_rate = 5.0 + min((1.1 ** k), 1e6)
                    e_acc += weight * np.exp(np.clip(-decay_rate * dist4, -60.0, 0.0))
                E[s_idx] = e_acc

            prod_p = np.ones_like(x)
            for p in range(1, P_max + 1):
                if cancel is not None and cancel():
                    raise RenderCancelled()
                fp = float(p)
                scale_p = (300.0 / 121.0) * ((253.0 / 250.0) ** p)
                sin_p6 = np.sin((fp ** 6) - 0.25)

                Up = scale_p * (1.0 + 8.0 * abs(np.sin(fp ** 4))) * (x + y * sin_p6 + 1.21 * np.cos(fp ** 2))
                Vp = scale_p * (1.0 + 6.0 * abs(np.sin(fp ** 5))) * (x * sin_p6 - y + 1.21 * np.cos(fp ** 3))

                rad_sq = Up ** 2 + Vp ** 2
                if np.min(rad_sq) > 3.5:
                    continue

                rad = np.sqrt(rad_sq + 1e-12)
                log_term = np.log(rad_sq + 0.1 * np.abs(np.cos(15.0 * rad)) + 1e-12)
                pitch_p = (5.0 + 2.0 * np.sin(fp ** 4)) / 4.0
                radial_sharp = 8.0 + 5.0 * np.sin(fp ** 4)
                arm_sharp = 7.0 * (1.0 + np.sin(fp ** 5))

                r_max = 15 if (p % 2 == 0) else 5
                for r in range(1, r_max + 1):
                    fr = float(r)
                    t1_exp = np.clip(radial_sharp * (((rad - 1.3 * (np.cos(fr ** 4) ** 2)) ** 2) - 0.1), -40.0, 40.0)
                    theta_r = (2.0 * fr * np.pi) / 8.0 - pitch_p * log_term
                    arm_dx = Up - rad * np.cos(theta_r)
                    arm_dy = Vp - rad * np.sin(theta_r)
                    t2 = arm_sharp * ((arm_dx ** 2 + arm_dy ** 2) ** 2)

                    A_rp = np.exp(np.clip(-np.exp(t1_exp) - t2, -60.0, 0.0))
                    prod_p *= (1.0 - A_rp)

            for s in range(3):
                fs = float(s)
                minus_one_s = 1.0 if (s % 2 == 0) else -1.0
                color_factor = (
                    0.2
                    + (0.2 - (fs ** 2) / 40.0) * C[1]
                    + ((4.0 + ((2.0 - fs) ** 2)) / 40.0) * C[2]
                    + ((20.0 + 3.0 * ((2.0 - fs) ** 2)) / 200.0) * C[3]
                    + (0.5 - 0.1 * (fs ** 2 + 0.5 * minus_one_s)) * E[1]
                    + ((fs ** 2) / 10.0) * E[2]
                    + 0.09 * (((2.0 - fs) ** 2) + 0.5 * minus_one_s) * E[3]
                )
                H_s = (1.0 + E[s + 1] / 30.0 - prod_p) * color_factor
                out_img[r_start:r_end, :, s] = AlgebraicCore.F_quantize(H_s)

            if progress_cb:
                progress_cb((idx + 1) / total_chunks, f"[Exact Galaxy Cluster] Scanlines {r_start+1}..{r_end}/{grid.height}")

        model.metrics = QualityMetrics(
            psnr_db=0.0, ssim=0.0, mae_255=0.0, max_err_255=0.0, measured=False,
            total_terms=P_max * 10 + K_star + K_wave,
            render_time_sec=round(time.perf_counter() - t0, 2)
        )
        return out_img


    @staticmethod
    def build_colliding_planets_model(grid: GridConfig, Z_terms: int = 50, K_terms: int = 50) -> VHRModel:
        W, H, m0, n0, S = grid.width, grid.height, grid.m0, grid.n0, int(grid.scale)
        latex = [
            rf"\mathrm{{Colliding\ Planets\ by\ Vihanga\ Himantha\ }} ({W}\times {H}).\ \mathrm{{For\ }} m=1,\dots,{W} \mathrm{{\ and\ }} n=1,\dots,{H}:",
            rf"\mathrm{{rgb}}\left(F\left(H_0\left(\frac{{m-{m0}}}{{{S}}},\frac{{{n0}-n}}{{{S}}}\right)\right), F\left(H_1\left(\frac{{m-{m0}}}{{{S}}},\frac{{{n0}-n}}{{{S}}}\right)\right), F\left(H_2\left(\frac{{m-{m0}}}{{{S}}},\frac{{{n0}-n}}{{{S}}}\right)\right)\right), \quad F(x)=\left\lfloor 255e^{{-e^{{-1000x}}}}|x|^{{e^{{-e^{{1000(x-1)}}}}}}\right\rfloor",
            rf"H_v(x,y) = \frac{{3v^2-11v+16}}{{4}}\left(\left(\frac{{6}}{{5}}-U_{{100,-3/20,7/10,1}}-\frac{{1}}{{5}}U_{{5,-3/20,7/10,1}}\right)B + \left(\frac{{6}}{{5}}-U_{{100,-17/100,2/5,0}}-\frac{{1}}{{5}}U_{{5,-17/100,2/5,0}}\right)A\right) + \frac{{3v^2-11v+16}}{{2}}C(x,y) + A\cdot\mathrm{{Shade}}_A + B\cdot\mathrm{{Shade}}_B",
            rf"A(x,y)=e^{{-e^{{70M(x,y)-28+\frac{{7}}{{20}}E(x,y)}}}}, \quad B(x,y)=e^{{-e^{{70N(x,y)-14+\frac{{7}}{{20}}E(x,y)}}}}, \quad M(x,y)=\left(x+\frac{{y}}{{4}}-\frac{{7}}{{20}}\right)^2+\left(y-\frac{{x}}{{4}}+\frac{{7}}{{20}}\right)^2, \quad N(x,y)=\left(x+\frac{{y}}{{4}}+\frac{{1}}{{5}}\right)^2+\left(y-\frac{{x}}{{4}}-\frac{{27}}{{50}}\right)^2",
            rf"E(x,y)=\sum_{{s=1}}^{{{Z_terms}}} Z_s(x,y), \quad R_s(x,y)=\arccos(\cos((30+10\cos(17s))x-15y+4\cos(12s)))+\frac{{2}}{{5}}\arccos(\cos(70x+85y+4\cos(34s)))+\dots"
        ]
        return VHRModel(
            title="Colliding Planets (VHR reference-style formula)",
            engine_mode="exact_colliding_planets",
            grid=grid,
            extra_params={"Z_terms": Z_terms, "K_terms": K_terms},
            latex_lines=latex
        )

    @staticmethod
    def render_colliding_planets(model: VHRModel, progress_cb: Optional[Callable[[float, str], None]] = None,
                                 cancel: Optional[Callable[[], bool]] = None) -> np.ndarray:
        t0 = time.perf_counter()
        grid = model.grid
        Z_max = int(model.extra_params.get("Z_terms", 50))
        K_max = int(model.extra_params.get("K_terms", 50))

        out_img = np.zeros((grid.height, grid.width, 3), dtype=np.uint8)
        m_row = np.arange(1, grid.width + 1, dtype=np.float64)
        x_row = (m_row - grid.m0) / grid.scale
        total_chunks = max(1, (grid.height + grid.chunk_rows - 1) // grid.chunk_rows)

        def eval_U(v_param: float, u_param: float, w_param: float, t_param: float, x: np.ndarray, y: np.ndarray, E: np.ndarray) -> np.ndarray:
            L_t = np.sqrt(np.abs(7600.0 - 4000.0 * t_param - ((100.0 * y - 100.0 * x + 65.0 - 142.0 * t_param) ** 2))) / 100.0
            prod_u = np.ones_like(x)
            sign_t = -1.0 if int(t_param) == 1 else 1.0
            xy_u = x + y + u_param
            yx_w = y - x + w_param

            for s in range(1, 6):
                fs = float(s)
                R_s = (
                    np.arccos(np.clip(np.cos((30.0 + 10.0 * np.cos(17.0 * fs)) * x - 15.0 * y + 4.0 * np.cos(12.0 * fs)), -1.0, 1.0))
                    + 0.4 * np.arccos(np.clip(np.cos(70.0 * x + 85.0 * y + 4.0 * np.cos(34.0 * fs)), -1.0, 1.0))
                    + 0.1 * np.arccos(np.clip(np.cos(140.0 * x + 100.0 * y + 4.0 * np.cos(36.0 * fs)), -1.0, 1.0))
                    + 0.05 * np.arccos(np.clip(np.cos(270.0 * x - 200.0 * y + 4.0 * np.cos(32.0 * fs)), -1.0, 1.0))
                )
                extra_ridge = np.arccos(np.clip(np.cos((10.0 + 4.0 * np.cos(15.0 * fs)) * x + 5.0 * y + 4.0 * np.cos(19.0 * fs)), -1.0, 1.0))
                cos_inner = (
                    (16.0 + 2.0 * np.cos(2.0 * fs)) * (100.0 * xy_u) / (1.0 + 100.0 * L_t)
                    + (1.0 + t_param) * (np.abs(xy_u) - L_t) * (extra_ridge + R_s)
                    + 4.0 * np.cos(27.0 * fs)
                )
                atan_inner = -400.0 * sign_t * (
                    yx_w + ((v_param - 5.0) / 475.0) * sign_t
                    + (np.cos(8.0 * fs) / 5.0) * np.cos(7.0 * xy_u + 3.0 * np.cos(14.0 * fs))
                )
                bracket = (
                    np.cos(cos_inner) - 3.01 + E / 25.0
                    + 0.3 * (yx_w ** 3) * sign_t
                    - (4.0 / np.pi) * np.arctan(atan_inner)
                )
                prod_u *= np.exp(-np.exp(np.clip(v_param * bracket, -45.0, 45.0)))
            return prod_u

        for idx, r_start in enumerate(range(0, grid.height, grid.chunk_rows)):
            if cancel is not None and cancel():
                raise RenderCancelled()
            r_end = min(grid.height, r_start + grid.chunk_rows)
            n_col = np.arange(r_start + 1, r_end + 1, dtype=np.float64)[:, None]
            y = (grid.n0 - n_col) / grid.scale
            x = np.broadcast_to(x_row, (r_end - r_start, grid.width))

            E = np.zeros_like(x)
            for s in range(1, Z_max + 1):
                if cancel is not None and cancel():
                    raise RenderCancelled()
                fs = float(s)
                amp = 0.3 * ((25.0 / 26.0) ** s)
                f_scale = (1.1 ** s)
                c10 = f_scale * (1.0 + np.cos(10.0 * fs))
                s2 = 2.0 * (fs ** 2)
                c_s2, s_s2 = np.cos(s2), np.sin(s2)

                term1 = np.cos(
                    c10 * (c_s2 * x + s_s2 * y + 2.0 * np.cos(17.0 * fs))
                    + 3.0 * np.cos(f_scale * (np.cos(7.0 * fs**2) * x + np.sin(7.0 * fs**2) * y))
                    + 2.0 * np.cos(5.0 * fs)
                )
                term2 = np.cos(
                    c10 * (c_s2 * y - s_s2 * x + 2.0 * np.cos(15.0 * fs))
                    + 3.0 * np.cos(f_scale * (np.cos(8.0 * fs**2) * x + np.sin(8.0 * fs**2) * y))
                    + 2.0 * np.cos(7.0 * fs)
                )
                E += amp * term1 * term2

            M = (x + 0.25 * y - 0.35) ** 2 + (y - 0.25 * x + 0.35) ** 2
            N = (x + 0.25 * y + 0.20) ** 2 + (y - 0.25 * x - 0.54) ** 2

            A = np.exp(-np.exp(np.clip(70.0 * M - 28.0 + 0.35 * E, -45.0, 45.0)))
            B = np.exp(-np.exp(np.clip(70.0 * N - 14.0 + 0.35 * E, -45.0, 45.0)))

            P = (
                1.5 * A * np.sqrt(M) * np.arctan((1000.0 * y - 250.0 * x + 350.0) / (1.0 + np.abs(1000.0 * x + 250.0 * y - 350.0)))
                + 1.5 * B * np.sqrt(N) * np.arctan((1000.0 * y - 250.0 * x - 540.0) / (1.0 + np.abs(1000.0 * x + 250.0 * y + 200.0)))
            )
            q_a = np.clip((2.0 ** -0.5) * A * np.sqrt(np.abs(2.0 - 5.0 * M)), -1.0, 1.0)
            q_b = np.clip((2.0 ** -0.5) * B * np.sqrt(np.abs(2.0 - 5.0 * N)), -1.0, 1.0)
            Q = A * np.arcsin(q_a) + B * np.arcsin(q_b)

            c_exp = (
                40.0 * ((y - x - 0.19 + 0.002 * ((10.0 * x + 10.0 * y - 1.0) ** 2)) ** 2)
                - 0.16 + 0.8 * E + 40.0 * ((0.5 * x + 0.5 * y - 0.07) ** 4)
            )
            C_val = np.exp(-np.exp(np.clip(c_exp, -45.0, 45.0)))

            D_val = np.exp(np.clip(2.5 - (5.0 * (100.0 * x + 25.0 * y + 20.0)) / (1.0 + np.sqrt(np.abs(2000.0 - ((100.0 * y - 25.0 * x - 54.0) ** 2)))), -40.0, 40.0))
            T_val = np.exp(np.clip(2.0 - (5.0 * (100.0 * x + 25.0 * y - 35.0)) / (1.0 + np.sqrt(np.abs(4000.0 - ((100.0 * y - 25.0 * x + 35.0) ** 2)))), -40.0, 40.0))

            U_100_B = eval_U(100.0, -0.15, 0.70, 1.0, x, y, E)
            U_5_B = eval_U(5.0, -0.15, 0.70, 1.0, x, y, E)
            U_100_A = eval_U(100.0, -0.17, 0.40, 0.0, x, y, E)
            U_5_A = eval_U(5.0, -0.17, 0.40, 0.0, x, y, E)

            K_channels = [np.zeros_like(x) for _ in range(3)]
            for s in range(1, K_max + 1):
                fs = float(s)
                freq_s = (5.0 + 4.0 * fs) / 5.0
                c4s, s4s = np.cos(4.0 * fs), np.sin(4.0 * fs)
                w1 = np.cos(freq_s * (c4s * Q + s4s * P) + 10.0 * np.sin(10.0 * fs)) ** 2
                w2 = np.cos(freq_s * (s4s * Q - c4s * P) + 10.0 * np.sin(9.0 * fs)) ** 2
                env = np.exp(-np.exp(np.clip(-20.0 * (w1 * w2 + (E - 7.0) / 10.0), -45.0, 45.0)))

                for v in range(3):
                    fv = float(v)
                    coeff = 3.0 * (
                        (A / 20.0) * ((5.0 - fv) / 5.0 + 3.0 * np.cos((5.0 + fv) * fs) / 20.0)
                        + (B / 20.0) * ((14.0 + 3.0 * fv - fv**2) / 20.0 + 3.0 * np.cos((4.0 + 2.0 * fv) * fs) / 20.0)
                    )
                    K_channels[v] += coeff * env

            hemisphere_A = np.exp(-np.exp(np.clip(500.0 * (y - x - 0.18), -45.0, 45.0)))
            hemisphere_B = np.exp(-np.exp(np.clip(-500.0 * (y - x - 0.18), -45.0, 45.0)))
            extra_B_decay = np.exp(np.clip(-D_val - np.exp(np.clip(-20.0 * (y - 0.25 * x - 0.30), -40.0, 40.0)), -50.0, 0.0))
            extra_A_decay = np.exp(np.clip(-T_val, -50.0, 0.0))

            for v in range(3):
                fv = float(v)
                fire_w = (3.0 * (fv ** 2) - 11.0 * fv + 16.0) / 4.0
                Kv = K_channels[v]

                term_cracks = fire_w * (
                    (1.2 - U_100_B - 0.2 * U_5_B) * B
                    + (1.2 - U_100_A - 0.2 * U_5_A) * A
                )
                term_core = (2.0 * fire_w) * C_val
                term_surf_A = A * hemisphere_A * (((10.0 - 3.0 * fv) / 100.0) * (1.25 + Kv / 3.0) + Kv * extra_A_decay)
                term_surf_B = B * hemisphere_B * (((10.0 - 3.0 * fv) / 100.0) * (1.25 + Kv / 3.0) + Kv * extra_B_decay)

                H_v = term_cracks + term_core + term_surf_A + term_surf_B
                out_img[r_start:r_end, :, v] = AlgebraicCore.F_quantize(H_v)

            if progress_cb:
                progress_cb((idx + 1) / total_chunks, f"[Exact Colliding Planets] Scanlines {r_start+1}..{r_end}/{grid.height}")

        model.metrics = QualityMetrics(
            psnr_db=0.0, ssim=0.0, mae_255=0.0, max_err_255=0.0, measured=False,
            total_terms=Z_max + K_max + 20,
            render_time_sec=round(time.perf_counter() - t0, 2)
        )
        return out_img


ExactYeganehCatalog = VHRExactCatalog  # Backward compatibility alias
