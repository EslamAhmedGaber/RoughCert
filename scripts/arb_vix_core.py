"""Rigorous 160-bit Arb (python-flint) engine for Rough Bergomi VIX Futures & Competitor Audits.

Implements:
1. Exact outward-rounded Arb evaluations of:
   - RL covariance R_RL(s, u) and forward-variance covariance C(x, y) = <k_x, k_y>
   - Flat-curve mean direction norm ||h||^2 and inner product <h, k_x> (closed-form 2F1)
   - Curved xi_0(t) monotone-enclosure fallback
2. Rigorous Arb end-to-end enclosure of the Tilted Girsanov-Holder VIX Futures Bracket:
     L = exp(-||h||^2 / 8) * m1^(3/2) / sqrt(m2)  <=  E[VIX_T]  <=  exp(-||h||^2 / 8) * sqrt(m1) = U
   using:
   - Exact 1D Arb integration (monotone corner box [0, x_1] + acb.integral on [x_i, x_{i+1}]) for W_i = int_{x_i}^{x_{i+1}} b(x) dx
   - Exact residual decomposition m2 = m1^2 + iint expm1(<d_x, d_y>) b(x) b(y) dx dy
   - Elementary monotone cell bounds on C(x,y) (decreasing) and -<h,k_x> (increasing) for 0 < H < 1/2
3. Independent Arb & floating-point evaluations of competitors:
   - Bourgey, De Marco, & Gobet (2023, Quantitative Finance, Thm. 2.2) Order-3 VIX futures expansion
   - Jacquier, Martini, & Muguruza (2018, Thm. 3.3) lower/upper bounds & moment-matched lognormal proxy
   - Discrete n-cell right-rectangle target for Bourgey & De Marco (2022, JCF, Sec. 3.3) MLMC audit
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from flint import arb, acb, ctx


def set_dps(dps: int = 50) -> None:
    """Set Arb working decimal precision (50 dps ~ 166 bits)."""
    ctx.dps = max(50, dps)


set_dps()


def arb_interval(lo: arb, hi: arb) -> arb:
    """Return a single Arb ball guaranteed to contain [lo.lower(), hi.upper()]."""
    return arb(lo.lower()).union(arb(hi.upper()))


def arb_lo_float(x: arb) -> float:
    """Strict outward lower float bound (rounded downward)."""
    # Convert Arb lower endpoint conservatively using mid - rad
    lo_arb = arb(x.lower())
    val = float(lo_arb)
    # Step down by 2 ULPs to guard float64 conversion rounding
    candidate = math.nextafter(math.nextafter(val, -math.inf), -math.inf)
    if math.isnan(candidate) or not arb(candidate) <= lo_arb:
        raise ArithmeticError('Unverified lower binary64 endpoint')
    return candidate


def arb_hi_float(x: arb) -> float:
    """Strict outward upper float bound (rounded upward)."""
    hi_arb = arb(x.upper())
    val = float(hi_arb)
    candidate = math.nextafter(math.nextafter(val, math.inf), math.inf)
    if math.isnan(candidate) or not arb(candidate) >= hi_arb:
        raise ArithmeticError('Unverified upper binary64 endpoint')
    return candidate


def arb_width_float(x: arb) -> float:
    return arb_hi_float(x) - arb_lo_float(x)


# =============================================================================
# 1. Exact Arb Volterra / RL forward-variance inner products
# =============================================================================

def rl_cov_arb(H: arb, s: arb, u: arb) -> arb:
    """Exact Riemann-Liouville covariance R_RL(s, u) = 2H int_0^{min(s,u)} (s-t)^{H-1/2}(u-t)^{H-1/2} dt.

    Assumes 0 < H < 1/2 and s, u >= 0 are exact/narrow Arb balls with s <= u (swaps automatically).
    """
    if s.is_zero():
        return arb(0)
    if u.is_zero():
        return arb(0)
    if s > u:
        s, u = u, s
    # Exact diagonal
    if s == u or s is u:
        return s ** (2 * H)

    if s.overlaps(u):
        # The RL integral increases in min(s,u) and decreases in |s-u|.
        # Cauchy--Schwarz alone suffices on an unordered input box.
        return arb_interval(arb(0), (s.upper()*u.upper())**H)
    z = s / u
    half = arb(1) / arb(2)
    # Use direct 2F1 for z <= 0.5, and exact connection formula for z > 0.5 (valid since 2H not integer for 0<H<1/2)
    if z <= arb(1)/2:
        hyp = z.hypgeom_2f1(arb(1), half - H, H + arb(3) / arb(2))
    else:
        omz = arb(1) - z
        term1 = ((H + half) / (2 * H)) * omz.hypgeom_2f1(arb(1), half - H, arb(1) - 2 * H)
        B = (H + arb(3) / arb(2)).gamma() * (-2 * H).gamma() / (half - H).gamma()
        term2 = B * (omz ** (2 * H)) * (z ** (-(H + half)))
        hyp = term1 + term2

    return (2 * H / (H + half)) * (s ** (H + half)) * (u ** (H - half)) * hyp


def fwd_cov_arb(H: arb, T: arb, a: arb, x: arb, y: arb, is_diag: bool = False) -> arb:
    """Exact C(x, y) = <k_x, k_y> = a^2 int_0^T (x+r)^{H-1/2} (y+r)^{H-1/2} dr
    = (a^2 / (2H)) [R_RL(T+x, T+y) - R_RL(x, y)].
    """
    eta_sq = (a * a) / (2 * H)
    if is_diag or (x is y):
        if x.is_zero():
            return eta_sq * (T ** (2 * H))
        return eta_sq * ((T + x) ** (2 * H) - x ** (2 * H))
    return eta_sq * (rl_cov_arb(H, T + x, T + y) - rl_cov_arb(H, x, y))


def norm_h_sq_flat_arb(H: arb, T: arb, Delta: arb, a: arb) -> arb:
    """Exact closed-form ||h||^2 for flat xi0 on [T, T+Delta].

    Uses Pfaff transformation on 2F1 so argument z = T/(T+Delta) lies strictly in (0, 1).
    """
    half = arb(1) / arb(2)
    p1 = 2 * H + 2
    term1 = ((T + Delta) ** p1 + T ** p1 - Delta ** p1) / p1
    z = T / (T + Delta)
    hyp = z.hypgeom_2f1(-(H + half), arb(1), H + arb(5) / arb(2))
    term2 = (2 * ((T + Delta) ** (H + half)) * (T ** (H + arb(3) / arb(2))) / (H + arb(3) / arb(2))) * hyp
    pref = (a / (Delta * (H + half))) ** 2
    return pref * (term1 - term2)


def _I_beta_sum_arb(H: arb, b: arb, sum_db: arb) -> arb:
    """Exact int_0^b (d + u)^{H+1/2} u^{H-1/2} du where sum_db = d + b > b >= 0."""
    if b.is_zero():
        return arb(0)
    half = arb(1) / arb(2)
    z = b / sum_db
    hyp = z.hypgeom_2f1(-(H + half), arb(1), H + arb(3) / arb(2))
    return (b ** (H + half)) * (sum_db ** (H + half)) * hyp / (H + half)


def inner_h_kx_flat_arb(
    H: arb, T: arb, Delta: arb, a: arb, x: arb, at_zero: bool = False, at_Delta: bool = False
) -> arb:
    """Exact closed-form <h, k_x> = (1/Delta) int_0^Delta C(x, y) dy for flat xi0, x in [0, Delta]."""
    half = arb(1) / arb(2)
    is_zero = at_zero or (x.is_zero())
    is_Delta = at_Delta or (x is Delta)

    # J2(x) = int_0^T r^{H+1/2} (x + r)^{H-1/2} dr
    if is_zero:
        J2 = (T ** (2 * H + 1)) / (2 * H + 1)
    else:
        z2 = T / (T + x)
        hyp2 = z2.hypgeom_2f1(half - H, arb(1), H + arb(5) / arb(2))
        J2 = (T ** (H + arb(3) / arb(2))) * ((T + x) ** (H - half)) * hyp2 / (H + arb(3) / arb(2))

    # J1(x) = int_0^T (Delta + r)^{H+1/2} (x + r)^{H-1/2} dr
    if is_Delta:
        J1 = ((T + Delta) ** (2 * H + 1) - Delta ** (2 * H + 1)) / (2 * H + 1)
    else:
        J1 = _I_beta_sum_arb(H, T + x, T + Delta) - (arb(0) if is_zero else _I_beta_sum_arb(H, x, Delta))

    return (a * a / (Delta * (H + half))) * (J1 - J2)


def _inner_h_kx_flat_acb(H_c: acb, T_c: acb, Delta_c: acb, a_c: acb, x: acb, analytic: bool) -> acb:
    """Analytic continuation of <h, k_x> for acb.integral on strictly interior intervals [x_lo, x_hi] subset (0, Delta)."""
    if analytic:
        if not (x.real > arb(0) and (Delta_c.real - x.real) > arb(0)):
            return acb(arb("nan"))
    half = acb(1) / acb(2)
    z2 = T_c / (T_c + x)
    z1a = (T_c + x) / (T_c + Delta_c)
    z1b = x / Delta_c
    if analytic:
        if not (z2.real < arb(1) and z1a.real < arb(1) and z1b.real < arb(1)):
            return acb(arb("nan"))
    hyp2 = z2.hypgeom_2f1(half - H_c, acb(1), H_c + acb(5) / acb(2))
    J2 = (T_c ** (H_c + acb(3) / acb(2))) * ((T_c + x) ** (H_c - half)) * hyp2 / (H_c + acb(3) / acb(2))

    I1 = ((T_c + x) ** (H_c + half)) * ((T_c + Delta_c) ** (H_c + half)) * z1a.hypgeom_2f1(-(H_c + half), acb(1), H_c + acb(3) / acb(2)) / (H_c + half)
    I2 = (x ** (H_c + half)) * (Delta_c ** (H_c + half)) * z1b.hypgeom_2f1(-(H_c + half), acb(1), H_c + acb(3) / acb(2)) / (H_c + half)
    J1 = I1 - I2
    return (a_c * a_c / (Delta_c * (H_c + half))) * (J1 - J2)


# =============================================================================
# 2. Rigorous Arb Tilted Girsanov-Holder VIX Futures Bracket (flat xi0)
# =============================================================================

def _arb_upper_min(*vals: arb) -> arb:
    """Given rigorous Arb upper bounds v_1, v_2, ... of the same nonnegative quantity,
    return a rigorous Arb upper bound <= min_k v_k.upper() (pure Arb, no float rounding loss)."""
    best = arb(vals[0].upper())
    for v in vals[1:]:
        u = arb(v.upper())
        if u < best:
            best = u
    return best


def _arb_upper_max(*vals: arb) -> arb:
    """Given rigorous Arb upper bounds v_1, v_2, ..., return a rigorous Arb upper bound >= max_k v_k.upper()."""
    best = arb(vals[0].upper())
    for v in vals[1:]:
        u = arb(v.upper())
        if u > best:
            best = u
    return best


def tilted_and_lognormal_flat_arb(
    H: arb,
    T: arb,
    Delta: arb,
    a: arb,
    xi0_val: arb,
    M: int = 240,
    grade_p: int = 3,
    dps: int = 35,
    tilt_scale: str = "1",
) -> dict:
    """Compute rigorous outward-rounded Arb enclosures for flat xi0 on [T, T+Delta]:
    1. Tilted Girsanov-Holder VIX Futures Bracket [L, U]:
         L = exp(-||h||^2 / 8) * m1^(3/2) / sqrt(m2)  <=  E[VIX_T]  <=  exp(-||h||^2 / 8) * sqrt(m1) = U
    2. JMM (2018, Thm. 3.3) lower/upper bounds [jmm_lo, jmm_up]
    3. Exact moment-matched lognormal proxy F_LN = M1^(3/4) * M2^(-1/8)

    Rigorous 2D integration method (zero wrapping effect, zero diagonal singularity):
    - On [0, Delta]: exact 1D Arb acb.integral for m1 and jmm_lo (with tiny monotone boxes at 0 and Delta).
    - On [d0, Delta]^2 with power-graded mesh x_i = Delta*(i/M)^p (i=1..M, d0 = Delta/M^p):
      2D composite midpoint rule with rigorous 2nd-derivative remainder bounds derived from
      the exact positive integral representations of -d_x C, d_xx C, -d_x <h,k_x>, d_xx <h,k_x>.
    - On the tiny L-shaped corner/edge strip ([0, d0] x [0, Delta] U [d0, Delta] x [0, d0]):
      exact elementary monotone box [0, strip_area * max_integrand].
    """
    set_dps(dps)
    eta_sq = (a * a) / (2 * H)
    a2 = a * a
    scale = arb(tilt_scale)
    if not scale > 0:
        raise ValueError("the monotone enclosure requires positive scalar tilt")
    nh2 = scale**2 * norm_h_sq_flat_arb(H, T, Delta, a)
    pref = (-nh2 / arb(8)).exp()

    # --- 1. Exact 1D Arb enclosure of m1 = (xi0/Delta) int_0^Delta exp(-0.5*(<h,k_x> - ||h||^2)) dx ---
    H_c, T_c, Delta_c, a_c, nh2_c = map(acb, (H, T, Delta, a, nh2))
    eta_sq_c = acb(eta_sq)

    def b_norm_acb(x: acb, analytic: bool) -> acb:
        val_hk = acb(scale) * _inner_h_kx_flat_acb(H_c, T_c, Delta_c, a_c, x, analytic)
        return (-(val_hk - nh2_c) / acb(2)).exp()

    xt_1d = Delta / arb(10_000_000)
    xd_1d = Delta - xt_1d
    hk_0 = scale * inner_h_kx_flat_arb(H, T, Delta, a, arb(0), at_zero=True)
    hk_xt = scale * inner_h_kx_flat_arb(H, T, Delta, a, xt_1d)
    hk_xd = scale * inner_h_kx_flat_arb(H, T, Delta, a, xd_1d)
    hk_D = scale * inner_h_kx_flat_arb(H, T, Delta, a, Delta, at_Delta=True)
    b0 = (-(hk_0 - nh2) / arb(2)).exp()
    b_xt = (-(hk_xt - nh2) / arb(2)).exp()
    b_xd = (-(hk_xd - nh2) / arb(2)).exp()
    b_D = (-(hk_D - nh2) / arb(2)).exp()

    m1_norm = (
        arb_interval(b0 * xt_1d, b_xt * xt_1d)
        + acb.integral(b_norm_acb, acb(xt_1d), acb(xd_1d), rel_tol=2 ** (-55), abs_tol=2 ** (-65)).real
        + arb_interval(b_xd * xt_1d, b_D * xt_1d)
    ) / Delta
    m1 = xi0_val * m1_norm

    # --- 2. Exact 1D Arb enclosure of JMM lower & upper bounds ---
    jmm_up = xi0_val.sqrt()

    def jmm_acb(x: acb, analytic: bool) -> acb:
        if analytic and not (x.real > arb(0)):
            return acb(arb("nan"))
        return (-eta_sq_c * ((T_c + x) ** (2 * H_c) - x ** (2 * H_c)) / acb(8)).exp()

    jf0 = (-eta_sq * (T ** (2 * H)) / arb(8)).exp()
    jf_xt = (-eta_sq * ((T + xt_1d) ** (2 * H) - xt_1d ** (2 * H)) / arb(8)).exp()
    jmm_lo = (xi0_val.sqrt() / Delta) * (
        arb_interval(jf0 * xt_1d, jf_xt * xt_1d)
        + acb.integral(jmm_acb, acb(xt_1d), acb(Delta), rel_tol=2 ** (-55), abs_tol=2 ** (-65)).real
    )

    # --- 3. Exact 2D power-graded midpoint + derivative-bound Arb enclosure of m2 and M2 ---
    alpha = arb(1) / arb(2) - H
    a_ap1 = alpha * (alpha + arb(1))
    B1_gen = (arb(1) - alpha).gamma() * (arb(2) * alpha).gamma() / (arb(1) + alpha).gamma()
    B2_gen = (arb(1) - alpha).gamma() * (arb(2) * alpha + arb(1)).gamma() / (arb(2) + alpha).gamma()
    c1_diag = a2 / arb(2)
    c2_diag = a2 * a_ap1 / (arb(2) * alpha + arb(1))
    c1_gen = a2 * alpha * B1_gen
    c2_gen = a2 * a_ap1 * B2_gen
    hk_c1 = scale * a2 * (Delta ** (H - arb(1) / arb(2))) / (H + arb(1) / arb(2))
    hk_c2 = scale * a2 * alpha * (Delta ** (H - arb(1) / arb(2))) / (H + arb(1) / arb(2))

    xs = [Delta * (arb(i) / arb(M)) ** grade_p for i in range(1, M)] + [Delta]
    N = len(xs) - 1
    xm = [(xs[i] + xs[i + 1]) / arb(2) for i in range(N)]
    hx = [(xs[i + 1] - xs[i]) / arb(2) for i in range(N)]

    hk_e = [scale * inner_h_kx_flat_arb(H, T, Delta, a, xs[i], at_Delta=(i == N)) for i in range(N + 1)]
    hk_m = [scale * inner_h_kx_flat_arb(H, T, Delta, a, xm[i]) for i in range(N)]
    C_00 = fwd_cov_arb(H, T, a, arb(0), arb(0), is_diag=True)

    C_e = [[None] * (N + 1) for _ in range(N + 1)]
    for i in range(N + 1):
        C_e[i][i] = fwd_cov_arb(H, T, a, xs[i], xs[i], is_diag=True)
        for j in range(i + 1, N + 1):
            v = fwd_cov_arb(H, T, a, xs[i], xs[j], is_diag=False)
            C_e[i][j] = C_e[j][i] = v

    C_m = [[None] * N for _ in range(N)]
    for i in range(N):
        C_m[i][i] = fwd_cov_arb(H, T, a, xm[i], xm[i], is_diag=True)
        for j in range(i + 1, N):
            v = fwd_cov_arb(H, T, a, xm[i], xm[j], is_diag=False)
            C_m[i][j] = C_m[j][i] = v

    # Precompute 1D derivative bounds and powers of xs[i]
    hkp_arr = []
    hkpp_arr = []
    x_neg_a = []
    x_neg_ap1 = []
    x_2Hm1 = []
    x_2Hm2 = []
    for i in range(N):
        xi = xs[i]
        hki_hi = hk_e[i]
        xna = xi ** (-alpha)
        xnap1 = xna / xi
        x_neg_a.append(xna)
        x_neg_ap1.append(xnap1)
        x_2Hm1.append(xi ** (arb(2) * H - arb(1)))
        x_2Hm2.append(xi ** (arb(2) * H - arb(2)))
        hkp_arr.append(_arb_upper_min(alpha * hki_hi / xi, hk_c1 * xna))
        hkpp_arr.append(_arb_upper_min(a_ap1 * hki_hi / (xi * xi), hk_c2 * xnap1))

    three_half = arb(3) / arb(2)
    two = arb(2)
    six = arb(6)
    four = arb(4)

    int_m2_mid = arb(0)
    int_m2_err = arb(0)
    int_M2_mid = arb(0)
    int_M2_err = arb(0)

    for i in range(N):
        xi = xs[i]
        hxi = hx[i]
        hxi2 = hxi * hxi
        hki_lo = hk_e[i + 1]
        hkp_i_15 = three_half * hkp_arr[i]
        hkpp_i_15 = three_half * hkpp_arr[i]
        xna_i = x_neg_a[i]
        xnap1_i = x_neg_ap1[i]
        x2hm1_i = x_2Hm1[i]
        x2hm2_i = x_2Hm2[i]

        for j in range(i, N):
            sym = arb(1) if i == j else two
            xj = xs[j]
            hxj = hx[j]
            hxj2 = hxj * hxj
            hkj_lo = hk_e[j + 1]
            area = four * hxi * hxj

            Em = C_m[i][j] - three_half * hk_m[i] - three_half * hk_m[j] + two * nh2
            int_m2_mid += sym * area * Em.exp()
            int_M2_mid += sym * area * C_m[i][j].exp()

            Cmax = C_e[i][j]
            Gmax = (Cmax - three_half * hki_lo - three_half * hkj_lo + two * nh2).exp()
            Mmax = Cmax.exp()

            xna_j = x_neg_a[j]
            xnap1_j = x_neg_ap1[j]

            # Valid for j >= i (so y >= x_j >= x_i on cell B_{ij})
            Cx_b = _arb_upper_min(alpha * Cmax / xi, c1_diag * x2hm1_i, a2 * xna_j * xna_i)
            Cxx_b = _arb_upper_min(a_ap1 * Cmax / (xi * xi), c2_diag * x2hm2_i, a2 * alpha * xna_j * xnap1_i)
            Cy_b = _arb_upper_min(alpha * Cmax / xj, c1_gen * x_2Hm1[j], a2 * xna_i * xna_j)
            Cyy_b = _arb_upper_min(a_ap1 * Cmax / (xj * xj), c2_gen * x_2Hm2[j], a2 * alpha * xna_i * xnap1_j)

            Ex_b = _arb_upper_max(Cx_b, hkp_i_15)
            Exx_b = _arb_upper_max(Cxx_b, hkpp_i_15)
            Ey_b = _arb_upper_max(Cy_b, three_half * hkp_arr[j])
            Eyy_b = _arb_upper_max(Cyy_b, three_half * hkpp_arr[j])

            int_m2_err += sym * (area / six) * Gmax * (hxi2 * (Exx_b + Ex_b * Ex_b) + hxj2 * (Eyy_b + Ey_b * Ey_b))
            int_M2_err += sym * (area / six) * Mmax * (hxi2 * (Cxx_b + Cx_b * Cx_b) + hxj2 * (Cyy_b + Cy_b * Cy_b))

    # Tiny L-shaped boundary strip ([0, d0] x [0, Delta] U [d0, Delta] x [0, d0])
    d0 = xs[0]
    strip_area = two * d0 * Delta - d0 * d0
    E_strip_max = C_00 - three_half * hk_D - three_half * hk_D + two * nh2
    strip_m2 = arb_interval(arb(0), strip_area * E_strip_max.exp())
    strip_M2 = arb_interval(arb(0), strip_area * C_00.exp())

    m2_norm = (int_m2_mid + arb(0, int_m2_err.upper()) + strip_m2) / (Delta * Delta)
    M2_norm = (int_M2_mid + arb(0, int_M2_err.upper()) + strip_M2) / (Delta * Delta)

    m2 = (xi0_val * xi0_val) * m2_norm
    M2 = (xi0_val * xi0_val) * M2_norm

    U_ball = pref * m1.sqrt()
    L_ball = pref * (m1 * m1.sqrt()) / m2.sqrt()
    ln_fut = (xi0_val ** (arb(3) / arb(4))) * (M2 ** (arb(-1) / arb(8)))

    return {
        "norm_h_sq": nh2,
        "m1": m1,
        "m2": m2,
        "M2": M2,
        "L_ball": L_ball,
        "U_ball": U_ball,
        "jmm_lo_ball": jmm_lo,
        "jmm_up_ball": jmm_up,
        "ln_fut_ball": ln_fut,
        "L_outer_volpts": arb_lo_float(arb(100) * L_ball),
        "L_inner_volpts": arb_hi_float(arb(100) * L_ball),
        "U_inner_volpts": arb_lo_float(arb(100) * U_ball),
        "U_outer_volpts": arb_hi_float(arb(100) * U_ball),
        "L_enclosure_width_volpts": arb_width_float(arb(100) * L_ball),
        "U_enclosure_width_volpts": arb_width_float(arb(100) * U_ball),
        "total_outer_width_volpts": arb_hi_float(arb(100) * U_ball) - arb_lo_float(arb(100) * L_ball),
        "jmm_lo_volpts": [arb_lo_float(arb(100) * jmm_lo), arb_hi_float(arb(100) * jmm_lo)],
        "jmm_up_volpts": [arb_lo_float(arb(100) * jmm_up), arb_hi_float(arb(100) * jmm_up)],
        "ln_fut_volpts": [arb_lo_float(arb(100) * ln_fut), arb_hi_float(arb(100) * ln_fut)],
        "ln_enclosure_width_volpts": arb_width_float(arb(100) * ln_fut),
        "M": M,
        "grade_p": grade_p,
    }


# =============================================================================
# 3. Rigorous Arb Enclosures of Competitors (BDG 2023 Thm. 2.2, JMM 2018, Lognormal Proxy)
# =============================================================================

def jmm_and_lognormal_flat_arb(
    H: arb,
    T: arb,
    Delta: arb,
    a: arb,
    xi0_val: arb,
    M: int = 96,
    grade_p: int = 3,
    dps: int = 50,
) -> dict:
    """Rigorous Arb enclosures for:
    - JMM (2018, Thm. 3.3) lower and upper VIX futures bounds
    - Exact moment-matched lognormal proxy F_LN = M1^(3/4) * M2^(-1/8) * ... wait:
      if Lambda ~ LogNormal(mu, s2) with E[Lambda]=M1, E[Lambda^2]=M2,
      then s2 = log(M2/M1^2), mu = log(M1) - s2/2, so
      E[sqrt(Lambda)] = exp(mu/2 + s2/8) = sqrt(M1) * (M2 / M1^2)^(-1/8) = M1^(3/4) / M2^(1/8)!
    """
    set_dps(dps)
    eta_sq = (a * a) / (2 * H)

    # JMM upper bound: sqrt(xi0) (exact for flat xi0)
    jmm_up = xi0_val.sqrt()

    # JMM lower bound: (sqrt(xi0)/Delta) int_0^Delta exp(-eta_sq * ((T+x)^{2H} - x^{2H}) / 8) dx
    # Since (T+x)^{2H} - x^{2H} is strictly DECREASING in x for 0 < H < 1/2,
    # the integrand is strictly INCREASING in x!
    H_c, T_c, eta_sq_c = acb(H), acb(T), acb(eta_sq)

    def jmm_integrand(x: acb, analytic: bool) -> acb:
        if analytic and not (x.real > arb(0)):
            return acb(arb("nan"))
        v = (T_c + x) ** (2 * H_c) - x ** (2 * H_c)
        return (-eta_sq_c * v / acb(8)).exp()

    xt = Delta / arb(1000000)
    f0 = (-eta_sq * (T ** (2 * H)) / arb(8)).exp()
    f_xt = (-eta_sq * ((T + xt) ** (2 * H) - xt ** (2 * H)) / arb(8)).exp()
    part0 = arb_interval(f0 * xt, f_xt * xt)
    part1 = acb.integral(jmm_integrand, acb(xt), acb(Delta), rel_tol=2 ** (-80), abs_tol=2 ** (-90)).real
    jmm_lo = (xi0_val.sqrt() / Delta) * (part0 + part1)

    # Exact M2 = E[Lambda^2] = (xi0/Delta)^2 iint exp(C(x, y)) dx dy
    # Using monotone cell bounds on C(x, y) (decreasing in x, y) on graded mesh
    nodes = [arb(0) if i == 0 else (Delta if i == M else Delta * (arb(i) / arb(M)) ** grade_p) for i in range(M + 1)]
    C_mat = [[None] * (M + 1) for _ in range(M + 1)]
    for i in range(M + 1):
        C_mat[i][i] = fwd_cov_arb(H, T, a, nodes[i], nodes[i], is_diag=True)
        for j in range(i + 1, M + 1):
            val = fwd_cov_arb(H, T, a, nodes[i], nodes[j], is_diag=False)
            C_mat[i][j] = val
            C_mat[j][i] = val

    w_dx = [(nodes[i + 1] - nodes[i]) / Delta for i in range(M)]
    m2_lo = arb(0)
    m2_hi = arb(0)
    for i in range(M):
        wi2 = w_dx[i] * w_dx[i]
        m2_lo += wi2 * C_mat[i + 1][i + 1].exp()
        m2_hi += wi2 * C_mat[i][i].exp()
        for j in range(i + 1, M):
            wij = arb(2) * w_dx[i] * w_dx[j]
            m2_lo += wij * C_mat[i + 1][j + 1].exp()
            m2_hi += wij * C_mat[i][j].exp()

    M1 = xi0_val
    M2 = (xi0_val * xi0_val) * arb_interval(m2_lo, m2_hi)
    # E[sqrt(Lambda_LN)] = M1^(3/4) * M2^(-1/8)
    ln_fut = (M1 ** (arb(3) / arb(4))) * (M2 ** (arb(-1) / arb(8)))

    return {
        "jmm_lo_ball": jmm_lo,
        "jmm_up_ball": jmm_up,
        "M2_ball": M2,
        "ln_fut_ball": ln_fut,
        "jmm_lo_volpts": [arb_lo_float(arb(100) * jmm_lo), arb_hi_float(arb(100) * jmm_lo)],
        "jmm_up_volpts": [arb_lo_float(arb(100) * jmm_up), arb_hi_float(arb(100) * jmm_up)],
        "ln_fut_volpts": [arb_lo_float(arb(100) * ln_fut), arb_hi_float(arb(100) * ln_fut)],
    }


def bdg_futures_flat_arb(
    H: arb,
    T: arb,
    Delta: arb,
    a: arb,
    xi0_val: arb,
    dps: int = 50,
) -> dict:
    """Rigorous Arb enclosure of Bourgey, De Marco, & Gobet (2023, QF, Thm. 2.2) VIX futures
    expansion F_BDG^(3) = S * (1 + gamma_1/2 + gamma_2/4 + gamma_3/8) for flat xi0!

    Uses ONLY the exact kernel identities (independent of Bourgey's closed-form gamma_2/gamma_3 code):
      meanp_0 = -(1/2) * (1/Delta) int_0^Delta ||k_x||^2 dx   (exact elementary formula)
      varp    = ||h||^2                                       (exact closed-form 2F1)
      q(x)    = ||k_x||^2 - avg_norm_sq                       (monotonically decreasing in x)
      c(x)    = <h, k_x> - ||h||^2                            (monotonically decreasing in x)
      gamma_1 = (1/8) * (1/Delta) int_0^Delta q(x)^2 dx + (1/2) * (avg_norm_sq - ||h||^2)
      gamma_2 = -(1/2) * (1/Delta) int_0^Delta c(x) * q(x) dx
      gamma_3 = (1/2) * (1/Delta) int_0^Delta c(x)^2 dx
    Every 1D integral is enclosed rigorously in Arb via tiny monotone boxes at 0 and Delta + acb.integral!
    """
    set_dps(dps)
    eta_sq = (a * a) / (2 * H)
    # avg_norm_sq = (1/Delta) int_0^Delta ||k_x||^2 dx
    # ||k_x||^2 = eta_sq * ((T+x)^{2H} - x^{2H})
    p = 2 * H + 1
    avg_norm_sq = (eta_sq / (Delta * p)) * ((T + Delta) ** p - T ** p - Delta ** p)
    nh2 = norm_h_sq_flat_arb(H, T, Delta, a)

    # S = sqrt(xi0) * exp(-avg_norm_sq / 4 + nh2 / 8)
    S = xi0_val.sqrt() * (-avg_norm_sq / arb(4) + nh2 / arb(8)).exp()

    # Exact closed-form (1/Delta) int_0^Delta q(x)^2 dx = (1/Delta) int_0^Delta ||k_x||^4 dx - avg_norm_sq^2
    # Wait: compute ALL THREE 1D integrals (q^2, c*q, c^2) via Arb acb.integral on [xt, Delta-xt] + monotone boxes!
    H_c, T_c, Delta_c, a_c = map(acb, (H, T, Delta, a))
    eta_sq_c, avg_c, nh2_c = map(acb, (eta_sq, avg_norm_sq, nh2))

    def q_real(x: arb) -> arb:
        n2 = eta_sq * (T ** (2 * H)) if (x.is_zero()) else eta_sq * ((T + x) ** (2 * H) - x ** (2 * H))
        return n2 - avg_norm_sq

    def c_real(x: arb, at_zero: bool = False, at_Delta: bool = False) -> arb:
        return inner_h_kx_flat_arb(H, T, Delta, a, x, at_zero=at_zero, at_Delta=at_Delta) - nh2

    def q_acb(x: acb, analytic: bool) -> acb:
        if analytic and not (x.real > arb(0)):
            return acb(arb("nan"))
        return eta_sq_c * ((T_c + x) ** (2 * H_c) - x ** (2 * H_c)) - avg_c

    def c_acb(x: acb, analytic: bool) -> acb:
        return _inner_h_kx_flat_acb(H_c, T_c, Delta_c, a_c, x, analytic) - nh2_c

    # Split [0, Delta] = [0, xt] U [xt, Delta - xt] U [Delta - xt, Delta] with xt = Delta * 10^-7
    xt = Delta / arb(10_000_000)
    xd = Delta - xt

    # Monotone enclosures of q(x) and c(x) on [0, xt] and [xd, Delta]
    q_box_0 = arb_interval(q_real(xt), q_real(arb(0)))
    q_box_1 = arb_interval(q_real(Delta), q_real(xd))
    c_box_0 = arb_interval(c_real(xt), c_real(arb(0), at_zero=True))
    c_box_1 = arb_interval(c_real(Delta, at_Delta=True), c_real(xd))

    rel_tol = 2 ** (-50)
    abs_tol = 2 ** (-60)

    int_q2 = (
        xt * (q_box_0 * q_box_0)
        + acb.integral(lambda x, an: q_acb(x, an) ** 2, acb(xt), acb(xd), rel_tol=rel_tol, abs_tol=abs_tol).real
        + xt * (q_box_1 * q_box_1)
    ) / Delta

    int_cq = (
        xt * (c_box_0 * q_box_0)
        + acb.integral(lambda x, an: c_acb(x, an) * q_acb(x, an), acb(xt), acb(xd), rel_tol=rel_tol, abs_tol=abs_tol).real
        + xt * (c_box_1 * q_box_1)
    ) / Delta

    int_c2 = (
        xt * (c_box_0 * c_box_0)
        + acb.integral(lambda x, an: c_acb(x, an) ** 2, acb(xt), acb(xd), rel_tol=rel_tol, abs_tol=abs_tol).real
        + xt * (c_box_1 * c_box_1)
    ) / Delta

    gamma_1 = int_q2 / arb(8) + (avg_norm_sq - nh2) / arb(2)
    gamma_2 = -int_cq / arb(2)
    gamma_3 = int_c2 / arb(2)

    F0 = S
    F1 = S * (arb(1) + gamma_1 / arb(2))
    F2 = S * (arb(1) + gamma_1 / arb(2) + gamma_2 / arb(4))
    F3 = S * (arb(1) + gamma_1 / arb(2) + gamma_2 / arb(4) + gamma_3 / arb(8))

    return {
        "S_ball": S,
        "gamma_1_ball": gamma_1,
        "gamma_2_ball": gamma_2,
        "gamma_3_ball": gamma_3,
        "F0_ball": F0,
        "F1_ball": F1,
        "F2_ball": F2,
        "F3_ball": F3,
        "F3_volpts": [arb_lo_float(arb(100) * F3), arb_hi_float(arb(100) * F3)],
        "F3_width_volpts": arb_width_float(arb(100) * F3),
    }


def bdg_futures_independent_scipy(H: float, T: float, Delta: float, a: float, xi0_val: float, n_quad: int = 128) -> dict:
    """Completely independent floating-point implementation of BDG (2023) Theorem 2.2
    directly matching Bourgey's rbergomi.py formulas (using scipy.special.hyp2f1 + Gauss-Legendre).
    """
    from scipy import special
    eta_p1 = a / math.sqrt(2.0 * H)
    # Bourgey's mean_proxy_flat
    meanp = -(eta_p1 ** 2) * ((T + Delta) ** (2 * H + 1) - Delta ** (2 * H + 1) - T ** (2 * H + 1)) / (2.0 * Delta * (2 * H + 1))
    # Bourgey's var_proxy_flat
    varp = (
        ((T + Delta) ** (2 * H + 2) + T ** (2 * H + 2) - Delta ** (2 * H + 2)) / (2 * H + 2)
        - 2.0 * (Delta ** (H + 0.5)) * (T ** (H + 1.5))
        * special.hyp2f1(-H - 0.5, H + 1.5, H + 2.5, -T / Delta) / (H + 1.5)
    ) * (a / (Delta * (H + 0.5))) ** 2

    S = math.sqrt(xi0_val) * math.exp(0.5 * meanp + 0.125 * varp)

    # Bourgey's gamma_1_proxy_flat
    tmp0 = ((T + Delta) ** (4 * H + 1) + Delta ** (4 * H + 1) - T ** (4 * H + 1)) / ((4 * H + 1) * Delta)
    tmp1 = -(((T ** (2 * H + 1) + Delta ** (2 * H + 1) - (T + Delta) ** (2 * H + 1)) / (Delta * (2 * H + 1))) ** 2)
    tmp2 = -2.0 * (T ** (2 * H)) * (Delta ** (2 * H)) * special.hyp2f1(-2 * H, 2 * H + 1, 2 * H + 2, -Delta / T) / (2 * H + 1)
    gamma_10 = (eta_p1 ** 4) * (tmp0 + tmp1 + tmp2) / 8.0
    gamma_11 = (eta_p1 ** 2) * ((T + Delta) ** (2 * H + 1) - T ** (2 * H + 1) - Delta ** (2 * H + 1)) / (2.0 * (2 * H + 1) * Delta) - 0.5 * varp
    gamma_1 = gamma_10 + gamma_11

    # Bourgey's gamma_2_proxy_flat (Gauss-Legendre on [0,1]^2)
    gx, gw = np.polynomial.legendre.leggauss(n_quad)
    v = 0.5 * (gx + 1.0)
    w = 0.5 * gw
    # Vectorised g2(t, alpha)
    t_grid = v[:, None]
    al_grid = v[None, :]
    g2 = (
        ((T * t_grid + Delta) ** (H + 0.5) - (T * t_grid) ** (H + 0.5))
        * ((T + al_grid * Delta) ** (2 * H) - (al_grid * Delta) ** (2 * H))
        * ((T * t_grid + al_grid * Delta) ** (H - 0.5))
    )
    int2 = float(w @ g2 @ w)
    gamma_2 = -int2 * T * (eta_p1 ** 4) * H / (Delta * (H + 0.5)) + (
        (eta_p1 ** 2) * ((T + Delta) ** (2 * H + 1) - T ** (2 * H + 1) - Delta ** (2 * H + 1)) * varp
    ) / (2.0 * (2 * H + 1) * Delta)

    # Bourgey's gamma_3_proxy_flat
    d = Delta / T
    # Use v for both t and alpha; compute tmps(alpha)
    ad = v * d
    tmp0_3 = (ad ** (H - 0.5)) * special.hyp2f1(-H + 0.5, H + 1.5, H + 2.5, -1.0 / ad) / (H + 1.5)
    tmp1_3 = (
        ((1.0 + ad) ** (H + 0.5)) * special.hyp2f1(-H - 0.5, H + 0.5, H + 1.5, -(1.0 + ad) / (d - ad))
        - (ad ** (H + 0.5)) * special.hyp2f1(-H - 0.5, H + 0.5, H + 1.5, -ad / (d - ad))
    ) * ((1.0 - v) ** (H + 0.5)) * (d ** (H + 0.5)) / (H + 0.5)
    tmps = tmp1_3 - tmp0_3
    g3 = (
        (T ** (4 * H))
        * ((t_grid + d) ** (H + 0.5) - t_grid ** (H + 0.5))
        * ((t_grid + al_grid * d) ** (H - 0.5))
        * tmps[None, :]
    )
    int3 = float(w @ g3 @ w)
    gamma_3 = 2.0 * (eta_p1 ** 4) * (H ** 2) * (T ** 2) / ((Delta ** 2) * ((H + 0.5) ** 2)) * int3 - 0.5 * (varp ** 2)

    F3 = S * (1.0 + 0.5 * gamma_1 + 0.25 * gamma_2 + 0.125 * gamma_3)
    return {
        "S": S,
        "varp": varp,
        "gamma_1": gamma_1,
        "gamma_2": gamma_2,
        "gamma_3": gamma_3,
        "F3_volpts": 100.0 * F3,
    }


def discrete_right_rect_tilted_arb(
    H: arb,
    T: arb,
    Delta: arb,
    a: arb,
    xi0_val: arb,
    n: int = 500,
    dps: int = 50,
) -> dict:
    """Exact Arb tilted Girsanov-Holder bracket for the n-cell RIGHT-RECTANGLE discrete target:
      Lambda_n = (1/n) sum_{j=1}^n xi_T(T + j*Delta/n)
    (exact discrete target used by Bourgey & De Marco 2022 MLMC reference!).
    ZERO spatial integration error: purely finite matrix algebra in Arb!
    """
    set_dps(dps)
    xs = [Delta * arb(j) / arb(n) for j in range(1, n + 1)]
    # Evaluate exact C_ij in float64? NO — let's do exact Arb at 30 dps (or check timing for n=500 -> 125k calls)
    # Wait: for fast exact verification, n=500 symmetric pairs = 125,250 hyp2f1 calls in Python could take ~10-20s.
    # Let's keep dps=30 (100 bits) for n=500 or vectorise cleanly if needed.
    set_dps(dps)
    w_atom = xi0_val / arb(n)
    C = [[None] * n for _ in range(n)]
    for i in range(n):
        C[i][i] = fwd_cov_arb(H, T, a, xs[i], xs[i], is_diag=True)
        for j in range(i + 1, n):
            val = fwd_cov_arb(H, T, a, xs[i], xs[j], is_diag=False)
            C[i][j] = val
            C[j][i] = val
    # h = (1/n) sum_j k_{x_j} => <h, k_{x_i}> = (1/n) sum_j C[i][j], ||h||^2 = (1/n) sum_i <h, k_{x_i}>
    hk = [sum(C[i], arb(0)) / arb(n) for i in range(n)]
    nh2 = sum(hk, arb(0)) / arb(n)
    b = [w_atom * (-(hk[i] - nh2) / arb(2)).exp() for i in range(n)]
    m1 = sum(b, arb(0))
    m2 = arb(0)
    for i in range(n):
        m2 += b[i] * b[i] * (C[i][i] - arb(2) * hk[i] + nh2).exp()
        for j in range(i + 1, n):
            m2 += arb(2) * b[i] * b[j] * (C[i][j] - hk[i] - hk[j] + nh2).exp()
    pref = (-nh2 / arb(8)).exp()
    U = pref * m1.sqrt()
    L = pref * (m1 * m1.sqrt()) / m2.sqrt()
    return {
        "n": n,
        "L_outer_volpts": arb_lo_float(arb(100) * L),
        "U_outer_volpts": arb_hi_float(arb(100) * U),
        "L_width": arb_width_float(arb(100) * L),
        "U_width": arb_width_float(arb(100) * U),
    }
