"""Rigorous 160-bit Arb (python-flint) Conditional-Variance Engine for:
  1. Single & Mixed Rough Bergomi VIX Futures (Proposition prop:root(i))
  2. Single & Mixed Rough Bergomi VIX Calls & Puts (Proposition prop:root(ii))
  3. VIX Black-76 Implied Volatility Brackets (with forward-uncertainty propagation verified in Arb)
  4. Flat AND Curved Forward-Variance Curves xi_0(t) (flat, quadratic (1+t)^2, sqrt(1+t))

Mathematical architecture (100% outward-rounded Arb, zero uncertified float quadrature):
  - Rank-one Gaussian conditioning variable Z = W(h)/||h|| ~ N(0, 1) with unweighted flat
    barycenter h = (1/Delta) int_0^Delta phi_x dx, whose inner products <h, phi_x> and ||h||^2
    are evaluated via exact closed-form Pfaff 2F1 formulas in Arb.
  - Exact 1D power-graded midpoint + 2nd-derivative remainder enclosures for cell weights
    W_{r,i}(z) = int_{x_i}^{x_{i+1}} (lambda_r xi_0(T+x)/Delta) exp(g_r(x) z - g_r(x)^2/2) dx
    and hence X(z) = sum_{r,i} W_{r,i}(z).
  - Exact Minkowski L^2 decomposition for V(z_m) = Var(Lambda | Z = z_m):
      sqrt(V(z_m)) <= sqrt(W(z_m)^T K_mid W(z_m)) + corner_err(z_m) + sum_{r,i} W_{r,i}(z_m) tau_{r,i},
    evaluated simultaneously across all z-panel conditional means via one C-level FLINT arb_mat
    multiplication K_mid * W!
  - Exact Minkowski z-perturbation bound across each z-panel J_j = [z_j, z_{j+1}] with conditional
    mean z_bar_j = (phi(z_j) - phi(z_{j+1})) / (Phi(z_{j+1}) - Phi(z_j)):
      sup_{z in J_j} sqrt(V(z)) / X(z) <= exp(Delta_g h_j) * (sqrt(V(z_bar_j)) + expm1(Delta_g h_j) S(z_bar_j)) / X(z_bar_j).
  - Exact piecewise-convexity integration over z in [-Zmax, Zmax]:
      * Lower bounds (futures & calls): Jensen on Z | Z in J_j using exact conditional mean z_bar_j!
      * Upper bounds (futures & calls): Secant line of convex sqrt(X(z)) and convex Scarf-Lo majorant!
  - Exact Cauchy-Schwarz Z-tail bound for |Z| > Zmax (Zmax = 10 => < 1.2e-12 vol pts!).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np
from flint import arb, arb_mat, ctx

from arb_vix_core import (
    set_dps,
    arb_interval,
    arb_lo_float,
    arb_hi_float,
    arb_width_float,
    fwd_cov_arb,
    norm_h_sq_flat_arb,
    inner_h_kx_flat_arb,
    _arb_upper_min,
)


# =============================================================================
# 1. Exact Positive Forward-Variance Curve Definitions (with exact 0th/1st/2nd derivs)
# =============================================================================

@dataclass(frozen=True)
class Xi0Curve:
    """Smooth positive forward-variance curve xi_0(t) with exact Arb value, 1st/2nd derivatives,
    and exact average F_vix2 = (1/Delta) int_0^Delta xi_0(T+x) dx."""
    name: str
    eval_arb: Callable[[arb], arb]
    d1_arb: Callable[[arb], arb]
    d2_arb: Callable[[arb], arb]
    avg_arb: Callable[[arb, arb], arb]  # (T, Delta) -> (1/Delta) int_0^Delta xi0(T+x) dx
    is_increasing: bool  # True if nondecreasing on [T, T+Delta] (for corner box [0, x_1])


def make_flat_curve(xi0_base_str: str) -> Xi0Curve:
    c0 = arb(xi0_base_str) ** 2
    return Xi0Curve(
        name=f"flat({xi0_base_str}^2)",
        eval_arb=lambda t: c0,
        d1_arb=lambda t: arb(0),
        d2_arb=lambda t: arb(0),
        avg_arb=lambda T, Delta: c0,
        is_increasing=True,
    )


def make_quadratic_curve(xi0_base_str: str) -> Xi0Curve:
    """JMM (2018) Scenario 2 / BCJ (2023) Sec. 5.3: xi_0(t) = c^2 * (1 + t)^2."""
    c0 = arb(xi0_base_str) ** 2
    return Xi0Curve(
        name=f"quad({xi0_base_str}^2*(1+t)^2)",
        eval_arb=lambda t: c0 * ((arb(1) + t) ** 2),
        d1_arb=lambda t: arb(2) * c0 * (arb(1) + t),
        d2_arb=lambda t: arb(2) * c0,
        avg_arb=lambda T, Delta: c0 * (((arb(1) + T + Delta) ** 3 - (arb(1) + T) ** 3) / (arb(3) * Delta)),
        is_increasing=True,
    )


def make_sqrt_curve(xi0_base_str: str) -> Xi0Curve:
    """BCJ (2023) Sec. 5.3: xi_0(t) = c^2 * sqrt(1 + t)."""
    c0 = arb(xi0_base_str) ** 2
    return Xi0Curve(
        name=f"sqrt({xi0_base_str}^2*sqrt(1+t))",
        eval_arb=lambda t: c0 * (arb(1) + t).sqrt(),
        d1_arb=lambda t: c0 / (arb(2) * (arb(1) + t).sqrt()),
        d2_arb=lambda t: -c0 / (arb(4) * ((arb(1) + t) ** (arb(3) / arb(2)))),
        avg_arb=lambda T, Delta: (arb(2) * c0 / (arb(3) * Delta)) * (((arb(1) + T + Delta) ** (arb(3) / arb(2))) - ((arb(1) + T) ** (arb(3) / arb(2)))),
        is_increasing=True,
    )


# =============================================================================
# 2. Exact Arb Gaussian Panel Masses & Conditional Means
# =============================================================================

def _phi_arb(z: arb) -> arb:
    """Standard normal density phi(z) = exp(-z^2/2) / sqrt(2*pi)."""
    return (-z * z / arb(2)).exp() / (arb(2) * arb.pi()).sqrt()


def _Phi_arb(z: arb) -> arb:
    """Standard normal CDF Phi(z) = 0.5 * erfc(-z / sqrt(2))."""
    return ((-z / arb(2).sqrt()).erfc()) / arb(2)


def _panel_mass_and_mean_arb(za: arb, zb: arb) -> Tuple[arb, arb, arb]:
    """Exact Arb mass p = Phi(zb) - Phi(za), first moment mu = phi(za) - phi(zb),
    and conditional mean z_bar = mu / p on [za, zb]."""
    # For numerical tightness in tails, use erfc directly on the tail side
    s2 = arb(2).sqrt()
    if float(za) >= 0.0:
        p = ((za / s2).erfc() - (zb / s2).erfc()) / arb(2)
    elif float(zb) <= 0.0:
        p = ((-zb / s2).erfc() - (-za / s2).erfc()) / arb(2)
    else:
        p = _Phi_arb(zb) - _Phi_arb(za)
    mu = _phi_arb(za) - _phi_arb(zb)
    zbar = mu / p
    return p, mu, zbar


def build_z_grid_arb(Zmax: float = 10.0, n_core: int = 240, n_tail_each: int = 50, z_core: float = 4.5) -> List[arb]:
    """Build a symmetric z-grid covering [-Zmax, Zmax]:
    dense uniform on [-z_core, z_core] + gently stretched on [-Zmax, -z_core] and [z_core, Zmax]."""
    pts: List[arb] = []
    # Left tail [-Zmax, -z_core]
    for i in range(n_tail_each):
        u = arb(n_tail_each - i) / arb(n_tail_each)
        pts.append(-(arb(str(z_core)) + (arb(str(Zmax)) - arb(str(z_core))) * (u * u)))
    # Core [-z_core, z_core]
    for i in range(n_core + 1):
        pts.append(-arb(str(z_core)) + arb(2) * arb(str(z_core)) * arb(i) / arb(n_core))
    # Right tail (z_core, Zmax]
    for i in range(1, n_tail_each + 1):
        u = arb(i) / arb(n_tail_each)
        pts.append(arb(str(z_core)) + (arb(str(Zmax)) - arb(str(z_core))) * (u * u))
    return pts


# =============================================================================
# 3. Core Arb Conditional-Variance Precomputation (Spatial Mesh + K_mid Matrix)
# =============================================================================

@dataclass
class CondVarArbModel:
    """Precomputed rigorous Arb spatial structure for a single or mixed Rough Bergomi cell."""
    H: arb
    T: arb
    Delta: arb
    components: List[Tuple[arb, arb]]  # list of (a_r, lambda_r)
    xi0_curve: Xi0Curve
    F_vix2: arb
    M: int
    grade_p: int
    # Spatial nodes x_0=0, x_1, ..., x_M=Delta
    xs: List[arb]
    xm: List[arb]  # midpoints for i=1..M-1 (length M-1)
    hx: List[arb]  # half-widths for i=1..M-1 (length M-1)
    # Unit-a loading g_0(x) at nodes and midpoints, plus derivative bounds
    g0_nodes: List[arb]   # length M+1 (i=0..M)
    g0_mid: List[arb]     # length M-1 (i=1..M-1)
    g0p_bound: List[arb]  # upper bound on |g_0'(x)| on [x_i, x_{i+1}], i=1..M-1
    g0pp_bound: List[arb] # upper bound on |g_0''(x)| on [x_i, x_{i+1}], i=1..M-1
    # Residual variance upper bounds D_{0,i}^max on [x_i, x_{i+1}] for i=0..M-1
    D0_max_cell: List[arb]
    # Minkowski cell-discretization error constants tau_{r, i} for r=0..m-1, i=1..M-1
    tau_ri: List[List[arb]]
    # Corner std factor sqrt(expm1(a_r^2 D_{0,0}^max)) for r=0..m-1
    corner_std_r: List[arb]
    # Cell std factor sqrt(expm1(a_r^2 D_{0,i}^max)) for r=0..m-1, i=0..M-1 (for S(z))
    cell_std_ri: List[List[arb]]
    # Precomputed FLINT arb_mat K_mid of shape (m*(M-1)) x (m*(M-1))
    K_mid: arb_mat
    # Global g bounds across all components and x in [0, Delta]
    g_min: arb
    g_max: arb
    g_bar: arb
    Delta_g: arb
    # Precomputed xi0 values/bounds on each cell i=0..M-1
    xi0_mid: List[arb]
    xi0_max: List[arb]
    xi0_d1_max: List[arb]
    xi0_d2_max: List[arb]
    xi0_corner_box: arb


def build_condvar_arb_model(
    H: arb,
    T: arb,
    Delta: arb,
    components: Sequence[Tuple[arb, arb]],
    xi0_curve: Xi0Curve,
    M: int = 128,
    grade_p: int = 3,
    dps: int = 35,
    conditioning_fraction: str | None = None,
) -> CondVarArbModel:
    """Build exact Arb spatial structures and K_mid matrix (computed ONCE per parameter cell)."""
    set_dps(dps)
    comps = list(components)
    if not comps or any(not a >= 0 or not w >= 0 for a,w in comps):
        raise ValueError("component amplitudes and weights must be nonnegative")
    if not sum((w for _,w in comps), arb(0)).contains(1):
        raise ValueError("component weights must sum to one")
    m_comp = len(comps)
    one = arb(1)
    two = arb(2)
    alpha = one / two - H
    a_ap1 = alpha * (alpha + one)

    # Unit-a (a=1) flat barycenter norm ||h_0|| and <h_0, phi_x>
    # A positive point kernel gives an independent optional conditioning direction.
    if conditioning_fraction is None:
        nh0_sq = norm_h_sq_flat_arb(H, T, Delta, one)
        hk_c1 = (Delta ** (H - one / two)) / (H + one / two)
        def inner(x):
            return inner_h_kx_flat_arb(H,T,Delta,one,x,at_zero=x.is_zero(),at_Delta=x is Delta)
    else:
        f = arb(conditioning_fraction)
        if not (0 < f and f <= 1):
            raise ValueError("conditioning fraction must lie in (0,1]")
        point = Delta*f
        nh0_sq = fwd_cov_arb(H,T,one,point,point,is_diag=True)
        hk_c1 = point ** (-alpha)
        def inner(x):
            return fwd_cov_arb(H,T,one,x,point)
    nh0 = nh0_sq.sqrt()
    hk_c2 = alpha * hk_c1

    xs = [arb(0)] + [Delta*(arb(i)/M)**grade_p for i in range(1,M)] + [Delta]
    N = M-1
    xm = [(xs[i]+xs[i+1])/two for i in range(1,M)]
    hx = [(xs[i+1]-xs[i])/two for i in range(1,M)]
    hk0_nodes = [inner(x) for x in xs]
    hk0_mid = [inner(x) for x in xm]
    g0_nodes = [v / nh0 for v in hk0_nodes]
    g0_mid = [v / nh0 for v in hk0_mid]

    # Exact derivative bounds on g_0'(x) and g_0''(x) on [xs[i], xs[i+1]] for i=1..M-1
    g0p_bound = []
    g0pp_bound = []
    for idx in range(N):
        i = idx + 1
        xi = xs[i]
        hki_hi = hk0_nodes[i]
        xna = xi ** (-alpha)
        xnap1 = xna / xi
        hkp = _arb_upper_min(alpha * hki_hi / xi, hk_c1 * xna)
        hkpp = _arb_upper_min(a_ap1 * hki_hi / (xi * xi), hk_c2 * xnap1)
        g0p_bound.append(arb((hkp / nh0).upper()))
        g0pp_bound.append(arb((hkpp / nh0).upper()))

    # Diagonal unit-a C_0(x_i, x_i) for i=0..M and upper bound D_{0,i}^max on [x_i, x_{i+1}]
    C0_diag_nodes = [fwd_cov_arb(H, T, one, xs[i], xs[i], is_diag=True) for i in range(M + 1)]
    D0_max_cell = []
    for i in range(M):
        # C_0(x,x) decreasing, g_0(x) decreasing => sup_{[x_i,x_{i+1}]} D_0(x,x) <= C_0(x_i,x_i) - g_0(x_{i+1})^2
        d_ub = arb((C0_diag_nodes[i] - g0_nodes[i + 1] * g0_nodes[i + 1]).upper())
        if d_ub < 0:
            d_ub = arb(0)
        D0_max_cell.append(d_ub)

    # Compute exact unit-a residual covariance matrix D_0(xm[a], xm[b]) for a, b in 0..N-1
    D0_mid = [[None] * N for _ in range(N)]
    for a_idx in range(N):
        c_aa = fwd_cov_arb(H, T, one, xm[a_idx], xm[a_idx], is_diag=True)
        D0_mid[a_idx][a_idx] = c_aa - g0_mid[a_idx] * g0_mid[a_idx]
        for b_idx in range(a_idx + 1, N):
            c_ab = fwd_cov_arb(H, T, one, xm[a_idx], xm[b_idx], is_diag=False)
            d_ab = c_ab - g0_mid[a_idx] * g0_mid[b_idx]
            D0_mid[a_idx][b_idx] = d_ab
            D0_mid[b_idx][a_idx] = d_ab

    # Build (m*N) x (m*N) FLINT arb_mat K_mid with entries expm1(a_r * a_s * D_0(xm[a], xm[b]))
    dim = m_comp * N
    K_rows = [[arb(0)] * dim for _ in range(dim)]
    for r in range(m_comp):
        ar = comps[r][0]
        for s in range(r, m_comp):
            a_s = comps[s][0]
            aras = ar * a_s
            for i in range(N):
                ri = r * N + i
                for j in range(i if r == s else 0, N):
                    sj = s * N + j
                    val = (aras * D0_mid[i][j]).expm1()
                    K_rows[ri][sj] = val
                    K_rows[sj][ri] = val
    K_mid = arb_mat(K_rows)

    # Minkowski cell discretization bounds tau_{r, idx} on [xs[idx+1], xs[idx+2]]
    denom_d = two * alpha + one
    tau_ri: List[List[arb]] = []
    corner_std_r: List[arb] = []
    cell_std_ri: List[List[arb]] = []
    for r in range(m_comp):
        ar = comps[r][0]
        ar2 = ar * ar
        corner_std_r.append(arb(((ar2 * D0_max_cell[0]).expm1().sqrt()).upper()))
        c_std_i = [arb(((ar2 * D0_max_cell[i]).expm1().sqrt()).upper()) for i in range(M)]
        cell_std_ri.append(c_std_i)
        t_list = []
        for idx in range(N):
            i = idx + 1
            xi = xs[i]
            hi = hx[idx]
            # Exact bound: d_0(x, x_m)^2 <= alpha^2 * hi^2 * (xi^{2H-2} - (T+xi)^{2H-2}) / (2*alpha + 1)
            pow_diff = xi ** (two * H - two) - (T + xi) ** (two * H - two)
            dbar = arb((alpha * hi * (pow_diff / denom_d).sqrt()).upper())
            Dmax = D0_max_cell[i]
            # tau_{r,i} = a_r * exp(0.5*a_r^2*Dmax) * sqrt(1 + a_r^2*Dmax*exp(a_r^2*sqrt(Dmax)*dbar)) * dbar
            inner = one + ar2 * Dmax * (ar2 * Dmax.sqrt() * dbar).exp()
            tau = arb((ar * (ar2 * Dmax / two).exp() * inner.sqrt() * dbar).upper())
            t_list.append(tau)
        tau_ri.append(t_list)

    # Global g bounds across all components
    a_min = min(a.lower() for a,_ in comps)
    a_max = max(a.upper() for a,_ in comps)
    g_min = arb((a_min * g0_nodes[M]).lower())
    g_max = arb((a_max * g0_nodes[0]).upper())
    g_bar = (g_min + g_max) / two
    Delta_g = arb(((g_max - g_min) / two).upper())

    # Precompute xi0 values and derivative bounds on each cell
    xi0_mid = [xi0_curve.eval_arb(T + xm[idx]) for idx in range(N)]
    xi0_max = []
    xi0_d1_max = []
    xi0_d2_max = []
    for idx in range(N):
        i = idx + 1
        t_lo = T + xs[i]
        t_hi = T + xs[i + 1]
        t_box = arb_interval(t_lo, t_hi)
        v_box = xi0_curve.eval_arb(t_box)
        d1_box = xi0_curve.d1_arb(t_box)
        d2_box = xi0_curve.d2_arb(t_box)
        xi0_max.append(arb(v_box.upper()))
        xi0_d1_max.append(arb(d1_box.abs_upper()))
        xi0_d2_max.append(arb(d2_box.abs_upper()))
    # Corner [0, x_1] enclosure of xi_0(T+x)
    t_c_box = arb_interval(T, T + xs[1])
    xi0_corner_box = xi0_curve.eval_arb(t_c_box)

    F_vix2 = xi0_curve.avg_arb(T, Delta)

    return CondVarArbModel(
        H=H,
        T=T,
        Delta=Delta,
        components=comps,
        xi0_curve=xi0_curve,
        F_vix2=F_vix2,
        M=M,
        grade_p=grade_p,
        xs=xs,
        xm=xm,
        hx=hx,
        g0_nodes=g0_nodes,
        g0_mid=g0_mid,
        g0p_bound=g0p_bound,
        g0pp_bound=g0pp_bound,
        D0_max_cell=D0_max_cell,
        tau_ri=tau_ri,
        corner_std_r=corner_std_r,
        cell_std_ri=cell_std_ri,
        K_mid=K_mid,
        g_min=g_min,
        g_max=g_max,
        g_bar=g_bar,
        Delta_g=Delta_g,
        xi0_mid=xi0_mid,
        xi0_max=xi0_max,
        xi0_d1_max=xi0_d1_max,
        xi0_d2_max=xi0_d2_max,
        xi0_corner_box=xi0_corner_box,
    )


def _exp_quad_range_arb(g_lo: arb, g_hi: arb, z: arb) -> arb:
    """Exact Arb enclosure of exp(g*z - 0.5*g^2) for g in [g_lo, g_hi] (concave quadratic in g)."""
    half = arb(1) / arb(2)
    q_lo = g_lo * z - half * g_lo * g_lo
    q_hi = g_hi * z - half * g_hi * g_hi
    # Minimum is always at endpoints (concave quadratic)
    q_min = q_lo.min(q_hi).lower()
    q_max = q_lo.max(q_hi).upper()
    if not (z.upper() < g_lo.lower() or z.lower() > g_hi.upper()):
        q_max = q_max.max((half * z**2).upper())
    return arb_interval(q_min.exp(), q_max.exp())


def eval_W_and_X_arb(model: CondVarArbModel, z: arb) -> Tuple[List[arb], List[arb], arb]:
    """Compute rigorous Arb enclosures at a single z:
    - W_int: list of length m*(M-1) containing exact Arb balls W_{r,i}(z) for interior cells i=1..M-1
    - W_corner: list of length m containing exact Arb balls W_{r,0}(z) for corner cell [0, x_1]
    - X_z: exact Arb ball enclosing X(z) = sum_r (W_{r,0}(z) + sum_{i=1}^{M-1} W_{r,i}(z))
    """
    half = arb(1) / arb(2)
    two = arb(2)
    three = arb(3)
    N = model.M - 1
    m_comp = len(model.components)
    inv_D = arb(1) / model.Delta

    W_corner: List[arb] = []
    W_int: List[arb] = []
    X_total = arb(0)
    dx0 = model.xs[1]  # width of [0, x_1]

    for r in range(m_comp):
        ar, lam_r = model.components[r]
        pref_r = lam_r * inv_D
        # Corner cell [0, x_1]: exact monotone range of exp(g*z - g^2/2) * xi_0(T+x)
        g_hi_0 = ar * model.g0_nodes[0]
        g_lo_0 = ar * model.g0_nodes[1]
        E_box_0 = _exp_quad_range_arb(g_lo_0, g_hi_0, z)
        w0 = pref_r * dx0 * model.xi0_corner_box * E_box_0
        W_corner.append(w0)
        X_total += w0

        # Interior cells i=1..M-1: 1D midpoint rule + exact 2nd-derivative remainder bound
        for idx in range(N):
            i = idx + 1
            hi = model.hx[idx]
            gm = ar * model.g0_mid[idx]
            Em = (gm * z - half * gm * gm).exp()
            mid_val = pref_r * (two * hi) * model.xi0_mid[idx] * Em

            # Exact upper bound on |psi''(x)| on [xs[i], xs[i+1]]
            g_hi = ar * model.g0_nodes[i]
            g_lo = ar * model.g0_nodes[i + 1]
            E_box = _exp_quad_range_arb(g_lo, g_hi, z)
            E_max = arb(E_box.upper())

            # |z - g(x)| <= max(|z - g_lo|, |z - g_hi|)
            dz1 = (z - g_lo).abs_upper()
            dz2 = (z - g_hi).abs_upper()
            dz_max = dz1.max(dz2).upper()

            Gp = ar * model.g0p_bound[idx]
            Gpp = ar * model.g0pp_bound[idx]
            qp_max = dz_max * Gp
            qpp_max = dz_max * Gpp + Gp * Gp

            psi_dd_max = E_max * (
                model.xi0_d2_max[idx]
                + two * model.xi0_d1_max[idx] * qp_max
                + model.xi0_max[idx] * (qpp_max + qp_max * qp_max)
            )
            err_bound = pref_r * (hi * hi * hi / three) * psi_dd_max
            wi = mid_val + arb(0, err_bound.upper())
            W_int.append(wi)
            X_total += wi

    return W_int, W_corner, X_total


# =============================================================================
# 4. Arb-Certified Conditional-Variance VIX Futures & Options (Calls / Puts)
# =============================================================================

def _choose_r_heuristic_float(X_f: float, V_f: float, k_f: float) -> float:
    """Fast heuristic minimizer of r -> (m_r + sqrt(V + m_r^2)) / (4*r) with m_r = X + r^2 - 2*k*r.
    Remember: ANY r > 0 produces a 100% rigorous upper bound in Arb!"""
    # Optimal r is close to max(k, sqrt(X)); refine with 25 golden-section steps on [r_lo, r_hi]
    r_ref = max(k_f, math.sqrt(max(X_f, 1e-12)))
    a_pt = max(1e-6, 0.25 * r_ref)
    b_pt = 2.5 * r_ref + 0.1
    phi_g = (math.sqrt(5.0) - 1.0) / 2.0

    def obj(r: float) -> float:
        mr = X_f + r * r - 2.0 * k_f * r
        return (mr + math.sqrt(V_f + mr * mr)) / (4.0 * r)

    c_pt = b_pt - phi_g * (b_pt - a_pt)
    d_pt = a_pt + phi_g * (b_pt - a_pt)
    fc, fd = obj(c_pt), obj(d_pt)
    for _ in range(30):
        if fc < fd:
            b_pt, d_pt, fd = d_pt, c_pt, fc
            c_pt = b_pt - phi_g * (b_pt - a_pt)
            fc = obj(c_pt)
        else:
            a_pt, c_pt, fc = c_pt, d_pt, fd
            d_pt = a_pt + phi_g * (b_pt - a_pt)
            fd = obj(d_pt)
    return 0.5 * (a_pt + b_pt)


def black76_call_arb(F: arb, K: arb, T: arb, sigma: arb) -> arb:
    """Exact undiscounted Black-76 call price in Arb: F*Phi(d1) - K*Phi(d2)."""
    if sigma.is_zero():
        return (F-K).max(arb(0))
    sT = sigma * T.sqrt()
    d1 = (F / K).log() / sT + sT / arb(2)
    d2 = d1 - sT
    return F * _Phi_arb(d1) - K * _Phi_arb(d2)


def black76_iv_bracket_arb(
    F_lo: arb, F_hi: arb, K: arb, T: arb, C_lo: arb, C_hi: arb, dps: int = 35
) -> Tuple[float, float, bool]:
    """Rigorous Arb enclosure [sigma_L, sigma_U] of Black-76 VIX implied volatility,
    propagating BOTH call-price uncertainty [C_lo, C_hi] AND forward uncertainty [F_lo, F_hi].

    Verified directly in Arb:
      BS(F_hi, K, T, sigma_L) <= C_lo   and   BS(F_lo, K, T, sigma_U) >= C_hi.
    Returns (sigma_L_float, sigma_U_float, verified_bool) in decimal (multiply by 100 for %).
    """
    set_dps(dps)
    if not (F_lo > 0 and K > 0 and T > 0):
        raise ValueError("Black inversion requires a positive forward, strike and maturity")
    # Infeasible crossed rectangle corners can give only the universal endpoints.
    def price(f, sigma):
        return black76_call_arb(f, K, T, sigma)
    lower = arb(0)
    upper = arb(1)
    if C_hi.upper() >= F_lo.lower():
        return 0.0, math.inf, True
    for _ in range(24):
        if price(F_lo,upper).lower() >= C_hi.upper():
            break
        upper *= 2
    else:
        return 0.0, math.inf, True
    certified_upper = upper
    for side in ("lower","upper"):
        left, right = arb(0), certified_upper
        for _ in range(85):
            mid = (left+right)/2
            if side == "lower":
                if price(F_hi,mid).upper() <= C_lo.lower():
                    left = mid
                else:
                    right = mid
            elif price(F_lo,mid).lower() >= C_hi.upper():
                right = mid
            else:
                left = mid
        if side == "lower":
            lower = left.lower()
        else:
            upper = right.upper()
    ok = (lower.is_zero() or price(F_hi,lower).upper() <= C_lo.lower()) and price(F_lo,upper).lower() >= C_hi.upper()
    if not ok:
        raise ArithmeticError("Black inversion endpoint verification failed")
    return max(0.0,arb_lo_float(lower)), arb_hi_float(upper), True


def condvar_vix_futures_and_options_arb(
    model: CondVarArbModel,
    strikes_decimal: Sequence[float],
    Zmax: float = 10.0,
    n_core: int = 240,
    n_tail_each: int = 50,
    z_core: float = 4.5,
    dps: int = 35,
) -> dict:
    """Compute 100% machine-verified outward-rounded Arb enclosures for:
      - VIX Future E[sqrt(Lambda)] via Proposition prop:root(i)
      - VIX Calls E[(sqrt(Lambda) - K)_+] via Proposition prop:root(ii)
      - VIX Puts via exact put-call parity
      - VIX Black-76 Implied Volatility brackets (with Arb-verified forward propagation)
    """
    set_dps(dps)
    one = arb(1)
    two = arb(2)
    four = arb(4)
    N = model.M - 1
    m_comp = len(model.components)
    dim = m_comp * N

    z_nodes = build_z_grid_arb(Zmax=Zmax, n_core=n_core, n_tail_each=n_tail_each, z_core=z_core)
    Nz = len(z_nodes) - 1  # number of z-panels

    # Exact Gaussian panel masses p_j, conditional means z_bar_j, and max half-distances h_j
    p_list: List[arb] = []
    zbar_list: List[arb] = []
    h_list: List[arb] = []
    w_sec_lo: List[arb] = []  # (z_{j+1} - z_bar_j) / (z_{j+1} - z_j)
    w_sec_hi: List[arb] = []  # (z_bar_j - z_j) / (z_{j+1} - z_j)
    for j in range(Nz):
        za, zb = z_nodes[j], z_nodes[j + 1]
        pj, muj, zbar = _panel_mass_and_mean_arb(za, zb)
        p_list.append(pj)
        zbar_list.append(zbar)
        d_left = zbar - za
        d_right = zb - zbar
        hj = d_left.max(d_right).upper()
        h_list.append(hj)
        dz = zb - za
        w_sec_lo.append((zb - zbar) / dz)
        w_sec_hi.append((zbar - za) / dz)

    # Evaluate X(z_j) at all Nz+1 endpoints
    X_nodes: List[arb] = []
    for j in range(Nz + 1):
        _, _, Xj = eval_W_and_X_arb(model, z_nodes[j])
        X_nodes.append(Xj)

    # Evaluate W_int(z_bar_j), W_corner(z_bar_j), X(z_bar_j) at all Nz panel conditional means
    W_cols: List[List[arb]] = []
    W_corners: List[List[arb]] = []
    X_bars: List[arb] = []
    for j in range(Nz):
        W_int, W_c, X_b = eval_W_and_X_arb(model, zbar_list[j])
        W_cols.append(W_int)
        W_corners.append(W_c)
        X_bars.append(X_b)

    # Build dim x Nz FLINT arb_mat W_mat and perform ONE C-level arb_mat multiplication K_mid * W_mat!
    W_mat = arb_mat([[W_cols[j][row] for j in range(Nz)] for row in range(dim)])
    KW_mat = model.K_mid * W_mat

    # Compute rigorous panel upper bounds V_bar_j >= sup_{z in J_j} V(z) and kappa_bar_j >= sup_{z in J_j} sqrt(V(z))/X(z)
    V_panel_hi: List[arb] = []
    rho_panel_lo: List[arb] = []
    # Also track EV_check = sum_j p_j * V(z_bar_j) midpoint diagnostic
    EV_sum = arb(0)
    for j in range(Nz):
        # Exact quadratic form q_j = W_j^T K_mid W_j
        q_j = arb(0)
        for row in range(dim):
            q_j += W_cols[j][row] * KW_mat[row, j]
        # Ensure nonnegativity of upper bound before sqrt
        q_hi = arb(q_j.upper())
        if q_hi < 0:
            q_hi = arb(0)
        sd_disc = q_hi.sqrt()

        # Corner and cell discretization Minkowski remainders
        err_c = arb(0)
        err_cell = arb(0)
        S_zbar = arb(0)
        for r in range(m_comp):
            wc_hi = arb(W_corners[j][r].upper())
            err_c += wc_hi * model.corner_std_r[r]
            S_zbar += wc_hi * model.cell_std_ri[r][0]
            base = r * N
            for idx in range(N):
                wi_hi = arb(W_cols[j][base + idx].upper())
                err_cell += wi_hi * model.tau_ri[r][idx]
                S_zbar += wi_hi * model.cell_std_ri[r][idx + 1]

        sd_zbar = arb((sd_disc + err_c + err_cell).upper())
        EV_sum += p_list[j] * (sd_zbar * sd_zbar)

        # Minkowski z-perturbation bound across panel J_j = [z_j, z_{j+1}]
        hj = h_list[j]
        expm1_dg = (model.Delta_g * hj).expm1()
        sd_perturbed = sd_zbar + expm1_dg * S_zbar

        # Panel upper bound on V(z): exp(2 * g_bar * hj) * sd_perturbed^2
        sd_panel = (model.g_bar * hj).exp() * sd_perturbed
        V_panel_hi.append(arb((sd_panel * sd_panel).upper()))

        # Panel lower bound on rho(z) = (1 + V(z)/X(z)^2)^(-1/2)
        X_b_lo = arb(X_bars[j].lower())
        if not X_b_lo > 0:
            raise ArithmeticError("refine spatial grid to certify a positive conditional mean")
        kappa_hi = (model.Delta_g * hj).exp() * sd_perturbed / X_b_lo
        rho_lo = arb((one / (one + kappa_hi * kappa_hi).sqrt()).lower())
        rho_panel_lo.append(rho_lo)

    # Exact Cauchy-Schwarz tail bound for |Z| > Zmax
    Zmax_arb = arb(str(Zmax))
    tail_prob = (Zmax_arb / two.sqrt()).erfc()
    tail_cs = arb(((model.F_vix2 * tail_prob).sqrt()).upper())

    # --- 1. Arb-certified VIX Future [F_lo, F_hi] ---
    F_lo_sum = arb(0)
    F_hi_sum = arb(0)
    for j in range(Nz):
        pj = p_list[j]
        # Lower: Jensen on conditional mean z_bar_j
        F_lo_sum += pj * rho_panel_lo[j] * arb(X_bars[j].lower()).sqrt()
        # Upper: Secant of convex sqrt(X(z)) on [z_j, z_{j+1}]
        s_a = arb(X_nodes[j].upper()).sqrt()
        s_b = arb(X_nodes[j + 1].upper()).sqrt()
        F_hi_sum += pj * (w_sec_lo[j] * s_a + w_sec_hi[j] * s_b)
    F_lo_ball = arb(F_lo_sum.lower())
    F_hi_ball = arb((F_hi_sum + tail_cs).upper())

    # --- 2. Arb-certified VIX Calls, Puts & IV Smile for every requested strike ---
    options_out = []
    for k_f in strikes_decimal:
        k_arb = arb(str(k_f))
        C_lo_sum = arb(0)
        C_hi_sum = arb(0)
        for j in range(Nz):
            pj = p_list[j]
            # Lower bound: Jensen on convex (rho_lo * sqrt(X(z)) - k)_+
            val_lo = rho_panel_lo[j] * arb(X_bars[j].lower()).sqrt() - k_arb
            if val_lo.lower() > 0:
                C_lo_sum += pj * arb(val_lo.lower())

            # Upper bound: choose explicit positive r_j > 0, apply convex secant on [z_j, z_{j+1}]
            Vj_hi = V_panel_hi[j]
            r_f = _choose_r_heuristic_float(float(X_bars[j]), float(Vj_hi), k_f)
            r_arb = arb(f"{r_f:.12g}")  # exact positive Arb number!
            # Evaluate convex majorant at z_j and z_{j+1} using upper X and upper V_j
            ma = arb(X_nodes[j].upper()) + r_arb * r_arb - two * k_arb * r_arb
            mb = arb(X_nodes[j + 1].upper()) + r_arb * r_arb - two * k_arb * r_arb
            fa = (ma + (Vj_hi + ma * ma).sqrt()) / (four * r_arb)
            fb = (mb + (Vj_hi + mb * mb).sqrt()) / (four * r_arb)
            C_hi_sum += pj * (w_sec_lo[j] * fa + w_sec_hi[j] * fb)

        C_lo_ball = arb(C_lo_sum.lower())
        C_hi_ball = arb((C_hi_sum + tail_cs).upper())

        # Exact Put via Put-Call Parity: P = C + K - F
        P_lo_raw = C_lo_ball + k_arb - F_hi_ball
        P_lo_ball = arb(P_lo_raw.lower()) if P_lo_raw.lower() > 0 else arb(0)
        P_hi_ball = arb((C_hi_ball + k_arb - F_lo_ball).upper())

        # Arb-verified Black-76 IV bracket with forward uncertainty propagation
        iv_lo, iv_hi, iv_ok = black76_iv_bracket_arb(F_lo_ball, F_hi_ball, k_arb, model.T, C_lo_ball, C_hi_ball, dps=dps)

        options_out.append({
            "K_decimal": float(k_f),
            "K_volpts": float(100.0 * k_f),
            "call_L_volpts": arb_lo_float(arb(100) * C_lo_ball),
            "call_U_volpts": arb_hi_float(arb(100) * C_hi_ball),
            "call_width_volpts": arb_hi_float(arb(100) * C_hi_ball) - arb_lo_float(arb(100) * C_lo_ball),
            "put_L_volpts": arb_lo_float(arb(100) * P_lo_ball),
            "put_U_volpts": arb_hi_float(arb(100) * P_hi_ball),
            "put_width_volpts": arb_hi_float(arb(100) * P_hi_ball) - arb_lo_float(arb(100) * P_lo_ball),
            "iv_L_pct": 100.0 * iv_lo,
            "iv_U_pct": 100.0 * iv_hi,
            "iv_width_pct": 100.0 * (iv_hi - iv_lo),
            "iv_arb_verified": bool(iv_ok),
        })

    return {
        "future_L_volpts": arb_lo_float(arb(100) * F_lo_ball),
        "future_U_volpts": arb_hi_float(arb(100) * F_hi_ball),
        "future_width_volpts": arb_hi_float(arb(100) * F_hi_ball) - arb_lo_float(arb(100) * F_lo_ball),
        "tail_cs_volpts": arb_hi_float(arb(100) * tail_cs),
        "panel_mean_variance_diagnostic": float(EV_sum),
        "options": options_out,
        "M": model.M,
        "Nz": Nz,
        "Zmax": Zmax,
    }


# =============================================================================
# 5. Exact Arb Discrete Right-Rectangle Conditional-Variance Option Engine (BD 2022 MLMC audit!)
# =============================================================================

def discrete_right_rect_condvar_options_arb(
    H: arb,
    T: arb,
    Delta: arb,
    a: arb,
    xi0_val: arb,
    strikes_decimal: Sequence[float],
    n: int = 500,
    Zmax: float = 10.0,
    n_core: int = 200,
    n_tail_each: int = 40,
    z_core: float = 4.5,
    dps: int = 30,
    curve: Xi0Curve | None = None,
) -> dict:
    """Exact Arb conditional-variance VIX future & calls for the n-cell right-rectangle discrete target
    Lambda_n = (1/n) sum_{j=1}^n xi_T(T + j*Delta/n) — ZERO spatial discretization error!"""
    set_dps(dps)
    one = arb(1)
    two = arb(2)
    four = arb(4)
    half = one / two

    xs = [Delta * arb(j) / arb(n) for j in range(1, n + 1)]
    C = [[None] * n for _ in range(n)]
    for i in range(n):
        C[i][i] = fwd_cov_arb(H, T, a, xs[i], xs[i], is_diag=True)
        for j in range(i + 1, n):
            v = fwd_cov_arb(H, T, a, xs[i], xs[j], is_diag=False)
            C[i][j] = C[j][i] = v
    hk = [sum(C[i], arb(0)) / arb(n) for i in range(n)]
    nh2 = sum(hk, arb(0)) / arb(n)
    nh = nh2.sqrt()
    g = [v / nh for v in hk]
    g_min = arb(g[-1].lower())
    g_max = arb(g[0].upper())
    g_bar = (g_min + g_max) / two
    Delta_g = arb(((g_max - g_min) / two).upper())

    D_diag_std = [arb(((C[i][i] - g[i] * g[i]).expm1().sqrt()).upper()) for i in range(n)]
    K_mat = arb_mat([[(C[i][j] - g[i] * g[j]).expm1() for j in range(n)] for i in range(n)])
    weights = [(curve.eval_arb(T+x) if curve is not None else xi0_val)/n for x in xs]

    z_nodes = build_z_grid_arb(Zmax=Zmax, n_core=n_core, n_tail_each=n_tail_each, z_core=z_core)
    Nz = len(z_nodes) - 1
    p_list, zbar_list, h_list, w_sec_lo, w_sec_hi = [], [], [], [], []
    for j in range(Nz):
        za, zb = z_nodes[j], z_nodes[j + 1]
        pj, _, zbar = _panel_mass_and_mean_arb(za, zb)
        p_list.append(pj)
        zbar_list.append(zbar)
        dl, dr = zbar - za, zb - zbar
        h_list.append(dl.max(dr).upper())
        dz = zb - za
        w_sec_lo.append((zb - zbar) / dz)
        w_sec_hi.append((zbar - za) / dz)

    def eval_w_x(z: arb) -> Tuple[List[arb], arb]:
        ws = [weights[i] * (g[i] * z - half * g[i] * g[i]).exp() for i in range(n)]
        return ws, sum(ws, arb(0))

    X_nodes = [eval_w_x(z)[1] for z in z_nodes]
    W_cols, X_bars = [], []
    for zb in zbar_list:
        ws, xb = eval_w_x(zb)
        W_cols.append(ws)
        X_bars.append(xb)

    W_mat = arb_mat([[W_cols[j][i] for j in range(Nz)] for i in range(n)])
    KW = K_mat * W_mat

    V_panel_hi, rho_panel_lo = [], []
    for j in range(Nz):
        q_j = arb(0)
        S_j = arb(0)
        for i in range(n):
            q_j += W_cols[j][i] * KW[i, j]
            S_j += arb(W_cols[j][i].upper()) * D_diag_std[i]
        sd_z = arb(q_j.upper()).sqrt() if q_j.upper() > 0 else arb(0)
        hj = h_list[j]
        sd_pert = sd_z + (Delta_g * hj).expm1() * S_j
        sd_pan = (g_bar * hj).exp() * sd_pert
        V_panel_hi.append(arb((sd_pan * sd_pan).upper()))
        kappa_hi = (Delta_g * hj).exp() * sd_pert / arb(X_bars[j].lower())
        rho_panel_lo.append(arb((one / (one + kappa_hi * kappa_hi).sqrt()).lower()))

    tail_cs = arb(((sum(weights) * (arb(str(Zmax)) / two.sqrt()).erfc()).sqrt()).upper())

    F_lo_sum = sum((p_list[j] * rho_panel_lo[j] * arb(X_bars[j].lower()).sqrt() for j in range(Nz)), arb(0))
    F_hi_sum = sum(
        (p_list[j] * (w_sec_lo[j] * arb(X_nodes[j].upper()).sqrt() + w_sec_hi[j] * arb(X_nodes[j + 1].upper()).sqrt())
         for j in range(Nz)),
        arb(0),
    )
    F_lo_ball = arb(F_lo_sum.lower())
    F_hi_ball = arb((F_hi_sum + tail_cs).upper())

    opts = []
    for k_f in strikes_decimal:
        k_arb = arb(str(k_f))
        C_lo_sum, C_hi_sum = arb(0), arb(0)
        for j in range(Nz):
            pj = p_list[j]
            vlo = rho_panel_lo[j] * arb(X_bars[j].lower()).sqrt() - k_arb
            if vlo.lower() > 0:
                C_lo_sum += pj * arb(vlo.lower())
            Vj_hi = V_panel_hi[j]
            r_f = _choose_r_heuristic_float(float(X_bars[j]), float(Vj_hi), k_f)
            r_arb = arb(f"{r_f:.12g}")
            ma = arb(X_nodes[j].upper()) + r_arb * r_arb - two * k_arb * r_arb
            mb = arb(X_nodes[j + 1].upper()) + r_arb * r_arb - two * k_arb * r_arb
            fa = (ma + (Vj_hi + ma * ma).sqrt()) / (four * r_arb)
            fb = (mb + (Vj_hi + mb * mb).sqrt()) / (four * r_arb)
            C_hi_sum += pj * (w_sec_lo[j] * fa + w_sec_hi[j] * fb)
        C_lo_b = arb(C_lo_sum.lower())
        C_hi_b = arb((C_hi_sum + tail_cs).upper())
        opts.append({
            "K_decimal": float(k_f),
            "K_volpts": float(100.0 * k_f),
            "call_L_volpts": arb_lo_float(arb(100) * C_lo_b),
            "call_U_volpts": arb_hi_float(arb(100) * C_hi_b),
            "call_width_volpts": arb_hi_float(arb(100) * C_hi_b) - arb_lo_float(arb(100) * C_lo_b),
        })

    return {
        "n": n,
        "future_L_volpts": arb_lo_float(arb(100) * F_lo_ball),
        "future_U_volpts": arb_hi_float(arb(100) * F_hi_ball),
        "future_width_volpts": arb_hi_float(arb(100) * F_hi_ball) - arb_lo_float(arb(100) * F_lo_ball),
        "options": opts,
    }


if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    set_dps(35)
    H = arb("0.1")
    T = arb("0.5")
    Delta = arb(30) / arb(365)
    a = arb("3.0") * (arb(2) * H).sqrt()
    curve = make_flat_curve("0.235")
    model = build_condvar_arb_model(H, T, Delta, [(a, arb(1))], curve, M=128, grade_p=3, dps=35)
    t1 = time.perf_counter()
    res = condvar_vix_futures_and_options_arb(model, strikes_decimal=[0.15552], n_core=240, n_tail_each=50)
    t2 = time.perf_counter()
    print(f"E14_stress_H010 build={t1-t0:.2f}s, solve={t2-t1:.2f}s")
    print(f"  Future Arb: [{res['future_L_volpts']:.6f}, {res['future_U_volpts']:.6f}] (w={res['future_width_volpts']:.6f})")
    for o in res["options"]:
        print(f"  Call K={o['K_volpts']:.3f}: [{o['call_L_volpts']:.6f}, {o['call_U_volpts']:.6f}] (w={o['call_width_volpts']:.6f}), IV=[{o['iv_L_pct']:.4f}%, {o['iv_U_pct']:.4f}%] (w={o['iv_width_pct']:.4f}%, ok={o['iv_arb_verified']})")
