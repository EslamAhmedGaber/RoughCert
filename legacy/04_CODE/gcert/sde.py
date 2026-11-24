"""Equation branch: residual-covariance Lawson--Hermite (RCLH) backend (Paper IX v13, Alg. 1).

Model (WIS):  dX = (alpha X + b(X)) dt + beta X <> dB^H,  X(0) = x0,  b(x) = sum_p c_p x^p.
Finite target: the driver on a uniform grid t_1..t_N, split as B_N = G xi + L_R eta (G from ANY
representation satisfying the projection contract), rows interpolated linearly.

Implemented here, following the v13 statements:
  * Lawson state Y = sum_nu y_nu H_nu(xi) on the total-degree set Lambda_P (M variables);
  * closure  g_j = sum_p c_p Y_j^p exp((p-1) q0_j + p(p-1) rho/2),  q0 = a.xi + |a|^2/2  (Cor. 2.5);
  * tensor Gauss--Hermite active rule (q >= P+1 per coordinate, Prop. gram-floor), RK4 in time;
  * coherent outputs (Thm. twotime):  E[X(t)X(s)] = e^{a(t).a(s) + v(t,s)} E[Y_t(xi + a(s)) Y_s(xi + a(t))];
    the same with v = 0 is E[Gamma(Pi)X(t) Gamma(Pi)X(s)] (Prop. D1 of the interface document);
  * reference: the full finite target sampled exactly through the double-Gjessing pull-back,
    X(t)(zeta) = exp(h_t.zeta - |h_t|^2/2) Y_full(t; zeta - h_t),  Y_full solving a pathwise ODE.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from .functionals import normalized_hermite


# ----------------------------------------------------------------------------- driver factor
@dataclass
class DriverFactor:
    grid: np.ndarray          # t_1..t_N (t_0 = 0 implied)
    G: np.ndarray             # N x M retained loading (projection of the grid vector)
    LR: np.ndarray            # N x r residual factor, LR LR^T = R_N - G G^T
    name: str

    @property
    def M(self):
        return self.G.shape[1]

    def _interp(self, A, t):
        t = np.atleast_1d(np.asarray(t, float))
        g = np.concatenate([[0.0], self.grid])
        A0 = np.vstack([np.zeros((1, A.shape[1])), A])
        idx = np.clip(np.searchsorted(g, t, side="right") - 1, 0, g.size - 2)
        th = (t - g[idx]) / (g[idx + 1] - g[idx])
        return (1 - th)[:, None] * A0[idx] + th[:, None] * A0[idx + 1]

    def a(self, t):
        return self._interp(self.G, t)

    def ar(self, t):
        return self._interp(self.LR, t)

    def u_trace(self):
        Rn = self.G @ self.G.T + self.LR @ self.LR.T
        return float(np.trace(self.LR @ self.LR.T) / np.trace(Rn))


def make_factor(field, N, M, kind="endpoint", rep=None):
    """Grid factor from a Paper IX construction or from any projection backend ``rep``."""
    T = field.T
    grid = T * np.arange(1, N + 1) / N
    C = field.R(grid[:, None], grid[None, :])
    if rep is not None:
        G = rep.loadings(grid)[:, :M]
        name = f"{rep.name}[M={M}]"
    elif kind == "spectral":
        lam, V = np.linalg.eigh(C)
        idx = np.argsort(lam)[::-1][:M]
        G = V[:, idx] * np.sqrt(np.maximum(lam[idx], 0))[None, :]
        name = f"spectral[M={M}]"
    elif kind == "endpoint":
        g0 = C[:, -1] / math.sqrt(C[-1, -1])
        Cp = C - np.outer(g0, g0)
        lam, V = np.linalg.eigh(Cp)
        idx = np.argsort(lam)[::-1][:M - 1]
        G = np.column_stack([g0, V[:, idx] * np.sqrt(np.maximum(lam[idx], 0))[None, :]]) if M > 1 else g0[:, None]
        name = f"endpoint[M={M}]"
    else:
        raise ValueError(kind)
    Res = C - G @ G.T
    lam, V = np.linalg.eigh(0.5 * (Res + Res.T))
    if lam.min() < -1e-10 * np.max(np.abs(C)):
        raise ValueError(f"factor {name} violates the projection contract (residual not PSD: {lam.min():.2e})")
    keep = lam > 1e-14 * np.max(np.abs(C))
    LR = V[:, keep] * np.sqrt(lam[keep])[None, :]
    return DriverFactor(grid, G, LR, name)


# ----------------------------------------------------------------------------- Hermite space
def total_degree_set(M, P):
    return [nu for nu in itertools.product(range(P + 1), repeat=M) if sum(nu) <= P]


class HermiteSpace:
    def __init__(self, M, P, q=None):
        self.M, self.P = M, P
        self.idx = total_degree_set(M, P)
        self.q = q or (P + 3)
        x, w = np.polynomial.hermite_e.hermegauss(self.q)
        w = w / math.sqrt(2 * math.pi)
        grids = np.meshgrid(*([x] * M), indexing="ij")
        wg = np.meshgrid(*([w] * M), indexing="ij")
        self.nodes = np.stack([g.ravel() for g in grids], axis=1)            # (Q, M)
        self.weights = np.prod(np.stack([g.ravel() for g in wg], axis=1), axis=1)
        self.Phi = self.basis(self.nodes)                                      # (Q, |Lambda|)
        gram = self.Phi.T @ (self.weights[:, None] * self.Phi)
        if np.max(np.abs(gram - np.eye(len(self.idx)))) > 1e-10:
            raise ValueError("active rule does not resolve the basis (q >= P+1 required)")

    def basis(self, Z):
        Z = np.atleast_2d(Z)
        H = [normalized_hermite(Z[:, k], self.P) for k in range(self.M)]       # each (P+1, Q)
        return np.stack([np.prod([H[k][nu[k]] for k in range(self.M)], axis=0) for nu in self.idx], axis=1)


# ----------------------------------------------------------------------------- RCLH solver
@dataclass
class EquationProblem:
    alpha: float
    beta: float
    poly: dict                # {p: c_p} for b(x) = sum c_p x^p
    x0: float
    name: str = ""


class RCLH:
    def __init__(self, prob, factor, P=3, substeps=4, closure=True, q=None):
        self.prob, self.f, self.P = prob, factor, P
        self.hs = HermiteSpace(factor.M, P, q)
        self.substeps, self.closure = substeps, closure

    def _rhs(self, t, y):
        hs, pb = self.hs, self.prob
        a = self.prob.beta * self.f.a(t)[0]
        rho = (self.prob.beta ** 2) * float(np.sum(self.f.ar(t)[0] ** 2)) if self.closure else 0.0
        Y = hs.Phi @ y
        q0 = hs.nodes @ a + 0.5 * float(a @ a)
        g = np.zeros_like(Y)
        for p, c in pb.poly.items():
            g += c * Y ** p * np.exp((p - 1) * q0 + 0.5 * p * (p - 1) * rho)
        return pb.alpha * y + hs.Phi.T @ (hs.weights * g)

    def solve(self):
        grid = np.concatenate([[0.0], self.f.grid])
        y = np.zeros(len(self.hs.idx))
        y[0] = self.prob.x0
        traj = [y.copy()]
        for k in range(grid.size - 1):
            h = (grid[k + 1] - grid[k]) / self.substeps
            t = grid[k]
            for _ in range(self.substeps):
                k1 = self._rhs(t, y)
                k2 = self._rhs(t + h / 2, y + h / 2 * k1)
                k3 = self._rhs(t + h / 2, y + h / 2 * k2)
                k4 = self._rhs(t + h, y + h * k3)
                y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                t += h
            traj.append(y.copy())
        self.times = grid
        self.Y = np.array(traj)              # (N+1, |Lambda|)
        return self

    # ---- coherent outputs ---------------------------------------------------------------
    def _shifted_eval(self, i, shift):
        return self.hs.basis(self.hs.nodes + shift[None, :]) @ self.Y[i]

    def two_time(self, i, j, residual=True):
        """E[X(t_i) X(t_j)] (coherent) or E[Gamma X(t_i) Gamma X(t_j)] (residual=False)."""
        beta = self.prob.beta
        ai = beta * self.f.a(self.times[i])[0]
        aj = beta * self.f.a(self.times[j])[0]
        Yi = self._shifted_eval(i, aj)
        Yj = self._shifted_eval(j, ai)
        v = beta ** 2 * float(self.f.ar(self.times[i])[0] @ self.f.ar(self.times[j])[0]) if residual else 0.0
        return math.exp(float(ai @ aj) + v) * float(self.hs.weights @ (Yi * Yj))

    def mean(self, i):
        return float(self.Y[i, 0])

    def kernel_matrix(self, ids, residual=True):
        K = np.empty((len(ids), len(ids)))
        for a_, i in enumerate(ids):
            for b_, j in enumerate(ids[a_:], start=a_):
                K[a_, b_] = K[b_, a_] = self.two_time(i, j, residual)
        return K

    def integrated(self, ids, w):
        """Mean, variance of J = sum w_i X(t_i), coherent and active-only, and the exact ledger P."""
        w = np.asarray(w, float)
        m = np.array([self.mean(i) for i in ids])
        Kc = self.kernel_matrix(ids, True)
        Ka = self.kernel_matrix(ids, False)
        mean = float(w @ m)
        var_c = float(w @ Kc @ w) - mean ** 2
        var_a = float(w @ Ka @ w) - mean ** 2
        return {"mean": mean, "var_coherent": var_c, "var_active": var_a,
                "P_coherent_vs_active": float(w @ (Kc - Ka) @ w)}

    def sample_coherent(self, ids, w, K, rng):
        """Samples of J_coh = sum w_i X_coh(t_i) (retained xi and one shared residual Gaussian space)."""
        beta = self.prob.beta
        M = self.f.M
        r = self.f.LR.shape[1]
        xi = rng.standard_normal((K, M))
        eta = rng.standard_normal((K, r))
        J = np.zeros(K)
        for wi, i in zip(w, ids):
            t = self.times[i]
            a = beta * self.f.a(t)[0]
            ar = beta * self.f.ar(t)[0]
            Yshift = self.hs.basis(xi - a[None, :]) @ self.Y[i]
            J += wi * np.exp(xi @ a - 0.5 * a @ a + eta @ ar - 0.5 * ar @ ar) * Yshift
        return J


# ----------------------------------------------------------------------------- full finite target
def full_target_samples(prob, factor, ids, w, K, rng, steps_per_cell=8, return_paths=False):
    """Exact samples (up to RK4) of J_full = sum w_i X_full(t_i) for the full finite factor.

    X_full(t)(zeta) = exp(h_t.zeta - |h_t|^2/2) Y(t; zeta - h_t), where for a fixed parameter vector
    z the pulled-back field solves  Y' = alpha Y + sum_p c_p e^{(p-1) Psi} Y^p,  Psi(s) = h_s.z + |h_s|^2/2.
    """
    beta = prob.beta
    M, r = factor.M, factor.LR.shape[1]
    zeta = rng.standard_normal((K, M + r))
    grid = np.concatenate([[0.0], factor.grid])
    out = np.zeros(K)
    Xt = {}
    for wi, i in zip(w, ids):
        t_end = grid[i]
        h_end = beta * np.concatenate([factor.a(t_end)[0], factor.ar(t_end)[0]])
        z = zeta - h_end[None, :]
        y = np.full(K, prob.x0, dtype=float)
        for k in range(i):
            hh = (grid[k + 1] - grid[k]) / steps_per_cell
            t = grid[k]
            for _ in range(steps_per_cell):
                def f(tt, yy):
                    h = beta * np.concatenate([factor.a(tt)[0], factor.ar(tt)[0]])
                    psi = z @ h + 0.5 * h @ h
                    out_ = prob.alpha * yy
                    for p, c in prob.poly.items():
                        out_ = out_ + c * np.exp((p - 1) * psi) * yy ** p
                    return out_
                k1 = f(t, y); k2 = f(t + hh / 2, y + hh / 2 * k1)
                k3 = f(t + hh / 2, y + hh / 2 * k2); k4 = f(t + hh, y + hh * k3)
                y = y + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                t += hh
        X = np.exp(zeta @ h_end - 0.5 * h_end @ h_end) * y
        Xt[i] = X
        out += wi * X
    return (out, Xt) if return_paths else out


def coupled_samples(prob, factor, solver, ids, w, K, rng, steps_per_cell=8):
    """J_full, J_coh, J_active on the SAME Gaussian vector zeta = (xi, eta) (natural coupling).

    J_coh(zeta) = sum w_i exp(h_i.zeta - |h_i|^2/2) Y_i(xi - a_i)  (coherent reconstruction),
    J_act(zeta) = sum w_i exp(a_i.xi - |a_i|^2/2)  Y_i(xi - a_i)   (= Gamma(Pi) J_coh).
    """
    beta = prob.beta
    M, r = factor.M, factor.LR.shape[1]
    zeta = rng.standard_normal((K, M + r))
    xi = zeta[:, :M]
    grid = np.concatenate([[0.0], factor.grid])
    Jf, Jc, Ja = np.zeros(K), np.zeros(K), np.zeros(K)
    for wi, i in zip(w, ids):
        t_end = grid[i]
        a = beta * factor.a(t_end)[0]
        ar = beta * factor.ar(t_end)[0]
        h_end = np.concatenate([a, ar])
        # coherent and active reconstructions from the RCLH coefficients
        Ysh = solver.hs.basis(xi - a[None, :]) @ solver.Y[i]
        ea = np.exp(xi @ a - 0.5 * a @ a)
        Jc += wi * ea * np.exp(zeta[:, M:] @ ar - 0.5 * ar @ ar) * Ysh
        Ja += wi * ea * Ysh
        # full finite target along the shifted parameter
        z = zeta - h_end[None, :]
        y = np.full(K, prob.x0, dtype=float)
        for k in range(i):
            hh = (grid[k + 1] - grid[k]) / steps_per_cell
            t = grid[k]
            for _ in range(steps_per_cell):
                def f(tt, yy):
                    h = beta * np.concatenate([factor.a(tt)[0], factor.ar(tt)[0]])
                    psi = z @ h + 0.5 * h @ h
                    o = prob.alpha * yy
                    for p, c in prob.poly.items():
                        o = o + c * np.exp((p - 1) * psi) * yy ** p
                    return o
                k1 = f(t, y); k2 = f(t + hh / 2, y + hh / 2 * k1)
                k3 = f(t + hh / 2, y + hh / 2 * k2); k4 = f(t + hh, y + hh * k3)
                y = y + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                t += hh
        Jf += wi * np.exp(zeta @ h_end - 0.5 * h_end @ h_end) * y
    return Jf, Jc, Ja


def solver_error_certificate(prob, factor, solver, ids, w, K, rng, conf=0.99, audit_K=2000):
    """Statistical solver certificate (route H for equations): ||J_full - J_coh||_2 from K exact coupled
    samples of the full finite target (pathwise ODE, RK4) and the coherent reconstruction.

    Returns the RMS estimate, a one-sided upper confidence bound at level conf for E[(J_full - J_coh)^2]
    (CLT on the squared error; label 'statistical (asymptotic)'), and an RK4 step-doubling audit
    (RMS change of J_full when the substeps per cell are doubled, on audit_K samples)."""
    from scipy.stats import norm as _norm
    st = rng.bit_generator.state
    Jf, Jc, _ = coupled_samples(prob, factor, solver, ids, w, K, rng)
    z2 = (Jf - Jc) ** 2
    m, s = float(z2.mean()), float(z2.std(ddof=1))
    ucb = m + _norm.ppf(conf) * s / math.sqrt(K)
    r1 = np.random.default_rng(12345)
    a8, _, _ = coupled_samples(prob, factor, solver, ids, w, audit_K, r1, steps_per_cell=8)
    r2 = np.random.default_rng(12345)
    a16, _, _ = coupled_samples(prob, factor, solver, ids, w, audit_K, r2, steps_per_cell=16)
    return {"rms": math.sqrt(m), "ucb_rms": math.sqrt(max(ucb, 0.0)), "conf": conf, "K": K,
            "rk4_audit_rms": float(np.sqrt(np.mean((a8 - a16) ** 2))), "label": "statistical (asymptotic)"}


def defect_bounds_tamed(solver, ids, n_gl=6):
    """Dissipation-aware ("self-tamed") variant of the v13 cubic full-defect bound.

    Keeping y^2 + y u + u^2 >= (3/4) u^2 in the one-sided inequality gives, pathwise for e = Y - U,
        |e|' <= (g - lam_s)|e| + |r_s|,   lam_s = (3/4) kappa e^{2Q_s} U_s^2 >= 0.
    Split r_s = (U'_s - g U_s) + kappa e^{2Q_s} U_s^3; the second part equals (4/3) lam_s U_s, and
        int_0^t e^{int_s^t (g - lam)} lam_s |U_s| ds <= e^{g_+ t} sup_{s<=t} |U_s|,
    so the term that carries the exponential moments in v13 is replaced by (4/3) e^{g_+ t} sup|U|:
        D'_t = e^{gt}||x0 - U_0|| + int_0^t e^{g(t-s)} ||U'_s - g U_s|| ds
               + (4/3) e^{g_+ t} (||U_0|| + int_0^t ||U'_s|| ds),
    all norms in the same shifted measure mu_t; B'_t = exp(|h_t|^2/2) D'_t >= ||X_full(t) - X_coh(t)||_2.
    Valid for a single odd saturation power p = 3 (kappa = -c_3 >= 0).  Returns {i: (B_v13, B_tamed)}."""
    pb = solver.prob
    sat = {p: c for p, c in pb.poly.items() if p >= 2}
    if set(sat) != {3} or sat[3] > 0 or pb.poly.get(0, 0.0) != 0.0:
        return None
    g = pb.alpha + pb.poly.get(1, 0.0)
    v13 = defect_bounds(solver, ids, n_gl)
    beta, fac, hs = pb.beta, solver.f, solver.hs
    M = fac.M
    q = max(hs.P * 3 + 1, hs.q)
    x, wq = np.polynomial.hermite_e.hermegauss(q)
    wq = wq / math.sqrt(2 * math.pi)
    Zg = np.stack([m.ravel() for m in np.meshgrid(*([x] * M), indexing="ij")], axis=1)
    Wg = np.prod(np.stack([m.ravel() for m in np.meshgrid(*([wq] * M), indexing="ij")], axis=1), axis=1)
    grid = solver.times
    xg, wg = np.polynomial.legendre.leggauss(n_gl)
    out = {}
    for i in ids:
        t = grid[i]
        a_t = beta * fac.a(t)[0]
        h_t = np.concatenate([a_t, beta * fac.ar(t)[0]])
        Bm = hs.basis(a_t[None, :] + Zg)                         # basis at the mu_t-shifted nodes (retained)
        nrm = lambda coef: math.sqrt(max(float(Wg @ (Bm @ coef) ** 2), 0.0))
        e0 = nrm(solver.Y[0] - np.eye(len(hs.idx))[0] * pb.x0)        # ||U_0 - x0|| (constant basis first)
        lin, var_int = 0.0, 0.0
        for k in range(i):
            t0, t1 = grid[k], grid[k + 1]
            dY = (solver.Y[k + 1] - solver.Y[k]) / (t1 - t0)
            for xx, ww in zip(xg, wg):
                s = t0 + 0.5 * (t1 - t0) * (xx + 1)
                th = (s - t0) / (t1 - t0)
                ycoef = (1 - th) * solver.Y[k] + th * solver.Y[k + 1]
                lin += 0.5 * (t1 - t0) * ww * math.exp(g * (t - s)) * nrm(dY - g * ycoef)
                var_int += 0.5 * (t1 - t0) * ww * nrm(dY)
        sup_u = nrm(solver.Y[0]) + var_int
        D_t = math.exp(g * t) * e0 + lin + (4.0 / 3.0) * math.exp(max(g, 0.0) * t) * sup_u
        out[i] = (v13[i], math.exp(0.5 * float(h_t @ h_t)) * D_t)
    return out


def defect_bounds(solver, ids, n_gl=6):
    """Paper IX v13 Thm (polynomial full-defect) + Cor (physical output), cubic/odd saturation.

    Target pulled-back field  B_s(y) = g y + sum_{p in sat} c_p e^{(p-1)Q_s} y^p,  c_p <= 0 for p >= 2,
    g = alpha + c_1.  Valid (v13) when every saturation power is odd (any x0) -- the quadratic case
    needs the positive-part modification and is not implemented here (returns None).
    U = computed Lawson polynomial, coefficients interpolated linearly between grid nodes.
    Returns B_t = exp(|h_t|^2/2) D_t >= ||X_full(t) - X_coh(t)||_2 for each requested grid index.
    All Gaussian expectations are exact polynomial Gauss--Hermite sums; the time integral is
    Gauss--Legendre per cell (numerical, not an enclosure).
    """
    pb = solver.prob
    sat = {p: c for p, c in pb.poly.items() if p >= 2}
    if any(c > 0 for c in sat.values()) or any(p % 2 == 0 for p in sat):
        return None
    g = pb.alpha + pb.poly.get(1, 0.0) + pb.poly.get(0, 0.0) * 0.0
    if pb.poly.get(0, 0.0) != 0.0:
        return None
    beta, fac, hs = pb.beta, solver.f, solver.hs
    M = fac.M
    pmax = max(sat) if sat else 1
    q = max(hs.P * pmax + 1, hs.q)
    x, wq = np.polynomial.hermite_e.hermegauss(q)
    wq = wq / math.sqrt(2 * math.pi)
    Zg = np.stack([m.ravel() for m in np.meshgrid(*([x] * M), indexing="ij")], axis=1)
    Wg = np.prod(np.stack([m.ravel() for m in np.meshgrid(*([wq] * M), indexing="ij")], axis=1), axis=1)
    grid = solver.times
    xg, wg = np.polynomial.legendre.leggauss(n_gl)
    out = {}
    for i in ids:
        t = grid[i]
        a_t = beta * fac.a(t)[0]
        h_t = np.concatenate([a_t, beta * fac.ar(t)[0]])
        total = 0.0
        for k in range(i):
            t0, t1 = grid[k], grid[k + 1]
            dY = (solver.Y[k + 1] - solver.Y[k]) / (t1 - t0)
            for xx, ww in zip(xg, wg):
                s = t0 + 0.5 * (t1 - t0) * (xx + 1)
                th = (s - t0) / (t1 - t0)
                ycoef = (1 - th) * solver.Y[k] + th * solver.Y[k + 1]
                a_s = beta * fac.a(s)[0]
                h_s = np.concatenate([a_s, beta * fac.ar(s)[0]])
                base = float(h_s @ h_t + 0.5 * h_s @ h_s)
                nh2 = float(h_s @ h_s)

                def E(Fvals_fn, c):
                    pts = a_t[None, :] + c * a_s[None, :] + Zg
                    return math.exp(c * base + 0.5 * c * c * nh2) * float(Wg @ Fvals_fn(pts))
                Bm = lambda pts: hs.basis(pts)
                A2 = E(lambda P_: (Bm(P_) @ (dY - g * ycoef)) ** 2, 0.0)
                cross, sq = 0.0, 0.0
                for p, c in sat.items():
                    cross += 2 * (-c) * E(lambda P_, p=p: (Bm(P_) @ (dY - g * ycoef)) * (Bm(P_) @ ycoef) ** p, p - 1)
                for p, c in sat.items():
                    for p2, c2 in sat.items():
                        sq += (-c) * (-c2) * E(lambda P_, p=p, p2=p2: (Bm(P_) @ ycoef) ** (p + p2), (p - 1) + (p2 - 1))
                r2 = max(A2 + cross + sq, 0.0)
                total += 0.5 * (t1 - t0) * ww * math.exp(g * (t - s)) * math.sqrt(r2)
        out[i] = math.exp(0.5 * float(h_t @ h_t)) * total
    return out
