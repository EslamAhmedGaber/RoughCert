"""Monitoring estimators beyond Riemann sums: the conditional-expectation monitor.

Given grid values Y_{t_1..t_n} (G = their sigma-field), the L^2-optimal estimator of Lambda_mu is
    Lambda_hat = E[Lambda_mu | G] = int (S_{D(s,s)} f_s)(m_s) mu(ds),   m_s = E[Y_s | G],
i.e. Gamma(Pi_grid) Lambda_mu with Pi_grid the projection onto span{Y_{t_i}} (residual smoothing lemma).
Its exact error is the ledger term P_mu(Pi_grid); the per-sample time integral uses a quadrature measure
nu_q, whose extra error ||Gamma(Pi)(Lambda_mu - Lambda_{nu_q})||_2 = sqrt(M_Q(nu_q)) is again a ledger term.

R = Lambda_mu - E[Lambda_mu | G] satisfies E[R | G] = 0, so Lambda is a mean-preserving spread of Lambda_hat
(martingale endpoints, gcert.martingale).  For exponential and quadratic outputs the conditional variance
V(G) = Var(Lambda_mu | G) is an explicit quadratic form in per-sample weights:
    Wick / pathwise exponential:  Cov(X_s, X_u | G) = S f(m_s) S f(m_u) (exp(eta^2 D(s,u)) - 1),
    a0 + a1 y + a2 y^2 (zero mean): a1^2 D + 2 a1 a2 (m_s + m_u) D + a2^2 (2 D^2 + 4 m_s m_u D),
with D(s,u) = R(s,u) - R_Pi(s,u) the kriging residual covariance.  Conditional Cantelli,
P(R >= d | G) <= V/(V + d^2), then gives quantile brackets of first order in sqrt(V).

Certified forms (P1 Proposition condvar):
  quadratic_cond_mean / quadratic_cond_var   closed-form quadratic forms d0 + l.xi + xi'A xi, c0 + b.xi + xi'Om xi
                                             (audited coefficients; gaps -> delta and a per-sample inflation)
  ExpCondVar                                 singularity-corrected per-sample V for exponential outputs
  cantelli_dominating                        dominating variables T+- = X +- sqrt(v) U, DKW-Massart band
  cantelli_dominating_binomial               the same draws with exact binomial order statistics
  cantelli_rb / rb_eps                       deterministic (Rao-Blackwellised) form, Jensen-Chernoff slack
  scarf_excess, quadratic_form_call          conditional two-moment call bound; exact E(Q - k)_+ of a quadratic form
cantelli_quantile (grid + Hoeffding union) is the earlier prototype kept for the E10 records.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import cholesky, solve_triangular

from .functionals import WickExponential, PathwiseExponential, Polynomial
from .quadrature import graded_panel, Measure
from .surrogate import smoothed


def _sigma(field, s):
    return np.sqrt(np.maximum(field.var(s), 0.0))


class ConditionalMonitor:
    """E[Lambda_mu | Y_grid] per sample, its conditional variance (where explicit), and exact radii."""

    def __init__(self, field, func, T, grid, q=6, levels=3, weight=1.0, nugget=0.0):
        """weight: constant density of mu on [0, T]; nugget: variance of independent noise on the grid
        values (conditioning on noisy observations is again a Gaussian projection)."""
        self.field, self.func, self.T = field, func, float(T)
        self.grid = np.asarray(grid, float)
        self.weight, self.nugget = float(weight), float(nugget)
        C = field.R(self.grid[:, None], self.grid[None, :]) + self.nugget * np.eye(self.grid.size)
        self.L = cholesky(C + 1e-14 * np.trace(C) / self.grid.size * np.eye(self.grid.size), lower=True)
        cuts = np.unique(np.concatenate([[0.0], self.grid, [self.T]]))
        xs, ws = [], []
        for a, b in zip(cuts[:-1], cuts[1:]):
            x, w = graded_panel(a, b, n=q, levels=levels)
            xs.append(x)
            ws.append(w)
        self.s, self.w = np.concatenate(xs), self.weight * np.concatenate(ws)
        RA = field.R(self.grid[:, None], self.s[None, :])
        self.G = solve_triangular(self.L, RA, lower=True).T          # (n_q, n): m_s = G xi
        self.Dd = np.maximum(field.var(self.s) - np.sum(self.G ** 2, axis=1), 0.0)
        self.nu = Measure.atomic(self.s, self.w, T=self.T, name="ce-quadrature")
        self._Dmat = None

    # -- per-sample quantities --------------------------------------------------------------
    def xi_from_grid(self, Y):
        return solve_triangular(self.L, np.atleast_2d(Y).T, lower=True).T

    def residual_cov(self):
        if self._Dmat is None:
            S, U = self.s[:, None], self.s[None, :]
            self._Dmat = self.field.R(S, U) - self.G @ self.G.T
        return self._Dmat

    def estimate(self, xi, with_variance=False, chunk=2000):
        """Lambda_hat (and V(G) when explicit) for whitened grid coordinates xi (K, n)."""
        K = xi.shape[0]
        lam = np.empty(K)
        var = np.empty(K) if with_variance else None
        f = self.func
        if with_variance:
            Dm = self.residual_cov()
            if isinstance(f, (WickExponential, PathwiseExponential)):
                Ek = np.expm1(f.eta ** 2 * Dm) * self.w[:, None] * self.w[None, :]
            elif isinstance(f, Polynomial) and len(f.a) <= 3:
                a = np.concatenate([f.a, np.zeros(3 - len(f.a))])
                W2 = self.w[:, None] * self.w[None, :]
            else:
                raise NotImplementedError("conditional variance is explicit for exponential and quadratic outputs")
        for c0 in range(0, K, chunk):
            sl = slice(c0, min(c0 + chunk, K))
            m = xi[sl] @ self.G.T                                       # (k, n_q)
            Sf = smoothed(f, self.s, m, self.Dd)
            lam[sl] = Sf @ self.w
            if with_variance:
                if isinstance(f, (WickExponential, PathwiseExponential)):
                    var[sl] = np.einsum("kq,kq->k", Sf @ Ek, Sf)
                else:
                    lin = a[1] * np.ones_like(m) + 2 * a[2] * m           # a1 + 2 a2 m_s  (gradient form)
                    var[sl] = (np.einsum("kq,kq->k", (lin * self.w) @ Dm, lin * self.w)
                               + 2 * a[2] ** 2 * float(np.sum(W2 * Dm * Dm)))
        return (lam, var) if with_variance else lam

    def sample(self, K, rng, with_variance=False):
        xi = rng.standard_normal((K, self.grid.size))                  # whitened grid values
        return self.estimate(xi, with_variance)


def _cell_rule(a, b, q, levels):
    return graded_panel(a, b, n=q, levels=levels)


def ce_radius(cm, near=(8, 4), far=3, audit=True):
    """P_mu(Pi_grid) = intint [K(R) - K(R_Pi)] dmu dmu for Lebesgue mu on [0, T], cell by cell.

    The integrand vanishes as D = R - R_Pi -> 0 and is concentrated on neighbouring cells.  Diagonal cell
    blocks use the graded triangle rule, neighbouring blocks graded tensor rules, all farther pairs one
    q x q Gauss--Legendre tensor rule per cell (one matrix product).  The integrand is symmetric, so each
    unordered pair is evaluated once.  Returns (P, gap), gap = change under refinement of every rule."""
    from .quadrature import _diag_triangle, gl
    field, f = cm.field, cm.func
    cuts = np.unique(np.concatenate([[0.0], cm.grid, [cm.T]]))
    cells = list(zip(cuts[:-1], cuts[1:]))
    nC = len(cells)

    def loads(s):
        return solve_triangular(cm.L, field.R(cm.grid[:, None], np.asarray(s, float)[None, :]), lower=True).T

    def kdiff(S, U, Rp):
        sg, su = _sigma(field, S), _sigma(field, U)
        return f.kernel(S, U, field.R(S, U), sg, su) - f.kernel(S, U, Rp, sg, su)

    def total(qn, ql, qf):
        tot = 0.0
        nodes = [graded_panel(a, b, n=qn, levels=ql) for a, b in cells]
        G = [loads(s) for s, _ in nodes]
        for i, (a, b) in enumerate(cells):                              # diagonal blocks
            # K(R) has the diagonal cusp: triangle rule, needs R only; K(R_Pi) is smooth across the
            # diagonal: tensor rule on the cell's graded nodes, needs loadings at 1-D nodes only
            S, U, W = _diag_triangle(a, b, qn, ql)
            sg, su = _sigma(field, S), _sigma(field, U)
            tot += 2.0 * float(np.sum(W * f.kernel(S, U, field.R(S, U), sg, su)))
            s, ws = nodes[i]
            Rp = G[i] @ G[i].T
            SS, UU = np.meshgrid(s, s, indexing="ij")
            gs = _sigma(field, s)
            Kp = f.kernel(SS.ravel(), UU.ravel(), Rp.ravel(), np.repeat(gs, s.size), np.tile(gs, s.size))
            tot -= float(np.sum(np.outer(ws, ws) * np.asarray(Kp).reshape(Rp.shape)))
        for i in range(nC - 1):                                         # neighbouring blocks
            (s, ws), (u, wu) = nodes[i], nodes[i + 1]
            Rp = G[i] @ G[i + 1].T
            SS, UU = np.meshgrid(s, u, indexing="ij")
            tot += 2.0 * float(np.sum(np.outer(ws, wu) * kdiff(SS.ravel(), UU.ravel(), Rp.ravel()).reshape(Rp.shape)))
        fn = [gl(a, b, qf) for a, b in cells]                           # far field, |i - j| >= 2
        s_all = np.concatenate([x for x, _ in fn])
        w_all = np.concatenate([w for _, w in fn])
        cid = np.repeat(np.arange(nC), qf)
        G_all = loads(s_all)
        for c0 in range(0, s_all.size, 512):
            sl = slice(c0, min(c0 + 512, s_all.size))
            mask = (cid[None, :] - cid[sl][:, None]) >= 2               # j >= i + 2 (unordered pairs once)
            if not mask.any():
                continue
            Rp = G_all[sl] @ G_all.T
            SS = np.broadcast_to(s_all[sl][:, None], Rp.shape)
            UU = np.broadcast_to(s_all[None, :], Rp.shape)
            v = kdiff(SS[mask], UU[mask], Rp[mask])
            tot += 2.0 * float(np.sum((w_all[sl][:, None] * w_all[None, :])[mask] * v))
        return tot

    P1 = cm.weight ** 2 * total(near[0], near[1], far)
    if not audit:
        return P1, 0.0
    P2 = cm.weight ** 2 * total(near[0] + 4, near[1] + 2, far + 2)
    return P2, abs(P2 - P1)


def quadratic_cond_var(cm, a1=0.0, a2=1.0, near=(8, 4), far=3, audit=True, norms=False):
    """Exact conditional variance of a quadratic output a0 + a1 y + a2 y^2 (zero mean function) given the grid:
        V(xi) = c0 + b.xi + xi' Om xi,   m_s = G(s).xi,
        c0 = a1^2 iint D + 2 a2^2 iint D^2,  b = 4 a1 a2 iint G(s) D(s,u),  Om = 4 a2^2 iint G(s) G(u)' D(s,u),
    all integrals over mu x mu (Lebesgue/T weight of the monitor), evaluated once with the cell-local rules of
    ce_radius (diagonal triangle rule, graded neighbours, far-field tensor).  Returns (c0, b, Om, gap); with
    norms=True, gap = (|d c0|, ||d b||_2, ||d Om||_2), so |dV(xi)| <= gap[0] + gap[1] |xi| + gap[2] |xi|^2."""
    from .quadrature import _diag_triangle, gl
    field = cm.field
    cuts = np.unique(np.concatenate([[0.0], cm.grid, [cm.T]]))
    cells = list(zip(cuts[:-1], cuts[1:]))
    n = cm.grid.size

    def loads(s):
        return solve_triangular(cm.L, field.R(cm.grid[:, None], np.asarray(s, float)[None, :]), lower=True).T

    def accumulate(S, U, W, GS, GU):
        D = field.R(S, U) - np.sum(GS * GU, axis=1)
        WD = W * D
        return (float(np.sum(WD)), float(np.sum(WD * D)), (GS * WD[:, None]).sum(axis=0),
                GS.T @ (GU * WD[:, None]))

    def total(qn, ql, qf, lf):
        I1, I2, Ib, IO = 0.0, 0.0, np.zeros(n), np.zeros((n, n))
        pairs = []
        nodes = [graded_panel(a, b, n=qn, levels=ql) for a, b in cells]
        G = [loads(s) for s, _ in nodes]
        for i, (a, b) in enumerate(cells):
            S, U, W = _diag_triangle(a, b, qn, ql)          # u > s triangle; mirror added below
            for c0 in range(0, S.size, 20_000):
                sl = slice(c0, min(c0 + 20_000, S.size))
                GS, GU = loads(S[sl]), loads(U[sl])
                pairs.append(accumulate(S[sl], U[sl], W[sl], GS, GU))
                pairs.append(accumulate(U[sl], S[sl], W[sl], GU, GS))
        for i in range(len(cells) - 1):                     # neighbouring cells, both orders
            for (p, r) in ((i, i + 1), (i + 1, i)):
                (s, ws), (u, wu) = nodes[p], nodes[r]
                SS, UU = np.meshgrid(s, u, indexing="ij")
                pairs.append(accumulate(SS.ravel(), UU.ravel(), np.outer(ws, wu).ravel(),
                                        np.repeat(G[p], u.size, axis=0), np.tile(G[r], (s.size, 1))))
        # far field |i - j| >= 2: the loadings have cusps at the grid points (cell ends), so graded 1-D
        # nodes in every cell, one masked tensor product in row chunks
        fn = [graded_panel(a, b, n=qf, levels=lf) for a, b in cells]
        s_all = np.concatenate([x for x, _ in fn])
        w_all = np.concatenate([w for _, w in fn])
        cid = np.concatenate([np.full(x.size, i) for i, (x, _) in enumerate(fn)])
        G_all = loads(s_all)
        for c0 in range(0, s_all.size, 512):
            sl = slice(c0, min(c0 + 512, s_all.size))
            mask = np.abs(cid[None, :] - cid[sl][:, None]) >= 2
            if not mask.any():
                continue
            Dm = field.R(s_all[sl][:, None], s_all[None, :]) - G_all[sl] @ G_all.T
            WD = np.where(mask, np.outer(w_all[sl], w_all) * Dm, 0.0)
            pairs.append((float(WD.sum()), float(np.sum(WD * Dm)), G_all[sl].T @ WD.sum(axis=1),
                          G_all[sl].T @ (WD @ G_all)))
        for p1, p2, pb, pO in pairs:
            I1 += p1; I2 += p2; Ib += pb; IO += pO
        w2 = cm.weight ** 2
        return w2 * (a1 ** 2 * I1 + 2 * a2 ** 2 * I2), w2 * 4 * a1 * a2 * Ib, w2 * 4 * a2 ** 2 * 0.5 * (IO + IO.T)

    lf = max(1, near[1] // 2 + 1)
    c0, b, Om = total(near[0], near[1], far, lf)
    if not audit:
        return c0, b, Om, ((0.0, 0.0, 0.0) if norms else 0.0)
    c0f, bf, Omf = total(near[0] + 4, near[1] + 2, far + 2, lf + 1)
    if norms:
        gap = (abs(c0f - c0), float(np.linalg.norm(bf - b)), float(np.linalg.norm(Omf - Om, 2)))
    else:
        gap = max(abs(c0f - c0), float(np.max(np.abs(bf - b))) if b.size else 0.0, float(np.max(np.abs(Omf - Om))))
    return c0f, bf, Omf, gap


def quadratic_cond_mean(cm, a0=0.0, a1=0.0, a2=1.0, q=8, levels=4, audit=True):
    """Lambda_hat(xi) = d0 + l.xi + xi' A xi for a quadratic output a0 + a1 y + a2 y^2 (zero mean function):
        d0 = a0 mu([0, T]) + a2 int D(s,s),   l = a1 int G(s),   A = a2 int G(s) G(s)',
    one-dimensional integrals on graded cell rules.  Returns (d0, l, A, gaps), gaps = (|d d0|, ||d l||_2,
    ||d A||_2) under refinement (q, levels) -> (q + 4, levels + 2); the per-sample error then has
    ||E||_2 <= gaps[0] + gaps[1] sqrt(n) + gaps[2] sqrt(n^2 + 2n)."""
    field = cm.field
    cuts = np.unique(np.concatenate([[0.0], cm.grid, [cm.T]]))

    def total(qn, ql):
        xs, ws = [], []
        for a, b in zip(cuts[:-1], cuts[1:]):
            x, w = graded_panel(a, b, n=qn, levels=ql)
            xs.append(x)
            ws.append(w)
        s, w = np.concatenate(xs), cm.weight * np.concatenate(ws)
        G = solve_triangular(cm.L, field.R(cm.grid[:, None], s[None, :]), lower=True).T
        Dd = field.var(s) - np.sum(G ** 2, axis=1)
        return a0 * float(np.sum(w)) + a2 * float(w @ Dd), a1 * (G.T @ w), a2 * (G.T @ (G * w[:, None]))

    d0, l, A = total(q, levels)
    if not audit:
        return d0, l, A, (0.0, 0.0, 0.0)
    d0f, lf, Af = total(q + 4, levels + 2)
    return d0f, lf, Af, (abs(d0f - d0), float(np.linalg.norm(lf - l)), float(np.linalg.norm(Af - A, 2)))


class ExpCondVar:
    """Per-sample conditional variance of an exponential output (Wick or pathwise),
        V(xi) = iint Xh_s Xh_u K(s,u) mu(ds) mu(du),   K = expm1(eta^2 D),   Xh_s = E[X_s | G],
    by a singularity-corrected node rule: with node weights w and the exact row integrals
    k_q = int K(s_q, u) mu(du) (graded 1-D rules split at s_q and at the grid points),
        V ~ Xh' (w w' o K) Xh + sum_q w_q (k_q - sum_q' w_q' K_qq') Xh_q^2,
    which equals V - (1/2) iint K (Xh_s - Xh_u)^2 evaluated by the tensor rule: the remaining integrand
    vanishes quadratically on the diagonal, where K has its cusp.  The rule is exact for constant Xh.
    values(xi) returns V on the node set (q, levels) and on a refined set; their difference is the
    per-sample audit gap (added as an inflation by the caller)."""

    def __init__(self, cm, levels_pair=((6, 3), (10, 5)), row=(8, 6)):
        f = cm.func
        if not isinstance(f, (WickExponential, PathwiseExponential)):
            raise NotImplementedError("ExpCondVar is for exponential outputs")
        self.sets = []
        for q, lv in levels_pair:
            c = ConditionalMonitor(cm.field, f, cm.T, cm.grid, q=q, levels=lv, weight=cm.weight, nugget=cm.nugget)
            K = np.expm1(f.eta ** 2 * c.residual_cov())
            k = self._row_integrals(c, f.eta, row)
            M = c.w[:, None] * c.w[None, :] * K
            d = c.w * k - M.sum(axis=1)
            self.sets.append((c, M, d))

    @staticmethod
    def _row_integrals(c, eta, row):
        field = c.field
        cuts = np.unique(np.concatenate([[0.0], c.grid, [c.T]]))
        loads = lambda u: solve_triangular(c.L, field.R(c.grid[:, None], np.asarray(u, float)[None, :]), lower=True).T
        base = [graded_panel(a, b, n=row[0], levels=row[1]) for a, b in zip(cuts[:-1], cuts[1:])]
        ub = [loads(u) for u, _ in base]
        out = np.empty(c.s.size)
        cell = np.clip(np.searchsorted(cuts, c.s, side="right") - 1, 0, len(base) - 1)
        for i, (sq, Gq) in enumerate(zip(c.s, c.G)):
            tot = 0.0
            for j, ((u, wu), Gu) in enumerate(zip(base, ub)):
                if j == cell[i]:
                    a, b = cuts[j], cuts[j + 1]
                    parts = [graded_panel(a, sq, n=row[0], levels=row[1]), graded_panel(sq, b, n=row[0], levels=row[1])]
                    u = np.concatenate([p[0] for p in parts])
                    wu = np.concatenate([p[1] for p in parts])
                    Gu = loads(u)
                D = field.R(np.full(u.size, sq), u) - Gu @ Gq
                tot += float(np.sum(wu * np.expm1(eta ** 2 * D)))
            out[i] = c.weight * tot
        return out

    def values(self, xi, chunk=4000):
        """(V on the first node set, V on the refined set) for whitened grid values xi (K, n)."""
        out = []
        for c, M, d in self.sets:
            V = np.empty(xi.shape[0])
            for c0 in range(0, xi.shape[0], chunk):
                sl = slice(c0, min(c0 + chunk, xi.shape[0]))
                m = xi[sl] @ c.G.T
                Sf = smoothed(c.func, c.s, m, c.Dd)
                V[sl] = np.sum((Sf @ M) * Sf, axis=1) + (Sf * Sf) @ d
            out.append(V)
        return out


def quadratic_delta(gaps, n):
    """L2 bound on the error of a quadratic form d0 + l.xi + xi'A xi (xi ~ N(0, I_n)) from its coefficient gaps."""
    return gaps[0] + gaps[1] * math.sqrt(n) + gaps[2] * math.sqrt(n * n + 2 * n)


# -- conditional-variance certificates (sharp given V) ------------------------------------------------------
def dkw_eps(K, a):
    """Level slack of the one-sided DKW--Massart bound at a/2 per side (total failure probability a)."""
    return math.sqrt(math.log(2.0 / a) / (2.0 * K))


def binomial_order_index(K, p, a_side, upper):
    """Exact distribution-free order-statistic index for one quantile of K iid draws.
    upper: smallest m with P(Bin(K, p) <= m - 1) >= 1 - a_side, so P(T_(m) >= Q_T(p)) >= 1 - a_side;
    lower: largest j with P(Bin(K, p) <= j - 1) <= a_side, so P(T_(j) <= Q_T(p)) >= 1 - a_side (0 if none)."""
    from scipy.stats import binom
    if upper:
        m = int(binom.ppf(1.0 - a_side, K, p)) + 1
        while binom.cdf(m - 1, K, p) < 1.0 - a_side:
            m += 1
        while m > 1 and binom.cdf(m - 2, K, p) >= 1.0 - a_side:
            m -= 1
        return m
    j = int(binom.ppf(a_side, K, p))
    while j >= 1 and binom.cdf(j - 1, K, p) > a_side:
        j -= 1
    while binom.cdf(j, K, p) <= a_side:
        j += 1
    return j


def cantelli_dominating_binomial(x, v, alpha, a=0.05, delta=0.0, rng=None, beta_c=1e-4):
    """cantelli_dominating with exact binomial order-statistic bounds instead of the DKW band.  The dominating
    variables reduce the certificate to ONE quantile of each of two iid samples (T+ and T-), so no uniformity in
    t is needed: q_alpha <= T+_(m) and q_alpha >= T-_(j) with the indices of binomial_order_index at a/2 per
    side.  A G-measurable error ||E||_2 <= delta uses one level shift fixed in advance, beta = beta_c (1 - alpha).
    Same draws as cantelli_dominating for the same rng state."""
    rng = np.random.default_rng() if rng is None else rng
    x, v = np.asarray(x, float), np.maximum(np.asarray(v, float), 0.0)
    K = x.size
    sv = np.sqrt(v)
    W1, W2 = rng.random(K), rng.random(K)
    Tp = np.sort(x + sv * np.sqrt(W1 / (1.0 - W1)))
    Tm = np.sort(x - sv * np.sqrt(W2 / (1.0 - W2)))
    b = beta_c * (1.0 - alpha) if delta > 0 else 0.0
    s = delta / math.sqrt(b) if b > 0 else 0.0
    m = binomial_order_index(K, min(alpha + b, 1.0), a / 2, upper=True)
    j = binomial_order_index(K, max(alpha - b, 0.0), a / 2, upper=False)
    up = float(Tp[m - 1]) + s if m <= K else math.inf
    lo = float(Tm[j - 1]) - s if j >= 1 else -math.inf
    return {"lower": lo, "upper": up, "index": (j, m), "center": float(np.quantile(x, alpha)),
            "method": "randomised, binomial order statistics", "Tp": Tp, "Tm": Tm}


def dkw_from_sorted(Tp, Tm, alpha, dkw_a, delta=0.0):
    """The DKW bracket of cantelli_dominating computed from already drawn (sorted) dominating samples."""
    K = Tp.size
    eps = dkw_eps(K, dkw_a)
    betas = [0.0] if delta <= 0 else [(1 - alpha) * c for c in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.2)]
    up, lo = math.inf, -math.inf
    for b in betas:
        s = delta / math.sqrt(b) if b > 0 else 0.0
        ku, kl = math.ceil(K * (alpha + b + eps)), math.ceil(K * (alpha - b - eps))
        if ku <= K:
            up = min(up, float(Tp[ku - 1]) + s)
        if kl >= 1:
            lo = max(lo, float(Tm[kl - 1]) - s)
    return {"lower": lo, "upper": up, "eps": eps}


def cantelli_dominating(x, v, alpha, dkw_a=0.05, delta=0.0, rng=None, betas=None, eps=None):
    """Quantile bracket from per-sample conditional means x_k = E[Lambda | G_k] and conditional variances
    v_k >= Var(Lambda | G_k), by randomised conditional Cantelli.

    With U ~ Phi, P(U <= y) = y^2/(1 + y^2) (y > 0), the variables
        T+ = x + sqrt(v) U,   T- = x - sqrt(v) U'
    satisfy P(Lambda < t) >= P(T+ < t) and P(Lambda <= t) <= P(T- <= t) for every t, and these are the exact
    worst cases over all conditional laws with the given means and variances (Cantelli is attained per
    sample and per level).  The T's are iid exact draws, so the one-sided DKW--Massart bound applies to
    their empirical laws with no grid and no union bound:
        q_alpha <= T+_(ceil(K(alpha + beta + eps))) + delta/sqrt(beta),
        q_alpha >= T-_(ceil(K(alpha - beta - eps))) - delta/sqrt(beta),
    eps = dkw_eps(K, a); a G-measurable error ||E||_2 <= delta in x is handled by the level shift beta
    (Chebyshev, optimised over a grid: one DKW event covers every beta).  eps overrides the DKW slack
    (eps = 0 gives the population-level endpoints of the empirical law, used for allocation only)."""
    rng = np.random.default_rng() if rng is None else rng
    x, v = np.asarray(x, float), np.maximum(np.asarray(v, float), 0.0)
    K = x.size
    eps = dkw_eps(K, dkw_a) if eps is None else float(eps)
    sv = np.sqrt(v)
    W1, W2 = rng.random(K), rng.random(K)
    Tp = np.sort(x + sv * np.sqrt(W1 / (1.0 - W1)))
    Tm = np.sort(x - sv * np.sqrt(W2 / (1.0 - W2)))
    if betas is None:
        betas = [0.0] if delta <= 0 else [(1 - alpha) * c for c in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.2)]
    up, lo = math.inf, -math.inf
    for b in betas:
        s = delta / math.sqrt(b) if b > 0 else 0.0
        ku, kl = math.ceil(K * (alpha + b + eps)), math.ceil(K * (alpha - b - eps))
        if ku <= K:
            up = min(up, float(Tp[ku - 1]) + s)
        if kl >= 1:
            lo = max(lo, float(Tm[kl - 1]) - s)
    return {"lower": lo, "upper": up, "eps": eps, "center": float(np.quantile(x, alpha)), "method": "randomised"}


def rb_eps(K, a_side):
    """Uniform one-sided slack for averages of CDF-valued kernels: if g_t(z) is a CDF in t for every z, then
    sup_t (E g_t(Z) - mean_k g_t(Z_k)) <= eps with probability >= 1 - a_side.

    mean_k g_t(Z_k) = E[F_K(t) | Z] for the empirical CDF F_K of a randomisation T_k ~ g_.(Z_k), so the
    deviation is at most Y = E[S | Z] with S the one-sided DKW supremum; Jensen and Chernoff give
    P(Y > eps) <= inf_lam e^{-lam eps} E e^{lam S}, and Massart's tail P(S > s) <= exp(-2 K s^2) for
    s >= s0 = sqrt(ln 2 / (2K)) (1 below s0) gives
        E e^{lam S} <= e^{lam s0} + lam e^{lam^2/(8K)} sqrt(pi/(8K)) erfc(sqrt(2K)(s0 - lam/(4K)))."""
    from scipy.optimize import brentq, minimize_scalar
    from scipy.special import erfc
    s0 = math.sqrt(math.log(2.0) / (2.0 * K))

    def log_bound(eps):
        def f(ll):
            lam = math.exp(ll)
            z = math.sqrt(2.0 * K) * (s0 - lam / (4.0 * K))
            t2 = math.log(lam) + lam * lam / (8.0 * K) + 0.5 * math.log(math.pi / (8.0 * K)) + math.log(erfc(z))
            return float(np.logaddexp(lam * s0, t2)) - lam * eps
        return minimize_scalar(f, bounds=(0.0, math.log(1e10)), method="bounded", options={"xatol": 1e-7}).fun

    return brentq(lambda e: log_bound(e) - math.log(a_side), s0 * 1.0001, 1.0, xtol=1e-12)


def cantelli_rb(x, v, alpha, dkw_a=0.05, delta=0.0, betas=None, iters=80):
    """Deterministic (Rao--Blackwellised) form of cantelli_dominating: the conditional-Cantelli probability
    curves averaged over the samples, corrected by rb_eps(K, a/2) on each side; endpoints by bisection
    (both curves are continuous and nondecreasing in t)."""
    x, v = np.asarray(x, float), np.maximum(np.asarray(v, float), 0.0)
    K = x.size
    eps = rb_eps(K, dkw_a / 2)
    sv = np.sqrt(v)

    def low_p(t):                                       # mean of P(T+ < t | Z) <= P(Lambda < t)
        d = t - x
        return float(np.mean(np.where(d > 0, d * d / (v + d * d + 1e-300), 0.0)))

    def up_p(t):                                        # mean of P(T- <= t | Z) >= P(Lambda <= t)
        d = x - t
        return float(np.mean(np.where(d > 0, v / (v + d * d + 1e-300), 1.0)))

    def cross(fun, level, a, b):                        # smallest t in [a, b] with fun(t) >= level
        for _ in range(iters):
            m = 0.5 * (a + b)
            a, b = (a, m) if fun(m) >= level else (m, b)
        return b

    if betas is None:
        betas = [0.0] if delta <= 0 else [(1 - alpha) * c for c in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.2)]
    up, lo = math.inf, -math.inf
    for b in betas:
        s = delta / math.sqrt(b) if b > 0 else 0.0
        Lu, Ll = alpha + b + eps, alpha - b - eps
        if Lu < 1:
            hi = float(np.max(x + sv * math.sqrt(Lu / (1 - Lu)))) + 1e-12
            up = min(up, cross(low_p, Lu, float(np.min(x)), hi) + s)
        if Ll > 0:
            a0 = float(np.min(x - sv * math.sqrt((1 - Ll) / Ll))) - 1e-12
            lo = max(lo, cross(up_p, Ll, a0, float(np.max(x))) - s)
    return {"lower": lo, "upper": up, "eps": eps, "center": float(np.quantile(x, alpha)), "method": "rao-blackwell"}


def quadratic_form_call(d0, l, A, k, tol=1e-13):
    """Exact E(Q - k)_+ for Q = d0 + l.xi + xi' A xi, xi ~ N(0, I_n) (the law of the monitor of a quadratic
    output).  With A = Q' diag(lam) Q and b = Q' l, Q is a sum of independent lam_j z_j^2 + b_j z_j, whose
    characteristic function is prod_j (1 - 2iu lam_j)^{-1/2} exp(-u^2 b_j^2 / (2(1 - 2iu lam_j))); then
    E(Q - k)_+ = (E Q - k)/2 + E|Q - k|/2 and E|Q - k| = (2/pi) int_0^inf (1 - Re phi(u) e^{-iuk}) u^{-2} du.
    Returns (value, quadrature error estimate)."""
    from scipy import integrate
    lam, V = np.linalg.eigh(0.5 * (A + A.T))
    b = V.T @ np.asarray(l, float)
    s = math.sqrt(float(np.sum(2 * lam ** 2 + b ** 2)))
    if s == 0.0:
        return max(d0 - k, 0.0), 0.0
    la, bb, kk = lam / s, b / s, (k - d0) / s

    def phi(u):
        z = 1.0 - 2j * u * la
        return np.exp(np.sum(-0.5 * np.log(z) - u * u * bb * bb / (2.0 * z)))

    U = 1.0
    while abs(phi(U)) > 1e-16 and U < 1e8:
        U *= 2.0
    f = lambda u: (1.0 - float(np.real(phi(u) * np.exp(-1j * u * kk)))) / (u * u) if u > 0 else 0.0
    val, err = integrate.quad(f, 0.0, U, limit=20000, epsabs=tol, epsrel=1e-11, points=[1e-3, 1e-2, 1e-1, 1.0])
    absdev = (2.0 / math.pi) * (val + 1.0 / U)
    mean = float(np.sum(la))
    return s * (0.5 * (mean - kk) + 0.5 * absdev), s * (2.0 / math.pi) * err


def scarf_excess(d, v):
    """Worst case of E(d + R)_+ - d_+ over mean-zero R with variance v (Scarf / Lo): (sqrt(v + d^2) - |d|)/2,
    at most sqrt(v)/2 (attained at d = 0) and at most v/(4|d|)."""
    d, v = np.asarray(d, float), np.maximum(np.asarray(v, float), 0.0)
    return 0.5 * v / (np.sqrt(v + d * d) + np.abs(d) + 1e-300)


def cantelli_quantile(x, v, alpha, conf_a=0.01, n_t=400, delta=0.0, beta=None, t_range=None):
    """Quantile bracket for Lambda = X + E + R with E[R | G] = 0, E[R^2 | G] <= V(G) per sample (x_k, v_k),
    and a G-measurable E with ||E||_2 <= delta.

    Upper: P(Lambda < t) >= E[g_t],  g_t = 1{x<t} (t-x)^2/(v+(t-x)^2)  (conditional Cantelli);
    lower: P(Lambda <= t) <= E[h_t], h_t = 1{x<=t} + 1{x>t} v/(v+(x-t)^2).
    Sample means are corrected by a Hoeffding bound with a union over the n_t points of a t-grid (both
    sides use the same grid; the test functions do not depend on beta, so one event covers every beta).
    With t_range given (chosen from exact moments, not from the samples) the bracket holds with probability
    >= 1 - conf_a; without it the grid is data-dependent and the bracket is only audited."""
    x, v = np.asarray(x, float), np.maximum(np.asarray(v, float), 0.0)
    K = x.size
    if t_range is not None:
        # fixed grid: probability curves once, then every beta (one Hoeffding event for all of them)
        grid = np.linspace(t_range[0], t_range[1], n_t)
        h = math.sqrt(math.log(2 * n_t / conf_a) / (2 * K))
        lowP, upP = np.empty(n_t), np.empty(n_t)
        for c0 in range(0, n_t, 100):
            tt = grid[c0:c0 + 100][:, None]
            d = tt - x[None, :]
            lowP[c0:c0 + 100] = np.mean(np.where(d > 0, d * d / (v[None, :] + d * d + 1e-300), 0.0), axis=1) - h
            upP[c0:c0 + 100] = np.mean(np.where(d < 0, v[None, :] / (v[None, :] + d * d + 1e-300), 1.0), axis=1) + h
        betas = [0.0] if delta <= 0 else [(1 - alpha) * c for c in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.2)]
        best_up, best_lo = math.inf, -math.inf
        for b in betas:
            shift = delta / math.sqrt(b) if b > 0 else 0.0
            iu = np.flatnonzero(lowP >= alpha + b)
            il = np.flatnonzero(upP < alpha - b)
            if iu.size:
                best_up = min(best_up, grid[iu[0]] + shift)
            if il.size:
                best_lo = max(best_lo, grid[il[-1]] - shift)
        return {"lower": best_lo, "upper": best_up, "center": float(np.quantile(x, alpha)), "hoeffding": h,
                "beta": "optimised", "grid": "fixed"}
    if delta > 0 and beta is None:                  # valid for every beta on the same event: optimise
        best = [cantelli_quantile(x, v, alpha, conf_a, n_t, delta, (1 - alpha) * c, t_range)
                for c in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.2)]
        return {"lower": max(b["lower"] for b in best), "upper": min(b["upper"] for b in best),
                "center": best[0]["center"], "hoeffding": best[0]["hoeffding"], "beta": "optimised",
                "grid": best[0]["grid"]}
    beta = 0.0 if delta <= 0 else beta
    shift = delta / math.sqrt(beta) if delta > 0 else 0.0
    h = math.sqrt(math.log(2 * n_t / conf_a) / (2 * K))
    a_up, a_lo = alpha + beta, alpha - beta
    qx = float(np.quantile(x, alpha))
    spread = float(np.quantile(x, 0.999) - np.quantile(x, 0.001)) + 10 * math.sqrt(float(np.mean(v)))

    def lower_prob_below(t):          # lower bound on P(Lambda < t)
        d = t - x
        g = np.where(d > 0, d * d / (v + d * d + 1e-300), 0.0)
        return float(np.mean(g)) - h

    def upper_prob_le(t):             # upper bound on P(Lambda <= t)
        d = x - t
        hh = np.where(d > 0, v / (v + d * d + 1e-300), 1.0)
        return float(np.mean(hh)) + h

    # upper endpoint: smallest grid t with P(Lambda < t) >= a_up guaranteed => q <= t
    ts = np.linspace(qx, qx + spread, n_t)
    up = next((t for t in ts if lower_prob_below(t) >= a_up), math.inf)
    # lower endpoint: largest grid t with P(Lambda <= t) < a_lo guaranteed => q > t
    ts = np.linspace(qx - spread, qx, n_t)
    lo = next((t for t in ts[::-1] if upper_prob_le(t) < a_lo), -math.inf)
    return {"lower": lo - shift, "upper": up + shift, "center": qx, "hoeffding": h, "beta": beta,
            "grid": "data-dependent (audited)"}


def lower_prob_below_fn(x, v, t):
    """Sample mean of the conditional-Cantelli lower bound on P(Lambda < t)."""
    d = t - x
    return float(np.mean(np.where(d > 0, d * d / (v + d * d + 1e-300), 0.0)))


def upper_prob_le_fn(x, v, t):
    """Sample mean of the conditional-Cantelli upper bound on P(Lambda <= t)."""
    d = x - t
    return float(np.mean(np.where(d > 0, v / (v + d * d + 1e-300), 1.0)))
