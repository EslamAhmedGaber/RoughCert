"""Cached exact ledgers over a grid of (n, k, N) and refinement strategies (Bridge D).

Surrogate:  Q_{k,N} Lambda_{nu_n}, nu_n = n-point midpoint rule of mu, Pi_k = projection on k nested
anchors, chaos level N.  Cost model (declared): C(n, k, N) = n * binom(k + N, N)  (coefficients
evaluated to assemble the surrogate).  Every candidate's error is the exact identity (Thm A3)
evaluated by audited quadrature; nothing below uses an error estimate in place of the identity.
"""
from __future__ import annotations

import math
from math import comb

import numpy as np

from .ledger import compute_ledger, _int_audit, KernelEvaluator
from .quadrature import Measure, rule_1d, rule_2d_full
from .representation import AnchorProjection, IdentityRepresentation

def _nested_dyadic(m=128):
    """Nested anchors on (0, 1]: 1, 1/2, then 3/4, 1/4, then odd multiples of 1/8 (descending), ..."""
    out, lev = [1.0, 0.5], 2
    while len(out) < m:
        den = 2 ** lev
        out += [j / den for j in range(den - 1, 0, -2)]
        lev += 1
    return out[:m]


NESTED = _nested_dyadic()


def cost(n, k, N):
    return n * comb(k + N, N)


def _sig(field, s):
    return np.sqrt(np.maximum(field.var(s), 0.0))


class LedgerGrid:
    def __init__(self, field, func, mu, ns, ks, Nmax, n_q=12, levels=8):
        self.field, self.func, self.mu = field, func, mu
        self.ns, self.ks, self.Nmax = list(ns), list(ks), Nmax
        T = field.T
        dens = float(mu.density(np.array([0.5 * (mu.lo + mu.hi)]))[0])
        self.nus = {n: Measure.riemann(n, T=T, rule="mid", lo=mu.lo, hi=mu.hi, weight=dens) for n in self.ns}
        bp_full = sorted(set(list(field.singular_points()) + [T]))
        ke = KernelEvaluator(field, IdentityRepresentation(field), func, None)
        self.A_mm, g1 = _int_audit(rule_2d_full, ke.full, mu, mu, bp_full, n_q, levels, True)
        self.X_mn, self.X_nn, self.gap = {}, {}, g1
        for n, nu in self.nus.items():
            v, g = _int_audit(rule_2d_full, ke.full, mu, nu, bp_full, n_q, levels, True)
            self.X_mn[n] = v
            t, wt = nu.atom_arrays()
            S, U = np.meshgrid(t, t, indexing="ij")
            self.X_nn[n] = float(wt @ ke.full(S, U) @ wt)
            self.gap = max(self.gap, g)
        # projected pieces per k (truncation levels 0..Nmax and untruncated)
        self.Bp_mm, self.BN_mm, self.BN_mn, self.BN_nn, self.Bp_nn, self.Bp_mn = {}, {}, {}, {}, {}, {}
        for k in self.ks:
            rep = AnchorProjection(field, [T * a for a in NESTED[:k]])
            bps = sorted(set(rep.breakpoints() + [T]))
            for lev, (nq, lv) in enumerate(((n_q, levels), (n_q + 4, levels + 3))):
                s, ws = rule_1d(mu, bps, nq, lv)
                gs, ss = rep.loadings(s), _sig(field, s)
                cs = func.chat(s, ss, Nmax)
                partial = np.zeros(Nmax + 1)
                bp = 0.0
                for i0 in range(0, s.size, 1024):          # row blocks keep memory bounded
                    sl = slice(i0, min(i0 + 1024, s.size))
                    X = gs[sl] @ gs.T
                    den = ss[sl][:, None] * ss[None, :]
                    r = np.where(den > 0, X / np.where(den > 0, den, 1.0), 0.0)
                    acc, rp = np.zeros_like(X), np.ones_like(X)
                    for m in range(Nmax + 1):
                        acc = acc + cs[m, sl][:, None] * cs[m][None, :] * rp
                        rp = rp * r
                        partial[m] += float(ws[sl] @ acc @ ws)
                    Kb = np.asarray(func.kernel(np.broadcast_to(s[sl][:, None], X.shape).ravel(),
                                                np.broadcast_to(s[None, :], X.shape).ravel(), X.ravel(),
                                                np.broadcast_to(ss[sl][:, None], X.shape).ravel(),
                                                np.broadcast_to(ss[None, :], X.shape).ravel())).reshape(X.shape)
                    bp += float(ws[sl] @ Kb @ ws)
                partial = list(partial)
                if lev == 0:
                    first = (partial, bp)
                else:
                    self.gap = max(self.gap, abs(bp - first[1]), max(abs(a - b) for a, b in zip(partial, first[0])))
                    self.BN_mm[k], self.Bp_mm[k] = partial, bp
                    s_fine, ws_fine, gs_fine, ss_fine, cs_fine = s, ws, gs, ss, cs
            for n, nu in self.nus.items():
                t, wt = nu.atom_arrays()
                gt, st = rep.loadings(t), _sig(field, t)
                ct = func.chat(t, st, Nmax)
                # mu x nu
                X = gs_fine @ gt.T
                den = ss_fine[:, None] * st[None, :]
                r = np.where(den > 0, X / np.where(den > 0, den, 1.0), 0.0)
                acc, rp, part_mn = np.zeros_like(X), np.ones_like(X), []
                for m in range(Nmax + 1):
                    acc = acc + cs_fine[m][:, None] * ct[m][None, :] * rp
                    rp = rp * r
                    part_mn.append(float(ws_fine @ acc @ wt))
                Kmn = np.asarray(func.kernel(np.broadcast_to(s_fine[:, None], X.shape).ravel(),
                                             np.broadcast_to(t[None, :], X.shape).ravel(), X.ravel(),
                                             np.broadcast_to(ss_fine[:, None], X.shape).ravel(),
                                             np.broadcast_to(st[None, :], X.shape).ravel())).reshape(X.shape)
                # nu x nu
                Xn = gt @ gt.T
                denn = st[:, None] * st[None, :]
                rn = np.where(denn > 0, Xn / np.where(denn > 0, denn, 1.0), 0.0)
                acc, rp, part_nn = np.zeros_like(Xn), np.ones_like(Xn), []
                for m in range(Nmax + 1):
                    acc = acc + ct[m][:, None] * ct[m][None, :] * rp
                    rp = rp * rn
                    part_nn.append(float(wt @ acc @ wt))
                Knn = np.asarray(func.kernel(np.broadcast_to(t[:, None], Xn.shape).ravel(),
                                             np.broadcast_to(t[None, :], Xn.shape).ravel(), Xn.ravel(),
                                             np.broadcast_to(st[:, None], Xn.shape).ravel(),
                                             np.broadcast_to(st[None, :], Xn.shape).ravel())).reshape(Xn.shape)
                self.BN_mn[(n, k)], self.BN_nn[(n, k)] = part_mn, part_nn
                self.Bp_mn[(n, k)], self.Bp_nn[(n, k)] = float(ws_fine @ Kmn @ wt), float(wt @ Knn @ wt)

    def ledger(self, n, k, N):
        A = self.A_mm
        BNmn, BNnn, BNmm = self.BN_mn[(n, k)][N], self.BN_nn[(n, k)][N], self.BN_mm[k][N]
        total = A - 2 * BNmn + BNnn
        M = A - 2 * self.X_mn[n] + self.X_nn[n]
        P_mu = A - self.Bp_mm[k]
        C_mu = self.Bp_mm[k] - BNmm
        M_Q = BNmm - 2 * BNmn + BNnn
        I = 2 * ((self.X_mn[n] - BNmn) - (self.X_nn[n] - BNnn))
        P_nu = self.X_nn[n] - self.Bp_nn[(n, k)]
        C_nu = self.Bp_nn[(n, k)] - BNnn
        return {"total": total, "M": M, "M_Q": M_Q, "P_mu": P_mu, "C_mu": C_mu, "P_nu": P_nu, "C_nu": C_nu,
                "I": I, "orth_check": M_Q + P_mu + C_mu - total, "seq_check": M + P_nu + C_nu + I - total}

    # single-axis "stage budgets" (each exact on its own axis, combined additively)
    def stage_M(self, n):
        return self.A_mm - 2 * self.X_mn[n] + self.X_nn[n]

    def stage_P(self, k):
        return self.A_mm - self.Bp_mm[k]

    def stage_C(self, k, N):
        return self.Bp_mm[k] - self.BN_mm[k][N]


class LazyLedger:
    """Ledger pieces of a (n, k, N) lattice computed on demand (same quadrature as LedgerGrid).

    Pythagoras (Q = J_{<=N} Gamma(Pi_k) is an orthogonal projection) gives, for every candidate,
        P_mu(k) + C_mu(k, N)  <=  ||Lambda_mu - Q Lambda_nu_n||^2  =  M_Q + P_mu + C_mu
                              <=  M(n) + P_mu(k) + C_mu(k, N),
    because M_Q = ||Q(Lambda_mu - Lambda_nu)||^2 <= M(n).  The lower bound needs no monitoring term, the
    upper bound only one-axis terms; the cross terms of (n, k) are computed only when the bounds disagree.
    """

    def __init__(self, field, func, mu, Nmax, n_q=12, levels=8, full_projected=False):
        self.field, self.func, self.mu, self.Nmax = field, func, mu, Nmax
        self.n_q, self.levels = n_q, levels
        # the search needs only truncated projected terms: P_mu + C_mu(k, N) = A - B_N(k); the untruncated
        # projected kernel (for reporting P_mu alone) is evaluated only on request
        self.full_projected = full_projected
        self.T = field.T
        self.dens = float(mu.density(np.array([0.5 * (mu.lo + mu.hi)]))[0])
        self.bp_full = sorted(set(list(field.singular_points()) + [self.T]))
        self.ke = KernelEvaluator(field, IdentityRepresentation(field), func, None)
        self.A_mm, self.gap = _int_audit(rule_2d_full, self.ke.full, mu, mu, self.bp_full, n_q, levels, True)
        self._M, self._proj, self._cross = {}, {}, {}
        self.counts = {"M": 0, "P": 0, "cross": 0}

    def nu(self, n):
        return Measure.riemann(n, T=self.T, rule="mid", lo=self.mu.lo, hi=self.mu.hi, weight=self.dens)

    def M(self, n):
        if n not in self._M:
            nu = self.nu(n)
            X_mn, g = _int_audit(rule_2d_full, self.ke.full, self.mu, nu, self.bp_full, self.n_q, self.levels, True)
            t, wt = nu.atom_arrays()
            S, U = np.meshgrid(t, t, indexing="ij")
            X_nn = float(wt @ self.ke.full(S, U) @ wt)
            self._M[n] = (self.A_mm - 2 * X_mn + X_nn, X_mn, X_nn)
            self.gap = max(self.gap, g)
            self.counts["M"] += 1
        return self._M[n][0]

    def _projected(self, k):
        if k not in self._proj:
            field, func, mu, Nmax = self.field, self.func, self.mu, self.Nmax
            rep = AnchorProjection(field, [self.T * a for a in NESTED[:k]])
            bps = sorted(set(rep.breakpoints() + [self.T]))
            vals = []
            for nq, lv in ((self.n_q, self.levels), (self.n_q + 4, self.levels + 3)):
                s, ws = rule_1d(mu, bps, nq, lv)
                gs, ss = rep.loadings(s), _sig(field, s)
                cs = func.chat(s, ss, Nmax)
                partial, bp = np.zeros(Nmax + 1), 0.0
                for i0 in range(0, s.size, 1024):
                    sl = slice(i0, min(i0 + 1024, s.size))
                    X = gs[sl] @ gs.T
                    den = ss[sl][:, None] * ss[None, :]
                    r = np.where(den > 0, X / np.where(den > 0, den, 1.0), 0.0)
                    acc, rp = np.zeros_like(X), np.ones_like(X)
                    for m in range(Nmax + 1):
                        acc = acc + cs[m, sl][:, None] * cs[m][None, :] * rp
                        rp = rp * r
                        partial[m] += float(ws[sl] @ acc @ ws)
                    if self.full_projected:
                        Kb = np.asarray(func.kernel(np.broadcast_to(s[sl][:, None], X.shape).ravel(),
                                                    np.broadcast_to(s[None, :], X.shape).ravel(), X.ravel(),
                                                    np.broadcast_to(ss[sl][:, None], X.shape).ravel(),
                                                    np.broadcast_to(ss[None, :], X.shape).ravel())).reshape(X.shape)
                        bp += float(ws[sl] @ Kb @ ws)
                    else:
                        bp = math.nan
                vals.append((partial, bp, (rep, s, ws, gs, ss, cs)))
            (p0, b0, _), (p1, b1, fine) = vals
            self.gap = max(self.gap, float(np.max(np.abs(p1 - p0))),
                           abs(b1 - b0) if self.full_projected else 0.0)
            self._proj[k] = (p1, b1, fine)
            self.counts["P"] += 1
        return self._proj[k]

    def lower(self, k, N):
        """P_mu(k) + C_mu(k, N) = A - B_N(k): needs truncated projected terms only."""
        return self.A_mm - self._projected(k)[0][N]

    def P(self, k):
        if not self.full_projected:
            raise ValueError("P_mu alone needs full_projected=True")
        return self.A_mm - self._projected(k)[1]

    def C(self, k, N):
        p, b, _ = self._projected(k)
        return b - p[N]

    def total(self, n, k, N):
        if (n, k) not in self._cross:
            p, b, (rep, s, ws, gs, ss, cs) = self._projected(k)
            t, wt = self.nu(n).atom_arrays()
            gt, st = rep.loadings(t), _sig(self.field, t)
            ct = self.func.chat(t, st, self.Nmax)
            X = gs @ gt.T
            den = ss[:, None] * st[None, :]
            r = np.where(den > 0, X / np.where(den > 0, den, 1.0), 0.0)
            acc, rp, part_mn = np.zeros_like(X), np.ones_like(X), []
            for m in range(self.Nmax + 1):
                acc = acc + cs[m][:, None] * ct[m][None, :] * rp
                rp = rp * r
                part_mn.append(float(ws @ acc @ wt))
            Xn = gt @ gt.T
            denn = st[:, None] * st[None, :]
            rn = np.where(denn > 0, Xn / np.where(denn > 0, denn, 1.0), 0.0)
            acc, rp, part_nn = np.zeros_like(Xn), np.ones_like(Xn), []
            for m in range(self.Nmax + 1):
                acc = acc + ct[m][:, None] * ct[m][None, :] * rp
                rp = rp * rn
                part_nn.append(float(wt @ acc @ wt))
            self._cross[(n, k)] = (part_mn, part_nn)
            self.counts["cross"] += 1
        part_mn, part_nn = self._cross[(n, k)]
        return self.A_mm - 2 * part_mn[N] + part_nn[N]


def best_first(L, ns, ks, Nmax, tol2):
    """Cheapest lattice candidate with certified error <= tol2, by enumeration in increasing cost.

    Each candidate is decided by the n-free lower bound P + C (prune), the separable upper bound
    M + P + C (accept without cross terms), or else the exact total.  The first accepted candidate is the
    lattice optimum (every cheaper one was excluded exactly or by a valid lower bound)."""
    cands = sorted((cost(n, k, N), n, k, N) for n in ns for k in ks for N in range(1, Nmax + 1))
    st = {"pruned_lower": 0, "accepted_upper": 0, "exact": 0, "visited": 0}
    for c, n, k, N in cands:
        st["visited"] += 1
        lb = L.lower(k, N)
        if lb - L.gap > tol2:
            st["pruned_lower"] += 1
            continue
        ub = L.M(n) + lb
        if ub + L.gap <= tol2:
            st["accepted_upper"] += 1
            return {"cost": c, "n": n, "k": k, "N": N, "bound": ub, "how": "upper bound", **st,
                    "counts": dict(L.counts)}
        e = L.total(n, k, N)
        st["exact"] += 1
        if e + L.gap <= tol2:
            return {"cost": c, "n": n, "k": k, "N": N, "bound": e, "how": "exact", **st, "counts": dict(L.counts)}
    return {"cost": None, **st, "counts": dict(L.counts)}


def strategies(grid, tol2):
    ns, ks, Nmax = grid.ns, grid.ks, grid.Nmax
    Ns = list(range(1, Nmax + 1))
    out = {}
    # oracle
    best = None
    for n in ns:
        for k in ks:
            for N in Ns:
                e = grid.ledger(n, k, N)["total"]
                if e + grid.gap <= tol2:
                    c = cost(n, k, N)
                    if best is None or c < best[0]:
                        best = (c, n, k, N, e)
    out["oracle"] = best

    def step_up(n, k, N):
        cand = []
        i, j = ns.index(n), ks.index(k)
        if i + 1 < len(ns):
            cand.append((ns[i + 1], k, N))
        if j + 1 < len(ks):
            cand.append((n, ks[j + 1], N))
        if N + 1 <= Nmax:
            cand.append((n, k, N + 1))
        return cand

    # greedy on exact ledger
    n, k, N = ns[0], ks[0], 1
    e = grid.ledger(n, k, N)["total"]
    evals = 1
    while e + grid.gap > tol2:
        cand = step_up(n, k, N)
        if not cand:
            break
        scored = []
        for (a, b, c) in cand:
            ee = grid.ledger(a, b, c)["total"]
            evals += 1
            scored.append(((e - ee) / max(cost(a, b, c) - cost(n, k, N), 1e-12), a, b, c, ee))
        _, n, k, N, e = max(scored)
    out["greedy_exact"] = (cost(n, k, N), n, k, N, e, evals, e + grid.gap <= tol2)

    # greedy on separate stage budgets (ignores interaction and the M -> M_Q change)
    n, k, N = ns[0], ks[0], 1
    sep = lambda n, k, N: grid.stage_M(n) + grid.stage_P(k) + grid.stage_C(k, N)
    e = sep(n, k, N)
    while e > tol2:
        cand = step_up(n, k, N)
        if not cand:
            break
        scored = [((e - sep(a, b, c)) / max(cost(a, b, c) - cost(n, k, N), 1e-12), a, b, c, sep(a, b, c))
                  for (a, b, c) in cand]
        _, n, k, N, e = max(scored)
    true_e = grid.ledger(n, k, N)["total"]
    out["greedy_separate"] = (cost(n, k, N), n, k, N, true_e, e, true_e + grid.gap <= tol2)

    # uniform refinement
    i = 0
    n, k, N = ns[0], ks[0], 1
    while True:
        e = grid.ledger(n, k, N)["total"]
        if e + grid.gap <= tol2:
            break
        i += 1
        if i >= max(len(ns), len(ks), Nmax):
            break
        n = ns[min(i, len(ns) - 1)]
        k = ks[min(i, len(ks) - 1)]
        N = min(1 + i, Nmax)
    out["uniform"] = (cost(n, k, N), n, k, N, e, e + grid.gap <= tol2)
    return out
