"""Separable evaluation of projected-kernel integrals.

For a finite-rank representation, R_Pi(s, u) = g(s) . g(u).  Integrals
    intint Phi(s, u; R_Pi(s, u)) a(ds) b(du)
are evaluated on a tensor rule built from 1-D graded rules, computing the loadings and
Hermite data on the 1-D nodes only and forming the kernel matrix in row blocks.
"""
from __future__ import annotations

import numpy as np

from .quadrature import rule_1d


def _sigma(field, s):
    return np.sqrt(np.maximum(field.var(s), 0.0))


class TensorProjected:
    """Projected-kernel integrals for one pair of measures and one rule level."""

    def __init__(self, field, rep, func, a, b, bps, n, levels, N, block=2048):
        self.field, self.rep, self.func, self.N = field, rep, func, N
        self.s, self.ws = rule_1d(a, bps, n, levels)
        self.u, self.wu = rule_1d(b, bps, n, levels)
        self.gs, self.gu = rep.loadings(self.s), rep.loadings(self.u)
        self.ss, self.su = _sigma(field, self.s), _sigma(field, self.u)
        self.block = block
        if N is not None:
            self.cs = func.chat(self.s, self.ss, N)
            self.cu = func.chat(self.u, self.su, N)

    def _blocks(self):
        for i0 in range(0, self.s.size, self.block):
            sl = slice(i0, min(i0 + self.block, self.s.size))
            X = self.gs[sl] @ self.gu.T
            yield sl, X

    def proj(self):
        """intint K(s,u; R_Pi) (untruncated chaos)."""
        tot = 0.0
        for sl, X in self._blocks():
            S = np.broadcast_to(self.s[sl][:, None], X.shape)
            U = np.broadcast_to(self.u[None, :], X.shape)
            Sg = np.broadcast_to(self.ss[sl][:, None], X.shape)
            Ug = np.broadcast_to(self.su[None, :], X.shape)
            K = np.asarray(self.func.kernel(S.ravel(), U.ravel(), X.ravel(), Sg.ravel(), Ug.ravel())).reshape(X.shape)
            tot += float(self.ws[sl] @ K @ self.wu)
        return tot

    def proj_trunc(self):
        """intint sum_{m<=N} chat_m(s) chat_m(u) r^m."""
        if self.N is None:
            return self.proj()
        tot = 0.0
        for sl, X in self._blocks():
            den = self.ss[sl][:, None] * self.su[None, :]
            r = np.where(den > 0, X / np.where(den > 0, den, 1.0), 0.0)
            acc = np.zeros_like(X)
            rp = np.ones_like(X)
            for m in range(self.N + 1):
                acc += self.cs[m, sl][:, None] * self.cu[m][None, :] * rp
                rp = rp * r
            tot += float(self.ws[sl] @ acc @ self.wu)
        return tot


def projected_integrals(field, rep, func, a, b, bps, n, levels, N, audit=True, audit_dn=4, audit_dl=3):
    """Return dict with 'proj', 'trunc' (and 'gap') for intint over a x b."""
    t1 = TensorProjected(field, rep, func, a, b, bps, n, levels, N)
    p1 = t1.proj()
    q1 = t1.proj_trunc() if N is not None else p1
    if not audit:
        return {"proj": p1, "trunc": q1, "gap": 0.0}
    t2 = TensorProjected(field, rep, func, a, b, bps, n + audit_dn, levels + audit_dl, N)
    p2 = t2.proj()
    q2 = t2.proj_trunc() if N is not None else p2
    return {"proj": p2, "trunc": q2, "gap": max(abs(p2 - p1), abs(q2 - q1))}
