"""Exact joint error ledger for measure-indexed subordinated functionals.

Target    Lambda_mu      = int f(s, Y_s) mu(ds)
Surrogate Q Lambda_nu    = int J_{<=N} Gamma(Pi) f(s, Y_s) nu(ds),   Q = J_{<=N} Gamma(Pi)

General master identity (extends Paper III Thm 1 and Paper V Thm 4.1 from Wick
exponentials/jets to every subordinated field):
  ||Lambda_mu - Q Lambda_nu||^2 = intint K(R) dmu dmu - 2 intint K_N(R_Pi) dmu dnu
                                  + intint K_N(R_Pi) dnu dnu,
with K(s,u;x) = sum_m chat_m(s) chat_m(u) (x/(sig_s sig_u))^m and K_N its truncation.
Both reconciled ledgers (sequential with signed interaction I, orthogonal with
nonnegative terms) follow from Q being an orthogonal projection.

All integrals are numerical (graded Gauss--Legendre) and are audited by a finer
rule; the reported ``quad_gap`` is the change under refinement, not an enclosure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field as dfield

import numpy as np

from .quadrature import rule_2d_full, rule_2d_tensor, Measure


def _sigma(field, s):
    return np.sqrt(np.maximum(field.var(s), 0.0))


class KernelEvaluator:
    """Evaluates K(R), K(R_Pi) and K_N(R_Pi) on node arrays, caching Hermite data."""

    def __init__(self, field, rep, func, N=None):
        self.field, self.rep, self.func, self.N = field, rep, func, N
        self._chat = {}

    def full(self, S, U):
        X = self.field.R(S, U)
        return self.func.kernel(S, U, X, _sigma(self.field, S), _sigma(self.field, U))

    def proj(self, S, U):
        X = self.rep.RPi_pairs(S, U)
        return self.func.kernel(S, U, X, _sigma(self.field, S), _sigma(self.field, U))

    def chat_at(self, S):
        S = np.asarray(S, float)
        uniq, inv = np.unique(S.ravel(), return_inverse=True)
        need = [x for x in uniq if x not in self._chat]
        if need:
            need = np.asarray(need)
            C = self.func.chat(need, _sigma(self.field, need), self.N)
            for j, x in enumerate(need):
                self._chat[float(x)] = C[:, j]
        return np.stack([self._chat[float(x)] for x in uniq], axis=1)[:, inv]

    def proj_trunc(self, S, U):
        if self.N is None:
            return self.proj(S, U)
        X = self.rep.RPi_pairs(S, U).ravel()
        cs, cu = self.chat_at(S), self.chat_at(U)
        den = (_sigma(self.field, S) * _sigma(self.field, U)).ravel()
        r = np.where(den > 0, X / np.where(den > 0, den, 1.0), 0.0)
        pw = r[None, :] ** np.arange(self.N + 1)[:, None]
        return np.sum(cs * cu * pw, axis=0).reshape(np.shape(S))


@dataclass
class LedgerResult:
    total: float
    M: float
    M_Q: float
    P_mu: float
    C_mu: float
    P_nu: float
    C_nu: float
    I: float
    sequential_check: float
    orthogonal_check: float
    quad_gap: float
    second_moment_target: float
    N: object
    rank: object
    notes: list = dfield(default_factory=list)

    def as_dict(self):
        return asdict(self)

    @property
    def rms(self):
        return math.sqrt(max(self.total, 0.0))


def _int(rule, phi, mu, nu, bps, n, levels):
    S, U, W = rule(mu, nu, bps, n, levels)
    return float(np.sum(W * phi(S, U))) if W.size else 0.0


def _int_audit(rule, phi, mu, nu, bps, n, levels, audit, audit_n=None, audit_levels=None):
    v = _int(rule, phi, mu, nu, bps, n, levels)
    if not audit:
        return v, 0.0
    v2 = _int(rule, phi, mu, nu, bps, audit_n or n + 4, audit_levels or levels + 3)
    return v2, abs(v2 - v)


def compute_ledger(field, rep, func, mu, nu, N=None, n=12, levels=8, audit=True):
    """Both reconciled ledgers of ||Lambda_mu - Q Lambda_nu||^2 (Q = J_{<=N} Gamma(Pi)).

    Quadrature routing: terms in the full kernel K(R) use the diagonal-aware rule (cusp
    |s-u|^{2H}).  Terms in K(R_Pi) use, for finite-rank backends, separable tensor rules graded
    at the backend's breakpoints (R_Pi(s,u) = g(s).g(u) is smooth across the diagonal); for the
    identity backend R_Pi = R and the diagonal rule is used.  Each integral is evaluated with two
    rules; the finer value is reported and quad_gap is the largest change (not an enclosure).
    """
    from .representation import IdentityRepresentation
    from ._tensor import projected_integrals
    ke = KernelEvaluator(field, rep, func, N)
    bp_full = sorted(set(list(field.singular_points()) + [field.T]))
    identity = isinstance(rep, IdentityRepresentation)
    gaps = []

    def run(phi, a, b, bps):
        v, g = _int_audit(rule_2d_full, phi, a, b, bps, n, levels, audit)
        gaps.append(g)
        return v

    full = lambda S, U: ke.full(S, U)
    A_mm = run(full, mu, mu, bp_full)
    X_mn = run(full, mu, nu, bp_full)
    X_nn = run(full, nu, nu, bp_full)
    if identity:
        projN = lambda S, U: ke.proj_trunc(S, U)
        proj = lambda S, U: ke.proj(S, U)
        Bp_mm, Bp_nn = run(proj, mu, mu, bp_full), run(proj, nu, nu, bp_full)
        if N is None:
            BN_mm, BN_nn, BN_mn = Bp_mm, Bp_nn, run(proj, mu, nu, bp_full)
        else:
            BN_mm, BN_nn, BN_mn = (run(projN, mu, mu, bp_full), run(projN, nu, nu, bp_full),
                                   run(projN, mu, nu, bp_full))
    else:
        bp_p = sorted(set(rep.breakpoints() + [field.T]))
        r_mm = projected_integrals(field, rep, func, mu, mu, bp_p, n, levels, N, audit)
        r_nn = projected_integrals(field, rep, func, nu, nu, bp_p, n, levels, N, audit)
        r_mn = projected_integrals(field, rep, func, mu, nu, bp_p, n, levels, N, audit)
        gaps += [r_mm["gap"], r_nn["gap"], r_mn["gap"]]
        Bp_mm, Bp_nn = r_mm["proj"], r_nn["proj"]
        BN_mm, BN_nn, BN_mn = r_mm["trunc"], r_nn["trunc"], r_mn["trunc"]
    C_mu = Bp_mm - BN_mm
    C_nu = Bp_nn - BN_nn
    P_mu = A_mm - Bp_mm
    P_nu = X_nn - Bp_nn
    M = A_mm - 2.0 * X_mn + X_nn
    M_Q = BN_mm - 2.0 * BN_mn + BN_nn
    I = 2.0 * ((X_mn - BN_mn) - (X_nn - BN_nn))
    total = A_mm - 2.0 * BN_mn + BN_nn
    seq = M + P_nu + C_nu + I
    orth = M_Q + P_mu + C_mu
    notes = []
    tolneg = 10 * (max(gaps) if gaps else 0.0) + 1e-13 * max(abs(A_mm), 1e-300)
    if min(P_mu, C_mu, M_Q) < -tolneg:
        notes.append("negative orthogonal-ledger component beyond audit: check representation/quadrature")
    return LedgerResult(total=total, M=M, M_Q=M_Q, P_mu=P_mu, C_mu=C_mu, P_nu=P_nu, C_nu=C_nu, I=I,
                        sequential_check=seq - total, orthogonal_check=orth - total,
                        quad_gap=float(max(gaps) if gaps else 0.0), second_moment_target=A_mm,
                        N=N, rank=getattr(rep, "rank", None), notes=notes)


def first_moment(field, func, mu, n=12, levels=8):
    """E[Lambda_mu] = int chat_0(s) dmu(s) (exact identity, numerical quadrature)."""
    from .quadrature import rule_1d
    s, w = rule_1d(mu, list(field.singular_points()) + [field.T], n, levels)
    c0 = func.chat(s, _sigma(field, s), 0)[0]
    return float(np.sum(w * c0))


def second_moment(field, func, mu, n=12, levels=8, audit=True):
    """E[Lambda_mu^2] by the full kernel (exact identity, audited quadrature)."""
    ke = KernelEvaluator(field, None, func, None)
    return _int_audit(rule_2d_full, ke.full, mu, mu, list(field.singular_points()) + [field.T],
                      n, levels, audit)


def covariance_matrix(field, funcs_measures, n=16, levels=10):
    """Joint covariance of several outputs (f_i, mu_i) from one field (reuse of R)."""
    k = len(funcs_measures)
    means = np.array([first_moment(field, f, m, n, levels) for f, m in funcs_measures])
    C = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            fi, mi = funcs_measures[i]
            fj, mj = funcs_measures[j]
            phi = _cross_kernel(field, fi, fj)
            S, U, W = rule_2d_full(mi, mj, list(field.singular_points()) + [field.T], n, levels)
            C[i, j] = C[j, i] = float(np.sum(W * phi(S, U))) - means[i] * means[j]
    return means, C


def _cross_kernel(field, fi, fj, M=80):
    """E[f_i(s, Y_s) f_j(u, Y_u)] via Hermite series (both families) -- generic but series-based."""
    if fi is fj:
        return lambda S, U: fi.kernel(S, U, field.R(S, U), _sigma(field, S), _sigma(field, U))

    def phi(S, U):
        cs = fi.chat(S.ravel(), _sigma(field, S).ravel(), M)
        cu = fj.chat(U.ravel(), _sigma(field, U).ravel(), M)
        den = (_sigma(field, S) * _sigma(field, U)).ravel()
        r = np.where(den > 0, field.R(S, U).ravel() / np.where(den > 0, den, 1.0), 0.0)
        return np.sum(cs * cu * r[None, :] ** np.arange(M + 1)[:, None], axis=0).reshape(S.shape)
    return phi
