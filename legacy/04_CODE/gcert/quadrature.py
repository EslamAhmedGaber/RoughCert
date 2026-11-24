"""Quadrature for measure integrals on [0, T] with power-type singularities.

Two kinds of non-smoothness occur in the ledger integrands:
  * endpoint/breakpoint singularities |s - b|^{2H} (from R(s, b), anchors, s = 0);
  * the diagonal cusp |s - u|^{2H} of R(s, u) in the full-kernel double integral.
Breakpoint singularities are handled by geometrically graded Gauss--Legendre panels
(hp-quadrature); the diagonal is handled by integrating over the triangle u > s with
the substitution u = s + (b - s) v and grading in v.  Accuracy is audited by
repeating every integral with a finer rule (``refine``) -- these are numerical
evaluations, not interval enclosures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=64)
def _gl01(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def gl(a, b, n):
    x, w = _gl01(int(n))
    return a + (b - a) * x, (b - a) * w


def graded_panel(a, b, n=16, levels=10, sigma=0.15, left=True, right=True):
    """hp rule on [a, b], graded geometrically toward singular endpoints."""
    if b <= a:
        return np.zeros(0), np.zeros(0)
    cuts = [0.0, 1.0]
    if left and right:
        mid = 0.5
        lc = [mid * sigma ** k for k in range(1, levels + 1)]
        rc = [1.0 - mid * sigma ** k for k in range(1, levels + 1)]
        cuts = sorted(set([0.0, mid, 1.0] + lc + rc))
    elif left:
        cuts = sorted(set([0.0, 1.0] + [sigma ** k for k in range(1, levels + 1)]))
    elif right:
        cuts = sorted(set([0.0, 1.0] + [1.0 - sigma ** k for k in range(1, levels + 1)]))
    xs, ws = [], []
    for c0, c1 in zip(cuts[:-1], cuts[1:]):
        x, w = gl(a + (b - a) * c0, a + (b - a) * c1, n)
        xs.append(x)
        ws.append(w)
    return np.concatenate(xs), np.concatenate(ws)


@dataclass
class Measure:
    """Finite signed measure on [0, T]: a density part g(s) ds on [lo, hi] plus atoms.

    ``density`` is a vectorised callable or None; ``density_breaks`` lists interior
    points where the density is not smooth.  ``atoms`` is a list of (t, w).
    """

    T: float = 1.0
    density: object = None
    lo: float = 0.0
    hi: float = None
    density_breaks: list = field(default_factory=list)
    atoms: list = field(default_factory=list)
    name: str = "measure"

    def __post_init__(self):
        if self.hi is None:
            self.hi = self.T

    @staticmethod
    def lebesgue(T=1.0, weight=1.0, lo=0.0, hi=None, name="Lebesgue"):
        c = float(weight)
        return Measure(T=T, density=lambda s: c * np.ones_like(np.asarray(s, float)), lo=lo,
                       hi=T if hi is None else hi, name=name)

    @staticmethod
    def atomic(times, weights, T=1.0, name="atomic"):
        return Measure(T=T, atoms=list(zip(map(float, times), map(float, weights))), name=name)

    @staticmethod
    def riemann(n, T=1.0, rule="left", weight=1.0, lo=0.0, hi=None, name=None):
        """Cell-matched n-point discretisation of weight * Lebesgue on [lo, hi]."""
        hi = T if hi is None else hi
        h = (hi - lo) / n
        if rule == "left":
            t = lo + h * np.arange(n)
        elif rule == "right":
            t = lo + h * np.arange(1, n + 1)
        elif rule == "mid":
            t = lo + h * (np.arange(n) + 0.5)
        else:
            raise ValueError(rule)
        return Measure.atomic(t, np.full(n, weight * h), T=T, name=name or f"{rule}-{n}")

    @property
    def has_density(self):
        return self.density is not None and self.hi > self.lo

    def atom_arrays(self):
        if not self.atoms:
            return np.zeros(0), np.zeros(0)
        t, w = zip(*self.atoms)
        return np.asarray(t, float), np.asarray(w, float)

    def total_variation(self, n=400):
        tv = float(np.sum(np.abs(self.atom_arrays()[1])))
        if self.has_density:
            x, w = gl(self.lo, self.hi, n)
            tv += float(np.sum(w * np.abs(self.density(x))))
        return tv

    def minus(self, other, name=None):
        """Signed difference as a list of components (density parts kept separate)."""
        return SignedCombination([(1.0, self), (-1.0, other)], name=name)


@dataclass
class SignedCombination:
    parts: list
    name: str = None


def _breaks_in(lo, hi, pts):
    return sorted(set([lo, hi] + [p for p in pts if lo < p < hi]))


def rule_1d(measure, breakpoints=(), n=16, levels=10):
    """Nodes/weights integrating phi(s) d measure(s); density part graded at breakpoints."""
    xs, ws = [], []
    if measure.has_density:
        cuts = _breaks_in(measure.lo, measure.hi, list(breakpoints) + list(measure.density_breaks))
        for a, b in zip(cuts[:-1], cuts[1:]):
            x, w = graded_panel(a, b, n=n, levels=levels)
            xs.append(x)
            ws.append(w * measure.density(x))
    t, wa = measure.atom_arrays()
    if t.size:
        xs.append(t)
        ws.append(wa)
    if not xs:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(xs), np.concatenate(ws)


def rule_2d_tensor(mu, nu, breakpoints=(), n=16, levels=10):
    """Tensor rule for integrands smooth across the diagonal (e.g. functions of R_Pi)."""
    s, ws = rule_1d(mu, breakpoints, n, levels)
    u, wu = rule_1d(nu, breakpoints, n, levels)
    S, U = np.meshgrid(s, u, indexing="ij")
    W = np.outer(ws, wu)
    return S.ravel(), U.ravel(), W.ravel()


def _diag_triangle(a, b, n, levels):
    """Nodes for int_a^b ds int_s^b du phi(s,u), graded at s=a,b and at u=s."""
    s, ws = graded_panel(a, b, n=n, levels=levels)
    v, wv = graded_panel(0.0, 1.0, n=n, levels=levels, left=True, right=True)
    S = np.repeat(s, v.size)
    L = np.repeat(b - s, v.size)
    V = np.tile(v, s.size)
    U = S + L * V
    W = np.repeat(ws, v.size) * L * np.tile(wv, s.size)
    return S, U, W


def rule_2d_full(mu, nu, breakpoints=(), n=16, levels=10):
    """Rule for int int phi(s,u) dmu(s) dnu(u) when phi has a diagonal cusp.

    Density x density blocks on the same panel use the triangle rule (both orders);
    different panels use tensor products of graded panels.  Atom blocks are 1-D.
    Returns (S, U, W) such that sum W phi(S, U) approximates the integral.
    """
    S_all, U_all, W_all = [], [], []
    if mu.has_density and nu.has_density:
        cuts_mu = _breaks_in(mu.lo, mu.hi, list(breakpoints) + list(mu.density_breaks) + [nu.lo, nu.hi]
                             + list(nu.density_breaks))
        cuts_nu = _breaks_in(nu.lo, nu.hi, list(breakpoints) + list(nu.density_breaks) + [mu.lo, mu.hi]
                             + list(mu.density_breaks))
        panels_mu = list(zip(cuts_mu[:-1], cuts_mu[1:]))
        panels_nu = list(zip(cuts_nu[:-1], cuts_nu[1:]))
        for (a, b) in panels_mu:
            for (c, d) in panels_nu:
                if abs(a - c) < 1e-15 and abs(b - d) < 1e-15:
                    S, U, W = _diag_triangle(a, b, n, levels)
                    # u > s triangle and its mirror u < s
                    for SS, UU in ((S, U), (U, S)):
                        S_all.append(SS)
                        U_all.append(UU)
                        W_all.append(W * mu.density(SS) * nu.density(UU))
                else:
                    s, ws = graded_panel(a, b, n=n, levels=levels)
                    u, wu = graded_panel(c, d, n=n, levels=levels)
                    SS, UU = np.meshgrid(s, u, indexing="ij")
                    WW = np.outer(ws * mu.density(s), wu * nu.density(u))
                    S_all.append(SS.ravel())
                    U_all.append(UU.ravel())
                    W_all.append(WW.ravel())
    # density x atom and atom x density
    tn, wn = nu.atom_arrays()
    if mu.has_density and tn.size:
        for t, w in zip(tn, wn):
            s, ws = rule_1d(Measure(T=mu.T, density=mu.density, lo=mu.lo, hi=mu.hi,
                                    density_breaks=mu.density_breaks), list(breakpoints) + [t], n, levels)
            S_all.append(s)
            U_all.append(np.full_like(s, t))
            W_all.append(ws * w)
    tm, wm = mu.atom_arrays()
    if nu.has_density and tm.size:
        for t, w in zip(tm, wm):
            u, wu = rule_1d(Measure(T=nu.T, density=nu.density, lo=nu.lo, hi=nu.hi,
                                    density_breaks=nu.density_breaks), list(breakpoints) + [t], n, levels)
            S_all.append(np.full_like(u, t))
            U_all.append(u)
            W_all.append(wu * w)
    if tm.size and tn.size:
        SS, UU = np.meshgrid(tm, tn, indexing="ij")
        S_all.append(SS.ravel())
        U_all.append(UU.ravel())
        W_all.append(np.outer(wm, wn).ravel())
    if not S_all:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    return np.concatenate(S_all), np.concatenate(U_all), np.concatenate(W_all)


def integrate_2d(phi, mu, nu, full=True, breakpoints=(), n=16, levels=10, audit=True):
    """Return (value, audit_gap).  ``phi(S, U)`` vectorised."""
    rule = rule_2d_full if full else rule_2d_tensor
    S, U, W = rule(mu, nu, breakpoints, n, levels)
    val = float(np.sum(W * phi(S, U))) if W.size else 0.0
    if not audit:
        return val, None
    S2, U2, W2 = rule(mu, nu, breakpoints, n + 8, levels + 3)
    val2 = float(np.sum(W2 * phi(S2, U2))) if W2.size else 0.0
    return val2, abs(val2 - val)
