"""One-dimensional surrogate laws and output-adapted projections.

For a rank-one projection Pi (W(Pi e_s) = g(s) z, z ~ N(0,1)) the conditional expectation
of the integrated output is a deterministic map of one Gaussian variable:
    Y = Gamma(Pi) Lambda_nu = G(z),   G(z) = int (S_{D(s,s)} f_s)(g(s) z) nu(ds),
where S_v is Gaussian smoothing by the residual variance v (the same operation as the
Paper IX residual closure, here applied to an output).  Its law is computed on a stratified
probability grid (transfer.law_from_map), and the exact ledger radius sqrt(P) makes the
sharp Wasserstein endpoints a two-sided quantile certificate.

OutputAdaptedProjection: projection onto span{h*}, h* = int c_1(s) e_s mu(ds), the direction
of the first Wiener chaos of the output itself.  It removes the first-chaos error exactly and is
an orthogonal projection, so every ledger statement applies.  (Related: gradient-based
dimension reduction, Zahm et al. 2020/2022; here the exact error is available.)
"""
from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import ndtr

from .functionals import (WickExponential, PathwiseExponential, Indicator, PositivePart, ExpCall, Polynomial,
                          npdf, _row_nodes, _cut_matrix, _pad)
from .quadrature import rule_1d, Measure, graded_panel
from .representation import Representation


def smoothed(func, s, y, D, field=None):
    """(S_D f_s)(y) = E[f(s, y + sqrt(D) Z)]; closed forms where available, else quadrature.

    s, y, D broadcast to a common shape.
    """
    s, y, D = np.asarray(s, float), np.asarray(y, float), np.maximum(np.asarray(D, float), 0.0)
    # Evaluate time-only functions on the node vector before broadcasting over samples.
    # In particular, RL variance evaluation otherwise repeats a hypergeometric series K times.
    if isinstance(func, WickExponential):
        var = func.field.var(s)
        return func.xi0 * np.exp(func.eta * y - 0.5 * func.eta ** 2 * (var - D))
    if isinstance(func, PathwiseExponential):
        return func.xi0 * np.exp(func.eta * y + func.drift(s) + 0.5 * func.eta ** 2 * D)
    s, y, D = np.broadcast_arrays(s, y, D)
    sq = np.sqrt(D)
    if isinstance(func, Indicator):
        d = y + func.mean(s) - func.c
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(sq > 0, ndtr(d / np.where(sq > 0, sq, 1.0)), (d > 0).astype(float))
        return func.scale * out
    if isinstance(func, PositivePart):
        d = y + func.mean(s) - func.c
        with np.errstate(divide="ignore", invalid="ignore"):
            k = d / np.where(sq > 0, sq, 1.0)
            out = np.where(sq > 0, d * ndtr(k) + sq * npdf(k), np.maximum(d, 0.0))
        return func.scale * out
    if isinstance(func, Polynomial):
        # E[(c + sqrt(D) Z)^k] = sum_{j even} C(k,j) c^(k-j) D^(j/2) (j-1)!!,  c = y + mean(s)
        c = y + np.broadcast_to(np.asarray(func.mean(s), float), y.shape)
        out = np.zeros(y.shape)
        for k, ak in enumerate(func.a):
            if ak == 0.0:
                continue
            for j in range(0, k + 1, 2):
                dfact = float(np.prod(np.arange(j - 1, 0, -2))) if j > 0 else 1.0
                out = out + ak * math.comb(k, j) * c ** (k - j) * D ** (j // 2) * dfact
        return out
    if isinstance(func, ExpCall):
        # E[(a e^{vZ} - B)_+] = a e^{v^2/2} Phi(d2 + v) - B Phi(d2),  a = A(s) e^{kappa y},  v = kappa sqrt(D)
        la = np.log(func.A(s)) + func.kappa * y
        v = func.kappa * sq
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            d2 = (la - math.log(func.B)) / np.where(v > 0, v, 1.0)
            out = np.where(v > 0, np.exp(la + 0.5 * v * v) * ndtr(d2 + v) - func.B * ndtr(d2),
                           np.maximum(np.exp(la) - func.B, 0.0))
        return func.scale * out
    # generic: composite quadrature with breakpoints
    flat_s, flat_y, flat_sq = s.ravel(), y.ravel(), sq.ravel()
    out = np.empty(flat_s.size)
    det = flat_sq <= 0
    if np.any(det):
        out[det] = func.f(flat_s[det], flat_y[det])
    idx = np.flatnonzero(~det)
    for c0 in range(0, idx.size, 8192):
        jj = idx[c0:c0 + 8192]
        ex = _pad([[(b - flat_y[j]) / flat_sq[j] for b in func.breakpoints_y(flat_s[j])] for j in jj])
        Z, W = _row_nodes(_cut_matrix(10.0, 0.5, ex), 10)
        V = flat_y[jj][:, None] + flat_sq[jj][:, None] * Z
        out[jj] = np.sum(W * npdf(Z) * func.f(np.broadcast_to(flat_s[jj][:, None], V.shape), V), axis=1)
    return out.reshape(s.shape)


class OneDimSurrogate:
    """G(z) for a rank-one representation; evaluated on a z-grid and spline-interpolated."""

    def __init__(self, field, rep, func, nu, n=16, levels=8, zmax=9.0, nz=1201):
        if rep.rank != 1:
            raise ValueError("OneDimSurrogate needs a rank-one representation")
        self.field, self.rep, self.func, self.nu = field, rep, func, nu
        bps = sorted(set(rep.breakpoints() + [field.T]))
        s, w = rule_1d(nu, bps, n, levels)
        self.s, self.w = s, w
        self.g = rep.loadings(s)[:, 0]
        var = field.var(s)
        self.D = np.maximum(var - self.g ** 2, 0.0)
        self.zgrid = np.linspace(-zmax, zmax, nz)
        Gv = self._direct(self.zgrid)
        self.spline = CubicSpline(self.zgrid, Gv)
        # interpolation audit on midpoints
        zm = 0.5 * (self.zgrid[1:] + self.zgrid[:-1])
        self.interp_gap = float(np.max(np.abs(self._direct(zm[::7]) - self.spline(zm[::7]))))
        self.zmax = zmax

    def _direct(self, z):
        z = np.atleast_1d(np.asarray(z, float))
        out = np.empty(z.size)
        for c0 in range(0, z.size, 64):
            zz = z[c0:c0 + 64]
            Y = self.g[None, :] * zz[:, None]
            vals = smoothed(self.func, np.broadcast_to(self.s, Y.shape), Y,
                            np.broadcast_to(self.D, Y.shape))
            out[c0:c0 + 64] = vals @ self.w
        return out

    def __call__(self, z):
        z = np.asarray(z, float)
        inside = np.abs(z) <= self.zmax
        out = np.empty(z.shape)
        out[inside] = self.spline(z[inside])
        if np.any(~inside):
            out[~inside] = self._direct(z[~inside])
        return out


class OutputAdaptedProjection(Representation):
    """Rank-one projection onto P_A h*, h* = int c_1(s) e_s mu(ds) (first chaos of the output).

    A is a finite anchor set; P_A h* = sum_j alpha_j e_{a_j} with alpha = R_AA^{-1} b,
    b_j = <e_{a_j}, h*> = int c_1(u) R(a_j, u) mu(du).  The loading
        g(s) = <e_s, P_A h*> / ||P_A h*|| = sum_j alpha_j R(s, a_j) / ||P_A h*||
    is exact and cheap, the projection is exactly orthogonal (span of one Gaussian variable),
    and its kinks sit only at the anchors.  As A becomes dense, P_A h* -> h*.
    """

    name = "output-adapted(first chaos)"

    def __init__(self, field, func, mu, anchors, n=16, levels=10):
        super().__init__(field)
        a = np.asarray(sorted(set(float(x) for x in anchors)), float)
        self.anchors = a
        bps = list(field.singular_points()) + [field.T]
        b = np.empty(a.size)
        for j, aj in enumerate(a):
            u, wu = rule_1d(mu, bps + [float(aj)], n, levels)
            sg = np.sqrt(np.maximum(field.var(u), 0.0))
            c1 = np.where(sg > 0, func.chat(u, sg, 1)[1] / np.where(sg > 0, sg, 1.0), 0.0)
            b[j] = float(np.sum(wu * c1 * field.R(np.full_like(u, aj), u)))
        RAA = field.R(a[:, None], a[None, :])
        alpha = np.linalg.solve(RAA, b)
        norm2 = float(alpha @ RAA @ alpha)
        if norm2 <= 0:
            raise ValueError("output has no first-chaos component on these anchors")
        self.alpha = alpha / math.sqrt(norm2)
        self.captured_first_chaos = norm2          # ||P_A h*||^2 (<= ||h*||^2)

    @property
    def rank(self):
        return 1

    def loadings(self, s):
        s = np.atleast_1d(np.asarray(s, float)).ravel()
        return (self.field.R(s[:, None], self.anchors[None, :]) @ self.alpha)[:, None]

    def breakpoints(self):
        return sorted(set(list(self.field.singular_points()) + list(self.anchors)))
