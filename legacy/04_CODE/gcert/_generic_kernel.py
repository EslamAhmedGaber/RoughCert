"""Fast generic Mehler kernel with a rigorous series tail bound.

K(s,u;x) = sum_m chat_m(s) chat_m(u) r^m,  r = x/(sig_s sig_u),  |r| <= 1.
Truncating at M leaves |tail| <= tau_M(s) tau_M(u) |r|^{M+1},
tau_M(s)^2 = E f(s,Y_s)^2 - sum_{m<=M} chat_m(s)^2   (Cauchy--Schwarz).
Pairs whose bound exceeds the tolerance (near-diagonal pairs of non-smooth f) are evaluated by
the breakpoint-aware conditional quadrature instead.  Hermite data are computed once per
unique s (the 1-D nodes) and cached.
"""
from __future__ import annotations

import numpy as np


class GenericKernelCache:
    def __init__(self, func, M=240, tol_abs=1e-13, tol_rel=1e-12):
        self.func, self.M, self.tol_abs, self.tol_rel = func, M, tol_abs, tol_rel
        self._c, self._tau = {}, {}

    def _sigma_interp(self, sig):
        """Chebyshev interpolation in sigma of (chat_0..M, tau) for homogeneous smooth f (audited)."""
        if not hasattr(self, "_cheb"):
            self._cheb = None
        smax = float(np.max(sig)) if sig.size else 1.0
        if self._cheb is None or smax > self._cheb[1] * (1 + 1e-12):
            lo, hi = 0.0, max(smax, 1e-12) * 1.02
            def build(nc):
                k = np.arange(nc)
                x = np.cos(np.pi * (k + 0.5) / nc)                     # Chebyshev nodes on [-1, 1]
                sg = 0.5 * (hi - lo) * (x + 1) + lo
                C = self.func.chat(np.zeros(nc), sg, self.M)
                m2 = self.func.second_moment_diag(np.zeros(nc), sg)
                tau = np.sqrt(np.maximum(m2 - np.sum(C ** 2, axis=0), 0.0))
                coef = np.polynomial.chebyshev.chebfit(x, np.vstack([C, tau]).T, nc - 1)
                return coef
            c1, c2 = build(96), build(128)
            xa = np.linspace(-1, 1, 257)
            v1 = np.polynomial.chebyshev.chebval(xa, c1)
            v2 = np.polynomial.chebyshev.chebval(xa, c2)
            self.interp_audit = float(np.max(np.abs(v1 - v2)))
            self._cheb = (c2, hi, lo)
        coef, hi, lo = self._cheb
        us, inv = np.unique(sig, return_inverse=True)
        x = 2 * (us - lo) / (hi - lo) - 1
        vals = np.polynomial.chebyshev.chebval(x, coef)                  # (M+2, n_unique)
        return vals[:-1][:, inv], np.maximum(vals[-1], 0.0)[inv]

    def _data(self, s, sig):
        s = np.asarray(s, float).ravel()
        sig = np.asarray(sig, float).ravel()
        if getattr(self.func, "homogeneous", False):
            return self._sigma_interp(sig)
        keys = np.round(s, 15)
        uniq, inv = np.unique(keys, return_inverse=True)
        need = [i for i, k in enumerate(uniq) if k not in self._c]
        if need:
            first = np.array([np.flatnonzero(keys == uniq[i])[0] for i in need])
            C = self.func.chat(s[first], sig[first], self.M)
            m2 = self.func.second_moment_diag(s[first], sig[first])
            tau = np.sqrt(np.maximum(m2 - np.sum(C ** 2, axis=0), 0.0))
            for j, i in enumerate(need):
                self._c[uniq[i]] = C[:, j]
                self._tau[uniq[i]] = tau[j]
        Cm = np.stack([self._c[k] for k in uniq], axis=1)[:, inv]
        tau = np.array([self._tau[k] for k in uniq])[inv]
        return Cm, tau

    def kernel(self, s, u, x, sig_s, sig_u, chunk=100_000):
        """Chunked over node pairs: the (M+1) x pairs coefficient arrays stay bounded in memory."""
        from .functionals import bivariate_expectation
        shape = np.shape(x)
        s = np.broadcast_to(np.asarray(s, float), shape).ravel()
        u = np.broadcast_to(np.asarray(u, float), shape).ravel()
        sig_s = np.broadcast_to(np.asarray(sig_s, float), shape).ravel()
        sig_u = np.broadcast_to(np.asarray(sig_u, float), shape).ravel()
        x = np.asarray(x, float).ravel()
        acc_all = np.empty(x.size)
        nbad = 0
        for c0 in range(0, x.size, chunk):
            sl = slice(c0, min(c0 + chunk, x.size))
            cs, ts = self._data(s[sl], sig_s[sl])
            cu, tu = self._data(u[sl], sig_u[sl])
            den = sig_s[sl] * sig_u[sl]
            r = np.clip(np.where(den > 0, x[sl] / np.where(den > 0, den, 1.0), 0.0), -1.0, 1.0)
            acc = np.zeros(r.size)
            rp = np.ones(r.size)
            for m in range(self.M + 1):
                acc += cs[m] * cu[m] * rp
                rp = rp * r
            bound = ts * tu * np.abs(rp)
            bad = bound > np.maximum(self.tol_abs, self.tol_rel * np.abs(acc))
            if np.any(bad):
                ss, uu, xx, gs, gu = s[sl][bad], u[sl][bad], x[sl][bad], sig_s[sl][bad], sig_u[sl][bad]
                has_bp = any(len(self.func.breakpoints_y(si)) for si in np.unique(ss)[:8])
                if has_bp:
                    acc[bad] = bivariate_expectation(self.func, ss, uu, xx, gs, gu)
                else:
                    acc[bad] = smooth_bivariate(self.func, ss, uu, xx, gs, gu)
            nbad += int(np.sum(bad))
            acc_all[sl] = acc
        self.last_fallback_fraction = nbad / max(x.size, 1)
        return acc_all.reshape(shape)


def smooth_bivariate(func, s, u, x, sig_s, sig_u, width=0.5, L=10.0, n_out=8, n_in=24, chunk=2048):
    """E[f(s,U) f(u,V)] for f smooth in y (no breakpoints): condition on U = sig_s z,
    V | U ~ N(c z, v); outer composite Gauss--Legendre in z (panels of width <= 0.5),
    inner Gauss--Hermite with n_in nodes for the residual smoothing (accurate for the small v of
    the near-diagonal pairs this route is used for; far pairs use the Hermite series)."""
    from .functionals import npdf
    zo = np.linspace(-L, L, int(round(2 * L / width)) + 1)
    xg, wg = np.polynomial.legendre.leggauss(n_out)
    Z1 = (0.5 * (zo[1:] - zo[:-1])[:, None] * (xg[None, :] + 1) + zo[:-1, None]).ravel()
    W1 = (0.5 * (zo[1:] - zo[:-1])[:, None] * wg[None, :]).ravel() * npdf(Z1)
    xh, wh = np.polynomial.hermite_e.hermegauss(n_in)
    wh = wh / np.sqrt(2 * np.pi)
    out = np.empty(s.size)
    for c0 in range(0, s.size, chunk):
        sl = slice(c0, min(c0 + chunk, s.size))
        ss, uu, gs, gu, xx = s[sl], u[sl], sig_s[sl], sig_u[sl], x[sl]
        pos = gs > 0
        c = np.where(pos, xx / np.where(pos, gs, 1.0), 0.0)
        sv = np.sqrt(np.maximum(gu * gu - c * c, 0.0))
        Uv = gs[:, None] * Z1[None, :]
        fU = func.f(np.broadcast_to(ss[:, None], Uv.shape), Uv)
        mean = c[:, None] * Z1[None, :]
        Vv = mean[:, :, None] + sv[:, None, None] * xh[None, None, :]
        fV = func.f(np.broadcast_to(uu[:, None, None], Vv.shape), Vv) @ wh
        out[sl] = np.sum(W1[None, :] * fU * fV, axis=1)
    return out
