"""Gaussian-subordinated output fields X_s = f(s, Y_s) and their chaos data.

For sigma_s^2 = R(s, s) and Z ~ N(0, 1), the normalised Hermite coefficients
    chat_m(s) = E[f(s, sigma_s Z) He_m(Z)] / sqrt(m!)
give the chaos decomposition X_s = sum_m chat_m(s) He_m(Y_s / sigma_s) / sqrt(m!), and
(Mehler) for any covariance x with |x| <= sigma_s sigma_u
    K(s, u; x) = sum_m chat_m(s) chat_m(u) (x / (sigma_s sigma_u))^m = E[f(s, U) f(u, V)],
(U, V) centred normal with variances (sigma_s^2, sigma_u^2) and covariance x.
Second quantisation of a projection Pi multiplies the m-th term by (R_Pi/R)^m, so
E[X_s Gamma(Pi) X_u] = K(s, u; R_Pi(s, u)); chaos truncation keeps m <= N.

Families: WickExponential (Paper III), WickJet (Paper V), PathwiseExponential,
PositivePart, Indicator, Polynomial, and Generic (any f with declared breakpoints).
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.special import ndtr, log_ndtr

SQ2PI = math.sqrt(2.0 * math.pi)


def npdf(x):
    return np.exp(-0.5 * np.asarray(x, float) ** 2) / SQ2PI


def normalized_hermite(x, M):
    """h_m(x) = He_m(x)/sqrt(m!), m = 0..M, stable recurrence; shape (M+1,) + x.shape."""
    x = np.asarray(x, float)
    out = np.empty((M + 1,) + x.shape)
    out[0] = 1.0
    if M >= 1:
        out[1] = x
    for m in range(1, M):
        out[m + 1] = (x * out[m] - math.sqrt(m) * out[m - 1]) / math.sqrt(m + 1)
    return out


@lru_cache(maxsize=32)
def _gauss_hermite(n):
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return x, w / SQ2PI


@lru_cache(maxsize=32)
def _gl(n):
    return np.polynomial.legendre.leggauss(n)


def gaussian_expectation_piecewise(g, breaks_z, n=48, L=12.0):
    """E[g(Z)] with g smooth between the given z-breakpoints (vectorised over leading axis).

    breaks_z: array (P, B) of breakpoints in z for each of P problems (nan = unused).
    g: callable(Z) with Z of shape (P, Q) -> (P, Q).
    """
    breaks_z = np.atleast_2d(np.asarray(breaks_z, float))
    P = breaks_z.shape[0]
    b = np.where(np.isfinite(breaks_z), np.clip(breaks_z, -L, L), np.nan)
    cuts = np.concatenate([np.full((P, 1), -L), b, np.full((P, 1), L)], axis=1)
    cuts = np.sort(np.where(np.isnan(cuts), L, cuts), axis=1)
    x, w = _gl(n)
    a, c = cuts[:, :-1], cuts[:, 1:]
    Z = 0.5 * (c - a)[:, :, None] * (x[None, None, :] + 1.0) + a[:, :, None]
    W = 0.5 * (c - a)[:, :, None] * w[None, None, :]
    Z = Z.reshape(P, -1)
    W = W.reshape(P, -1) * npdf(Z)
    return np.sum(W * g(Z), axis=1)


class SubordinatedFunctional:
    """Base class: X_s = f(s, Y_s).  Subclasses supply chat(s, sigma, M) and kernel(...)."""

    name = "subordinated"
    smooth = True

    def f(self, s, y):  # pragma: no cover - interface
        raise NotImplementedError

    def breakpoints_y(self, s):
        """y-values where f(s, .) is not smooth (list per s); default none."""
        return []

    # ---- chaos data -----------------------------------------------------------------
    def chat(self, s, sigma, M):
        """Normalised Hermite coefficients, shape (M+1, len(s)); generic composite quadrature.

        Composite Gauss--Legendre panels of width <= 0.5 in z, split at the declared
        breakpoints, keep the rule accurate for nonlinearities with complex singularities near
        the real axis (e.g. tanh) and for kinks/jumps.
        """
        s = np.atleast_1d(np.asarray(s, float)).ravel()
        sigma = np.broadcast_to(np.asarray(sigma, float), s.shape).ravel()
        out = np.zeros((M + 1, s.size))
        zero = sigma <= 0
        if np.any(zero):
            out[0, zero] = self.f(s[zero], np.zeros(int(np.sum(zero))))
        idx = np.flatnonzero(~zero)
        for c0 in range(0, idx.size, 64):
            ii = idx[c0:c0 + 64]
            extra = _pad([[b / sigma[i] for b in self.breakpoints_y(s[i])] for i in ii])
            Z, W = _row_nodes(_cut_matrix(12.0, 0.5, extra), 16)
            vals = self.f(np.broadcast_to(s[ii][:, None], Z.shape), sigma[ii][:, None] * Z)
            Hm = normalized_hermite(Z, M)                       # (M+1, R, Q)
            out[:, ii] = np.einsum("mrq,rq->mr", Hm, W * npdf(Z) * vals)
        return out

    def second_moment_diag(self, s, sigma):
        """E[f(s, Y_s)^2] (used for chaos-tail bounds)."""
        s = np.atleast_1d(np.asarray(s, float)).ravel()
        sigma = np.broadcast_to(np.asarray(sigma, float), s.shape).ravel()
        out = np.empty(s.size)
        for c0 in range(0, s.size, 256):
            ii = np.arange(c0, min(c0 + 256, s.size))
            sg = np.where(sigma[ii] > 0, sigma[ii], 0.0)
            extra = _pad([[b / sigma[i] for b in self.breakpoints_y(s[i])] if sigma[i] > 0 else []
                          for i in ii])
            Z, W = _row_nodes(_cut_matrix(12.0, 0.5, extra), 16)
            vals = self.f(np.broadcast_to(s[ii][:, None], Z.shape), sg[:, None] * Z)
            out[ii] = np.sum(W * npdf(Z) * vals ** 2, axis=1)
        return out

    # ---- two-point kernel -----------------------------------------------------------
    def kernel(self, s, u, x, sig_s, sig_u):
        """E[f(s,U) f(u,V)] (Mehler); generic route = Hermite series with rigorous tail bound,
        falling back to breakpoint-aware conditional quadrature where the bound is not met."""
        from ._generic_kernel import GenericKernelCache
        if not hasattr(self, "_gk"):
            self._gk = GenericKernelCache(self)
        return self._gk.kernel(s, u, x, sig_s, sig_u)

    def kernel_derivative(self, s, u, x, sig_s, sig_u, M=120):
        """K'(s,u;x) = E[d_y f(s,U) d_y f(u,V)] (Price); generic route = Hermite series, data cached
        per unique node (no tail bound is attached to this derivative series)."""
        from ._generic_kernel import GenericKernelCache
        if not hasattr(self, "_gkd"):
            self._gkd = GenericKernelCache(self, M=M)
        shape = np.shape(x)
        s_ = np.broadcast_to(np.asarray(s, float), shape).ravel()
        u_ = np.broadcast_to(np.asarray(u, float), shape).ravel()
        ss = np.broadcast_to(np.asarray(sig_s, float), shape).ravel()
        su = np.broadcast_to(np.asarray(sig_u, float), shape).ravel()
        cs, _ = self._gkd._data(s_, ss)
        cu, _ = self._gkd._data(u_, su)
        den = ss * su
        r = np.where(den > 0, np.asarray(x, float).ravel() / np.where(den > 0, den, 1.0), 0.0)
        acc = np.zeros(r.size)
        rp = np.ones(r.size)
        for m in range(1, cs.shape[0]):
            acc += m * cs[m] * cu[m] * rp
            rp = rp * r
        return (acc / np.where(den > 0, den, 1.0)).reshape(shape)

    def kernel_truncated(self, s, u, x, sig_s, sig_u, N, chat_s=None, chat_u=None):
        """sum_{m<=N} chat_m(s) chat_m(u) r^m with r = x/(sig_s sig_u)."""
        s = np.asarray(s, float)
        u = np.asarray(u, float)
        cs = self.chat(s.ravel(), np.asarray(sig_s, float).ravel(), N) if chat_s is None else chat_s
        cu = self.chat(u.ravel(), np.asarray(sig_u, float).ravel(), N) if chat_u is None else chat_u
        den = (np.asarray(sig_s, float) * np.asarray(sig_u, float)).ravel()
        r = np.where(den > 0, np.asarray(x, float).ravel() / np.where(den > 0, den, 1.0), 0.0)
        powers = r[None, :] ** np.arange(N + 1)[:, None]
        return np.sum(cs[: N + 1] * cu[: N + 1] * powers, axis=0).reshape(np.shape(x))


def _pad(lists):
    """Ragged list-of-lists -> 2-D array padded with nan (at least one column)."""
    width = max([len(x) for x in lists] + [1])
    out = np.full((len(lists), width), np.nan)
    for i, x in enumerate(lists):
        if len(x):
            out[i, :len(x)] = x
    return out


def _cut_matrix(L, width, extra):
    """Sorted panel cuts per row: uniform cuts on [-L, L] plus row-specific extra cuts."""
    base = np.linspace(-L, L, int(round(2 * L / width)) + 1)
    extra = np.atleast_2d(extra)
    R = extra.shape[0]
    allc = np.concatenate([np.broadcast_to(base, (R, base.size)), np.clip(extra, -L, L)], axis=1)
    allc = np.where(np.isnan(allc), L, allc)
    return np.sort(allc, axis=1)


def _row_nodes(cuts, n):
    """Composite Gauss--Legendre nodes/weights for each row of panel cuts (zero-length ok)."""
    x, w = _gl(n)
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    a, b = cuts[:, :-1], cuts[:, 1:]
    Z = a[..., None] + (b - a)[..., None] * x
    W = (b - a)[..., None] * w
    return Z.reshape(cuts.shape[0], -1), W.reshape(cuts.shape[0], -1)


def bivariate_expectation(func, s, u, x, sig_s, sig_u, n=12, width=0.5, L=10.0, chunk=24):
    """Generic E[f(s,U) f(u,V)] (Mehler kernel), vectorised composite quadrature.

    Condition on U = sig_s z:  V | U ~ N(c z, v),  c = x/sig_s,  v = sig_u^2 - c^2.
    Outer z-panels: uniform width <= 0.5, split at breakpoints of f(s,.) and at the points
    z* = b/c where c z crosses a breakpoint b of f(u,.), with geometric grading of scale
    sqrt(v)/|c| around z* (near-diagonal pairs).  Inner panels split at (b - c z)/sqrt(v).
    """
    s = np.atleast_1d(np.asarray(s, float)).ravel()
    u = np.atleast_1d(np.asarray(u, float)).ravel()
    x = np.atleast_1d(np.asarray(x, float)).ravel()
    sig_s = np.broadcast_to(np.asarray(sig_s, float), s.shape).ravel()
    sig_u = np.broadcast_to(np.asarray(sig_u, float), u.shape).ravel()
    P = s.size
    out = np.empty(P)
    for c0 in range(0, P, chunk):
        sl = slice(c0, min(c0 + chunk, P))
        ss, uu, gs, gu, xx = s[sl], u[sl], sig_s[sl], sig_u[sl], x[sl]
        R = ss.size
        pos = gs > 0
        c = np.where(pos, xx / np.where(pos, gs, 1.0), 0.0)
        v = np.maximum(gu * gu - c * c, 0.0)
        sv = np.sqrt(v)
        bu_list = [func.breakpoints_y(ui) for ui in uu]
        extra = []
        for i in range(R):
            e = [b / gs[i] for b in func.breakpoints_y(ss[i])] if pos[i] else []
            if abs(c[i]) > 0:
                for b in bu_list[i]:
                    zc = b / c[i]
                    e.append(zc)
                    wd = max(sv[i] / abs(c[i]), 1e-13)
                    e += [zc + sg * wd * 4.0 ** (k - 3) for k in range(1, 9) for sg in (-1, 1)]
            extra.append(e)
        Z1, W1 = _row_nodes(_cut_matrix(L, width, _pad(extra)), n)          # (R, Q1)
        U = np.where(pos[:, None], gs[:, None] * Z1, 0.0)
        fU = func.f(np.broadcast_to(ss[:, None], U.shape), U)
        mean = c[:, None] * Z1                                               # (R, Q1)
        Q1 = Z1.shape[1]
        # inner expectation per (row, outer node)
        rows_u = np.repeat(uu, Q1)
        rows_m = mean.ravel()
        rows_sd = np.repeat(sv, Q1)
        inner = np.empty(rows_m.size)
        det = rows_sd <= 0
        if np.any(det):
            inner[det] = func.f(rows_u[det], rows_m[det])
        nd = np.flatnonzero(~det)
        for d0 in range(0, nd.size, 4096):
            jj = nd[d0:d0 + 4096]
            ex = _pad([[(b - rows_m[j]) / rows_sd[j] for b in bu_list[j // Q1]] for j in jj])
            Z2, W2 = _row_nodes(_cut_matrix(L, width, ex), n)
            V = rows_m[jj][:, None] + rows_sd[jj][:, None] * Z2
            inner[jj] = np.sum(W2 * npdf(Z2) * func.f(np.broadcast_to(rows_u[jj][:, None], V.shape), V), axis=1)
        inner = inner.reshape(R, Q1)
        out[sl] = np.sum(W1 * npdf(Z1) * fU * inner, axis=1)
    return out


# =====================================================================================
class WickExponential(SubordinatedFunctional):
    """X_s = xi0 exp(eta Y_s - eta^2 R(s,s)/2)  (Paper III).  K = xi0^2 exp(eta^2 x)."""

    def __init__(self, xi0=1.0, eta=1.0, field=None):
        self.xi0, self.eta, self.field = float(xi0), float(eta), field
        self.name = f"WickExp(eta={eta})"

    def f(self, s, y):
        var = self.field.var(s)
        return self.xi0 * np.exp(self.eta * y - 0.5 * self.eta ** 2 * var)

    def chat(self, s, sigma, M):
        sigma = np.atleast_1d(np.asarray(sigma, float))
        m = np.arange(M + 1)
        lg = np.array([math.lgamma(k + 1) for k in m])
        es = self.eta * sigma
        with np.errstate(divide="ignore"):
            logv = m[:, None] * np.log(np.where(es > 0, es, 1.0))[None, :] - 0.5 * lg[:, None]
        out = self.xi0 * np.exp(logv)
        out[1:, es <= 0] = 0.0
        return out

    def kernel(self, s, u, x, sig_s, sig_u):
        return self.xi0 ** 2 * np.exp(self.eta ** 2 * np.asarray(x, float))

    def second_moment_diag(self, s, sigma):
        return self.xi0 ** 2 * np.exp(self.eta ** 2 * np.asarray(sigma, float) ** 2)

    def kernel_derivative(self, s, u, x, sig_s, sig_u, M=None):
        return self.xi0 ** 2 * self.eta ** 2 * np.exp(self.eta ** 2 * np.asarray(x, float))


class WickJet(SubordinatedFunctional):
    """Paper V jet: X_s = xi0 sum_j q_j d^j/deta^j E_eta(e_s);  c_m = xi0 sum_j q_j eta^{m-j}/(m-j)!."""

    def __init__(self, xi0=1.0, eta=1.0, q=(1.0,), field=None):
        self.xi0, self.eta, self.q, self.field = float(xi0), float(eta), tuple(map(float, q)), field
        self.name = f"WickJet(eta={eta}, q={self.q})"

    def cm(self, M):
        c = np.zeros(M + 1)
        for m in range(M + 1):
            c[m] = sum(self.q[j] * self.eta ** (m - j) / math.factorial(m - j)
                       for j in range(min(m, len(self.q) - 1) + 1))
        return self.xi0 * c

    def chat(self, s, sigma, M):
        sigma = np.atleast_1d(np.asarray(sigma, float))
        c = self.cm(M)
        m = np.arange(M + 1)
        fac = np.array([math.sqrt(math.factorial(k)) for k in m])
        return (c * fac)[:, None] * sigma[None, :] ** m[:, None]

    def f(self, s, y):
        var = self.field.var(s)
        w = y - self.eta * var             # Wick powers of y times E_eta: shift identity
        base = self.xi0 * np.exp(self.eta * y - 0.5 * self.eta ** 2 * var)
        fac = np.zeros_like(base)
        for j, qj in enumerate(self.q):
            # W^{<>j} <> E = He_j-type polynomial in shifted variable: sum over Hermite He_j(w; var)
            fac = fac + qj * _wick_power_poly(w, var, j)
        return base * fac

    def kernel(self, s, u, x, sig_s, sig_u):
        x = np.asarray(x, float)
        a = self.eta ** 2
        d = len(self.q) - 1
        tot = np.zeros_like(x)
        for j in range(d + 1):
            for k in range(d + 1):
                ssum = sum(math.comb(j, i) * math.comb(k, i) * math.factorial(i)
                           * self.eta ** (j + k - 2 * i) * x ** (j + k - i) for i in range(min(j, k) + 1))
                tot = tot + self.q[j] * self.q[k] * ssum
        return self.xi0 ** 2 * np.exp(a * x) * tot


def _wick_power_poly(w, var, j):
    """Hermite polynomial He_j(w; var) = var^{j/2} He_j(w / sqrt(var)) (Wick power in variance var)."""
    if j == 0:
        return np.ones_like(w)
    h_prev, h = np.ones_like(w), w.copy()
    for m in range(1, j):
        h_prev, h = h, w * h - m * var * h_prev
    return h


class PathwiseExponential(SubordinatedFunctional):
    """X_s = xi0 exp(eta Y_s + drift(s)) (ordinary exponential, e.g. lognormal variance)."""

    def __init__(self, xi0=1.0, eta=1.0, drift=None):
        self.xi0, self.eta = float(xi0), float(eta)
        self.drift = (lambda s: np.zeros_like(np.asarray(s, float))) if drift is None else drift
        self.name = f"Exp(eta={eta})"

    def f(self, s, y):
        return self.xi0 * np.exp(self.eta * y + self.drift(s))

    def chat(self, s, sigma, M):
        s = np.atleast_1d(np.asarray(s, float))
        sigma = np.atleast_1d(np.asarray(sigma, float))
        m = np.arange(M + 1)
        lg = np.array([math.lgamma(k + 1) for k in m])
        es = self.eta * sigma
        pref = self.xi0 * np.exp(0.5 * es ** 2 + self.drift(s))
        with np.errstate(divide="ignore"):
            logv = m[:, None] * np.log(np.where(es > 0, es, 1.0))[None, :] - 0.5 * lg[:, None]
        out = pref[None, :] * np.exp(logv)
        out[1:, es <= 0] = 0.0
        return out

    def kernel(self, s, u, x, sig_s, sig_u):
        s, u, x = map(lambda z: np.asarray(z, float), (s, u, x))
        return self.xi0 ** 2 * np.exp(self.drift(s) + self.drift(u)
                                      + 0.5 * self.eta ** 2 * (np.asarray(sig_s) ** 2 + np.asarray(sig_u) ** 2)
                                      + self.eta ** 2 * x)


class PositivePart(SubordinatedFunctional):
    """X_s = scale * (Y_s + m(s) - c)_+  (exceedance / degree-day type output)."""

    smooth = False

    def __init__(self, c=0.0, mean=None, scale=1.0):
        self.c, self.scale = float(c), float(scale)
        self.mean = (lambda s: np.zeros_like(np.asarray(s, float))) if mean is None else mean
        self.name = f"PositivePart(c={c})"

    def f(self, s, y):
        return self.scale * np.maximum(y + self.mean(s) - self.c, 0.0)

    def breakpoints_y(self, s):
        return [float(self.c - self.mean(np.array([s]))[0])]

    def chat(self, s, sigma, M):
        s = np.atleast_1d(np.asarray(s, float))
        sigma = np.atleast_1d(np.asarray(sigma, float))
        out = np.zeros((M + 1, s.size))
        thr = self.c - self.mean(s)
        pos = sigma > 0
        alpha = np.where(pos, thr / np.where(pos, sigma, 1.0), 0.0)
        # m = 0: sigma (phi(a) - a Phibar(a)); m = 1: sigma Phibar(a); m >= 2: sigma phi(a) h_{m-2}(a) sqrt((m-2)!/m!)
        Phibar = ndtr(-alpha)
        out[0] = np.where(pos, sigma * (npdf(alpha) - alpha * Phibar), np.maximum(-thr, 0.0))
        if M >= 1:
            out[1] = np.where(pos, sigma * Phibar, 0.0)
        if M >= 2:
            H = normalized_hermite(alpha, M - 2)                  # He_{m-2}(a)/sqrt((m-2)!)
            for m in range(2, M + 1):
                out[m] = np.where(pos, sigma * npdf(alpha) * H[m - 2] / math.sqrt(m * (m - 1)), 0.0)
        return self.scale * out

    def kernel(self, s, u, x, sig_s, sig_u):
        """E[(U - a)_+ (V - b)_+] in closed form (bivariate normal partial moments)."""
        s, u = np.asarray(s, float), np.asarray(u, float)
        a = self.c - self.mean(s)
        b = self.c - self.mean(u)
        return self.scale ** 2 * _bivariate_call_product(a, b, np.asarray(sig_s, float),
                                                         np.asarray(sig_u, float), np.asarray(x, float))

    def kernel_derivative(self, s, u, x, sig_s, sig_u, M=None):
        """Price: d/dx E[(U-a)_+(V-b)_+] = P(U > a, V > b)."""
        s, u = np.asarray(s, float), np.asarray(u, float)
        a = self.c - self.mean(s)
        b = self.c - self.mean(u)
        return self.scale ** 2 * _bivariate_orthant(a, b, np.asarray(sig_s, float),
                                                    np.asarray(sig_u, float), np.asarray(x, float))


class Indicator(SubordinatedFunctional):
    """X_s = scale * 1{Y_s + m(s) > c}  (occupation / exceedance-time type output)."""

    smooth = False

    def __init__(self, c=0.0, mean=None, scale=1.0):
        self.c, self.scale = float(c), float(scale)
        self.mean = (lambda s: np.zeros_like(np.asarray(s, float))) if mean is None else mean
        self.name = f"Indicator(c={c})"

    def f(self, s, y):
        return self.scale * (y + self.mean(s) > self.c).astype(float)

    def breakpoints_y(self, s):
        return [float(self.c - self.mean(np.array([s]))[0])]

    def chat(self, s, sigma, M):
        s = np.atleast_1d(np.asarray(s, float))
        sigma = np.atleast_1d(np.asarray(sigma, float))
        out = np.zeros((M + 1, s.size))
        thr = self.c - self.mean(s)
        pos = sigma > 0
        alpha = np.where(pos, thr / np.where(pos, sigma, 1.0), 0.0)
        out[0] = np.where(pos, ndtr(-alpha), (thr < 0).astype(float))
        if M >= 1:
            H = normalized_hermite(alpha, M - 1)                  # He_{m-1}(a)/sqrt((m-1)!)
            for m in range(1, M + 1):
                out[m] = np.where(pos, npdf(alpha) * H[m - 1] / math.sqrt(m), 0.0)
        return self.scale * out

    def kernel(self, s, u, x, sig_s, sig_u):
        s, u = np.asarray(s, float), np.asarray(u, float)
        a = self.c - self.mean(s)
        b = self.c - self.mean(u)
        return self.scale ** 2 * _bivariate_orthant(a, b, np.asarray(sig_s, float),
                                                    np.asarray(sig_u, float), np.asarray(x, float))

    def kernel_derivative(self, s, u, x, sig_s, sig_u, M=None):
        """Price: d/dx P(U > a, V > b) = bivariate normal density at (a, b)."""
        s, u = np.asarray(s, float), np.asarray(u, float)
        a = self.c - self.mean(s)
        b = self.c - self.mean(u)
        sa, sb, xx = (np.asarray(v, float) for v in np.broadcast_arrays(sig_s, sig_u, x))
        a, b = np.broadcast_arrays(a, b)
        det = sa ** 2 * sb ** 2 - xx ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            q = (sb ** 2 * a ** 2 - 2 * xx * a * b + sa ** 2 * b ** 2) / det
            dens = np.where(det > 0, np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(np.where(det > 0, det, 1.0))), np.inf)
        return self.scale ** 2 * dens


class Polynomial(SubordinatedFunctional):
    """X_s = sum_k a_k (Y_s + m(s))^k  (finite chaos; exact)."""

    def __init__(self, coeffs, mean=None):
        self.a = np.asarray(coeffs, float)
        self.mean = (lambda s: np.zeros_like(np.asarray(s, float))) if mean is None else mean
        self.name = f"Polynomial(deg={len(self.a) - 1})"

    def f(self, s, y):
        z = y + self.mean(s)
        return sum(ak * z ** k for k, ak in enumerate(self.a))

    def chat(self, s, sigma, M):
        """Closed form: (sigma Z + m)^k = sum_l C(k,l) m^(k-l) sigma^l Z^l and
        Z^l = sum_i l!/(2^i i! (l-2i)!) He_{l-2i}(Z), so chat_j = sqrt(j!) [He_j-coefficient]."""
        s = np.atleast_1d(np.asarray(s, float)).ravel()
        sg = np.broadcast_to(np.asarray(sigma, float), s.shape).ravel()
        m = np.broadcast_to(np.asarray(self.mean(s), float), s.shape).ravel()
        out = np.zeros((M + 1, s.size))
        for k, ak in enumerate(self.a):
            if ak == 0.0:
                continue
            for l in range(k + 1):
                base = ak * math.comb(k, l) * m ** (k - l) * sg ** l
                for i in range(l // 2 + 1):
                    j = l - 2 * i
                    if j <= M:
                        c = math.factorial(l) / (2 ** i * math.factorial(i) * math.factorial(j))
                        out[j] += base * c * math.sqrt(math.factorial(j))
        return out

    def kernel(self, s, u, x, sig_s, sig_u):
        deg = len(self.a) - 1
        M = 2 * deg + 2
        cs = self.chat(np.atleast_1d(s).ravel(), np.atleast_1d(sig_s).ravel(), M)
        cu = self.chat(np.atleast_1d(u).ravel(), np.atleast_1d(sig_u).ravel(), M)
        den = (np.asarray(sig_s) * np.asarray(sig_u)).ravel()
        r = np.where(den > 0, np.asarray(x, float).ravel() / np.where(den > 0, den, 1), 0.0)
        return np.sum(cs * cu * r[None, :] ** np.arange(M + 1)[:, None], axis=0).reshape(np.shape(x))


class ExpCall(SubordinatedFunctional):
    """X_s = scale * (A(s) exp(kappa Y_s) - B)_+  ('call on a lognormal', e.g. excess variance).

    Closed-form kernel: with k_i = log(B/A_i)/kappa and Cameron--Martin shifts,
    E[(A_s e^{kU} - B)_+ (A_u e^{kV} - B)_+] = sum over t in {(1,1),(1,0),(0,1),(0,0)} of
    coef_t * exp(t'S t / 2) * P(U + (S t)_1 > k_s, V + (S t)_2 > k_u), S the covariance of kappa(U,V).
    """

    smooth = False

    def __init__(self, A, kappa, B, scale=1.0, name="exp-call"):
        self.A, self.kappa, self.B, self.scale, self.name = A, float(kappa), float(B), float(scale), name

    def f(self, s, y):
        return self.scale * np.maximum(self.A(s) * np.exp(self.kappa * y) - self.B, 0.0)

    def breakpoints_y(self, s):
        a = float(self.A(np.array([s]))[0])
        return [math.log(self.B / a) / self.kappa]

    def kernel(self, s, u, x, sig_s, sig_u):
        s, u, x = (np.asarray(v, float) for v in np.broadcast_arrays(s, u, x))
        ss, su = (np.asarray(v, float) for v in np.broadcast_arrays(sig_s, sig_u, x)[:2])
        k = self.kappa
        As, Au = self.A(s), self.A(u)
        ks, ku = np.log(self.B / As) / k, np.log(self.B / Au) / k      # thresholds on U, V
        Sss, Suu, Ssu = k * k * ss * ss, k * k * su * su, k * k * x      # covariance of (kU, kV)
        out = np.zeros(s.shape)
        terms = (((1, 1), As * Au), ((1, 0), -self.B * As), ((0, 1), -self.B * Au), ((0, 0), self.B ** 2 * np.ones_like(As)))
        for (t1, t2), coef in terms:
            m1 = (t1 * Sss + t2 * Ssu) / k                                # shift of U
            m2 = (t1 * Ssu + t2 * Suu) / k                                # shift of V
            quad = 0.5 * (t1 * t1 * Sss + 2 * t1 * t2 * Ssu + t2 * t2 * Suu)
            prob = _bivariate_orthant(ks - m1, ku - m2, ss, su, x)
            out = out + coef * np.exp(quad) * prob
        return self.scale ** 2 * out


class Generic(SubordinatedFunctional):
    """Any user nonlinearity f(s, y) with declared y-breakpoints; no closed forms used."""

    def __init__(self, f, breakpoints=None, name="generic", smooth=True, homogeneous=False):
        """homogeneous=True declares f(s, y) = f(y) (no explicit time dependence): Hermite data then
        depend on s only through sigma_s and are interpolated in sigma (Chebyshev, audited)."""
        self._f, self._bp, self.name, self.smooth = f, breakpoints, name, smooth
        self.homogeneous = homogeneous

    def f(self, s, y):
        return self._f(np.asarray(s, float), np.asarray(y, float))

    def breakpoints_y(self, s):
        return [] if self._bp is None else list(self._bp(s))


# ------------------------------------------------------------------- closed-form helpers
from ._bvn import (bvn_upper, bvn_upper_integral, std_call_product as _std_call_product,
                   bivariate_orthant as _bivariate_orthant, bivariate_call_product as _bivariate_call_product)
