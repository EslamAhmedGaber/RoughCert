"""Martingale-sharp risk endpoints (convex order).

Setting.  Lambda = X + R with E[R | X] = 0 and E[R^2] <= P.  This holds whenever X = E[Lambda | G] for a
sigma-field G: the conditional-expectation monitor (X = Gamma(Pi_grid) Lambda), and the deterministic
surrogate G(z) = Gamma(Pi) Lambda of route D (full chaos).  Law(Lambda) is then a mean-preserving
spread of Law(X), a smaller set than the W_2 ball of radius sqrt(P).

Quantiles.  To raise the alpha-quantile above t, mass F(t-) - alpha below t must reach t.  An atom (x, w),
x < t, that sends mass m to t must send the rest to z = (w x - m t)/(w - m) (mean preservation, one point
by Jensen), at variance cost (t-x)^2 w m/(w-m).  The cheapest lift is the separable convex programme
    min sum_i (t-x_i)^2 w_i m_i/(w_i-m_i)   s.t.  sum_i m_i = F(t-) - alpha,
with KKT solution m_i = w_i (1 - (t-x_i)/theta)_+ and value
    C(t) = sum_{0 < t-x_i < theta} w_i (t-x_i) (theta - (t-x_i)),
theta solving sum_{x_i < t} w_i (1 - (t-x_i)/theta)_+ = F(t-) - alpha.  The sharp upper endpoint is
U = sup{t : C(t) <= P}; the lower endpoint is symmetric.  Near a quantile with density f the shift is
(3P/(4f))^{1/3}, a factor 4^{1/3} below the W_2 endpoint (3P/f)^{1/3}; for a degenerate X it is
sqrt(P alpha/(1-alpha)) against sqrt(P/(1-alpha)).

Both endpoints are monotone in the first-order stochastic order of Law(X) (comonotone coupling of the
base laws; the spread is carried along), so a DKW band is pushed through by evaluating U at the upper
envelope (bottom mass moved to +inf) and L at the lower envelope (top mass moved to -inf).  Atoms beyond
the candidate t never move, so the position of the moved mass does not matter.

A grid-measurable perturbation E (Lambda = X + E + R with E[R | G] = 0, ||E||_2 <= delta) is absorbed by
a level shift: q_alpha(Lambda) <= U_{alpha+beta} + delta/sqrt(beta) (Chebyshev), and symmetrically.

Expected shortfall.  X <=_cx Lambda gives ES_a(X) <= ES_a(Lambda); subadditivity and the mean-variance
bound ES_a(R) <= E R + sd(R) sqrt(a/(1-a)) give ES_a(Lambda) <= ES_a(X) + E R + sd(R) sqrt(a/(1-a)).
"""
from __future__ import annotations

import math

import numpy as np

from .transfer import dkw_delta, es_bracket, law_from_samples


class _Atoms:
    """Sorted finite atoms with prefix sums (missing mass sits at +inf or -inf, never moved)."""

    def __init__(self, x, w):
        o = np.argsort(x, kind="stable")
        self.x = np.asarray(x, float)[o]
        self.w = np.asarray(w, float)[o]
        self.W = np.concatenate([[0.0], np.cumsum(self.w)])
        self.S = np.concatenate([[0.0], np.cumsum(self.w * self.x)])
        self.S2 = np.concatenate([[0.0], np.cumsum(self.w * self.x * self.x)])

    def window(self, a, b):
        """Sums over atoms with a < x < b."""
        ia = int(np.searchsorted(self.x, a, side="right"))
        ib = int(np.searchsorted(self.x, b, side="left"))
        if ib <= ia:
            return 0.0, 0.0, 0.0
        return self.W[ib] - self.W[ia], self.S[ib] - self.S[ia], self.S2[ib] - self.S2[ia]

    def mass_below(self, t, strict=True):
        i = int(np.searchsorted(self.x, t, side="left" if strict else "right"))
        return float(self.W[i])


def _solve_theta(mass_fn, need, start):
    lo, hi = 0.0, max(start, 1e-300)
    while mass_fn(hi) < need:
        lo, hi = hi, 2.0 * hi
        if hi > 1e300:
            return math.inf
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mass_fn(mid) < need:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-15 * hi:
            break
    return hi


def _cost_up(A, t, alpha, bottom=0.0):
    need = bottom + A.mass_below(t, strict=True) - alpha
    if need <= 0.0:
        return 0.0
    span = t - A.x[0] if A.x.size else 1.0

    def mass(theta):
        Wn, Sn, _ = A.window(t - theta, t)
        return Wn - (t * Wn - Sn) / theta

    theta = _solve_theta(mass, need, max(span, 1e-12))
    if not math.isfinite(theta):
        return math.inf
    Wn, Sn, S2n = A.window(t - theta, t)
    return theta * (t * Wn - Sn) - (t * t * Wn - 2.0 * t * Sn + S2n)


def _cost_down(A, t, alpha, bottom=0.0):
    need = alpha - (bottom + A.mass_below(t, strict=False))
    if need <= 0.0:
        return 0.0
    span = A.x[-1] - t if A.x.size else 1.0

    def mass(theta):
        Wn, Sn, _ = A.window(t, t + theta)
        return Wn - (Sn - t * Wn) / theta

    theta = _solve_theta(mass, need, max(span, 1e-12))
    if not math.isfinite(theta):
        return math.inf
    Wn, Sn, S2n = A.window(t, t + theta)
    return theta * (Sn - t * Wn) - (S2n - 2.0 * t * Sn + t * t * Wn)


def _quantile(A, alpha, bottom=0.0):
    c = bottom + np.cumsum(A.w)
    i = int(np.searchsorted(c, alpha - 1e-15, side="left"))
    return float(A.x[min(i, A.x.size - 1)])


def mart_upper(x, w, alpha, P, bottom=0.0, iters=200):
    """sup q_alpha(Lambda) over mean-preserving spreads of sum w_i delta_{x_i} (+ mass `bottom` at -inf)
    with E(Lambda - X)^2 <= P.  Mass 1 - bottom - sum(w) is understood to sit at +inf."""
    # The upper DKW envelope can put more than 1-alpha mass at +infinity.
    # In that case no finite upper certificate exists (also avoids an endless expansion).
    if bottom + float(np.sum(w)) < alpha or (P > 0 and bottom + float(np.sum(w)) <= alpha):
        return math.inf
    A = _Atoms(x, w)
    q = _quantile(A, alpha, bottom)
    if P <= 0:
        return q
    lo, hi = q, q + math.sqrt(P / max(1.0 - alpha, 1e-300)) + 1e-12
    while _cost_up(A, hi, alpha, bottom) <= P:
        lo, hi = hi, hi + 2.0 * (hi - q)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _cost_up(A, mid, alpha, bottom) <= P:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-13 * max(1.0, abs(hi)):
            break
    return lo


def mart_lower(x, w, alpha, P, bottom=0.0, iters=200):
    """inf q_alpha(Lambda) over the same set (mass `bottom` at -inf is counted, never moved)."""
    if bottom >= alpha:
        return -math.inf
    A = _Atoms(x, w)
    q = _quantile(A, alpha, bottom)
    if P <= 0:
        return q
    lo, hi = q - math.sqrt(P / max(alpha, 1e-300)) - 1e-12, q
    while _cost_down(A, lo, alpha, bottom) <= P:
        lo, hi = lo - 2.0 * (q - lo), lo
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _cost_down(A, mid, alpha, bottom) <= P:
            hi = mid
        else:
            lo = mid
        if hi - lo <= 1e-13 * max(1.0, abs(lo)):
            break
    return hi


def _envelopes(samples, d):
    """DKW envelopes as atoms: upper = bottom mass d moved to +inf; lower = top mass d moved to -inf."""
    v = np.sort(np.asarray(samples, float))
    K = v.size
    w = np.full(K, 1.0 / K)
    c = np.cumsum(w)
    # upper envelope: remove mass d from the bottom
    wu = np.clip(np.minimum(c, 1.0) - np.maximum(c - w, d), 0.0, None)
    # lower envelope: remove mass d from the top
    wl = np.clip(np.minimum(c, 1.0 - d) - (c - w), 0.0, None)
    return v, wu, wl


def certify_quantile_martingale(samples=None, alpha=0.95, P=0.0, dkw_a=None, delta=0.0, beta=None,
                                law=None):
    """Martingale-sharp alpha-quantile bracket for Lambda = X + E + R (E[R|G]=0, E R^2 <= P, ||E|| <= delta).

    `samples` (with dkw_a) or a deterministic discrete `law` (values, probs) of X.  Returns a dict with
    lower, upper, center, label; with dkw_a the statement holds with probability >= 1 - dkw_a.

    The level-shift bound holds for every beta > 0 simultaneously (same event), so with delta > 0 the
    endpoints are optimised over a grid of beta values (min of the upper, max of the lower endpoints)."""
    if delta <= 0:
        betas = [0.0]
    elif beta is not None:
        betas = [beta]
    else:
        betas = [(1 - alpha) * c for c in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.2)]
    if law is not None:
        x, w = np.asarray(law.values, float), np.asarray(law.probs, float)
        up = lambda a: mart_upper(x, w, a, P)
        lo = lambda a: mart_lower(x, w, a, P)
        center, label, d = law.quantile(alpha), "certified-numerical", 0.0
    else:
        d = dkw_delta(len(samples), dkw_a) if dkw_a else 0.0
        v, wu, wl = _envelopes(samples, d)
        up = lambda a: mart_upper(v, wu, a, P)
        lo = lambda a: mart_lower(v, wl, a, P, bottom=d)
        center = float(np.quantile(samples, alpha))
        label = f"statistical(1-{dkw_a:g})" if dkw_a else "deterministic"
    best_U, best_L, bU, bL = math.inf, -math.inf, None, None
    for b in betas:
        shift = delta / math.sqrt(b) if b > 0 else 0.0
        U = up(min(alpha + b, 1 - 1e-12)) + shift
        L = lo(max(alpha - b, 1e-12)) - shift
        if U < best_U:
            best_U, bU = U, b
        if L > best_L:
            best_L, bL = L, b
    return {"lower": best_L, "upper": best_U, "center": center, "label": label, "beta": (bL, bU), "dkw": d}


def es_bracket_martingale(samples, alpha, P, second_moment_X=None, dkw_a=None, delta=0.0, mean_E=None, law=None):
    """ES_alpha(Lambda) bracket from convex order: [ES(X) + E[E] - delta k, ES(X) + E[E] + sqrt(P+delta^2) k],
    k = sqrt(alpha/(1-alpha)); ES(X) from the DKW band with the exact second moment of X (samples), or from a
    deterministic discrete `law`.  If the mean of the perturbation E is unknown, |E[E]| <= delta is used."""
    k = math.sqrt(alpha / (1.0 - alpha))
    if law is None:
        law = law_from_samples(np.asarray(samples, float))
        b = es_bracket(law, alpha, 0.0, second_moment_Y=second_moment_X, dkw_a=dkw_a) if dkw_a else \
            {"lower": law.es_upper_tail(alpha), "upper": law.es_upper_tail(alpha)}
        label = f"statistical(1-{dkw_a:g})" if dkw_a else "deterministic"
    else:
        es = law.es_upper_tail(alpha)
        b, label = {"lower": es, "upper": es}, "certified-numerical"
    m_lo, m_hi = (-delta, delta) if mean_E is None else (mean_E, mean_E)
    return {"lower": b["lower"] + m_lo - delta * k, "upper": b["upper"] + m_hi + math.sqrt(P + delta ** 2) * k,
            "center": law.es_upper_tail(alpha), "label": label}
