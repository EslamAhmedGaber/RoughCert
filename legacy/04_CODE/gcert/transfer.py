"""From a coupled error radius to quantile and expected-shortfall statements.

Primary transfer (classical; Liu--Mao--Wang--Wei 2022, Han--Liu 2026):
  if W_p(L(X), L(Y)) <= eps then  L_p(alpha) <= Q_X(alpha) <= U_p(alpha), where
  int_alpha^1 (U - Q_Y(t))_+^p dt = eps^p,   int_0^alpha (Q_Y(t) - L)_+^p dt = eps^p.
ES transfer (classical Hoelder form): |ES_a(X) - ES_a(Y)| <= W_2 / sqrt(1 - a).

The law of Y is supplied as a *discrete quantile representation* (values sorted with
probability masses).  Three ways of obtaining it are provided:
  * ``law_from_map``: Y = G(Z), Z ~ N(0, 1), on a stratified probability grid, with a
    computed discretisation radius delta (added to eps by the triangle inequality in W_p);
  * ``law_from_samples`` + DKW: an empirical law with a simultaneous confidence band
    (Massart's constant); the band is pushed through the endpoint maps, which are monotone
    in the quantile function, giving brackets valid with probability >= 1 - a;
  * exact atoms.
Labels: the bracket is CERTIFIED-NUMERICAL when eps is an exact identity evaluated by audited
quadrature and the surrogate law is deterministic; STATISTICAL(1-a) when the DKW band is used.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtri, ndtr


@dataclass
class DiscreteLaw:
    values: np.ndarray       # sorted ascending
    probs: np.ndarray        # nonnegative, sum to 1
    radius_disc: float = 0.0  # W_p discretisation radius already included where relevant
    label: str = "deterministic"

    @property
    def cdf_right(self):
        return np.cumsum(self.probs)

    def quantile(self, a):
        c = self.cdf_right
        i = int(np.searchsorted(c, a - 1e-15, side="left"))
        return float(self.values[min(i, self.values.size - 1)])

    def mean(self):
        return float(np.sum(self.values * self.probs))

    def es_upper_tail(self, a):
        """(1-a)^{-1} int_a^1 Q(t) dt for the discrete law (upper-tail ES)."""
        c = self.cdf_right
        lo = np.concatenate([[0.0], c[:-1]])
        ell = np.clip(np.minimum(c, 1.0) - np.maximum(lo, a), 0.0, None)
        return float(np.sum(ell * self.values) / (1.0 - a))


def probability_grid(n_bulk=1 << 14, tail=1e-3, depth=1e-14, per_decade=24):
    """Cell edges in (0,1): uniform in [tail, 1-tail], geometric towards 0 and 1 down to depth."""
    bulk = np.linspace(tail, 1.0 - tail, n_bulk + 1)
    k = int(per_decade * math.log10(tail / depth))
    lo = tail * np.logspace(-math.log10(tail / depth), 0.0, k + 1)[:-1]
    edges = np.concatenate([[0.0], lo, bulk, (1.0 - lo)[::-1], [1.0]])
    return edges


def law_from_map(G, p=2, n_bulk=1 << 14, sub=8, tail_check=True):
    """Law of Y = G(Z) on a tail-refined stratified probability grid, with a W_p radius.

    Cell j = (t_j, t_{j+1}] in probability, value G(z_j) at the cell's probability midpoint.
    Coupling Z with its cell value bounds W_p(L(G(Z)), grid law)^p by
    sum_j E[|G(Z) - G(z_j)|^p ; Z in cell j], estimated with ``sub`` stratified sub-points per cell
    (numerical evaluation); the two outermost cells (probability <= 1e-14 each) are integrated
    directly over the Gaussian tail.
    """
    edges = probability_grid(n_bulk)
    t_mid = 0.5 * (edges[:-1] + edges[1:])
    prob = np.diff(edges)
    z = ndtri(t_mid)
    y = G(z)
    order = np.argsort(y, kind="mergesort")
    vals, probs = y[order], prob[order]
    # sub-sampling inside cells (interior cells only)
    inner = np.arange(1, z.size - 1)
    fr = (np.arange(sub) + 0.5) / sub
    tt = edges[inner][:, None] + prob[inner][:, None] * fr[None, :]
    gg = G(ndtri(tt.ravel())).reshape(inner.size, sub)
    dp = float(np.sum(prob[inner] * np.mean(np.abs(gg - y[inner][:, None]) ** p, axis=1)))
    if tail_check:
        for edge, sgn in ((0, -1.0), (z.size - 1, 1.0)):
            zb = ndtri(edges[1]) if edge == 0 else ndtri(edges[-2])
            zt = zb + sgn * np.linspace(0.0, 14.0, 6001)
            dens = np.exp(-0.5 * zt ** 2) / math.sqrt(2 * math.pi)
            dp += abs(float(np.trapz(np.abs(G(zt) - y[edge]) ** p * dens, zt)))
    return DiscreteLaw(vals, probs, radius_disc=dp ** (1.0 / p), label="deterministic-map")


def law_from_samples(samples):
    v = np.sort(np.asarray(samples, float))
    return DiscreteLaw(v, np.full(v.size, 1.0 / v.size), label="empirical")


def dkw_delta(K, a):
    """Massart-DKW: P(sup|F_K - F| > d) <= 2 exp(-2 K d^2)."""
    return math.sqrt(math.log(2.0 / a) / (2.0 * K))


def _upper_endpoint(vals, probs, alpha, eps, p, shift=0.0):
    """Solve int_alpha^1 (U - Q(t + shift))_+^p dt = eps^p for a discrete law (shift >= 0)."""
    c = np.cumsum(probs)
    if eps <= 0:
        level = alpha + shift
        if level > 1.0:
            return math.inf
        i = int(np.searchsorted(c, level - 1e-15, side="left"))
        return float(vals[min(i, vals.size - 1)])
    lo = np.concatenate([[0.0], c[:-1]])
    # quantile function evaluated at t + shift: atom i covers t in (lo_i - shift, c_i - shift]
    a_i = np.maximum(lo - shift, alpha)
    b_i = np.minimum(c - shift, 1.0)
    ell = np.clip(b_i - a_i, 0.0, None)
    keep = ell > 0
    v, l = vals[keep], ell[keep]
    if l.sum() <= 0:
        return math.inf

    def F(U):
        return float(np.sum(l * np.clip(U - v, 0.0, None) ** p)) - eps ** p
    lo_U = float(v[0])
    hi_U = float(v[-1]) + eps / max(l.sum(), 1e-300) ** (1.0 / p) + 1.0
    while F(hi_U) < 0:
        hi_U = 2 * hi_U + 1.0
    return brentq(F, lo_U, hi_U, xtol=1e-14, rtol=1e-13)


def _lower_endpoint(vals, probs, alpha, eps, p, shift=0.0):
    """Solve int_0^alpha (Q(t - shift) - L)_+^p dt = eps^p (shift >= 0)."""
    if eps <= 0:
        level = alpha - shift
        if level <= 0.0:
            return -math.inf
        i = int(np.searchsorted(np.cumsum(probs), level - 1e-15, side="left"))
        return float(vals[min(i, vals.size - 1)])
    neg = DiscreteLaw(-vals[::-1], probs[::-1])
    return -_upper_endpoint(neg.values, neg.probs, 1.0 - alpha, eps, p, shift)


@dataclass
class QuantileCertificate:
    alpha: float
    lower: float
    upper: float
    center: float
    eps: dict
    label: str
    notes: str = ""

    @property
    def width(self):
        return self.upper - self.lower


def certify_quantile(law, alpha, eps_by_p, support=None, dkw_a=None):
    """Sharp Wasserstein bracket for Q_X(alpha), intersected over the supplied orders p.

    eps_by_p: {p: radius} with W_p(L(X), L(Y)) <= radius (law's radius_disc is added).
    dkw_a: if given, the law is empirical and the bracket holds with probability >= 1 - dkw_a.
    """
    L, U = -math.inf, math.inf
    shift = 0.0
    label = "certified-numerical"
    if dkw_a is not None:
        shift = dkw_delta(law.values.size, dkw_a)
        label = f"statistical(1-{dkw_a:g})"
    used = {}
    for p, eps in eps_by_p.items():
        e = eps + law.radius_disc
        used[p] = e
        U = min(U, _upper_endpoint(law.values, law.probs, alpha, e, p, shift))
        L = max(L, _lower_endpoint(law.values, law.probs, alpha, e, p, shift))
    if support is not None:
        L, U = max(L, support[0]), min(U, support[1])
    return QuantileCertificate(alpha, L, U, law.quantile(alpha), used, label)


def es_bracket(law, alpha, eps2, second_moment_Y=None, dkw_a=None):
    """Upper-tail ES_alpha(X) bracket from W_2 <= eps2 and the surrogate law.

    Deterministic law: ES(Y) +- (eps2 + delta)/sqrt(1-alpha).
    Empirical law: set d = min(1, DKW allowance). Integrate Q_K(t-d) on
    [max(alpha,d),1] and Q_K(t+d) on [alpha,max(alpha,1-d)]. Empty
    intervals contribute zero. Subtract sqrt(E Y^2 * max(d-alpha,0))
    below and add sqrt(E Y^2 * min(d,1-alpha)) above, then divide by
    1-alpha. These Cauchy--Schwarz guards cover both missing tails,
    including signed outputs and small samples. A valid upper bound
    on the second moment may replace its exact value.
    """
    rad = (eps2 + law.radius_disc) / math.sqrt(1.0 - alpha)
    if dkw_a is None:
        es = law.es_upper_tail(alpha)
        return {"lower": es - rad, "upper": es + rad, "center": es, "label": "certified-numerical"}
    if second_moment_Y is None:
        raise ValueError("DKW ES bracket needs the exact second moment of the surrogate")
    if not math.isfinite(second_moment_Y) or second_moment_Y < 0:
        raise ValueError("The second-moment bound must be finite and nonnegative")
    d = min(1.0, dkw_delta(law.values.size, dkw_a))
    c = np.cumsum(law.probs)
    lo_c = np.concatenate([[0.0], c[:-1]])
    v = law.values
    # lower: Q_K(t - d) on (alpha, 1]
    a_i = np.maximum(lo_c + d, alpha)
    b_i = np.minimum(c + d, 1.0)
    lo_int = float(np.sum(np.clip(b_i - a_i, 0, None) * v))
    lo_int -= math.sqrt(second_moment_Y) * math.sqrt(max(d - alpha, 0.0))
    # upper: Q_K(t + d) on (alpha, 1 - d]
    a_j = np.maximum(lo_c - d, alpha)
    b_j = np.minimum(c - d, 1.0 - d)
    hi_int = (float(np.sum(np.clip(b_j - a_j, 0, None) * v))
              + math.sqrt(second_moment_Y) * math.sqrt(min(d, 1.0 - alpha)))
    es_lo, es_hi = lo_int / (1 - alpha), hi_int / (1 - alpha)
    return {"lower": es_lo - rad, "upper": es_hi + rad, "center": law.es_upper_tail(alpha),
            "label": f"statistical(1-{dkw_a:g})"}


def local_density_gate(Mp, m_alpha, delta, p=2):
    """Paper II/III/V sufficient gate: |Q_X - Q_Y| <= B_p if B_p <= delta (else refuse)."""
    c_p = (1.0 + 1.0 / p) * p ** (1.0 / (p + 1.0))
    B = c_p * (Mp / m_alpha) ** (1.0 / (p + 1.0))
    return {"B": B, "issued": bool(B <= delta), "c_p": c_p}
