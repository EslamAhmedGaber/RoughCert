"""Centred Gaussian fields Y_s = W(e_s) on [0, T], specified by their covariance.

Only the covariance R(s, u) is used downstream.  Every field declares the
H-range for which its closed form is valid; nothing here is approximated.
"""
from __future__ import annotations

import numpy as np
from scipy.special import hyp2f1


class GaussianField:
    """Base class.  Subclasses implement ``R(s, u)`` (vectorised, broadcasting)."""

    name = "generic"
    H = None
    T = 1.0

    def R(self, s, u):  # pragma: no cover - interface
        raise NotImplementedError

    def var(self, s):
        s = np.asarray(s, dtype=float)
        return self.R(s, s)

    def singular_points(self):
        """Points of [0, T] where R(., u) may lose smoothness for every u (e.g. 0)."""
        return [0.0]

    def diag_exponent(self):
        """Exponent q with ||e_s - e_u||^2 ~ |s-u|^q near the diagonal (None if smooth)."""
        return None

    def describe(self):
        return {"field": self.name, "H": self.H, "T": self.T}


class FBM(GaussianField):
    """Standard fractional Brownian motion, any 0 < H < 1."""

    def __init__(self, H, T=1.0, scale=1.0):
        if not 0.0 < H < 1.0:
            raise ValueError("fBm requires 0 < H < 1")
        self.H, self.T, self.scale = float(H), float(T), float(scale)
        self.name = "fBm"

    def R(self, s, u):
        s = np.asarray(s, dtype=float)
        u = np.asarray(u, dtype=float)
        h2 = 2.0 * self.H
        return self.scale ** 2 * 0.5 * (np.abs(s) ** h2 + np.abs(u) ** h2 - np.abs(s - u) ** h2)

    def diag_exponent(self):
        return 2.0 * self.H


def _rl_hyp(H, x, terms=64):
    """2F1(1, 1/2-H; H+3/2; x) on [0, 1], vectorised: power series for x <= 1/2 and, for x > 1/2, the
    connection formula (c-a-b = 2H not an integer)
        F(x) = (H+1/2)/(2H) 2F1(1, 1/2-H; 1-2H; 1-x) + G(H+3/2) G(-2H)/G(1/2-H) (1-x)^{2H} x^{-(H+1/2)},
    both series converging at rate <= 1/2 (64 terms: below 1e-18)."""
    from math import gamma
    x = np.asarray(x, float)
    out = np.empty(x.shape)
    lo = x <= 0.5
    if np.any(lo):
        z = x[lo]
        term, acc = np.ones(z.shape), np.ones(z.shape)
        for k in range(terms):
            term = term * (0.5 - H + k) / (H + 1.5 + k) * z
            acc += term
        out[lo] = acc
    hi = ~lo
    if np.any(hi):
        z = 1.0 - x[hi]
        term, acc = np.ones(z.shape), np.ones(z.shape)
        for k in range(terms):
            term = term * (1.0 + k) * (0.5 - H + k) / ((1.0 - 2.0 * H + k) * (k + 1.0)) * z
            acc += term
        B = gamma(H + 1.5) * gamma(-2.0 * H) / gamma(0.5 - H)
        out[hi] = (H + 0.5) / (2.0 * H) * acc + B * z ** (2.0 * H) * x[hi] ** (-(H + 0.5))
    return out


class RiemannLiouville(GaussianField):
    """Riemann--Liouville fBm  int_0^s (s-r)^(H-1/2) dW_r * sqrt(2H)  (R(s,s) = s^{2H})."""

    def __init__(self, H, T=1.0, scale=1.0):
        if not 0.0 < H < 1.0:
            raise ValueError("RL requires 0 < H < 1")
        self.H, self.T, self.scale = float(H), float(T), float(scale)
        self.name = "RL"
        # fast vectorised hypergeometric away from H = 1/2 (where c-a-b is an integer); scipy otherwise
        self._fast = abs(self.H - 0.5) > 0.02

    def R(self, s, u):
        S = np.asarray(s, dtype=float)
        U = np.asarray(u, dtype=float)
        S, U = np.broadcast_arrays(S, U)
        lo, hi = np.minimum(S, U), np.maximum(S, U)
        m = lo > 0
        x = np.clip(np.where(m, lo / np.where(hi > 0, hi, 1.0), 0.0), 0.0, 1.0)
        F = _rl_hyp(self.H, x) if self._fast else hyp2f1(1.0, 0.5 - self.H, self.H + 1.5, x)
        with np.errstate(divide="ignore", invalid="ignore"):
            val = (2 * self.H / (self.H + 0.5)) * lo ** (self.H + 0.5) * hi ** (self.H - 0.5) * F
        out = np.where(m, val, 0.0)
        return self.scale ** 2 * out

    def diag_exponent(self):
        return 2.0 * self.H


class Brownian(FBM):
    def __init__(self, T=1.0, scale=1.0):
        super().__init__(0.5, T, scale)
        self.name = "Brownian"


class FractionalOU(GaussianField):
    """Stationary-start-free fractional OU  Y_s = int_0^s e^{-lam (s-r)} dB^H_r (Wiener integral).

    For H = 1/2 the covariance is closed form; for H != 1/2 it is evaluated by the
    CovFourier backend (see representation.CovFourierProjection), so this class only
    serves the Brownian case directly.
    """

    def __init__(self, lam, T=1.0, sigma=1.0):
        self.lam, self.T, self.sigma, self.H = float(lam), float(T), float(sigma), 0.5
        self.name = "OU(H=1/2)"

    def R(self, s, u):
        s = np.asarray(s, dtype=float)
        u = np.asarray(u, dtype=float)
        lo, hi = np.minimum(s, u), np.maximum(s, u)
        lam = self.lam
        return self.sigma ** 2 * np.exp(-lam * (hi - lo)) * (1 - np.exp(-2 * lam * lo)) / (2 * lam)

    def diag_exponent(self):
        return 1.0


class CallableField(GaussianField):
    """User-supplied covariance.  The user declares H-range/regularity information."""

    def __init__(self, R, T=1.0, name="callable", H=None, diag_exponent=None, singular=(0.0,)):
        self._R, self.T, self.name, self.H = R, float(T), name, H
        self._q, self._sing = diag_exponent, list(singular)

    def R(self, s, u):
        return self._R(np.asarray(s, dtype=float), np.asarray(u, dtype=float))

    def diag_exponent(self):
        return self._q

    def singular_points(self):
        return self._sing
