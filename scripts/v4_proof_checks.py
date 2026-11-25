"""First-principles numerical checks of the stated theorems (v4 audit, status: diagnostic).

Every check recomputes a claim from its own definition, independently of the Arb engines:
  T1   Mehler ledger: closed-form Var(Lambda) and residual share against 2-D Gaussian quadrature.
  T3a  Brownian boundary of Theorem 3: n^2 M(mu_n) -> xi0^2 T^2 (e^{eta^2 T} - 1)/3 (exact second moments).
  T3b  rough case: fitted slope of M(mu_n) against the predicted n^{-(1+2H)} (N=4096 reference grid).
  T4   fourth-order relative width law of the symmetric two-atom tilted control.
  P4c  Minkowski perturbation bound tau_{r,i} on random admissible cells.
  P4e  Black-76 rectangular corner inversion (monotone in F and sigma).
Output: results/v4_proof_checks.json
"""
from pathlib import Path
import json
import random

import mpmath as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
mp.mp.dps = 40


def mehler_ledger(a=1.2, rho=mp.mpf("0.6"), xi0=mp.mpf("0.0552")):
    """Lambda = (X1 + X2)/2 with X_i Wick exponentials of a correlated standard Gaussian pair."""
    wick = lambda z: xi0 * mp.e ** (a * z - a * a / 2)
    var_closed = xi0 ** 2 * (2 * mp.e ** (a * a) + 2 * mp.e ** (a * a * rho)) / 4 - xi0 ** 2
    f = lambda z1, z2: ((wick(z1) + wick(rho * z1 + mp.sqrt(1 - rho ** 2) * z2)) / 2 - xi0) ** 2 \
        * mp.npdf(z1) * mp.npdf(z2)
    var_quad = mp.quad(lambda z1: mp.quad(lambda z2: f(z1, z2), [-9, 0, 9]), [-9, 0, 9])
    g = (1 + rho) / mp.sqrt(2 + 2 * rho)                      # loading on the barycentre direction
    var_cond = xi0 ** 2 * mp.e ** (a * a * g * g) - xi0 ** 2
    return dict(var_closed=float(var_closed), var_quadrature=float(var_quad),
                rel_diff=float(abs(var_closed - var_quad) / var_closed),
                residual_share=float((var_closed - var_cond) / var_closed))


def _M(cov, w_fine, w_coarse, idx, xi0, eta):
    """||Lambda_fine - Lambda_coarse||_2^2 for Wick exponentials: E[f(s)f(u)] = xi0^2 exp(eta^2 R(s,u))."""
    G = xi0 ** 2 * np.exp(eta ** 2 * cov)
    a = w_fine
    b = np.zeros_like(w_fine)
    np.add.at(b, idx, w_coarse)
    d = a - b
    return float(d @ G @ d)


def brownian_boundary(xi0=0.235 ** 2, eta=1.9, T=1.0, ns=(256, 512, 1024, 2048, 4096)):
    """Exact M(mu_n) for Brownian covariance and left-point nodes: no reference grid is involved.

    E[f(s)f(u)] = xi0^2 exp(eta^2 min(s,u)), so every term of ||Lambda - Lambda_n||_2^2 is closed form.
    """
    c = eta ** 2
    I2 = 2 * ((T * (np.exp(c * T) - 1) / c) - (T * np.exp(c * T) / c - (np.exp(c * T) - 1) / c ** 2))
    I1 = lambda x: (np.exp(c * x) - 1) / c + (T - x) * np.exp(c * x)
    out = []
    for n in ns:
        h = T / n
        nodes = np.arange(n) * h                               # left-point sampling
        M = xi0 ** 2 * (I2 - 2 * h * I1(nodes).sum() + h * h * np.exp(c * np.minimum.outer(nodes, nodes)).sum())
        out.append((n, float(M)))
    limit = xi0 ** 2 * T ** 2 * (np.exp(c * T) - 1) / 3
    return dict(eta=eta, xi0=xi0, T=T, values=out, predicted_limit=float(limit),
                n2M=[(n, v * n ** 2) for n, v in out],
                rel_gap=float(abs(out[-1][1] * ns[-1] ** 2 - limit) / limit))


def rough_rate(H, xi0=0.235 ** 2, eta=1.9, T=1.0, ns=(8, 16, 32, 64)):
    """fBm covariance; fine midpoint grid as the continuum reference."""
    N = 4096                                                   # fine midpoint reference grid
    s = (np.arange(N) + 0.5) * T / N
    cov = 0.5 * (s[:, None] ** (2 * H) + s[None, :] ** (2 * H) - np.abs(s[:, None] - s[None, :]) ** (2 * H))
    w_fine = np.full(N, T / N)
    out = []
    for n in ns:
        mid = (np.arange(n) + 0.5) * T / n
        idx = np.searchsorted(s, mid) - 1
        out.append((n, _M(cov, w_fine, np.full(n, T / n), idx, xi0, eta)))
    logs = np.log(np.array([v for _, v in out])); logn = np.log(np.array([n for n, _ in out]))
    slope = float(np.polyfit(logn, logs, 1)[0])
    return dict(H=H, values=out, fitted_slope=slope, predicted_slope=-(1 + 2 * H))


def two_atom(eps):
    width = 1 - 1 / mp.sqrt(mp.cosh(eps * eps))
    return dict(eps=float(eps), rel_width=float(width), over_eps4=float(width / eps ** 4))


def minkowski_tau(trials=400, seed=11):
    """e^{a^2 C}(e^{a^2(A-C)} + e^{a^2(B-C)} - 2) <= tau^2 with u = v = a^2 sqrt(Dmax) d_0."""
    rng = random.Random(seed)
    worst = -mp.inf
    for _ in range(trials):
        a = mp.mpf(rng.uniform(0.2, 2.0))
        A = mp.mpf(rng.uniform(0.01, 1.0)); B = A * mp.mpf(rng.uniform(0.8, 1.2))
        C = mp.sqrt(A * B) * mp.mpf(rng.uniform(0.80, 0.999))
        d0 = mp.sqrt(A + B - 2 * C)
        lhs = mp.e ** (a * a * C) * (mp.e ** (a * a * (A - C)) + mp.e ** (a * a * (B - C)) - 2)
        u = a * a * mp.sqrt(max(A, B)) * d0
        tau2 = mp.e ** (a * a * C) * (2 * mp.e ** u - 2)
        worst = max(worst, lhs - tau2)
    return dict(trials=trials, max_violation=float(worst), holds=bool(worst <= 0))


def black76_corners(trials=400, seed=5):
    rng = random.Random(seed)
    bad = 0
    for _ in range(trials):
        F = mp.mpf(rng.uniform(10, 40)); K = F * mp.mpf(rng.uniform(0.6, 1.6))
        T = mp.mpf(rng.uniform(0.05, 1.0)); sg = mp.mpf(rng.uniform(0.3, 2.5))
        eF = mp.mpf(rng.uniform(0, 0.01)); es = mp.mpf(rng.uniform(0, 0.01))
        def bs(f, v):
            d1 = mp.log(f / K) / (v * mp.sqrt(T)) + v * mp.sqrt(T) / 2
            return f * mp.ncdf(d1) - K * mp.ncdf(d1 - v * mp.sqrt(T))
        if not bs(F - eF, sg - es) <= bs(F, sg) <= bs(F + eF, sg + es):
            bad += 1
    return dict(trials=trials, monotone_violations=bad)


def main():
    res = dict(status="diagnostic", mpmath_dps=mp.mp.dps,
               T1_mehler_ledger=mehler_ledger(),
               T3a_brownian_boundary=brownian_boundary(),
               T3b_rough_rate=[rough_rate(h) for h in (0.1, 0.3)],
               T4_two_atom=[two_atom(mp.mpf(e)) for e in ("0.2", "0.1", "0.05", "0.01")],
               P4c_minkowski=minkowski_tau(),
               P4e_black76=black76_corners())
    (ROOT / "results/v4_proof_checks.json").write_text(json.dumps(res, indent=1))
    t1 = res["T1_mehler_ledger"]; t3 = res["T3a_brownian_boundary"]
    print("T1 closed form vs 2-D quadrature rel diff: %.2e" % t1["rel_diff"])
    print("T3a n^2 M: %s -> predicted %.6f (rel gap %.3f)" %
          ([round(v, 5) for _, v in t3["n2M"]], t3["predicted_limit"], t3["rel_gap"]))
    for r in res["T3b_rough_rate"]:
        print("T3b H=%.2f fitted slope %.3f vs predicted %.2f" % (r["H"], r["fitted_slope"], r["predicted_slope"]))
    print("T4 width/eps^4:", [round(r["over_eps4"], 6) for r in res["T4_two_atom"]])
    print("P4c max violation %.3e (holds=%s) | P4e violations %d" %
          (res["P4c_minkowski"]["max_violation"], res["P4c_minkowski"]["holds"],
           res["P4e_black76"]["monotone_violations"]))


if __name__ == "__main__":
    main()
