"""Independent continuum reference for the v4 audit (no Arb, no shared engine code).

The delivery-window Gaussian field is built from its covariance evaluated by adaptive quadrature
(not by the Pfaff hypergeometric identity used in the certified engine), factorised by Cholesky and
sampled with scrambled Sobol points.  Two delivery meshes are run, so the discretisation direction is
visible.  Output: results/v4_independent_check.json (status: diagnostic).
"""
from pathlib import Path
import json
import numpy as np
from scipy import integrate, optimize, special, stats

ROOT = Path(__file__).resolve().parents[1]


def cov(u, v, T, H):
    """Cov(W(k_u), W(k_v)) = int_0^T (u-s)^{H-1/2} (v-s)^{H-1/2} ds, by adaptive quadrature."""
    p = H - 0.5
    f = lambda s: (u - s) ** p * (v - s) ** p
    val, _ = integrate.quad(f, 0.0, T, epsabs=1e-13, epsrel=1e-13, limit=400)
    return val


def field(T, D, H, n):
    """Gauss-Legendre delivery nodes on [T, T+D] with weights, and the Cholesky factor of their covariance."""
    x, w = np.polynomial.legendre.leggauss(n)
    u = T + D * (x + 1) / 2
    w = w / 2                              # (1/D) * int_T^{T+D} . du under the GL change of variable
    C = np.array([[cov(a, b, T, H) for b in u] for a in u])
    C = (C + C.T) / 2
    # The delivery covariance is numerically low rank: factorise by eigenvalues and record what is dropped.
    lam, vec = np.linalg.eigh(C)
    ok = lam > lam.max() * 1e-14
    L = vec[:, ok] * np.sqrt(lam[ok])
    return u, w, np.sqrt(np.diag(C)), L, int(ok.sum()), float(lam[~ok].clip(min=0).sum() / lam.sum())


def black76(F, K, T, C):
    g = lambda v: F * special.ndtr(np.log(F / K) / (v * np.sqrt(T)) + v * np.sqrt(T) / 2) \
        - K * special.ndtr(np.log(F / K) / (v * np.sqrt(T)) - v * np.sqrt(T) / 2) - C
    return optimize.brentq(g, 1e-6, 8.0)


def run(T, D, H, comps, xi, strikes, n, power=16, reps=8, seed=20261124):
    u, w, sd, L, rank, dropped = field(T, D, H, n)
    out = []
    for rep in range(reps):
        gen = stats.qmc.Sobol(d=rank, scramble=True, seed=seed + rep)
        tot = np.zeros(1 + len(strikes))
        for _ in range(2 ** (power - 12)):
            z = special.ndtri(np.clip(gen.random(4096), 1e-16, 1 - 1e-16))
            g = z @ L.T                                     # Gaussian field at the delivery nodes
            lam = sum(lw * np.exp(a * g - 0.5 * a * a * sd ** 2) for a, lw in comps)
            vix = 100 * np.sqrt((lam * xi(u) * w).sum(axis=1))
            tot += np.array([vix.mean(), *[np.maximum(vix - k, 0).mean() for k in strikes]])
        out.append(tot / 2 ** (power - 12))
    out = np.array(out)
    mean, se = out.mean(axis=0), out.std(axis=0, ddof=1) / np.sqrt(reps)
    return dict(nodes=n, power=power, reps=reps, rank=rank, dropped_variance_share=dropped,
                mean=mean.tolist(), se=se.tolist())


def main():
    o = json.loads((ROOT / "results/vix_options_smile_suite.json").read_text())
    cases = []
    for key, T, D, H, comps, xi in [
        ("bdg2023_fig6", 0.25, 1 / 12, 0.10, [(1.0, 1.0)], (lambda u: 0.235 ** 2 * np.ones_like(u))),
        ("mixed_rbergomi_smile", 0.25, 1 / 12, 0.10, [(1.4, 0.3), (0.7, 0.7)], (lambda u: 0.235 ** 2 * np.ones_like(u))),
    ]:
        z = o[key]
        strikes = [t["K_decimal"] * 100 for t in z["strikes"]]  # strikes in volatility points
        runs = [run(T, D, H, comps, xi, strikes, n) for n in (48, 96)]
        arb = dict(future=[z["condvar_future_L"], z["condvar_future_U"]],
                   calls=[[t["call_L_volpts"], t["call_U_volpts"]] for t in z["strikes"]],
                   iv=[[t["iv_L_pct"], t["iv_U_pct"]] for t in z["strikes"]])
        fine = runs[-1]
        inside = []
        for i, b in enumerate([arb["future"], *arb["calls"]]):
            m, s = fine["mean"][i], fine["se"][i]
            inside.append(bool(b[0] - 3 * s <= m <= b[1] + 3 * s))
        iv_mc = [100 * black76(fine["mean"][0], k, T, c) for k, c in zip(strikes, fine["mean"][1:])]
        cases.append(dict(case=key, strikes=strikes, runs=runs, arb=arb, inside=inside,
                          iv_independent_pct=iv_mc,
                          iv_inside=[bool(b[0] - 0.05 <= v <= b[1] + 0.05) for v, b in zip(iv_mc, arb["iv"])]))
    res = dict(status="diagnostic", method="adaptive-quadrature covariance + Cholesky + scrambled Sobol",
               cases=cases, all_inside=all(all(c["inside"]) for c in cases))
    (ROOT / "results/v4_independent_check.json").write_text(json.dumps(res, indent=1))
    for c in cases:
        print(c["case"], "inside:", sum(c["inside"]), "/", len(c["inside"]),
              "| future mean %.5f vs Arb [%.5f, %.5f]" % (c["runs"][-1]["mean"][0], *c["arb"]["future"]))
    print("ALL INSIDE:", res["all_inside"])


if __name__ == "__main__":
    main()
