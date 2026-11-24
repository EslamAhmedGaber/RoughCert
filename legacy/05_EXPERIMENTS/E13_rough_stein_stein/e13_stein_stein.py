"""E13 driver (see PROTOCOL.md)."""
import json, math, sys, time
from pathlib import Path
import numpy as np
from scipy import integrate, optimize
LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB / "04_CODE"))
from gcert import RiemannLiouville, Measure, Polynomial, IdentityRepresentation, compute_ledger
from gcert.ledger import second_moment
from gcert.monitor import (ConditionalMonitor, ce_radius, quadratic_cond_mean, quadratic_cond_var, quadratic_delta,
                           cantelli_dominating, cantelli_rb, scarf_excess)
from gcert.martingale import certify_quantile_martingale
from gcert.transfer import law_from_samples, certify_quantile, dkw_delta

OUT = Path(__file__).parent
SIG0, NU, T, NDAY = 0.2, 0.1, 1.0 / 12.0, 21
ALPHA, K, R, A_Q, A_P = 0.99, 200_000, 50, 0.01, 0.005
HS = [float(x) for x in sys.argv[1:]] or [0.05, 0.10, 0.30]


def call_bracket(x, k, second_moment_x, d):
    """[lower, upper] for E (X - k)_+ from samples x on the DKW event (sup |F_K - F| <= d) (as E6b)."""
    v = np.sort(np.asarray(x, float))
    Kn = v.size
    lo_c, hi_c = np.arange(Kn) / Kn, np.arange(1, Kn + 1) / Kn
    g = np.maximum(v - k, 0.0)
    low = float(np.sum(g * np.clip(np.minimum(hi_c + d, 1.0) - np.maximum(lo_c + d, 0.0), 0, None)))
    up = float(np.sum(g * np.clip(np.minimum(hi_c - d, 1.0 - d) - np.maximum(lo_c - d, 0.0), 0, None)))
    return low, up + math.sqrt(d * second_moment_x)


# ------------------------------------------------------------------ exact law (KL + characteristic function)
def exact_law(fld, n=2500, J=1500):
    x, w = np.polynomial.legendre.leggauss(n)
    t, w = 0.5 * T * (x + 1), 0.5 * T * w
    sw = np.sqrt(w)
    lam, E = np.linalg.eigh(sw[:, None] * fld.R(t[:, None], t[None, :]) * sw[None, :])
    lam, E = np.clip(lam[::-1], 0, None), E[:, ::-1]
    c = sw @ E                                                    # int phi_j
    a, da = NU ** 2 * lam, SIG0 ** 2 * c ** 2                     # Lambda = sum a_j (Z_j + b_j)^2, da = a b^2
    return {"a": a[:J], "da": da[:J], "shift": float(np.sum(a[J:] + da[J:])),
            "rest_var": float(np.sum(2 * a[J:] ** 2 + 4 * a[J:] * da[J:])),
            "mean": float(np.sum(a + da)), "var": float(np.sum(2 * a ** 2 + 4 * a * da))}


def _phi(u, law, s):
    """Characteristic function of (Lambda - shift)/s at u (vector)."""
    a, da = law["a"] / s, law["da"] / s
    z = 1.0 - 2j * np.outer(u, a)
    return np.exp(np.sum(-0.5 * np.log(z) + 1j * u[:, None] * da[None, :] / z, axis=1))


def _umax(law, s):
    u = 1.0
    while abs(_phi(np.array([u]), law, s)[0]) > 1e-15:
        u *= 2.0
    return u


def cdf(y, law, s, U):
    yy = (y - law["shift"]) / s
    f = lambda u: float(np.imag(np.exp(-1j * u * yy) * _phi(np.array([u]), law, s)[0])) / u
    val, _ = integrate.quad(f, 0.0, U, limit=20000, epsabs=1e-13, epsrel=1e-11)
    return 0.5 - val / math.pi


def call_price(k, law, s, U):
    yy = (k - law["shift"]) / s
    m = (law["mean"] - k) / s
    f = lambda u: (1.0 - float(np.real(np.exp(-1j * u * yy) * _phi(np.array([u]), law, s)[0]))) / (u * u) if u > 0 else 0.0
    val, _ = integrate.quad(f, 0.0, U, limit=20000, epsabs=1e-13, epsrel=1e-11, points=[1e-3, 1e-2, 1e-1, 1.0])
    absdev = (2.0 / math.pi) * (val + 1.0 / U)                    # tail of (1 - Re)/u^2 beyond U, |phi| < 1e-15
    return s * (0.5 * m + 0.5 * absdev)


rows = []
RESULTS = OUT / ("e13_results.json" if not sys.argv[1:] else f"e13_results_H{'_'.join(sys.argv[1:])}.json")
for H in HS:
    t0 = time.time()
    fld = RiemannLiouville(H, T=T)
    f = Polynomial([SIG0 ** 2, 2 * SIG0 * NU, NU ** 2])
    mu = Measure.lebesgue(T)
    law = exact_law(fld)
    mean_exact = SIG0 ** 2 * T + NU ** 2 * T ** (2 * H + 1) / (2 * H + 1)
    A2, A2gap = second_moment(fld, f, mu)
    var_quad = A2 - mean_exact ** 2
    sd = math.sqrt(law["var"])
    U = _umax(law, sd)
    q_true = optimize.brentq(lambda y: cdf(y, law, sd, U) - ALPHA, mean_exact, mean_exact + 30 * sd, xtol=1e-14)
    strikes = {"ATM": mean_exact, "mean+2sd": mean_exact + 2 * sd}
    price_true = {lab: call_price(k, law, sd, U) for lab, k in strikes.items()}
    # monitor quantities (once)
    daily = T * np.arange(1, NDAY + 1) / NDAY
    cm = ConditionalMonitor(fld, f, T, daily, q=6, levels=3)
    d0, l, Am, ga = quadratic_cond_mean(cm, a0=SIG0 ** 2, a1=2 * SIG0 * NU, a2=NU ** 2)
    c0, bv, Om, gv = quadratic_cond_var(cm, a1=2 * SIG0 * NU, a2=NU ** 2, norms=True)
    P_ce, P_gap = ce_radius(cm)
    delta = quadratic_delta(ga, NDAY)
    EV = c0 + float(np.trace(Om))
    P_budget = EV + gv[0] + gv[1] * math.sqrt(NDAY) + gv[2] * NDAY
    EX2 = (d0 + float(np.trace(Am))) ** 2 + float(l @ l) + 2 * float(np.sum(Am * Am))
    riem = Measure.riemann(NDAY, T=T, rule="right")
    Lr = compute_ledger(fld, IdentityRepresentation(fld), f, mu, riem, N=None)
    eps_r = math.sqrt(max(Lr.M, 0.0) + Lr.quad_gap)
    tg, wg = riem.atom_arrays()
    S, Uu = np.meshgrid(tg, tg, indexing="ij")
    sg = np.sqrt(fld.var(tg))
    EXr2 = float(wg @ f.kernel(S, Uu, fld.R(S, Uu), sg[:, None] + 0 * S, sg[None, :] + 0 * S) @ wg)
    dP = dkw_delta(K, A_P)
    t_cheb = math.sqrt((P_budget / 4.0) * (1 - A_P) / (K * A_P))
    rng = np.random.default_rng(13_000 + int(1000 * H))
    aux = np.random.default_rng(31_000 + int(1000 * H))
    qkeys = ("a", "c", "d", "d_rb")
    cov = {k: 0 for k in qkeys}
    wid = {k: [] for k in qkeys}
    naive_q, pr = [], {lab: {"convex": [], "scarf": [], "lipschitz": [], "naive": [], "cov": {"convex": 0, "scarf": 0, "lipschitz": 0}}
                      for lab in strikes}
    for r in range(R):
        xi = rng.standard_normal((K, NDAY))
        lam = d0 + xi @ l + np.sum((xi @ Am) * xi, axis=1)
        nx2 = np.sum(xi * xi, axis=1)
        v = c0 + xi @ bv + np.sum((xi @ Om) * xi, axis=1) + gv[0] + gv[1] * np.sqrt(nx2) + gv[2] * nx2
        Y = xi @ cm.L.T
        lam_r = f.f(np.broadcast_to(daily, Y.shape), Y) @ wg
        ivs = {"a": certify_quantile(law_from_samples(lam_r), ALPHA, {2: eps_r}, dkw_a=A_Q),
               "c": certify_quantile_martingale(lam, ALPHA, P_budget, dkw_a=A_Q, delta=delta),
               "d": cantelli_dominating(lam, v, ALPHA, A_Q, delta, rng=aux),
               "d_rb": cantelli_rb(lam, v, ALPHA, A_Q, delta)}
        for key, iv in ivs.items():
            lo, hi = (iv.lower, iv.upper) if key == "a" else (iv["lower"], iv["upper"])
            cov[key] += int(lo <= q_true <= hi)
            wid[key].append(hi - lo)
        naive_q.append(float(np.quantile(lam_r, ALPHA)))
        for lab, k in strikes.items():
            lo, hi = call_bracket(lam, k, EX2, dP)
            phi = scarf_excess(lam - k, v)
            lo_r, hi_r = call_bracket(lam_r, k, EXr2, dP)
            br = {"convex": (lo - delta, hi + 0.5 * math.sqrt(P_budget) + delta),
                  "scarf": (lo - delta, hi + float(np.mean(phi)) + t_cheb + 2 * delta),
                  "lipschitz": (lo_r - eps_r, hi_r + eps_r)}
            for name, (blo, bhi) in br.items():
                pr[lab][name].append([blo, bhi])
                pr[lab]["cov"][name] += int(blo <= price_true[lab] <= bhi)
            pr[lab]["naive"].append(float(np.mean(np.maximum(lam_r - k, 0.0))))
    med = {k: float(np.median(w)) for k, w in wid.items()}
    prices = {}
    for lab in strikes:
        p = pr[lab]
        prices[lab] = {"k": strikes[lab], "exact": price_true[lab],
                       **{f"{nm}_median": np.median(np.array(p[nm]), axis=0).tolist() for nm in ("convex", "scarf", "lipschitz")},
                       **{f"{nm}_width_rel": float(np.median(np.diff(np.array(p[nm]), axis=1))) / price_true[lab]
                          for nm in ("convex", "scarf", "lipschitz")},
                       "coverage": {nm: c / R for nm, c in p["cov"].items()},
                       "naive_daily_mean": float(np.mean(p["naive"])), "naive_daily_bias_rel": float(np.mean(p["naive"])) / price_true[lab] - 1}
    row = {"H": H, "mean": mean_exact, "mean_kl": law["mean"], "sd": sd, "var_quad": var_quad, "A2_gap": A2gap,
           "rest_var_share": law["rest_var"] / law["var"], "q99": q_true,
           "M_daily": Lr.M, "riemann_rel_rms": math.sqrt(max(Lr.M, 0)) / sd, "ce_rel_rms": math.sqrt(EV) / sd,
           "P_ce": P_ce, "P_ce_gap": P_gap, "EV": EV, "P_budget": P_budget, "gap_V": list(gv), "gap_mean": list(ga),
           "delta": delta,
           **{f"coverage_{k}": c / R for k, c in cov.items()}, **{f"width_{k}_sd": w / sd for k, w in med.items()},
           "halfwidth_d_over_sd": 0.5 * med["d"] / sd, "status_d": "ISSUED" if 0.5 * med["d"] <= 0.05 * sd else "UNRESOLVED",
           "naive_q99_bias_sd": (float(np.mean(naive_q)) - q_true) / sd, "naive_q99_sd_over_reps": float(np.std(naive_q)) / sd,
           "prices": prices, "seconds": time.time() - t0}
    rows.append(row)
    print(json.dumps(row, default=float), flush=True)
    RESULTS.write_text(json.dumps(rows, indent=1, default=float))
