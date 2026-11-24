"""E13b driver (see PROTOCOL.md, E13b)."""
import json, math, sys, time
from pathlib import Path
import numpy as np
LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB / "04_CODE"))
from gcert import RiemannLiouville, Polynomial
from gcert.monitor import ConditionalMonitor, quadratic_cond_mean, quadratic_cond_var, quadratic_delta, cantelli_dominating, dkw_eps
from gcert.martingale import certify_quantile_martingale

OUT = Path(__file__).parent
SIG0, NU, T = 0.2, 0.1, 1.0 / 12.0
ALPHA, K, R, A = 0.99, 200_000, 10, 0.01
HS = [float(x) for x in sys.argv[1:]] or [0.05, 0.10]
ref = {}
for q in sorted(OUT.glob("e13_results*.json")):
    for r in json.loads(q.read_text()):
        ref[r["H"]] = r
rows = []
RESULTS = OUT / ("e13b_results.json" if not sys.argv[1:] else f"e13b_results_H{'_'.join(sys.argv[1:])}.json")
for H in HS:
    q_true, sd, var = ref[H]["q99"], ref[H]["sd"], ref[H]["var_quad"]
    for n in (21, 42, 84, 168):
        t0 = time.time()
        grid = T * np.arange(1, n + 1) / n
        cm = ConditionalMonitor(RiemannLiouville(H, T=T), Polynomial([SIG0 ** 2, 2 * SIG0 * NU, NU ** 2]), T, grid, q=6, levels=3)
        d0, l, Am, ga = quadratic_cond_mean(cm, a0=SIG0 ** 2, a1=2 * SIG0 * NU, a2=NU ** 2)
        c0, bv, Om, gv = quadratic_cond_var(cm, a1=2 * SIG0 * NU, a2=NU ** 2, norms=True)
        delta = quadratic_delta(ga, n)
        P = c0 + float(np.trace(Om)) + gv[0] + gv[1] * math.sqrt(n) + gv[2] * n
        t_coef = time.time() - t0
        rng = np.random.default_rng(130_000 + int(1000 * H) + n)
        aux = np.random.default_rng(310_000 + int(1000 * H) + n)
        cov = {"c": 0, "d": 0}
        wid = {"c": [], "d": [], "floor": []}
        eps = dkw_eps(K, A)
        for r in range(R):
            lam, v = np.empty(K), np.empty(K)
            for c in range(0, K, 20_000):
                xi = rng.standard_normal((20_000, n))
                r2 = np.sum(xi * xi, axis=1)
                lam[c:c + 20_000] = d0 + xi @ l + np.sum((xi @ Am) * xi, axis=1)
                v[c:c + 20_000] = c0 + xi @ bv + np.sum((xi @ Om) * xi, axis=1) + gv[0] + gv[1] * np.sqrt(r2) + gv[2] * r2
            ic = certify_quantile_martingale(lam, ALPHA, P, dkw_a=A, delta=delta)
            idd = cantelli_dominating(lam, v, ALPHA, A, delta, rng=aux)
            for key, iv in (("c", ic), ("d", idd)):
                cov[key] += int(iv["lower"] <= q_true <= iv["upper"])
                wid[key].append(iv["upper"] - iv["lower"])
            s = np.sort(lam)
            wid["floor"].append(float(s[math.ceil(K * (ALPHA + eps)) - 1] - s[math.ceil(K * (ALPHA - eps)) - 1]))
        med = {k: float(np.median(w)) / sd for k, w in wid.items()}
        row = {"H": H, "n": n, "per_day": n // 21, "residual_share": P / var, "ce_rel_rms": math.sqrt(P / var),
               "gap_V": list(gv), "delta": delta, **{f"coverage_{k}": c / R for k, c in cov.items()},
               **{f"width_{k}_sd": w for k, w in med.items()}, "coef_seconds": t_coef, "seconds": time.time() - t0}
        rows.append(row)
        print(json.dumps(row), flush=True)
        RESULTS.write_text(json.dumps(rows, indent=1))
