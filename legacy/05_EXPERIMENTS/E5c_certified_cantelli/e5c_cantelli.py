"""E5c driver (see PROTOCOL.md)."""
import json, math, sys, time
from pathlib import Path
import numpy as np
LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB / "04_CODE"))
from gcert import FBM, Polynomial
from gcert.monitor import (ConditionalMonitor, ce_radius, quadratic_cond_mean, quadratic_cond_var, quadratic_delta,
                           cantelli_dominating, cantelli_rb)
from gcert.martingale import certify_quantile_martingale

OUT = Path(__file__).parent
ALPHA, K, R, A = 0.95, 20_000, 100, 0.05
e5 = json.loads((LAB / "05_EXPERIMENTS/E5_bias_certified_mc/e5_results_A.json").read_text())
truth = {r["H"]: r["truth"] for r in e5}
var_exact = {r["H"]: r["var_quad"] for r in e5}
e5b = {(r["H"], r["n"]): r for r in json.loads((LAB / "05_EXPERIMENTS/E5b_conditional_certificates/e5b_results.json").read_text())}
f = Polynomial([0.0, 0.0, 1.0])
rows = []
for H in (0.10, 0.30, 0.50, 0.75):
    fld = FBM(H)
    for n in (4, 16, 64):
        t0 = time.time()
        grid = (np.arange(n) + 0.5) / n
        cm = ConditionalMonitor(fld, f, 1.0, grid, q=6, levels=3)
        d0, _, Am, ga = quadratic_cond_mean(cm)
        c0, _, Om, gv = quadratic_cond_var(cm, norms=True)
        P_ce, P_gap = ce_radius(cm)
        delta = quadratic_delta(ga, n)
        P_budget = c0 + gv[0] + float(np.trace(Om)) + n * gv[2]          # = E[v_k]
        rng = np.random.default_rng(1000 * int(100 * H) + n)               # E5b seeds: the same xi
        aux = np.random.default_rng(5000 + 1000 * int(100 * H) + n)
        cov = {"c": 0, "d": 0, "d_rb": 0}
        wid = {"c": [], "d": [], "d_rb": []}
        for r in range(R):
            xi = rng.standard_normal((K, n))
            lam = d0 + np.sum((xi @ Am) * xi, axis=1)
            v = c0 + gv[0] + np.sum((xi @ Om) * xi, axis=1) + gv[2] * np.sum(xi * xi, axis=1)
            ic = certify_quantile_martingale(lam, ALPHA, P_budget, dkw_a=A, delta=delta)
            idd = cantelli_dominating(lam, v, ALPHA, A, delta, rng=aux)
            irb = cantelli_rb(lam, v, ALPHA, A, delta)
            for key, iv in (("c", ic), ("d", idd), ("d_rb", irb)):
                cov[key] += int(iv["lower"] <= truth[H] <= iv["upper"])
                wid[key].append(iv["upper"] - iv["lower"])
        med = {k: float(np.median(w)) for k, w in wid.items()}
        row = {"H": H, "n": n, "truth": truth[H], "var_lambda": var_exact[H],
               "EV": c0 + float(np.trace(Om)), "P_ce": P_ce, "P_ce_gap": P_gap, "gap_V": list(gv), "gap_mean": list(ga),
               "delta": delta, "P_budget": P_budget, "residual_share": P_budget / var_exact[H],
               **{f"coverage_{k}": c / R for k, c in cov.items()}, **{f"width_{k}": w for k, w in med.items()},
               "width_a_e5b": e5b[(H, n)]["width_a"], "width_c_e5b": e5b[(H, n)]["width_c"],
               "ratio_c_d": med["c"] / med["d"], "ratio_a_d": e5b[(H, n)]["width_a"] / med["d"],
               "ratio_rb_d": med["d_rb"] / med["d"], "seconds": time.time() - t0}
        rows.append(row)
        print(json.dumps(row), flush=True)
        (OUT / "e5c_results.json").write_text(json.dumps(rows, indent=1))
