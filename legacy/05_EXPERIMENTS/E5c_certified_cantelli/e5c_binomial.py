"""E5c Part C driver (see PROTOCOL.md, Part C)."""
import json, math, sys, time
from pathlib import Path
import numpy as np
LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB / "04_CODE"))
from gcert import FBM, Polynomial
from gcert.monitor import (ConditionalMonitor, quadratic_cond_mean, quadratic_cond_var, quadratic_delta,
                           cantelli_dominating_binomial, dkw_from_sorted, binomial_order_index)

OUT = Path(__file__).parent
ALPHA, K, R, A = 0.95, 20_000, 100, 0.05
truth = {r["H"]: r["truth"] for r in json.loads((LAB / "05_EXPERIMENTS/E5_bias_certified_mc/e5_results_A.json").read_text())}
partA = {(r["H"], r["n"]): r for r in json.loads((OUT / "e5c_results.json").read_text())}
f = Polynomial([0.0, 0.0, 1.0])
mU, mL = binomial_order_index(K, ALPHA, A / 2, True), binomial_order_index(K, ALPHA, A / 2, False)
rows = []
for H in (0.10, 0.30, 0.50, 0.75):
    for n in (4, 16, 64):
        t0 = time.time()
        cm = ConditionalMonitor(FBM(H), f, 1.0, (np.arange(n) + 0.5) / n, q=6, levels=3)
        d0, _, Am, ga = quadratic_cond_mean(cm)
        c0, _, Om, gv = quadratic_cond_var(cm, norms=True)
        delta = quadratic_delta(ga, n)
        rng = np.random.default_rng(1000 * int(100 * H) + n)               # Part A seeds
        aux = np.random.default_rng(5000 + 1000 * int(100 * H) + n)
        cov = 0
        wd, wb, fb = [], [], []
        for r in range(R):
            xi = rng.standard_normal((K, n))
            lam = d0 + np.sum((xi @ Am) * xi, axis=1)
            v = c0 + gv[0] + np.sum((xi @ Om) * xi, axis=1) + gv[2] * np.sum(xi * xi, axis=1)
            b = cantelli_dominating_binomial(lam, v, ALPHA, A, delta, rng=aux)
            d = dkw_from_sorted(b["Tp"], b["Tm"], ALPHA, A, delta)
            cov += int(b["lower"] <= truth[H] <= b["upper"])
            wb.append(b["upper"] - b["lower"])
            wd.append(d["upper"] - d["lower"])
            s = np.sort(lam)
            fb.append(float(s[mU - 1] - s[mL - 1]))
        pa = partA[(H, n)]
        row = {"H": H, "n": n, "coverage_dbin": cov / R, "width_dbin": float(np.median(wb)),
               "width_d_recomputed": float(np.median(wd)), "width_d_partA": pa["width_d"], "width_c": pa["width_c"],
               "floor_bin": float(np.median(fb)), "ratio_d_dbin": pa["width_d"] / float(np.median(wb)),
               "ratio_c_dbin": pa["width_c"] / float(np.median(wb)), "seconds": time.time() - t0}
        rows.append(row)
        print(json.dumps(row), flush=True)
        (OUT / "e5c_binomial.json").write_text(json.dumps(rows, indent=1))
