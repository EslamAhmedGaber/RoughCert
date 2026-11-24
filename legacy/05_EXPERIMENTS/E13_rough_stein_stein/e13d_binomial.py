"""E13d driver (see PROTOCOL.md, E13d)."""
import json, math, sys, time
from pathlib import Path
import numpy as np
LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB / "04_CODE"))
from gcert import RiemannLiouville, Polynomial
from gcert.monitor import (ConditionalMonitor, quadratic_cond_mean, quadratic_cond_var, quadratic_delta,
                           cantelli_dominating_binomial, dkw_from_sorted, binomial_order_index)

OUT = Path(__file__).parent
SIG0, NU, T, NDAY = 0.2, 0.1, 1.0 / 12.0, 21
ALPHA, K, R, A = 0.99, 200_000, 50, 0.01
ref = {}
for q in sorted(OUT.glob("e13_results*.json")):
    for r in json.loads(q.read_text()):
        ref[r["H"]] = r
mU, mL = binomial_order_index(K, ALPHA, A / 2, True), binomial_order_index(K, ALPHA, A / 2, False)
rows = []
for H in sorted(ref):
    t0 = time.time()
    daily = T * np.arange(1, NDAY + 1) / NDAY
    cm = ConditionalMonitor(RiemannLiouville(H, T=T), Polynomial([SIG0 ** 2, 2 * SIG0 * NU, NU ** 2]), T, daily, q=6, levels=3)
    d0, l, Am, ga = quadratic_cond_mean(cm, a0=SIG0 ** 2, a1=2 * SIG0 * NU, a2=NU ** 2)
    c0, bv, Om, gv = quadratic_cond_var(cm, a1=2 * SIG0 * NU, a2=NU ** 2, norms=True)
    delta = quadratic_delta(ga, NDAY)
    rng = np.random.default_rng(13_000 + int(1000 * H))                  # E13 seeds
    aux = np.random.default_rng(31_000 + int(1000 * H))
    q, sd = ref[H]["q99"], ref[H]["sd"]
    cov = 0
    wb, wd, fb = [], [], []
    for r in range(R):
        xi = rng.standard_normal((K, NDAY))
        lam = d0 + xi @ l + np.sum((xi @ Am) * xi, axis=1)
        nx2 = np.sum(xi * xi, axis=1)
        v = c0 + xi @ bv + np.sum((xi @ Om) * xi, axis=1) + gv[0] + gv[1] * np.sqrt(nx2) + gv[2] * nx2
        b = cantelli_dominating_binomial(lam, v, ALPHA, A, delta, rng=aux)
        d = dkw_from_sorted(b["Tp"], b["Tm"], ALPHA, A, delta)
        cov += int(b["lower"] <= q <= b["upper"])
        wb.append(b["upper"] - b["lower"])
        wd.append(d["upper"] - d["lower"])
        s = np.sort(lam)
        fb.append(float(s[mU - 1] - s[mL - 1]))
    row = {"H": H, "coverage_dbin": cov / R, "width_dbin_sd": float(np.median(wb)) / sd,
           "width_d_recomputed_sd": float(np.median(wd)) / sd, "width_d_e13_sd": ref[H]["width_d_sd"],
           "width_c_e13_sd": ref[H]["width_c_sd"], "width_a_e13_sd": ref[H]["width_a_sd"],
           "floor_bin_sd": float(np.median(fb)) / sd, "seconds": time.time() - t0}
    rows.append(row)
    print(json.dumps(row), flush=True)
    (OUT / "e13d_results.json").write_text(json.dumps(rows, indent=1))
