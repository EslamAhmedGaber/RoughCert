"""E13c driver (see PROTOCOL.md, E13c)."""
import json, math, sys, time
from pathlib import Path
import numpy as np
LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB / "04_CODE"))
from gcert import RiemannLiouville, Polynomial
from gcert.monitor import (ConditionalMonitor, quadratic_cond_mean, quadratic_cond_var, quadratic_delta,
                           quadratic_form_call, scarf_excess)

OUT = Path(__file__).parent
SIG0, NU, T, NDAY = 0.2, 0.1, 1.0 / 12.0, 21
K, R, A_P = 200_000, 50, 0.005
ref = {}
for q in sorted(OUT.glob("e13_results*.json")):
    for r in json.loads(q.read_text()):
        ref[r["H"]] = r
rows = []
for H in sorted(ref):
    t0 = time.time()
    daily = T * np.arange(1, NDAY + 1) / NDAY
    cm = ConditionalMonitor(RiemannLiouville(H, T=T), Polynomial([SIG0 ** 2, 2 * SIG0 * NU, NU ** 2]), T, daily, q=6, levels=3)
    d0, l, Am, ga = quadratic_cond_mean(cm, a0=SIG0 ** 2, a1=2 * SIG0 * NU, a2=NU ** 2)
    c0, bv, Om, gv = quadratic_cond_var(cm, a1=2 * SIG0 * NU, a2=NU ** 2, norms=True)
    delta = quadratic_delta(ga, NDAY)
    P = c0 + float(np.trace(Om)) + gv[0] + gv[1] * math.sqrt(NDAY) + gv[2] * NDAY
    t_cheb = math.sqrt((P / 4.0) * (1 - A_P) / (K * A_P))
    strikes = {lab: p["k"] for lab, p in ref[H]["prices"].items()}
    J = {lab: quadratic_form_call(d0, l, Am, k) for lab, k in strikes.items()}
    rng = np.random.default_rng(13_000 + int(1000 * H))                  # E13 seeds: the same xi
    br = {lab: [] for lab in strikes}
    for r in range(R):
        xi = rng.standard_normal((K, NDAY))
        lam = d0 + xi @ l + np.sum((xi @ Am) * xi, axis=1)
        nx2 = np.sum(xi * xi, axis=1)
        v = c0 + xi @ bv + np.sum((xi @ Om) * xi, axis=1) + gv[0] + gv[1] * np.sqrt(nx2) + gv[2] * nx2
        for lab, k in strikes.items():
            br[lab].append([J[lab][0] - delta - J[lab][1], J[lab][0] + float(np.mean(scarf_excess(lam - k, v))) + t_cheb
                            + 2 * delta + J[lab][1]])
    out = {}
    for lab in strikes:
        b = np.array(br[lab])
        ex = ref[H]["prices"][lab]["exact"]
        out[lab] = {"k": strikes[lab], "exact": ex, "jensen_exact": J[lab][0], "jensen_quad_err": J[lab][1],
                    "median": np.median(b, axis=0).tolist(), "coverage": float(np.mean((b[:, 0] <= ex) & (ex <= b[:, 1]))),
                    "width_rel": float(np.median(b[:, 1] - b[:, 0])) / ex, "sampling_term_rel": t_cheb / ex,
                    "convex_e13_width_rel": ref[H]["prices"][lab]["convex_width_rel"],
                    "scarf_e13_width_rel": ref[H]["prices"][lab]["scarf_width_rel"]}
    row = {"H": H, "P": P, "delta": delta, "t_cheb": t_cheb, "prices": out, "seconds": time.time() - t0}
    rows.append(row)
    print(json.dumps(row, default=float), flush=True)
    (OUT / "e13c_results.json").write_text(json.dumps(rows, indent=1, default=float))
