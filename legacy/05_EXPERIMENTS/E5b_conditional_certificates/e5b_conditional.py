"""E5b driver (see PROTOCOL.md)."""
import json, math, sys, time
from pathlib import Path
import numpy as np
LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB / "04_CODE"))
from gcert import FBM, Measure, Polynomial, IdentityRepresentation, compute_ledger
from gcert.monitor import ConditionalMonitor, ce_radius
from gcert.martingale import certify_quantile_martingale
from gcert.transfer import law_from_samples, certify_quantile

OUT = Path(__file__).parent
ALPHA, K, R, A = 0.95, 20_000, 100, 0.05
e5 = json.loads((LAB / "05_EXPERIMENTS/E5_bias_certified_mc/e5_results_A.json").read_text())
truth = {r["H"]: r["truth"] for r in e5}
f = Polynomial([0.0, 0.0, 1.0])
mu = Measure.lebesgue(1.0)
rows = []
for H in (0.10, 0.30, 0.50, 0.75):
    fld = FBM(H)
    for n in (4, 16, 64):
        t0 = time.time()
        grid = (np.arange(n) + 0.5) / n
        M = compute_ledger(fld, IdentityRepresentation(fld), f, mu, Measure.riemann(n, rule="mid"), N=None)
        eps_r = math.sqrt(max(M.M, 0.0) + M.quad_gap)
        cm = ConditionalMonitor(fld, f, 1.0, grid, q=6, levels=3)
        P, gap = ce_radius(cm)
        cmf = ConditionalMonitor(fld, f, 1.0, grid, q=12, levels=6)
        xa = np.random.default_rng(7).standard_normal((2000, n))
        dq = 10.0 * float(np.sqrt(np.mean((cm.estimate(xa) - cmf.estimate(xa)) ** 2)))
        Pb = P + gap
        rng = np.random.default_rng(1000 * int(100 * H) + n)
        cov = {"a": 0, "b": 0, "c": 0}
        wid = {"a": [], "b": [], "c": []}
        for r in range(R):
            xi = rng.standard_normal((K, n))
            Y = xi @ cm.L.T                                         # exact grid values
            lam_r = (Y ** 2).mean(axis=1)                           # Riemann (midpoint) sum
            lam_c = cm.estimate(xi)                                 # E[Lambda | grid]
            ia = certify_quantile(law_from_samples(lam_r), ALPHA, {2: eps_r}, dkw_a=A)
            ib = certify_quantile(law_from_samples(lam_c), ALPHA, {2: math.sqrt(Pb) + dq}, dkw_a=A)
            ic = certify_quantile_martingale(lam_c, ALPHA, Pb, dkw_a=A, delta=dq)
            for key, (lo, hi) in (("a", (ia.lower, ia.upper)), ("b", (ib.lower, ib.upper)),
                                  ("c", (ic["lower"], ic["upper"]))):
                cov[key] += int(lo <= truth[H] <= hi)
                wid[key].append(hi - lo)
        row = {"H": H, "n": n, "truth": truth[H], "M_riemann": M.M, "P_ce": P, "P_gap": gap, "delta_q": dq,
               **{f"coverage_{k}": cov[k] / R for k in cov}, **{f"width_{k}": float(np.median(wid[k])) for k in wid},
               "seconds": time.time() - t0}
        rows.append(row)
        print(json.dumps(row), flush=True)
        (OUT / "e5b_results.json").write_text(json.dumps(rows, indent=1))
