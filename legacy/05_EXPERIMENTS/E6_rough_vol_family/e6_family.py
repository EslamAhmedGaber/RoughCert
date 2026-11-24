import json, math, sys, time
from pathlib import Path
import numpy as np
LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB / "04_CODE"))
from gcert import RiemannLiouville, Measure, WickExponential, PathwiseExponential, Indicator, Generic, IdentityRepresentation, compute_ledger
from gcert.solve import solve, FunctionalProblem
from gcert.functionals import ExpCall
from gcert.solve import _sample_field

OUT = Path(__file__).parent
XI0, ETA, T, NDAY = 0.235 ** 2, 1.9, 1.0 / 12.0, 21
B = 2 * XI0
HS = [float(x) for x in sys.argv[1:]] or [0.05, 0.10, 0.30]

def products(fld):
    var = lambda s: fld.var(s)
    lb = math.log(B / XI0)
    return {
        "IV": WickExponential(XI0, ETA, field=fld),
        "IVol": PathwiseExponential(math.sqrt(XI0), ETA / 2, drift=lambda s: -ETA ** 2 * var(s) / 4),
        "OCC": Indicator(c=lb / ETA, mean=lambda s: -0.5 * ETA * var(s)),
        "EXC": ExpCall(lambda s: XI0 * np.exp(-0.5 * ETA ** 2 * var(s)), ETA, B, name="excess variance"),
    }

RES = OUT / f"e6_results_{'_'.join(map(str, HS))}.json"
rows = json.loads(RES.read_text()) if RES.exists() else []   # resume: completed cells are kept, not rerun
done = {(r["H"], r["output"]) for r in rows}
t0 = time.time()
for H in HS:
    fld = RiemannLiouville(H, T=T)
    mu = Measure.lebesgue(T)
    daily = Measure.riemann(NDAY, T=T, rule="right")
    for name, f in products(fld).items():
        if (H, name) in done:
            continue
        tc = time.time()
        pb = FunctionalProblem(fld, f, mu, support=(0.0, math.inf), name=f"{name} H={H}")
        L = compute_ledger(fld, IdentityRepresentation(fld), f, mu, daily, N=None)
        rep = solve(pb, ["mean", "variance"], {"moment": 1e-10}, {})
        mean = rep.outputs[0].value; var_ = rep.outputs[1].value
        # tolerance relative to the output scale: re-run quantile with tol = 5% of sd
        tol_q = 0.05 * math.sqrt(max(var_, 0.0))
        rep = solve(pb, [("quantile", 0.99)], {"quantile": tol_q},
                    {"mc_max": 400_000, "n_max": 256, "adapted_anchor_sizes": (4,), "confidence_a": 0.01})
        q = rep.outputs[0]
        # diagnostic MC on a fine grid
        rng = np.random.default_rng(int(1e4 * H) + len(rows))
        n_fine = 512
        tt = T * (np.arange(n_fine) + 0.5) / n_fine
        smp = np.concatenate([f.f(np.broadcast_to(tt, Y.shape), Y) @ np.full(n_fine, T / n_fine)
                              for Y in _sample_field(fld, tt, 100_000, rng)])
        q_mc = float(np.quantile(smp, 0.99))
        row = {"H": H, "output": name, "mean": mean, "sd": math.sqrt(max(var_, 0)),
               "monitor_rel_rms": math.sqrt(max(L.M, 0)) / math.sqrt(max(var_, 1e-300)),
               "monitor_M": L.M, "q99_lower": q.lower, "q99_upper": q.upper, "q99_center": q.value,
               "q99_label": q.label, "q99_status": q.status, "q99_route": q.details.get("route"),
               "q99_n": q.details.get("n"), "q99_K": q.details.get("K"), "tol_halfwidth": tol_q,
               "mc_q99_diag": q_mc, "mc_inside": bool(q.lower <= q_mc <= q.upper), "seconds": time.time() - tc}
        rows.append(row)
        print(json.dumps(row, default=float), flush=True)
        (OUT / f"e6_results_{'_'.join(map(str, HS))}.json").write_text(json.dumps(rows, indent=1, default=float))
print("done", time.time() - t0)
