import json, math, sys, time
from pathlib import Path
import numpy as np
LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB / "04_CODE"))
from gcert import FBM, Measure, WickExponential, PositivePart, Indicator, Generic, IdentityRepresentation, compute_ledger

# Run order (amendment A1): closed-form kernels for every H first, the generic tanh last.
# Design, grids and analysis are unchanged.
HS = (0.1, 0.3, 0.5, 0.7)
MAKERS = [lambda fld: WickExponential(1.0, 1.0, field=fld),
          lambda fld: PositivePart(c=0.5),
          lambda fld: Indicator(c=0.5),
          lambda fld: Generic(lambda s, y: np.tanh(2 * y), name="tanh2y", homogeneous=True)]
OUT = Path(__file__).parent / "e9_results.json"
rows = []
t0 = time.time()
for make in MAKERS:
    for H in HS:
        fld = FBM(H)
        mu = Measure.lebesgue(1.0)
        f = make(fld)
        Ms = []
        for e in range(2, 11):
            n = 2 ** e
            L = compute_ledger(fld, IdentityRepresentation(fld), f, mu, Measure.riemann(n, rule="mid"), N=None,
                               n=10, levels=7)
            Ms.append((n, L.M, L.quad_gap))
            rows.append({"H": H, "f": f.name, "n": n, "M": L.M, "quad_gap": L.quad_gap})
            print(f"  H={H} {f.name} n={n} M={L.M:.4e} t={time.time()-t0:.0f}s", flush=True)
            OUT.write_text(json.dumps(rows, indent=1))
        ns = np.array([m[0] for m in Ms if m[0] >= 64], float)
        Mv = np.array([m[1] for m in Ms if m[0] >= 64])
        slope = float(np.polyfit(np.log(ns), np.log(Mv), 1)[0]) if np.all(Mv > 0) else float("nan")
        rows.append({"H": H, "f": f.name, "slope_64_1024": slope, "theory_exp_class": -(1 + 2 * H)})
        print(f"H={H} {f.name:22s} slope={slope:7.3f}  (exp-class theory {-(1+2*H):.2f})  M(1024)={Ms[-1][1]:.3e}  "
              f"gap={max(m[2] for m in Ms):.1e}  t={time.time()-t0:.0f}s", flush=True)
        OUT.write_text(json.dumps(rows, indent=1))
