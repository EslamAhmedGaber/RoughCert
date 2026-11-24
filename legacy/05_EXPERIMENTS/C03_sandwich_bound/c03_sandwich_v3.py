"""C03 - Theorem B2 check: intint K'(R_Pi) D  <=  P  <=  intint K'(R) D, plus first-chaos lower bound."""
import json, math, sys, time
from pathlib import Path
import numpy as np
LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB / "04_CODE"))
from gcert import *
from gcert.quadrature import rule_2d_full, rule_2d_tensor
from gcert.surrogate import OutputAdaptedProjection

def sig(fld, s):
    return np.sqrt(np.maximum(fld.var(s), 0.0))

def bounds(fld, rep, f, mu, n=12, levels=8):
    bps = sorted(set(rep.breakpoints() + [fld.T]))
    S, U, W = rule_2d_full(mu, mu, bps, n, levels)
    D = fld.R(S, U) - rep.RPi_pairs(S, U)
    up = float(np.sum(W * D * f.kernel_derivative(S, U, fld.R(S, U), sig(fld, S), sig(fld, U))))
    lo = float(np.sum(W * D * f.kernel_derivative(S, U, rep.RPi_pairs(S, U), sig(fld, S), sig(fld, U))))
    c1s = f.chat(S, sig(fld, S), 1)[1] / np.where(sig(fld, S) > 0, sig(fld, S), 1)
    c1u = f.chat(U, sig(fld, U), 1)[1] / np.where(sig(fld, U) > 0, sig(fld, U), 1)
    first = float(np.sum(W * D * c1s * c1u))
    return first, lo, up

rows = []
cases = []
for H in (0.1, 0.3, 0.7):
    fld = FBM(H, T=1.0)
    mu = Measure.lebesgue(1.0)
    funcs = [WickExponential(1.0, 1.0, field=fld), PositivePart(c=0.5), Indicator(c=0.5)]
    if "--generic" in sys.argv:
        funcs = [Generic(lambda s, y: np.tanh(2 * y), name="tanh(2y)", homogeneous=True)]
    for f in funcs:
        for rep in (AnchorProjection(fld, [0.5]), AnchorProjection(fld, [0.25, 0.5, 0.75, 1.0])):
            cases.append((H, fld, f, mu, rep))
t0 = time.time()
OUTF = Path(__file__).parent / ("c03_results_generic.json" if "--generic" in sys.argv else "c03_results_closed.json")
if "--resume" in sys.argv and OUTF.exists():          # keep completed rows, compute only the missing cases
    rows = json.loads(OUTF.read_text())
done = {(r["H"], r["f"], r["rank"]) for r in rows}
for H, fld, f, mu, rep in cases:
    if (H, f.name, rep.rank) in done:
        continue
    L = compute_ledger(fld, rep, f, mu, mu, N=None)
    first, lo, up = bounds(fld, rep, f, mu)
    ok = (first <= lo * (1 + 1e-9) + 1e-12) and (lo <= L.P_mu * (1 + 1e-6) + 1e-10) and (L.P_mu <= up * (1 + 1e-6) + 1e-10)
    rows.append(dict(H=H, f=f.name, rank=rep.rank, P=L.P_mu, quad_gap=L.quad_gap, first_chaos=first,
                     lower=lo, upper=up, lower_over_P=lo / L.P_mu, upper_over_P=up / L.P_mu, ordered=bool(ok)))
    print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    OUTF.write_text(json.dumps(rows, indent=1))          # after every row, so a stopped run loses nothing
OUTF.write_text(json.dumps(rows, indent=1))
print("all ordered:", all(r["ordered"] for r in rows), " elapsed", round(time.time() - t0, 1))
