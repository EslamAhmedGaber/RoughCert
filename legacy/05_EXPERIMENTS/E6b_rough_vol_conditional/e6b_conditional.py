"""E6b driver (see PROTOCOL.md)."""
import json, math, sys, time
from pathlib import Path
import numpy as np
LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB / "04_CODE"))
from gcert import RiemannLiouville, Measure, WickExponential, PathwiseExponential, Indicator, IdentityRepresentation
from gcert import compute_ledger
from gcert.ledger import second_moment
from gcert.functionals import ExpCall
from gcert.monitor import ConditionalMonitor, ce_radius
from gcert.martingale import certify_quantile_martingale
from gcert.transfer import dkw_delta
from gcert.solve import _sample_field

OUT = Path(__file__).parent
XI0, ETA, T, NDAY = 0.235 ** 2, 1.9, 1.0 / 12.0, 21
B = 2 * XI0
K, A_Q, N_CE = 200_000, 0.01, 256
e6 = {(r["H"], r["output"]): r for r in
      json.loads((LAB / "05_EXPERIMENTS/E6_rough_vol_family/e6_results_0.05_0.1_0.3.json").read_text())}


def products(fld):
    var = lambda s: fld.var(s)
    lb = math.log(B / XI0)
    return {"IV": WickExponential(XI0, ETA, field=fld),
            "IVol": PathwiseExponential(math.sqrt(XI0), ETA / 2, drift=lambda s: -ETA ** 2 * var(s) / 4),
            "OCC": Indicator(c=lb / ETA, mean=lambda s: -0.5 * ETA * var(s)),
            "EXC": ExpCall(lambda s: XI0 * np.exp(-0.5 * ETA ** 2 * var(s)), ETA, B, name="excess variance")}


def call_bracket(x, k, second_moment_x, d):
    """[lower, upper] for E (X - k)_+ from samples x on the DKW event (sup |F_K - F| <= d)."""
    v = np.sort(np.asarray(x, float))
    Kn = v.size
    lo_c, hi_c = np.arange(Kn) / Kn, np.arange(1, Kn + 1) / Kn
    g = np.maximum(v - k, 0.0)
    low = float(np.sum(g * np.clip(np.minimum(hi_c + d, 1.0) - np.maximum(lo_c + d, 0.0), 0, None)))
    up = float(np.sum(g * np.clip(np.minimum(hi_c - d, 1.0 - d) - np.maximum(lo_c - d, 0.0), 0, None)))
    return low, up + math.sqrt(d * second_moment_x)


rows = []
HS = [float(x) for x in sys.argv[1:]] or [0.05, 0.10, 0.30]      # optional subset (parallel runs)
RESULTS = OUT / ("e6b_results.json" if not sys.argv[1:] else f"e6b_results_H{'_'.join(sys.argv[1:])}.json")
for H in HS:
    fld = RiemannLiouville(H, T=T)
    mu = Measure.lebesgue(T)
    for name, f in products(fld).items():
        t0 = time.time()
        ref = e6[(H, name)]
        daily = T * np.arange(1, NDAY + 1) / NDAY
        cm_d = ConditionalMonitor(fld, f, T, daily, q=6, levels=3)
        P_d, gap_d = ce_radius(cm_d)
        grid = T * (np.arange(N_CE) + 0.5) / N_CE
        cm = ConditionalMonitor(fld, f, T, grid, q=6, levels=3)
        P, gap = ce_radius(cm)
        cmf = ConditionalMonitor(fld, f, T, grid, q=12, levels=6)
        xa = np.random.default_rng(7).standard_normal((1000, N_CE))
        dq = 10.0 * float(np.sqrt(np.mean((cm.estimate(xa) - cmf.estimate(xa)) ** 2)))
        rng = np.random.default_rng(int(1e4 * H) + len(rows))
        xi = rng.standard_normal((K, N_CE))
        lam = cm.estimate(xi)
        q = certify_quantile_martingale(lam, 0.99, P + gap, dkw_a=A_Q, delta=dq)
        tol = ref["tol_halfwidth"]
        row = {"H": H, "output": name, "mean": ref["mean"], "sd": ref["sd"],
               "daily_riemann_rel_rms": ref["monitor_rel_rms"], "daily_ce_rel_rms": math.sqrt(max(P_d, 0)) / ref["sd"],
               "ce256_rel_rms": math.sqrt(max(P, 0)) / ref["sd"], "P_gap": gap, "delta_q": dq,
               "q99_lower": q["lower"], "q99_upper": q["upper"], "q99_center": q["center"],
               "width_rel_E6": (q["upper"] - q["lower"]) / (ref["q99_upper"] - ref["q99_lower"]),
               "halfwidth_over_sd": 0.5 * (q["upper"] - q["lower"]) / ref["sd"],
               "status": "ISSUED" if 0.5 * (q["upper"] - q["lower"]) <= tol else "UNRESOLVED",
               "mc_q99_diag": ref["mc_q99_diag"],
               "mc_inside": bool(q["lower"] <= ref["mc_q99_diag"] <= q["upper"]),
               "e6_bracket": [ref["q99_lower"], ref["q99_upper"]]}
        if name == "IV":
            A2, A2_gap = second_moment(fld, f, mu)
            A2_upper = max(A2, 0.0) + abs(A2_gap)
            P_lower = max(P - abs(gap), 0.0)
            EX2 = A2_upper - P_lower + 2 * math.sqrt(A2_upper) * dq + dq ** 2
            row["price_moment_audit"] = {"correction": "C-020", "second_moment_upper": EX2}
            d = dkw_delta(K, A_Q)
            # Riemann n = 256 on the same grid values (Lipschitz bracket) and its exact second moment
            Y = xi @ cm.L.T
            lam_r = f.f(np.broadcast_to(grid, Y.shape), Y).mean(axis=1) * T
            Lr = compute_ledger(fld, IdentityRepresentation(fld), f, mu, Measure.riemann(N_CE, T=T, rule="mid"), N=None)
            tg, wg = Measure.riemann(N_CE, T=T, rule="mid").atom_arrays()
            S, U = np.meshgrid(tg, tg, indexing="ij")
            sg = np.sqrt(fld.var(tg))
            EXr2 = float(wg @ f.kernel(S, U, fld.R(S, U), sg[:, None] + 0 * S, sg[None, :] + 0 * S) @ wg)
            # daily Riemann (naive) and fine-grid (diagnostic) prices
            rng2 = np.random.default_rng(99 + len(rows))
            Yd = next(_sample_field(fld, daily, 200_000, rng2, chunk=200_000))
            lam_daily = f.f(np.broadcast_to(daily, Yd.shape), Yd).mean(axis=1) * T
            fine = T * (np.arange(1024) + 0.5) / 1024
            lam_fine = np.concatenate([f.f(np.broadcast_to(fine, Yf.shape), Yf).mean(axis=1) * T
                                       for Yf in _sample_field(fld, fine, 200_000, rng2, chunk=20_000)])
            prices = []
            for label, kk in (("ATM", ref["mean"]), ("mean+2sd", ref["mean"] + 2 * ref["sd"])):
                lo, hi = call_bracket(lam, kk, EX2, d)
                lo_r, hi_r = call_bracket(lam_r, kk, EXr2, d)
                eps_r = math.sqrt(max(Lr.M, 0) + Lr.quad_gap)
                prices.append({"strike": label, "k": kk,
                               "convex_bracket": [lo - dq, hi + 0.5 * math.sqrt(P + gap) + dq],
                               "lipschitz_bracket_riemann256": [lo_r - eps_r, hi_r + eps_r],
                               "daily_mc_price": float(np.mean(np.maximum(lam_daily - kk, 0))),
                               "daily_mc_se": float(np.std(np.maximum(lam_daily - kk, 0)) / math.sqrt(lam_daily.size)),
                               "fine_mc_price": float(np.mean(np.maximum(lam_fine - kk, 0))),
                               "fine_mc_se": float(np.std(np.maximum(lam_fine - kk, 0)) / math.sqrt(lam_fine.size))})
            row["prices"] = prices
            row["M256"] = Lr.M
        row["seconds"] = time.time() - t0
        rows.append(row)
        print(json.dumps(row, default=float), flush=True)
        RESULTS.write_text(json.dumps(rows, indent=1, default=float))
