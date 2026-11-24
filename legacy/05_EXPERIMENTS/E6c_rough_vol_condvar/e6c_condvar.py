"""E6c driver (see PROTOCOL.md)."""
import json, math, sys, time
from pathlib import Path
import numpy as np
LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB / "04_CODE"))
from gcert import RiemannLiouville, WickExponential, PathwiseExponential, Measure
from gcert.ledger import second_moment
from gcert.monitor import ConditionalMonitor, ce_radius, ExpCondVar, cantelli_dominating, scarf_excess, dkw_eps
from gcert.martingale import certify_quantile_martingale
from gcert.transfer import dkw_delta

OUT = Path(__file__).parent
XI0, ETA, T, NDAY = 0.235 ** 2, 1.9, 1.0 / 12.0, 21
ALPHA, K, R, A_Q, A_P = 0.99, 200_000, 5, 0.01, 0.005
RESUME = "--resume" in sys.argv          # keep finished cells; the seeds use len(rows), so they stay identical
ARGS = [a for a in sys.argv[1:] if a != "--resume"]
HS = [float(x) for x in ARGS] or [0.05, 0.10, 0.30]
e6 = {(r["H"], r["output"]): r for r in
      json.loads((LAB / "05_EXPERIMENTS/E6_rough_vol_family/e6_results_0.05_0.1_0.3.json").read_text())}
e6b = {}
for q in sorted((LAB / "05_EXPERIMENTS/E6b_rough_vol_conditional").glob("e6b_results*.json")):
    for r in json.loads(q.read_text()):
        e6b[(r["H"], r["output"])] = r


def call_bracket(x, k, second_moment_x, d):
    v = np.sort(np.asarray(x, float))
    Kn = v.size
    lo_c, hi_c = np.arange(Kn) / Kn, np.arange(1, Kn + 1) / Kn
    g = np.maximum(v - k, 0.0)
    low = float(np.sum(g * np.clip(np.minimum(hi_c + d, 1.0) - np.maximum(lo_c + d, 0.0), 0, None)))
    up = float(np.sum(g * np.clip(np.minimum(hi_c - d, 1.0 - d) - np.maximum(lo_c - d, 0.0), 0, None)))
    return low, up + math.sqrt(d * second_moment_x)


RESULTS = OUT / ("e6c_results.json" if not ARGS else f"e6c_results_H{'_'.join(ARGS)}.json")
rows = json.loads(RESULTS.read_text()) if RESUME and RESULTS.exists() else []

if RESUME:
    expected = [(h, name) for h in HS for name in ("IV", "IVol")]
    actual = [(r["H"], r["output"]) for r in rows]
    if actual != expected[:len(actual)]:
        raise ValueError("Resume requires a completed prefix of the requested cells to preserve row-index seeds")
    print(f"Resuming {len(rows)}/{len(expected)} completed cells from {RESULTS.name}", flush=True)
done = {(r["H"], r["output"]) for r in rows}
for H in HS:
    fld = RiemannLiouville(H, T=T)
    prods = {"IV": WickExponential(XI0, ETA, field=fld),
             "IVol": PathwiseExponential(math.sqrt(XI0), ETA / 2, drift=lambda s: -ETA ** 2 * fld.var(s) / 4)}
    for name, f in prods.items():
        if (H, name) in done:
            continue
        t0 = time.time()
        print(f"Starting H={H:g}, output={name}, seed row={len(rows)}", flush=True)
        ref = e6[(H, name)]
        daily = T * np.arange(1, NDAY + 1) / NDAY
        cm = ConditionalMonitor(fld, f, T, daily, q=6, levels=3)
        P, gap = ce_radius(cm)
        cmf = ConditionalMonitor(fld, f, T, daily, q=12, levels=6)
        xa = np.random.default_rng(7).standard_normal((1000, NDAY))
        dq = 10.0 * float(np.sqrt(np.mean((cm.estimate(xa) - cmf.estimate(xa)) ** 2)))
        ecv = ExpCondVar(cm, levels_pair=((6, 3), (8, 4)))
        if name == "IV":
            A2, A2_gap = second_moment(fld, f, Measure.lebesgue(T))
            A2_upper = max(ref["sd"] ** 2 + ref["mean"] ** 2, A2 + abs(A2_gap))
            EX2 = (math.sqrt(A2_upper) + dq) ** 2
        t_setup = time.time() - t0
        print(f"Setup complete in {t_setup:.1f}s; running {R} replications of K={K}", flush=True)
        rng = np.random.default_rng(60_000 + int(1000 * H) + 7 * len(rows))
        aux = np.random.default_rng(61_000 + int(1000 * H) + 7 * len(rows))
        wid = {"c": [], "d": [], "floor": []}
        inside = {"c": 0, "d": 0}
        prices = {"ATM": ref["mean"], "mean+2sd": ref["mean"] + 2 * ref["sd"]}
        pr = {lab: {"convex": [], "scarf": []} for lab in prices}
        eps = dkw_eps(K, A_Q)
        audit_rel = []
        for r in range(R):
            xi = rng.standard_normal((K, NDAY))
            lam = cm.estimate(xi)
            V1, V2 = ecv.values(xi)
            v = V2 + np.abs(V2 - V1)
            audit_rel.append(float(np.max(np.abs(V2 - V1) / np.maximum(V2, 1e-300))))
            ic = certify_quantile_martingale(lam, ALPHA, P + gap, dkw_a=A_Q, delta=dq)
            idd = cantelli_dominating(lam, v, ALPHA, A_Q, dq, rng=aux)
            for key, iv in (("c", ic), ("d", idd)):
                wid[key].append(iv["upper"] - iv["lower"])
                inside[key] += int(iv["lower"] <= ref["mc_q99_diag"] <= iv["upper"])
            s = np.sort(lam)
            wid["floor"].append(float(s[math.ceil(K * (ALPHA + eps)) - 1] - s[math.ceil(K * (ALPHA - eps)) - 1]))
            if name == "IV":
                dP = dkw_delta(K, A_P)
                t_cheb = math.sqrt(((P + gap) / 4.0) * (1 - A_P) / (K * A_P))
                for lab, k in prices.items():
                    lo, hi = call_bracket(lam, k, EX2, dP)
                    phi = scarf_excess(lam - k, v)
                    pr[lab]["convex"].append([lo - dq, hi + 0.5 * math.sqrt(P + gap) + dq])
                    pr[lab]["scarf"].append([lo - dq, hi + float(np.mean(phi)) + t_cheb + 2 * dq])
            print(f"H={H:g} {name}: replication {r + 1}/{R}, elapsed {time.time() - t0:.1f}s", flush=True)
        sd = ref["sd"]
        row = {"H": H, "output": name, "mean": ref["mean"], "sd": sd, "P": P, "P_gap": gap, "ce_rel_rms": math.sqrt(P) / sd,
               "delta_q": dq, "audit_rel_max": max(audit_rel), "setup_seconds": t_setup,
               **{f"width_{k}_sd": float(np.median(w)) / sd for k, w in wid.items()},
               "ratio_c_d": float(np.median(wid["c"]) / np.median(wid["d"])),
               "mc_diag": ref["mc_q99_diag"], **{f"mc_inside_{k}": c / R for k, c in inside.items()},
               "e6b_width_sd": (e6b[(H, name)]["q99_upper"] - e6b[(H, name)]["q99_lower"]) / sd if (H, name) in e6b else None,
               "e6_width_sd": (ref["q99_upper"] - ref["q99_lower"]) / sd}
        if name == "IV":
            row["price_moment_audit"] = {"correction": "C-020", "second_moment_upper": EX2}
            fine = {p["strike"]: p["fine_mc_price"] for p in e6b[(H, name)]["prices"]} if (H, name) in e6b else {}
            row["prices"] = {lab: {"k": k, **{f"{nm}_median": np.median(np.array(pr[lab][nm]), axis=0).tolist() for nm in ("convex", "scarf")},
                                   "fine_mc_e6b": fine.get(lab)} for lab, k in prices.items()}
        row["seconds"] = time.time() - t0
        rows.append(row)
        print(json.dumps(row, default=float), flush=True)
        temporary = RESULTS.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(rows, indent=1, default=float))
        temporary.replace(RESULTS)
