"""Sampling floors: the width of the DKW band of the monitor alone (no bias term), on the same samples and seeds
as E5c (argument e5c) or E13 (argument e13).  Post-processing only; the floor is the width any certificate built
from these samples would have if the residual were zero."""
import json, math, sys
from pathlib import Path
import numpy as np
LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB / "04_CODE"))
from gcert import FBM, Polynomial, RiemannLiouville
from gcert.monitor import ConditionalMonitor, quadratic_cond_mean, dkw_eps


def band(lam, alpha, eps):
    s = np.sort(lam)
    K = s.size
    return float(s[min(math.ceil(K * (alpha + eps)), K) - 1] - s[max(math.ceil(K * (alpha - eps)), 1) - 1])


which = sys.argv[1] if len(sys.argv) > 1 else "e5c"
if which == "e5c":
    ALPHA, K, R, A = 0.95, 20_000, 100, 0.05
    out = {}
    for H in (0.10, 0.30, 0.50, 0.75):
        for n in (4, 16, 64):
            grid = (np.arange(n) + 0.5) / n
            cm = ConditionalMonitor(FBM(H), Polynomial([0.0, 0.0, 1.0]), 1.0, grid, q=6, levels=3)
            d0, _, Am, _ = quadratic_cond_mean(cm, audit=False)
            rng = np.random.default_rng(1000 * int(100 * H) + n)             # E5b/E5c seeds
            w = [band(d0 + np.sum((xi @ Am) * xi, axis=1), ALPHA, dkw_eps(K, A))
                 for xi in (rng.standard_normal((K, n)) for _ in range(R))]
            out[f"{H}_{n}"] = {"H": H, "n": n, "floor": float(np.median(w))}
            print(H, n, out[f"{H}_{n}"]["floor"], flush=True)
    (Path(__file__).parent / "e5c_floor.json").write_text(json.dumps(out, indent=1))
else:
    SIG0, NU, T, NDAY = 0.2, 0.1, 1.0 / 12.0, 21
    ALPHA, K, R, A = 0.99, 200_000, 50, 0.01
    e13 = LAB / "05_EXPERIMENTS/E13_rough_stein_stein"
    rows = {}
    for q in sorted(e13.glob("e13_results*.json")):
        for r in json.loads(q.read_text()):
            rows[r["H"]] = r
    out = {}
    for H, r in sorted(rows.items()):
        daily = T * np.arange(1, NDAY + 1) / NDAY
        cm = ConditionalMonitor(RiemannLiouville(H, T=T), Polynomial([SIG0 ** 2, 2 * SIG0 * NU, NU ** 2]), T, daily,
                                q=6, levels=3)
        d0, l, Am, _ = quadratic_cond_mean(cm, a0=SIG0 ** 2, a1=2 * SIG0 * NU, a2=NU ** 2)
        rng = np.random.default_rng(13_000 + int(1000 * H))                  # E13 seeds
        w = [band(d0 + xi @ l + np.sum((xi @ Am) * xi, axis=1), ALPHA, dkw_eps(K, A))
             for xi in (rng.standard_normal((K, NDAY)) for _ in range(R))]
        out[str(H)] = {"H": H, "floor": float(np.median(w)), "floor_sd": float(np.median(w)) / r["sd"]}
        print(H, out[str(H)], flush=True)
    (e13 / "e13_floor.json").write_text(json.dumps(out, indent=1))
