"""E5c Part B: rates of the certified bias terms (see PROTOCOL.md)."""
import json, math, sys, time
from pathlib import Path
import numpy as np
LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB / "04_CODE"))
from gcert import FBM, Polynomial
from gcert.monitor import ConditionalMonitor, quadratic_cond_mean, quadratic_cond_var
from gcert.martingale import mart_upper

OUT = Path(__file__).parent
ALPHA, K = 0.95, 200_000
f = Polynomial([0.0, 0.0, 1.0])
rows = []
for H in (0.10, 0.30):
    fld = FBM(H)
    for n in (4, 8, 16, 32, 64, 128):
        t0 = time.time()
        grid = (np.arange(n) + 0.5) / n
        cm = ConditionalMonitor(fld, f, 1.0, grid, q=6, levels=3)
        d0, _, Am, ga = quadratic_cond_mean(cm)
        c0, _, Om, gv = quadratic_cond_var(cm, norms=True)
        rng = np.random.default_rng(777 + 1000 * int(100 * H) + n)
        x = np.empty(K); v = np.empty(K)
        for c in range(0, K, 20_000):
            xi = rng.standard_normal((20_000, n))
            x[c:c + 20_000] = d0 + np.sum((xi @ Am) * xi, axis=1)
            v[c:c + 20_000] = c0 + gv[0] + np.sum((xi @ Om) * xi, axis=1) + gv[2] * np.sum(xi * xi, axis=1)
        P = float(np.mean(v))
        qh = float(np.quantile(x, ALPHA))
        Um = mart_upper(np.sort(x), np.full(K, 1.0 / K), ALPHA, P)

        def low_p(t):
            d = t - x
            return float(np.mean(np.where(d > 0, d * d / (v + d * d), 0.0)))
        a, b = qh, qh + 50 * math.sqrt(P) + 1e-9
        for _ in range(100):
            m = 0.5 * (a + b)
            a, b = (a, m) if low_p(m) >= ALPHA else (m, b)
        qd = b
        sd = float(np.std(x))
        h = 0.02 * sd
        wk = np.exp(-0.5 * ((x - qh) / h) ** 2)
        fq = float(np.sum(wk)) / (K * h * math.sqrt(2 * math.pi))
        Esq = float(np.sum(wk * np.sqrt(v)) / np.sum(wk))
        row = {"H": H, "n": n, "P": P, "q_hat": qh, "B_c": Um - qh, "B_d": qd - qh,
               "pred_c": (3 * P / (4 * fq)) ** (1 / 3), "pred_d": 0.5 * math.pi * Esq, "f_q": fq,
               "sqrtP": math.sqrt(P), "E_sqrtV_at_q": Esq, "seconds": time.time() - t0}
        rows.append(row)
        print(json.dumps(row), flush=True)
        (OUT / "e5c_rates.json").write_text(json.dumps(rows, indent=1))
for H in (0.10, 0.30):
    r = [x for x in rows if x["H"] == H]
    lp = np.log([x["P"] for x in r])
    for key in ("B_c", "B_d"):
        s = np.polyfit(lp, np.log([x[key] for x in r]), 1)[0]
        print(f"H={H} slope {key} vs P: {s:.3f}")
