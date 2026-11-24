"""E14 driver (see PROTOCOL.md): certified VIX futures and options under rough Bergomi.

VIX_T^2 = Lambda = Delta^{-1} int_T^{T+Delta} xi_T(u) du is a Wick-exponential output of the forward field
Y_u = sqrt(2H) int_0^T (u-s)^{H-1/2} dW_s.  With x = u - T the covariance is R_RL(T+x, T+y) - R_RL(x, y).
A rank-one projection gives E[Lambda | Z] = X(Z), Var(Lambda | Z) = V(Z) in closed form, so the brackets below
are one-dimensional quadratures: no sampling enters them.  Sampling appears only in the reference."""
import json, math, sys, time
from pathlib import Path
import numpy as np
from scipy.special import ndtr
LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB / "04_CODE"))
from gcert.fields import RiemannLiouville
from gcert.quadrature import graded_panel
from gcert.martingale import mart_upper, mart_lower

OUT = Path(__file__).parent
DELTA = 30.0 / 365.0
TICK = 5e-4                                     # 0.05 vol points, in VIX units (VIX = 0.20 means 20 points)
FLAT = 0.235 ** 2
STRIKES = (0.8, 1.0, 1.25, 1.5, 2.0)
RULES = ((14, 10, 160), (20, 14, 320))
MC_BATCHES, MC_BATCH = 40, 25_000


def flat(_):
    return FLAT


def scenario2(t):
    return 0.234 ** 2 * (1.0 + t) ** 2


class Cell:
    def __init__(self, H, T, eta, curve, group):
        self.H, self.T, self.eta, self.curve, self.group = H, T, eta, curve, group
        self.rl = RiemannLiouville(H)

    def R(self, x, y):
        return self.rl.R(self.T + x, self.T + y) - self.rl.R(x, y)

    def nodes(self, n, levels):
        x, w = graded_panel(0.0, DELTA, n=n, levels=levels, left=True, right=False)
        xi = np.broadcast_to(np.asarray(self.curve(self.T + x), float), x.shape).astype(float)
        return x, w / DELTA, xi, self.R(x[:, None], x[None, :])


def zrule(panels, zmax=9.5, q=8):
    e = np.linspace(-zmax, zmax, panels + 1)
    gx, gw = np.polynomial.legendre.leggauss(q)
    z = np.concatenate([0.5 * (b - a) * gx + 0.5 * (a + b) for a, b in zip(e[:-1], e[1:])])
    w = np.concatenate([0.5 * (b - a) * gw for a, b in zip(e[:-1], e[1:])])
    return z, w * np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def phi_upper(X, v, k):
    """inf_r (1/(4r)) [m + sqrt(v + m^2)], m = X + r^2 - 2 k r: every r gives a valid bound (Prop. root (ii))."""
    def val(r):
        m = X + r * r - 2.0 * k * r
        return (m + np.sqrt(v + m * m)) / (4.0 * r)
    base = np.sqrt(np.maximum(X, k * k))
    fac = np.exp(np.linspace(-1.0, 1.0, 121))
    cand = np.stack([val(base * f) for f in fac], axis=1)
    j = np.argmin(cand, axis=1)
    lo, hi = base * fac[np.maximum(j - 1, 0)], base * fac[np.minimum(j + 1, fac.size - 1)]
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo.copy(), hi.copy()
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(60):
        vc, vd = val(c), val(d)
        keep = vc < vd
        b = np.where(keep, d, b); a = np.where(keep, a, c)
        c, d = b - gr * (b - a), a + gr * (b - a)
    return np.minimum(val(0.5 * (a + b)), cand.min(axis=1))


def cantelli_quantiles(X, v, pz, alpha):
    """Quantiles of the dominating laws T+ and T- of Proposition condvar (i), on the z-rule."""
    def up_prob(t):                                  # P(T+ < t)
        d = t - X
        return float(np.sum(pz[d > 0] * d[d > 0] ** 2 / (v[d > 0] + d[d > 0] ** 2)))

    def low_prob(t):                                 # P(T- <= t)
        d = X - t
        out = pz[d <= 0].sum() + np.sum(pz[d > 0] * v[d > 0] / (v[d > 0] + d[d > 0] ** 2))
        return float(out)

    def cross(fun, a, b):
        for _ in range(200):
            c = 0.5 * (a + b)
            if fun(c) >= alpha:
                b = c
            else:
                a = c
        return b
    top = float(X.max() + 1e3 * math.sqrt(v.max() + 1e-30))
    return cross(low_prob, 0.0, top), cross(up_prob, 0.0, top)


def analyse(cell, n, levels, panels, spot_strikes=None):
    t0 = time.perf_counter()
    eta = cell.eta
    x, w, xi, C = cell.nodes(n, levels)
    m1 = float(w @ xi)
    m2 = float(np.sum(np.outer(w * xi, w * xi) * np.exp(eta ** 2 * C)))
    wxi = w * xi
    b = C @ wxi
    nh = math.sqrt(float(wxi @ b))
    g = b / nh
    D = C - np.outer(g, g)
    P = float(np.sum(np.outer(wxi, wxi) * (np.exp(eta ** 2 * C) - np.exp(eta ** 2 * np.outer(g, g)))))
    z, pz = zrule(panels)
    Gh = xi * np.exp(eta * np.outer(z, g) - 0.5 * eta ** 2 * g ** 2)
    X = Gh @ w
    GW = Gh * w
    V = np.maximum(np.einsum("ij,ij->i", GW @ np.expm1(eta ** 2 * D), GW), 0.0)
    rho = 1.0 / np.sqrt(1.0 + V / X ** 2)
    sq = np.sqrt(X)
    res = {"nodes": x.size, "mean": m1, "m2": m2, "var": m2 - m1 ** 2, "P": P, "EX": float(pz @ X), "EV": float(pz @ V),
           "fut_lo": float(pz @ (rho * sq)), "fut_up": float(pz @ sq),
           "fut_width_bound": float(pz @ (0.5 * V * X ** -1.5)),
           "jmm_lo": float(np.sum(w * np.sqrt(xi) * np.exp(-eta ** 2 * np.diag(C) / 8.0))), "jmm_up": math.sqrt(m1)}
    s2 = math.log(m2 / m1 ** 2)
    mu = math.log(m1) - 0.5 * s2
    res["ln_fut"] = math.exp(0.5 * mu + s2 / 8.0)
    res["brackets_seconds"] = None
    if spot_strikes is not None:
        calls = {}
        sg = math.sqrt(s2)
        for mult, k in spot_strikes.items():
            d2 = (mu - 2.0 * math.log(k)) / sg
            ln = math.exp(0.5 * mu + s2 / 8.0) * ndtr(d2 + 0.5 * sg) - k * ndtr(d2)
            calls[mult] = {"k": k, "lo": float(pz @ np.maximum(rho * sq - k, 0.0)),
                           "up": float(pz @ phi_upper(X, V, k)), "lip_half": math.sqrt(P) / (2.0 * k),
                           "center": float(pz @ np.maximum(sq - k, 0.0)), "lognormal": float(ln)}
        res["calls"] = calls
        q = {}
        q["cv"] = cantelli_quantiles(X, V, pz, 0.99)
        q["mart"] = (float(mart_lower(X, pz, 0.99, P)), float(mart_upper(X, pz, 0.99, P)))
        res["q"] = q
        res["q_surrogate"] = float(X[np.searchsorted(np.cumsum(pz[np.argsort(X)]), 0.99)]) if False else None
    res["brackets_seconds"] = time.perf_counter() - t0
    res["g"], res["w"], res["x"] = g, w, x
    return res


def reference(cell, seed):
    """Independent of g and D: draws of the full node vector from C, with exact X(Z) as control variate."""
    eta = cell.eta
    x, w, xi, C = cell.nodes(*RULES[1][:2])
    wxi = w * xi
    b = C @ wxi
    nh = math.sqrt(float(wxi @ b))
    g = b / nh
    lam, U = np.linalg.eigh(C)
    keep = lam > 1e-14 * lam.max()
    L = U[:, keep] * np.sqrt(lam[keep])
    rng = np.random.default_rng(seed)
    dv = np.diag(C)
    root = np.empty(MC_BATCHES * MC_BATCH)
    root_cv = np.empty_like(root)
    for i in range(MC_BATCHES):
        Y = rng.standard_normal((MC_BATCH, L.shape[1])) @ L.T
        lamb = (xi * np.exp(eta * Y - 0.5 * eta ** 2 * dv)) @ w
        Z = (Y @ wxi) / nh
        Xe = (xi * np.exp(eta * np.outer(Z, g) - 0.5 * eta ** 2 * g ** 2)) @ w
        root[i * MC_BATCH:(i + 1) * MC_BATCH] = np.sqrt(lamb)
        root_cv[i * MC_BATCH:(i + 1) * MC_BATCH] = np.sqrt(Xe)
    return root, root_cv


def node_plan(cell, res_fine):
    """Exact RMS error of uniform rectangle/trapezoid rules for Lambda (identity; no sampling)."""
    eta = cell.eta
    x, w, xi, C = cell.nodes(*RULES[1][:2])
    A = float(np.sum(np.outer(w * xi, w * xi) * np.exp(eta ** 2 * C)))
    out = {}
    for rule in ("rect", "trap"):
        rows = []
        for n in (4, 8, 16, 32, 64, 128, 256, 512, 1024):
            h = DELTA / n
            if rule == "rect":
                t, nu = h * np.arange(1, n + 1), np.full(n, 1.0 / n)
            else:
                t = h * np.arange(0, n + 1)
                nu = np.full(n + 1, 1.0 / n); nu[0] = nu[-1] = 0.5 / n
            xit = np.broadcast_to(np.asarray(cell.curve(cell.T + t), float), t.shape).astype(float)
            cross = float((w * xi) @ np.exp(eta ** 2 * cell.R(x[:, None], t[None, :])) @ (nu * xit))
            atoms = float((nu * xit) @ np.exp(eta ** 2 * cell.R(t[:, None], t[None, :])) @ (nu * xit))
            rows.append({"n": n, "rms": math.sqrt(max(A - 2.0 * cross + atoms, 0.0))})
        out[rule] = rows
    return out


def run_cell(cell, seed):
    root, root_cv = reference(cell, seed)
    K = root.size
    coarse = analyse(cell, *RULES[0])
    fine = analyse(cell, *RULES[1])
    d = root - root_cv
    fut_ref = fine["fut_up"] + float(d.mean())
    fut_se = float(d.std() / math.sqrt(K))
    strikes = {m: m * fut_ref for m in STRIKES}
    t0 = time.perf_counter()
    coarse = analyse(cell, *RULES[0], strikes)
    sec = time.perf_counter() - t0
    fine = analyse(cell, *RULES[1], strikes)
    gap_lo = abs(fine["fut_lo"] - coarse["fut_lo"])
    gap_up = abs(fine["fut_up"] - coarse["fut_up"])
    row = {"group": cell.group, "H": cell.H, "T": cell.T, "eta": cell.eta, "K_mc": K, "nodes": fine["nodes"],
           "mean": fine["mean"], "sd": math.sqrt(fine["var"]), "P": fine["P"], "share": fine["P"] / fine["var"],
           "check_EX": fine["EX"] / fine["mean"] - 1.0, "check_EV": fine["EV"] / fine["P"] - 1.0,
           "fut_ref": fut_ref, "fut_se": fut_se, "fut_plain": float(root.mean()), "fut_plain_se": float(root.std() / math.sqrt(K)),
           "vr": float(root.var() / d.var()),
           "fut_lo": fine["fut_lo"] - gap_lo, "fut_up": fine["fut_up"] + gap_up, "fut_gap": max(gap_lo, gap_up),
           "fut_width_bound": fine["fut_width_bound"],
           "jmm_lo": fine["jmm_lo"], "jmm_up": fine["jmm_up"], "lognormal": fine["ln_fut"],
           "bracket_seconds_coarse": sec, "calls": []}
    for m in STRIKES:
        cf, cc = fine["calls"][m], coarse["calls"][m]
        cv = np.maximum(root_cv - strikes[m], 0.0)
        dd = np.maximum(root - strikes[m], 0.0) - cv
        ref = cf["center"] + float(dd.mean())
        se = float(dd.std() / math.sqrt(K))
        gl, gu = abs(cf["lo"] - cc["lo"]), abs(cf["up"] - cc["up"])
        row["calls"].append({"mult": m, "k": cf["k"], "ref": ref, "se": se, "lo": cf["lo"] - gl, "up": cf["up"] + gu,
                             "gap": max(gl, gu), "lip_width": 2.0 * cf["lip_half"], "lognormal": cf["lognormal"]})
    qs = np.sort(root)
    jq = int(math.ceil(0.99 * K)) - 1
    se_q = math.sqrt(0.99 * 0.01 * K)
    row["q99"] = {"mc": float(qs[jq]), "mc_lo": float(qs[int(jq - 3 * se_q)]), "mc_hi": float(qs[int(jq + 3 * se_q)]),
                  "cv": [math.sqrt(v) for v in fine["q"]["cv"]], "mart": [math.sqrt(v) for v in fine["q"]["mart"]],
                  "gap": max(abs(math.sqrt(a) - math.sqrt(b)) for a, b in zip(fine["q"]["cv"] + fine["q"]["mart"], coarse["q"]["cv"] + coarse["q"]["mart"]))}
    row["plan"] = node_plan(cell, fine)
    return row


def main():
    cells = []
    for H in (0.05, 0.10, 0.30):
        for T in (1 / 52, 1 / 12, 3 / 12):
            cells.append(Cell(H, T, 1.9, flat, "base"))
    for H in (0.05, 0.10, 0.30):
        cells.append(Cell(H, 1 / 12, 1.9, scenario2, "curve"))
    for H in (0.05, 0.10, 0.30):
        cells.append(Cell(H, 0.5, 3.0, flat, "stress"))
    rows = []
    for i, c in enumerate(cells):
        t0 = time.time()
        r = run_cell(c, 14000 + i)
        r["seconds"] = time.time() - t0
        rows.append(r)
        print(json.dumps({k: r[k] for k in ("group", "H", "T", "eta", "share", "check_EV", "fut_ref", "fut_se", "fut_lo", "fut_up", "jmm_lo", "jmm_up", "lognormal", "seconds")}), flush=True)
        (OUT / "e14_results.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
