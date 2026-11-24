"""E5 - coverage of naive vs bias-certified Monte Carlo intervals for continuum quantiles (see PROTOCOL.md)."""
import json, math, sys, time
from pathlib import Path
import numpy as np
from scipy import integrate, optimize, stats
from scipy.linalg import cholesky

LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB / "04_CODE"))
from gcert import FBM, Measure, Polynomial, PositivePart, IdentityRepresentation, compute_ledger
from gcert.transfer import law_from_samples, certify_quantile, dkw_delta
from gcert.quadrature import rule_2d_full

OUT = Path(__file__).parent
ALPHA, K, R, A_DKW = 0.95, 20_000, 200, 0.05
HS = (0.10, 0.30, 0.50, 0.75)
NS = (2, 4, 8, 16, 32, 64)


# ------------------------------------------------------------------ truth for int B_H^2 (KL + Imhof)
def kl_eigs(H, n=3000):
    x, w = np.polynomial.legendre.leggauss(n)
    t, w = 0.5 * (x + 1), 0.5 * w
    fld = FBM(H)
    C = fld.R(t[:, None], t[None, :])
    sw = np.sqrt(w)
    lam = np.linalg.eigvalsh(sw[:, None] * C * sw[None, :])[::-1]
    return np.clip(lam, 0, None)


def imhof_cdf(x, lam):
    def integrand(u):
        th = 0.5 * np.sum(np.arctan(lam * u)) - 0.5 * x * u
        rho = np.exp(0.25 * np.sum(np.log1p((lam * u) ** 2)))
        return math.sin(th) / (u * rho)
    val, err = integrate.quad(integrand, 0, np.inf, limit=2000, epsabs=1e-13, epsrel=1e-11)
    return 0.5 - val / math.pi


def truth_quadratic(H):
    lam = kl_eigs(H)
    trace = 1.0 / (2 * H + 1)
    K_use = 1500
    lead, rest = lam[:K_use], max(trace - lam[:K_use].sum(), 0.0)
    # checks: trace and variance 2 intint R^2 (independent 2-D quadrature)
    fld = FBM(H)
    mu = Measure.lebesgue(1.0)
    S, U, W = rule_2d_full(mu, mu, [0.0, 1.0], 14, 10)
    var_exact = 2 * float(np.sum(W * fld.R(S, U) ** 2))
    q = optimize.brentq(lambda x: imhof_cdf(x - rest, lead) - ALPHA, 1e-6, 50 * trace, xtol=1e-12)
    return {"H": H, "quantile": q, "trace": trace, "sum_lambda": float(lam.sum()),
            "var_kl": float(2 * np.sum(lam ** 2)), "var_quad": var_exact, "tail_mass": rest}


# ------------------------------------------------------------------ naive and certified intervals
def order_stat_ci(x_sorted, alpha, conf=0.95):
    n = x_sorted.size
    lo = int(stats.binom.ppf((1 - conf) / 2, n, alpha))
    hi = int(stats.binom.ppf(1 - (1 - conf) / 2, n, alpha)) + 1
    return x_sorted[max(lo - 1, 0)], x_sorted[min(hi - 1, n - 1)]


def run_cell(fld, func, n, truth, seed, eps):
    t = (np.arange(n) + 0.5) / n
    w = np.full(n, 1.0 / n)
    C = fld.R(t[:, None], t[None, :])
    L = cholesky(C + 1e-14 * np.eye(n), lower=True)
    rng = np.random.default_rng(seed)
    cov_n = cov_c = 0
    wn, wc = [], []
    for r in range(R):
        Y = rng.standard_normal((K, n)) @ L.T
        lam = func.f(np.broadcast_to(t, Y.shape), Y) @ w
        xs = np.sort(lam)
        a, b = order_stat_ci(xs, ALPHA)
        cov_n += (a <= truth <= b)
        wn.append(b - a)
        law = law_from_samples(xs)
        cert = certify_quantile(law, ALPHA, {2: eps}, dkw_a=A_DKW)
        cov_c += (cert.lower <= truth <= cert.upper)
        wc.append(cert.width)
    return {"coverage_naive": cov_n / R, "coverage_cert": cov_c / R,
            "median_width_naive": float(np.median(wn)), "median_width_cert": float(np.median(wc))}


def main(which="A"):
    rows = []
    t0 = time.time()
    for H in HS:
        fld = FBM(H)
        if which == "A":
            tr = truth_quadratic(H)
            func = Polynomial([0.0, 0.0, 1.0])
            truth = tr["quantile"]
            print("truth", tr, flush=True)
        else:
            raise NotImplementedError
        mu = Measure.lebesgue(1.0)
        for n in NS:
            nu = Measure.riemann(n, T=1.0, rule="mid")
            Lg = compute_ledger(fld, IdentityRepresentation(fld), func, mu, nu, N=None)
            eps = math.sqrt(max(Lg.M, 0.0) + Lg.quad_gap)
            res = run_cell(fld, func, n, truth, seed=1000 * int(100 * H) + n, eps=eps)
            row = {"output": which, "H": H, "n": n, "eps": eps, "M": Lg.M, "quad_gap": Lg.quad_gap,
                   "truth": truth, **res, "seconds": time.time() - t0}
            if which == "A":
                row.update({k: tr[k] for k in ("trace", "sum_lambda", "var_kl", "var_quad")})
            rows.append(row)
            print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            (OUT / f"e5_results_{which}.json").write_text(json.dumps(rows, indent=1))
    return rows


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "A")
