"""Generalised (non-exponential) subordinated functionals: closed forms and ledger vs Monte Carlo."""
import math, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gcert import *
from gcert.functionals import bivariate_expectation, SubordinatedFunctional, normalized_hermite

rng = np.random.default_rng(20261031)

def brute_kernel(func, s, u, x, ss, su, n=400):
    """E[f(s,U) f(u,V)] by dense tensor Gauss-Legendre on a box (independent of code paths)."""
    z, w = np.polynomial.legendre.leggauss(n); z = 9 * z; w = 9 * w
    Z1, Z2 = np.meshgrid(z, z, indexing="ij"); W = np.outer(w, w) * np.exp(-0.5 * (Z1**2 + Z2**2)) / (2 * np.pi)
    c = x / ss; v = max(su * su - c * c, 0.0)
    U = ss * Z1; V = c * Z1 + math.sqrt(v) * Z2
    return float(np.sum(W * func.f(np.full_like(U, s), U) * func.f(np.full_like(V, u), V)))

def test_closed_forms():
    fld = FBM(0.3)
    out = []
    fams = [PositivePart(c=0.4, mean=lambda s: 0.2 * np.asarray(s)), Indicator(c=-0.1),
            WickExponential(1.0, 0.8, field=fld), PathwiseExponential(1.0, 0.7)]
    for fam in fams:
        for (s, u) in ((0.3, 0.8), (0.55, 0.6)):
            ss, su = math.sqrt(fld.var(s)), math.sqrt(fld.var(u)); x = float(fld.R(s, u))
            kc = float(np.asarray(fam.kernel(np.array(s), np.array(u), np.array(x), ss, su)))
            kb = brute_kernel(fam, s, u, x, ss, su)
            kg = float(bivariate_expectation(fam, s, u, x, ss, su)[0])
            # Hermite coefficients: closed vs generic quadrature
            M = 25
            cc = fam.chat(np.array([s]), np.array([ss]), M)[:, 0]
            cg = SubordinatedFunctional.chat(fam, np.array([s]), np.array([ss]), M)[:, 0]
            out.append((fam.name, s, u, kc, abs(kc - kb) / abs(kb), abs(kg - kb) / abs(kb),
                        float(np.max(np.abs(cc - cg)))))
            # reference = breakpoint-aware conditional quadrature (independent of the closed forms);
            # the naive box rule kb is reported only (it does not resolve kinks/jumps)
            assert abs(kc - kg) <= 1e-9 * abs(kg) + 1e-13, (fam.name, kc, kg)
            assert np.max(np.abs(cc - cg)) < 1e-9, (fam.name, np.max(np.abs(cc - cg)))
    return out

def mc_ledger(fld, rep, func, mu_t, mu_w, nu_t, nu_w, N, nsamp=2_000_000, seed=1):
    """Independent MC of E|Lambda_mu - Q Lambda_nu|^2 for atomic measures (exact Gaussian sampling)."""
    times = np.unique(np.concatenate([mu_t, nu_t]))
    g = rep.loadings(times)                                     # (n, k)
    C = fld.R(times[:, None], times[None, :]); Dm = C - g @ g.T
    ev, V = np.linalg.eigh(0.5 * (Dm + Dm.T)); ev = np.maximum(ev, 0)
    Lr = V * np.sqrt(ev)[None, :]
    sig = np.sqrt(fld.var(times)); sigP = np.sqrt(np.maximum(np.einsum('ij,ij->i', g, g), 0))
    rng = np.random.default_rng(seed); acc = []; M2 = 0.0; cnt = 0
    chat = func.chat(times, sig, max(N or 0, 60))
    for _ in range(nsamp // 200_000):
        xi = rng.standard_normal((200_000, g.shape[1])); eta = rng.standard_normal((200_000, times.size))
        Yp = xi @ g.T; Y = Yp + eta @ Lr.T
        X = func.f(np.broadcast_to(times, Y.shape), Y)
        # Q X_t = sum_{m<=N} chat_m(t) (sigP/sig)^m h_m(Yp/sigP)   (h_m = He_m/sqrt(m!))
        Nn = N if N is not None else 60
        QX = np.zeros_like(Y)
        Hm = normalized_hermite(Yp / np.where(sigP > 0, sigP, 1)[None, :], Nn)
        for m in range(Nn + 1):
            QX += chat[m][None, :] * (sigP / sig)[None, :] ** m * Hm[m]
        idx = {t: i for i, t in enumerate(times)}
        Lm = sum(w * X[:, idx[t]] for t, w in zip(mu_t, mu_w))
        Ln = sum(w * QX[:, idx[t]] for t, w in zip(nu_t, nu_w))
        d2 = (Lm - Ln) ** 2
        acc.append(d2)
    d2 = np.concatenate(acc)
    return float(d2.mean()), float(d2.std() / math.sqrt(d2.size))

def test_ledger_vs_mc():
    fld = FBM(0.3, T=1.0)
    rep = AnchorProjection(fld, [0.5, 1.0])
    mu_t = np.array([0.15, 0.35, 0.55, 0.75, 0.95]); mu_w = np.array([0.2, -0.1, 0.3, 0.25, 0.2])
    nu_t = np.array([0.2, 0.6, 1.0]); nu_w = np.array([0.3, 0.35, 0.35])
    mu, nu = Measure.atomic(mu_t, mu_w), Measure.atomic(nu_t, nu_w)
    rows = []
    for func, N in ((PositivePart(c=0.3), 3), (Indicator(c=0.2), 4),
                    (Generic(lambda s, y: np.tanh(2 * y) + 0.3 * y ** 2, name="tanh+quad"), 2),
                    (WickExponential(1.0, 1.2, field=fld), None)):
        L = compute_ledger(fld, rep, func, mu, nu, N=N, audit=False)
        mean, se = mc_ledger(fld, rep, func, mu_t, mu_w, nu_t, nu_w, N)
        z = (L.total - mean) / se
        rows.append((func.name, N, L.total, mean, se, z, L.M, L.P_mu, L.C_mu, L.I))
        assert abs(z) < 4.5, (func.name, L.total, mean, se)
        assert min(L.M_Q, L.P_mu, L.C_mu) > -1e-12
    return rows

if __name__ == "__main__":
    t0 = time.time()
    for r in test_closed_forms():
        print("closed-form", r)
    for r in test_ledger_vs_mc():
        print("ledger-vs-MC", r)
    print(f"elapsed {time.time()-t0:.1f}s")
