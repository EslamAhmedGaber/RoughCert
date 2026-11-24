"""One entry point:  solve(problem, outputs, tol, options).

The same six steps for every admissible problem:
  1. specify & validate   (class membership, H range of the backend, square integrability)
  2. represent once       (orthogonal-projection backend: anchors / output-adapted / CovFourier)
  3. expand               (Hermite data of the output field; chaos level)
  4. evaluate outputs     (exact moment identities; surrogate laws)
  5. assess error         (exact ledger -> W_p radius -> sharp quantile/ES brackets)
  6. refine & reuse       (rank, monitoring, chaos, Monte Carlo size) until the tolerance is met,
                           or return UNRESOLVED with the reason.
Every reported number carries a guarantee label:
  EXACT-IDENTITY+QUADRATURE  identity proved; value from audited quadrature (gap reported)
  CERTIFIED-NUMERICAL        bracket from an exact-identity radius and a deterministic surrogate law
  STATISTICAL(1-a)           bracket valid with probability >= 1-a (DKW), bias covered by exact radius
  DIAGNOSTIC                 indicator only, no guarantee
  UNRESOLVED / UNSUPPORTED   no statement issued (reason given)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field as dfield, asdict

import numpy as np
from scipy.linalg import cholesky

from .fields import FBM, GaussianField
from .quadrature import Measure, rule_1d
from .representation import AnchorProjection, IdentityRepresentation, CovFourierProjection
from .functionals import SubordinatedFunctional
from .ledger import compute_ledger, first_moment, second_moment
from .surrogate import OneDimSurrogate, OutputAdaptedProjection
from .transfer import law_from_map, law_from_samples, certify_quantile, es_bracket, dkw_delta


@dataclass
class FunctionalProblem:
    field: GaussianField
    functional: SubordinatedFunctional
    measure: Measure
    support: tuple = None          # known support of the output (used only to clamp brackets)
    name: str = ""


@dataclass
class OutputResult:
    kind: str
    level: float = None
    value: float = None
    lower: float = None
    upper: float = None
    label: str = ""
    status: str = "ISSUED"
    details: dict = dfield(default_factory=dict)


@dataclass
class SolveReport:
    problem: str
    outputs: list
    resources: dict
    log: list

    def as_dict(self):
        return {"problem": self.problem, "outputs": [asdict(o) for o in self.outputs],
                "resources": self.resources, "log": self.log}

    def table(self):
        rows = []
        for o in self.outputs:
            rows.append(f"{o.kind:>12s} {'' if o.level is None else o.level:>6} "
                        f"value={o.value if o.value is None else round(o.value, 6)!s:>12} "
                        f"[{o.lower if o.lower is None else round(o.lower, 6)}, "
                        f"{o.upper if o.upper is None else round(o.upper, 6)}]  {o.label}  {o.status}")
        return "\n".join(rows)


# ----------------------------------------------------------------------------- validation
def validate(problem, options, log):
    f, fld, mu = problem.functional, problem.field, problem.measure
    issues = []
    backend = options.get("backend", "auto")
    if backend == "covfourier":
        if not (isinstance(fld, FBM) and 0.5 <= fld.H < 1.0):
            issues.append("UNSUPPORTED: CovFourier backend requires fBm with 1/2 <= H < 1")
    s, w = rule_1d(mu, list(fld.singular_points()) + [fld.T], 8, 6)
    sig = np.sqrt(np.maximum(fld.var(s), 0.0))
    with np.errstate(all="ignore"):
        m2 = f.second_moment_diag(s, sig)
    if not np.all(np.isfinite(m2)) or np.max(m2) > options.get("max_second_moment", 1e12):
        issues.append("UNSUPPORTED: f(s, Y_s) not square integrable on the support of the measure")
    log.append({"step": "validate", "max_E_f2": float(np.nanmax(m2)) if m2.size else None, "issues": issues})
    return issues


# ----------------------------------------------------------------------------- routes
def _moments(problem, log, tol_mom):
    fld, f, mu = problem.field, problem.functional, problem.measure
    t0 = time.perf_counter()
    m1 = first_moment(fld, f, mu)
    m1b = first_moment(fld, f, mu, n=16, levels=11)
    m2, gap2 = second_moment(fld, f, mu)
    var = m2 - m1b ** 2
    gap_mean = abs(m1b - m1)
    gap_var = gap2 + 2 * abs(m1b) * gap_mean
    log.append({"step": "moments", "seconds": time.perf_counter() - t0, "mean_gap": gap_mean, "m2_gap": gap2})
    lab = "EXACT-IDENTITY+QUADRATURE"
    out = [OutputResult("mean", None, m1b, m1b - gap_mean, m1b + gap_mean, lab,
                        "ISSUED" if gap_mean <= tol_mom else "UNRESOLVED", {"quad_gap": gap_mean}),
           OutputResult("variance", None, var, var - gap_var, var + gap_var, lab,
                        "ISSUED" if gap_var <= tol_mom else "UNRESOLVED", {"quad_gap": gap_var})]
    return out, m1b, m2


def _deterministic_quantile(problem, alpha, tol, options, log, want_es):
    """Route D: rank-one output-adapted surrogate with exact radius; grow the anchor set."""
    fld, f, mu = problem.field, problem.functional, problem.measure
    best = None
    for k in options.get("adapted_anchor_sizes", (1, 4, 16)):
        t0 = time.perf_counter()
        anchors = fld.T * np.arange(1, k + 1) / k
        try:
            rep = OutputAdaptedProjection(fld, f, mu, anchors) if k > 1 or options.get("adapted_k1", True) \
                else AnchorProjection(fld, [fld.T / 2])
        except ValueError:
            rep = AnchorProjection(fld, [fld.T / 2])
        L = compute_ledger(fld, rep, f, mu, mu, N=None)
        G = OneDimSurrogate(fld, rep, f, mu)
        law = law_from_map(G)
        eps = math.sqrt(max(L.P_mu, 0.0) + L.quad_gap)
        w2 = certify_quantile(law, alpha, {2: eps}, support=problem.support)
        if options.get("endpoints", "martingale") == "martingale":
            # G(Z) = E[Lambda | Z] (full chaos): martingale-sharp endpoints; the law discretisation and spline
            # interpolation are sigma(Z)-measurable perturbations (level shift)
            from .martingale import certify_quantile_martingale, es_bracket_martingale
            dlt = law.radius_disc + G.interp_gap
            c = certify_quantile_martingale(law=law, alpha=alpha, P=eps ** 2, delta=dlt)
            cert = _Cert(c["lower"], c["upper"], c["center"], "certified-numerical")
            if problem.support is not None:
                cert.lower, cert.upper = max(cert.lower, problem.support[0]), min(cert.upper, problem.support[1])
            es = es_bracket_martingale(None, alpha, eps ** 2, law=law, delta=dlt) if want_es else None
        else:
            cert = w2
            es = es_bracket(law, alpha, eps) if want_es else None
        rec = {"route": "D", "k_anchors": k, "P": L.P_mu, "eps": eps, "disc": law.radius_disc,
               "width": cert.width, "width_w2": w2.width, "seconds": time.perf_counter() - t0}
        log.append(rec)
        best = (cert, es, rec)
        if cert.width <= 2 * tol and (es is None or es["upper"] - es["lower"] <= 2 * tol):
            return best, True
    return best, False


def _sample_field(field, times, K, rng, chunk=20000):
    C = field.R(times[:, None], times[None, :])
    L = cholesky(C + 1e-14 * np.trace(C) / times.size * np.eye(times.size), lower=True)
    for c0 in range(0, K, chunk):
        k = min(chunk, K - c0)
        yield rng.standard_normal((k, times.size)) @ L.T


class _Cert:
    """Quantile bracket record (same fields as transfer.QuantileCertificate)."""

    def __init__(self, lower, upper, center, label):
        self.lower, self.upper, self.center, self.label = lower, upper, center, label

    @property
    def width(self):
        return self.upper - self.lower


_CE_FAMILIES = ("WickExponential", "PathwiseExponential", "Indicator", "PositivePart", "ExpCall", "Polynomial")


def _mc_schedule(options, stop_after_first=False):
    """Fix all candidate sample sizes before inspecting the certification samples."""
    initial, cap = options.get("mc_initial", 50_000), options.get("mc_max", 2_000_000)
    if initial < 1 or cap < 1 or int(initial) != initial or int(cap) != cap:
        raise ValueError("mc_initial and mc_max must be positive integers")
    sizes = [min(int(initial), int(cap))]
    while sizes[-1] < cap and not stop_after_first:
        sizes.append(min(4 * sizes[-1], int(cap)))
    return sizes


def _stage_confidence(a, sizes, families=1):
    if not 0 < a < 1:
        raise ValueError("confidence_a must lie strictly between 0 and 1")
    # A union bound over the fixed schedule preserves confidence at the selected stopping stage.
    return a / (len(sizes) * families)


def _ce_supported(problem):
    mu = problem.measure
    return (type(problem.functional).__name__ in _CE_FAMILIES and mu.has_density and not mu.density_breaks
            and not mu.atoms and abs(mu.lo) < 1e-15 and abs(mu.hi - problem.field.T) < 1e-12)


def _ce_quantile(problem, alpha, tol, options, log, want_es):
    """Route H with the conditional-expectation monitor E[Lambda | Y_grid] (Lemma cemonitor): exact radius
    P_mu(Pi_n) <= M(n), martingale-sharp endpoints with DKW band, convex-order ES bracket.  With
    options['ce_endpoints'] = 'condvar' and a quadratic output (zero mean function), the endpoints are sharp
    given the conditional variance (Proposition condvar), from closed-form quadratic forms."""
    from .monitor import (ConditionalMonitor, ce_radius, quadratic_cond_mean, quadratic_cond_var, quadratic_delta,
                          cantelli_dominating, cantelli_dominating_binomial)
    from .martingale import certify_quantile_martingale, es_bracket_martingale
    from .functionals import Polynomial
    fld, f, mu = problem.field, problem.functional, problem.measure
    a = options.get("confidence_a", 0.01)
    phi = options.get("bias_fraction", 0.6)
    rng = np.random.default_rng(options.get("seed", 12345))
    max_K, max_n = options.get("mc_max", 2_000_000), options.get("n_max", 1024)
    dens = float(mu.density(np.array([0.5 * (mu.lo + mu.hi)]))[0])
    T = mu.hi
    use_cv = (options.get("ce_endpoints", "martingale") == "condvar" and isinstance(f, Polynomial)
              and len(f.a) <= 3 and not np.any(f.mean(np.linspace(mu.lo, mu.hi, 7))))

    def monitor(n):
        cm = ConditionalMonitor(fld, f, T, T * (np.arange(n) + 0.5) / n, weight=dens)
        if use_cv:
            a0, a1, a2 = np.concatenate([f.a, np.zeros(3 - len(f.a))])
            d0, l, A, ga = quadratic_cond_mean(cm, a0=a0, a1=a1, a2=a2)
            c0, b, Om, gv = quadratic_cond_var(cm, a1=a1, a2=a2, norms=True)
            cm.qf = (d0, l, A, c0, b, Om, gv)
            P = c0 + float(np.trace(Om)) + gv[0] + gv[1] * math.sqrt(n) + gv[2] * n     # = E v
            return cm, P, quadratic_delta(ga, n)
        P, gap = ce_radius(cm)
        cmf = ConditionalMonitor(fld, f, T, cm.grid, q=12, levels=6, weight=dens)
        xa = np.random.default_rng(7).standard_normal((1000, n))
        dq = 10.0 * float(np.sqrt(np.mean((cm.estimate(xa) - cmf.estimate(xa)) ** 2)))
        return cm, P + gap, dq

    def draw(cm, K):
        if not use_cv:
            return cm.sample(K, rng), None
        d0, l, A, c0, b, Om, gv = cm.qf
        xi = rng.standard_normal((K, cm.grid.size))
        r2 = np.sum(xi * xi, axis=1)
        return (d0 + xi @ l + np.sum((xi @ A) * xi, axis=1),
                c0 + xi @ b + np.sum((xi @ Om) * xi, axis=1) + gv[0] + gv[1] * np.sqrt(r2) + gv[2] * r2)

    def bracket(x, v, P2, dq, dkw):
        """dkw = None: population-level endpoints of the empirical law (allocation only)."""
        if use_cv and dkw is not None and options.get("ce_order_statistics", False):
            c = cantelli_dominating_binomial(x, v, alpha, dkw, dq, rng=rng)      # exact order statistics
            c["label"] = f"statistical(1-{dkw:g}), sharp given V, order statistics"
            return c
        if use_cv:
            c = cantelli_dominating(x, v, alpha, 0.05 if dkw is None else dkw, dq, rng=rng,
                                    eps=0.0 if dkw is None else None)
            c["label"] = "population" if dkw is None else f"statistical(1-{dkw:g}), sharp given V"
            return c
        if dkw is None:
            return certify_quantile_martingale(law=law_from_samples(x), alpha=alpha, P=P2, delta=dq)
        return certify_quantile_martingale(x, alpha, P2, dkw_a=dkw, delta=dq)

    n = options.get("n_initial", 16)
    cm, P2, dq = monitor(n)
    pilot, vpil = draw(cm, options.get("mc_pilot", 20_000))
    while True:
        wb = bracket(pilot, vpil, P2, dq, None)
        wb = wb["upper"] - wb["lower"]
        log.append({"route": "C-bias", "n": n, "P": P2, "delta_q": dq, "bias_width": wb,
                    "endpoints": "condvar" if use_cv else "martingale"})
        if wb <= phi * 2 * tol or n >= max_n:
            break
        n *= 2
        cm, P2, dq = monitor(n)
        if use_cv:
            pilot, vpil = draw(cm, options.get("mc_pilot", 20_000))
    bias_floor_exceeded = wb > 2 * tol
    A2, A2_gap = second_moment(fld, f, mu)
    A2_upper = max(A2, 0.0) + abs(A2_gap)
    sizes = _mc_schedule(options, bias_floor_exceeded)
    # The conditional-variance endpoints use T+/T- events; ES uses the empirical law of X.
    # The martingale endpoints and ES, in contrast, share one two-sided DKW event for X.
    families = 2 if use_cv and want_es else 1
    stage_a = _stage_confidence(a, sizes, families)
    for attempt, K in enumerate(sizes, 1):
        t0 = time.perf_counter()
        samples, vs = draw(cm, K)
        c = bracket(samples, vs, P2, dq, stage_a)
        label = c["label"].replace(f"1-{stage_a:g}", f"1-{a:g}")
        cert = _Cert(c["lower"], c["upper"], c["center"], label)
        es = None
        if want_es:
            # Jensen gives E[X_exact^2] <= E[Lambda^2]. P2 is an UPPER bound on the residual
            # variance, so subtracting it from A2 would not give an upper bound on E[X_exact^2].
            ex2 = (math.sqrt(A2_upper) + dq) ** 2
            es = es_bracket_martingale(samples, alpha, P2, second_moment_X=ex2, dkw_a=stage_a, delta=dq)
        rec = {"route": "C", "n": n, "K": K, "P": P2, "delta_q": dq, "dkw": dkw_delta(K, stage_a), "width": cert.width,
               "confidence_a": a, "stage_a": stage_a, "mc_stages": len(sizes), "attempt": attempt,
               "confidence_families": families,
               "es_second_moment_upper": None if es is None else ex2,
               "es_width": None if es is None else es["upper"] - es["lower"],
               "bias_floor_exceeded": bool(bias_floor_exceeded), "seconds": time.perf_counter() - t0}
        log.append(rec)
        ok = cert.width <= 2 * tol and (es is None or es["upper"] - es["lower"] <= 2 * tol)
        if ok or K >= max_K or bias_floor_exceeded:
            if bias_floor_exceeded and not ok:
                rec["unresolved_reason"] = f"pilot bracket at n_max={n} exceeds the tolerance before sampling allowance"
            return (cert, es, rec), ok


def _hybrid_quantile(problem, alpha, tol, options, log, want_es):
    """Route H: Monte Carlo on the n-point monitored functional + exact bias radius + DKW band.

    With options['monitor'] == 'ce' (default where supported) the conditional-expectation monitor and
    martingale-sharp endpoints are used (_ce_quantile); 'riemann' keeps the n-point sum and W_2 endpoints.
    Allocation: (i) pilot law at n0; (ii) smallest n (doubling) whose exact bias radius
    eps(n) = sqrt(M(n)) alone uses at most a fraction phi of the tolerance budget; if no
    n <= n_max qualifies, stop: UNRESOLVED (bias floor); (iii) smallest K (x4 steps) whose DKW band
    brings the total width within tolerance; if K_max is reached, UNRESOLVED (sampling budget).
    """
    if options.get("monitor", "ce") == "ce" and _ce_supported(problem):
        return _ce_quantile(problem, alpha, tol, options, log, want_es)
    fld, f, mu = problem.field, problem.functional, problem.measure
    a = options.get("confidence_a", 0.01)
    phi = options.get("bias_fraction", 0.6)
    rng = np.random.default_rng(options.get("seed", 12345))
    max_K, max_n = options.get("mc_max", 2_000_000), options.get("n_max", 1024)
    atomic_target = not mu.has_density
    if mu.has_density and mu.density_breaks:
        raise NotImplementedError("hybrid route implemented for constant densities")
    dens = None if atomic_target else float(mu.density(np.array([0.5 * (mu.lo + mu.hi)]))[0])

    def monitor(n):
        return mu if atomic_target else Measure.riemann(n, T=mu.hi, rule="mid", lo=mu.lo, weight=dens)

    def draw(nu, K):
        t_nu, w_nu = nu.atom_arrays()
        return np.concatenate([f.f(np.broadcast_to(t_nu, Y.shape), Y) @ w_nu
                               for Y in _sample_field(fld, t_nu, K, rng)])

    def widths(law, eps, d):
        if d is None:
            c = certify_quantile(law, alpha, {2: eps}, support=problem.support)
        else:
            c = certify_quantile(law, alpha, {2: eps}, support=problem.support, dkw_a=d)
        w = c.width
        if want_es:
            w = max(w, 2 * eps / math.sqrt(1 - alpha))
        return c, w

    n = options.get("n_initial", 16)
    pilot = law_from_samples(draw(monitor(n), options.get("mc_pilot", 20_000)))
    # (ii) monitoring level from the exact bias radius
    chosen = None
    while True:
        nu = monitor(n)
        L = compute_ledger(fld, IdentityRepresentation(fld), f, mu, nu, N=None)
        eps = math.sqrt(max(L.M, 0.0) + L.quad_gap)
        _, wb = widths(pilot, eps, None)
        log.append({"route": "H-bias", "n": len(nu.atoms), "M": L.M, "eps": eps, "bias_width": wb})
        if wb <= phi * 2 * tol:
            chosen = (n, nu, L, eps)
            break
        if atomic_target or n >= max_n:
            chosen = (n, nu, L, eps)
            break
        n *= 2
    n, nu, L, eps = chosen
    bias_floor_exceeded = wb > 2 * tol           # pilot-based stop; not a proof of population infeasibility
    sizes = _mc_schedule(options, bias_floor_exceeded)
    stage_a = _stage_confidence(a, sizes)
    result = None
    for attempt, K in enumerate(sizes, 1):
        t0 = time.perf_counter()
        samples = draw(nu, K)
        law = law_from_samples(samples)
        cert, _ = widths(law, eps, stage_a)
        cert.label = f"statistical(1-{a:g})"
        es = None
        if want_es:
            t_nu, w_nu = nu.atom_arrays()
            es = es_bracket(law, alpha, eps, second_moment_Y=_second_moment_atomic(fld, f, t_nu, w_nu), dkw_a=stage_a)
        rec = {"route": "H", "n": len(nu.atoms), "K": K, "M": L.M, "eps": eps, "dkw": dkw_delta(K, stage_a),
               "confidence_a": a, "stage_a": stage_a, "mc_stages": len(sizes), "attempt": attempt,
               "confidence_families": 1,
               "width": cert.width, "es_width": None if es is None else es["upper"] - es["lower"],
               "bias_floor_exceeded": bool(bias_floor_exceeded), "seconds": time.perf_counter() - t0}
        log.append(rec)
        result = (cert, es, rec)
        ok = cert.width <= 2 * tol and (es is None or es["upper"] - es["lower"] <= 2 * tol)
        if ok or K >= max_K or bias_floor_exceeded:
            if bias_floor_exceeded and not ok:
                rec["unresolved_reason"] = f"pilot bracket at n_max={len(nu.atoms)} exceeds the tolerance before sampling allowance"
            return result, ok


def _second_moment_atomic(field, func, t, w):
    T1, T2 = np.meshgrid(t, t, indexing="ij")
    sg = np.sqrt(np.maximum(field.var(t), 0.0))
    S1, S2 = np.meshgrid(sg, sg, indexing="ij")
    K = func.kernel(T1.ravel(), T2.ravel(), field.R(T1, T2).ravel(), S1.ravel(), S2.ravel())
    return float(w @ np.asarray(K).reshape(T1.shape) @ w)


def _surrogate(problem, tol_rms, options, log):
    """Certified chaos surrogate Q_{k,N} Lambda_{nu_n} at minimal declared cost within a lattice.

    The exact ledger factorises by axis (terms in n only, in k (all N at once), and in (n, k)), so the
    whole lattice is evaluated from cached pieces (allocation.LedgerGrid) and the cheapest candidate whose
    exact error plus quadrature allowance meets the tolerance is returned -- optimal within the declared
    lattice by construction (E4/E4b showed greedy rules to be unreliable).  If no lattice point qualifies,
    the status is UNRESOLVED and the most accurate candidate is reported with the reason.
    """
    from .allocation import LedgerGrid, LazyLedger, best_first, cost
    fld, f, mu = problem.field, problem.functional, problem.measure
    if not mu.has_density:
        raise NotImplementedError("surrogate route implemented for density measures")
    ns = options.get("sur_ns", [2, 4, 8, 16, 32, 64, 128])
    ks = options.get("sur_ks", [1, 2, 4, 8, 16])
    Nmax = options.get("sur_Nmax", 8)
    t0 = time.perf_counter()
    tol2 = tol_rms ** 2
    if options.get("allocation", "best_first") == "best_first":
        # certified best-first search with the Pythagorean bounds (Proposition pythagoras): exact optimum
        L = LazyLedger(fld, f, mu, Nmax)
        bf = best_first(L, ns, ks, Nmax, tol2)
        log.append({"step": "surrogate-best-first", "candidates": len(ns) * len(ks) * Nmax, **{
            k: bf[k] for k in ("visited", "pruned_lower", "accepted_upper", "exact")}, "counts": bf["counts"],
            "seconds": time.perf_counter() - t0, "quad_gap": L.gap})
        if bf["cost"] is None:
            return OutputResult("surrogate", None, None, 0.0, None, "EXACT-IDENTITY+QUADRATURE", "UNRESOLVED",
                                {"reason": "no lattice point meets the tolerance (certified by the lower bound "
                                           "P_mu + C_mu or the exact total)", "counts": bf["counts"]})
        e = bf["bound"]
        return OutputResult("surrogate", None, math.sqrt(max(e, 0.0)), 0.0, math.sqrt(max(e + L.gap, 0.0)),
                            "EXACT-IDENTITY+QUADRATURE", "ISSUED",
                            {"n": bf["n"], "k": bf["k"], "N": bf["N"], "cost": bf["cost"], "how": bf["how"],
                             "counts": bf["counts"]})
    grid = LedgerGrid(fld, f, mu, ns=ns, ks=ks, Nmax=Nmax)
    tol2 = tol_rms ** 2
    best, most_accurate = None, None
    for n in ns:
        for k in ks:
            for N in range(1, Nmax + 1):
                e = grid.ledger(n, k, N)["total"]
                if most_accurate is None or e < most_accurate[0]:
                    most_accurate = (e, n, k, N)
                if e + grid.gap <= tol2 and (best is None or cost(n, k, N) < best[0]):
                    best = (cost(n, k, N), n, k, N, e)
    log.append({"step": "surrogate-lattice", "candidates": len(ns) * len(ks) * Nmax,
                "seconds": time.perf_counter() - t0, "quad_gap": grid.gap})
    if best is None:
        e, n, k, N = most_accurate
        return OutputResult("surrogate", None, math.sqrt(max(e, 0.0)), 0.0, math.sqrt(max(e + grid.gap, 0.0)),
                            "EXACT-IDENTITY+QUADRATURE", "UNRESOLVED",
                            {"n": n, "k": k, "N": N, "cost": cost(n, k, N), "ledger": grid.ledger(n, k, N),
                             "reason": "no lattice point meets the tolerance (certified by the exact ledger)"})
    c, n, k, N, e = best
    return OutputResult("surrogate", None, math.sqrt(max(e, 0.0)), 0.0, math.sqrt(max(e + grid.gap, 0.0)),
                        "EXACT-IDENTITY+QUADRATURE", "ISSUED",
                        {"n": n, "k": k, "N": N, "cost": c, "ledger": grid.ledger(n, k, N)})


# ----------------------------------------------------------------------------- entry point
def solve(problem, outputs, tol, options=None):
    """problem: FunctionalProblem; outputs: list like ["mean","variance",("quantile",0.975),("ES",0.975),
    ("surrogate",)]; tol: dict with keys 'moment', 'quantile', 'es', 'surrogate_rms' (absolute)."""
    options = dict(options or {})
    log, res = [], []
    t_start = time.perf_counter()
    issues = validate(problem, options, log)
    if issues:
        for o in outputs:
            kind = o if isinstance(o, str) else o[0]
            res.append(OutputResult(kind, None if isinstance(o, str) else (o[1] if len(o) > 1 else None),
                                    label="UNSUPPORTED", status="UNSUPPORTED", details={"reasons": issues}))
        return SolveReport(problem.name, res, {"seconds": time.perf_counter() - t_start}, log)
    kinds = [o if isinstance(o, str) else o[0] for o in outputs]
    if "mean" in kinds or "variance" in kinds:
        mom, _, _ = _moments(problem, log, tol.get("moment", 1e-6))
        res += [m for m in mom if m.kind in kinds]
    for o in outputs:
        kind = o if isinstance(o, str) else o[0]
        if kind in ("quantile", "ES"):
            alpha = o[1]
            if any(r.kind in ("quantile", "ES") and r.level == alpha for r in res):
                continue
            want_es = any((not isinstance(x, str)) and x[0] == "ES" and x[1] == alpha for x in outputs)
            tq = tol.get("quantile", 0.05)
            (cert, es, rec), ok = _deterministic_quantile(problem, alpha, tq, options, log, want_es)
            if not ok and options.get("allow_mc", True):
                (cert, es, rec), ok = _hybrid_quantile(problem, alpha, tq, options, log, want_es)
            status = "ISSUED" if ok else "UNRESOLVED"
            lab = "CERTIFIED-NUMERICAL" if rec["route"] == "D" else cert.label.upper()
            res.append(OutputResult("quantile", alpha, cert.center, cert.lower, cert.upper, lab, status, rec))
            if es is not None:
                res.append(OutputResult("ES", alpha, es["center"], es["lower"], es["upper"],
                                        lab, status, rec))
        elif kind == "surrogate":
            res.append(_surrogate(problem, tol.get("surrogate_rms", 1e-2), options, log))
    return SolveReport(problem.name, res, {"seconds": time.perf_counter() - t_start}, log)
