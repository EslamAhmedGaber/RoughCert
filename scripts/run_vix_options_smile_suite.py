"""Full Arb-Certified VIX Options & Implied-Volatility Smile Competitor Suite.

Covers five modern published VIX option / smile benchmarks (all 160-bit outward-rounded Arb):
  1. BDG (2023, Quantitative Finance, Fig. 6): H=0.10, T=0.25, a=1.0, Delta=1/12, xi0=0.235^2
     9-strike VIX call/put/IV smile comparing Arb CondVar vs BDG Order-3 Arb vs Lognormal Proxy
     vs Tilted Option Lipschitz control sqrt(2*U*(U-L)).
  2. Stress Cell VIX Smile (E14_stress_H010): H=0.10, T=0.5, eta_P1=3.0 (a=1.34164), Delta=30/365
     6 strikes K/F in {0.70, 0.85, 1.00, 1.25, 1.50, 2.00} — Arb CondVar vs BDG Order-3 Arb vs LN!
  3. Bourgey & De Marco (2022, JCF, Sec. 3.3) MLMC ATM Call Audit:
     H=0.10, T=0.5, a=0.5, Delta=1/12, xi0=0.235^2, K=0.2218135 (published ATM call = 2.98840) —
     contrasting Continuous-Time Arb Call vs 500-cell Right-Rectangle Arb Call!
  4. Liao, Agarwal, & Bourgey (2026, arXiv:2604.25123) VIX IV Smile Benchmark:
     xi0=0.24^2, T=0.25, Delta=1/12, (H, a) = (0.10, 1.0) across 7 log-moneyness points.
  5. Mixed Rough Bergomi Upward-Sloping VIX IV Smile (BDG 2023 Sec. 4, Scenario 1):
     H=0.10, T=0.25, Delta=1/12, xi0=0.235^2, (a1, a2, lambda) = (1.4, 0.7, 0.3) across 7 strikes!

Outputs verified JSON to: results/vix_options_smile_suite.json
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from flint import arb

from arb_vix_core import (
    set_dps,
    arb_lo_float,
    arb_hi_float,
    tilted_and_lognormal_flat_arb,
    bdg_futures_flat_arb,
)
from arb_vix_options import (
    make_flat_curve,
    build_condvar_arb_model,
    condvar_vix_futures_and_options_arb,
    discrete_right_rect_condvar_options_arb,
    black76_call_arb,
    black76_iv_bracket_arb,
    _phi_arb,
    _Phi_arb,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def bdg_call_and_iv_flat_arb(
    H: arb, T: arb, Delta: arb, a: arb, xi0_val: arb, strikes_decimal: list[float], dps: int = 40
) -> dict:
    """Exact 160-bit Arb evaluation of Bourgey, De Marco, & Gobet (2023, QF, Thm. 2.2)
    Order-0 and Order-3 VIX Call prices and Black-76 implied volatilities (matching price_vix_approx)."""
    set_dps(dps)
    bdg = bdg_futures_flat_arb(H, T, Delta, a, xi0_val, dps=dps)
    S = bdg["S_ball"]
    F3 = bdg["F3_ball"]
    g1 = bdg["gamma_1_ball"]
    g2 = bdg["gamma_2_ball"]
    g3 = bdg["gamma_3_ball"]

    # tot_varp = ||h||^2; tau = 0.5 * sqrt(tot_varp)
    from arb_vix_core import norm_h_sq_flat_arb
    varp = norm_h_sq_flat_arb(H, T, Delta, a)
    tau = varp.sqrt() / arb(2)
    sigma0 = tau / T.sqrt()

    out_rows = []
    for k_f in strikes_decimal:
        K = arb(str(k_f))
        d1 = (S / K).log() / tau + tau / arb(2)
        d2 = d1 - tau
        P0 = S * _Phi_arb(d1) - K * _Phi_arb(d2)
        P1 = (S / arb(2)) * _Phi_arb(d1)
        P2 = P1 / arb(2) + (S / arb(4)) * _phi_arb(d1) / tau
        P3 = -P1 / arb(2) + (arb(3) / arb(2)) * P2 - (S / arb(8)) * _phi_arb(d1) * (arb(1) + d1 / tau) / tau
        C3 = P0 + g1 * P1 + g2 * P2 + g3 * P3

        # Implied vol of BDG Order-3 call using BDG's own Order-3 forward F3 (exact convention of implied_vol_vix_approx)
        intrinsic = (F3-K).max(arb(0))
        if C3.lower() > intrinsic.upper() and C3.upper() < F3.lower():
            iv3_lo, iv3_hi, iv3_ok = black76_iv_bracket_arb(F3.lower(),F3.upper(),K,T,C3.lower(),C3.upper(),dps=dps)
        else:
            iv3_lo, iv3_hi, iv3_ok = 0.0, float("inf"), False

        out_rows.append({
            "K_decimal": float(k_f),
            "K_volpts": float(100.0 * k_f),
            "bdg_C0_volpts": [arb_lo_float(arb(100) * P0), arb_hi_float(arb(100) * P0)],
            "bdg_C3_volpts": [arb_lo_float(arb(100) * C3), arb_hi_float(arb(100) * C3)],
            "bdg_iv0_pct": 100.0 * float(sigma0),
            "bdg_iv3_pct": [100.0 * iv3_lo, 100.0 * iv3_hi] if iv3_ok else None,
        })

    return {
        "bdg_F0_volpts": [arb_lo_float(arb(100) * S), arb_hi_float(arb(100) * S)],
        "bdg_F3_volpts": [arb_lo_float(arb(100) * F3), arb_hi_float(arb(100) * F3)],
        "bdg_sigma0_pct": 100.0 * float(sigma0),
        "options": out_rows,
    }


def lognormal_call_and_iv_arb(xi0_val: arb, M2_ball: arb, T: arb, strikes_decimal: list[float], dps: int = 35) -> dict:
    """Exact moment-matched Lognormal VIX Call & flat Black-76 IV in Arb.
    For Lambda_LN ~ LogNormal(mu, s2) with E[Lambda]=M1=xi0, E[Lambda^2]=M2:
      s2 = log(M2 / M1^2)  =>  sqrt(Lambda_LN) ~ LogNormal(mu/2, s2/4)
      Forward F_LN = M1^(3/4) * M2^(-1/8), Black-76 vol sigma_LN = sqrt(s2 / (4*T))!
    """
    set_dps(dps)
    s2 = (M2_ball / (xi0_val * xi0_val)).log()
    F_LN = (xi0_val ** (arb(3) / arb(4))) * (M2_ball ** (arb(-1) / arb(8)))
    sigma_LN = (s2 / (arb(4) * T)).sqrt()
    rows = []
    for k_f in strikes_decimal:
        K = arb(str(k_f))
        C_LN = black76_call_arb(F_LN, K, T, sigma_LN)
        rows.append({
            "K_decimal": float(k_f),
            "K_volpts": float(100.0 * k_f),
            "ln_call_volpts": [arb_lo_float(arb(100) * C_LN), arb_hi_float(arb(100) * C_LN)],
            "ln_iv_pct": [arb_lo_float(arb(100) * sigma_LN), arb_hi_float(arb(100) * sigma_LN)],
        })
    return {
        "ln_F_volpts": [arb_lo_float(arb(100) * F_LN), arb_hi_float(arb(100) * F_LN)],
        "ln_iv_pct": [arb_lo_float(arb(100) * sigma_LN), arb_hi_float(arb(100) * sigma_LN)],
        "options": rows,
    }


def bdg_mixed_smile_float(H: float, T: float, Delta: float, a1: float, a2: float, lbd: float, xi0_val: float, strikes_decimal: list[float]) -> dict:
    """BDG (2023) mixed rough Bergomi order-0 and order-3 VIX calls and IVs (matching price_vix_approx_mixed)."""
    from arb_vix_core import bdg_futures_independent_scipy
    b1 = bdg_futures_independent_scipy(H, T, Delta, a1, xi0_val, n_quad=96)
    b2 = bdg_futures_independent_scipy(H, T, Delta, a2, xi0_val, n_quad=96)
    e1, e2 = a1 / math.sqrt(2.0 * H), a2 / math.sqrt(2.0 * H)
    mp_unit = -((T + Delta) ** (2 * H + 1) - Delta ** (2 * H + 1) - T ** (2 * H + 1)) / (2.0 * Delta * (2 * H + 1))
    mp1, mp2 = (e1 ** 2) * mp_unit, (e2 ** 2) * mp_unit
    sp1, sp2 = math.sqrt(b1["varp"]), math.sqrt(b2["varp"])

    n_gh = 100
    gx, gw = np.polynomial.hermite.hermgauss(n_gh)
    nodes = math.sqrt(2.0) * gx
    weights = gw / math.sqrt(math.pi)
    mu0 = mp_unit
    eps = 1e-4

    def eval_payoff(payoff_fn):
        p0 = float(np.sum(weights * payoff_fn(xi0_val * (lbd * np.exp(mp1 + sp1 * nodes) + (1.0 - lbd) * np.exp(mp2 + sp2 * nodes)))))

        def psi(x: float, idx: int) -> float:
            if idx == 1:
                def f(y: float) -> float:
                    t0 = lbd * math.exp(mp1 + sp1 * x + y)
                    t1 = (1.0 - lbd) * math.exp(e2 * (e1 - e2) * (-mu0) + (e2 / e1) * (mp1 + sp1 * x))
                    return float(payoff_fn(xi0_val * (t0 + t1)))
            else:
                def f(y: float) -> float:
                    t0 = (1.0 - lbd) * math.exp(mp2 + sp2 * x + y)
                    t1 = lbd * math.exp(e1 * (e2 - e1) * (-mu0) + (e1 / e2) * (mp2 + sp2 * x))
                    return float(payoff_fn(xi0_val * (t0 + t1)))
            return (f(eps) - f(-eps)) / (2.0 * eps)

        psi1 = np.array([psi(float(x), 1) for x in nodes])
        psi2 = np.array([psi(float(x), 2) for x in nodes])
        p1 = b1["gamma_1"] * float(np.sum(weights * psi1)) + b2["gamma_1"] * float(np.sum(weights * psi2))
        p2 = (
            b1["gamma_2"] * float(np.sum(weights * nodes * psi1)) / sp1
            + b2["gamma_2"] * float(np.sum(weights * nodes * psi2)) / sp2
        )
        p3 = (
            b1["gamma_3"] * float(np.sum(weights * (nodes ** 2 - 1.0) * psi1)) / (sp1 ** 2)
            + b2["gamma_3"] * float(np.sum(weights * (nodes ** 2 - 1.0) * psi2)) / (sp2 ** 2)
        )
        return p0, p0 + p1 + p2 + p3

    F0, F3 = eval_payoff(lambda v: np.sqrt(v))
    rows = []
    for k_f in strikes_decimal:
        C0, C3 = eval_payoff(lambda v, K=k_f: np.maximum(np.sqrt(v) - K, 0.0))
        # Black-76 IV of C3 with forward F3
        F3_a, K_a, T_a, C3_a = arb(f"{F3:.15g}"), arb(str(k_f)), arb(f"{T:.15g}"), arb(f"{C3:.15g}")
        iv_lo, iv_hi, ok = black76_iv_bracket_arb(F3_a, F3_a, K_a, T_a, C3_a, C3_a, dps=30)
        rows.append({
            "K_decimal": float(k_f),
            "K_volpts": float(100.0 * k_f),
            "bdg_mix_C0_volpts": 100.0 * C0,
            "bdg_mix_C3_volpts": 100.0 * C3,
            "bdg_mix_iv3_pct": 50.0 * (iv_lo + iv_hi) if ok else None,
        })
    return {"bdg_mix_F0_volpts": 100.0 * F0, "bdg_mix_F3_volpts": 100.0 * F3, "options": rows}


def run_smile_case(
    name: str,
    H: arb,
    T: arb,
    Delta: arb,
    a: arb,
    xi0_str: str,
    log_moneyness_grid: list[float] | None = None,
    strike_ratios: list[float] | None = None,
    M_cv: int = 160,
    n_core: int = 280,
) -> dict:
    """Run full flat single-rBergomi smile comparison: Arb CondVar vs BDG Order-3 Arb vs LN Arb vs Tilted Lipschitz."""
    t0 = time.perf_counter()
    set_dps(35)
    xi0_val = arb(xi0_str) ** 2
    curve = make_flat_curve(xi0_str)

    # First compute Tilted + M2 (gives exact F_mid and Tilted Lipschitz control!)
    M_t = 240 if float(a) < 1.3 else 300
    til = tilted_and_lognormal_flat_arb(H, T, Delta, a, xi0_val, M=M_t, grade_p=3, dps=35)
    F_mid_dec = 0.005 * (til["L_outer_volpts"] + til["U_outer_volpts"])
    U_dec = 0.01 * til["U_outer_volpts"]
    L_dec = 0.01 * til["L_outer_volpts"]
    tilted_lipschitz_opt_volpts = 100.0 * math.sqrt(2.0 * U_dec * max(U_dec - L_dec, 0.0))

    if log_moneyness_grid is not None:
        strikes = [round(F_mid_dec * math.exp(k), 6) for k in log_moneyness_grid]
        k_labels = log_moneyness_grid
    else:
        strikes = [round(F_mid_dec * r, 6) for r in strike_ratios]
        k_labels = [math.log(r) for r in strike_ratios]

    t_build0 = time.perf_counter()
    mod = build_condvar_arb_model(H, T, Delta, [(a, arb(1))], curve, M=M_cv, grade_p=3, dps=35)
    t_setup = time.perf_counter() - t_build0
    t_s0 = time.perf_counter()
    cv = condvar_vix_futures_and_options_arb(mod, strikes, Zmax=10.0, n_core=n_core, n_tail_each=50, dps=35)
    t_solve = time.perf_counter() - t_s0

    bdg = bdg_call_and_iv_flat_arb(H, T, Delta, a, xi0_val, strikes, dps=40)
    ln = lognormal_call_and_iv_arb(xi0_val, til["M2"], T, strikes, dps=35)

    merged = []
    for i in range(len(strikes)):
        co = cv["options"][i]
        bo = bdg["options"][i]
        lo = ln["options"][i]
        # Signed certified deviations: positive if competitor > Arb_U, negative if competitor < Arb_L, 0 if inside
        bdg_c_lo, bdg_c_hi = bo["bdg_C3_volpts"]
        if bdg_c_lo > co["call_U_volpts"]:
            bdg_call_cert_err = bdg_c_lo - co["call_U_volpts"]
        elif bdg_c_hi < co["call_L_volpts"]:
            bdg_call_cert_err = bdg_c_hi - co["call_L_volpts"]
        else:
            bdg_call_cert_err = 0.0

        ln_c_lo, ln_c_hi = lo["ln_call_volpts"]
        if ln_c_lo > co["call_U_volpts"]:
            ln_call_cert_err = ln_c_lo - co["call_U_volpts"]
        elif ln_c_hi < co["call_L_volpts"]:
            ln_call_cert_err = ln_c_hi - co["call_L_volpts"]
        else:
            ln_call_cert_err = 0.0

        bdg_iv_cert_err = None
        if bo["bdg_iv3_pct"] is not None:
            bi_lo, bi_hi = bo["bdg_iv3_pct"]
            if bi_lo > co["iv_U_pct"]:
                bdg_iv_cert_err = bi_lo - co["iv_U_pct"]
            elif bi_hi < co["iv_L_pct"]:
                bdg_iv_cert_err = bi_hi - co["iv_L_pct"]
            else:
                bdg_iv_cert_err = 0.0

        row = {
            "log_moneyness": round(k_labels[i], 4),
            "K_over_F": round(strikes[i] / F_mid_dec, 4),
            **co,
            "bdg_C0_volpts": bo["bdg_C0_volpts"],
            "bdg_C3_volpts": bo["bdg_C3_volpts"],
            "bdg_iv3_pct": bo["bdg_iv3_pct"],
            "bdg_call_cert_sep_volpts": bdg_call_cert_err,
            "bdg_iv_cert_sep_pct": bdg_iv_cert_err,
            "ln_call_volpts": lo["ln_call_volpts"],
            "ln_iv_pct": lo["ln_iv_pct"],
            "ln_call_cert_sep_volpts": ln_call_cert_err,
        }
        merged.append(row)

    print(
        f"\n[{name}] Fut Arb=[{cv['future_L_volpts']:.5f}, {cv['future_U_volpts']:.5f}] (w={cv['future_width_volpts']:.5f}) | "
        f"Tilted Lip opt bound={tilted_lipschitz_opt_volpts:.4f} | BDG3 F={bdg['bdg_F3_volpts'][0]:.5f} | setup={t_setup:.1f}s solve={t_solve:.1f}s",
        flush=True,
    )
    for r in merged:
        bi_s = f"{r['bdg_iv3_pct'][0]:.2f}%" if r["bdg_iv3_pct"] is not None else "N/A"
        print(
            f"   k={r['log_moneyness']:+.2f} (K={r['K_volpts']:6.3f}): Call Arb=[{r['call_L_volpts']:.5f}, {r['call_U_volpts']:.5f}] (w={r['call_width_volpts']:.5f}) | "
            f"BDG3={r['bdg_C3_volpts'][0]:.5f} (sep={r['bdg_call_cert_sep_volpts']:+.5f}) | LN={r['ln_call_volpts'][0]:.5f} | "
            f"IV Arb=[{r['iv_L_pct']:.2f}%, {r['iv_U_pct']:.2f}%] vs BDG3={bi_s}, LN={r['ln_iv_pct'][0]:.2f}% (ok={r['iv_arb_verified']})",
            flush=True,
        )

    return {
        "name": name,
        "H": float(H),
        "T": float(T),
        "Delta": float(Delta),
        "a": float(a),
        "eta_P1": float(a) / math.sqrt(2.0 * float(H)),
        "xi0_base": float(xi0_str),
        "condvar_future_L": cv["future_L_volpts"],
        "condvar_future_U": cv["future_U_volpts"],
        "condvar_future_width": cv["future_width_volpts"],
        "tilted_future_L": til["L_outer_volpts"],
        "tilted_future_U": til["U_outer_volpts"],
        "tilted_lipschitz_opt_bound_volpts": tilted_lipschitz_opt_volpts,
        "bdg_F3_volpts": bdg["bdg_F3_volpts"],
        "ln_F_volpts": ln["ln_F_volpts"],
        "setup_s": round(t_setup, 2),
        "solve_s": round(t_solve, 2),
        "total_s": round(time.perf_counter() - t0, 2),
        "strikes": merged,
    }


def main():
    t_all = time.perf_counter()

    # 1. BDG (2023) Fig. 6 VIX Smile (H=0.10, T=0.25, a=1.0, Delta=1/12, xi0=0.235^2)
    case_bdg_fig6 = run_smile_case(
        "BDG2023_Fig6_T3m",
        H=arb("0.10"),
        T=arb(1) / arb(4),
        Delta=arb(1) / arb(12),
        a=arb("1.0"),
        xi0_str="0.235",
        log_moneyness_grid=[-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4],
        M_cv=160,
        n_core=280,
    )

    # 2. E14 Stress Cell VIX Smile (H=0.10, T=0.5, eta_P1=3.0, Delta=30/365, xi0=0.235^2)
    H10 = arb("0.10")
    case_stress = run_smile_case(
        "E14_stress_H010_T6m",
        H=H10,
        T=arb(1) / arb(2),
        Delta=arb(30) / arb(365),
        a=arb("3.0") * (arb(2) * H10).sqrt(),
        xi0_str="0.235",
        strike_ratios=[0.70, 0.85, 1.00, 1.25, 1.50, 2.00],
        M_cv=200,
        n_core=320,
    )

    # 3. Bourgey & De Marco (2022, JCF, Sec. 3.3) MLMC ATM Call Audit: Continuum vs 500-cell Right-Rect!
    print("\n=== BD2022 MLMC ATM Call Audit (Continuum Arb vs 500-cell Right-Rect Arb vs Published 2.98840) ===", flush=True)
    t_bd0 = time.perf_counter()
    H_bd, T_bd, D_bd, a_bd, xi_bd = arb("0.10"), arb(1) / arb(2), arb(1) / arb(12), arb("0.5"), arb("0.235") ** 2
    K_bd = 0.2218135
    mod_bd = build_condvar_arb_model(H_bd, T_bd, D_bd, [(a_bd, arb(1))], make_flat_curve("0.235"), M=160, grade_p=3, dps=35)
    cv_bd_cont = condvar_vix_futures_and_options_arb(mod_bd, [K_bd], Zmax=10.0, n_core=280, n_tail_each=45, dps=35)
    cv_bd_500 = discrete_right_rect_condvar_options_arb(H_bd, T_bd, D_bd, a_bd, xi_bd, [K_bd], n=500, Zmax=10.0, n_core=220, n_tail_each=40, dps=30)
    bdg_bd = bdg_call_and_iv_flat_arb(H_bd, T_bd, D_bd, a_bd, xi_bd, [K_bd], dps=40)
    case_bd2022 = {
        "name": "BD2022_MLMC_ATM_Call",
        "H": 0.10, "T": 0.5, "Delta": 1.0 / 12.0, "a": 0.5, "xi0_base": 0.235,
        "K_decimal": K_bd,
        "published_fut_volpts": 22.18135,
        "published_atm_call_volpts": 2.98840,
        "continuum_arb": cv_bd_cont,
        "grid500_arb": cv_bd_500,
        "bdg_arb": bdg_bd,
        "grid500_contains_pub_call": bool(cv_bd_500["options"][0]["call_L_volpts"] <= 2.98840 <= cv_bd_500["options"][0]["call_U_volpts"]),
        "continuum_contains_pub_call": bool(cv_bd_cont["options"][0]["call_L_volpts"] <= 2.98840 <= cv_bd_cont["options"][0]["call_U_volpts"]),
        "seconds": round(time.perf_counter() - t_bd0, 2),
    }
    print(
        f"  Continuum Fut=[{cv_bd_cont['future_L_volpts']:.6f}, {cv_bd_cont['future_U_volpts']:.6f}], "
        f"Call=[{cv_bd_cont['options'][0]['call_L_volpts']:.6f}, {cv_bd_cont['options'][0]['call_U_volpts']:.6f}] (w={cv_bd_cont['options'][0]['call_width_volpts']:.6f})\n"
        f"  Grid-500  Fut=[{cv_bd_500['future_L_volpts']:.6f}, {cv_bd_500['future_U_volpts']:.6f}], "
        f"Call=[{cv_bd_500['options'][0]['call_L_volpts']:.6f}, {cv_bd_500['options'][0]['call_U_volpts']:.6f}] (w={cv_bd_500['options'][0]['call_width_volpts']:.6f})\n"
        f"  Published Call=2.98840 (in_grid500={case_bd2022['grid500_contains_pub_call']}, in_cont={case_bd2022['continuum_contains_pub_call']}) | BDG3 Call={bdg_bd['options'][0]['bdg_C3_volpts'][0]:.6f}",
        flush=True,
    )

    # 4. Liao, Agarwal, & Bourgey (2026, arXiv:2604.25123) VIX IV Smile Benchmark (xi0=0.24^2, H=0.10, a=1.0, T=0.25)
    case_lab2026 = run_smile_case(
        "LAB2026_VIX_IV_H010_a10_T3m",
        H=arb("0.10"),
        T=arb(1) / arb(4),
        Delta=arb(1) / arb(12),
        a=arb("1.0"),
        xi0_str="0.24",
        log_moneyness_grid=[-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
        M_cv=160,
        n_core=280,
    )

    # 5. Mixed Rough Bergomi Upward-Sloping VIX IV Smile (BDG 2023 Sec. 4 Scenario 1, T=3m)
    print("\n=== Mixed Rough Bergomi VIX Smile (a1=1.4, a2=0.7, lambda=0.3, T=3m) ===", flush=True)
    t_m0 = time.perf_counter()
    H_m, T_m, D_m = arb("0.10"), arb(1) / arb(4), arb(1) / arb(12)
    comps_m = [(arb("1.4"), arb("0.3")), (arb("0.7"), arb("0.7"))]
    mod_m = build_condvar_arb_model(H_m, T_m, D_m, comps_m, make_flat_curve("0.235"), M=160, grade_p=3, dps=35)
    # Get quick future center to center log-moneyness grid
    cv_m0 = condvar_vix_futures_and_options_arb(mod_m, [], Zmax=10.0, n_core=260, n_tail_each=45, dps=35)
    F_m_mid = 0.005 * (cv_m0["future_L_volpts"] + cv_m0["future_U_volpts"])
    km_grid = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    strikes_m = [round(F_m_mid * math.exp(k), 6) for k in km_grid]
    cv_m = condvar_vix_futures_and_options_arb(mod_m, strikes_m, Zmax=10.0, n_core=280, n_tail_each=50, dps=35)
    bdg_m = bdg_mixed_smile_float(0.10, 0.25, 1.0 / 12.0, 1.4, 0.7, 0.3, 0.235 ** 2, strikes_m)
    mixed_rows = []
    for i in range(len(strikes_m)):
        co = cv_m["options"][i]
        bo = bdg_m["options"][i]
        mixed_rows.append({"log_moneyness": km_grid[i], **co, **bo})
        print(
            f"   k={km_grid[i]:+.2f} (K={co['K_volpts']:6.3f}): Call Arb=[{co['call_L_volpts']:.5f}, {co['call_U_volpts']:.5f}] (w={co['call_width_volpts']:.5f}) | "
            f"BDG_mix3={bo['bdg_mix_C3_volpts']:.5f} | IV Arb=[{co['iv_L_pct']:.2f}%, {co['iv_U_pct']:.2f}%] vs BDG_mix3={bo['bdg_mix_iv3_pct']:.2f}% (ok={co['iv_arb_verified']})",
            flush=True,
        )
    case_mixed = {
        "name": "Mixed_rBergomi_Scen1_Smile_T3m",
        "H": 0.10, "T": 0.25, "Delta": 1.0 / 12.0, "a1": 1.4, "a2": 0.7, "lambda": 0.3, "xi0_base": 0.235,
        "condvar_future_L": cv_m["future_L_volpts"],
        "condvar_future_U": cv_m["future_U_volpts"],
        "condvar_future_width": cv_m["future_width_volpts"],
        "bdg_mix_F0_volpts": bdg_m["bdg_mix_F0_volpts"],
        "bdg_mix_F3_volpts": bdg_m["bdg_mix_F3_volpts"],
        "strikes": mixed_rows,
        "seconds": round(time.perf_counter() - t_m0, 2),
    }

    out = {
        "total_seconds": round(time.perf_counter() - t_all, 2),
        "bdg2023_fig6": case_bdg_fig6,
        "e14_stress_smile": case_stress,
        "bd2022_mlmc_call": case_bd2022,
        "lab2026_smile": case_lab2026,
        "mixed_rbergomi_smile": case_mixed,
    }
    out_path = RESULTS_DIR / "vix_options_smile_suite.json"
    out_path.write_text(json.dumps(out, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    print(f"\nDONE in {out['total_seconds']}s -> Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
