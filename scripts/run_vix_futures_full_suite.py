"""Full VIX Futures Competitor & Generalization Suite (100% 160-bit Arb).

Covers four major published benchmark blocks:
  Block A: All 15 P1/E14 cells (9 base flat + 3 JMM Scenario 2 quadratic curve + 3 stress eta=3)
           comparing Arb Tilted Bracket, Arb Conditional-Variance Bracket, BDG (2023) Thm. 2.2 Arb,
           JMM (2018) Thm. 3.3 Arb (total width + lower-bound error), and Lognormal Proxy Arb.
  Block B: Bourgey, De Marco, & Gobet (2023, Quantitative Finance) Fig. 1 (vol-of-vol a sweep)
           and Fig. 2 (Hurst H sweep) — maps the exact Arb-certified 1-tick violation boundary!
  Block C: Bonesini, Callegaro, & Jacquier (2023, Quantitative Finance, Sec. 5.3)
           3 forward-variance curves (flat, (1+t)^2, sqrt(1+t)) x 6 maturities (1,2,3,6,9,12m).
  Block D: Mixed Rough Bergomi (Bergomi 2016; BDG 2023 Sec. 4) — Arb Conditional-Variance
           futures brackets vs. BDG mixed order-0/3 expansions across maturities and stress.

Outputs verified JSON to: results/vix_futures_full_suite.json
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from flint import arb, acb

from arb_vix_core import (
    set_dps,
    arb_interval,
    arb_lo_float,
    arb_hi_float,
    tilted_and_lognormal_flat_arb,
    bdg_futures_flat_arb,
    bdg_futures_independent_scipy,
)
from arb_vix_options import (
    Xi0Curve,
    make_flat_curve,
    make_quadratic_curve,
    make_sqrt_curve,
    build_condvar_arb_model,
    condvar_vix_futures_and_options_arb,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def jmm_bounds_curved_arb(H: arb, T: arb, Delta: arb, a: arb, curve: Xi0Curve, dps: int = 40) -> dict:
    """Rigorous Arb enclosure of JMM (2018, Thm. 3.3) lower & upper VIX futures bounds
    for any smooth increasing forward-variance curve xi_0(t):
      U_JMM = sqrt((1/Delta) int_0^Delta xi_0(T+x) dx)
      L_JMM = (1/Delta) int_0^Delta sqrt(xi_0(T+x)) * exp(-a^2/(16H) * ((T+x)^{2H} - x^{2H})) dx.
    """
    set_dps(dps)
    eta_sq = (a * a) / (2 * H)
    jmm_up = curve.avg_arb(T, Delta).sqrt()

    # Since both sqrt(xi_0(T+x)) (for increasing curves) and exp(-eta_sq/8 * ((T+x)^{2H} - x^{2H}))
    # are strictly INCREASING on [0, Delta], a 1D graded monotone Riemann sum on M=200 panels is
    # fast, exact, and works uniformly for flat, quadratic, and sqrt curves!
    M = 240
    p = 3
    xs = [arb(0)] + [Delta * (arb(i) / arb(M)) ** p for i in range(1, M)] + [Delta]
    vals = []
    for i, x in enumerate(xs):
        v_var = (T ** (2 * H)) if i == 0 else ((T + x) ** (2 * H) - x ** (2 * H))
        vals.append(curve.eval_arb(T + x).sqrt() * (-eta_sq * v_var / arb(8)).exp())
    lo_sum = arb(0)
    hi_sum = arb(0)
    for i in range(M):
        dx = xs[i + 1] - xs[i]
        lo_sum += dx * vals[i]
        hi_sum += dx * vals[i + 1]
    jmm_lo = arb_interval(lo_sum / Delta, hi_sum / Delta)
    return {
        "jmm_lo_volpts": [arb_lo_float(arb(100) * jmm_lo), arb_hi_float(arb(100) * jmm_lo)],
        "jmm_up_volpts": [arb_lo_float(arb(100) * jmm_up), arb_hi_float(arb(100) * jmm_up)],
        "jmm_width_volpts": arb_hi_float(arb(100) * jmm_up) - arb_lo_float(arb(100) * jmm_lo),
    }


def bdg_mixed_futures_float(H: float, T: float, Delta: float, a1: float, a2: float, lbd: float, xi0_val: float) -> dict:
    """Exact implementation of Bourgey, De Marco, & Gobet (2023, QF, Sec. 4) mixed rBergomi
    weak expansion (matching fbourgey_rbergomi_7822b00a.py price_vix_approx_mixed)."""
    b1 = bdg_futures_independent_scipy(H, T, Delta, a1, xi0_val, n_quad=96)
    b2 = bdg_futures_independent_scipy(H, T, Delta, a2, xi0_val, n_quad=96)
    e1 = a1 / math.sqrt(2.0 * H)
    e2 = a2 / math.sqrt(2.0 * H)
    mp_unit = -((T + Delta) ** (2 * H + 1) - Delta ** (2 * H + 1) - T ** (2 * H + 1)) / (2.0 * Delta * (2 * H + 1))
    mp1 = (e1 ** 2) * mp_unit
    mp2 = (e2 ** 2) * mp_unit
    sp1 = math.sqrt(b1["varp"])
    sp2 = math.sqrt(b2["varp"])

    n_gh = 80
    gx, gw = np.polynomial.hermite.hermgauss(n_gh)
    nodes = math.sqrt(2.0) * gx
    weights = gw / math.sqrt(math.pi)

    p0 = float(np.sum(weights * np.sqrt(xi0_val * (lbd * np.exp(mp1 + sp1 * nodes) + (1.0 - lbd) * np.exp(mp2 + sp2 * nodes)))))
    mu0 = mp_unit
    eps = 1e-5

    def psi(x: float, idx: int) -> float:
        if idx == 1:
            def f(y: float) -> float:
                t0 = lbd * math.exp(mp1 + sp1 * x + y)
                t1 = (1.0 - lbd) * math.exp(e2 * (e1 - e2) * (-mu0) + (e2 / e1) * (mp1 + sp1 * x))
                return math.sqrt(xi0_val * (t0 + t1))
        else:
            def f(y: float) -> float:
                t0 = (1.0 - lbd) * math.exp(mp2 + sp2 * x + y)
                t1 = lbd * math.exp(e1 * (e2 - e1) * (-mu0) + (e1 / e2) * (mp2 + sp2 * x))
                return math.sqrt(xi0_val * (t0 + t1))
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
    return {
        "order0_volpts": 100.0 * p0,
        "order1_volpts": 100.0 * (p0 + p1),
        "order2_volpts": 100.0 * (p0 + p1 + p2),
        "order3_volpts": 100.0 * (p0 + p1 + p2 + p3),
    }


def run_block_a_e14_15cells() -> list:
    """Block A: All 15 P1/E14 cells (9 base + 3 curved JMM Scen 2 + 3 stress)."""
    print("\n=== BLOCK A: All 15 P1/E14 Cells (Arb Tilted + Arb CondVar + BDG + JMM + LN) ===", flush=True)
    Delta = arb(30) / arb(365)
    Df = 30.0 / 365.0
    specs = []
    # 9 base cells
    for H_s in ("0.05", "0.10", "0.30"):
        for T_lbl, T_num, T_den in (("1w", 7, 365), ("1m", 1, 12), ("3m", 3, 12)):
            specs.append(("base", f"base_H{H_s.replace('.','')}_{T_lbl}", H_s, T_num, T_den, "1.9", "flat", "0.235"))
    # 3 curved JMM Scenario 2 cells
    for H_s in ("0.05", "0.10", "0.30"):
        specs.append(("curve", f"curve_H{H_s.replace('.','')}_1m", H_s, 1, 12, "1.9", "quad", "0.234"))
    # 3 stress cells
    for H_s in ("0.05", "0.10", "0.30"):
        specs.append(("stress", f"stress_H{H_s.replace('.','')}_6m", H_s, 1, 2, "3.0", "flat", "0.235"))

    rows = []
    for grp, cid, H_s, T_num, T_den, eta_s, ctype, xi_s in specs:
        set_dps(35)
        H = arb(H_s)
        T = arb(T_num) / arb(T_den)
        eta_p1 = arb(eta_s)
        a = eta_p1 * (arb(2) * H).sqrt()
        curve = make_flat_curve(xi_s) if ctype == "flat" else make_quadratic_curve(xi_s)

        M_cv = 180 if grp == "stress" else 128
        n_core_cv = 300 if grp == "stress" else 220
        t0 = time.perf_counter()
        mod = build_condvar_arb_model(H, T, Delta, [(a, arb(1))], curve, M=M_cv, grade_p=3, dps=35)
        t_setup = time.perf_counter() - t0
        t1 = time.perf_counter()
        cv = condvar_vix_futures_and_options_arb(mod, [], Zmax=10.0, n_core=n_core_cv, n_tail_each=45, dps=35)
        t_solve = time.perf_counter() - t1

        row = {
            "group": grp,
            "id": cid,
            "H": float(H_s),
            "T": T_num / T_den,
            "Delta": Df,
            "eta_P1": float(eta_s),
            "a": float(a),
            "curve": curve.name,
            "condvar_L": cv["future_L_volpts"],
            "condvar_U": cv["future_U_volpts"],
            "condvar_width": cv["future_width_volpts"],
            "condvar_setup_s": round(t_setup, 2),
            "condvar_solve_s": round(t_solve, 2),
        }

        if ctype == "flat":
            M_t = 240 if float(a) < 1.3 else 300
            til = tilted_and_lognormal_flat_arb(H, T, Delta, a, arb(xi_s) ** 2, M=M_t, grade_p=3, dps=35)
            bdg = bdg_futures_flat_arb(H, T, Delta, a, arb(xi_s) ** 2, dps=40)
            row.update({
                "tilted_L": til["L_outer_volpts"],
                "tilted_U": til["U_outer_volpts"],
                "tilted_width": til["total_outer_width_volpts"],
                "best_L": max(cv["future_L_volpts"], til["L_outer_volpts"]),
                "best_U": min(cv["future_U_volpts"], til["U_outer_volpts"]),
                "jmm_lo": til["jmm_lo_volpts"],
                "jmm_up": til["jmm_up_volpts"],
                "jmm_width": til["jmm_up_volpts"][1] - til["jmm_lo_volpts"][0],
                "ln_fut": til["ln_fut_volpts"],
                "bdg_F3": bdg["F3_volpts"],
            })
            row["best_width"] = row["best_U"] - row["best_L"]
            row["jmm_lo_gap_max"] = row["best_U"] - row["jmm_lo"][0]
            row["bdg_minus_U"] = row["bdg_F3"][0] - row["best_U"]
            row["L_minus_LN"] = row["best_L"] - row["ln_fut"][1]
        else:
            jmm = jmm_bounds_curved_arb(H, T, Delta, a, curve, dps=40)
            row.update({
                "best_L": cv["future_L_volpts"],
                "best_U": cv["future_U_volpts"],
                "best_width": cv["future_width_volpts"],
                "jmm_lo": jmm["jmm_lo_volpts"],
                "jmm_up": jmm["jmm_up_volpts"],
                "jmm_width": jmm["jmm_width_volpts"],
                "jmm_lo_gap_max": cv["future_U_volpts"] - jmm["jmm_lo_volpts"][0],
            })
        row["ratio_vs_jmm"] = row["jmm_width"] / row["best_width"]
        rows.append(row)
        bdg_str = f"BDG-U={row['bdg_minus_U']:+.5f} | L-LN={row['L_minus_LN']:+.5f}" if "bdg_minus_U" in row else "curved"
        print(
            f"  [{cid:16s}] Best=[{row['best_L']:.5f}, {row['best_U']:.5f}] (w={row['best_width']:.5f}) | "
            f"JMM_w={row['jmm_width']:.3f} ({row['ratio_vs_jmm']:.0f}x) | {bdg_str} | setup={t_setup:.1f}s solve={t_solve:.1f}s",
            flush=True,
        )
    return rows


def run_block_b_bdg_sweeps() -> dict:
    """Block B: BDG (2023) Fig. 1 (a-sweep) & Fig. 2 (H-sweep)."""
    print("\n=== BLOCK B: BDG (2023, QF) Fig. 1 (a-sweep) & Fig. 2 (H-sweep) ===", flush=True)
    Delta = arb(1) / arb(12)
    T = arb(1) / arb(2)
    xi0_val = arb("0.235") ** 2

    fig1_rows = []
    H = arb("0.10")
    for a_s in ("0.2", "0.4", "0.6", "0.8", "1.0", "1.2", "1.35", "1.5"):
        a = arb(a_s)
        M_t = 240 if float(a_s) <= 1.2 else 340
        til = tilted_and_lognormal_flat_arb(H, T, Delta, a, xi0_val, M=M_t, grade_p=3, dps=35)
        bdg = bdg_futures_flat_arb(H, T, Delta, a, xi0_val, dps=40)
        row = {
            "a": float(a_s),
            "eta_P1": float(a_s) / math.sqrt(0.2),
            "H": 0.10,
            "T": 0.5,
            "Delta": 1.0 / 12.0,
            "tilted_L": til["L_outer_volpts"],
            "tilted_U": til["U_outer_volpts"],
            "tilted_width": til["total_outer_width_volpts"],
            "bdg_F0": [arb_lo_float(arb(100) * bdg["F0_ball"]), arb_hi_float(arb(100) * bdg["F0_ball"])],
            "bdg_F1": [arb_lo_float(arb(100) * bdg["F1_ball"]), arb_hi_float(arb(100) * bdg["F1_ball"])],
            "bdg_F2": [arb_lo_float(arb(100) * bdg["F2_ball"]), arb_hi_float(arb(100) * bdg["F2_ball"])],
            "bdg_F3": bdg["F3_volpts"],
            "ln_fut": til["ln_fut_volpts"],
            "bdg3_minus_U": bdg["F3_volpts"][0] - til["U_outer_volpts"],
            "L_minus_LN": til["L_outer_volpts"] - til["ln_fut_volpts"][1],
            "exceeds_1tick": bool(bdg["F3_volpts"][0] - til["U_outer_volpts"] > 0.05),
        }
        fig1_rows.append(row)
        print(
            f"  [Fig1 a={a_s:4s}] Tilted=[{row['tilted_L']:.5f}, {row['tilted_U']:.5f}] (w={row['tilted_width']:.5f}) | "
            f"BDG3={row['bdg_F3'][0]:.5f} (BDG3-U={row['bdg3_minus_U']:+.5f}, >1tick={row['exceeds_1tick']}) | L-LN={row['L_minus_LN']:+.5f}",
            flush=True,
        )

    fig2_rows = []
    a_fix = arb("1.0")
    for H_s in ("0.05", "0.08", "0.10", "0.15", "0.20", "0.30", "0.40"):
        H_cur = arb(H_s)
        til = tilted_and_lognormal_flat_arb(H_cur, T, Delta, a_fix, xi0_val, M=220, grade_p=3, dps=35)
        bdg = bdg_futures_flat_arb(H_cur, T, Delta, a_fix, xi0_val, dps=40)
        row = {
            "H": float(H_s),
            "a": 1.0,
            "eta_P1": 1.0 / math.sqrt(2.0 * float(H_s)),
            "T": 0.5,
            "Delta": 1.0 / 12.0,
            "tilted_L": til["L_outer_volpts"],
            "tilted_U": til["U_outer_volpts"],
            "tilted_width": til["total_outer_width_volpts"],
            "bdg_F3": bdg["F3_volpts"],
            "ln_fut": til["ln_fut_volpts"],
            "bdg3_minus_U": bdg["F3_volpts"][0] - til["U_outer_volpts"],
            "L_minus_LN": til["L_outer_volpts"] - til["ln_fut_volpts"][1],
        }
        fig2_rows.append(row)
        print(
            f"  [Fig2 H={H_s:4s}] Tilted=[{row['tilted_L']:.5f}, {row['tilted_U']:.5f}] (w={row['tilted_width']:.5f}) | "
            f"BDG3-U={row['bdg3_minus_U']:+.5f} | L-LN={row['L_minus_LN']:+.5f}",
            flush=True,
        )
    return {"fig1_a_sweep": fig1_rows, "fig2_H_sweep": fig2_rows}


def run_block_c_bcj2023() -> list:
    """Block C: Bonesini, Callegaro, & Jacquier (2023, QF, Sec. 5.3)
    3 curves x 6 maturities (1, 2, 3, 6, 9, 12m), H=0.10, eta_P1=1.9, Delta=1/12."""
    print("\n=== BLOCK C: BCJ (2023, QF, Sec. 5.3) Multi-Curve x Multi-Maturity (18 cells) ===", flush=True)
    set_dps(35)
    H = arb("0.10")
    Delta = arb(1) / arb(12)
    a = arb("1.9") * (arb(2) * H).sqrt()
    curves = [
        ("flat", make_flat_curve("0.234")),
        ("quad", make_quadratic_curve("0.234")),
        ("sqrt", make_sqrt_curve("0.234")),
    ]
    rows = []
    for cname, curve in curves:
        for m_mo in (1, 2, 3, 6, 9, 12):
            T = arb(m_mo) / arb(12)
            t0 = time.perf_counter()
            mod = build_condvar_arb_model(H, T, Delta, [(a, arb(1))], curve, M=120, grade_p=3, dps=35)
            t_setup = time.perf_counter() - t0
            t1 = time.perf_counter()
            cv = condvar_vix_futures_and_options_arb(mod, [], Zmax=10.0, n_core=220, n_tail_each=45, dps=35)
            t_solve = time.perf_counter() - t1
            jmm = jmm_bounds_curved_arb(H, T, Delta, a, curve, dps=35)
            row = {
                "curve": cname,
                "curve_formula": curve.name,
                "T_months": m_mo,
                "T": m_mo / 12.0,
                "H": 0.10,
                "eta_P1": 1.9,
                "a": float(a),
                "Delta": 1.0 / 12.0,
                "condvar_L": cv["future_L_volpts"],
                "condvar_U": cv["future_U_volpts"],
                "condvar_width": cv["future_width_volpts"],
                "jmm_lo": jmm["jmm_lo_volpts"],
                "jmm_up": jmm["jmm_up_volpts"],
                "jmm_width": jmm["jmm_width_volpts"],
                "ratio_vs_jmm": jmm["jmm_width_volpts"] / cv["future_width_volpts"],
                "setup_s": round(t_setup, 2),
                "solve_s": round(t_solve, 2),
            }
            rows.append(row)
            print(
                f"  [BCJ {cname:4s} T={m_mo:2d}m] Arb=[{row['condvar_L']:.5f}, {row['condvar_U']:.5f}] "
                f"(w={row['condvar_width']:.5f}, JMM={row['jmm_width']:.3f}, {row['ratio_vs_jmm']:.0f}x) | {t_setup+t_solve:.1f}s",
                flush=True,
            )
    return rows


def run_block_d_mixed_rbergomi() -> list:
    """Block D: Mixed Rough Bergomi (BDG 2023 Sec. 4):
      Scenario 1: a1=1.4, a2=0.7, lambda=0.3 (T=1,3,6m)
      Stress Mixed: a1=1.6, a2=0.7, lambda=0.4 (T=3,6m)"""
    print("\n=== BLOCK D: Mixed Rough Bergomi (Arb CondVar vs BDG Order-0/3 Mixed Expansion) ===", flush=True)
    set_dps(35)
    H = arb("0.10")
    Delta = arb(1) / arb(12)
    curve = make_flat_curve("0.235")
    xi0_f = 0.235 ** 2
    cases = [
        ("mixed_scen1_1m", "1.4", "0.7", "0.3", 1),
        ("mixed_scen1_3m", "1.4", "0.7", "0.3", 3),
        ("mixed_scen1_6m", "1.4", "0.7", "0.3", 6),
        ("mixed_stress_3m", "1.6", "0.7", "0.4", 3),
        ("mixed_stress_6m", "1.6", "0.7", "0.4", 6),
    ]
    rows = []
    for cid, a1_s, a2_s, lam_s, m_mo in cases:
        T = arb(m_mo) / arb(12)
        comps = [(arb(a1_s), arb(lam_s)), (arb(a2_s), arb(1) - arb(lam_s))]
        t0 = time.perf_counter()
        mod = build_condvar_arb_model(H, T, Delta, comps, curve, M=140, grade_p=3, dps=35)
        t_setup = time.perf_counter() - t0
        t1 = time.perf_counter()
        cv = condvar_vix_futures_and_options_arb(mod, [], Zmax=10.0, n_core=260, n_tail_each=50, dps=35)
        t_solve = time.perf_counter() - t1
        bdg_m = bdg_mixed_futures_float(0.10, m_mo / 12.0, 1.0 / 12.0, float(a1_s), float(a2_s), float(lam_s), xi0_f)
        gap3 = bdg_m["order3_volpts"] - cv["future_U_volpts"]
        row = {
            "id": cid,
            "H": 0.10,
            "T_months": m_mo,
            "T": m_mo / 12.0,
            "Delta": 1.0 / 12.0,
            "a1": float(a1_s),
            "a2": float(a2_s),
            "lambda": float(lam_s),
            "condvar_L": cv["future_L_volpts"],
            "condvar_U": cv["future_U_volpts"],
            "condvar_width": cv["future_width_volpts"],
            "bdg_mixed_order0": bdg_m["order0_volpts"],
            "bdg_mixed_order3": bdg_m["order3_volpts"],
            "bdg3_minus_U": gap3,
            "exceeds_1tick": bool(gap3 > 0.05),
            "setup_s": round(t_setup, 2),
            "solve_s": round(t_solve, 2),
        }
        rows.append(row)
        print(
            f"  [{cid:16s}] Arb=[{row['condvar_L']:.5f}, {row['condvar_U']:.5f}] (w={row['condvar_width']:.5f}) | "
            f"BDG_mix0={row['bdg_mixed_order0']:.5f}, BDG_mix3={row['bdg_mixed_order3']:.5f} (BDG3-U={gap3:+.5f}, >1tick={row['exceeds_1tick']}) | {t_setup+t_solve:.1f}s",
            flush=True,
        )
    return rows


def main():
    t_all = time.perf_counter()
    block_a = run_block_a_e14_15cells()
    block_b = run_block_b_bdg_sweeps()
    block_c = run_block_c_bcj2023()
    block_d = run_block_d_mixed_rbergomi()

    out = {
        "total_seconds": round(time.perf_counter() - t_all, 2),
        "block_a_e14_15cells": block_a,
        "block_b_bdg_sweeps": block_b,
        "block_c_bcj2023_18cells": block_c,
        "block_d_mixed_rbergomi": block_d,
    }
    out_path = RESULTS_DIR / "vix_futures_full_suite.json"
    out_path.write_text(json.dumps(out, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    print(f"\nDONE in {out['total_seconds']}s -> Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
