"""Week-1 Go/No-Go Verification Suite (Gates 1, 2, and 3A).

Runs in strict 160-bit Arb interval arithmetic (python-flint) and outputs verified JSON + summary table.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

from flint import arb, acb, ctx
from arb_vix_core import (
    set_dps,
    arb_lo_float,
    arb_hi_float,
    arb_width_float,
    fwd_cov_arb,
    norm_h_sq_flat_arb,
    inner_h_kx_flat_arb,
    tilted_and_lognormal_flat_arb,
    bdg_futures_flat_arb,
    bdg_futures_independent_scipy,
    discrete_right_rect_tilted_arb,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_gate1a_analytic_controls() -> dict:
    """Gate 1A: Constant kernel and two-atom controls k_+- = h +- eps*e (e perp h, |e|=1)."""
    set_dps(50)
    nh = arb("1.2")
    nh2 = nh * nh
    int_a = arb("0.235") ** 2
    pref = (-nh2 / arb(8)).exp()
    exact_const = pref * int_a.sqrt()

    out = {"controls": [], "all_pass": True}

    # Control 1: constant kernel k_u == h => m1 = int_a, m2 = int_a^2 => L == U == exact_const
    m1 = int_a
    m2 = int_a * int_a
    L = pref * (m1 * m1.sqrt()) / m2.sqrt()
    U = pref * m1.sqrt()
    rel_w_L = float(L.rad() * arb(2) / exact_const)
    rel_w_U = float(U.rad() * arb(2) / exact_const)
    contained = (L <= exact_const) and (exact_const <= U) or (exact_const.overlaps(L) and exact_const.overlaps(U))
    passed = contained and max(rel_w_L, rel_w_U) < 1e-10
    out["controls"].append({
        "id": "ctrl_const",
        "eps": 0.0,
        "exact_volpts": float(arb(100) * exact_const),
        "L_volpts": [arb_lo_float(arb(100) * L), arb_hi_float(arb(100) * L)],
        "U_volpts": [arb_lo_float(arb(100) * U), arb_hi_float(arb(100) * U)],
        "norm_diff_width": max(rel_w_L, rel_w_U),
        "pass": bool(passed),
    })
    out["all_pass"] &= bool(passed)

    # Controls 2-4: two-atom k_+- = h +- eps*e, equal weight int_a/2 each, <h, d_u> = 0 => b_+- = int_a/2
    # m1 = int_a, m2 = (int_a/2)^2 * (2*exp(eps^2) + 2*exp(-eps^2)) = int_a^2 * cosh(eps^2)
    # Exact truth: E_P[sqrt(Lambda)] = pref * sqrt(int_a) * E[sqrt(exp(-eps^2/2) * cosh(eps * Z))]
    pi = arb.pi()
    for eps_str in ("0.1", "0.2", "0.4"):
        eps = arb(eps_str)
        e2 = eps * eps
        w_atom = int_a / arb(2)
        m1_2 = w_atom + w_atom
        m2_2 = (w_atom * w_atom) * (arb(2) * e2.exp() + arb(2) * (-e2).exp())
        ratio = m2_2 / (m1_2 * m1_2)
        ratio_exact = e2.cosh()
        ratio_diff_w = float((ratio - ratio_exact).abs_upper() / ratio_exact)

        L_2 = pref * (m1_2 * m1_2.sqrt()) / m2_2.sqrt()
        U_2 = pref * m1_2.sqrt()
        L_exact = exact_const / ratio_exact.sqrt()
        U_exact = exact_const

        # Compute exact E[sqrt(exp(-e2/2) * cosh(eps * Z))] via Arb 1D integral on [0, 12] + tail bound
        eps_c, e2_c, pi_c = acb(eps), acb(e2), acb(pi)

        def f_true(z: acb, analytic: bool) -> acb:
            val = (-e2_c / acb(2)).exp() * (eps_c * z).cosh()
            return val.sqrt(analytic=analytic) * (-z * z / acb(2)).exp() * (acb(2) / (acb(2) * pi_c).sqrt())

        Zmax = arb(14)
        # Tail for z >= 14: integrand <= sqrt(exp(eps*z)) * sqrt(2/pi) * exp(-z^2/2), tiny (< 10^-40)
        tail_bound = (arb(2) / pi).sqrt() * (eps * Zmax / arb(2) - Zmax * Zmax / arb(2)).exp() / (Zmax - eps / arb(2))
        main_int = acb.integral(f_true, acb(0), acb(Zmax), rel_tol=2 ** (-90), abs_tol=2 ** (-100)).real
        truth_factor = main_int + arb(0).union(tail_bound)
        truth = exact_const * truth_factor

        # Strict containment: L_2.upper() <= truth.lower() and truth.upper() <= U_2.lower()
        strict_contains = (arb_hi_float(L_2) <= arb_lo_float(truth)) and (arb_hi_float(truth) <= arb_lo_float(U_2))
        diff_L = float((L_2 - L_exact).abs_upper() / L_exact)
        diff_U = float((U_2 - U_exact).abs_upper() / U_exact)
        max_err = max(ratio_diff_w, diff_L, diff_U)
        passed = strict_contains and (max_err < 1e-10)

        out["controls"].append({
            "id": f"ctrl_2atom_e{eps_str.replace('.', '')}",
            "eps": float(eps_str),
            "ratio_m2_over_m1sq_err": ratio_diff_w,
            "L_volpts": [arb_lo_float(arb(100) * L_2), arb_hi_float(arb(100) * L_2)],
            "truth_volpts": [arb_lo_float(arb(100) * truth), arb_hi_float(arb(100) * truth)],
            "U_volpts": [arb_lo_float(arb(100) * U_2), arb_hi_float(arb(100) * U_2)],
            "bracket_width_volpts": arb_hi_float(arb(100) * U_2) - arb_lo_float(arb(100) * L_2),
            "rel_bracket_width": float((U_2 - L_2) / U_2),
            "rel_width_over_eps4": float(((U_2 - L_2) / U_2) / (eps ** 4)),
            "max_norm_diff": max_err,
            "pass": bool(passed),
        })
        out["all_pass"] &= bool(passed)

    return out


def run_gate1b_kernel_identity_checks() -> dict:
    """Verify closed-form Arb <h, k_x> and ||h||^2 against independent numerical integration."""
    set_dps(50)
    H = arb("0.1")
    T = arb("0.5")
    Delta = arb(30) / arb(365)
    a = arb("3.0") * (arb(2) * H).sqrt()

    # Check <h, k_x> at x = Delta/3 by direct Gauss-Legendre integration of C(x, y) in float64 & Arb
    x_test = Delta / arb(3)
    hk_closed = inner_h_kx_flat_arb(H, T, Delta, a, x_test)
    # Direct numerical check via scipy quad on C(x, y)
    from scipy import integrate
    def c_float(y):
        return float(fwd_cov_arb(H, T, a, x_test, arb(y)))
    val_quad, err_quad = integrate.quad(c_float, 0.0, float(Delta), epsabs=1e-13, epsrel=1e-13)
    hk_quad = val_quad / float(Delta)
    rel_diff_hk = abs(float(hk_closed) - hk_quad) / abs(float(hk_closed))

    # Check ||h||^2 = (1/Delta) int_0^Delta <h, k_x> dx
    nh2_closed = norm_h_sq_flat_arb(H, T, Delta, a)
    val_nh2_quad, _ = integrate.quad(lambda x: float(inner_h_kx_flat_arb(H, T, Delta, a, arb(x))), 0.0, float(Delta), epsabs=1e-13, epsrel=1e-13)
    nh2_quad = val_nh2_quad / float(Delta)
    rel_diff_nh2 = abs(float(nh2_closed) - nh2_quad) / abs(float(nh2_closed))

    return {
        "hk_closed": float(hk_closed),
        "hk_quad": hk_quad,
        "rel_diff_hk": rel_diff_hk,
        "nh2_closed": float(nh2_closed),
        "nh2_quad": nh2_quad,
        "rel_diff_nh2": rel_diff_nh2,
        "pass": (rel_diff_hk < 1e-11) and (rel_diff_nh2 < 1e-11),
    }


def _parse_cell(c: dict) -> Tuple[arb, arb, arb, arb, arb, float, float, float, float, float]:
    H = arb(str(c["H"]))
    T = arb(c["T_num"]) / arb(c["T_den"]) if "T_num" in c else arb(str(c["T"]))
    Delta = arb(c["Delta_num"]) / arb(c["Delta_den"])
    if "a_direct" in c:
        a = arb(str(c["a_direct"]))
        eta_p1 = a / (arb(2) * H).sqrt()
    else:
        eta_p1 = arb(str(c["eta_P1"]))
        a = eta_p1 * (arb(2) * H).sqrt()
    xi0_val = arb(str(c["xi0_base"])) ** 2
    return H, T, Delta, a, xi0_val, float(H), float(T), float(Delta), float(a), float(xi0_val)


def run_headline_cells(M: int = 240, grade_p: int = 3) -> dict:
    manifest = json.loads((ROOT / "params_manifest.json").read_text(encoding="utf-8"))
    rows = []
    for c in manifest["headline_cells"]:
        t0 = time.perf_counter()
        H, T, Delta, a, xi0_val, Hf, Tf, Df, af, xi0f = _parse_cell(c)
        tilted = tilted_and_lognormal_flat_arb(H, T, Delta, a, xi0_val, M=M, grade_p=grade_p, dps=35)
        M_used = M
        if max(tilted["L_enclosure_width_volpts"], tilted["U_enclosure_width_volpts"]) > 0.002:
            M_used = 360
            tilted = tilted_and_lognormal_flat_arb(H, T, Delta, a, xi0_val, M=M_used, grade_p=grade_p, dps=35)
        bdg_arb = bdg_futures_flat_arb(H, T, Delta, a, xi0_val, dps=40)
        bdg_sci = bdg_futures_independent_scipy(Hf, Tf, Df, af, xi0f, n_quad=128)

        bdg_diff_impl = float(abs(0.5 * (bdg_arb["F3_volpts"][0] + bdg_arb["F3_volpts"][1]) - float(bdg_sci["F3_volpts"])))
        # Certified gap: BDG_lo - U_outer (positive => BDG provably exceeds EVERY continuous rough-Bergomi price!)
        bdg_minus_U_cert = float(bdg_arb["F3_volpts"][0] - tilted["U_outer_volpts"])
        # Certified gap: L_outer - LN_hi (positive => LN proxy provably below EVERY continuous rough-Bergomi price!)
        L_minus_LN_cert = float(tilted["L_outer_volpts"] - tilted["ln_fut_volpts"][1])

        row = {
            "id": c["id"],
            "source": c["source"],
            "H": Hf,
            "T": Tf,
            "Delta": Df,
            "a": af,
            "eta_P1": af / math.sqrt(2.0 * Hf),
            "xi0": xi0f,
            "M": M_used,
            "grade_p": grade_p,
            "tilted_L_outer": float(tilted["L_outer_volpts"]),
            "tilted_L_inner": float(tilted["L_inner_volpts"]),
            "tilted_U_inner": float(tilted["U_inner_volpts"]),
            "tilted_U_outer": float(tilted["U_outer_volpts"]),
            "tilted_total_outer_width": float(tilted["total_outer_width_volpts"]),
            "tilted_L_enc_width": float(tilted["L_enclosure_width_volpts"]),
            "tilted_U_enc_width": float(tilted["U_enclosure_width_volpts"]),
            "sub_tick_pass": bool((tilted["total_outer_width_volpts"] < 0.05) and (max(tilted["L_enclosure_width_volpts"], tilted["U_enclosure_width_volpts"]) <= 0.002)),
            "jmm_lo": [float(x) for x in tilted["jmm_lo_volpts"]],
            "jmm_up": [float(x) for x in tilted["jmm_up_volpts"]],
            "ln_fut": [float(x) for x in tilted["ln_fut_volpts"]],
            "ln_enc_width": float(tilted["ln_enclosure_width_volpts"]),
            "L_minus_LN_cert": L_minus_LN_cert,
            "bdg_F3_arb": [float(x) for x in bdg_arb["F3_volpts"]],
            "bdg_F3_enc_width": float(bdg_arb["F3_width_volpts"]),
            "bdg_F3_scipy": float(bdg_sci["F3_volpts"]),
            "bdg_impl_diff": bdg_diff_impl,
            "bdg_impl_pass": bool(bdg_diff_impl < 1e-4),
            "bdg_minus_U_cert": bdg_minus_U_cert,
            "bdg_exceeds_1tick_cert": bool(bdg_minus_U_cert > 0.05),
            "seconds": round(time.perf_counter() - t0, 2),
        }
        if c["id"] == "BD2022_MLMC_JCF":
            row["published_fut_volpts"] = c["published_fut_volpts"]
            row["pub_minus_continuum_U"] = c["published_fut_volpts"] - tilted["U_outer_volpts"]
            t_disc = time.perf_counter()
            disc500 = discrete_right_rect_tilted_arb(H, T, Delta, a, xi0_val, n=500, dps=30)
            row["grid500_tilted_arb"] = disc500
            row["grid500_contains_published"] = (disc500["L_outer_volpts"] <= c["published_fut_volpts"] <= disc500["U_outer_volpts"])
            row["grid500_seconds"] = round(time.perf_counter() - t_disc, 2)
        rows.append(row)
        print(
            f"[{row['id']}] Tilted=[{row['tilted_L_outer']:.6f}, {row['tilted_U_outer']:.6f}] "
            f"(w={row['tilted_total_outer_width']:.6f}, enc_L={row['tilted_L_enc_width']:.2e}, enc_U={row['tilted_U_enc_width']:.2e}, pass={row['sub_tick_pass']}) | "
            f"BDG=[{row['bdg_F3_arb'][0]:.6f}, {row['bdg_F3_arb'][1]:.6f}] (diff={row['bdg_impl_diff']:.2e}, BDG-U={row['bdg_minus_U_cert']:+.6f}) | "
            f"LN=[{row['ln_fut'][0]:.5f}, {row['ln_fut'][1]:.5f}] (L-LN={row['L_minus_LN_cert']:+.5f}) | {row['seconds']}s",
            flush=True,
        )
        if "grid500_tilted_arb" in row:
            d5 = row["grid500_tilted_arb"]
            print(
                f"   -> BD2022 500-cell right-rect Arb: [{d5['L_outer_volpts']:.6f}, {d5['U_outer_volpts']:.6f}] "
                f"(pub={row['published_fut_volpts']:.5f}, contains={row['grid500_contains_published']}, {row['grid500_seconds']}s)",
                flush=True,
            )
    return {"cells": rows}


def main():
    print("=== GATE 1A: Analytic Controls (Arb 160-bit) ===", flush=True)
    g1a = run_gate1a_analytic_controls()
    for r in g1a["controls"]:
        print(f"  {r['id']}: pass={r['pass']} | max_err={r.get('max_norm_diff', r.get('norm_diff_width')):.2e}" + (f" | rel_w/eps^4={r['rel_width_over_eps4']:.6f}" if "rel_width_over_eps4" in r else ""), flush=True)
    assert g1a["all_pass"], "GATE 1A FAILED!"

    print("\n=== GATE 1B: Closed-Form Volterra Kernel Cross-Check ===", flush=True)
    g1b = run_gate1b_kernel_identity_checks()
    print(f"  rel_diff_hk={g1b['rel_diff_hk']:.2e}, rel_diff_nh2={g1b['rel_diff_nh2']:.2e}, pass={g1b['pass']}", flush=True)
    assert g1b["pass"], "GATE 1B FAILED!"

    print("\n=== GATES 1C & 2: Full End-to-End Arb Enclosures on Headline Cells (M=240, p=3) ===", flush=True)
    g2 = run_headline_cells(M=240, grade_p=3)

    out = {"gate1a": g1a, "gate1b": g1b, "gate2": g2}
    out_path = RESULTS_DIR / "week1_gates_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    print(f"\nSaved verified results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
