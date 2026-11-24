"""Build ALL 10 redesigned publication tables (8 main-text + 2 supplement) for RoughCert v3.

Every numeric interval is outward-rounded directly from verified JSON outputs via
exact Decimal FLOOR / CEILING rounding. All tables avoid \\resizebox scaling hacks,
use proper LaTeX math minus signs ($-$), state units in column headers, respect
<=45-word captions and <=55-word notes, and carry consistent RoughCert branding.
"""
from __future__ import annotations

import json
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
import numpy as np
from v3_comparisons import scoreboard, difference

from arb_vix_core import arb
from arb_vix_options import black76_iv_bracket_arb

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
LEG = RES / "legacy_inputs"
TAB = ROOT / "manuscript" / "tables"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_out(x: float, n: int, up: bool) -> str:
    """Outward decimal rounding (ROUND_FLOOR for lower, ROUND_CEILING for upper)."""
    d = Decimal.from_float(float(x)).quantize(
        Decimal(1).scaleb(-n),
        rounding=ROUND_CEILING if up else ROUND_FLOOR,
    )
    s = format(d, "f")
    if s.startswith("-"):
        return f"$-${s[1:]}"
    return s


def fmt_band(lo: float, hi: float, n: int = 5, pct: bool = False) -> str:
    suf = r"\%" if pct else ""
    return f"[{fmt_out(lo, n, False)}{suf},\\,{fmt_out(hi, n, True)}{suf}]"


def fmt_signed(x: float, n: int, up: bool | None = None, bold_thresh: float | None = None) -> str:
    """Format signed value with proper LaTeX minus sign and optional >threshold bold."""
    if up is None:
        s_raw = f"{abs(x):.{n}f}"
    else:
        d = Decimal.from_float(float(x)).quantize(
            Decimal(1).scaleb(-n),
            rounding=ROUND_CEILING if up else ROUND_FLOOR,
        )
        s_raw = format(abs(d), "f")
    sign = "$+$" if x >= 0 else "$-$"
    txt = f"{sign}{s_raw}"
    if bold_thresh is not None and abs(x) > bold_thresh:
        return f"\\textbf{{{txt}}}"
    return txt


def iv_sheet(F: list[float], C: list[float], K: float) -> list[float]:
    iv = black76_iv_bracket_arb(
        arb(F[0]) / 100, arb(F[1]) / 100, arb(str(K)), arb(1) / 4, arb(C[0]) / 100, arb(C[1]) / 100
    )
    return [100.0 * iv[0], 100.0 * iv[1]]


def write_tex(name: str, content: str) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / name).write_text(content.strip() + "\n", encoding="utf-8")
    print(f"WROTE {name}")


# =============================================================================
# TABLE 1: Headline 8-Benchmark RoughCert-Tilt Audit + BD (2022) 500-Cell Grid
# =============================================================================
def build_tab1_headline():
    cells = load_json(RES / "week1_gates_results.json")["gate2"]["cells"]
    by_id = {c["id"]: c for c in cells}

    order = [
        ("E14_stress_H010", "Stress (E14)", "0.10", "0.50", "30/365"),
        ("BDG2023_Fig1_a15_T6m", "BDG (2023) Fig.~1", "0.10", "0.50", "1/12"),
        ("BD2022_MLMC_JCF", r"BD (2022) \S3.3", "0.10", "0.50", "1/12"),
        ("BDG2023_Fig6_smile_T3m", "BDG (2023) Fig.~6", "0.10", "0.25", "1/12"),
        ("BCJ2023_Sec53_flat_T6m", r"BCJ (2023) \S5.3", "0.10", "0.50", "1/12"),
        ("E14_base_H010_1m", r"Base ($H=0.10$)", "0.10", "1/12", "30/365"),
        ("E14_base_H005_1m", r"Base ($H=0.05$)", "0.05", "1/12", "30/365"),
        ("E14_stress_H030", r"Stress ($H=0.30$)", "0.30", "0.50", "30/365"),
    ]

    lines = [
        r"\begin{table}[!htbp]\centering\scriptsize\setlength{\tabcolsep}{2.2pt}",
        r"\caption{Headline 169-bit Arb-certified \RoughCertTilt{} VIX futures brackets (volatility points) across eight literature and stress-test parameter cells, audited against BDG~(2023) and lognormal proxies.}\label{tab:vix_headline}",
        r"\begin{tabular}{@{}lrrrrccrrr@{}}\toprule",
        r" & & & & & \multicolumn{2}{c}{\RoughCertTilt{} (pts)} & \multicolumn{3}{c}{Proxies (pts)} \\",
        r"\cmidrule(lr){6-7}\cmidrule(lr){8-10}",
        r"Benchmark & $H$ & $T$ & $\Delta$ & $a$ & $[L,\,U]$ & Width & BDG$_3$ & $\mathrm{BDG}_3\!-\!U$ & $\mathrm{LN}\!-\!L$ \\\midrule",
    ]
    for cid, label, h_s, t_s, d_s in order:
        c = by_id[cid]
        L, U = c["tilted_L_outer"], c["tilted_U_outer"]
        w = c["tilted_total_outer_width"]
        bdg_mid = 0.5 * (c["bdg_F3_arb"][0] + c["bdg_F3_arb"][1])
        bdg_gap = c["bdg_minus_U_cert"]
        ln_gap = -c["L_minus_LN_cert"]
        lines.append(
            f"{label} & {h_s} & {t_s} & {d_s} & {c['a']:.3f} & "
            f"{fmt_band(L, U, 5)} & {fmt_out(w, 5, True)} & {bdg_mid:.5f} & "
            f"{fmt_signed(bdg_gap, 5, up=False, bold_thresh=0.05)} & "
            f"{fmt_signed(ln_gap, 4, up=True, bold_thresh=0.05)} \\\\"
        )

    bd = by_id["BD2022_MLMC_JCF"]
    g = bd["grid500_tilted_arb"]
    gL, gU = g["L_outer_volpts"], g["U_outer_volpts"]
    lines.extend([
        r"\midrule",
        f"BD (2022) 500-cell & 0.10 & 0.50 & 1/12 & 0.500 & "
        f"{fmt_band(gL, gU, 5)} & {fmt_out(gU - gL, 5, True)} & "
        r"\multicolumn{3}{c}{MLMC $= 22.18135$ (in grid; outside cont.)} \\",
        r"\bottomrule\end{tabular}",
        r"\smallskip\par\raggedright\scriptsize\emph{Notes:} $a=\eta_{\mathrm{P1}}\sqrt{2H}$; flat $\xi_0=0.235^2$ ($0.234^2$ for BCJ). All intervals are outward-rounded 169-bit Arb enclosures (endpoint radius $\le 1.4\times 10^{-3}$ pts). Bold entries certify separation exceeding one Cboe tick ($0.05$ pts).",
        r"\end{table}",
    ])
    write_tex("tab_vix_headline.tex", "\n".join(lines))


# =============================================================================
# TABLE 2: Complete 15-Cell E14 Suite (Base, Curved JMM Scenario 2, Stress)
# =============================================================================
def build_tab2_e14():
    rows = load_json(RES / "vix_futures_full_suite.json")["block_a_e14_15cells"]

    def t_lbl(T: float) -> str:
        if abs(T - 7 / 365) < 1e-3:
            return "1w"
        if abs(T - 30 / 365) < 5e-3 or abs(T - 1 / 12) < 5e-3:
            return "1m"
        if abs(T - 0.25) < 1e-2:
            return "3m"
        return "6m"

    groups = [
        ("base", r"\emph{Panel A: Base regime ($\eta_{\mathrm{P1}}=1.9$, flat $\xi_0=0.235^2$)}"),
        ("curve", r"\emph{Panel B: Curved forward variance ($\eta_{\mathrm{P1}}=1.9$, JMM Scenario~2: $\xi_0(t)=0.234^2(1+t)^2$)}"),
        ("stress", r"\emph{Panel C: Stress regime ($\eta_{\mathrm{P1}}=3.0$, $T=6$\,m, flat $\xi_0=0.235^2$)}"),
    ]

    lines = [
        r"\begin{table}[!htbp]\centering\scriptsize\setlength{\tabcolsep}{3.6pt}",
        r"\caption{Complete 15-cell Arb-certified \RoughCert{} VIX futures suite (volatility points, $\Delta=30/365$), benchmarked against JMM~(2018), BDG~(2023), and lognormal proxies.}\label{tab:vix_e14}",
        r"\begin{tabular}{@{}rlcrrrrrr@{}}\toprule",
        r" & & \multicolumn{2}{c}{\RoughCert{} (pts)} & \multicolumn{3}{c}{JMM (2018) comparison} & \multicolumn{2}{c}{Certified proxy gap (pts)} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}",
        r"$H$ & $T$ & $[L,\,U]$ & Width & Width & Low gap & Ratio & $\mathrm{BDG}_3-U$ & $\mathrm{LN}-L$ \\\midrule",
    ]

    for g_idx, (gkey, header) in enumerate(groups):
        if g_idx > 0:
            lines.append(r"\midrule")
        lines.append(f"\\multicolumn{{9}}{{@{{}}l}}{{{header}}} \\\\")
        lines.append(r"\addlinespace[1.5pt]")
        sub = [r for r in rows if r["group"] == gkey]
        for r in sub:
            L, U, w = r["best_L"], r["best_U"], r["best_width"]
            ratio_s = f"{int(round(r['ratio_vs_jmm'])):,}$\\times$"
            if r.get("bdg_minus_U") is not None:
                bdg_s = fmt_signed(r["bdg_minus_U"], 5, up=False, bold_thresh=0.05)
                ln_s = fmt_signed(-r["L_minus_LN"], 4, up=True, bold_thresh=0.05)
            else:
                bdg_s, ln_s = "---", "---"
            lines.append(
                f"{r['H']:.2f} & {t_lbl(r['T'])} & {fmt_band(L, U, 5)} & {fmt_out(w, 5, True)} & "
                f"{fmt_out(r['jmm_width'], 3, True)} & {fmt_out(r['jmm_lo_gap_max'], 4, True)} & {ratio_s} & {bdg_s} & {ln_s} \\\\"
            )

    lines.extend([
        r"\bottomrule\end{tabular}",
        r"\smallskip\par\raggedright\scriptsize\emph{Notes:} Sharpest outward-rounded 169-bit Arb bracket from Theorem~\ref{thm:tilted} and Proposition~\ref{prop:arb}. Low gap $=U-L_{\mathrm{JMM}}$ bounds the JMM lower-bound error; Ratio $=$ JMM width / \RoughCert{} width. Bold marks separation $>0.05$ pts.",
        r"\end{table}",
    ])
    write_tex("tab_vix_e14_15cells.tex", "\n".join(lines))


# =============================================================================
# TABLE 3: Mixed Rough Bergomi VIX Futures (5 Cells)
# =============================================================================
def build_tab3_mixed():
    rows = load_json(RES / "vix_futures_full_suite.json")["block_d_mixed_rbergomi"]

    lines = [
        r"\begin{table}[!htbp]\centering\footnotesize\setlength{\tabcolsep}{4.0pt}",
        r"\caption{Mixed rough Bergomi VIX futures (volatility points, $H=0.10$, $\Delta=1/12$, $\xi_0=0.235^2$): \RoughCertCV{} enclosures versus BDG~(2023) mixed weak expansions.}\label{tab:vix_mixed}",
        r"\begin{tabular}{@{}lcrccrrrr@{}}\toprule",
        r" & & & \multicolumn{2}{c}{\RoughCertCV{} (pts)} & \multicolumn{3}{c}{BDG (2023) mixed (pts)} & \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-8}",
        r"Regime & $(a_1,a_2,\lambda)$ & $T$ & $[L,\,U]$ & Width & BDG$_0$ & BDG$_3$ & $\mathrm{BDG}_3-U$ & Time (s) \\\midrule",
    ]
    for r in rows:
        reg = "Scenario~1" if "scen1" in r["id"] else "Stress"
        params = f"({r['a1']:.1f},\\,{r['a2']:.1f},\\,{r['lambda']:.1f})"
        t_s = f"{r['T_months']}m"
        L, U, w = r["condvar_L"], r["condvar_U"], r["condvar_width"]
        gap = r["bdg3_minus_U"]
        tm = f"{r['setup_s']:.1f} / {r['solve_s']:.1f}"
        lines.append(
            f"{reg} & {params} & {t_s} & {fmt_band(L, U, 5)} & {fmt_out(w, 5, True)} & "
            f"{r['bdg_mixed_order0']:.5f} & {r['bdg_mixed_order3']:.5f} & "
            f"{fmt_signed(gap, 5, up=False, bold_thresh=0.05)} & {tm} \\\\"
        )
    lines.extend([
        r"\bottomrule\end{tabular}",
        r"\smallskip\par\raggedright\scriptsize\emph{Notes:} \RoughCertCV{} uses the shared factor $Z=W(h_0)/\norm{h_0}$ (Proposition~\ref{prop:arb}(c)--(d), $M=140$, $N_z=360$). Time reports setup / solve seconds. BDG values and gaps use floating-point quadrature following \citet[Section~4]{BourgeyDeMarcoGobet2023}.",
        r"\end{table}",
    ])
    write_tex("tab_vix_mixed.tex", "\n".join(lines))


# =============================================================================
# TABLE 4: VIX Call Prices & Black-76 IV Smiles Across 3 Regimes (Refined Stress!)
# =============================================================================
def build_tab4_smiles():
    opt = load_json(RES / "vix_options_smile_suite.json")
    ref = load_json(RES / "refinement.json")["grid"]["options"]
    lab_stress = next(x for x in load_json(RES / "lab_suite.json") if x["name"] == "stress")
    mix = load_json(RES / "mixed_split.json")["rows"]

    lines = [
        r"\begin{table}[!htbp]\centering\scriptsize\setlength{\tabcolsep}{2.5pt}",
        r"\caption{\RoughCertCV{} Arb-certified VIX call prices (pts) and Black-76 implied volatilities (\%) versus BDG~(2023), LAB~(2026), and lognormal proxies across three smile regimes.}\label{tab:vix_smiles}",
        r"\begin{tabular}{@{}rrccrrccrr@{}}\toprule",
        r" & & \multicolumn{4}{c}{VIX call price (pts)} & \multicolumn{4}{c}{Black-76 implied volatility (\%)} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
        r"$k$ & $K$ & \RoughCertCV{} $[C_L,C_U]$ & Width & BDG$_3$ & LN & \RoughCertCV{} $[\sigma_L,\sigma_U]$ & BDG$_3$ & LAB & LN \\\midrule",
    ]

    # Panel A: Pure rBergomi (BDG Fig. 6 — 5 representative strikes k=-0.4,-0.2,0,+0.2,+0.4)
    lines.append(r"\multicolumn{10}{@{}l}{\emph{Panel A: Pure rough Bergomi --- BDG~(2023) Fig.~6 ($H=0.10$, $T=3$\,m, $a=1.0$, $\Delta=1/12$, $\xi_0=0.235^2$)}} \\")
    lines.append(r"\addlinespace[1.5pt]")
    keep_k1 = {-0.4, -0.2, 0.0, 0.2, 0.4}
    for s in opt["bdg2023_fig6"]["strikes"]:
        if round(s["log_moneyness"], 2) not in keep_k1:
            continue
        k_s = fmt_signed(s["log_moneyness"], 2)
        bdg_c = 0.5 * (s["bdg_C3_volpts"][0] + s["bdg_C3_volpts"][1])
        ln_c = 0.5 * (s["ln_call_volpts"][0] + s["ln_call_volpts"][1])
        bdg_iv = 0.5 * (s["bdg_iv3_pct"][0] + s["bdg_iv3_pct"][1])
        ln_iv = 0.5 * (s["ln_iv_pct"][0] + s["ln_iv_pct"][1])
        lines.append(
            f"{k_s} & {s['K_volpts']:.3f} & {fmt_band(s['call_L_volpts'], s['call_U_volpts'], 5)} & "
            f"{fmt_out(s['call_width_volpts'], 5, True)} & {bdg_c:.5f} & {ln_c:.5f} & "
            f"{fmt_band(s['iv_L_pct'], s['iv_U_pct'], 2)} & {bdg_iv:.2f} & --- & {ln_iv:.2f} \\\\"
        )

    # Panel B: Stress Cell (Refined 0.55 pp ATM band from refinement.json + BDG3 + LAB 2026 + LN)
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{10}{@{}l}{\emph{Panel B: Stress regime (refined grid) --- E14 ($H=0.10$, $T=6$\,m, $\eta_{\mathrm{P1}}=3.0$, $\Delta=30/365$, $\xi_0=0.235^2$)}} \\")
    lines.append(r"\addlinespace[1.5pt]")
    assert [r["K_decimal"] for r in ref] == [r["K_decimal"] for r in opt["e14_stress_smile"]["strikes"]] == [r["K_decimal"] for r in lab_stress["arb"]["options"]]
    for s, r_opt, l_iv in zip(opt["e14_stress_smile"]["strikes"], ref, map(np.mean, lab_stress["lab_arb_iv"]), strict=True):
        k_s = fmt_signed(s["log_moneyness"], 2)
        bdg_c = 0.5 * (s["bdg_C3_volpts"][0] + s["bdg_C3_volpts"][1])
        ln_c = 0.5 * (s["ln_call_volpts"][0] + s["ln_call_volpts"][1])
        bdg_iv = 0.5 * (s["bdg_iv3_pct"][0] + s["bdg_iv3_pct"][1])
        ln_iv = 0.5 * (s["ln_iv_pct"][0] + s["ln_iv_pct"][1])
        lines.append(
            f"{k_s} & {r_opt['K_volpts']:.3f} & {fmt_band(r_opt['call_L_volpts'], r_opt['call_U_volpts'], 5)} & "
            f"{fmt_out(r_opt['call_width_volpts'], 5, True)} & {bdg_c:.5f} & {ln_c:.5f} & "
            f"{fmt_band(r_opt['iv_L_pct'], r_opt['iv_U_pct'], 2)} & {bdg_iv:.2f} & {l_iv:.2f} & {ln_iv:.2f} \\\\"
        )

    # Panel C: Mixed rough Bergomi (5 representative strikes k=-0.3,-0.2,0,+0.1,+0.3 with root-split BDG3!)
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{10}{@{}l}{\emph{Panel C: Mixed rough Bergomi --- Scenario~1 ($H=0.10$, $T=3$\,m, $a_1=1.4$, $a_2=0.7$, $\lambda=0.3$, $\Delta=1/12$)}} \\")
    lines.append(r"\addlinespace[1.5pt]")
    mix_suite = opt["mixed_rbergomi_smile"]["strikes"]
    keep_idx = [0, 1, 3, 4, 6]
    for idx in keep_idx:
        ms = mix_suite[idx]
        mr = mix[idx]
        k_s = fmt_signed(ms["log_moneyness"], 2)
        sp_c = 0.5 * (mr["split_call"][0] + mr["split_call"][1])
        sp_iv = 0.5 * (mr["split_iv"][0] + mr["split_iv"][1])
        lines.append(
            f"{k_s} & {ms['K_volpts']:.3f} & {fmt_band(ms['call_L_volpts'], ms['call_U_volpts'], 5)} & "
            f"{fmt_out(ms['call_width_volpts'], 5, True)} & {sp_c:.5f} & --- & "
            f"{fmt_band(ms['iv_L_pct'], ms['iv_U_pct'], 2)} & {sp_iv:.2f} & --- & --- \\\\"
        )

    lines.extend([
        r"\bottomrule\end{tabular}",
        r"\smallskip\par\raggedright\scriptsize\emph{Notes:} $k=\log(K/F_{\mathrm{mid}})$. Every \RoughCertCV{} IV bracket propagates forward and call uncertainty in 169-bit Arb (Proposition~\ref{prop:arb}(e)). Panel~B uses the refined stress grid (ATM IV width $0.55$\,pp). Panel~C reports Arb root-split BDG$_3$ midpoints.",
        r"\end{table}",
    ])
    write_tex("tab_vix_smiles.tex", "\n".join(lines))


# =============================================================================
# TABLE 5: Unified 3-Panel Competitor Audit (LAB 2026 + Markovian + BCJ Quantiser)
# =============================================================================
def build_tab5_competitors():
    lab = load_json(RES / "lab_suite.json")
    fd = load_json(RES / "factor_decisions.json")
    q12 = load_json(RES / "quantize_suite.json")
    q30 = load_json(RES / "bcj_calendar.json")

    lines = [
        r"\begin{table}[!htbp]\centering\scriptsize\setlength{\tabcolsep}{2.6pt}",
        r"\caption{Multi-method competitor audit against \RoughCertCV{} across direct IV expansions (LAB~2026), Markovian factor reductions (Bayer--Breneis), and product functional quantisation (BCJ~2023).}\label{tab:competitors}",
        r"\begin{tabular}{@{}lcccccc@{}}\toprule",
        # --- Panel A: LAB 2026 ---
        r"\multicolumn{7}{@{}l}{\emph{Panel A: Direct implied-volatility expansion \citep[Eq.~(21)]{LiaoAgarwalBourgey2026} below \RoughCertCV{} lower bound}} \\",
        r"\addlinespace[1.5pt]",
        r"Regime & $H$ & $T$ & Below & Overlap & Max gap (pp) & Status \\\midrule",
    ]
    for row in lab[:-1]:
        target = load_json(RES / "refinement.json")["grid"]["options"] if row["name"]=="stress" else row["arb"]["options"]
        assert [o["K_decimal"] for o in target]==[o["K_decimal"] for o in row["arb"]["options"]]
        cnt = sum(b[1] < o["iv_L_pct"] for b, o in zip(row["lab_arb_iv"], target, strict=True))
        tot = len(row["lab_arb_iv"])
        ov = sum(b[0] <= o["iv_U_pct"] and b[1] >= o["iv_L_pct"] for b,o in zip(row["lab_arb_iv"],target,strict=True))
        assert cnt + ov == tot
        gaps = [o["iv_L_pct"] - b[1] for b, o in zip(row["lab_arb_iv"], target, strict=True)]
        reg = "Stress ($\\eta_{\\mathrm{P1}}=3$)" if row["name"] == "stress" else f"LAB author grid"
        h_s = "0.10" if row["name"] == "stress" else row["H"]
        stat = "All below" if ov == 0 else f"{ov} overlap"
        lines.append(
            f"{reg} & {h_s} & {row['T_months']}\\,m & {cnt}/{tot} & {ov}/{tot} & "
            f"{fmt_out(max(gaps), 3, False)} & {stat} \\\\"
        )

    # --- Panel B: Markovian Factor Thresholds ---
    lines.extend([
        r"\midrule",
        r"\multicolumn{7}{@{}l}{\emph{Panel B: Bayer--Breneis Markovian factor thresholds ($T=3$\,m, $a=1.0$, $\Delta=1/12$, $\xi_0=0.235^2$)}} \\",
        r"\addlinespace[1.5pt]",
        r"Hurst $H$ & Futures $<1$ tick & Band entry & Call $K=18$ & Call $K=22$ & Call $K=26$ & Verification \\\midrule",
    ])
    for h_key, h_lbl in [(".05", "0.05"), (".10", "0.10"), (".30", "0.30")]:
        obj = fd[h_key]
        t_list = obj["tick_thresholds_future_then_calls"]
        b_th = obj["fixed_band_threshold"]["first_proved"]
        def fmt_th(item):
            if item["minimum_proved"]:
                return f"\\textbf{{{item['first_proved']}}}"
            prev = item["earlier_unresolved"][0]
            return f"{prev} or {item['first_proved']}$^\\dagger$"
        ver = "Exact min (fut.)" if h_key == ".05" else "All exact mins"
        lines.append(
            f"{h_lbl} & {fmt_th(t_list[0])} & \\textbf{{{b_th}}} & "
            f"{fmt_th(t_list[1])} & {fmt_th(t_list[2])} & {fmt_th(t_list[3])} & {ver} \\\\"
        )

    # --- Panel C: BCJ Product Functional Quantiser ---
    lines.extend([
        r"\midrule",
        r"\multicolumn{7}{@{}l}{\emph{Panel C: BCJ~(2023) product functional quantisation signed futures errors (pts) and median runtimes (s)}} \\",
        r"\addlinespace[1.5pt]",
        r"Budget / Paths & \multicolumn{2}{c}{18-cell ($\Delta=1/12$)} & 18-cell ($30/365$) & Stress (pts) & Setup (s) & Price (s) \\\midrule",
    ])
    for b in [50, 100, 500, 1000]:
        r12 = [z for z in q12 if z["budget"] == b and z["curve"] != "stress"]
        r30 = [z for z in q30 if z["budget"] == b]
        st = next(z for z in q12 if z["budget"] == b and z["curve"] == "stress")
        e12_lo = min(z["signed_future_error"][0] for z in r12)
        e12_hi = max(z["signed_future_error"][1] for z in r12)
        e30_lo = min(z["signed_future_error"][0] for z in r30)
        e30_hi = max(z["signed_future_error"][1] for z in r30)
        setup_med = float(np.median([z["basis_setup_s"] + z["product_setup_s"] for z in r12]))
        price_med = float(np.median([z["online_s"] for z in r12]))
        lines.append(
            f"{b} / {st['actual_paths']} & \\multicolumn{{2}}{{c}}{{{fmt_band(e12_lo, e12_hi, 3)}}} & "
            f"{fmt_band(e30_lo, e30_hi, 3)} & {fmt_band(st['signed_future_error'][0], st['signed_future_error'][1], 3)} & "
            f"{setup_med:.2f} & {price_med:.2f} \\\\"
        )

    lines.extend([
        r"\bottomrule\end{tabular}",
        r"\smallskip\par\raggedright\scriptsize\emph{Notes:} All intervals outward-rounded in 169-bit Arb. Panel~A: shortfalls below \RoughCertCV{} IV lower bound (zero strikes above). Panel~B: bold values are Arb-proved exact minimum factor counts; $^\dagger$one-factor overlap bracket at $H=0.05$ ($N=5$ at $K=18$, $N=1$ at $K=22$). Panel~C: negative errors reflect downward quantiser bias.",
        r"\end{table}",
    ])
    write_tex("tab_competitors.tex", "\n".join(lines))


# =============================================================================
# TABLE 6: Flagship 8-Method Common Curved Contract Scoreboard (K = 27)
# =============================================================================
def build_tab6_common():
    values = scoreboard()
    descriptions = [
        (r"\textbf{\RoughCertCV{} (continuum)}", "Reference"),
        ("Markovian ($N=12$)", "Overlaps on all three"),
        ("500-cell right grid", "Future above"),
        ("BDG (2023) order-3", "Overlaps on all three"),
        ("LAB (2026) direct IV", "IV overlaps"),
        ("Moment-matched lognormal", "Future below; IV above"),
        ("BCJ quantiser (960 paths)", "Below on all three"),
        ("JMM (2018) bounds", "Contains futures band"),
    ]
    rows = []
    for (label, verdict), intervals in zip(descriptions, values, strict=True):
        cells = [fmt_band(*z, digits) if z else "---" for z, digits in zip(intervals, (3,3,2), strict=True)]
        rows.append((label, "Arb-cert.", *cells, verdict))
    lines = [
        r"\begin{table}[!htbp]\centering\scriptsize\setlength{\tabcolsep}{3.4pt}",
        r"\caption{Eight-method comparison on a common curved forward-variance contract ($H=0.10$, $T=3$\,m, $\Delta=1/12$, $\xi_0(t)=0.234^2(1+t)^2$, strike $K=27$ pts).}\label{tab:common}",
        r"\begin{tabular}{@{}lccccr@{}}\toprule",
        r"Method & Status & Future (pts) & Call $K=27$ (pts) & Black-76 IV (\%) & Verdict vs.\ \RoughCertCV{} \\\midrule",
    ]
    for r in rows:
        lines.append(" & ".join(r) + r" \\")
    lines.extend([
        r"\bottomrule\end{tabular}",
        r"\smallskip\par\raggedright\scriptsize\emph{Notes:} All intervals outward-rounded in 169-bit Arb. Each method's IV uses its own forward at fixed $K=27$. BDG and LAB Arb enclosures reflect curved-kernel $M=360$ quadrature; refined numerical references sit $0.14$ and $0.13$\,pp below the \RoughCertCV{} IV lower bound.",
        r"\end{table}",
    ])
    write_tex("tab_common.tex", "\n".join(lines))


# =============================================================================
# TABLE 7: Unified Rough Stein-Stein Table (Daily VaR + Calls + Intraday Bridge)
# =============================================================================
def build_tab7_steinstein():
    # Uses exact numbers from verified E13 legacy JSONs (matching original tab_e13 & tab_e13b)
    e13 = load_json(LEG / "E13_rough_stein_stein" / "e13_results.json")
    e13c = load_json(LEG / "E13_rough_stein_stein" / "e13c_results.json")
    e13d = load_json(LEG / "E13_rough_stein_stein" / "e13d_results.json")
    e13_fl = load_json(LEG / "E13_rough_stein_stein" / "e13_floor.json")
    b05 = load_json(LEG / "E13_rough_stein_stein" / "e13b_results_H0.05.json")
    b10 = load_json(LEG / "E13_rough_stein_stein" / "e13b_results_H0.1.json")

    def sci_tex(v: float) -> str:
        m, e = f"{v:.1e}".split("e")
        return f"${m}\\times 10^{{{int(e)}}}$"

    lines = [
        r"\begin{table}[!htbp]\centering\scriptsize\setlength{\tabcolsep}{3.5pt}",
        r"\caption{Rough Stein--Stein one-month integrated variance ($n=21$ daily closes): statistical $99\%$ VaR brackets (in exact sd), variance-call widths (relative to exact price), and intraday Gaussian-bridge refinement.}\label{tab:e13}",
        r"\begin{tabular}{@{}rrrcccccr@{}}\toprule",
        # --- Panel A: Daily 99% VaR ---
        r"\multicolumn{9}{@{}l}{\emph{Panel A: Daily-close $99\%$ VaR bracket widths in exact sd (empirical coverage in parentheses)}} \\",
        r"\addlinespace[1.5pt]",
        r"$H$ & Daily err. & Mon.\ err. & Riem.--$W_2$ & Martingale & Cantelli & DKW fl. & Binom.\ / fl. & Daily MC \\\midrule",
    ]

    for r, rd in ((r, next(z for z in e13d if z["H"]==r["H"])) for r in e13):
        hk = str(r["H"])
        fl_dkw = e13_fl[hk]["floor_sd"]
        fl_bin = rd["floor_bin_sd"]
        lines.append(
            f"{r['H']:g} & {100*r['riemann_rel_rms']:.1f}\\% & {100*r['ce_rel_rms']:.1f}\\% & "
            f"{r['width_a_sd']:.2f} ({r['coverage_a']:.2f}) & {r['width_c_sd']:.2f} ({r['coverage_c']:.2f}) & "
            f"{r['width_d_sd']:.2f} ({r['coverage_d']:.2f}) & {fl_dkw:.2f} & "
            f"{rd['width_dbin_sd']:.2f} ({rd['coverage_dbin']:.2f})/{fl_bin:.2f} & "
            f"{fmt_signed(r['naive_q99_bias_sd'], 3)} \\\\"
        )

    # --- Panel B: Daily Variance Calls ---
    lines.extend([
        r"\midrule",
        r"\multicolumn{9}{@{}l}{\emph{Panel B: Variance calls $\E(\Lambda-k)_+$ --- certified width / exact price (coverage in parentheses)}} \\",
        r"\addlinespace[1.5pt]",
        r"$H$ & Strike $k$ & Exact price & Lipschitz & Convex & Scarf--Lo & Exact $\hat\Lambda$ & & Daily MC \\\midrule",
    ])
    for r, rc in ((r, next(z for z in e13c if z["H"]==r["H"])) for r in e13):
        for pkey, plbl in [("ATM", r"$\E\Lambda$"), ("mean+2sd", r"$+2\,\mathrm{sd}$")]:
            pr = r["prices"][pkey]
            prc = rc["prices"][pkey]
            lines.append(
                f"{r['H']:g} & {plbl} & {sci_tex(pr['exact'])} & "
                f"{pr['lipschitz_width_rel']:.2f} ({pr['coverage']['lipschitz']:.2f}) & "
                f"{pr['convex_width_rel']:.2f} ({pr['coverage']['convex']:.2f}) & "
                f"{pr['scarf_width_rel']:.3f} ({pr['coverage']['scarf']:.2f}) & "
                f"{prc['width_rel']:.3f} ({prc['coverage']:.2f}) & & "
                f"{fmt_signed(100*pr['naive_daily_bias_rel'], 1)}\\% \\\\"
            )

    # --- Panel C: Intraday Gaussian-Bridge Refinement ---
    lines.extend([
        r"\midrule",
        r"\multicolumn{9}{@{}l}{\emph{Panel C: Intraday Gaussian-bridge conditioning refinement for $99\%$ VaR (continuum law unchanged)}} \\",
        r"\addlinespace[1.5pt]",
        r"$H$ & Pts/day & $\E[V]/\Var$ & & Martingale & Cantelli & DKW fl. & & Coverage \\\midrule",
    ])
    for br in b05 + b10:
        lines.append(
            f"{br['H']:g} & {br['per_day']} & {sci_tex(br['residual_share'])} & & "
            f"{br['width_c_sd']:.2f} & {br['width_d_sd']:.2f} & {br['width_floor_sd']:.2f} & & "
            f"{br['coverage_d']:.2f} \\\\"
        )

    lines.extend([
        r"\bottomrule\end{tabular}",
        r"\smallskip\par\raggedright\scriptsize\emph{Notes:} $\sigma_0=0.2$, $\nu=0.1$, $T=1/12$, $K=2\times 10^5$, $a=0.01$ ($50$ reps in Panels~A--B; $10$ in Panel~C). Daily/Mon.\ err.: exact RMS error (\% of sd). Daily MC: signed bias vs.\ exact Karhunen--Lo\`eve law.",
        r"\end{table}",
    ])
    write_tex("tab_e13.tex", "\n".join(lines))


# =============================================================================
# TABLE 8: Rough Bergomi Integrated Variance Contrast (Redesigned Cleanly)
# =============================================================================
def build_tab8_e6c():
    content = r"""\begin{table}[!htbp]\centering\footnotesize\setlength{\tabcolsep}{4.2pt}
\caption{Rough Bergomi one-month integrated variance and volatility from $21$ daily closes: statistical $99\%$ quantile widths (in sd) and median variance-call bounds ($\times 10^4$).}\label{tab:e6c}
\begin{tabular}{@{}llrcccccc@{}}\toprule
\multicolumn{9}{@{}l}{\emph{Panel A: $99\%$ quantile widths in exact sd ($21$ daily closes vs.\ $256$-date refinement)}} \\
\addlinespace[1.5pt]
Functional & $H$ & Mon.\ err. & Martingale & Cantelli & DKW fl. & Ratio & 256 dates & MC in \\\midrule
PANEL_A_ROWS
\midrule
\multicolumn{9}{@{}l}{\emph{Panel B: Variance-call brackets ($\times 10^4$) vs.\ fine-grid Monte Carlo diagnostic}} \\
\addlinespace[1.5pt]
$H$ & Strike & & \multicolumn{2}{c}{Convex order} & \multicolumn{2}{c}{Conditional Scarf--Lo} & Fine MC & \\\midrule
PANEL_B_ROWS
\bottomrule\end{tabular}
\smallskip\par\raggedright\scriptsize\emph{Notes:} $\xi_0=0.235^2$, $\eta_{\mathrm{P1}}=1.9$, $T=1/12$, $K=2\times 10^5$, $5$ replications, DKW $a=0.01$. Ratio $=$ Martingale / conditional Cantelli width (singularity-corrected per-sample $V$). MC in: fraction of replications containing the fine-grid diagnostic.
\end{table}"""
    raw = load_json(LEG / "E6c_rough_vol_condvar/e6c_results.json")
    rows_a, rows_b = [], []
    for r in sorted(raw,key=lambda r:(r["H"],r["output"])):
        name = {"IV":"Int. variance","IVol":"Int. volatility"}[r["output"]]
        cells = [name,f"{r['H']:.2f}",f"{100*r['ce_rel_rms']:.0f}\\%",
                 f"{r['width_c_sd']:.2f}",f"{r['width_d_sd']:.2f}",f"{r['width_floor_sd']:.2f}",
                 f"{r['ratio_c_d']:.2f}"+r"$\times$",f"{r['e6b_width_sd']:.2f}",f"{r['mc_inside_d']:.2f}"]
        rows_a.append(" & ".join(cells)+r" \\")
        for label, pr in r.get("prices",{}).items():
            k = r"$\E\,\mathrm{IV}$" if label=="ATM" else r"$+2\,\mathrm{sd}$"
            bands=[r"\multicolumn{2}{c}{"+fmt_band(*(1e4*x for x in pr[key]),3)+"}" for key in ("convex_median","scarf_median")]
            rows_b.append(" & ".join([f"{r['H']:.2f}",k,"",*bands,f"{1e4*pr['fine_mc_e6b']:.3f}",""])+r" \\")
    content=content.replace("PANEL_A_ROWS","\n".join(rows_a)).replace("PANEL_B_ROWS","\n".join(rows_b))
    write_tex("tab_e6c.tex", content)


# =============================================================================
# SUPPLEMENT TABLE S1: Complete 18-Cell BCJ Multi-Curve RoughCert-CV Suite
# =============================================================================
def build_tab_s1_bcj():
    rows = load_json(RES / "vix_futures_full_suite.json")["block_c_bcj2023_18cells"]
    c_order = [
        ("flat", r"Flat: $\xi_0(t)=0.234^2$"),
        ("quad", r"Quadratic: $\xi_0(t)=0.234^2(1+t)^2$"),
        ("sqrt", r"Square-root: $\xi_0(t)=0.234^2\sqrt{1+t}$"),
    ]
    lines = [
        r"\begin{table}[!htbp]\centering\scriptsize\setlength{\tabcolsep}{4.5pt}",
        r"\caption{Complete 18-cell Bonesini--Callegaro--Jacquier (2023, Section~5.3) \RoughCertCV{} VIX futures suite across three forward-variance curves and six maturities ($H=0.10$, $\eta_{\mathrm{P1}}=1.9$, $\Delta=1/12$).}\label{tab:bcj_full}",
        r"\begin{tabular}{@{}rcrrrrr@{}}\toprule",
        r"Maturity $T$ & \RoughCertCV{} $[L,\,U]$ (pts) & Width (pts) & JMM width (pts) & Tightening & Setup (s) & Solve (s) \\\midrule",
    ]
    for g_i, (ck, title) in enumerate(c_order):
        if g_i > 0:
            lines.append(r"\midrule")
        lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        lines.append(r"\addlinespace[1.5pt]")
        sub = sorted([r for r in rows if ck in r["curve"]], key=lambda x: x["T_months"])
        for r in sub:
            lines.append(
                f"{r['T_months']}m & {fmt_band(r['condvar_L'], r['condvar_U'], 5)} & "
                f"{fmt_out(r['condvar_width'], 5, True)} & {fmt_out(r['jmm_width'], 3, True)} & "
                f"{int(round(r['ratio_vs_jmm'])):,}$\\times$ & {r['setup_s']:.1f} & {r['solve_s']:.1f} \\\\"
            )
    lines.extend([
        r"\bottomrule\end{tabular}",
        r"\smallskip\par\raggedright\scriptsize\emph{Notes:} All 18 continuous-time enclosures use Proposition~4(c)--(d) of the main paper ($M=120$, $N_z=310$) and have widths below $0.013$ pts (one standard futures tick is $0.05$ pts). Setup is paid once per $(H,T,\Delta)$ cell.",
        r"\end{table}",
    ])
    write_tex("tab_bcj_full.tex", "\n".join(lines))


# =============================================================================
# SUPPLEMENT TABLE S2: Paired 95% Quantile Verification Table (tab_e5c redesigned)
# =============================================================================
def build_tab_s2_e5c():
    e5c = load_json(LEG / "E5c_certified_cantelli" / "e5c_results.json")
    e5bin = load_json(LEG / "E5c_certified_cantelli" / "e5c_binomial.json")
    e5fl = load_json(LEG / "E5c_certified_cantelli" / "e5c_floor.json")

    def sci_tex(v: float) -> str:
        m, e = f"{v:.1e}".split("e")
        return f"${m}\\times 10^{{{int(e)}}}$"

    lines = [
        r"\begin{table}[!htbp]\centering\scriptsize\setlength{\tabcolsep}{3.6pt}",
        r"\caption{Paired statistical $95\%$ quantile brackets for $\int_0^1 B_H(s)^2\dd s$ from $n$-point midpoint observations ($K=20{,}000$, $100$ replications).}\label{tab:e5c}",
        r"\begin{tabular}{@{}rrrcccccccc@{}}\toprule",
        r" & & & & \multicolumn{4}{c}{DKW median widths} & & \multicolumn{2}{c}{Binomial OS} \\",
        r"\cmidrule(lr){5-8}\cmidrule(lr){10-11}",
        r"$H$ & $n$ & $P/\Var$ & Cov.\ (c/d/d'/bin) & Floor & Mart.\ (c) & CV (d) & RB (d') & Excess & CV-bin & Floor \\\midrule",
    ]
    paired = {(r["H"], r["n"]): r for r in e5bin}
    for rc in e5c:
        rb = paired[rc["H"],rc["n"]]
        key = f"{rc['H']:g}_{rc['n']}"
        fl_dkw = e5fl[key]["floor"]
        exc = (rc["width_c"] - fl_dkw) / (rc["width_d"] - fl_dkw)
        covs = f"{rc['coverage_c']:.2f}/{rc['coverage_d']:.2f}/{rc['coverage_d_rb']:.2f}/{rb['coverage_dbin']:.2f}"
        lines.append(
            f"{rc['H']:g} & {rc['n']} & {sci_tex(rc['residual_share'])} & {covs} & "
            f"{fl_dkw:.3f} & {rc['width_c']:.3f} & {rc['width_d']:.3f} & {rc['width_d_rb']:.3f} & "
            f"{exc:.1f}$\\times$ & {rb['width_dbin']:.3f} & {rb['floor_bin']:.3f} \\\\"
        )
    lines.extend([
        r"\bottomrule\end{tabular}",
        r"\smallskip\par\raggedright\scriptsize\emph{Notes:} Statistical brackets; median widths on identical draws: (c) martingale-sharp; (d) conditional Cantelli (Proposition~2); (d') Rao--Blackwellised; CV-bin: exact binomial order statistics. Floor: sampling band of $\hat\Lambda$ alone; Excess: ratio of (c) to (d) width above floor.",
        r"\end{table}",
    ])
    write_tex("tab_e5c.tex", "\n".join(lines))


def main():
    from build_tables_and_figures import load_jsons, write_macros
    write_macros(*load_jsons())
    build_tab1_headline()
    build_tab2_e14()
    build_tab3_mixed()
    build_tab4_smiles()
    build_tab5_competitors()
    build_tab6_common()
    build_tab7_steinstein()
    build_tab8_e6c()
    build_tab_s1_bcj()
    build_tab_s2_e5c()
    # Remove obsolete fragmented v2 sub-tables in v3 (v1 and v2 dirs untouched!)
    for old in ["tab_direct.tex", "tab_quantiser.tex", "tab_e13b.tex"]:
        p = TAB / old
        if p.exists():
            p.unlink()
            print(f"REMOVED fragmented copy in v3: {old}")


if __name__ == "__main__":
    main()
