"""Generate all LaTeX tables, summary macros, and publication-quality PDF figures
directly from the verified 169-bit Arb JSON datasets.

Inputs (verified JSONs):
  - results/week1_gates_results.json
  - results/vix_futures_full_suite.json
  - results/vix_options_smile_suite.json

Outputs (inside manuscript/):
  - tables/vix_arb_numbers.tex
  - tables/tab_vix_headline.tex
  - tables/tab_vix_e14_15cells.tex
  - tables/tab_vix_mixed.tex
  - tables/tab_vix_smiles.tex
  - tables/tab_bcj_full.tex
  - figures/fig_vix_phase_boundary.pdf
  - figures/fig_vix_smiles.pdf
  - figures/fig_vix_bcj_disjoint.pdf
"""
from __future__ import annotations

import json
from build_comparison import num
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MAN = ROOT / "manuscript"
TAB_DIR = MAN / "tables"
FIG_DIR = MAN / "figures"
TAB_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Okabe-Ito CVD-safe palette
C_BLUE = "#0072B2"
C_VERM = "#D55E00"
C_TEAL = "#009E73"
C_PURP = "#CC79A7"
C_GOLD = "#E69F00"
C_GRAY = "#555555"
C_DARK = "#1A1A1A"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Computer Modern Roman", "Times New Roman"],
    "mathtext.fontset": "cm",
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.4,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.4,
    "grid.linewidth": 0.45,
    "grid.alpha": 0.35,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})


def load_jsons():
    g = json.loads((RES / "week1_gates_results.json").read_text(encoding="utf-8"))
    f = json.loads((RES / "vix_futures_full_suite.json").read_text(encoding="utf-8"))
    o = json.loads((RES / "vix_options_smile_suite.json").read_text(encoding="utf-8"))
    return g, f, o


def write_macros(g, f, o):
    cells = {c["id"]: c for c in g["gate2"]["cells"]}
    e14 = {r["id"]: r for r in f["block_a_e14_15cells"]}
    bcj = f["block_c_bcj2023_18cells"]
    mix = f["block_d_mixed_rbergomi"]

    max_e14_w = max(r["best_width"] for r in f["block_a_e14_15cells"])
    max_e14_base_w = max(r["best_width"] for r in f["block_a_e14_15cells"] if r["group"] in ("base", "curve"))
    min_jmm_ratio = min(r["ratio_vs_jmm"] for r in f["block_a_e14_15cells"])
    max_jmm_ratio = max(r["ratio_vs_jmm"] for r in f["block_a_e14_15cells"])
    max_bcj_w = max(r["condvar_width"] for r in bcj)
    min_bcj_ratio = min(r["ratio_vs_jmm"] for r in bcj)
    max_bcj_ratio = max(r["ratio_vs_jmm"] for r in bcj)
    max_mix_s1_w = max(r["condvar_width"] for r in mix if "scen1" in r["id"])
    max_mix_all_w = max(r["condvar_width"] for r in mix)
    max_mix_bdg_gap = max(r["bdg3_minus_U"] for r in mix)

    lines = [
        "% Auto-generated from verified 169-bit Arb JSON results -- DO NOT EDIT BY HAND",
        f"\\newcommand{{\\ArbStressTiltedL}}{{{num(cells['E14_stress_H010']['tilted_L_outer'],6,False)}}}",
        f"\\newcommand{{\\ArbStressTiltedU}}{{{num(cells['E14_stress_H010']['tilted_U_outer'],6,True)}}}",
        f"\\newcommand{{\\ArbStressTiltedW}}{{{num(cells['E14_stress_H010']['tilted_total_outer_width'],5,True)}}}",
        f"\\newcommand{{\\ArbStressBDG}}{{{cells['E14_stress_H010']['bdg_F3_arb'][0]:.6f}}}",
        f"\\newcommand{{\\ArbStressBDGGap}}{{{num(cells['E14_stress_H010']['bdg_minus_U_cert'],5,False)}}}",
        f"\\newcommand{{\\ArbStressLNGap}}{{{num(cells['E14_stress_H010']['L_minus_LN_cert'],4,False)}}}",
        f"\\newcommand{{\\ArbStressCondL}}{{{num(e14['stress_H010_6m']['condvar_L'],5,False)}}}",
        f"\\newcommand{{\\ArbStressCondU}}{{{num(e14['stress_H010_6m']['condvar_U'],5,True)}}}",
        f"\\newcommand{{\\ArbStressCondW}}{{{num(e14['stress_H010_6m']['condvar_width'],5,True)}}}",
        f"\\newcommand{{\\ArbFigOneL}}{{{num(cells['BDG2023_Fig1_a15_T6m']['tilted_L_outer'],6,False)}}}",
        f"\\newcommand{{\\ArbFigOneU}}{{{num(cells['BDG2023_Fig1_a15_T6m']['tilted_U_outer'],6,True)}}}",
        f"\\newcommand{{\\ArbFigOneW}}{{{num(cells['BDG2023_Fig1_a15_T6m']['tilted_total_outer_width'],5,True)}}}",
        f"\\newcommand{{\\ArbFigOneBDG}}{{{cells['BDG2023_Fig1_a15_T6m']['bdg_F3_arb'][0]:.6f}}}",
        f"\\newcommand{{\\ArbFigOneBDGGap}}{{{num(cells['BDG2023_Fig1_a15_T6m']['bdg_minus_U_cert'],5,False)}}}",
        f"\\newcommand{{\\ArbFigOneLNGap}}{{{num(cells['BDG2023_Fig1_a15_T6m']['L_minus_LN_cert'],4,False)}}}",
        f"\\newcommand{{\\ArbStressHZeroFiveBDGGap}}{{{num(e14['stress_H005_6m']['bdg_minus_U'],5,False)}}}",
        f"\\newcommand{{\\ArbBDContL}}{{{num(cells['BD2022_MLMC_JCF']['tilted_L_outer'],6,False)}}}",
        f"\\newcommand{{\\ArbBDContU}}{{{num(cells['BD2022_MLMC_JCF']['tilted_U_outer'],6,True)}}}",
        f"\\newcommand{{\\ArbBDContW}}{{{num(cells['BD2022_MLMC_JCF']['tilted_total_outer_width'],6,True)}}}",
        f"\\newcommand{{\\ArbBDGridL}}{{{num(cells['BD2022_MLMC_JCF']['grid500_tilted_arb']['L_outer_volpts'],6,False)}}}",
        f"\\newcommand{{\\ArbBDGridU}}{{{num(cells['BD2022_MLMC_JCF']['grid500_tilted_arb']['U_outer_volpts'],6,True)}}}",
        f"\\newcommand{{\\ArbBDDisjointGap}}{{{num(cells['BD2022_MLMC_JCF']['grid500_tilted_arb']['L_outer_volpts'] - cells['BD2022_MLMC_JCF']['tilted_U_outer'],6,False)}}}",
        f"\\newcommand{{\\ArbEFourteenMaxW}}{{{num(max_e14_w,5,True)}}}",
        f"\\newcommand{{\\ArbEFourteenBaseMaxW}}{{{num(max_e14_base_w,5,True)}}}",
        f"\\newcommand{{\\ArbJMMRatioLo}}{{{int(round(min_jmm_ratio)):,}}}".replace(",", "{,}"),
        f"\\newcommand{{\\ArbJMMRatioHi}}{{{int(round(max_jmm_ratio)):,}}}".replace(",", "{,}"),
        f"\\newcommand{{\\ArbBCJMaxW}}{{{num(max_bcj_w,5,True)}}}",
        f"\\newcommand{{\\ArbBCJRatioLo}}{{{int(round(min_bcj_ratio)):,}}}".replace(",", "{,}"),
        f"\\newcommand{{\\ArbBCJRatioHi}}{{{int(round(max_bcj_ratio)):,}}}".replace(",", "{,}"),
        f"\\newcommand{{\\ArbMixScenMaxW}}{{{num(max_mix_s1_w,5,True)}}}",
        f"\\newcommand{{\\ArbMixAllMaxW}}{{{num(max_mix_all_w,5,True)}}}",
        f"\\newcommand{{\\ArbMixMaxBDGGap}}{{{max_mix_bdg_gap:.5f}}}",
    ]
    (TAB_DIR / "vix_arb_numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tab_headline(g):
    label_map = {
        "E14_stress_H010": "Stress (E14)",
        "BDG2023_Fig1_a15_T6m": "BDG (2023) Fig.~1",
        "BD2022_MLMC_JCF": "BD (2022) \\S3.3",
        "BDG2023_Fig6_smile_T3m": "BDG (2023) Fig.~6",
        "BCJ2023_Sec53_flat_T6m": "BCJ (2023) \\S5.3",
        "E14_base_H010_1m": "Base $H=0.10$",
        "E14_base_H005_1m": "Base $H=0.05$",
        "E14_stress_H030": "Stress $H=0.30$",
    }
    order = [
        "E14_stress_H010",
        "BDG2023_Fig1_a15_T6m",
        "BD2022_MLMC_JCF",
        "BDG2023_Fig6_smile_T3m",
        "BCJ2023_Sec53_flat_T6m",
        "E14_base_H010_1m",
        "E14_base_H005_1m",
        "E14_stress_H030",
    ]
    cmap = {c["id"]: c for c in g["gate2"]["cells"]}
    rows_tex = []
    for cid in order:
        c = cmap[cid]
        d_str = "30/365" if abs(c["Delta"] - 30 / 365) < 1e-6 else "1/12"
        t_str = f"{c['T']:.2f}" if c["T"] in (0.25, 0.5) else "1/12"
        gap_b = c["bdg_minus_U_cert"]
        gap_b_s = f"\\textbf{{+{gap_b:.5f}}}" if gap_b > 0.05 else f"+{gap_b:.5f}"
        gap_ln = c["L_minus_LN_cert"]
        gap_ln_s = f"\\textbf{{-{gap_ln:.4f}}}" if gap_ln > 0.05 else f"-{gap_ln:.4f}"
        rows_tex.append(
            f"{label_map[cid]} & {c['H']:.2f} & {t_str} & {d_str} & {c['a']:.3f} & "
            f"[{num(c['tilted_L_outer'],5,False)}, {num(c['tilted_U_outer'],5,True)}] & {c['tilted_total_outer_width']:.5f} & "
            f"{c['bdg_F3_arb'][0]:.5f} & {gap_b_s} & {gap_ln_s} \\\\"
        )
    # Add 500-cell right-rect audit row for BD 2022
    bd = cmap["BD2022_MLMC_JCF"]
    d5 = bd["grid500_tilted_arb"]
    rows_tex.append("\\midrule")
    rows_tex.append(
        f"BD (2022) 500-cell grid & 0.10 & 0.50 & 1/12 & 0.500 & "
        f"[{num(d5['L_outer_volpts'],5,False)}, {num(d5['U_outer_volpts'],5,True)}] & {d5['U_outer_volpts'] - d5['L_outer_volpts']:.5f} & "
        f"\\multicolumn{{3}}{{c}}{{published MLMC = $22.18135$ (inside grid, outside continuum)}} \\\\"
    )

    tex = (
        "\\begin{table}[!htbp]\\centering\\scriptsize\\setlength{\\tabcolsep}{2.6pt}\n"
        "\\caption{Headline 160-bit Arb-certified VIX futures enclosures (volatility points) across eight selected "
        "rough-Bergomi benchmarks, audited against BDG~(2023) order-three and lognormal proxies.}\\label{tab:vix_headline}\n"
        "\\resizebox{\\textwidth}{!}{\\begin{tabular}{@{}lrrrrccrrr@{}}\\toprule\n"
        "Benchmark & $H$ & $T$ & $\\Delta$ & $a$ & Arb Tilted $[L, U]$ & Width & BDG~(2023) & $\\mathrm{BDG}-U$ & $\\mathrm{LN}-L$ \\\\\\midrule\n"
        + "\n".join(rows_tex)
        + "\n\\bottomrule\\end{tabular}}\n"
        "\\smallskip\\par\\raggedright\\scriptsize\\emph{Notes:} $a=\\eta_{\\mathrm{P1}}\\sqrt{2H}$, flat "
        "$\\xi_0=0.235^2$ ($0.234^2$ for BCJ). All intervals are outward-rounded 160-bit Arb enclosures with "
        "enclosure radius $\\le1.4\\times10^{-3}$ points. Bold entries certify separation exceeding one Cboe tick "
        "($0.05$ points).\n"
        "\\end{table}\n"
    )
    (TAB_DIR / "tab_vix_headline.tex").write_text(tex, encoding="utf-8")


def write_tab_e14(f):
    rows_tex = []
    cur_grp = None
    grp_titles = {"base": "Base ($\\eta_{\\mathrm{P1}}=1.9$, flat $\\xi_0=0.235^2$)",
                  "curve": "Curved forward variance ($\\eta_{\\mathrm{P1}}=1.9$, JMM Scenario~2: $\\xi_0(t)=0.234^2(1+t)^2$)",
                  "stress": "Stress ($\\eta_{\\mathrm{P1}}=3.0$, $T=6$\\,m, flat $\\xi_0=0.235^2$)"}
    for r in f["block_a_e14_15cells"]:
        if r["group"] != cur_grp:
            if cur_grp is not None:
                rows_tex.append("\\midrule")
            rows_tex.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{{grp_titles[r['group']]}}}}} \\\\")
            cur_grp = r["group"]
        t_lbl = "1w" if r["T"] < 0.05 else ("1m" if r["T"] < 0.15 else ("3m" if r["T"] < 0.35 else "6m"))
        bdg_s = f"+{r['bdg_minus_U']:.5f}" if "bdg_minus_U" in r else "---"
        if "bdg_minus_U" in r and r["bdg_minus_U"] > 0.05:
            bdg_s = f"\\textbf{{{bdg_s}}}"
        ln_s = f"-{r['L_minus_LN']:.4f}" if "L_minus_LN" in r else "---"
        if "L_minus_LN" in r and r["L_minus_LN"] > 0.05:
            ln_s = f"\\textbf{{{ln_s}}}"
        rows_tex.append(
            f"{r['H']:.2f} & {t_lbl} & [{num(r['best_L'],5,False)}, {num(r['best_U'],5,True)}] & {r['best_width']:.5f} & "
            f"{r['jmm_width']:.3f} & {r['jmm_lo_gap_max']:.4f} & {r['ratio_vs_jmm']:,.0f}$\\times$ & {bdg_s} & {ln_s} \\\\"
        )
    tex = (
        "\\begin{table}[!htbp]\\centering\\scriptsize\\setlength{\\tabcolsep}{4pt}\n"
        "\\caption{Complete 15-cell Arb-certified VIX futures suite (volatility points, $\\Delta=30/365$), "
        "comparing Arb widths against JMM~(2018), BDG~(2023), and lognormal proxies.}\\label{tab:vix_e14}\n"
        "\\begin{tabular}{@{}rlcrrrrrr@{}}\\toprule\n"
        "$H$ & $T$ & Arb $[L, U]$ & Width & JMM width & JMM low gap & Ratio & $\\mathrm{BDG}-U$ & $\\mathrm{LN}-L$ \\\\\\midrule\n"
        + "\n".join(rows_tex)
        + "\n\\bottomrule\\end{tabular}\n"
        "\\smallskip\\par\\raggedright\\scriptsize\\emph{Notes:} Best outward-rounded Arb bracket from "
        "Theorem~\\ref{thm:tilted} and Proposition~\\ref{prop:arb}. JMM low gap $=U-L_{\\mathrm{JMM}}$ bounds "
        "the error of the JMM lower bound; Ratio $=$ JMM width / Arb width. Bold entries certify separation $>0.05$ points.\n"
        "\\end{table}\n"
    )
    (TAB_DIR / "tab_vix_e14_15cells.tex").write_text(tex, encoding="utf-8")


def write_tab_mixed(f):
    rows_tex = []
    for r in f["block_d_mixed_rbergomi"]:
        scen = "Scenario~1" if "scen1" in r["id"] else "Stress"
        g3 = r["bdg3_minus_U"]
        g3_s = f"\\textbf{{+{g3:.5f}}}" if g3 > 0.05 else f"+{g3:.5f}"
        rows_tex.append(
            f"{scen} & ({r['a1']:.1f}, {r['a2']:.1f}, {r['lambda']:.1f}) & {r['T_months']}m & "
            f"[{num(r['condvar_L'],5,False)}, {num(r['condvar_U'],5,True)}] & {r['condvar_width']:.5f} & "
            f"{r['bdg_mixed_order0']:.5f} & {r['bdg_mixed_order3']:.5f} & {g3_s} & {r['setup_s']:.1f} / {r['solve_s']:.1f} \\\\"
        )
    tex = (
        "\\begin{table}[!htbp]\\centering\\scriptsize\\setlength{\\tabcolsep}{4pt}\n"
        "\\caption{Mixed rough Bergomi VIX futures (volatility points, $H=0.10$, $\\Delta=1/12$, $\\xi_0=0.235^2$): "
        "Arb conditional-variance enclosures versus BDG~(2023) mixed weak expansions.}\\label{tab:vix_mixed}\n"
        "\\begin{tabular}{@{}lcrccrrrr@{}}\\toprule\n"
        "Regime & $(a_1,a_2,\\lambda)$ & $T$ & Arb CondVar $[L, U]$ & Width & BDG$_0$ & BDG$_3$ & $\\mathrm{BDG}_3-U$ & Setup/Solve (s) \\\\\\midrule\n"
        + "\n".join(rows_tex)
        + "\n\\bottomrule\\end{tabular}\n"
        "\\smallskip\\par\\raggedright\\scriptsize\\emph{Notes:} Conditional-variance Arb brackets use the shared "
        "rank-one direction $Z=W(h_0)/\\norm{h_0}$ (Proposition~\\ref{prop:arb}(c)--(d), $M=140$, $N_z=360$). "
        "BDG$_0$, BDG$_3$ follow \\citet[Section~4]{BourgeyDeMarcoGobet2023}. Bold marks $>0.05$ points.\n"
        "\\end{table}\n"
    )
    (TAB_DIR / "tab_vix_mixed.tex").write_text(tex, encoding="utf-8")


def write_tab_smiles(o):
    rows_tex = []
    # Panel 1: BDG Fig 6 (selected 5 representative strikes: -0.4, -0.2, 0.0, +0.2, +0.4)
    rows_tex.append("\\multicolumn{9}{@{}l}{\\emph{Pure rough Bergomi: BDG~(2023) Fig.~6 ($H=0.10$, $T=3$\\,m, $a=1.0$, $\\Delta=1/12$, $\\xi_0=0.235^2$)}} \\\\")
    for r in o["bdg2023_fig6"]["strikes"]:
        if round(r["log_moneyness"], 2) not in (-0.4, -0.2, 0.0, 0.2, 0.4):
            continue
        rows_tex.append(
            f"${r['log_moneyness']:+.2f}$ & {r['K_volpts']:.3f} & [{num(r['call_L_volpts'],5,False)}, {num(r['call_U_volpts'],5,True)}] & "
            f"{r['call_width_volpts']:.5f} & {r['bdg_C3_volpts'][0]:.5f} & {r['ln_call_volpts'][0]:.5f} & "
            f"[{num(r['iv_L_pct'],2,False)}\\%, {num(r['iv_U_pct'],2,True)}\\%] & {r['bdg_iv3_pct'][0]:.2f}\\% & {r['ln_iv_pct'][0]:.2f}\\% \\\\"
        )
    rows_tex.append("\\midrule")
    # Panel 2: Stress cell (all 6 strikes)
    rows_tex.append("\\multicolumn{9}{@{}l}{\\emph{Pure rough Bergomi stress: E14 ($H=0.10$, $T=6$\\,m, $\\eta_{\\mathrm{P1}}=3.0$, $\\Delta=30/365$, $\\xi_0=0.235^2$)}} \\\\")
    for r in o["e14_stress_smile"]["strikes"]:
        rows_tex.append(
            f"${r['log_moneyness']:+.2f}$ & {r['K_volpts']:.3f} & [{num(r['call_L_volpts'],5,False)}, {num(r['call_U_volpts'],5,True)}] & "
            f"{r['call_width_volpts']:.5f} & {r['bdg_C3_volpts'][0]:.5f} & {r['ln_call_volpts'][0]:.5f} & "
            f"[{num(r['iv_L_pct'],2,False)}\\%, {num(r['iv_U_pct'],2,True)}\\%] & {r['bdg_iv3_pct'][0]:.2f}\\% & {r['ln_iv_pct'][0]:.2f}\\% \\\\"
        )
    rows_tex.append("\\midrule")
    # Panel 3: Mixed rough Bergomi (5 strikes: -0.3, -0.2, 0.0, +0.1, +0.3)
    rows_tex.append("\\multicolumn{9}{@{}l}{\\emph{Mixed rough Bergomi: Scenario~1 ($H=0.10$, $T=3$\\,m, $a_1=1.4$, $a_2=0.7$, $\\lambda=0.3$, $\\Delta=1/12$)}} \\\\")
    split = json.loads((RES/"mixed_split.json").read_text())["rows"]
    for r, corrected in zip(o["mixed_rbergomi_smile"]["strikes"], split, strict=True):
        if round(r["log_moneyness"], 2) not in (-0.3, -0.2, 0.0, 0.1, 0.3):
            continue
        rows_tex.append(
            f"${r['log_moneyness']:+.2f}$ & {r['K_volpts']:.3f} & [{num(r['call_L_volpts'],5,False)}, {num(r['call_U_volpts'],5,True)}] & "
            f"{r['call_width_volpts']:.5f} & {np.mean(corrected['split_call']):.5f} & --- & "
            f"[{num(r['iv_L_pct'],2,False)}\\%, {num(r['iv_U_pct'],2,True)}\\%] & {np.mean(corrected['split_iv']):.2f}\\% & --- \\\\"
        )

    tex = (
        "\\begin{table}[!htbp]\\centering\\scriptsize\\setlength{\\tabcolsep}{2.6pt}\n"
        "\\caption{Arb-certified VIX call prices (volatility points) and Black-76 implied volatilities versus "
        "BDG~(2023) order-three and lognormal proxies across three smile regimes.}\\label{tab:vix_smiles}\n"
        "\\resizebox{\\textwidth}{!}{\\begin{tabular}{@{}rrccrrccr@{}}\\toprule\n"
        "$k$ & $K$ & Arb Call $[C_L, C_U]$ & Width & BDG$_3$ Call & LN Call & Arb IV $[\\sigma_L, \\sigma_U]$ & BDG$_3$ IV & LN IV \\\\\\midrule\n"
        + "\n".join(rows_tex)
        + "\n\\bottomrule\\end{tabular}}\n"
        "\\smallskip\\par\\raggedright\\scriptsize\\emph{Notes:} $k=\\log(K/F_{\\mathrm{mid}})$. Every Arb IV "
        "bracket propagates both call-price and forward-price uncertainty and is independently verified in 160-bit "
        "Arb (Proposition~\\ref{prop:arb}(e)). BDG$_3$ IV uses the expansion forward. Mixed entries show root-split midpoints.\n"
        "\\end{table}\n"
    )
    (TAB_DIR / "tab_vix_smiles.tex").write_text(tex, encoding="utf-8")


def write_tab_bcj(f):
    rows_tex = []
    cmap_lbl = {"flat": "$0.234^2$", "quad": "$0.234^2(1+t)^2$", "sqrt": "$0.234^2\\sqrt{1+t}$"}
    cur = None
    for r in f["block_c_bcj2023_18cells"]:
        if r["curve"] != cur:
            if cur is not None:
                rows_tex.append("\\midrule")
            cur = r["curve"]
        rows_tex.append(
            f"{cmap_lbl[r['curve']]} & {r['T_months']}m & [{num(r['condvar_L'],5,False)}, {num(r['condvar_U'],5,True)}] & "
            f"{r['condvar_width']:.5f} & {r['jmm_width']:.3f} & {r['ratio_vs_jmm']:,.0f}$\\times$ & "
            f"{r['setup_s']:.1f} & {r['solve_s']:.1f} \\\\"
        )
    tex = (
        "\\begin{table}[!htbp]\\centering\\scriptsize\\setlength{\\tabcolsep}{4.5pt}\n"
        "\\caption{Complete 18-cell Bonesini--Callegaro--Jacquier (2023, Section~5.3) Arb VIX futures suite "
        "across three forward-variance curves and six maturities ($H=0.10$, $\\eta_{\\mathrm{P1}}=1.9$, $\\Delta=1/12$).}\\label{tab:bcj_full}\n"
        "\\begin{tabular}{@{}lrcrrrrr@{}}\\toprule\n"
        "Curve $\\xi_0(t)$ & $T$ & Arb CondVar $[L, U]$ & Width & JMM width & Ratio & Setup (s) & Solve (s) \\\\\\midrule\n"
        + "\n".join(rows_tex)
        + "\n\\bottomrule\\end{tabular}\n"
        "\\smallskip\\par\\raggedright\\scriptsize\\emph{Notes:} All 18 continuous-time enclosures use "
        "Proposition~4(c)--(d) of the main manuscript ($M=120$, $N_z=310$) and achieve widths at least $4\\times$ below "
        "one Cboe tick ($0.05$ points). Setup is paid once per $(H,T,\\Delta)$ cell.\n"
        "\\end{table}\n"
    )
    (TAB_DIR / "tab_bcj_full.tex").write_text(tex, encoding="utf-8")


def make_fig_phase_boundary(f):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.65), constrained_layout=True)

    # Panel (a): BDG Fig. 1 a-sweep
    r1 = f["block_b_bdg_sweeps"]["fig1_a_sweep"]
    a_vals = np.array([r["a"] for r in r1])
    bdg_gap = np.array([r["bdg3_minus_U"] for r in r1])
    ln_gap = np.array([r["L_minus_LN"] for r in r1])
    arb_w = np.array([r["tilted_width"] for r in r1])

    ax1.axhspan(0.05, 1.0, color=C_VERM, alpha=0.07, zorder=0)
    ax1.axhline(0.05, color=C_DARK, linestyle="--", linewidth=0.9, label="1 Cboe tick (0.05 pts)")
    ax1.semilogy(a_vals, ln_gap, "s--", color=C_VERM, markersize=4.2, label=r"Lognormal shortfall ($L - \mathrm{LN}$)")
    ax1.semilogy(a_vals, bdg_gap, "o-", color=C_BLUE, markersize=4.5, label=r"BDG (2023) excess ($\mathrm{BDG}_3 - U$)")
    ax1.semilogy(a_vals, arb_w, "^:", color=C_TEAL, markersize=4.2, label=r"Arb bracket width ($U - L$)")
    ax1.annotate("Arb-certified $>1$ tick\nBDG violation", xy=(1.35, 0.062), xytext=(0.24, 0.20),
                 fontsize=7.1, color=C_BLUE, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=0.85, connectionstyle="arc3,rad=-0.12"))
    ax1.set_xlabel(r"Vol-of-vol $a = \eta_{\mathrm{P1}}\sqrt{2H}$")
    ax1.set_ylabel("Volatility points (log scale)")
    ax1.set_title(r"(a) Vol-of-vol sweep ($H=0.10,\;T=6\,$m)")
    ax1.set_ylim(1e-5, 0.9)
    ax1.grid(True, which="both")
    ax1.legend(loc="lower right", frameon=True, framealpha=0.92)

    # Panel (b): BDG Fig. 2 H-sweep
    r2 = f["block_b_bdg_sweeps"]["fig2_H_sweep"]
    H_vals = np.array([r["H"] for r in r2])
    bdg_h = np.array([r["bdg3_minus_U"] for r in r2])
    ln_h = np.array([r["L_minus_LN"] for r in r2])
    arb_h = np.array([r["tilted_width"] for r in r2])

    ax2.axhspan(0.05, 1.0, color=C_VERM, alpha=0.07, zorder=0)
    ax2.axhline(0.05, color=C_DARK, linestyle="--", linewidth=0.9)
    ax2.semilogy(H_vals, ln_h, "s--", color=C_VERM, markersize=4.2, label=r"$L - \mathrm{LN}$")
    ax2.semilogy(H_vals, bdg_h, "o-", color=C_BLUE, markersize=4.5, label=r"$\mathrm{BDG}_3 - U$")
    ax2.semilogy(H_vals, arb_h, "^:", color=C_TEAL, markersize=4.2, label=r"Arb width $U - L$")
    ax2.set_xlabel(r"Hurst index $H$")
    ax2.set_title(r"(b) Roughness sweep ($a=1.0,\;T=6\,$m)")
    ax2.set_ylim(1e-5, 0.9)
    ax2.grid(True, which="both")
    ax2.legend(loc="lower left", frameon=True, framealpha=0.92)

    fig.savefig(FIG_DIR / "fig_vix_phase_boundary.pdf")
    plt.close(fig)


def make_fig_smiles(o):
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.45), constrained_layout=True)

    # (a) BDG Fig 6
    ax = axes[0]
    s = o["bdg2023_fig6"]["strikes"]
    ks = [r["log_moneyness"] for r in s]
    iv_l = [r["iv_L_pct"] for r in s]
    iv_u = [r["iv_U_pct"] for r in s]
    bdg_iv = [r["bdg_iv3_pct"][0] for r in s]
    ln_iv = [r["ln_iv_pct"][0] for r in s]
    ax.fill_between(ks, iv_l, iv_u, color=C_TEAL, alpha=0.35, label="Arb IV band")
    ax.plot(ks, iv_l, color=C_TEAL, lw=0.9)
    ax.plot(ks, iv_u, color=C_TEAL, lw=0.9)
    ax.plot(ks, bdg_iv, "o-", color=C_BLUE, ms=3.4, lw=1.1, label=r"BDG$_3$ IV")
    ax.plot(ks, ln_iv, "--", color=C_VERM, lw=1.1, label="Lognormal IV")
    ax.set_title(r"(a) Pure rBergomi ($a{=}1$, $3\,$m)")
    ax.set_xlabel(r"Log-moneyness $k$")
    ax.set_ylabel("Black-76 IV (%)")
    ax.grid(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.87), frameon=True, framealpha=0.94, fontsize=6.8)

    # (b) Stress smile
    ax = axes[1]
    s = o["e14_stress_smile"]["strikes"]
    ks = [r["log_moneyness"] for r in s]
    iv_l = [r["iv_L_pct"] for r in s]
    iv_u = [r["iv_U_pct"] for r in s]
    bdg_iv = [r["bdg_iv3_pct"][0] for r in s]
    ln_iv = [r["ln_iv_pct"][0] for r in s]
    ax.fill_between(ks, iv_l, iv_u, color=C_TEAL, alpha=0.35, label="Arb IV band")
    ax.plot(ks, iv_l, color=C_TEAL, lw=0.9)
    ax.plot(ks, iv_u, color=C_TEAL, lw=0.9)
    ax.plot(ks, bdg_iv, "o-", color=C_BLUE, ms=3.4, lw=1.1, label=r"BDG$_3$ IV")
    ax.plot(ks, ln_iv, "--", color=C_VERM, lw=1.1, label="Lognormal IV")
    ax.set_title(r"(b) Stress ($\eta_{\mathrm{P1}}{=}3$, $6\,$m)")
    ax.set_xlabel(r"Log-moneyness $k$")
    ax.grid(True)

    # (c) Mixed rough Bergomi smile
    ax = axes[2]
    s = o["mixed_rbergomi_smile"]["strikes"]
    ks = [r["log_moneyness"] for r in s]
    iv_l = [r["iv_L_pct"] for r in s]
    iv_u = [r["iv_U_pct"] for r in s]
    split = json.loads((RES/"mixed_split.json").read_text())["rows"]
    bdg_iv = [np.mean(r["split_iv"]) for r in split]
    ax.fill_between(ks, iv_l, iv_u, color=C_TEAL, alpha=0.40, label="Arb IV band")
    ax.plot(ks, iv_l, color=C_TEAL, lw=0.9)
    ax.plot(ks, iv_u, color=C_TEAL, lw=0.9)
    ax.plot(ks, bdg_iv, "s--", color=C_BLUE, ms=3.5, lw=1.1, label="Root split")
    ax.plot(ks, [r["code_iv"] for r in split], "x:", color=C_VERM, ms=3, lw=0.9, label="100-node code")
    ax.set_title(r"(c) Mixed rBergomi ($3\,$m)")
    ax.set_xlabel(r"Log-moneyness $k$")
    ax.grid(True)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=6.8)

    fig.savefig(FIG_DIR / "fig_vix_smiles.pdf")
    plt.close(fig)


def make_fig_bcj_and_disjoint(g, f):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.55), constrained_layout=True)

    # Panel (a): BD (2022) Continuum vs 500-cell right-rectangle disjoint Arb audit
    bd = {c["id"]: c for c in g["gate2"]["cells"]}["BD2022_MLMC_JCF"]
    cL, cU = bd["tilted_L_outer"], bd["tilted_U_outer"]
    gL, gU = bd["grid500_tilted_arb"]["L_outer_volpts"], bd["grid500_tilted_arb"]["U_outer_volpts"]
    pub = bd["published_fut_volpts"]
    bdg3 = bd["bdg_F3_arb"][0]

    ax1.barh(2.0, cU - cL, left=cL, height=0.38, color=C_TEAL, alpha=0.85, edgecolor=C_DARK, lw=0.7, label="Continuum Arb")
    ax1.barh(1.0, gU - gL, left=gL, height=0.38, color=C_GOLD, alpha=0.85, edgecolor=C_DARK, lw=0.7, label="500-cell grid Arb")
    ax1.plot(pub, 1.0, "D", color=C_VERM, ms=5.5, zorder=5, label="BD (2022) MLMC (22.18135)")
    ax1.axvline(pub, color=C_VERM, ls=":", lw=0.9, alpha=0.7)
    ax1.plot(bdg3, 2.0, "s", color=C_BLUE, ms=5.0, zorder=5, label="BDG (2023) order-3 (22.18192)")
    ax1.axvspan(cU, gL, color="#888888", alpha=0.18)
    ax1.annotate("Certified disjoint\ngap $\\geq 6.5\\times10^{-4}$", xy=(0.5 * (cU + gL), 1.50), xytext=(22.18025, 0.22),
                 fontsize=6.9, ha="center", arrowprops=dict(arrowstyle="->", color=C_DARK, lw=0.75))
    ax1.set_yticks([1.0, 2.0])
    ax1.set_yticklabels(["500-cell\nright-rect", "Continuous\nmodel"])
    ax1.set_ylim(-0.15, 3.35)
    ax1.set_xlim(22.1797, 22.1822)
    ax1.set_xlabel("VIX futures price (volatility points)")
    ax1.set_title("(a) BD (2022) MLMC continuum vs. grid audit")
    ax1.grid(True, axis="x")
    ax1.legend(loc="upper center", ncol=2, frameon=True, framealpha=0.94, fontsize=6.0, columnspacing=0.8, handletextpad=0.4)

    # Panel (b): BCJ (2023) Sec 5.3 3-curve term structures
    bcj = f["block_c_bcj2023_18cells"]
    styles = {
        "flat": (C_TEAL, "o-", r"Flat $\xi_0=0.234^2$"),
        "sqrt": (C_BLUE, "s--", r"$\xi_0(t)=0.234^2\sqrt{1+t}$"),
        "quad": (C_VERM, "^-.", r"$\xi_0(t)=0.234^2(1+t)^2$"),
    }
    for cname in ("quad", "sqrt", "flat"):
        sub = sorted([r for r in bcj if r["curve"] == cname], key=lambda x: x["T_months"])
        Ts = [r["T_months"] for r in sub]
        mids = [0.5 * (r["condvar_L"] + r["condvar_U"]) for r in sub]
        ws = [r["condvar_width"] for r in sub]
        col, sty, lbl = styles[cname]
        ax2.plot(Ts, mids, sty, color=col, ms=4.0, label=f"{lbl} (max $w={max(ws):.4f}$)")
    ax2.set_xlabel("Maturity $T$ (months)")
    ax2.set_ylabel("Arb VIX future (volatility points)")
    ax2.set_xticks([1, 2, 3, 6, 9, 12])
    ax2.set_ylim(17.0, 44.5)
    ax2.set_title(r"(b) BCJ (2023) Sec. 5.3 term structures ($H=0.10$)")
    ax2.grid(True)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.94, fontsize=6.5)

    fig.savefig(FIG_DIR / "fig_vix_bcj_disjoint.pdf")
    plt.close(fig)


def main():
    g, f, o = load_jsons()
    write_macros(g, f, o)
    write_tab_headline(g)
    write_tab_e14(f)
    write_tab_mixed(f)
    write_tab_smiles(o)
    write_tab_bcj(f)
    make_fig_phase_boundary(f)
    make_fig_smiles(o)
    make_fig_bcj_and_disjoint(g, f)
    print("Generated all LaTeX macros, 5 tables, and 3 publication-quality PDF figures successfully!")


if __name__ == "__main__":
    main()

