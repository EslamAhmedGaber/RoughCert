"""Build ALL 9 publication figures (6 main-text + 3 supplement) from scratch for RoughCert v3.

Every figure is generated deterministically from verified JSON datasets under one
unified Quantitative Finance visual system (v3_style.py) at exact print width
(\\textwidth = 6.50 in) so typography prints at exact 8-9 pt with zero rescaling.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, FormatStrFormatter

from v3_style import (
    TEXTWIDTH_IN,
    PALETTE,
    H_COLORS,
    CURVE_COLORS,
    apply_style,
    style_axes,
    panel_title,
    save_figure,
)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
LEG = RES / "legacy_inputs"
OUT_DIR = ROOT / "manuscript" / "figures"


from v3_comparisons import scoreboard, factor_paths, absolute, difference
from build_v3_tables import fmt_out


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# =============================================================================
# FIGURE 1: RoughCert Pipeline Schematic + Quantile Bias Rates (P^1/3 vs P^1/2)
# =============================================================================
def build_fig1():
    fig, (ax_pipe, ax_rate) = plt.subplots(
        1, 2,
        figsize=(TEXTWIDTH_IN, 2.90),
        gridspec_kw={"width_ratios": [1.15, 0.98], "wspace": 0.36},
    )

    # --- Panel (a): Clean 3-stage RoughCert architecture schematic ---
    ax_pipe.set_axis_off()
    ax_pipe.set_xlim(0.0, 1.0)
    ax_pipe.set_ylim(0.0, 1.0)
    panel_title(ax_pipe, "a", "RoughCert certification pipeline")

    def box(x, y, w, h, title, body, fc="#FFFFFF", ec=PALETTE["ink"], title_c=PALETTE["ink"], lw=0.95):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            fc=fc, ec=ec, lw=lw, zorder=2,
        )
        ax_pipe.add_patch(patch)
        ax_pipe.text(
            x + w / 2, y + h * 0.72, title,
            ha="center", va="center", fontsize=6.25, fontweight="bold", color=title_c, zorder=3,
        )
        ax_pipe.text(
            x + w / 2, y + h * 0.29, body,
            ha="center", va="center", fontsize=5.35, color=PALETTE["ink"], linespacing=1.24, zorder=3,
        )

    def arrow(x0, y0, x1, y1, color=PALETTE["roughcert"]):
        ax_pipe.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-|>", mutation_scale=8.0, lw=0.9, color=color, zorder=4,
            shrinkA=1.5, shrinkB=1.5,
        ))

    # Top input box
    box(
        0.02, 0.76, 0.96, 0.20,
        "1. Continuous Model & Mehler Ledger",
        "Single or mixed rBergomi, curved $\\xi_0(t)$\nCompute residual share $P/\\mathrm{Var}(\\Lambda)$",
        fc="#F8FAFC", ec=PALETTE["ink"], title_c=PALETTE["ink"],
    )

    # Middle dual branches: RoughCert-Tilt vs RoughCert-CV
    box(
        0.02, 0.40, 0.45, 0.25,
        "2a. RoughCert-Tilt",
        "Static Girsanov–Hölder\nNo $Z$-quadrature; $O(\\varepsilon^4)$",
        fc=PALETTE["roughcert_light"], ec=PALETTE["roughcert"], title_c=PALETTE["roughcert"], lw=1.1,
    )
    box(
        0.53, 0.40, 0.45, 0.25,
        "2b. RoughCert-CV",
        "Cond. var. + Minkowski $L^2$\nFutures, options and IV",
        fc=PALETTE["roughcert_light"], ec=PALETTE["roughcert"], title_c=PALETTE["roughcert"], lw=1.1,
    )

    # Bottom Arb engine & certified verdict
    box(
        0.02, 0.03, 0.96, 0.26,
        "3. 169-bit Arb Ball Engine & Verdict",
        "Pfaff ${}_2F_1$ products + convex panels + tail $<2\\times10^{-10}$\n"
        "$\\Rightarrow$ Verified $[L,U]$, IV band & $\\leq 1$ tick ($0.05$ pts) test",
        fc="#F0FDFA", ec=PALETTE["roughcert"], title_c=PALETTE["roughcert"], lw=1.2,
    )

    # Arrows
    arrow(0.245, 0.75, 0.245, 0.665)
    arrow(0.755, 0.75, 0.755, 0.665)
    arrow(0.245, 0.39, 0.245, 0.305)
    arrow(0.755, 0.39, 0.755, 0.305)

    # Compact branch cues
    ax_pipe.text(0.22, 0.708, "flat futures", ha="right", va="center", fontsize=5.8, color=PALETTE["muted"], style="italic")
    ax_pipe.text(0.78, 0.708, "kinks / curves", ha="left", va="center", fontsize=5.8, color=PALETTE["muted"], style="italic")

    # --- Panel (b): Quantile Bias Rate Verification (P^1/3 vs P^1/2) ---
    rates = load_json(LEG / "E5c_certified_cantelli" / "e5c_rates.json")
    style_axes(ax_rate, grid_axis="both")
    panel_title(ax_rate, "b", r"Quantile bias rate ($P^{1/3}$ vs. $P^{1/2}$)")

    for H_val, c_mart, c_cv, mk_m, mk_c in [
        (0.1, PALETTE["bcj"], PALETTE["roughcert"], "s", "o"),
        (0.3, PALETTE["markov"], PALETTE["bdg"], "^", "D"),
    ]:
        rows = sorted([r for r in rates if abs(r["H"] - H_val) < 1e-6], key=lambda x: x["P"])
        P = np.array([r["P"] for r in rows])
        Bc = np.array([r["B_c"] for r in rows])
        Bd = np.array([r["B_d"] for r in rows])
        pc = np.array([r["pred_c"] for r in rows])
        pd = np.array([r["pred_d"] for r in rows])

        sc = np.polyfit(np.log(P), np.log(Bc), 1)[0]
        sd = np.polyfit(np.log(P), np.log(Bd), 1)[0]

        ax_rate.loglog(P, Bc, "-", color=c_mart, marker=mk_m, ms=3.7, lw=1.25,
                       label=f"Martingale $H={H_val:g}$ ({sc:.2f})")
        ax_rate.loglog(P, Bd, "-", color=c_cv, marker=mk_c, ms=3.7, lw=1.35,
                       label=f"CondVar $H={H_val:g}$ ({sd:.2f})")
        ax_rate.loglog(P, pc, ":", color=c_mart, lw=0.85, alpha=0.75)
        ax_rate.loglog(P, pd, "--", color=c_cv, lw=0.85, alpha=0.75)

    ax_rate.text(1.9e-4, 0.19, r"$\propto P^{1/3}$", fontsize=7.3, color=PALETTE["markov"], fontweight="bold")
    ax_rate.text(2.4e-4, 0.024, r"$\propto P^{1/2}$", fontsize=7.3, color=PALETTE["bdg"], fontweight="bold")

    ax_rate.set_xlabel(r"Residual variance $P = \mathbb{E}[V]$")
    ax_rate.set_ylabel(r"Upper shift $U - Q_{\hat\Lambda}(0.95)$")
    ax_rate.set_xlim(1.3e-4, 1.8e-1)
    ax_rate.set_ylim(1.9e-2, 3.8)
    ax_rate.legend(
        loc="upper left", fontsize=6.3, frameon=True, facecolor="white",
        edgecolor="none", framealpha=0.90, handlelength=1.35, labelspacing=0.22,
    )

    return save_figure(fig, OUT_DIR, "fig_v3_01_pipeline_and_rates")


# =============================================================================
# FIGURE 2: Phase Boundaries — BDG Figs. 1 & 2 Parameter Sweeps
# =============================================================================
def build_fig2():
    fut = load_json(RES / "vix_futures_full_suite.json")
    a_rows = fut["block_b_bdg_sweeps"]["fig1_a_sweep"]
    h_rows = fut["block_b_bdg_sweeps"]["fig2_H_sweep"]

    fig, (ax_a, ax_h) = plt.subplots(
        1, 2, figsize=(TEXTWIDTH_IN, 2.75), sharey=True,
        gridspec_kw={"wspace": 0.14},
    )

    # --- Panel (a): a-sweep ---
    style_axes(ax_a, grid_axis="y")
    panel_title(ax_a, "a", r"Vol-of-vol sweep ($H=0.10$, $T=6$ m)")

    a = np.array([r["a"] for r in a_rows])
    w_a = np.array([r["tilted_width"] for r in a_rows])
    bdg_a = np.array([r["bdg3_minus_U"] for r in a_rows])
    ln_a = np.array([r["L_minus_LN"] for r in a_rows])

    ax_a.axhspan(0.05, 1.6, color=PALETTE["tick_fill"], alpha=0.52, zorder=0)
    ax_a.axhline(0.05, color=PALETTE["tick"], ls="--", lw=0.95, zorder=1)
    ax_a.text(0.19, 0.058, "1 tick ($0.05$)", fontsize=6.8, color=PALETTE["tick"], fontweight="bold")

    ax_a.semilogy(a, ln_a, ":", color=PALETTE["ln"], marker="D", ms=3.9, lw=1.25, label=r"Lognormal shortfall ($L - F_{\mathrm{LN}}$)")
    ax_a.semilogy(a, bdg_a, "-", color=PALETTE["bdg"], marker="o", ms=4.1, lw=1.35, label=r"BDG (2023) excess ($F_{\mathrm{BDG}_3} - U$)")
    ax_a.semilogy(a, w_a, "-", color=PALETTE["roughcert"], marker="s", ms=3.9, lw=1.6, label=r"RoughCert-Tilt width ($U - L$)")
    ax_a.fill_between(a, 1e-6, w_a, color=PALETTE["roughcert_fill"], alpha=0.12, zorder=1)

    a_ref = np.linspace(0.35, 1.15, 30)
    c4 = (w_a[2] * 0.42) / (a[2] ** 4)
    ax_a.semilogy(a_ref, c4 * a_ref ** 4, ":", color=PALETTE["muted"], lw=0.8)
    ax_a.text(0.75, c4 * (0.75 ** 4) * 0.48, r"$\propto a^4$", fontsize=7.2, color=PALETTE["muted"])

    first = np.flatnonzero(ln_a > 0.05)[0]
    ax_a.annotate(f"LN $>1$ tick\nat $a={a[first]:g}$",
                  xy=(a[first],ln_a[first]),xytext=(0.25,0.22),fontsize=6.2,
                  arrowprops=dict(arrowstyle="-|>",color=PALETTE["ln"],lw=.7))
    ax_a.annotate(
        r"BDG$_3$: $+0.081$ pts" + "\n(1.62 ticks)",
        xy=(1.50, bdg_a[-1]), xytext=(0.87, 0.44),
        fontsize=6.6, color=PALETTE["bdg"], fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["bdg"], lw=0.75, shrinkA=1, shrinkB=3.0),
    )

    ax_a.set_xlabel(r"Kernel amplitude $a = \eta_{\mathrm{P1}}\sqrt{2H}$")
    ax_a.set_ylabel("Certified separation or width (vol. pts)")
    ax_a.set_xlim(0.16, 1.57)
    ax_a.set_ylim(min(min(w_a),min(r["tilted_width"] for r in h_rows)) * 0.65, 1.4)

    # --- Panel (b): H-sweep ---
    style_axes(ax_h, grid_axis="y")
    panel_title(ax_h, "b", r"Roughness sweep ($a=1.0$, $T=6$ m)")

    H = np.array([r["H"] for r in h_rows])
    w_h = np.array([r["tilted_width"] for r in h_rows])
    bdg_h = np.array([r["bdg3_minus_U"] for r in h_rows])
    ln_h = np.array([r["L_minus_LN"] for r in h_rows])

    ax_h.axhspan(0.05, 1.6, color=PALETTE["tick_fill"], alpha=0.52, zorder=0)
    ax_h.axhline(0.05, color=PALETTE["tick"], ls="--", lw=0.95, zorder=1)

    ax_h.semilogy(H, ln_h, ":", color=PALETTE["ln"], marker="D", ms=3.9, lw=1.25)
    ax_h.semilogy(H, bdg_h, "-", color=PALETTE["bdg"], marker="o", ms=4.1, lw=1.35)
    ax_h.semilogy(H, w_h, "-", color=PALETTE["roughcert"], marker="s", ms=3.9, lw=1.6)
    ax_h.fill_between(H, 1e-6, w_h, color=PALETTE["roughcert_fill"], alpha=0.12, zorder=1)

    ax_h.annotate(
        r"BDG$_3$: $+0.066$ pts" + "\n(1.32 ticks at $H=0.05$)",
        xy=(0.05, bdg_h[0]), xytext=(0.145, 0.34),
        fontsize=6.6, color=PALETTE["bdg"], fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["bdg"], lw=0.75, shrinkA=1, shrinkB=3.0),
    )
    ax_h.annotate(
        "RoughCert-Tilt:\n" + r"$0.025$ pts ($\frac{1}{2}$ tick)",
        xy=(0.05, w_h[0]), xytext=(0.065, 0.00018),
        fontsize=6.6, color=PALETTE["roughcert"], fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["roughcert"], lw=0.75, shrinkA=1, shrinkB=3.0),
    )

    ax_h.set_xlabel(r"Hurst parameter $H$")
    ax_h.set_xlim(0.03, 0.42)
    ax_h.set_xticks([0.05, 0.10, 0.15, 0.20, 0.30, 0.40])

    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.52, 1.07),
        ncol=3, frameon=False, fontsize=7.4, columnspacing=1.2,
    )
    fig.subplots_adjust(top=0.86)
    return save_figure(fig, OUT_DIR, "fig_v3_02_phase_boundaries")


# =============================================================================
# FIGURE 3: BD (2022) Disjoint Audit + BCJ (2023) Multi-Curve Term Structures
# =============================================================================
def build_fig3():
    w1 = load_json(RES / "week1_gates_results.json")["gate2"]["cells"]
    cont_c = next(c for c in w1 if c["id"] == "BD2022_MLMC_JCF")
    grid_c = cont_c["grid500_tilted_arb"]
    fut = load_json(RES / "vix_futures_full_suite.json")
    bcj18 = fut["block_c_bcj2023_18cells"]
    q_suite = load_json(RES / "quantize_suite.json")

    fig, (ax_bd, ax_bcj) = plt.subplots(
        1, 2, figsize=(TEXTWIDTH_IN, 2.75),
        gridspec_kw={"width_ratios": [1.02, 1.08], "wspace": 0.34},
    )

    # --- Panel (a): Disjoint horizontal dot-and-bar chart (BD 2022 §3.3) ---
    style_axes(ax_bd, grid_axis="x")
    ax_bd.spines["left"].set_visible(False)
    ax_bd.tick_params(axis="y", length=0)
    panel_title(ax_bd, "a", "BD (2022) §3.3: disjoint target audit")

    cL, cU = cont_c["tilted_L_outer"], cont_c["tilted_U_outer"]
    gL, gU = grid_c["L_outer_volpts"], grid_c["U_outer_volpts"]
    mlmc = 22.18135
    bdg3 = 0.5 * (cont_c["bdg_F3_arb"][0] + cont_c["bdg_F3_arb"][1])

    ax_bd.axvspan(cU, gL, color=PALETTE["gold_fill"], alpha=0.85, zorder=0)
    ax_bd.axvline(cU, color=PALETTE["roughcert"], ls=":", lw=0.8, zorder=1)
    ax_bd.axvline(gL, color=PALETTE["markov"], ls=":", lw=0.8, zorder=1)
    ax_bd.text(
        0.5 * (cU + gL), 3.55,
        f"Certified disjoint gap\n$\\geq {fmt_out(gL-cU,5,False)}$ pts",
        ha="center", va="center", fontsize=6.5, color="#92400E", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.22", fc="#FFFBEB", ec="#F59E0B", lw=0.6),
    )

    y_rows = [2.6, 1.75, 0.95, 0.15]
    ax_bd.hlines(y_rows[0], cL, cU, color=PALETTE["roughcert"], lw=5.5, alpha=0.35, zorder=2)
    ax_bd.plot([cL, cU], [y_rows[0], y_rows[0]], "|", color=PALETTE["roughcert"], ms=8, mew=1.5, zorder=3)
    ax_bd.plot(0.5 * (cL + cU), y_rows[0], "o", color=PALETTE["roughcert"], ms=4.2, zorder=4)
    ax_bd.text(cL - 0.00004, y_rows[0] - 0.30, f"[{fmt_out(cL,5,False)}, {fmt_out(cU,5,True)}]", fontsize=6.2, color=PALETTE["roughcert"], ha="left")

    ax_bd.hlines(y_rows[1], gL, gU, color=PALETTE["markov"], lw=5.5, alpha=0.35, zorder=2)
    ax_bd.plot([gL, gU], [y_rows[1], y_rows[1]], "|", color=PALETTE["markov"], ms=8, mew=1.5, zorder=3)
    ax_bd.plot(0.5 * (gL + gU), y_rows[1], "s", color=PALETTE["markov"], ms=4.0, zorder=4)
    ax_bd.text(gU + 0.00004, y_rows[1] - 0.30, f"[{fmt_out(gL,5,False)}, {fmt_out(gU,5,True)}]", fontsize=6.2, color="#B45309", ha="right")

    ax_bd.axvline(mlmc, ymin=0.18, ymax=0.52, color=PALETTE["ink"], ls="--", lw=0.65, alpha=0.5, zorder=1)
    ax_bd.plot(mlmc, y_rows[2], "D", color=PALETTE["ink"], ms=4.8, zorder=4)
    ax_bd.text(mlmc - 0.00008, y_rows[2], "22.18135 (in 500-cell)", fontsize=6.3, color=PALETTE["ink"], ha="right", va="center")

    ax_bd.plot(bdg3, y_rows[3], "^", color=PALETTE["bdg"], ms=5.0, zorder=4)
    ax_bd.text(bdg3 - 0.00008, y_rows[3], f"{bdg3:.5f} (outside both)", fontsize=6.3, color=PALETTE["bdg"], ha="right", va="center")

    ax_bd.set_yticks(y_rows)
    ax_bd.set_yticklabels([
        "RoughCert\nContinuum",
        "RoughCert\n500-cell grid",
        "Published MLMC\n(BD 2022)",
        "BDG (2023)\norder-3",
    ], fontsize=6.9)
    ax_bd.set_xlim(22.17975, 22.18212)
    ax_bd.set_ylim(-0.32, 4.05)
    ax_bd.set_xticks([22.1800, 22.1807, 22.1814, 22.1821])
    ax_bd.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    ax_bd.set_xlabel("VIX futures price (volatility points)")

    # --- Panel (b): BCJ (2023) 3 forward-variance curves + JMM + 960-path Quantiser ---
    style_axes(ax_bcj, grid_axis="y")
    panel_title(ax_bcj, "b", "BCJ (2023) §5.3: multi-curve futures")

    curve_specs = [
        ("quad", r"$\xi_0(t)=0.234^2(1+t)^2$", CURVE_COLORS["quad"], "o"),
        ("sqrt", r"$\xi_0(t)=0.234^2\sqrt{1+t}$", CURVE_COLORS["sqrt"], "s"),
        ("flat", r"$\xi_0(t)=0.234^2$ (flat)", CURVE_COLORS["flat"], "^"),
    ]
    for c_key, label, color, mk in curve_specs:
        rows = sorted([r for r in bcj18 if c_key in r["curve"]], key=lambda x: x["T_months"])
        T_m = np.array([r["T_months"] for r in rows])
        mid = np.array([0.5 * (r["condvar_L"] + r["condvar_U"]) for r in rows])
        j_lo = np.array([r["jmm_lo"][0] for r in rows])
        j_hi = np.array([r["jmm_up"][1] for r in rows])

        if c_key == "quad":
            ax_bcj.fill_between(T_m, j_lo, j_hi, color=PALETTE["jmm"], alpha=0.18, lw=0, label="JMM (2018) band (quad.)")

        ax_bcj.plot(T_m, mid, "-", color=color, marker=mk, ms=3.9, lw=1.55, label=label)

        q_rows = sorted([r for r in q_suite if r["curve"] == c_key and r["budget"] == 1000], key=lambda x: x["months"])
        if q_rows:
            Tq = np.array([r["months"] for r in q_rows])
            Fq = np.array([0.5 * (r["future"][0] + r["future"][1]) for r in q_rows])
            ax_bcj.plot(Tq, Fq, "--", color=color, marker=mk, ms=3.2, mfc="white", mew=0.9, lw=0.95, alpha=0.88)

    # Add dummy handle for BCJ hollow dashed style right inside the legend
    ax_bcj.plot([], [], "--", color=PALETTE["ink"], marker="o", mfc="white", ms=3.2, lw=0.95,
                label="BCJ 960-path quantiser (hollow)")

    # Crisp badge placed in the wide-open mid-right channel (y=26..30, x=6..12)
    ax_bcj.text(
        12.3, 27.6,
        "RoughCert-CV (solid):\nmax width $0.013$ pts\n($483$–$817\\times$ tighter)",
        ha="right", va="center", fontsize=6.1, color=PALETTE["roughcert"], fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="#F0FDFA", ec=PALETTE["roughcert"], lw=0.6),
    )

    ax_bcj.set_xlabel("Maturity $T$ (months)")
    ax_bcj.set_ylabel("VIX futures price (volatility points)")
    ax_bcj.set_xticks([1, 2, 3, 6, 9, 12])
    ax_bcj.set_xlim(0.5, 12.6)
    ax_bcj.set_ylim(16.6, 51.0)
    ax_bcj.legend(loc="upper left", fontsize=6.15, handlelength=1.4, labelspacing=0.20,
                  frameon=True, facecolor="white", edgecolor="none", framealpha=0.85)

    return save_figure(fig, OUT_DIR, "fig_v3_03_disjoint_and_curves")


# =============================================================================
# FIGURE 4: VIX Black-76 Implied-Volatility Smiles (3 Regimes)
# =============================================================================
def build_fig4():
    opt = load_json(RES / "vix_options_smile_suite.json")
    ref = load_json(RES / "refinement.json")
    lab = load_json(RES / "lab_suite.json")
    mix = load_json(RES / "mixed_split.json")

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(TEXTWIDTH_IN, 2.60),
        gridspec_kw={"wspace": 0.34},
    )

    # --- Panel (a): Pure rBergomi (BDG Fig. 6, T=3m, a=1.0) ---
    style_axes(ax1, grid_axis="y")
    panel_title(ax1, "a", r"Pure ($a=1.0$, 3 m)")
    st1 = opt["bdg2023_fig6"]["strikes"]
    k1 = np.array([s["log_moneyness"] for s in st1])
    ivL1 = np.array([s["iv_L_pct"] for s in st1])
    ivU1 = np.array([s["iv_U_pct"] for s in st1])
    bdg1 = np.array([0.5 * (s["bdg_iv3_pct"][0] + s["bdg_iv3_pct"][1]) for s in st1])
    ln1 = 0.5 * (st1[0]["ln_iv_pct"][0] + st1[0]["ln_iv_pct"][1])

    ax1.fill_between(k1, ivL1, ivU1, color=PALETTE["roughcert_fill"], alpha=0.30, lw=0)
    ax1.plot(k1, ivL1, color=PALETTE["roughcert"], lw=0.9)
    ax1.plot(k1, ivU1, color=PALETTE["roughcert"], lw=0.9)
    ax1.plot(k1, bdg1, "-", color=PALETTE["bdg"], marker="o", ms=3.4, lw=1.2)

    ax1.text(
        0.50, 0.96, f"LN flat: {ln1:.2f}%",
        transform=ax1.transAxes, ha="center", va="top", fontsize=5.8, color=PALETTE["ln"],
        bbox=dict(boxstyle="round,pad=0.20", fc="#F1F5F9", ec=PALETTE["ln"], lw=0.5),
    )
    ax1.text(0.97, 0.05, "BDG below at all strikes",
             transform=ax1.transAxes, fontsize=5.9, color=PALETTE["ink"], ha="right", va="bottom")
    ax1.set_xlabel(r"Log-moneyness $k$")
    ax1.set_ylabel("Black-76 implied vol. (%)")
    ax1.set_ylim(112.95, 115.85)
    ax1.set_xticks([-0.4, -0.2, 0.0, 0.2, 0.4])

    # --- Panel (b): Stress Cell (Refined 0.55 pp ATM band + BDG3 + LAB 2026) ---
    style_axes(ax2, grid_axis="y")
    panel_title(ax2, "b", r"Stress ($\eta_{\mathrm{P1}}=3$, 6 m)")
    st2 = opt["e14_stress_smile"]["strikes"]
    ref2 = ref["grid"]["options"]
    lab2 = next(x for x in lab if x["name"] == "stress")

    k2 = np.array([s["log_moneyness"] for s in st2])
    ivL2 = np.array([o["iv_L_pct"] for o in ref2])
    ivU2 = np.array([o["iv_U_pct"] for o in ref2])
    bdg2 = np.array([0.5 * (s["bdg_iv3_pct"][0] + s["bdg_iv3_pct"][1]) for s in st2])
    lab_iv2 = np.array(lab2["lab_iv"])
    ln2 = 0.5 * (st2[0]["ln_iv_pct"][0] + st2[0]["ln_iv_pct"][1])

    ax2.fill_between(k2, ivL2, ivU2, color=PALETTE["roughcert_fill"], alpha=0.30, lw=0)
    ax2.plot(k2, ivL2, color=PALETTE["roughcert"], lw=0.9)
    ax2.plot(k2, ivU2, color=PALETTE["roughcert"], lw=0.9)
    ax2.plot(k2, bdg2, "-", color=PALETTE["bdg"], marker="o", ms=3.4, lw=1.2)
    ax2.plot(k2, lab_iv2, "--", color=PALETTE["lab"], marker="D", ms=3.1, lw=1.15)

    ax2.text(
        0.50, 0.96, f"LN flat: {ln2:.2f}%",
        transform=ax2.transAxes, ha="center", va="top", fontsize=5.8, color=PALETTE["ln"],
        bbox=dict(boxstyle="round,pad=0.20", fc="#F1F5F9", ec=PALETTE["ln"], lw=0.5),
    )
    ax2.text(0.97, 0.05, "Refined ATM: 0.55 pp\nBDG$_3$ & LAB: 6/6 below",
             transform=ax2.transAxes, fontsize=5.9, color=PALETTE["ink"], ha="right", va="bottom")
    ax2.set_xlabel(r"Log-moneyness $k$")
    ax2.set_ylim(125.75, 129.00)
    ax2.set_xticks([-0.3, 0.0, 0.3, 0.6])

    # --- Panel (c): Mixed rBergomi Steep Smile (Root-split vs 100-node FD) ---
    style_axes(ax3, grid_axis="y")
    panel_title(ax3, "c", "Mixed rBergomi (3 m)")
    k3 = np.array([s["log_moneyness"] for s in load_json(RES / "vix_options_smile_suite.json")["mixed_rbergomi_smile"]["strikes"]])
    ivL3 = np.array([r["target_iv"][0] for r in mix["rows"]])
    ivU3 = np.array([r["target_iv"][1] for r in mix["rows"]])
    sp3 = np.array([0.5 * (r["split_iv"][0] + r["split_iv"][1]) for r in mix["rows"]])
    fd3 = np.array([r["code_iv"] for r in mix["rows"]])

    ax3.fill_between(k3, ivL3, ivU3, color=PALETTE["roughcert_fill"], alpha=0.30, lw=0)
    ax3.plot(k3, ivL3, color=PALETTE["roughcert"], lw=0.9)
    ax3.plot(k3, ivU3, color=PALETTE["roughcert"], lw=0.9)
    ax3.plot(k3, sp3, "-", color=PALETTE["bdg"], marker="o", ms=3.4, lw=1.2)
    ax3.plot(k3, fd3, ":", color=PALETTE["bdg_fd"], marker="s", mfc="white", ms=3.1, lw=1.0)

    ax3.text(0.97, 0.05, "Root-split BDG$_3$:\n7/7 strictly below",
             transform=ax3.transAxes, fontsize=5.9, color=PALETTE["bdg"], ha="right", va="bottom")
    ax3.set_xlabel(r"Log-moneyness $k$")
    ax3.set_ylim(92.0, 100.5)
    ax3.set_xticks([-0.3, -0.1, 0.1, 0.3])

    handles = [
        Line2D([0], [0], color=PALETTE["roughcert"], lw=5.0, alpha=0.38, label="RoughCert-CV IV band"),
        Line2D([0], [0], color=PALETTE["bdg"], marker="o", ms=3.7, lw=1.25, label="BDG (2023) order-3 / root-split"),
        Line2D([0], [0], color=PALETTE["lab"], ls="--", marker="D", ms=3.4, lw=1.2, label="LAB (2026) direct IV"),
        Line2D([0], [0], color=PALETTE["bdg_fd"], ls=":", marker="s", mfc="white", ms=3.4, lw=1.05, label="BDG$_3$ 100-node FD (mixed)"),
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.51, 1.07),
        ncol=4, frameon=False, fontsize=6.9, columnspacing=0.85, handlelength=1.45,
    )
    fig.subplots_adjust(top=0.85)
    return save_figure(fig, OUT_DIR, "fig_v3_04_vix_smiles")


# =============================================================================
# FIGURE 5: Mixed rBergomi IV Residuals + Bayer-Breneis Markovian Convergence N=1..19
# =============================================================================
def build_fig5():
    mix = load_json(RES / "mixed_split.json")
    be = load_json(RES / "band_entry.json")
    fd = load_json(RES / "factor_decisions.json")
    fb = load_json(RES / "factor_boundary.json")

    fig, (ax_m, ax_k) = plt.subplots(
        1, 2, figsize=(TEXTWIDTH_IN, 2.85),
        gridspec_kw={"width_ratios": [0.95, 1.15], "wspace": 0.33},
    )

    # --- Panel (a): Mixed rBergomi IV residuals relative to RoughCert-CV midpoint ---
    style_axes(ax_m, grid_axis="y")
    panel_title(ax_m, "a", "Mixed IV residual vs. RoughCert-CV")

    k = np.array([s["log_moneyness"] for s in load_json(RES / "vix_options_smile_suite.json")["mixed_rbergomi_smile"]["strikes"]])
    rows = mix["rows"]
    mid = np.array([0.5 * (r["target_iv"][0] + r["target_iv"][1]) for r in rows])
    lo_r = np.array([r["target_iv"][0] - m for r, m in zip(rows, mid)])
    hi_r = np.array([r["target_iv"][1] - m for r, m in zip(rows, mid)])
    sp_r = np.array([0.5 * (r["split_iv"][0] + r["split_iv"][1]) - m for r, m in zip(rows, mid)])
    fd_r = np.array([r["code_iv"] - m for r, m in zip(rows, mid)])

    ax_m.axhline(0.0, color=PALETTE["roughcert"], ls=":", lw=0.75, alpha=0.7)
    ax_m.fill_between(k, lo_r, hi_r, color=PALETTE["roughcert_fill"], alpha=0.30, lw=0, label="RoughCert-CV band")
    ax_m.plot(k, lo_r, color=PALETTE["roughcert"], lw=0.9)
    ax_m.plot(k, hi_r, color=PALETTE["roughcert"], lw=0.9)
    ax_m.plot(k, fd_r, "--", color=PALETTE["bcj"], marker="s", mfc="white", ms=3.8, lw=1.05, label="100-node FD (kink noise)")
    ax_m.plot(k, sp_r, "-", color=PALETTE["bdg"], marker="o", ms=4.0, lw=1.45, label="Root-split BDG$_3$ (Arb)")

    ax_m.annotate(
        "7/7 below\nlower edge",
        xy=(k[5], sp_r[5]), xytext=(0.01, -0.78),
        fontsize=6.2, color=PALETTE["bdg"], fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["bdg"], lw=0.75, shrinkA=1, shrinkB=2.5),
    )

    ax_m.set_xlabel(r"Log-moneyness $k$")
    ax_m.set_ylabel("IV minus band midpoint (pp)")
    ax_m.set_xticks([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
    ax_m.set_ylim(-0.92, 0.42)
    ax_m.legend(loc="upper center", bbox_to_anchor=(0.50, 1.00), fontsize=6.1,
                ncol=1, handlelength=1.4, labelspacing=0.18,
                frameon=True, facecolor="white", edgecolor="none", framealpha=0.88)

    # --- Panel (b): Bayer-Breneis Markovian factor convergence N=1..19 (log |error|) ---
    style_axes(ax_k, grid_axis="y")
    panel_title(ax_k, "b", "Bayer–Breneis Markovian factors ($N=1..19$)")

    ax_k.axhspan(0.05, 1.6, color=PALETTE["tick_fill"], alpha=0.45, zorder=0)
    ax_k.axhline(0.05, color=PALETTE["tick"], ls="--", lw=0.95, zorder=1)
    ax_k.text(19.5, 0.056, "1 tick ($0.05$)", ha="right", va="bottom",
              fontsize=6.5, color=PALETTE["tick"], fontweight="bold")

    for h, series in factor_paths().items():
        d=fd[h]
        tick=d["tick_thresholds_future_then_calls"][0]["first_proved"]
        band=d["fixed_band_threshold"]["first_proved"]
        lo,hi=np.array(series).T
        ns=np.arange(1,len(series)+1)
        mid=(lo+hi)/2
        col=H_COLORS[h]
        ax_k.plot(ns,mid,color=col,marker={".05":"s",".10":"o",".30":"^"}[h],ms=3,lw=1.1,
                  label=f"$H=0{h}$   {tick} / {band}")
        ax_k.fill_between(ns,lo,hi,color=col,alpha=.18,lw=0)
        ax_k.plot(tick,mid[tick-1],"o",mfc="none",mec=col,ms=8)
        ax_k.plot(band,mid[band-1],"D",mfc="white",mec=col,ms=5)
    ax_k.set_yscale("symlog",linthresh=.004)
    ax_k.set_yticks([0,.01,.05,.1,1],labels=["0",".01",".05",".1","1"])
    ax_k.set_xlabel("Number of Markovian exponential factors $N$")
    ax_k.set_ylabel(r"Absolute futures error $|F_N - F_{\mathrm{cont}}|$ (pts)")
    ax_k.set_xlim(0.4, 19.9)
    ax_k.set_xticks([1, 2, 5, 7, 9, 11, 14, 17, 19])
    ax_k.set_ylim(-.0003,1.45)
    # Place legend cleanly in the wide-open upper-right corner!
    ax_k.legend(title="Tick / band order", title_fontsize=6, loc="upper right", fontsize=6.05, handlelength=1.3, labelspacing=0.20,
                frameon=True, facecolor="white", edgecolor="none", framealpha=0.90)

    return save_figure(fig, OUT_DIR, "fig_v3_05_mixed_and_markov")


# =============================================================================
# FIGURE 6: Flagship 8-Method Curved Scoreboard (dot_table) + BCJ Quantiser Plateau
# =============================================================================
def build_fig6():
    q_suite = load_json(RES / "quantize_suite.json")
    values=scoreboard()
    fig,(ax_dot,ax_q)=plt.subplots(1,2,figsize=(TEXTWIDTH_IN,3.45),gridspec_kw={"wspace":.48})
    style_axes(ax_dot,grid_axis="x")
    panel_title(ax_dot,"a","Common curved contract")
    labels=["RoughCert-CV","Markovian (12)","Right grid (500)","BDG order-3","LAB direct IV","Lognormal","BCJ (960)","JMM bounds"]
    ref=values[0][0]
    for i,(F,_,_) in enumerate(values):
        y=7-i
        if F is None:
            ax_dot.text(.004,y,"IV only",va="center",fontsize=7)
        elif i in (0,7):
            ax_dot.plot((F[1]-F[0])/2,y,"D",color=PALETTE["roughcert"],ms=4.5)
        else:
            lo,hi=absolute(difference(F,ref))
            ax_dot.hlines(y,lo,hi,color=PALETTE["bdg"],lw=2)
            ax_dot.plot([lo,hi],[y,y],"|",color=PALETTE["bdg"],ms=5)
            ax_dot.plot((lo+hi)/2,y,"o",color=PALETTE["bdg"],ms=3.4)
    ax_dot.axvline(.05,color=PALETTE["tick"],ls="--",lw=.9)
    ax_dot.set_xscale("symlog",linthresh=.008)
    ax_dot.set_xlim(-.0006,3)
    ax_dot.set_xticks([0,.01,.05,.2,1],labels=["0",".01",".05",".2","1"])
    from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
    def signed_iv(x, rounding):
        d=Decimal.from_float(x).quantize(Decimal(".01"),rounding=rounding)
        return f"{d:+.2f}".replace("-", "−")
    iv_labels=[]
    for i,(name,(_,_,iv_band)) in enumerate(zip(labels,values,strict=True)):
        if i==0:
            subtitle="IV reference"
        elif iv_band is None:
            subtitle="Futures only"
        else:
            lo,hi=difference(iv_band,values[0][2])
            subtitle=f"IV [{signed_iv(lo,ROUND_FLOOR)}, {signed_iv(hi,ROUND_CEILING)}] pp"
        iv_labels.append(f"{name}\n{subtitle}")
    ax_dot.set_yticks(range(7,-1,-1),labels=iv_labels,fontsize=6.5)
    ax_dot.set_ylim(-.7,7.7)
    ax_dot.set_xlabel("Absolute futures error (pts)\nDiamonds: enclosure half-width")
    # --- Panel (b): BCJ (2023) Quantiser Bias Across 4 Budgets (48 -> 960 paths) ---
    style_axes(ax_q, grid_axis="y")
    panel_title(ax_q, "b", "BCJ quantiser error vs. paths")

    budgets = [50, 100, 500, 1000]
    paths = [48, 96, 480, 960]

    bcj_max, bcj_min, st_mid, q3_mid = [], [], [], []
    for b in budgets:
        r18 = [z for z in q_suite if z["budget"] == b and z["curve"] != "stress"]
        bcj_max.append(max(abs(z["signed_future_error"][0]) for z in r18))
        bcj_min.append(min(abs(z["signed_future_error"][1]) for z in r18))
        st = next(z for z in q_suite if z["budget"] == b and z["curve"] == "stress")
        st_mid.append(abs(0.5 * (st["signed_future_error"][0] + st["signed_future_error"][1])))
        q3 = next(z for z in q_suite if z["budget"] == b and z["curve"] == "quad" and z["months"] == 3)
        q3_mid.append(abs(0.5 * (q3["signed_future_error"][0] + q3["signed_future_error"][1])))

    ax_q.axhspan(0.05, 14.0, color=PALETTE["tick_fill"], alpha=0.45, zorder=0)
    ax_q.axhline(0.05, color=PALETTE["tick"], ls="--", lw=0.95, zorder=1)
    ax_q.text(44, 0.058, "1 Cboe tick ($0.05$ pts)", fontsize=6.4, color=PALETTE["tick"], fontweight="bold")

    max_w=max(r["condvar_width"] for r in load_json(RES/"vix_futures_full_suite.json")["block_c_bcj2023_18cells"])
    ax_q.axhline(max_w, color=PALETTE["roughcert"], ls="-.", lw=1.05, zorder=2)
    ax_q.text(1050, 0.0148, "RoughCert-CV max width ($0.013$)", ha="right",
              fontsize=6.1, color=PALETTE["roughcert"], fontweight="bold")

    ax_q.fill_between(paths, bcj_min, bcj_max, color=PALETTE["bcj"], alpha=0.16, lw=0, label="18-cell BCJ range")
    ax_q.loglog(paths, bcj_max, "-", color=PALETTE["bcj"], marker="^", ms=3.9, lw=1.35, label="Worst cell ($12$ m quad)")
    ax_q.loglog(paths, st_mid, "--", color=PALETTE["markov"], marker="s", ms=3.7, lw=1.25, label="Stress ($6$ m, $\\eta_{\\mathrm{P1}}=3$)")
    ax_q.loglog(paths, q3_mid, "-", color=PALETTE["bdg"], marker="o", ms=3.7, lw=1.25, label="Curved $3$ m (Panel a)")
    ax_q.loglog(paths, bcj_min, ":", color=PALETTE["bcj"], lw=0.95)

    ax_q.set_xlabel(r"Actual quantiser paths $N_{\mathrm{paths}}$")
    ax_q.set_ylabel(r"Absolute futures error $|F_{\mathrm{BCJ}} - F_{\mathrm{cont}}|$ (pts)")
    ax_q.set_xlim(38, 1250)
    ax_q.set_xticks(paths)
    ax_q.set_xticklabels([str(p) for p in paths])
    ax_q.xaxis.set_minor_formatter(NullFormatter())
    ax_q.set_ylim(0.008, 13.5)
    # Clean headroom above curves for legend!
    ax_q.legend(loc="upper right", fontsize=5.95, handlelength=1.35, labelspacing=0.18,
                frameon=True, facecolor="white", edgecolor="none", framealpha=0.90)

    return save_figure(fig, OUT_DIR, "fig_v3_06_scoreboard_and_quantizer")


# =============================================================================
# SUPPLEMENT FIGURE S1: Residual-to-Output Sandwich (24 tests, redesigned)
# =============================================================================
def build_fig_s1():
    c_closed = load_json(LEG / "C03_sandwich_bound" / "c03_results_closed.json")
    c_gen = load_json(LEG / "C03_sandwich_bound" / "c03_results_generic.json")
    all_rows = c_closed + c_gen

    funcs = [
        ("WickExp(eta=1.0)", r"Wick exp. $\mathcal{E}(y)$"),
        ("tanh(2y)", r"Smooth $\tanh(2y)$"),
        ("PositivePart(c=0.5)", r"Exceedance $(y-0.5)_+$"),
        ("Indicator(c=0.5)", r"Indicator $\mathbf{1}_{y>0.5}$"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(TEXTWIDTH_IN, 2.35), sharey=True, gridspec_kw={"wspace": 0.12})

    for ax, (fkey, title) in zip(axes, funcs):
        style_axes(ax, grid_axis="x")
        ax.set_title(title, loc="left", fontsize=7.6, pad=5.0)
        rows = sorted([r for r in all_rows if r["f"] == fkey], key=lambda x: (x["H"], x["rank"]))
        ys = np.arange(len(rows))[::-1]
        ax.axvline(1.0, color=PALETTE["ink"], ls="--", lw=0.8, zorder=1)
        for y, r in zip(ys, rows):
            lo = r["lower_over_P"]
            up = r["upper_over_P"]
            if np.isfinite(up):
                ax.hlines(y, lo, up, color=PALETTE["roughcert"], lw=3.2, alpha=0.32, zorder=2)
                ax.plot(up, y, "s", color=PALETTE["bdg"], ms=3.8, zorder=3)
            else:
                ax.hlines(y, lo, 1.24, color=PALETTE["muted"], lw=1.2, ls=":", zorder=2)
                ax.annotate("", xy=(1.255, y), xytext=(1.21, y),
                            arrowprops=dict(arrowstyle="-|>", color=PALETTE["muted"], lw=0.8))
            ax.plot(lo, y, "o", color=PALETTE["roughcert"], ms=4.0, zorder=4)

        ax.set_xlim(0.60, 1.28)
        ax.set_xticks([0.7, 1.0, 1.2])
        ax.set_xlabel("Bound / $P_\\rho$", fontsize=7.5)
        ax.set_yticks(ys)
        if ax is axes[0]:
            ax.set_yticklabels([f"$H={r['H']:g},\\,r={r['rank']}$" for r in rows], fontsize=6.9)
        ax.tick_params(axis="y", length=0)

    handles = [
        Line2D([0], [0], color=PALETTE["roughcert"], marker="o", ls="none", ms=4.2, label="Lower bound / $P_\\rho$"),
        Line2D([0], [0], color=PALETTE["bdg"], marker="s", ls="none", ms=4.0, label="Poincaré upper bound / $P_\\rho$"),
        Line2D([0], [0], color=PALETTE["ink"], ls="--", lw=0.85, label="Exact $P_\\rho$ ($=1$)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.51, 1.07), ncol=3, frameon=False, fontsize=7.3)
    fig.subplots_adjust(top=0.84)
    return save_figure(fig, OUT_DIR, "fig_v3_s1_sandwich")


# =============================================================================
# SUPPLEMENT FIGURE S2: Continuum Coverage & Width (E5, redesigned)
# =============================================================================
def build_fig_s2():
    e5 = load_json(LEG / "E5_bias_certified_mc" / "e5_results_A.json")
    Hs = [0.1, 0.3, 0.5, 0.75]
    h_keys = ["0.10", "0.30", "0.50", "0.75"]

    fig, (ax_c, ax_w) = plt.subplots(
        2, 1, figsize=(TEXTWIDTH_IN, 3.65), sharex=True,
        gridspec_kw={"hspace": 0.24},
    )

    style_axes(ax_c, grid_axis="y")
    style_axes(ax_w, grid_axis="y")
    panel_title(ax_c, "a", r"Empirical continuum coverage ($95\%$ quantile of $\Lambda = \int_0^1 B_H^2\,\mathrm{d}s$)")
    panel_title(ax_w, "b", "Median interval width across grid resolutions $n$")

    ax_c.axhline(0.95, color=PALETTE["tick"], ls="--", lw=0.9, zorder=1)
    ax_c.text(2.08, 0.87, "Nominal 0.95", ha="left", va="top", fontsize=6.5, color=PALETTE["tick"], fontweight="bold")

    markers = ["o", "s", "^", "D"]
    for H, hk, mk in zip(Hs, h_keys, markers):
        rows = sorted([r for r in e5 if abs(r["H"] - H) < 1e-6], key=lambda x: x["n"])
        ns = np.array([r["n"] for r in rows])
        cov_c = np.array([r["coverage_cert"] for r in rows])
        cov_n = np.array([r["coverage_naive"] for r in rows])
        w_c = np.array([r["median_width_cert"] for r in rows])
        w_n = np.array([r["median_width_naive"] for r in rows])
        col = H_COLORS[hk]

        ax_c.semilogx(ns, cov_c, "-", color=col, marker=mk, ms=4.0, lw=1.4)
        ax_c.semilogx(ns, cov_n, "--", color=col, marker=mk, mfc="white", ms=3.8, lw=1.05, alpha=0.85)
        ax_w.loglog(ns, w_c, "-", color=col, marker=mk, ms=4.0, lw=1.4)
        ax_w.loglog(ns, w_n, "--", color=col, marker=mk, mfc="white", ms=3.8, lw=1.05, alpha=0.85)

    ax_c.set_ylabel("Empirical coverage")
    ax_c.set_ylim(-0.05, 1.09)
    ax_w.set_ylabel("Median bracket width")
    ax_w.set_xlabel("Monitoring grid size $n$")
    ax_w.set_xticks([2, 4, 8, 16, 32, 64])
    ax_w.set_xticklabels(["2", "4", "8", "16", "32", "64"])
    ax_w.xaxis.set_minor_formatter(NullFormatter())

    handles = [
        Line2D([0], [0], color=H_COLORS[hk], marker=mk, lw=1.4, ms=4.0, label=f"$H={H:g}$")
        for H, hk, mk in zip(Hs, h_keys, markers)
    ] + [
        Line2D([0], [0], color=PALETTE["ink"], ls="-", lw=1.35, label="Bias-certified (solid)"),
        Line2D([0], [0], color=PALETTE["ink"], ls="--", lw=1.05, marker="o", mfc="white", ms=3.8, label="Naive Riemann MC (dashed)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.51, 1.03), ncol=6, frameon=False, fontsize=6.9, columnspacing=0.85)
    fig.subplots_adjust(top=0.88)
    return save_figure(fig, OUT_DIR, "fig_v3_s2_coverage")


# =============================================================================
# SUPPLEMENT FIGURE S3: Progressive Width Tightening Across Certificate Layers
# =============================================================================
def build_fig_s3():
    e5b = load_json(LEG / "E5b_conditional_certificates" / "e5b_results.json")
    e5c = load_json(LEG / "E5c_certified_cantelli" / "e5c_results.json")
    e5bin = load_json(LEG / "E5c_certified_cantelli" / "e5c_binomial.json")

    Hs = [0.1, 0.3, 0.5, 0.75]
    fig, axes = plt.subplots(1, 4, figsize=(TEXTWIDTH_IN, 2.35), sharey=True, gridspec_kw={"wspace": 0.12})

    for ax, H in zip(axes, Hs):
        style_axes(ax, grid_axis="y")
        ax.set_title(f"$H = {H:g}$", loc="left", fontsize=8.0, pad=5.0)
        b_rows = sorted([r for r in e5b if abs(r["H"] - H) < 1e-6], key=lambda x: x["n"])
        c_rows = sorted([r for r in e5c if abs(r["H"] - H) < 1e-6], key=lambda x: x["n"])
        bin_rows = sorted([r for r in e5bin if abs(r["H"] - H) < 1e-6], key=lambda x: x["n"])

        ns = np.array([r["n"] for r in b_rows])
        wa = np.array([r["width_a"] for r in b_rows])
        wb = np.array([r["width_b"] for r in b_rows])
        wc = np.array([r["width_c"] for r in c_rows])
        wd = np.array([r["width_d"] for r in c_rows])
        wdb = np.array([r["width_dbin"] for r in bin_rows])

        ax.loglog(ns, wa, ":", color=PALETTE["jmm"], marker="v", ms=3.6, lw=1.1, label="Riemann–$W_2$")
        ax.loglog(ns, wb, "--", color=PALETTE["markov"], marker="^", ms=3.6, lw=1.1, label="Cond. monitor")
        ax.loglog(ns, wc, "-", color=PALETTE["bdg"], marker="s", ms=3.6, lw=1.2, label="Martingale-sharp")
        ax.loglog(ns, wd, "-", color=PALETTE["roughcert"], marker="o", ms=3.8, lw=1.45, label="Cantelli (DKW)")
        ax.loglog(ns, wdb, "--", color=PALETTE["roughcert"], marker="D", mfc="white", ms=3.8, lw=1.35, label="Cantelli (binomial)")

        ax.set_xticks([4, 16, 64])
        ax.set_xticklabels(["4", "16", "64"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("$n$")

    axes[0].set_ylabel("Median 95% quantile width")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.51, 1.07), ncol=5, frameon=False, fontsize=6.7, columnspacing=0.8)
    fig.subplots_adjust(top=0.83)
    return save_figure(fig, OUT_DIR, "fig_v3_s3_layers")


def main():
    apply_style(base_size=8.5)
    builders = [
        ("Fig 1 (Pipeline & Rates)", build_fig1),
        ("Fig 2 (Phase Boundaries)", build_fig2),
        ("Fig 3 (Disjoint & Multi-Curve)", build_fig3),
        ("Fig 4 (VIX Smiles)", build_fig4),
        ("Fig 5 (Mixed Residuals & Markov N=1..19)", build_fig5),
        ("Fig 6 (Curved Scoreboard & BCJ Quantiser)", build_fig6),
        ("Fig S1 (Sandwich Validation)", build_fig_s1),
        ("Fig S2 (Continuum Coverage)", build_fig_s2),
        ("Fig S3 (Certificate Layers)", build_fig_s3),
    ]
    for label, fn in builders:
        paths = fn()
        print(f"BUILT {label}: {[p.name for p in paths]}")


if __name__ == "__main__":
    main()
