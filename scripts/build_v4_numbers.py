"""Generate the narrative number macros (tables/v2_numbers.tex and tables/v4_numbers.tex) from the JSON results.

Every macro records its rounding rule: 'down' (floor; certified lower bounds, gaps and tick counts),
'up' (ceiling; certified upper bounds and widths), 'zero' (toward zero; signed separations) and
'nearest' (point values such as fitted slopes).  The specification list SPEC is also read by the audit,
which recomputes every macro independently.
"""
import json
from decimal import Decimal, ROUND_CEILING, ROUND_DOWN, ROUND_FLOOR, ROUND_HALF_EVEN
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
LEG = RES / "legacy_inputs"
TAB = ROOT / "manuscript" / "tables"
TICK = 0.05
MODES = {"down": ROUND_FLOOR, "up": ROUND_CEILING, "zero": ROUND_DOWN, "nearest": ROUND_HALF_EVEN}


def rounded(x, n, mode):
    """Decimal text of the binary64 value x rounded at n decimals with the named rule (no sign handling)."""
    return format(Decimal.from_float(float(x)).quantize(Decimal(1).scaleb(-n), rounding=MODES[mode]), "f")


def load(name, legacy=None):
    path = (LEG / legacy / (name + ".json")) if legacy else (RES / (name + ".json"))
    return json.loads(path.read_text(encoding="utf-8"))


def spec():
    """(macro, value, decimals, rule, meaning); magnitudes of negative gaps are stated in the text."""
    g = {c["id"]: c for c in load("week1_gates_results")["gate2"]["cells"]}
    f = load("vix_futures_full_suite")
    o = load("vix_options_smile_suite")
    ref = load("refinement")
    e14 = {r["id"]: r for r in f["block_a_e14_15cells"]}
    asw = {round(r["a"], 2): r for r in f["block_b_bdg_sweeps"]["fig1_a_sweep"]}
    hsw = {round(r["H"], 2): r for r in f["block_b_bdg_sweeps"]["fig2_H_sweep"]}
    pure = o["bdg2023_fig6"]["strikes"]
    sep = {round(t["log_moneyness"], 2): t["bdg_call_cert_sep_volpts"] for t in pure}
    stress_gap = o["e14_stress_smile"]["bdg_F3_volpts"][0] - ref["grid"]["future_U_volpts"]
    ln_fig6 = g["BDG2023_Fig6_smile_T3m"]["tilted_L_outer"] - g["BDG2023_Fig6_smile_T3m"]["ln_fut"][1]
    ln_bcj = g["BCJ2023_Sec53_flat_T6m"]["tilted_L_outer"] - g["BCJ2023_Sec53_flat_T6m"]["ln_fut"][1]
    c03 = load("c03_results_closed", "C03_sandwich_bound") + load("c03_results_generic", "C03_sandwich_bound")
    occ = [r["lower_over_P"] for r in c03 if r["f"].startswith("Indicator")]
    smooth = [r["lower_over_P"] for r in c03 if not r["f"].startswith("Indicator")]
    finite = [r["upper_over_P"] for r in c03 if r["upper_over_P"] != float("inf")]
    e9 = {(r["f"], r["H"]): r["slope_64_1024"] for r in load("e9_results", "E9_measure_axis_rates") if "slope_64_1024" in r}
    bpc = load("v4_bdg_public_code_check")
    s = [
        ("TickStress", g["E14_stress_H010"]["bdg_minus_U_cert"] / TICK, 2, "down", "stress BDG excess in ticks"),
        ("TickFigOne", g["BDG2023_Fig1_a15_T6m"]["bdg_minus_U_cert"] / TICK, 2, "down", "Fig. 1 BDG excess in ticks"),
        ("TickStressHZeroFive", e14["stress_H005_6m"]["bdg_minus_U"] / TICK, 2, "down", "H=0.05 stress BDG excess in ticks"),
        ("GapStressRefined", stress_gap, 5, "down", "BDG excess over refined CV upper bound"),
        ("TickStressRefined", stress_gap / TICK, 2, "down", "same in ticks"),
        ("TickLNStress", g["E14_stress_H010"]["L_minus_LN_cert"] / TICK, 1, "down", "LN shortfall stress, ticks"),
        ("TickLNFigOne", g["BDG2023_Fig1_a15_T6m"]["L_minus_LN_cert"] / TICK, 1, "down", "LN shortfall Fig. 1, ticks"),
        ("LNGapFigSix", ln_fig6, 4, "down", "LN shortfall BDG Fig. 6 (magnitude)"),
        ("TickLNFigSix", ln_fig6 / TICK, 2, "down", "same in ticks"),
        ("LNGapBCJ", ln_bcj, 4, "down", "LN shortfall BCJ 5.3 (magnitude)"),
        ("TickLNBCJ", ln_bcj / TICK, 2, "down", "same in ticks"),
        ("SweepAOneTwo", asw[1.2]["bdg3_minus_U"], 4, "down", "a-sweep BDG excess at a=1.20"),
        ("SweepAOneThreeFive", asw[1.35]["bdg3_minus_U"], 4, "down", "a-sweep BDG excess at a=1.35"),
        ("SweepAOneFive", asw[1.5]["bdg3_minus_U"], 4, "down", "a-sweep BDG excess at a=1.50"),
        ("SweepHBDG", hsw[0.05]["bdg3_minus_U"], 5, "down", "H-sweep BDG excess at H=0.05"),
        ("SweepHBDGTicks", hsw[0.05]["bdg3_minus_U"] / TICK, 2, "down", "same in ticks"),
        ("SweepHLN", hsw[0.05]["L_minus_LN"], 4, "down", "H-sweep LN shortfall at H=0.05"),
        ("SweepHLNTicks", hsw[0.05]["L_minus_LN"] / TICK, 1, "down", "same in ticks"),
        ("SweepHWidth", hsw[0.05]["tilted_width"], 4, "up", "tilted width at H=0.05"),
        ("SweepHWidthRest", max(r["tilted_width"] for h, r in hsw.items() if h >= 0.1), 4, "up", "tilted width for H>=0.10"),
        ("LipFigSix", o["bdg2023_fig6"]["tilted_lipschitz_opt_bound_volpts"], 4, "up", "Lipschitz option bound, Fig. 6"),
        ("LipStress", o["e14_stress_smile"]["tilted_lipschitz_opt_bound_volpts"], 4, "up", "Lipschitz option bound, stress"),
        ("SepKmFour", sep[-0.4], 5, "zero", "BDG call above upper bound at k=-0.40"),
        ("SepKmThree", sep[-0.3], 5, "zero", "BDG call above upper bound at k=-0.30"),
        ("SepKpFour", -sep[0.4], 5, "zero", "BDG call below lower bound at k=+0.40 (magnitude)"),
        ("SandLoSmooth", 100 * min(smooth), 1, "down", "sandwich lower bound / P, smooth and kinked (%)"),
        ("SandHiSmooth", 100 * max(smooth), 1, "up", "same, maximum (%)"),
        ("SandLoOcc", 100 * min(occ), 1, "down", "sandwich lower bound / P, occupation (%)"),
        ("SandHiOcc", 100 * max(occ), 1, "up", "same, maximum (%)"),
        ("SandOverLo", 100 * (min(finite) - 1), 1, "down", "Poincare overshoot minimum (%)"),
        ("SandOverHi", 100 * (max(finite) - 1), 0, "up", "Poincare overshoot maximum (%)"),
        ("BDGDefaultExcessHi", bpc["max_default_excess"], 4, "up", "public BDG code, default 30-node excess (max)"),
    ]
    # Section 6-7 ranges: widths rounded up, certified separations rounded down (each end).
    lab = load("lab_suite")
    mixed = o["mixed_rbergomi_smile"]["strikes"]
    split = load("mixed_split")["rows"]
    scale = o["lab2026_smile"]["strikes"]
    rs = ref["grid"]["options"]
    lip = o["bdg2023_fig6"]["tilted_lipschitz_opt_bound_volpts"]
    pw = [t["call_width_volpts"] for t in pure]
    pgap = [t["iv_L_pct"] - t["bdg_iv3_pct"][1] for t in pure]
    sgap = [t["iv_L_pct"] - t["bdg_iv3_pct"][1] for t in scale]
    stress_lab = next(x for x in lab if x["name"] == "stress")["lab_arb_iv"]
    rgap = [v["iv_L_pct"] - b for t, v, l in zip(o["e14_stress_smile"]["strikes"], rs, stress_lab, strict=True)
            for b in (t["bdg_iv3_pct"][1], l[1])]
    msep = [x["target_iv"][0] - x["split_iv"][1] for x in split]
    mixg = [r["bdg3_minus_U"] for r in f["block_d_mixed_rbergomi"] if r["id"].startswith("mixed_scen1")]
    # A printed range must contain every value it summarises: minimum floored, maximum ceiled.
    for name, vals, n, meaning in [
        ("FigSixCallW", pw, 5, "BDG Fig. 6 call widths"),
        ("FigSixIVGap", pgap, 2, "BDG Fig. 6 IV below CV lower edge"),
        ("ScaleCallW", [t["call_width_volpts"] for t in scale], 5, "flat-scaling call widths"),
        ("ScaleIVGap", sgap, 2, "flat-scaling BDG IV below CV lower edge"),
        ("StressCallW", [t["call_width_volpts"] for t in rs], 5, "refined stress call widths"),
        ("StressIVGap", rgap, 2, "stress BDG and LAB IV below CV lower edge"),
        ("MixCallW", [t["call_width_volpts"] for t in mixed], 5, "mixed smile call widths"),
        ("MixIVW", [t["iv_width_pct"] for t in mixed], 2, "mixed smile IV widths"),
        ("MixSplitSep", msep, 3, "root-split IV below CV lower edge"),
    ]:
        s.append((name + "Lo", min(vals), n, "down", meaning + " (minimum, floored)"))
        s.append((name + "Hi", max(vals), n, "up", meaning + " (maximum, ceiled)"))
    ss = o["e14_stress_smile"]["strikes"]
    s += [
        ("BDGIVPureLo", min(t["bdg_iv3_pct"][0] for t in pure), 2, "down", "BDG order-three IV, Fig. 6 smile (minimum)"),
        ("BDGIVPureHi", max(t["bdg_iv3_pct"][1] for t in pure), 2, "up", "BDG order-three IV, Fig. 6 smile (maximum)"),
        ("LNIVPureLo", min(t["ln_iv_pct"][0] for t in pure), 2, "down", "lognormal IV, Fig. 6 smile (minimum)"),
        ("LNIVPureHi", max(t["ln_iv_pct"][1] for t in pure), 2, "up", "lognormal IV, Fig. 6 smile (maximum)"),
        ("LNAbovePureLo", min(t["ln_iv_pct"][0] - t["iv_U_pct"] for t in pure), 2, "down", "lognormal IV above CV upper edge, Fig. 6 (minimum)"),
        ("LNAbovePureHi", max(t["ln_iv_pct"][1] - t["iv_U_pct"] for t in pure), 2, "up", "lognormal IV above CV upper edge, Fig. 6 (maximum)"),
        ("BDGIVStressLo", min(t["bdg_iv3_pct"][0] for t in ss), 2, "down", "BDG order-three IV, stress smile (minimum)"),
        ("BDGIVStressHi", max(t["bdg_iv3_pct"][1] for t in ss), 2, "up", "BDG order-three IV, stress smile (maximum)"),
        ("LABIVStressLo", min(v[0] for v in stress_lab), 2, "down", "LAB 2026 IV, stress smile (minimum)"),
        ("LABIVStressHi", max(v[1] for v in stress_lab), 2, "up", "LAB 2026 IV, stress smile (maximum)"),
        ("LNIVStressLo", min(t["ln_iv_pct"][0] for t in ss), 2, "down", "lognormal IV, stress smile (minimum)"),
        ("LNIVStressHi", max(t["ln_iv_pct"][1] for t in ss), 2, "up", "lognormal IV, stress smile (maximum)"),
        ("LNAboveStressLo", min(t["ln_iv_pct"][0] - v["iv_U_pct"] for t, v in zip(ss, rs, strict=True)), 2, "down", "lognormal IV above refined CV upper edge (minimum)"),
        ("LNAboveStressHi", max(t["ln_iv_pct"][1] - v["iv_U_pct"] for t, v in zip(ss, rs, strict=True)), 2, "up", "lognormal IV above refined CV upper edge (maximum)"),
        ("StressFutW", ref["grid"]["future_U_volpts"] - ref["grid"]["future_L_volpts"], 5, "up", "refined stress futures bracket width"),
        ("StressExcWing", ss[0]["bdg_C3_volpts"][0] - rs[0]["call_U_volpts"], 5, "down", "BDG call above refined CV upper bound at K/F=0.70"),
        ("StressExcATM", ss[2]["bdg_C3_volpts"][0] - rs[2]["call_U_volpts"], 5, "down", "BDG call above refined CV upper bound at the money"),
        ("LipRatioLo", lip / max(pw), 0, "down", "Lipschitz bound / widest Fig. 6 call bracket"),
        ("LipRatioHi", lip / min(pw), 0, "up", "Lipschitz bound / narrowest Fig. 6 call bracket"),
        ("PureSkew", pure[-1]["iv_L_pct"] - pure[0]["iv_U_pct"], 2, "down", "certified IV rise k=-0.4 to +0.4"),
        ("MixSkew", mixed[-1]["iv_L_pct"] - mixed[0]["iv_U_pct"], 2, "down", "certified IV rise, mixed smile"),
        ("MixScenGapLo", min(mixg), 4, "down", "Scenario 1 mixed BDG excess over U (minimum)"),
        ("MixScenGapHi", max(mixg), 4, "up", "Scenario 1 mixed BDG excess over U (maximum)"),
        ("MLMCBelowGridU", g["BD2022_MLMC_JCF"]["grid500_tilted_arb"]["U_outer_volpts"] - o["bd2022_mlmc_call"]["published_fut_volpts"], 5, "down", "published MLMC below the 500-cell tilted upper edge (certified margin)"),
    ]
    e13 = load("e13_results", "E13_rough_stein_stein")
    e6c = load("e6c_results", "E6c_rough_vol_condvar")
    e14res = load("e14_results", "E14_vix_forward_variance")
    rough13 = [100 * r["P_ce"] / r["var_quad"] for r in e13 if r["H"] <= 0.1]
    s += [
        ("ResShareDailyLo", min(rough13), 1, "down", "rough Stein-Stein residual share, daily closes, H<=0.10 (minimum)"),
        ("ResShareDailyHi", max(rough13), 1, "up", "same (maximum)"),
        ("ShareVIXBase", max(100 * z["share"] for z in e14res if z["group"] == "base"), 3, "up", "VIX forward-variance residual share, base cells"),
        ("ShareVIXStress", max(100 * z["share"] for z in e14res if z["group"] == "stress"), 3, "up", "same, six-month stress cells"),
        ("ShareIVHi", max(100 * r["P"] / r["sd"] ** 2 for r in e6c), 1, "up", "rough Bergomi integrated-variance residual share, daily closes"),
        ("EsixcWidthLo", min(r["width_d_sd"] for r in e6c), 2, "down", "daily-close quantile bracket width in sd (minimum)"),
        ("EsixcWidthHi", max(r["width_d_sd"] for r in e6c), 2, "up", "same (maximum)"),
    ]
    futw = ([r["best_width"] for r in f["block_a_e14_15cells"]]
            + [r["tilted_width"] for r in list(asw.values()) + list(hsw.values())]
            + [r["condvar_width"] for r in f["block_c_bcj2023_18cells"] + f["block_d_mixed_rbergomi"]]
            + [r["tilted_total_outer_width"] for r in g.values()])
    s.append(("MaxFutW", max(futw), 4, "up", "widest certified VIX futures bracket in the study (%d brackets)" % len(futw)))
    for tag, name in [("Wick", "WickExp(eta=1.0)"), ("Kink", "PositivePart(c=0.5)"), ("Ind", "Indicator(c=0.5)"), ("Tanh", "tanh2y")]:
        for htag, h in [("A", 0.1), ("B", 0.3), ("C", 0.5), ("D", 0.7)]:
            s.append((f"Slope{tag}{htag}", -e9[(name, h)], 3, "nearest", f"fitted -slope {name} H={h}"))
    return s


def v2_values():
    legacy = load("vix_options_smile_suite")
    ref = load("refinement")
    checks = load("qmc_check")
    hits = sum(sum(c["contained"]) for c in checks)
    assert hits == sum(len(c["contained"]) for c in checks) == 35
    return [("SimChecks", str(hits)),
            ("StressATMOld", rounded(legacy["e14_stress_smile"]["strikes"][2]["iv_width_pct"], 3, "up")),
            ("StressATMNew", rounded(ref["grid"]["options"][2]["iv_width_pct"], 3, "up"))]


def write_all():
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "v2_numbers.tex").write_text("\n".join(r"\newcommand{\%s}{%s}" % kv for kv in v2_values()) + "\n", encoding="utf-8")
    lines = ["% Generated by scripts/build_v4_numbers.py from results/*.json -- do not edit by hand"]
    for name, value, n, mode, meaning in spec():
        lines.append(r"\newcommand{\%s}{%s}%% %s; %s" % (name, rounded(value, n, mode), mode, meaning))
    (TAB / "v4_numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("WROTE v2_numbers.tex, v4_numbers.tex")


if __name__ == "__main__":
    write_all()
