"""Evaluate the pre-declared E14 criteria from the frozen results; write e14_audit.json."""
import json, math
from pathlib import Path

OUT = Path(__file__).parent
rows = json.loads((OUT / "e14_results.json").read_text())
vp, tick = 100.0, 0.05
main = [r for r in rows if r["group"] in ("base", "curve")]
report = {"cells": len(rows)}
fut_in = [r["fut_lo"] - 3 * r["fut_se"] <= r["fut_ref"] <= r["fut_up"] + 3 * r["fut_se"] for r in rows]
call_in = [c["lo"] - 3 * c["se"] <= c["ref"] <= c["up"] + 3 * c["se"] for r in rows for c in r["calls"]]
report["C1_future_agreement_with_3se"] = f"{sum(fut_in)}/{len(fut_in)}"
report["C1_calls_agreement_with_3se"] = f"{sum(call_in)}/{len(call_in)}"
report["C1_strict_inside_no_se_future"] = sum(r["fut_lo"] <= r["fut_ref"] <= r["fut_up"] for r in rows)
report["C1_strict_inside_no_se_calls"] = sum(c["lo"] <= c["ref"] <= c["up"] for r in rows for c in r["calls"])
report["C1_interpretation"] = "Monte Carlo point agreement within three standard errors, not simultaneous coverage; quadrature errors are refinement-audited."
report["C2_max_abs_EV_minus_P_rel"] = max(abs(r["check_EV"]) for r in rows)
report["C2_max_abs_EX_minus_mean_rel"] = max(abs(r["check_EX"]) for r in rows)
report["C2_max_gap_vol_points"] = vp * max([r["fut_gap"] for r in rows] + [c["gap"] for r in rows for c in r["calls"]])
fw = {(r["group"], r["H"], round(r["T"], 4)): vp * (r["fut_up"] - r["fut_lo"]) for r in rows}
report["C3_cells_below_tick"] = f"{sum(vp * (r['fut_up'] - r['fut_lo']) < tick for r in main)}/{len(main)}"
report["C3_max_width"] = max(vp * (r["fut_up"] - r["fut_lo"]) for r in main)
stress = [r for r in rows if r["group"] == "stress"]
report["C4_stress"] = [{"H": r["H"], "share": r["share"], "width": vp * (r["fut_up"] - r["fut_lo"]),
                        "published": vp * (r["jmm_up"] - r["jmm_lo"]), "ref": vp * r["fut_ref"],
                        "lognormal_err": vp * (r["lognormal"] - r["fut_ref"]),
                        "call_max_width": max(vp * (c["up"] - c["lo"]) for c in r["calls"])} for r in stress]
report["C5_ratio_min_main"] = min((r["jmm_up"] - r["jmm_lo"]) / (r["fut_up"] - r["fut_lo"]) for r in main)
report["C5_ratio_max_all"] = max((r["jmm_up"] - r["jmm_lo"]) / (r["fut_up"] - r["fut_lo"]) for r in rows)
report["lognormal_outside"] = [{"H": r["H"], "T": r["T"], "group": r["group"], "below": r["lognormal"] < r["fut_lo"],
                                "above": r["lognormal"] > r["fut_up"]} for r in rows]
report["q99_mc_inside_cv"] = sum(r["q99"]["cv"][0] <= r["q99"]["mc"] <= r["q99"]["cv"][1] for r in rows)
report["q99_mc_ci_overlap_cv"] = sum(r["q99"]["cv"][0] <= r["q99"]["mc_hi"] and r["q99"]["mc_lo"] <= r["q99"]["cv"][1] for r in rows)
report["q99_mc_inside_mart"] = sum(r["q99"]["mart"][0] <= r["q99"]["mc"] <= r["q99"]["mart"][1] for r in rows)
plan = []
for r in rows:
    if r["group"] == "base" and abs(r["T"] - 1 / 12) < 1e-12:
        p = {"H": r["H"], "rank1_rms_over_sd": math.sqrt(r["P"]) / r["sd"], "rank1_rms": math.sqrt(r["P"])}
        for rule in ("trap", "rect"):
            p[rule] = [(q["n"], q["rms"] / r["sd"]) for q in r["plan"][rule]]
            ok = [q["n"] for q in r["plan"][rule] if q["rms"] / (2 * r["fut_ref"]) * vp <= tick]
            p[rule + "_tick_n"] = min(ok) if ok else None
            better = [q["n"] for q in r["plan"][rule] if q["rms"] <= math.sqrt(r["P"])]
            p[rule + "_beats_rank1_from_n"] = min(better) if better else None
        plan.append(p)
report["plan"] = plan
(OUT / "e14_audit.json").write_text(json.dumps(report, indent=1))
print(json.dumps({k: v for k, v in report.items() if k not in ("plan", "lognormal_outside")}, indent=1))
print(report["lognormal_outside"])
for p in plan:
    print(p["H"], "rank1", round(p["rank1_rms_over_sd"], 4), "trapN", p["trap_tick_n"], "rectN", p["rect_tick_n"], "trap beats from", p["trap_beats_rank1_from_n"], "rect from", p["rect_beats_rank1_from_n"])
    print("  trap", [(n, round(v, 4)) for n, v in p["trap"]])
    print("  rect", [(n, round(v, 4)) for n, v in p["rect"]])
