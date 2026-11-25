"""Run the authors' public BDG (2023) futures expansion at default and refined quadrature.

The public routine price_vix_fut_approx (rbergomi.py, commit 7822b00a) evaluates the Theorem 2.2
coefficients gamma_2 and gamma_3 with an n_quad-point Gauss-Legendre rule (default 30).  This script
compares the default rule, refined rules and adaptive quadrature with the 169-bit Arb evaluation of the
same coefficient formulas stored in results/week1_gates_results.json.  Status: diagnostic (floating point).
"""
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "00_references/legacy"))
spec = importlib.util.spec_from_file_location("original_bdg", ROOT / "00_references/fbourgey_rbergomi_7822b00a.py")
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


def price(cell, n_quad=30, quad_scipy=False):
    rb = original.RoughBergomi(1.0, lambda t: np.full_like(t, cell["xi0"]), cell["H"], cell["eta_P1"], -0.5)
    rb.delta_vix = cell["Delta"]
    T = cell["T"]
    S = np.exp((rb.mean_proxy_flat(T) + np.log(rb.fut_vix2(T))) / 2.0 + rb.var_proxy_flat(T) / 8.0)
    g1 = rb.gamma_1_proxy_flat(T=T)
    g2 = rb.gamma_2_proxy_flat(T=T, quad_scipy=quad_scipy, n_quad=n_quad)
    g3 = rb.gamma_3_proxy_flat(T=T, quad_scipy=quad_scipy, n_quad=n_quad)
    return 100.0 * S * (1.0 + g1 / 2.0 + g2 / 4.0 + g3 / 8.0)


def main():
    warnings.filterwarnings("ignore")
    cells = json.loads((ROOT / "results/week1_gates_results.json").read_text())["gate2"]["cells"]
    rows = []
    for c in cells:
        arb_lo, arb_hi = c["bdg_F3_arb"]
        row = dict(id=c["id"], arb=[arb_lo, arb_hi], default_n30=price(c, 30), n1000=price(c, 1000),
                   adaptive=price(c, quad_scipy=True))
        row["default_excess"] = row["default_n30"] - arb_hi
        row["n1000_offset"] = max(abs(row["n1000"] - arb_lo), abs(row["n1000"] - arb_hi))
        row["adaptive_offset"] = max(abs(row["adaptive"] - arb_lo), abs(row["adaptive"] - arb_hi))
        rows.append(row)
        print(row, flush=True)
    out = dict(status="diagnostic", source="00_references/fbourgey_rbergomi_7822b00a.py", rows=rows,
               max_default_excess=max(r["default_excess"] for r in rows),
               min_default_excess=min(r["default_excess"] for r in rows),
               max_refined_offset=max(max(r["n1000_offset"], r["adaptive_offset"]) for r in rows))
    (ROOT / "results/v4_bdg_public_code_check.json").write_text(json.dumps(out, indent=2))
    print({k: v for k, v in out.items() if k != "rows"})


if __name__ == "__main__":
    main()
