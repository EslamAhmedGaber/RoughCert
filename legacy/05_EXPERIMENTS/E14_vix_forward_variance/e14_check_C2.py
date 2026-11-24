"""Diagnostic for criterion C2 (E V = P): does the truncation |z| <= 9.5 explain the stress-cell gap?
Recomputes E V and the future bracket with zmax = 9.5 and zmax = 16; frozen results are not altered."""
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import e14_vix as e

rows = json.loads((e.OUT / "e14_results.json").read_text())
print("worst cells by |check_EV|:")
for r in sorted(rows, key=lambda r: -abs(r["check_EV"]))[:4]:
    print(" ", r["group"], r["H"], round(r["T"], 3), f"{r['check_EV']:.2e}")
out = []
for H in (0.05, 0.10, 0.30):
    c = e.Cell(H, 0.5, 3.0, e.flat, "stress")
    x, w, xi, C = c.nodes(20, 14)
    eta = c.eta
    wxi = w * xi
    b = C @ wxi
    g = b / math.sqrt(float(wxi @ b))
    D = C - np.outer(g, g)
    P = float(np.sum(np.outer(wxi, wxi) * (np.exp(eta ** 2 * C) - np.exp(eta ** 2 * np.outer(g, g)))))
    for zmax, panels in ((9.5, 320), (16.0, 540)):
        z, pz = e.zrule(panels, zmax=zmax)
        Gh = xi * np.exp(eta * np.outer(z, g) - 0.5 * eta ** 2 * g ** 2)
        X = Gh @ w
        GW = Gh * w
        V = np.maximum(np.einsum("ij,ij->i", GW @ np.expm1(eta ** 2 * D), GW), 0.0)
        rho = 1.0 / np.sqrt(1.0 + V / X ** 2)
        row = {"H": H, "zmax": zmax, "EV_over_P_minus_1": float(pz @ V) / P - 1.0,
               "fut_lo": float(pz @ (rho * np.sqrt(X))), "fut_up": float(pz @ np.sqrt(X))}
        out.append(row)
        print(row)
(e.OUT / "e14_check_C2.json").write_text(json.dumps(out, indent=1))
