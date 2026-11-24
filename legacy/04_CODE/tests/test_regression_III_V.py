"""Regression of the generalised ledger code against frozen Paper III/V results."""
import math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gcert import *
from gcert.fields import CallableField

def test_paper_III_two_atom_table():
    t1, t2 = 0.3, 0.7
    def R(s, u):
        s, u = np.broadcast_arrays(np.asarray(s, float), np.asarray(u, float))
        return np.where((np.abs(s - u) < 1e-12) & ((np.abs(s - t1) < 1e-12) | (np.abs(s - t2) < 1e-12)), 1.0, 0.0)
    fld = CallableField(R, T=1.0, name="two-atom")
    rep = AnchorProjection(fld, [t1])
    f = WickExponential(xi0=1.0, eta=math.sqrt(math.log(2.0)), field=fld)
    nu = Measure.atomic([t1, t2], [1.0, 1.0])
    out = {}
    for lab, mu_w, total in (("plus", (0.5, 1.5), 2.5), ("minus", (1.5, 0.5), 0.5)):
        mu = Measure.atomic([t1, t2], mu_w)
        L = compute_ledger(fld, rep, f, mu, nu, N=None, audit=False)
        assert abs(L.total - total) < 1e-13, (lab, L.total)
        assert abs(L.M - 0.5) < 1e-13 and abs(L.P_nu + L.C_nu - 1.0) < 1e-13
        assert abs(abs(L.I) - 1.0) < 1e-13
        assert abs(L.sequential_check) < 1e-13 and abs(L.orthogonal_check) < 1e-13
        out[lab] = L
    assert out["plus"].I > 0 > out["minus"].I
    return {k: v.as_dict() for k, v in out.items()}

def test_paper_V_jet_two_atom():
    t1, t2 = 0.3, 0.7
    def R(s, u):
        s, u = np.broadcast_arrays(np.asarray(s, float), np.asarray(u, float))
        return np.where((np.abs(s - u) < 1e-12) & ((np.abs(s - t1) < 1e-12) | (np.abs(s - t2) < 1e-12)), 1.0, 0.0)
    fld = CallableField(R, T=1.0, name="two-atom")
    rep = AnchorProjection(fld, [t1])
    f = WickJet(xi0=1.0, eta=math.sqrt(math.log(2.0)), q=(1.0, 0.25), field=fld)
    L = compute_ledger(fld, rep, f, Measure.atomic([t1, t2], [0.5, 1.5]), Measure.atomic([t1, t2], [1, 1]), audit=False)
    # Paper V Remark (E0): q=(1,0.25): M=1.0221, I=+2.0442, total=5.1105
    assert abs(L.M - 1.0221) < 1e-4 and abs(L.I - 2.0442) < 1e-4 and abs(L.total - 5.1105) < 1e-4
    return L.as_dict()

def test_paper_V_E1_projection_budget():
    # Paper V E1 (fbm, H=0.1, T=0.5, eta=1.9, one anchor t1=T/2): Egen = 0.5197889 (audit 1.9e-4)
    fld = FBM(0.1, T=0.5)
    rep = AnchorProjection(fld, [0.25])
    f = WickExponential(1.0, 1.9, field=fld)
    mu = Measure.lebesgue(0.5)
    L = compute_ledger(fld, rep, f, mu, mu, N=None, n=16, levels=10, audit=True)
    # 30-digit mpmath reference (this lab): 0.51955877521387; Paper V reported 0.5197889 (conservative,
    # error 2.3e-4 > its stated audit 1.9e-4) -- see 00_ADMIN/CORRECTIONS_LOG.md
    assert abs(L.P_mu - 0.51955877521387) < 1e-11, L.P_mu
    assert L.P_mu < 0.5197889340609881
    return L.as_dict()

def test_paper_V_E3_measure_axis():
    # Paper V E3: d0, H=0.1, mid rule, n=4  ->  M_n = 0.526962962 (T=0.5, eta=1.9)
    fld = FBM(0.1, T=0.5)
    f = WickExponential(1.0, 1.9, field=fld)
    mu = Measure.lebesgue(0.5)
    nu = Measure.riemann(4, T=0.5, rule="mid")
    L = compute_ledger(fld, IdentityRepresentation(fld), f, mu, nu, N=None, audit=True)
    # 30-digit mpmath reference: 0.52695080649542; Paper V reported 0.526962962 (error 1.2e-5, conservative)
    assert abs(L.M - 0.52695080649542) < 1e-11, L.M
    assert L.M < 0.526962962334643
    return L.as_dict()

if __name__ == "__main__":
    import json
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            res = fn()
            keys = ("total", "M", "M_Q", "P_mu", "C_mu", "P_nu", "I", "quad_gap", "sequential_check", "orthogonal_check")
            if isinstance(res, dict) and "total" in res:
                print(name, {k: res[k] for k in keys})
            else:
                print(name, {kk: {k: vv[k] for k in keys} for kk, vv in res.items()})
