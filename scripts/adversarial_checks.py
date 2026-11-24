"""Deterministic endpoint, domain and inversion regressions for the released Arb engines."""
from pathlib import Path
from flint import arb,ctx
import json,math
from arb_vix_core import set_dps,rl_cov_arb,arb_interval,arb_lo_float,arb_hi_float
from arb_vix_options import _exp_quad_range_arb,black76_iv_bracket_arb,black76_call_arb
root=Path(__file__).resolve().parents[1]
def main():
    set_dps(24)
    checks={}
    def check(name,condition):
        checks[name]=bool(condition)
        assert condition,name
    check("precision_floor_bits",ctx.prec>=160)
    h=arb(".1");a=arb(1);b=a+arb(2)**-80
    check("distinct_times_identical_binary64",float(a)==float(b) and rl_cov_arb(h,a,b)<1)
    t=arb(2)**-1200
    check("positive_time_underflows_binary64",float(t)==0 and rl_cov_arb(h,t,a)>0)
    check("exact_zero",rl_cov_arb(h,arb(0),a).is_zero())
    p=arb_interval(arb(".99"),arb("1.01"));q=arb_interval(arb("1.001"),arb("1.02"))
    check("overlapping_boxes",rl_cov_arb(h,p,q).contains(rl_cov_arb(h,arb(1),arb("1.01"))))
    check("symmetric_order",rl_cov_arb(h,a,b).overlaps(rl_cov_arb(h,b,a)))
    for x in [arb(0),-arb(2)**-1200,arb(2)**-1200,arb("1.00000000000000000000001"),arb_interval(-a,b)]:
        check("signed_binary64_export_"+str(len(checks)),arb(arb_lo_float(x))<=x.lower() and arb(arb_hi_float(x))>=x.upper())
    zbox=arb_interval(arb(".98"),arb("1.03"))
    e=_exp_quad_range_arb(p,q,zbox)
    for g in [arb(".99"),arb("1.001"),arb("1.02")]:
        for z in [arb(".98"),arb("1.001"),arb("1.03")]:
            check("quadratic_vertex_"+str(len(checks)),e.contains((g*z-g**2/2).exp()))
    T=arb(1)/4;K=arb(1);sigma=arb(".37")
    exact=black76_call_arb(a,K,T,sigma)
    l,u,ok=black76_iv_bracket_arb(a,a,K,T,exact.lower(),exact.upper())
    check("black_known_root",ok and arb(l)<sigma<arb(u))
    l,u,ok=black76_iv_bracket_arb(a,arb("1.5"),K,T,arb(0),arb(".2"))
    check("black_zero_generalised_lower",ok and l==0 and u>0)
    l,u,ok=black76_iv_bracket_arb(a,arb("1.5"),K,T,arb(0),a)
    check("black_crossed_upper",ok and l==0 and math.isinf(u))
    check("black_intrinsic",black76_call_arb(arb(2),a,T,arb(0))==1)
    (root/"results/adversarial_checks.json").write_text(json.dumps(dict(bits=ctx.prec,checks=checks),indent=2))
    print(len(checks),"passed at",ctx.prec,"bits")
if __name__=="__main__":main()

