"""Common target, lognormal moments and right-grid settlement audit on BCJ curve."""
from pathlib import Path
import json
from flint import arb
from arb_vix_core import arb_interval,arb_lo_float,arb_hi_float,fwd_cov_arb,set_dps
from arb_vix_options import make_quadratic_curve,discrete_right_rect_condvar_options_arb
from run_vix_futures_full_suite import jmm_bounds_curved_arb
from run_vix_options_smile_suite import lognormal_call_and_iv_arb
ROOT=Path(__file__).resolve().parents[1]
def main():
    set_dps(50)
    H=arb(".1");T=arb(1)/4;D=arb(1)/12;a=arb("1.9")*(2*H).sqrt();curve=make_quadratic_curve(".234")
    strikes=[.25,.27,.30]
    M=200;xs=[D*(arb(i)/M)**3 for i in range(M+1)]
    ws=[curve.avg_arb(T+x,y-x)*(y-x)/D for x,y in zip(xs,xs[1:])]
    lo=arb(0);hi=arb(0)
    cache=[[None]*(M+1) for _ in xs]
    for i in range(M+1):
        for j in range(i,M+1):
            cache[i][j]=fwd_cov_arb(H,T,a,xs[i],xs[j],i==j).exp()
    for i in range(M):
        for j in range(i,M):
            weight=ws[i]*ws[j]*(1 if i==j else 2)
            lo+=weight*cache[i+1][j+1];hi+=weight*cache[i][j]
    proxy=lognormal_call_and_iv_arb(curve.avg_arb(T,D),arb_interval(lo,hi),T,strikes)
    jmm=jmm_bounds_curved_arb(H,T,D,a,curve)
    grid=discrete_right_rect_condvar_options_arb(H,T,D,a,arb(".234")**2,strikes,n=500,n_core=250,curve=curve)
    out=dict(H=.1,T=.25,formula="0.234^2*(1+t)^2",strikes=strikes,status="Arb-certified",jmm={k:v for k,v in jmm.items() if isinstance(v,(float,list))},moment_matched=proxy,grid500=grid)
    (ROOT/"results/curved_comparison.json").write_text(json.dumps(out,indent=2,allow_nan=False))
if __name__=="__main__":main()

