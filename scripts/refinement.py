import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.optimize import brentq
from flint import arb
from arb_vix_core import (tilted_and_lognormal_flat_arb,inner_h_kx_flat_arb,
                          norm_h_sq_flat_arb,set_dps)
from arb_vix_options import build_condvar_arb_model,make_flat_curve,condvar_vix_futures_and_options_arb
ROOT=Path(__file__).resolve().parents[1]
def main():
    set_dps(50)
    H,T=arb("0.1"),arb(1)/2
    out=[]
    for a,D in [(3*(2*H).sqrt(),arb(30)/365),(arb("1.5"),arb(1)/12)]:
        nh=float(norm_h_sq_flat_arb(H,T,D,a))
        x,w=np.polynomial.legendre.leggauss(80)
        vals=np.array([float(inner_h_kx_flat_arb(H,T,D,a,D*arb((1+v)/2))) for v in x])
        def grad(c):
            weight=w*np.exp(-c*vals/2)
            return c*nh-weight@vals/weight.sum()
        c=brentq(grad,.5,1.5)
        for scale in ["1",str(c)]:
            start=perf_counter()
            r=tilted_and_lognormal_flat_arb(H,T,D,a,arb(".235")**2,M=320,tilt_scale=scale)
            out.append(dict(a=str(a),c=scale,L=[r["L_outer_volpts"],r["L_inner_volpts"]],U=[r["U_inner_volpts"],r["U_outer_volpts"]],seconds=perf_counter()-start))
    old=json.loads((ROOT/"results/vix_options_smile_suite.json").read_text())["e14_stress_smile"]
    strikes=[r["K_decimal"] for r in old["strikes"]]
    start=perf_counter()
    model=build_condvar_arb_model(H,T,arb(30)/365,[(3*(2*H).sqrt(),arb(1))],make_flat_curve(".235"),M=320)
    cv=condvar_vix_futures_and_options_arb(model,strikes,n_core=480,n_tail_each=70)
    (ROOT/"results/refinement.json").write_text(json.dumps(dict(tilt=out,grid=cv,seconds=perf_counter()-start),indent=2))
if __name__=="__main__":main()

