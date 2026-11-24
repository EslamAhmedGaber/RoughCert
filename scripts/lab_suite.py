"""LAB Eq. (21), author-code comparison and fixed physical strike Arb IV bands."""
from pathlib import Path
import sys, json, math
import numpy as np
from flint import arb
from arb_vix_options import (build_condvar_arb_model,condvar_vix_futures_and_options_arb,
                             make_flat_curve,make_quadratic_curve)
from arb_vix_core import set_dps, norm_h_sq_flat_arb, bdg_futures_flat_arb,arb_lo_float,arb_hi_float
from run_vix_options_smile_suite import bdg_call_and_iv_flat_arb
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"00_references/bourgey"))
from rbergomi import RoughBergomiModel
def main():
    set_dps(50)
    output=[]
    for name,h,amp,month in [(f"LAB-{H}-{m}",H,a,m) for H,a in [("0.10","1"),("0.23","1.02")] for m in [1,3,6]]+[("stress","0.10","stress",6),("curved","0.10","bcj",3)]:
        H,T=arb(h),arb(month)/12
        a={"stress":3*(2*H).sqrt(),"bcj":arb("1.9")*(2*H).sqrt()}.get(amp, None)
        if a is None: a=arb(amp)
        curved=name=="curved"
        D=arb(1)/12 if curved else arb(30)/365
        base="0.234" if curved else "0.235" if name=="stress" else "0.24"
        curve=make_quadratic_curve(base) if curved else make_flat_curve(base)
        model=RoughBergomiModel(dict(H=float(H),eta=float(a)/math.sqrt(2*float(H))),lambda t:float(base)**2*(1+t)**2 if curved else np.full_like(t,float(base)**2))
        model.delta_vix=float(D)
        tt=float(T)
        F=float(model.price_vix_fut_approx(T=tt,order=3,n_quad=30))
        klist=F*np.exp(np.linspace(-.1,.4,10))
        if name=="stress":
            legacy=json.loads((ROOT/"results/vix_options_smile_suite.json").read_text())
            klist=np.array([r["K_decimal"] for r in legacy["e14_stress_smile"]["strikes"]])
        if curved:klist=np.array([.25,.27,.30])
        mp=np.log(klist/F)
        direct=model.implied_vol_vix_expansion(mp,tt,order=1,n_quad=30)*100
        fine_F=float(model.price_vix_fut_approx(T=tt,order=3,n_quad=60))
        fine_mp=np.log(klist/fine_F)
        fine=model.implied_vol_vix_expansion(fine_mp,tt,order=1,n_quad=60)*100
        weak=model.implied_vol_vix_approx(k=mp,T=tt,order=3,n_quad=30)*100
        ref=build_condvar_arb_model(H,T,D,[(a,arb(1))],curve,M=152,grade_p=3)
        arbcase=condvar_vix_futures_and_options_arb(ref,klist.tolist(),n_core=280,n_tail_each=45)
        verified=[]
        if not curved:
            xi=arb(base)**2
            bdg=bdg_futures_flat_arb(H,T,D,a,xi)
            var=norm_h_sq_flat_arb(H,T,D,a)
            vol=(var/T).sqrt()
            g2,g3=bdg["gamma_2_ball"],bdg["gamma_3_ball"]
            for k in klist:
                d=vol/2+g2/(2*vol*T)+3*g3/(8*vol*T)-g3*(bdg["S_ball"]/arb(str(float(k)))).log()/(vol**3*T**2)
                verified.append([arb_lo_float(100*d),arb_hi_float(100*d)])
        output.append(dict(name=name,H=h,amplitude=str(a),T_months=month,Delta=str(D),forward_BDG=100*F,
                           author_nodes=30,refinement_nodes=60,refined_forward=100*fine_F,refined_iv=fine.tolist(),arb=arbcase,lab_iv=direct.tolist(),bdg_iv=weak.tolist(),lab_arb_iv=verified,
                           source="2604.25123v3 Eq.21; d11feba",status="Arb-certified for target and flat direct IV intervals; diagnostic for source-code points"))
        (ROOT/"results/lab_suite.json").write_text(json.dumps(output,indent=2,allow_nan=False))
        print(name,min(direct),max(direct),flush=True)
if __name__=="__main__":main()


