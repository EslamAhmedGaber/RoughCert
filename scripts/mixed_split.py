"""BDG mixed Thm 3.1 coefficient expression, evaluated at the exercise boundary."""
import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np
from flint import arb, acb
from arb_vix_core import (set_dps, norm_h_sq_flat_arb, bdg_futures_flat_arb,
                          arb_lo_float, arb_hi_float, arb_interval)
from arb_vix_options import black76_iv_bracket_arb

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"00_references/legacy"))
spec=importlib.util.spec_from_file_location("original_bdg",ROOT/"00_references/fbourgey_rbergomi_7822b00a.py")
original=importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)

def bounds(ball):
    return [arb_lo_float(100*ball),arb_hi_float(100*ball)]

def main():
    set_dps(50)
    case=json.loads((ROOT/"results/vix_options_smile_suite.json").read_text())["mixed_rbergomi_smile"]
    H,T,D=arb("0.1"),arb(1)/4,arb(1)/12
    amplitudes=[arb("1.4"),arb("0.7")]
    weights=[arb("0.3"),arb("0.7")]
    xi=arb("0.235")**2
    scales=[norm_h_sq_flat_arb(H,T,D,a).sqrt() for a in amplitudes]
    p=2*H+1
    mus=[-a*a*((T+D)**p-T**p-D**p)/(4*H*D*p) for a in amplitudes]
    coeff=[]
    for a,s in zip(amplitudes,scales):
        state=bdg_futures_flat_arb(H,T,D,a,xi)
        coeff.append((state["gamma_1_ball"],state["gamma_2_ball"]/s,state["gamma_3_ball"]/s**2))
    def vparts(z):
        return [xi*w*(m+s*z).exp() for w,m,s in zip(weights,mus,scales)]
    B=arb(16)
    normal=(2*arb.pi()).sqrt()
    def integrated(K):
        # E[(sqrt(Y)-K)+] plus the Hermite-weighted exact weak derivatives.
        def price_density(z,an):
            vv=vparts(z)
            root=sum(vv).sqrt(analytic=an)
            add=sum(y*(g1+g2*z+g3*(z*z-1)) for y,(g1,g2,g3) in zip(vv,coeff))
            return ((root-K)+add/(2*root))*(-z*z/2).exp()/normal
        lo,hi=-B,B
        if K.is_zero():
            hi=lo
        else:
            assert sum(vparts(lo))<K*K and sum(vparts(hi))>K*K
            for _ in range(110):
                mid=(lo+hi)/2
                y=sum(vparts(mid))
                if y<K*K: lo=mid
                elif y>K*K: hi=mid
                else: break
        integral=acb.integral(price_density,acb(hi),acb(B),rel_tol=1e-14,abs_tol=1e-17).real
        if hi>lo:
            gap=(hi-lo)*price_density(acb(arb_interval(lo,hi)),False).real.abs_upper()
            integral+=arb(0,gap.upper())
        # |psi_j| <= sqrt(Y)/2. Both tails are dominated by
        # A*(1+z^2)*exp(-z^2/4)/sqrt(2*pi) since max sigma <= B/2.
        assert max(s.upper() for s in scales)<B/2 and all(m<0 for m in mus)
        A=K+xi.sqrt()*(1+sum(abs(g1)+abs(g2)+2*abs(g3) for g1,g2,g3 in coeff))
        tail=2*A*(3*arb.pi().sqrt()*(B/2).erfc()+2*B*(-B**2/4).exp())/normal
        return integral+arb(0,tail.upper())
    ref=original.RoughBergomi(1.0,lambda t:np.full_like(t,0.235**2),.1,1.4/math.sqrt(.2),-.5)
    def published(k,kind):
        return float(ref.price_vix_approx_mixed(.25,k,.3,.7/math.sqrt(.2),kind,order=3,n_quad=100,eps=1e-4))
    F=integrated(arb(0))
    origF=published(0.0,"fut")
    rows=[]
    for old in case["strikes"]:
        k=old["K_decimal"]; K=arb(str(k))
        C=integrated(K)
        iv=black76_iv_bracket_arb(F.lower(),F.upper(),K,T,C.lower(),C.upper())
        raw=published(k,"call")
        orig_iv=black76_iv_bracket_arb(arb(origF),arb(origF),K,T,arb(raw),arb(raw))
        rows.append(dict(K=k,split_call=bounds(C),split_iv=[100*iv[0],100*iv[1]],
                         code_call=100*raw,code_iv=50*(orig_iv[0]+orig_iv[1]),
                         target_iv=[old["iv_L_pct"],old["iv_U_pct"]]))
    out=dict(status_split="Arb-certified",status_original="diagnostic",formula="BDG Theorem 3.1 coefficient functional",quadrature_order=100,eps=.0001,
             split_forward=bounds(F),code_forward=100*origF,rows=rows)
    (ROOT/"results/mixed_split.json").write_text(json.dumps(out,indent=2,allow_nan=False))
    print(out)
if __name__=="__main__":
    main()

