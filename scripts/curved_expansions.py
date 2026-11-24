"""Rigorous positive-cell integration of BDG and LAB coefficients for the curved scoreboard."""
from pathlib import Path
import json
from flint import arb
from arb_vix_core import set_dps,fwd_cov_arb,arb_interval,arb_lo_float,arb_hi_float
from arb_vix_options import make_quadratic_curve,_phi_arb,_Phi_arb,black76_iv_bracket_arb
ROOT=Path(__file__).resolve().parents[1]
def report(x):return [arb_lo_float(100*x),arb_hi_float(100*x)]
def main():
    set_dps(50)
    H=arb(".1");a=arb("1.9")*(2*H).sqrt();T=arb(1)/4;D=arb(1)/12
    curve=make_quadratic_curve(".234");xi=curve.avg_arb(T,D);M=360
    xs=[D*(arb(i)/M)**3 for i in range(M+1)]
    w=[curve.avg_arb(T+x,y-x)*(y-x)/(D*xi) for x,y in zip(xs,xs[1:])]
    C=[[None]*(M+1) for _ in xs]
    for i in range(M+1):
        for j in range(i,M+1):
            C[i][j]=C[j][i]=fwd_cov_arb(H,T,a,xs[i],xs[j],i==j)
    v=[arb_interval(C[i+1][i+1],C[i][i]) for i in range(M)]
    h=[arb_interval(sum(q*c for q,c in zip(w,C[i+1][1:])),sum(q*c for q,c in zip(w,C[i][:-1]))) for i in range(M)]
    avg=lambda z:sum(l*r for l,r in zip(w,z))
    var=avg(h);m=avg(v);S=xi.sqrt()*(-m/4+var/8).exp()
    q=[i-m for i in v];c=[i-var for i in h]
    g1=avg([i**2 for i in q])/8+(m-var)/2
    g2=-avg([i*j for i,j in zip(q,c)])/2
    g3=avg([i**2 for i in c])/2
    F=S*(1+g1/2+g2/4+g3/8);vol=(var/T).sqrt();tau=var.sqrt()/2
    rows=[]
    for k in [".25",".27",".30"]:
        K=arb(k);d=(S/K).log()/tau+tau/2
        P0=S*_Phi_arb(d)-K*_Phi_arb(d-tau)
        P1=S*_Phi_arb(d)/2
        P2=P1/2+S*_phi_arb(d)/(4*tau)
        P3=-P1/2+3*P2/2-S*_phi_arb(d)*(1+d/tau)/(8*tau)
        call=P0+g1*P1+g2*P2+g3*P3
        iv=black76_iv_bracket_arb(F.lower(),F.upper(),K,T,call.lower(),call.upper())
        lab=vol/2+g2/(2*vol*T)+3*g3/(8*vol*T)-g3*(S/K).log()/(vol**3*T**2)
        rows.append(dict(K=k,bdg_call=report(call),bdg_iv=[100*iv[0],100*iv[1]],lab_iv=report(lab)))
    (ROOT/"results/curved_expansions.json").write_text(json.dumps(dict(status="Arb-certified",M=M,forward=report(F),rows=rows),indent=2))
if __name__=="__main__":main()

