"""Condition the tilted residual on a kernel contrast to resolve factor crossovers."""
from pathlib import Path
import json,math
from flint import arb,arb_mat
from arb_vix_core import (fwd_cov_arb,arb_interval,norm_h_sq_flat_arb,
                          inner_h_kx_flat_arb,arb_lo_float,arb_hi_float)
from arb_vix_options import build_z_grid_arb,_panel_mass_and_mean_arb,_exp_quad_range_arb
from markov_suite import integral,interval_diff
ROOT=Path(__file__).resolve().parents[1]
def solve(H,T,D,r=None,M=180,scale='.99'):
    p=arb(1)/2-H; one=arb(1); xi=arb(".235")**2; c=arb(scale)
    if r is None:
        C=lambda x,y:fwd_cov_arb(H,T,one,x,y,x is y or x==y)
        hk=lambda x:c*inner_h_kx_flat_arb(H,T,D,one,x,at_Delta=(x is D))
        n=c**2*norm_h_sq_flat_arb(H,T,D,one)
        def der(x,j):
            return (p if j==1 else p*(p+1))*((x**(1-2*p-2*j)-(T+x)**(1-2*p-2*j))/(2*p+2*j-1)).sqrt()
    else:
        rates=[arb(s) for s in r["nodes"]]; weights=[arb(s) for s in r["weights"]]
        G=arb_mat([[integral(x+y,T) for x in rates] for y in rates])
        from functools import lru_cache
        def vec(x,j=0):return arb_mat([[w*(-a*x).exp()*a**j for w,a in zip(weights,rates)]])
        h=arb_mat([[c*w*integral(a,D)/D for w,a in zip(weights,rates)]])
        # Cache by node identity since Arb balls are unhashable.
        cache={}
        def pair(x):
            i=id(x)
            if i not in cache:
                z=vec(x);cache[i]=(x,z*G,z.transpose())
            return cache[i]
        C=lambda x,y:(pair(x)[1]*pair(y)[2])[0,0]
        hk=lambda x:(pair(x)[1]*h.transpose())[0,0]
        n=(h*G*h.transpose())[0,0]
        def der(x,j):
            z=vec(x,j);return (z*G*z.transpose())[0,0].sqrt()
    t=D/14
    en=(C(t,t)+C(D,D)-2*C(t,D)).sqrt()
    he=(hk(t)-hk(D))/en
    def g(x):return (C(x,t)-C(x,D))/en-he
    edge=[D*(arb(i)/M)**3 for i in range(M)]+[D]
    width=[y-x for x,y in zip(edge,edge[1:])]
    mid=[(x+y)/2 for x,y in zip(edge,edge[1:])]
    ge=[g(x) for x in edge]; gm=[g(x) for x in mid]; hedc=[hk(x) for x in edge];hm=[hk(x) for x in mid]
    B=[((n-v)/2).exp() for v in hm]
    gs=[arb_interval(ge[i+1],ge[i]) for i in range(M)]
    dmax=[(C(edge[i],edge[i])-2*hedc[i+1]+n-gs[i]**2).upper().max(arb(0)) for i in range(M)]
    sd=[z.expm1().sqrt().upper() for z in dmax]
    K=arb_mat(M-1,M-1)
    for i in range(1,M):
        for j in range(i,M):
            k=(C(mid[i],mid[j])-hm[i]-hm[j]+n-gm[i]*gm[j]).expm1()
            K[i-1,j-1]=K[j-1,i-1]=k
    d1=[None]+[der(x,1).upper() for x in edge[1:-1]]
    d2=[None]+[der(x,2).upper() for x in edge[1:-1]]
    tau=[sd[0]]
    for i in range(1,M):
        db=d1[i]*width[i]/2;dm=dmax[i]
        tau.append((db*(dm/2).exp()*(1+dm*(dm.sqrt()*db).exp()).sqrt()).upper())
    def moments(z):
        wk=[]
        for i,dx in enumerate(width):
            factor=(dx*xi/D)*((n-hedc[i+1])/2).exp()
            e=_exp_quad_range_arb(ge[i+1],ge[i],z)
            if not i:
                b=arb_interval(((n-hedc[i])/2).exp(),((n-hedc[i+1])/2).exp())
                wk.append((dx*xi/D)*b*e);continue
            dz=(z-gs[i]).abs_upper()+n.sqrt()/2
            e2=dz*d2[i]+d1[i]**2+(dz*d1[i])**2
            err=factor*e.upper()*dx**2*e2/24
            wk.append(dx*xi/D*B[i]*(gm[i]*z-gm[i]**2/2).exp()+arb(0,err.upper()))
        return wk,sum(wk)
    nodes=build_z_grid_arb(n_core=260,n_tail_each=40)
    X=[moments(z)[1] for z in nodes];bars=[];panels=[]
    for a,b in zip(nodes,nodes[1:]):
        mass,_,z=_panel_mass_and_mean_arb(a,b)
        wk,xb=moments(z)
        panels.append((mass,z,xb,(z-a).max(b-z).upper(),(z-a)/(b-a)));bars.append(wk)
    A=arb_mat([[w[i] for w in bars] for i in range(1,M)])
    KA=K*A; lo=arb(0);up=arb(0)
    dg=(ge[0]-ge[-1])/2
    for j,(mass,z,xb,h,w) in enumerate(panels):
        wk=bars[j]
        variance=sum(A[i,j]*KA[i,j] for i in range(M-1)).upper().max(arb(0))
        s=variance.sqrt()+sum(x.upper()*y for x,y in zip(wk,tau))
        S=sum(x.upper()*y for x,y in zip(wk,sd))
        kap=(dg*h).exp()*(s+(dg*h).expm1()*S)/xb.lower()
        assert xb>0
        lo+=mass*xb.lower().sqrt()/(1+kap**2).sqrt()
        up+=mass*((1-w)*X[j].upper().sqrt()+w*X[j+1].upper().sqrt())
    up+=(xi*((n-hedc[-1])/2).exp()*(arb(10)/arb(2).sqrt()).erfc()).sqrt()
    pref=100*(-n/8).exp()
    return [arb_lo_float(pref*lo),arb_hi_float(pref*up)]
def main():
    rows=json.loads((ROOT/"results/factor_thresholds.json").read_text())
    H=arb(".05");T=arb(1)/4;D=arb(1)/12
    target=solve(H,T,D)
    results=[]
    for n in [12,13,14,15]:
        r=next(x for x in rows if x["H"]==".05" and x["N"]==n)
        value=solve(H,T,D,r)
        results.append(dict(N=n,target=target,model=value,error=interval_diff(value,target)))
        (ROOT/"results/tilted_conditioning.json").write_text(json.dumps(results,indent=2))
        print(results[-1],flush=True)
if __name__=="__main__":main()

