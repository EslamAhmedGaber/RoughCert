"""Bayer--Breneis QF geometric Gaussian rule, theorem parameters, with optimal zero weight."""
from pathlib import Path
import json,time
from flint import arb,arb_mat
from arb_vix_core import set_dps, arb_interval,arb_lo_float,arb_hi_float
from arb_vix_options import (CondVarArbModel,build_condvar_arb_model,make_flat_curve,make_quadratic_curve,
                             condvar_vix_futures_and_options_arb)
ROOT=Path(__file__).resolve().parents[1]
def nodes_weights(H,N,L):
    # We approximate the unnormalised power, so its Laplace measure is
    # x^(-H-1/2) dx / Gamma(1/2-H).
    p=arb(1)/2-H
    if N==1:return [arb(0)],[L**(-p)/(1-p)]
    n=N-1; beta=arb(".4275"); alpha=arb("1.06418")
    A=(1/H+1/(arb("1.5")-H)).sqrt(); d=(alpha*beta).exp()
    exponent=1/(3*d/(8*(d-1))+6*H-4*H**2)
    b1=((9-6*H)/(2*H))**(d/(8*(d-1)))
    b2=5*arb.pi()**3*d*(d-1)*A**(2-2*H)*arb(n)**(1-H)/beta**(2-2*H)
    xmin=(b1*(b2*(3-2*H)/(768*H))**(2*H))**exponent*(-alpha*arb(n).sqrt()/((arb("1.5")-H)*A)).exp()/L
    xmax=(b1*(b2/1152)**(2*H-3))**exponent*(alpha*arb(n).sqrt()/(H*A)).exp()/L
    assert beta*arb(n).sqrt()/A < arb("1.5")
    x=[xmin*(xmax/xmin)**(arb(i)/n) for i in range(n+1)]
    nodes=[]; weights=[]
    for l,r in zip(x,x[1:]):
        mass=(r**p-l**p)/p
        nodes.append((r**(1+p)-l**(1+p))/((1+p)*mass))
        weights.append(mass/p.gamma())
    w0=(L**(1-p)/(1-p)-sum(w*(-x*L).expm1()/(-x) for x,w in zip(nodes,weights)))/L
    assert w0>0
    return [arb(0),*nodes],[w0,*weights]
def integral(u,L):
    return L if u.is_zero() else -(-u*L).expm1()/u
def finite_model(H,T,D,a,curve,nodes,weights,M):
    # All frequency weights positive, so g, its absolute derivatives and
    # ||k'_x|| decrease. Every endpoint and midpoint is a ball.
    q=len(nodes);one=arb(1)
    gram=arb_mat([[integral(x+y,T) for y in nodes] for x in nodes])
    def row(vals):return arb_mat([vals])
    def form(l,r):return (l*gram*r.transpose())[0,0]
    h=row([a*w*integral(u,D)/D for u,w in zip(nodes,weights)])
    norm=form(h,h).sqrt()
    def loading(x,der=0):
        return row([a*w*(-u*x).exp()*(u**der if der else 1) for u,w in zip(nodes,weights)])
    xs=[D*(arb(i)/M)**3 for i in range(M+1)]
    xm=[(xs[i]+xs[i+1])/2 for i in range(1,M)]
    hx=[(xs[i+1]-xs[i])/2 for i in range(1,M)]
    edge=[loading(x) for x in xs]; mid=[loading(x) for x in xm]
    g=[form(l,h)/norm for l in edge]; gm=[form(l,h)/norm for l in mid]
    gp=[];gpp=[];tau=[];std=[];dmax=[]
    for i in range(M):
        dm=(form(edge[i],edge[i])-g[i+1]**2).upper().max(arb(0))
        dmax.append(dm);std.append(dm.expm1().sqrt().upper())
        if i:
            deriv=loading(xs[i],1)
            gp.append((form(deriv,h)/norm).upper())
            gpp.append((form(loading(xs[i],2),h)/norm).upper())
            db=hx[i-1]*form(deriv,deriv).sqrt()
            tau.append(((dm/2).exp()*(1+dm*(dm.sqrt()*db).exp()).sqrt()*db).upper())
    A=arb_mat([[mid[i][0,j] for j in range(q)] for i in range(M-1)])
    C=A*gram*A.transpose()
    K=arb_mat([[(C[i,j]-gm[i]*gm[j]).expm1() for j in range(M-1)] for i in range(M-1)])
    boxes=[arb_interval(T+xs[i],T+xs[i+1]) for i in range(M)]
    return CondVarArbModel(H,T,D,[(one,one)],curve,curve.avg_arb(T,D),M,3,xs,xm,hx,g,gm,gp,gpp,dmax,[tau],[std[0]],[std],K,g[-1].lower(),g[0].upper(),(g[0].upper()+g[-1].lower())/2,((g[0].upper()-g[-1].lower())/2).upper(),
        [curve.eval_arb(T+x) for x in xm],[curve.eval_arb(b).upper() for b in boxes[1:]],[curve.d1_arb(b).abs_upper() for b in boxes[1:]],[curve.d2_arb(b).abs_upper() for b in boxes[1:]],curve.eval_arb(boxes[0]))
def interval_diff(A,B):
    d=arb_interval(arb(A[0]),arb(A[1]))-arb_interval(arb(B[0]),arb(B[1]))
    return [arb_lo_float(d),arb_hi_float(d)]
def main(rule=nodes_weights,filename="markov_suite"):
    set_dps(50);out=[]
    for label,h in [("flat",".05"),("flat",".10"),("flat",".30"),("curved",".10")]:
        H=arb(h); T=arb(1)/4;D=arb(1)/12
        a=arb("1.9")*(2*H).sqrt() if label=="curved" else arb(1)
        curve=make_quadratic_curve(".234") if label=="curved" else make_flat_curve(".235")
        strikes=[.25,.27,.30] if label=="curved" else [.18,.22,.26]
        target=condvar_vix_futures_and_options_arb(build_condvar_arb_model(H,T,D,[(a,arb(1))],curve,M=144),strikes,n_core=240,n_tail_each=40)
        for N in [1,2,3,5,8,12]:
            t=time.perf_counter()
            nodes,weights=rule(H,N,T+D)
            model=finite_model(H,T,D,a,curve,nodes,weights,128)
            cv=condvar_vix_futures_and_options_arb(model,strikes,n_core=256,n_tail_each=45)
            def bands(c):return [[c["future_L_volpts"],c["future_U_volpts"]],*[[r["call_L_volpts"],r["call_U_volpts"]] for r in c["options"]]]
            errs=[interval_diff(x,y) for x,y in zip(bands(cv),bands(target))]
            out.append(dict(curve=label,H=h,N=N,nodes=list(map(str,nodes)),weights_unit_a=list(map(str,weights)),status="Arb-certified",model=cv,target=target,error_volpts=errs,tick_states=["within" if max(abs(l),abs(u))<.05 else "outside" if l>.05 or u<-.05 else "unresolved" for l,u in errs],seconds=time.perf_counter()-t))
            (ROOT/"results"/(filename+".json")).write_text(json.dumps(out,indent=2,allow_nan=False))
            print(label,h,N,errs[0],flush=True)
if __name__=="__main__":main()


