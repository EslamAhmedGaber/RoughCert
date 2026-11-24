"""Independent quadrature of the Brownian kernel followed by Gaussian PCA and RQMC."""
from pathlib import Path
import json, math
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import special, integrate, optimize, stats
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]

def gauss(n,p):
    x,w=leggauss(n); u=(x+1)/2
    return u**p,.5*p*w*u**(p-1)

def iv(F,K,T,C):
    def residual(v):
        s=v*math.sqrt(T);d=math.log(F/K)/s+s/2
        return F*special.ndtr(d)-K*special.ndtr(d-s)-C
    return optimize.brentq(residual,1e-6,8)

def estimate(T,D,H,components,xi,strikes,n,power):
    x,w=gauss(n,4); r,wr=gauss(224,8)
    x=x*D; w=w*xi(T+x)
    kernel=(x[:,None]+T*r)**(H-.5)*np.sqrt(wr*T)
    # SVD of the Brownian loading avoids loss of tiny positive eigenvalues.
    u,s,_=np.linalg.svd(kernel,full_matrices=False)
    ok=s>s[0]*1e-13
    A=u[:,ok]*s[ok]; v=np.sum(kernel**2,axis=1); g=A[:,0]
    if g.mean()<0:g=-g;A[:,0]=g
    def proxy(z):
        return np.sqrt(sum(weight*(np.exp(amp*z*g-.5*amp**2*g*g)@w) for amp,weight in components))
    K=np.array([0.,*strikes])
    cv=[]
    for k in K:
        lo=-16 if not k else optimize.brentq(lambda z:proxy(z)-k,-16,16)
        cv.append(integrate.quad(lambda z:(proxy(z)-k)*np.exp(-z*z/2)/np.sqrt(2*np.pi),lo,16,epsabs=1e-14)[0])
    est=[]
    for rep in range(12):
        gen=stats.qmc.Sobol(d=A.shape[1],scramble=True,seed=12480+rep)
        total=np.zeros(len(K))
        for batch in range(2**(power-11)):
            z=special.ndtri(np.clip(gen.random(2048),1e-16,1-1e-16))
            fields=z@A.T
            actual=np.sqrt(sum(lam*(np.exp(a*fields-.5*a*a*v)@w) for a,lam in components))
            approx=proxy(z[:,0,None])
            total+=(np.maximum(actual[:,None]-K,0)-np.maximum(approx[:,None]-K,0)).sum(axis=0)
        vals=total/2**power+cv
        est.append([*(100*vals),*[100*iv(vals[0],k,T,c) for k,c in zip(K[1:],vals[1:])]])
    est=np.array(est)
    return dict(nodes=n,power=power,effective_rank=int(ok.sum()),max_discarded_sd=float(np.linalg.norm(s[~ok])),mean=est.mean(axis=0).tolist(),se=(est.std(axis=0,ddof=1)/np.sqrt(len(est))).tolist())
def main():
    data=json.loads((ROOT/"results/vix_options_smile_suite.json").read_text())
    cases=[]
    for key in ["mixed_rbergomi_smile","e14_stress_smile","curved"]:
        if key=="curved":
            lab=json.loads((ROOT/"results/lab_suite.json").read_text())
            c=next(i["arb"] for i in lab if i["name"]=="curved")
            rows=c["options"]; T=.25; D=1/12; comps=[(1.9*np.sqrt(.2),1)]
            F=[c["future_L_volpts"],c["future_U_volpts"]];xi=lambda t:.234**2*(1+t)**2
        else:
            c=data[key]; rows=c["strikes"]; T=c["T"];D=c["Delta"]
            F=[c["condvar_future_L"],c["condvar_future_U"]]
            comps=[(1.4,.3),(.7,.7)] if key.startswith("mixed") else [(3*np.sqrt(.2),1.)]
            xi=lambda t:.235**2
        strikes=[r["K_decimal"] for r in rows]
        bands=[F,*[[r["call_L_volpts"],r["call_U_volpts"]] for r in rows],*[[r["iv_L_pct"],r["iv_U_pct"]] for r in rows]]
        runs=[estimate(T,D,.1,comps,xi,strikes,n,p) for n,p in [(48,14),(96,17)]]
        fine=np.array(runs[-1]["mean"]);lo,hi=np.array(bands).T
        cases.append(dict(name=key,status="diagnostic",scrambles=12,order="forward, calls, Black IV%",strikes=strikes,bands=bands,runs=runs,contained=((lo<=fine)&(fine<=hi)).tolist()))
        (ROOT/"results/qmc_check.json").write_text(json.dumps(cases,indent=2))
        print(key,cases[-1]["contained"],flush=True)
if __name__=="__main__":
    with threadpool_limits(limits=1):main()

