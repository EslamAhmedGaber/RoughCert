"""Independent high-precision hypergeometric Gram matrix and Cholesky on the delivery grid."""
from pathlib import Path
from functools import lru_cache
import json,math
import numpy as np
import mpmath as mp
from scipy import integrate,stats,special
from threadpoolctl import threadpool_limits
from qmc_check import gauss,iv
root=Path(__file__).resolve().parents[1]

@lru_cache(None)
def factor(tn,td,dn,dd,n):
    x,w=gauss(n,4)
    with mp.workdps(130):
        T=mp.mpf(tn)/td;D=mp.mpf(dn)/dd;p=mp.mpf(2)/5
        xm=[D*mp.mpf(float(y)) for y in x]
        def primitive(u,v):
            if u==v:return u**(1-2*p)/(1-2*p)
            gap=v-u
            return u**(1-p)*gap**(-p)*mp.hyp2f1(p,1-p,2-p,-u/gap)/(1-p)
        C=mp.matrix(n)
        for i,a in enumerate(xm):
            for j in range(i,n):
                b=xm[j];C[i,j]=C[j,i]=primitive(a+T,b+T)-primitive(a,b)
        L=mp.cholesky(C)
        residue=mp.mnorm(C-L*L.T,"inf")/mp.mnorm(C,"inf")
        assert residue < mp.mpf("1e-105")
        # A float rotation concentrates Sobol variance in the first directions.
        diag=np.array([float(C[i,i]) for i in range(n)])
        U,S,_=np.linalg.svd(np.array(L.tolist(),dtype=float),full_matrices=False)
        A=U*S
        if A[:,0].mean()<0:A[:,0]*=-1
        return np.array(list(map(float,xm))),w,A,diag,float(residue)

def main():
    source=json.loads((root/"results/qmc_check.json").read_text());out=[]
    for row in source:
        name=row["name"];stress=name=="e14_stress_smile";curved=name=="curved"
        x,w,A,diag,res=factor(1,2 if stress else 4,30 if stress else 1,365 if stress else 12,40)
        T=.5 if stress else .25
        w=w*(.234**2*(1+T+x)**2 if curved else .235**2)
        comps=[(3*math.sqrt(.2),1)] if stress else [(1.9*math.sqrt(.2),1)] if curved else [(1.4,.3),(.7,.7)]
        K=np.array([0,*row["strikes"]]);g=A[:,0]
        def proxy(z):
            return np.sqrt(sum(l*(np.exp(a*z*g-.5*a*a*g*g)@w) for a,l in comps))
        base=[]
        from scipy.optimize import brentq
        for k in K:
            z=-16 if not k else brentq(lambda y:proxy(y)-k,-16,16)
            base.append(integrate.quad(lambda y:(proxy(y)-k)*math.exp(-y*y/2)/math.sqrt(2*math.pi),z,16,epsabs=1e-13)[0])
        estimates=[]
        for seed in range(12):
            sob=stats.qmc.Sobol(d=A.shape[1],scramble=True,seed=87351+seed)
            accum=np.zeros(len(K))
            for _ in range(64):
                z=special.ndtri(np.clip(sob.random(2048),1e-16,1-1e-16))
                y=z@A.T
                y=np.sqrt(sum(l*(np.exp(a*y-.5*a*a*diag)@w) for a,l in comps))
                c=proxy(z[:,0,None])
                accum+=(np.maximum(y[:,None]-K,0)-np.maximum(c[:,None]-K,0)).sum(axis=0)
            prices=accum/2**17+base
            estimates.append([*(100*prices),*[100*iv(prices[0],k,T,p) for k,p in zip(K[1:],prices[1:])]])
        vals=np.array(estimates);avg=vals.mean(axis=0);lo,hi=np.array(row["bands"]).T
        out.append(dict(name=name,dps=130,delivery_nodes=40,gaussian_dimensions=A.shape[1],scrambles=12,power=17,relative_cholesky_residual=res,mean=avg.tolist(),se=(vals.std(axis=0,ddof=1)/np.sqrt(len(vals))).tolist(),contained=((lo<=avg)&(avg<=hi)).tolist(),status="diagnostic"))
        print(name,sum(out[-1]["contained"]),len(avg),res,flush=True)
        (root/"results/high_precision_qmc.json").write_text(json.dumps(out,indent=2,allow_nan=False))
if __name__=="__main__":
    with threadpool_limits(limits=1):main()

