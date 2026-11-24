"""Forward-field KL/Nystrom comparator, distinct from BCJ's Brownian cosine expansion."""
import json
from itertools import product
from pathlib import Path
from time import perf_counter
from math import prod
import numpy as np
from qmc_check import gauss
from quantize_suite import ALLOC,grid
from arb_vix_core import set_dps
ROOT=Path(__file__).resolve().parents[1]
def main():
    set_dps(50)
    refs=json.loads((ROOT/"results/quantize_suite.json").read_text())
    out=[]
    for count in [64,112]:
        for mo,is_stress in [(i,False) for i in [1,2,3,6,9,12]]+[(6,True)]:
            start=perf_counter()
            T=mo/12;D=30/365 if is_stress else 1/12
            amp=(3 if is_stress else 1.9)*np.sqrt(.2)
            u,w=gauss(count,4);x=D*u
            r,wr=gauss(256,8)
            A=amp*(x[:,None]+T*r)**(-.4)*np.sqrt(wr*T)
            U,s,_=np.linalg.svd(np.sqrt(w)[:,None]*A,full_matrices=False)
            modes=U[:,:6]*s[:6]/np.sqrt(w)[:,None]
            diag=amp**2*((T+x)**.2-x**.2)/.2
            basis_time=perf_counter()-start
            for n,alloc in ALLOC.items():
                start=perf_counter()
                indices=np.array(list(product(*(range(a) for a in alloc))))
                zs=np.zeros(indices.shape);prob=np.ones(len(indices))
                for j,sz in enumerate(alloc):
                    centers,mass=grid(sz)
                    zs[:,j]=np.array(list(map(float,centers)))[indices[:,j]]
                    prob*=np.array(list(map(float,mass)))[indices[:,j]]
                expf=np.exp(zs@modes[:,:len(alloc)].T-diag/2)
                setup_time=perf_counter()-start
                for name in (["stress"] if is_stress else ["flat","quad","sqrt"]):
                    start=perf_counter()
                    level=.235 if is_stress else .234
                    xi=level**2*(1+T+x)**{"quad":2,"sqrt":.5}.get(name,0)
                    F=100*float(prob@np.sqrt(expf@(w*xi)))
                    ref=next(z for z in refs if z["curve"]==name and z["months"]==mo and z["budget"]==n)
                    out.append(dict(nodes=count,months=mo,curve=name,budget=n,price=F,target=ref["target"],basis_s=basis_time,product_setup_s=setup_time,price_s=perf_counter()-start,status="diagnostic"))
    (ROOT/"results/field_kl.json").write_text(json.dumps(out,indent=2))
if __name__=="__main__":main()

