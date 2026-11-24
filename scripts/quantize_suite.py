"""BCJ (2023), Eq. (14) and Remark 4.9. Certified tensor Gaussian cubature prices."""
import itertools,json,math
from pathlib import Path
from time import perf_counter
from functools import lru_cache
import numpy as np
from scipy import special,optimize
from flint import arb,acb,arb_mat
from arb_vix_core import arb_interval,arb_lo_float,arb_hi_float,set_dps
from arb_vix_options import make_flat_curve,make_quadratic_curve,make_sqrt_curve
ROOT=Path(__file__).resolve().parents[1]
ALLOC={50:[6,4,2],100:[8,3,2,2],500:[10,4,3,2,2],1000:[10,4,3,2,2,2]}

@lru_cache(None)
def grid(n):
    def mapping(c):
        b=np.r_[-np.inf,(c[1:]+c[:-1])/2,np.inf]
        return -np.diff(np.exp(-b*b/2)/np.sqrt(2*np.pi))/np.diff(special.ndtr(b))
    c=optimize.root(lambda z:mapping(z)-z,special.ndtri((np.arange(n)+.5)/n),tol=1e-12).x
    center=[arb(float(x)) for x in c]
    def calc(z):
        b=[(l+r)/2 for l,r in zip(z,z[1:])]
        f=[arb(0),*[(-x/arb(2).sqrt()).erfc()/2 for x in b],arb(1)]
        d=[arb(0),*[(-x*x/2).exp()/(2*arb.pi()).sqrt() for x in b],arb(0)]
        probs=[r-l for l,r in zip(f,f[1:])]
        return [(l-r)/m for l,r,m in zip(d,d[1:],probs)],probs
    for eps in ["1e-12","1e-10","1e-8"]:
        left=[x-arb(eps) for x in center];right=[x+arb(eps) for x in center]
        if all(z>l for z,l in zip(calc(left)[0],left)) and all(z<r for z,r in zip(calc(right)[0],right)):
            # Truncated normal conditional means increase in both thresholds.
            # Brouwer on this ordered box, uniqueness for a strictly log-concave normal law.
            z=[arb_interval(l,r) for l,r in zip(left,right)]
            return z,calc(z)[1]
    raise ArithmeticError("failed to enclose scalar quantizer")
def spatial(T,D,H,a,M):
    p=arb(1)/2-H; b=1-p; L=T+D; gain=a*(2/L).sqrt()
    edge=[D*(arb(i)/M)**3 for i in range(M+1)]
    half=[(v-u)/2 for u,v in zip(edge,edge[1:])]
    mid=[u+h for u,h in zip(edge,half)]
    loads=[]
    for j in range(1,7):
        freq=(arb(j)-arb(1)/2)*arb.pi()/L
        def primitive(t):
            return t**b*acb(0,-freq*t).hypgeom_1f1(b,b+1)/b
        loads.append([gain*(acb(0,freq*(T+x)).exp()*(primitive(T+x)-primitive(x))).real for x in mid])
    def drift(x):return -a*a*((T+x)**(2*H)-x**(2*H))/(4*H)
    diffs=[gain*(v**b-u**b)/b for u,v in zip(edge,edge[1:])]
    return dict(loads=loads,half=half,mid=mid,edge=edge,drift=[drift(x) for x in mid],
                drift_delta=[drift(v)-drift(u) for u,v in zip(edge,edge[1:])],diffs=diffs,
                der1=[(gain*u**(-p),a*a*u**(-2*p)/2) for u in edge[1:-1]],
                der2=[(gain*p*u**(-p-1),a*a*p*u**(-2*p-1)) for u in edge[1:-1]])
def paths(data,allocation):
    m=len(allocation); rules=[grid(n) for n in allocation]
    zs=[];probs=[];abss=[]
    for idx in itertools.product(*[range(n) for n in allocation]):
        zs.append([r[0][i] for i,r in zip(idx,rules)])
        probs.append(math.prod(r[1][i] for i,r in zip(idx,rules)))
        abss.append(sum(x.abs_upper() for x in zs[-1]))
    assert sum(probs).contains(1)
    values=arb_mat(zs)*arb_mat(data["loads"][:m])
    return values,probs,abss
def price(data,cache,T,D,curve,strikes):
    fields,prob,abss=cache
    mid=data["mid"];half=data["half"];edge=data["edge"]
    xis=[curve.eval_arb(T+x) for x in mid]
    boxes=[arb_interval(T+l,T+r) for l,r in zip(edge,edge[1:])]
    curves=[(curve.eval_arb(b),curve.d1_arb(b).abs_upper(),curve.d2_arb(b).abs_upper()) for b in boxes]
    prices=[arb(0) for _ in [0,*strikes]]
    for j,(weight,sz) in enumerate(zip(prob,abss)):
        I=arb(0)
        for i,h in enumerate(half):
            z=fields[j,i]+data["drift"][i]
            rad=(sz*data["diffs"][i]+data["drift_delta"][i]).upper()
            E=(z+arb(0,rad)).exp()
            curvebox,d1,d2=curves[i]
            box=2*h*curvebox*E
            if i:
                a,b=data["der1"][i-1]; e1=a*sz+b
                a,b=data["der2"][i-1]; e2=a*sz+b
                err=h**3/3*E.upper()*(d2+2*d1*e1+curvebox.upper()*(e1**2+e2))
                narrow=2*h*xis[i]*z.exp()+arb(0,err.upper())
                box=box.intersection(narrow)
            I+=box
        V=(I/D).sqrt()
        assert V.is_finite()
        prices[0]+=weight*V
        for k,K in enumerate(strikes,1):prices[k]+=weight*(V-arb(str(K))).max(arb(0))
    return [[arb_lo_float(100*x),arb_hi_float(100*x)] for x in prices]
def main():
    set_dps(50); rows=[]
    refs=json.loads((ROOT/"results/vix_futures_full_suite.json").read_text())["block_c_bcj2023_18cells"]
    stress=json.loads((ROOT/"results/vix_options_smile_suite.json").read_text())["e14_stress_smile"]
    for mo,stressflag in [(i,False) for i in [1,2,3,6,9,12]]+[(6,True)]:
        H=arb(".1");T=arb(mo)/12;D=arb(30)/365 if stressflag else arb(1)/12
        a=(3 if stressflag else arb("1.9"))*(2*H).sqrt()
        start=perf_counter()
        data=spatial(T,D,H,a,96)
        setup0=perf_counter()-start
        for budget,allocation in ALLOC.items():
            start=perf_counter();cache=paths(data,allocation);setup=perf_counter()-start
            curves={"stress":make_flat_curve(".235")} if stressflag else {"flat":make_flat_curve(".234"),"quad":make_quadratic_curve(".234"),"sqrt":make_sqrt_curve(".234")}
            for name,curve in curves.items():
                strikes=[.25,.27,.30] if mo==3 and name=="quad" else []
                start=perf_counter();vals=price(data,cache,T,D,curve,strikes);elapsed=perf_counter()-start
                if stressflag:
                    target=[stress["condvar_future_L"],stress["condvar_future_U"]]
                else:
                    ref=next(r for r in refs if r["curve"]==name and r["T_months"]==mo)
                    target=[ref["condvar_L"],ref["condvar_U"]]
                err=arb_interval(*map(arb,vals[0]))-arb_interval(*map(arb,target))
                rows.append(dict(curve=name,months=mo,budget=budget,allocation=allocation,actual_paths=math.prod(allocation),
                                 status="Arb-certified",future=vals[0],target=target,signed_future_error=[arb_lo_float(err),arb_hi_float(err)],
                                 strikes=strikes,calls=vals[1:],basis_setup_s=setup0,product_setup_s=setup,online_s=elapsed))
            print(mo,stressflag,budget,flush=True)
            (ROOT/"results/quantize_suite.json").write_text(json.dumps(rows,indent=2,allow_nan=False))
if __name__=="__main__":main()

