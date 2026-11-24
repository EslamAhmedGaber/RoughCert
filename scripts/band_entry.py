"""Entry into the original fixed rough-model futures interval for the optimised kernel sequence."""
import json
from tilted_conditioning import solve,arb,ROOT
from markov_optimized import optimized

def location(interval, target):
    l,u=interval
    if u<target[0] or l>target[1]: return "outside"
    if target[0]<=l and u<=target[1]: return "inside"
    return "unresolved"

def main():
    cached=json.loads((ROOT/"results/markov_optimized.json").read_text())
    seq=json.loads((ROOT/"results/factor_thresholds.json").read_text())
    results=[]
    for h in [".30",".10",".05"]:
        H,T,D=arb(h),arb(1)/4,arb(1)/12
        base=next(r["target"] for r in cached if r["H"]==h and r["curve"]=="flat")
        target=[base["future_L_volpts"],base["future_U_volpts"]]
        for n in range(1,29):
            found=[r for r in seq if r["N"]==n and r["H"]==h]
            found+= [r for r in cached if r["N"]==n and r["H"]==h and r["curve"]=="flat"]
            if found:
                r=found[0]
                nodes=r["nodes"]; weights=r.get("weights",r.get("weights_unit_a"))
                price=[max(s["model"]["future_L_volpts"] for s in found),min(s["model"]["future_U_volpts"] for s in found)]
            else:
                a,b=optimized(H,n,T+D)
                nodes,weights=list(map(str,a)),list(map(str,b))
                price=[0,100]
            grids=[]
            for m in [135,260,450,640]:
                if location(price,target)!="unresolved":break
                z=solve(H,T,D,{"nodes":nodes,"weights":weights},M=m)
                price=[max(price[0],z[0]),min(price[1],z[1])]
                assert price[0]<=price[1]
                grids.append(m)
            state=location(price,target)
            results.append(dict(H=h,N=n,target=target,model=price,state=state,contrast_mesh=grids,nodes=nodes,weights=weights))
            (ROOT/"results/band_entry.json").write_text(json.dumps(results,indent=2))
            print(h,n,price,state,flush=True)
            if state=="inside":break
if __name__=="__main__":main()

