"""Refine only a strike for which a smaller factor count could meet the tolerance."""
import json
from arb_vix_core import arb,set_dps
from arb_vix_options import make_flat_curve,build_condvar_arb_model,condvar_vix_futures_and_options_arb as price
from markov_suite import ROOT,finite_model,interval_diff
def bands(r):
    return [[x["call_L_volpts"],x["call_U_volpts"]] for x in r["options"]]
def intersect(a,b):return [max(a[0],b[0]),min(a[1],b[1])]
def state(e):
    if max(map(abs,e))<=.05:return "within"
    if e[0]>.05 or e[1]<-.05:return "outside"
    return "unresolved"
def main():
    set_dps(50)
    data=json.loads((ROOT/"results/factor_thresholds.json").read_text()); results=[]
    for label in [".30",".10",".05"]:
        rows=[r for r in data if r["H"]==label]
        H,T,D=arb(label),arb(1)/4,arb(1)/12
        curve=make_flat_curve(".235");strikes=[.18,.22,.26]
        if not any("unresolved" in r["states"][1:] for r in rows):continue
        refined=bands(price(build_condvar_arb_model(H,T,D,[(arb(1),arb(1))],curve,M=400,grade_p=3),strikes,n_core=540,n_tail_each=76))
        hits=[False]*3
        for r in rows:
            n=r["N"];target=[intersect(x,y) for x,y in zip(refined,bands(r["target"]))]
            vals=bands(r["model"]); st=[state(interval_diff(x,y)) for x,y in zip(vals,target)]
            if any(s=="unresolved" and not hit for s,hit in zip(st,hits)):
                nodes=list(map(arb,r["nodes"]));weights=list(map(arb,r["weights"]))
                for m in [290,460]:
                    mod=finite_model(H,T,D,arb(1),curve,nodes,weights,m)
                    vv=bands(price(mod,strikes,n_core=540,n_tail_each=76))
                    vals=[intersect(x,y) for x,y in zip(vals,vv)]
                    st=[state(interval_diff(x,y)) for x,y in zip(vals,target)]
                    if all(hit or s!="unresolved" for s,hit in zip(st,hits)):break
            err=[interval_diff(x,y) for x,y in zip(vals,target)]
            results.append(dict(H=label,N=n,strikes=strikes,target=target,model=vals,signed_error=err,states=st))
            print(label,n,st,flush=True)
            (ROOT/"results/option_thresholds.json").write_text(json.dumps(results,indent=2))
            hits=[h or s=="within" for h,s in zip(hits,st)]
            if all(hits):break
if __name__=="__main__":main()

