"""Independent point-kernel projections for the two unresolved call comparisons."""
from pathlib import Path
import json
from flint import arb
from arb_vix_options import build_condvar_arb_model,make_flat_curve,condvar_vix_futures_and_options_arb
from markov_suite import interval_diff
from factor_decisions import tick,ix
root=Path(__file__).resolve().parents[1]
def main():
    prior=json.loads((root/"results/option_thresholds.json").read_text())
    cases=[(5,0),(1,1)]
    base=[next(x for x in prior if x["H"]==".05" and x["N"]==n) for n,j in cases]
    refs=[r["target"][j] for r,(n,j) in zip(base,cases)]
    out=[]
    for f in [".3381",".4517",".2523",".5619",".1937"]:
        H,T,D=arb(".05"),arb(1)/4,arb(1)/12
        args=(H,T,D,[(arb(1),arb(1))],make_flat_curve(".235"))
        mod=build_condvar_arb_model(*args,M=280,conditioning_fraction=f)
        prices=condvar_vix_futures_and_options_arb(mod,[.18,.22],n_core=440,n_tail_each=64)
        for i,(o,(n,j),r) in enumerate(zip(prices["options"],cases,base)):
            new=[o["call_L_volpts"],o["call_U_volpts"]];refs[i]=ix(refs[i],new)
            err=interval_diff(r["model"][j],refs[i])
            out.append(dict(H=".05",N=n,strike_index=j,conditioning_fraction=f,target=refs[i],model=r["model"][j],error=err,state=tick(err)))
        (root/"results/strike_directions.json").write_text(json.dumps(out,indent=2))
        print(f,[(r["error"],r["state"]) for r in out[-2:]],flush=True)
        if all(r["state"]!="unresolved" for r in out[-2:]):break
if __name__=="__main__":main()

