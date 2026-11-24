"""Sequential factor-order audit, fixed optimizer initialisation and fixed price tolerances."""
import json,time
from pathlib import Path
from flint import arb
from markov_suite import finite_model,interval_diff
from markov_optimized import optimized
from arb_vix_options import make_flat_curve,condvar_vix_futures_and_options_arb
ROOT=Path(__file__).resolve().parents[1]
def bands(c):
    return [[c["future_L_volpts"],c["future_U_volpts"]],*[[r["call_L_volpts"],r["call_U_volpts"]] for r in c["options"]]]
def main():
    previous=json.loads((ROOT/"results/markov_optimized.json").read_text())
    rows=[]
    for h in [".05",".10",".30"]:
        known={r["N"]:r for r in previous if r["H"]==h and r["curve"]=="flat"}
        target=known[1]["target"];tb=bands(target)
        H=arb(h);T=arb(1)/4;D=arb(1)/12
        hits=[False]*4
        for n in range(1,23):
            record=known.get(n,{})
            ms=[240,360] if record else [160,300,420]
            states=record.get("tick_states",["unresolved"]*4)
            cv=record.get("model")
            for M in ms:
                if cv and all(h or s!="unresolved" for h,s in zip(hits,states)):break
                nodes,weights=optimized(H,n,T+D)
                mod=finite_model(H,T,D,arb(1),make_flat_curve(".235"),nodes,weights,M)
                cv=condvar_vix_futures_and_options_arb(mod,[.18,.22,.26],n_core=360,n_tail_each=52)
                err=[interval_diff(x,y) for x,y in zip(bands(cv),tb)]
                states=["within" if max(abs(l),abs(u))<.05 else "outside" if l>.05 or u<-.05 else "unresolved" for l,u in err]
                record=dict(nodes=list(map(str,nodes)),weights_unit_a=list(map(str,weights)))
            rows.append(dict(H=h,N=n,model=cv,target=target,states=states,nodes=record["nodes"],weights=record["weights_unit_a"],signed_error=[interval_diff(x,y) for x,y in zip(bands(cv),tb)],model_enclosed_in_target=[y[0]<=x[0] and x[1]<=y[1] for x,y in zip(bands(cv),tb)]))
            (ROOT/"results/factor_thresholds.json").write_text(json.dumps(rows,indent=2,allow_nan=False))
            hits=[b or s=="within" for b,s in zip(hits,states)]
            print(h,n,states,flush=True)
            if all(hits):break
if __name__=="__main__":main()

