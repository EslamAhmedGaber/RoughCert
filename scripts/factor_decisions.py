"""Consolidate all separately certified comparisons at fixed fitting specifications."""
import json
from markov_suite import ROOT,interval_diff
from band_entry import location
def load(name):return json.loads((ROOT/"results"/(name+".json")).read_text())
def ix(a,b):
    z=[max(a[0],b[0]),min(a[1],b[1])]
    assert z[0]<=z[1]
    return z
def tick(e):return "within" if max(map(abs,e))<=.05 else "outside" if e[0]>.05 or e[1]<-.05 else "unresolved"
def first(states,winner):
    n=next(i for i,s in states if s==winner)
    undecided=[i for i,s in states if i<n and s=="unresolved"]
    return dict(first_proved=n,earlier_unresolved=undecided,minimum_proved=not undecided)
def main():
    seq=load("factor_thresholds"); refined=load("option_thresholds"); contrast=load("tilted_conditioning")
    edge=load("factor_boundary");entry=load("band_entry");extra=load("band_refine")
    out={}
    for h in [".05",".10",".30"]:
        rows=[]; band=[]
        for r in [x for x in seq if x["H"]==h]:
            n=r["N"];e=r["signed_error"][0]
            if h==".05":
                for t in [*contrast,edge]:
                    if t["N"]==n:e=ix(e,t["error"])
            ce=r["signed_error"][1:]
            for t in refined:
                if t["N"]==n and t["H"]==h:ce=[ix(a,b) for a,b in zip(ce,t["signed_error"])]
            rows.append(dict(N=n,futures_error=e,call_errors=ce,states=[tick(x) for x in [e,*ce]]))
        for e in entry:
            if e["H"]!=h:continue
            z=next((x for x in extra if x["H"]==h and x["N"]==e["N"]),e)
            band.append(dict(N=e["N"],model=z["model"],target=e["target"],state=location(z["model"],e["target"])))
        threshold=[first([(x["N"],x["states"][j]) for x in rows],"within") for j in range(4)]
        out[h]=dict(strikes_points=[18,22,26],tick_thresholds_future_then_calls=threshold,
                fixed_band_threshold=first([(x["N"],x["state"]) for x in band],"inside"),tick_evidence=rows,band_evidence=band)
    (ROOT/"results/factor_decisions.json").write_text(json.dumps(out,indent=2))
    print([(h,x["tick_thresholds_future_then_calls"],x["fixed_band_threshold"]) for h,x in out.items()])
if __name__=="__main__":main()

