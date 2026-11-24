"""Test the ordinary kernel mean as a second tilt for disputed band entries."""
import json
from tilted_conditioning import ROOT,arb,solve
from band_entry import location
def main():
    rows=json.loads((ROOT/"results/band_entry.json").read_text()); result=[]
    for r in rows:
        if r["state"]!="unresolved":continue
        p=r["model"]; variants=[]
        for c in ["1",".999",".996"]:
            z=solve(arb(r["H"]),arb(1)/4,arb(1)/12,r,M=360,scale=c)
            p=[max(p[0],z[0]),min(p[1],z[1])]
            assert p[0]<=p[1]
            variants.append(dict(c=c,model=z))
            if location(p,r["target"])!="unresolved":break
        result.append({**r,"model":p,"state":location(p,r["target"]),"additional_tilts":variants})
        (ROOT/"results/band_refine.json").write_text(json.dumps(result,indent=2))
        print(r["H"],r["N"],p,result[-1]["state"],flush=True)
if __name__=="__main__":main()

