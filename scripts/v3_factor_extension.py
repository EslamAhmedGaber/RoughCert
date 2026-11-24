"""Complete the factor paths against the original, fixed rough band."""
import json
from band_entry import location
from tilted_conditioning import ROOT, arb, solve
from markov_optimized import optimized

def main():
    res=ROOT/'results'
    existing=json.loads((res/'band_entry.json').read_text())
    known=json.loads((res/'markov_optimized.json').read_text())+json.loads((res/'factor_thresholds.json').read_text())
    out=[]
    for h in ['.10','.30','.05']:
        rows={r['N']:r for r in existing if r['H']==h}
        target=rows[1]['target']
        for n in sorted(set(range(1,20))-rows.keys()):
            found=next((r for r in known if r['H']==h and r['N']==n and r.get('curve','flat')=='flat'),None)
            if found:
                nodes=found['nodes']; weights=found.get('weights',found.get('weights_unit_a'))
                z=[found['model'][k] for k in ['future_L_volpts','future_U_volpts']]
            else:
                H,T,D=arb(h),arb(1)/4,arb(1)/12
                a,b=optimized(H,n,T+D)
                nodes,weights=list(map(str,a)),list(map(str,b))
                z=solve(H,T,D,dict(nodes=nodes,weights=weights),M=135)
            out.append(dict(H=h,N=n,target=target,model=z,state=location(z,target),nodes=nodes,weights=weights))
            print(h,n,z,flush=True)
    (res/'v3_factor_extension.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__': main()

