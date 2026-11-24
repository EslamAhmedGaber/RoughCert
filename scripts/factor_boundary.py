from tilted_conditioning import solve,ROOT,arb,interval_diff,json
def main():
    H=arb(".05");T=arb(1)/4;D=arb(1)/12
    records=json.loads((ROOT/"results/factor_thresholds.json").read_text())
    r=next(x for x in records if x["H"]==".05" and x["N"]==14)
    for M in [360,560]:
        target=solve(H,T,D,M=M); model=solve(H,T,D,r,M=M)
        err=interval_diff(model,target)
        out=dict(status="Arb-certified",N=14,H=".05",spatial=M,target=target,model=model,error=err)
        (ROOT/"results/factor_boundary.json").write_text(json.dumps(out,indent=2))
        print(out,flush=True)
        if max(map(abs,err))<.05:break
if __name__=="__main__":main()

