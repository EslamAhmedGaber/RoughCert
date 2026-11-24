"""BCJ Section 5.3's 30-day window on ACT/365, at all six maturities and three curves."""
import json
from time import perf_counter
from flint import arb
from quantize_suite import ROOT,ALLOC,spatial,paths,price,set_dps
from arb_vix_options import build_condvar_arb_model,condvar_vix_futures_and_options_arb,make_flat_curve,make_quadratic_curve,make_sqrt_curve
from markov_suite import interval_diff
def main():
    set_dps(50);records=[];H=arb(".1");D=arb(30)/365;a=arb("1.9")*(2*H).sqrt()
    curves={"flat":make_flat_curve(".234"),"quad":make_quadratic_curve(".234"),"sqrt":make_sqrt_curve(".234")}
    for mo in [1,2,3,6,9,12]:
        T=arb(mo)/12;start=perf_counter();basis=spatial(T,D,H,a,96);basis_s=perf_counter()-start
        targets={}
        for name,curve in curves.items():
            rv=condvar_vix_futures_and_options_arb(build_condvar_arb_model(H,T,D,[(a,arb(1))],curve,M=140),[],n_core=260,n_tail_each=45)
            targets[name]=[rv["future_L_volpts"],rv["future_U_volpts"]]
        for budget,allocation in ALLOC.items():
            start=perf_counter();cache=paths(basis,allocation);setup=perf_counter()-start
            for name,curve in curves.items():
                start=perf_counter();F=price(basis,cache,T,D,curve,[])[0];elapsed=perf_counter()-start
                records.append(dict(curve=name,months=mo,window="30/365",H=".10",amplitude="1.9*sqrt(2H)",budget=budget,allocation=allocation,actual_paths=len(cache[1]),future=F,target=targets[name],signed_future_error=interval_diff(F,targets[name]),basis_setup_s=basis_s,product_setup_s=setup,online_s=elapsed,status="Arb-certified"))
            (ROOT/"results/bcj_calendar.json").write_text(json.dumps(records,indent=2))
        print(mo,len(records),flush=True)
if __name__=="__main__":main()

