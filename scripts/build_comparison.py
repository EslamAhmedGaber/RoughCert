"""Build outward-rounded comparison tables and a vector figure from final JSON outputs."""
import math,re,json
from decimal import Decimal,ROUND_CEILING,ROUND_FLOOR
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from arb_vix_core import arb,arb_interval
from arb_vix_options import black76_iv_bracket_arb
ROOT=Path(__file__).resolve().parents[1]
M=ROOT/"manuscript"
def read(s):return json.loads((ROOT/"results"/(s+".json")).read_text())
def num(x,n,up):
    return format(Decimal.from_float(float(x)).quantize(Decimal(1).scaleb(-n),rounding=ROUND_CEILING if up else ROUND_FLOOR),"f")
def band(x,n=3):return "["+num(x[0],n,False)+", "+num(x[1],n,True)+"]"
def table(key,title,cols,rows):
    text=[r"\begin{table}[!htbp]\centering\small",r"\caption{"+title+r"}\label{tab:"+key+"}",r"\begin{tabular}{@{}l"+"c"*(len(cols)-1)+r"@{}}\toprule"," & ".join(cols)+r" \\\midrule"]
    text.extend(" & ".join(row)+r" \\" for row in rows)
    text.extend([r"\bottomrule\end{tabular}",r"\end{table}"])
    (M/"tables"/("tab_"+key+".tex")).write_text("\n".join(text)+"\n")
def forward(c):return [c["future_L_volpts"],c["future_U_volpts"]]
def call(o):return [o["call_L_volpts"],o["call_U_volpts"]]
def iv_sheet(F,C,K):
    iv=black76_iv_bracket_arb(arb(F[0])/100,arb(F[1])/100,arb(str(K)),arb(1)/4,arb(C[0])/100,arb(C[1])/100)
    return [100*iv[0],100*iv[1]]
def main():
    lab=read("lab_suite");mix=read("mixed_split");refs=read("refinement")
    q=read("quantize_suite");mark=read("markov_optimized");comp=read("curved_comparison");exp=read("curved_expansions")
    curve=lab[-1]; k=1; bench=curve["arb"];o=bench["options"][k];strike=o["K_decimal"]
    bb=next(r["model"] for r in mark if r["curve"]=="curved" and r["N"]==12)
    qc=next(r for r in q if r["months"]==3 and r["curve"]=="quad" and r["budget"]==1000)
    grid=comp["grid500"];gopt=grid["options"][k];ln=comp["moment_matched"]
    def cvrow(name,c):
        o=c["options"][k]
        return [name,band(forward(c)),band(call(o)),band([o["iv_L_pct"],o["iv_U_pct"]],2)]
    rows=[cvrow("Continuous model",bench),
          ["BDG weak price",band(exp["forward"]),band(exp["rows"][k]["bdg_call"]),band(exp["rows"][k]["bdg_iv"],2)],
          ["LAB direct IV",r"---",r"---",band(exp["rows"][k]["lab_iv"],2)],
          cvrow("Markovian, 12 factors",bb),
          ["BCJ, 960 paths",band(qc["future"]),band(qc["calls"][k]),band(iv_sheet(qc["future"],qc["calls"][k],strike),2)],
          ["JMM bounds",band([comp["jmm"]["jmm_lo_volpts"][0],comp["jmm"]["jmm_up_volpts"][1]]),r"---",r"---"],
          ["Moment-matched lognormal",band(ln["ln_F_volpts"]),band(ln["options"][k]["ln_call_volpts"]),band(ln["ln_iv_pct"],2)],
          ["Right grid, 500 dates",band(forward(grid)),band(call(gopt)),band(iv_sheet(forward(grid),call(gopt),strike),2)]]
    table("common","Three-month comparison on the upward-sloping forward-variance curve. Price units are index points. Each IV uses its own forward and the same strike of 27 points. All displayed intervals are outward rounded.",["Method","Future","Call",r"Black IV (\%)"],rows)
    rows=[]
    for row in lab[:-1]:
        counts=sum(b[1]<o["iv_L_pct"] for b,o in zip(row["lab_arb_iv"],row["arb"]["options"]))
        gaps=[o["iv_L_pct"]-b[1] for b,o in zip(row["lab_arb_iv"],row["arb"]["options"])]
        rows.append(["Stress" if row["name"]=="stress" else row["H"],str(row["T_months"]),str(counts)+"/"+str(len(gaps)),num(max(gaps),3,False)])
    table("direct","Direct IV expansion below the continuous-model lower bound at fixed strikes. The last column gives the largest certified shortfall in percentage points.",["$H$","Months","Separated","Largest shortfall"],rows)
    rows=[]
    for b in [50,100,500,1000]:
        r=[z for z in q if z["budget"]==b and z["curve"]!="stress"]
        st=next(z for z in q if z["budget"]==b and z["curve"]=="stress")
        rows.append([str(b),str(st["actual_paths"]),band([min(z["signed_future_error"][0] for z in r),max(z["signed_future_error"][1] for z in r)]),band(st["signed_future_error"]),f"{np.median([z['basis_setup_s']+z['product_setup_s'] for z in r]):.2f}",f"{np.median([z['online_s'] for z in r]):.2f}"])
    table("quantiser","Signed product-quantiser futures errors over 18 cells with $\\Delta=1/12$ and one 30-day stress cell. Times are medians in seconds, with spatial basis and tensor-grid preparation included in setup.",["Budget","Paths","Literature range","Stress error","Setup","Price"],rows)
    legacy=read("vix_options_smile_suite");allr=read("qmc_check")
    totals=sum(len(c["contained"]) for c in allr); hits=sum(sum(c["contained"]) for c in allr)
    assert totals==35 and hits==35
    values={"SimChecks":str(hits),"StressATMOld":num(legacy["e14_stress_smile"]["strikes"][2]["iv_width_pct"],3,True),"StressATMNew":num(refs["grid"]["options"][2]["iv_width_pct"],3,True)}
    (M/"tables/v2_numbers.tex").write_text("\n".join(r"\newcommand{\%s}{%s}"%(k,v) for k,v in values.items())+"\n")
    fig,(ax,ay)=plt.subplots(1,2,figsize=(7.0,2.9),layout="constrained")
    z=np.linspace(-.3,.3,7)
    rr=mix["rows"];center=np.array([np.mean(i["target_iv"]) for i in rr])
    ax.fill_between(z,[i["target_iv"][0]-m for i,m in zip(rr,center)],[i["target_iv"][1]-m for i,m in zip(rr,center)],color="#9ecae1",label="Model interval")
    ax.plot(z,[i["code_iv"]-m for i,m in zip(rr,center)],"s--",ms=3,color="#D55E00",label="100-node code")
    ax.plot(z,[np.mean(i["split_iv"])-m for i,m in zip(rr,center)],"o-",ms=3,color="#003f5c",label="Root split")
    ax.set(xlabel="Log-moneyness",ylabel="IV minus band midpoint (pp)",title="Mixed smile")
    ax.legend(fontsize=7,loc="lower right");ax.grid(axis="y",alpha=.2)
    for h,color,marker in [(".05","#D55E00","s"),(".10","#0072B2","o"),(".30","#009E73","^")]:
        rr=[i for i in mark if i["curve"]=="flat" and i["H"]==h]
        x=[i["N"] for i in rr];lo=np.array([i["error_volpts"][0][0] for i in rr]);hi=np.array([i["error_volpts"][0][1] for i in rr]);mid=(hi+lo)/2
        ay.errorbar(x,mid,yerr=np.stack([mid-lo,hi-mid]),marker=marker,ms=3,color=color,lw=1,capsize=2,label="H = 0"+h)
    ay.axhspan(-.05,.05,color="#cccccc",alpha=.45);ay.set(xlabel="Number of factors",ylabel="Signed futures error (points)",title="Optimised kernel approximation",xticks=[1,3,5,8,12]);ay.grid(axis="y",alpha=.2);ay.legend(fontsize=7,loc="lower right")
    fig.savefig(M/"figures/fig_method_errors.pdf");plt.close(fig)
if __name__=="__main__":main()

