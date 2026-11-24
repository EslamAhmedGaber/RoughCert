"""Compare plotted Matplotlib coordinates and band vertices with numerical source rows."""
import json,math,re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import fitz
import build_v3_figures as gen
from v3_comparisons import data,RES,scoreboard
from audit_v3_tables import legacy
checks=[]
def require(label,ok):
    assert ok,label
    checks.append(label)
def eq(a,b):
    return np.shape(a)==np.shape(b) and np.allclose(a,b,rtol=1e-12,atol=1e-13)
def line(ax,x,y,label):
    require(label,any(eq(z.get_xdata(),x) and eq(z.get_ydata(),y) for z in ax.lines))
def band(ax,x,lo,hi,label):
    for artist in ax.collections:
        for poly in artist.get_paths():
            points=poly.vertices
            if all(any(eq(p,(t,y)) for p in points) for t,l,u in zip(x,lo,hi) for y in (l,u)):
                checks.append(label);return
    raise AssertionError(label)
def marker(ax,x,y,label):line(ax,[x],[y],label)
def capture(fig,out,stem):
    path=out/(stem+'.pdf')
    with fitz.open(path) as doc:
        require(stem+' vector 6.50in',len(doc)==1 and abs(doc[0].rect.width-468)<.001 and bool(doc[0].get_fonts()))
    return fig.axes
def cols(rows,key):return [r[key] for r in rows]
def mid(z):return sum(z)/2
def main():
    gen.save_figure=capture; gen.apply_style()
    a,b=gen.build_fig1();r=legacy('E5c_certified_cantelli','e5c_rates')
    for h in [.1,.3]:
        z=sorted((i for i in r if i['H']==h),key=lambda i:i['P'])
        for k in ('B_c','B_d','pred_c','pred_d'):line(b,cols(z,'P'),cols(z,k),f'F1/{h}/{k}')
    f=data('vix_futures_full_suite');a,b=gen.build_fig2()
    for ax,design,var in [(a,'fig1_a_sweep','a'),(b,'fig2_H_sweep','H')]:
        z=f['block_b_bdg_sweeps'][design]
        for k in ['L_minus_LN','bdg3_minus_U','tilted_width']:
            line(ax,cols(z,var),cols(z,k),'F2/'+design+'/'+k)
            require('all plotted points on scale '+design+k,min(cols(z,k))>=ax.get_ylim()[0])
    a,b=gen.build_fig3();r=next(z for z in data('week1_gates_results')['gate2']['cells'] if z['id']=='BD2022_MLMC_JCF');g=r['grid500_tilted_arb']
    for y,z in [(2.6,[r['tilted_L_outer'],r['tilted_U_outer']]),(1.75,[g['L_outer_volpts'],g['U_outer_volpts']])]:line(a,z,[y,y],'F3/interval/'+str(y))
    q=data('quantize_suite')
    for curve in ['flat','quad','sqrt']:
        r=sorted((z for z in f['block_c_bcj2023_18cells'] if z['curve']==curve),key=lambda z:z['T_months']);t=cols(r,'T_months')
        line(b,t,[(z['condvar_L']+z['condvar_U'])/2 for z in r],'F3/continuum/'+curve)
        qr=sorted((z for z in q if z['curve']==curve and z['budget']==1000),key=lambda z:z['months'])
        line(b,t,[mid(z['future']) for z in qr],'F3/BCJ/'+curve)
        if curve=='quad':band(b,t,[z['jmm_lo'][0] for z in r],[z['jmm_up'][1] for z in r],'F3/JMM')
    axs=gen.build_fig4();o=data('vix_options_smile_suite');mix=data('mixed_split')['rows']
    for i,(ax,key) in enumerate(zip(axs,['bdg2023_fig6','e14_stress_smile','mixed_rbergomi_smile'])):
        z=o[key]['strikes'];t=cols(z,'log_moneyness');ref=data('refinement')['grid']['options'] if i==1 else z
        band(ax,t,cols(ref,'iv_L_pct'),cols(ref,'iv_U_pct'),'F4/band/'+key)
        line(ax,t,[mid(row['split_iv'] if i==2 else row['bdg_iv3_pct']) for row in (mix if i==2 else z)],'F4/BDG/'+key)
        if i==1:line(ax,t,data('lab_suite')[-2]['lab_iv'],'F4/LAB')
        if i==2:line(ax,t,cols(mix,'code_iv'),'F4/FD')
    a,b=gen.build_fig5();midpt=np.array([mid(z['target_iv']) for z in mix]);x=t
    band(a,x,[z['target_iv'][0]-v for z,v in zip(mix,midpt)],[z['target_iv'][1]-v for z,v in zip(mix,midpt)],'F5/mixed band')
    for key in ('code_iv','split_iv'):
        line(a,x,[(mid(z[key]) if isinstance(z[key],list) else z[key])-v for z,v in zip(mix,midpt)],'F5/'+key)
    d=data('factor_decisions');extra=data('v3_factor_extension')
    for h,r in d.items():
        rr={v['N']:v for v in extra if v['H']==h};rr.update({v['N']:v for v in r['band_evidence']});te={v['N']:v['futures_error'] for v in r['tick_evidence']};lo=[];hi=[];ns=range(1,20)
        for n in ns:
            z=rr[n];l=z['model'][0]-z['target'][1];u=z['model'][1]-z['target'][0]
            if n in te:l=max(l,te[n][0]);u=min(u,te[n][1])
            assert l<=u;lo.append(max(0,l,-u));hi.append(max(-l,u))
        mm=(np.array(lo)+hi)/2;line(b,ns,mm,'F5/19 means/'+h);band(b,ns,lo,hi,'F5/19 intervals/'+h)
        for n in [r['fixed_band_threshold']['first_proved'],r['tick_thresholds_future_then_calls'][0]['first_proved']]:marker(b,n,mm[n-1],'F5/threshold/'+h+'/'+str(n))
    a,b=gen.build_fig6();s=scoreboard();ref=s[0][0]
    for i,(_,_,z) in enumerate(s):
        text=a.get_yticklabels()[i].get_text().replace("−","-")
        if z and i:
            ivlo,ivhi=map(float,re.search(r"IV \[([+-][\d.]+), ([+-][\d.]+)\]",text).groups())
            target=s[0][2]
            require("F6/signed IV offset "+str(i),0<=z[0]-target[1]-ivlo<.01 and 0<=ivhi-(z[1]-target[0])<.01)
    for i,(F,_,_) in enumerate(s):
        if F is None:require('F6/LAB IV only',any('IV only' in t.get_text() for t in a.texts));continue
        if i in (0,7):marker(a,(F[1]-F[0])/2,7-i,'F6/half width/'+str(i));continue
        lower=F[0]-ref[1];upper=F[1]-ref[0];lo=max(0,lower,-upper);hi=max(-lower,upper)
        require('F6/absolute interval/'+str(i),any(eq(seg,[[lo,7-i],[hi,7-i]]) for c in a.collections if isinstance(c,LineCollection) for seg in c.get_segments()))
    budgets=[50,100,500,1000];ns=[];lo=[];hi=[];stress=[];curve=[]
    for n in budgets:
        rows=[r for r in q if r['budget']==n]; st=next(r for r in rows if r['curve']=='stress');ns.append(st['actual_paths']);stress.append(-mid(st['signed_future_error']))
        cells=[r for r in rows if r['curve']!='stress'];lo.append(min(-v['signed_future_error'][1] for v in cells));hi.append(max(-v['signed_future_error'][0] for v in cells));curve.append(-mid(next(r['signed_future_error'] for r in cells if r['curve']=='quad' and r['months']==3)))
    band(b,ns,lo,hi,'F6/18-cell envelope')
    for label,series in [('min',lo),('max',hi),('stress',stress),('curved',curve)]:line(b,ns,series,'F6/BCJ/'+label)
    axs=gen.build_fig_s1();r=legacy('C03_sandwich_bound','c03_results_closed')+legacy('C03_sandwich_bound','c03_results_generic')
    for ax,name in zip(axs,['WickExp(eta=1.0)','tanh(2y)','PositivePart(c=0.5)','Indicator(c=0.5)']):
        for i,z in enumerate(sorted((v for v in r if v['f']==name),key=lambda v:(v['H'],v['rank']))):
            for k in ('lower_over_P','upper_over_P'):
                if math.isfinite(z[k]):marker(ax,z[k],5-i,f'S1/{name}/{i}/{k}')
                else:require('S1/infinite upper arrow/'+str(i),sum(hasattr(t,'arrow_patch') for t in ax.texts)>=6)
    a,b=gen.build_fig_s2();r=legacy('E5_bias_certified_mc','e5_results_A')
    for h in (.1,.3,.5,.75):
        z=sorted((v for v in r if v['H']==h),key=lambda v:v['n'])
        for k in ('coverage_cert','coverage_naive','median_width_cert','median_width_naive'):line(a if k.startswith('coverage') else b,cols(z,'n'),cols(z,k),f'S2/{h}/{k}')
    axs=gen.build_fig_s3()
    for ax,h in zip(axs,(.1,.3,.5,.75)):
        for fld,fn,k in [('E5b_conditional_certificates','e5b_results','width_a'),('E5b_conditional_certificates','e5b_results','width_b'),('E5c_certified_cantelli','e5c_results','width_c'),('E5c_certified_cantelli','e5c_results','width_d'),('E5c_certified_cantelli','e5c_binomial','width_dbin')]:
            z=sorted((v for v in legacy(fld,fn) if v['H']==h),key=lambda v:v['n'])
            line(ax,cols(z,'n'),cols(z,k),f'S3/{h}/{k}')
    plt.close('all');(RES.parent/'audit/v3_figure_checks.json').write_text(json.dumps(checks,indent=1));print(len(checks),'figure checks')
if __name__=='__main__':main()

