"""Independent check of the numeric cells in the ten printed TeX tables."""
import json, re, statistics
from decimal import Decimal as D, ROUND_FLOOR, ROUND_CEILING, ROUND_HALF_EVEN
from fractions import Fraction
from pathlib import Path
from v3_comparisons import data, RES, scoreboard

TAB=RES.parent/'manuscript/tables'
def legacy(folder,stem):return json.loads((RES/'legacy_inputs'/folder/(stem+'.json')).read_text())
def take(table,pattern):
    return [[c.strip() for c in line.removesuffix(r'\\').strip().split('&')] for line in (TAB/(table+'.tex')).read_text(encoding='utf8').splitlines() if re.match(pattern,line) and ' & ' in line]
def fields(r,names):return [r[x] for x in names.split()]
def average(z):return sum(z)/2
def interval(z):return [(z[0],'down'),(z[1],'up')]
def up(z):return (z,'up')
def down(z):return (z,'down')
def numbers(c):
    t=re.sub(r'\\multicolumn\{\d+\}\{c\}\{','',c).replace(r'\,',' ').replace('$','').replace(',','')
    t=re.sub(r'\\times\s*10\^\{(-?\d+)\}',r'e\1',t)
    return re.findall(r'[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?',t)
records=[]
def check(label,row,spec):
    assert len(row)==len(spec),(label,row,spec)
    for i,(c,v) in enumerate(zip(row,spec)):
        if v is None:continue # text label; numeric contract identifiers checked separately
        vals=numbers(c);specs=v if isinstance(v,list) else [v]
        assert len(vals)==len(specs),(label,i,c,specs)
        for txt,raw in zip(vals,specs):
            val,mode=raw if isinstance(raw,tuple) else (raw,'nearest')
            actual=D(txt); source=D.from_float(float(val)); quantum=D(1).scaleb(actual.as_tuple().exponent)
            expected=source.quantize(quantum,rounding={'down':ROUND_FLOOR,'up':ROUND_CEILING,'nearest':ROUND_HALF_EVEN}[mode])
            assert actual==expected,(label,i,txt,val,mode,expected)
            records.append({'row':label,'column':i+1,'printed':txt,'raw':val,'rule':mode})
def checkrows(label,rows,specs):
    assert len(rows)==len(specs),(label,len(rows),len(specs))
    for i,(r,s) in enumerate(zip(rows,specs)):check(f'{label}/{i+1}',r,s)

def main():
    records.clear()
    g=data('week1_gates_results')['gate2']['cells']; f=data('vix_futures_full_suite');o=data('vix_options_smile_suite')
    table=take('tab_vix_headline',r'^(BD|BCJ|Stress|Base)');order=[0,5,4,6,7,1,2,3]
    specs=[]
    for idx,r in zip(order,table):
        z=g[idx];assert float(Fraction(r[2]))==z['T'] and float(Fraction(r[3]))==z['Delta']
        specs.append([None,z['H'],None,None,z['a'],interval(fields(z,'tilted_L_outer tilted_U_outer')),up(z['tilted_U_outer']-z['tilted_L_outer']),average(z['bdg_F3_arb']),down(z['bdg_F3_arb'][0]-z['tilted_U_outer']),up(z['ln_fut'][1]-z['tilted_L_outer'])])
    checkrows('T1',table[:-1],specs);a=g[4]['grid500_tilted_arb'];lo,hi=fields(a,'L_outer_volpts U_outer_volpts')
    check('T1/grid',table[-1],[None,g[4]['H'],g[4]['T'],[1,12],g[4]['a'],interval([lo,hi]),up(hi-lo),[o['bd2022_mlmc_call']['published_fut_volpts']]])
    tr=take('tab_vix_e14_15cells',r'^\d')
    for r,z in zip(tr,f['block_a_e14_15cells'],strict=True):
        assert r[1]=={7/365:'1w',1/52:'1w',1/12:'1m',.25:'3m',.5:'6m'}[z['T']]
        L,U=z['best_L'],z['best_U'];wj=z['jmm_up'][1]-z['jmm_lo'][0];w=U-L
        check('T2/'+z['id'],r,[z['H'],None,interval([L,U]),up(w),up(wj),up(U-z['jmm_lo'][0]),wj/w,down(z['bdg_F3'][0]-U) if 'bdg_F3' in z else [],up(z['ln_fut'][1]-L) if 'ln_fut' in z else []])
    specs=[]
    for z in f['block_d_mixed_rbergomi']:
        L,U=fields(z,'condvar_L condvar_U');specs.append([None,fields(z,'a1 a2 lambda'),z['T_months'],interval([L,U]),up(U-L),z['bdg_mixed_order0'],z['bdg_mixed_order3'],down(z['bdg_mixed_order3']-U),fields(z,'setup_s solve_s')])
    checkrows('T3',take('tab_vix_mixed',r'^(Stress|Scenario)'),specs)
    specs=[];ref=data('refinement')['grid']['options'];lab=data('lab_suite');m=data('mixed_split')['rows']
    for group in ['bdg2023_fig6','e14_stress_smile','mixed_rbergomi_smile']:
        for i,z in enumerate(o[group]['strikes']):
            if group=='bdg2023_fig6' and i%2:continue
            if group=='mixed_rbergomi_smile' and i not in (0,1,3,4,6):continue
            stress=group=='e14_stress_smile';mixed=group=='mixed_rbergomi_smile'; t=ref[i] if stress else z
            if stress:assert t['K_decimal']==z['K_decimal']==lab[-2]['arb']['options'][i]['K_decimal']
            if mixed:assert t['K_decimal']==m[i]['K']
            C=fields(t,'call_L_volpts call_U_volpts');IV=fields(t,'iv_L_pct iv_U_pct')
            specs.append([z['log_moneyness'],z['K_volpts'],interval(C),up(C[1]-C[0]),average(m[i]['split_call'] if mixed else z['bdg_C3_volpts']),[] if mixed else average(z['ln_call_volpts']),interval(IV),average(m[i]['split_iv'] if mixed else z['bdg_iv3_pct']),average(lab[-2]['lab_arb_iv'][i]) if stress else [],[] if mixed else average(z['ln_iv_pct'])])
    checkrows('T4',take('tab_vix_smiles',r'^\$[+-]'),specs)
    specs=[]
    for z in lab[:-1]:
        opts=ref if z['name']=='stress' else z['arb']['options'];gaps=[r['iv_L_pct']-a[1] for r,a in zip(opts,z['lab_arb_iv'],strict=True)]
        below=sum(x>0 for x in gaps);overlap=sum(r['iv_U_pct']>=a[0] and r['iv_L_pct']<=a[1] for r,a in zip(opts,z['lab_arb_iv']))
        assert below+overlap==len(opts)
        specs.append([None,float(z['H']),z['T_months'],[below,len(opts)],[overlap,len(opts)],down(max(gaps)),[overlap] if overlap else []])
    checkrows('T5A',take('tab_competitors',r'^(LAB author|Stress)'),specs)
    specs=[];br=data('band_refine')
    for h,d in data('factor_decisions').items():
        for r in d['band_evidence']:
            n=r['N'];a,b=r['model'];l,u=r['target']
            assert r['state']==('inside' if l<=a<=b<=u else 'outside' if b<l or a>u else 'unresolved')
            if any(x['H']==h and x['N']==n for x in br):assert r['state']==next(x['state'] for x in br if x['H']==h and x['N']==n)
        first=next(r['N'] for r in d['band_evidence'] if r['state']=='inside')
        assert all(r['state']=='outside' for r in d['band_evidence'] if r['N']<first)
        items=[]
        for j in range(4):
            states=[]
            for r in d['tick_evidence']:
                e=([r['futures_error']]+r['call_errors'])[j]
                states.append((r['N'],'within' if max(map(abs,e))<.05 else 'outside' if e[0]>.05 or e[1]<-.05 else 'unresolved'))
            n=next(n for n,state in states if state=='within')
            items.append([i for i,state in states if i<n and state=='unresolved']+[n])
        specs.append([float(h),items[0],first,*items[1:],None])
    checkrows('T5B',take('tab_competitors',r'^0\.'),specs)
    specs=[]
    for b in (50,100,500,1000):
        q=[r for r in data('quantize_suite') if r['budget']==b];u=[r for r in q if r['curve']!='stress'];st=next(r for r in q if r['curve']=='stress')
        def span(rows):return interval([min(r['signed_future_error'][0] for r in rows),max(r['signed_future_error'][1] for r in rows)])
        specs.append([[b,st['actual_paths']],span(u),span([r for r in data('bcj_calendar') if r['budget']==b]),span([st]),statistics.median(r['basis_setup_s']+r['product_setup_s'] for r in u),statistics.median(r['online_s'] for r in u)])
    checkrows('T5C',take('tab_competitors',r'^[1-9]'),specs)
    vals=scoreboard(); checkrows('T6',take('tab_common',r'^(\\textbf|Markovian|500-cell|BDG|LAB|Moment|BCJ|JMM)'),[[None,None,*[interval(x) if x else [] for x in row],None] for row in vals])
    specs=[];E=lambda n:legacy('E13_rough_stein_stein','e13'+n)
    for r in E('_results'):
        h=r['H'];d=next(v for v in E('d_results') if v['H']==h)
        specs.append([h,100*r['riemann_rel_rms'],100*r['ce_rel_rms'],*[fields(r,'width_'+j+'_sd coverage_'+j) for j in 'acd'],E('_floor')[str(h)]['floor_sd'],fields(d,'width_dbin_sd coverage_dbin floor_bin_sd'),r['naive_q99_bias_sd']])
    for r in E('_results'):
        h=r['H'];c=next(z for z in E('c_results') if z['H']==h)
        for k in ('ATM','mean+2sd'):
            a=r['prices'][k];d=c['prices'][k]; specs.append([h,[] if k=='ATM' else 2,a['exact'],*[[a[j+'_width_rel'],a['coverage'][j]] for j in ('lipschitz','convex','scarf')],fields(d,'width_rel coverage'),[],100*a['naive_daily_bias_rel']])
    for r in E('b_results_H0.05')+E('b_results_H0.1'):
        specs.append([r['H'],r['per_day'],r['residual_share'],[],r['width_c_sd'],r['width_d_sd'],r['width_floor_sd'],[],r['coverage_d']])
    checkrows('T7',take('tab_e13',r'^\d'),specs)
    rs=sorted(legacy('E6c_rough_vol_condvar','e6c_results'),key=lambda z:(z['H'],z['output']));specs=[]
    for r in rs:
        specs.append([None,r['H'],100*r['ce_rel_rms'],*fields(r,'width_c_sd width_d_sd width_floor_sd'),r['width_c_sd']/r['width_d_sd'],r['e6b_width_sd'],r['mc_inside_d']])
    for r in rs:
        for k,a in r.get('prices',{}).items():
            specs.append([r['H'],[] if k=='ATM' else 2,[],interval([v*1e4 for v in a['convex_median']]),interval([v*1e4 for v in a['scarf_median']]),a['fine_mc_e6b']*1e4,[]])
    checkrows('T8',take('tab_e6c',r'^(Int\.|\d)'),specs)
    specs=[]
    for curve in ('flat','quad','sqrt'):
        for r in sorted((z for z in f['block_c_bcj2023_18cells'] if z['curve']==curve),key=lambda z:z['T_months']):
            L,U=fields(r,'condvar_L condvar_U');w=U-L;wj=r['jmm_up'][1]-r['jmm_lo'][0];specs.append([r['T_months'],interval([L,U]),up(w),up(wj),wj/w,r['setup_s'],r['solve_s']])
    checkrows('S1',take('tab_bcj_full',r'^\d'),specs)
    E=lambda n:legacy('E5c_certified_cantelli','e5c_'+n);specs=[]
    for r in E('results'):
        h,n=r['H'],r['n'];b=next(v for v in E('binomial') if (v['H'],v['n'])==(h,n));fl=E('floor')[f'{h:g}_{n}']['floor'];c,d,rb=fields(r,'width_c width_d width_d_rb')
        specs.append([h,n,r['residual_share'],fields(r,'coverage_c coverage_d coverage_d_rb')+[b['coverage_dbin']],fl,c,d,rb,(c-fl)/(d-fl),b['width_dbin'],b['floor_bin']])
    checkrows('S2',take('tab_e5c',r'^\d'),specs)
    out={'checked_numeric_tokens':len(records),'rules':{m:sum(r['rule']==m for r in records) for m in ['down','up','nearest']},'records':records}
    (RES.parent/'audit/v3_table_cells.json').write_text(json.dumps(out,indent=1))
    print(out['checked_numeric_tokens'],out['rules'])
if __name__=='__main__':main()

