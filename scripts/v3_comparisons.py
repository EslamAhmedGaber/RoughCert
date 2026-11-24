"""Contract-matched numerical inputs shared by the table and figure generators."""
import json, math
from pathlib import Path
from arb_vix_core import arb,arb_lo_float,arb_hi_float
from arb_vix_options import black76_iv_bracket_arb

RES=Path(__file__).resolve().parents[1]/'results'
def data(name): return json.loads((RES/(name+'.json')).read_text(encoding='utf8'))
def bounds(row,prefix,unit='volpts'):
    return [row[f'{prefix}_L_{unit}'],row[f'{prefix}_U_{unit}']]
def iv(F,C,k):
    l,u,verified=black76_iv_bracket_arb(arb(F[0])/100,arb(F[1])/100,arb(str(k)),arb(1)/4,arb(C[0])/100,arb(C[1])/100)
    assert verified
    return [arb_lo_float(arb(l)*100),arb_hi_float(arb(u)*100)]
def scoreboard():
    curve=next(r for r in data('lab_suite') if r['name']=='curved')
    b=curve['arb']; k=.27
    def option(model): return next(r for r in model['options'] if r['K_decimal']==k)
    m=next(r['model'] for r in data('markov_optimized') if r['curve']=='curved' and r['N']==12)
    comp=data('curved_comparison'); g=comp['grid500']; l=comp['moment_matched']; j=comp['jmm']
    exp=data('curved_expansions'); e=next(r for r in exp['rows'] if float(r['K'])==k)
    q=next(r for r in data('quantize_suite') if (r['months'],r['curve'],r['budget'])==(3,'quad',1000))
    i=[r['K_decimal'] for r in b['options']].index(k)
    def model(r):return [bounds(r,'future'),bounds(option(r),'call'),bounds(option(r),'iv','pct')]
    gf,gc=bounds(g,'future'),bounds(option(g),'call')
    return [model(b),model(m),[gf,gc,iv(gf,gc,k)],[exp['forward'],e['bdg_call'],e['bdg_iv']],
            [None,None,e['lab_iv']],[l['ln_F_volpts'],l['options'][i]['ln_call_volpts'],l['ln_iv_pct']],
            [q['future'],q['calls'][i],iv(q['future'],q['calls'][i],k)],[[j['jmm_lo_volpts'][0],j['jmm_up_volpts'][1]],None,None]]
def difference(a,b):
    return [math.nextafter(a[0]-b[1],-math.inf),math.nextafter(a[1]-b[0],math.inf)]
def absolute(e):
    return [0 if e[0]<=0<=e[1] else min(map(abs,e)),max(map(abs,e))]
def factor_paths():
    decisions=data('factor_decisions'); extra=data('v3_factor_extension')
    out={}
    for h,d in decisions.items():
        series={r['N']:r for r in extra if r['H']==h}
        series.update({r['N']:r for r in d['band_evidence']})
        tick={r['N']:r['futures_error'] for r in d['tick_evidence']}
        path=[]
        for n in range(1,20):
            r=series[n]; lo,hi=difference(r['model'],r['target'])
            if n in tick:lo,hi=max(lo,tick[n][0]),min(hi,tick[n][1])
            assert lo<=hi
            path.append(absolute([lo,hi]))
        out[h]=path
    return out

