"""Audit final prose, equation/display references, PDFs and evidence from the independent numerical checks."""
import hashlib,json,re,math
from pathlib import Path
import fitz,mpmath as mp
from v3_comparisons import data,RES,scoreboard
from audit_v3_tables import fields,legacy
from build_v4_numbers import rounded as rnd,spec as v4spec
ROOT=RES.parent; M=ROOT/'manuscript';A=ROOT/'audit';S=M/'sections'
tests=[]
def check(label,ok,detail=None):
    assert ok,(label,detail)
    tests.append({'test':label,'evidence':detail if detail is not None else True})
def proofids(p):return re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}',p.read_text(encoding='utf8'))
MACROS=dict(kv for p in sorted((RES.parent/'manuscript/tables').glob('*numbers.tex'))
           for kv in re.findall(r'\\newcommand\{\\(\w+)\}\{([^{}]*)\}',p.read_text(encoding='utf8')))
def expand(t):return re.sub(r'\\([A-Za-z]+)',lambda m:MACROS.get(m.group(1),m.group(0)),t)
def prose(name):return expand(next(S.glob(name+'*.tex')).read_text(encoding='utf8'))
def norm(t):return t.replace('\\%','').replace('\\,','').replace('\n',' ').replace('$','')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def hull(name,text,printed,raw,joined='--'):
    l,u=printed
    check(name+' printed',l+joined+u in norm(text),(l,u))
    check(name+' outward',float(l)<=min(raw)<=max(raw)<=float(u),list(raw))
def iv_bounds(r):return fields(r,'iv_L_pct iv_U_pct')
def prices(r):return fields(r,'call_L_volpts call_U_volpts')
def forwards(r):return fields(r,'future_L_volpts future_U_volpts')
def close(x,y,tol=1e-11):return abs(x-y)<=tol

def prose_checks():
    v4=prose('04_');v5=prose('05_');v6=prose('06_');v7=prose('06b');v8=prose('07_');v9=prose('08_')
    g=data('week1_gates_results');f=data('vix_futures_full_suite');o=data('vix_options_smile_suite');ref=data('refinement');lab=data('lab_suite')
    control=g['gate1a']['controls'];cn=control[0]
    check('constant-kernel independent closed form',close(23.5*math.exp(-1.2**2/8),cn['exact_volpts']) and f"{cn['exact_volpts']:.6f}" in v4 and f"{cn['norm_diff_width']/1e-50:.2f}" in v4)
    for r in control[1:]:
        eps=r['eps'];expect=(1-1/math.sqrt(math.cosh(eps*eps)))/eps**4
        check('two-atom coefficient '+str(eps),abs(expect-r['rel_width_over_eps4'])<2e-12 and f"{expect:.6f}" in v4 and r['max_norm_diff']<3.5e-50)
    check('Pfaff cross-checks',[f"{g['gate1b'][k]*1e16:.2f}" in v4 for k in ['rel_diff_hk','rel_diff_nh2']]==[True,True])
    chk=data('v3_signed_atom_controls');check('signed two-atom controls',all(abs(abs(r['I'])-1)<1e-13 for r in chk['two_atom'].values()))
    check('two million signed-sample reproductions',all(f'{r[5]:.2f}' in v4 for r in chk['five_atom']),[r[5] for r in chk['five_atom']])
    c=g['gate2']['cells'];check('headline tick and endpoint widths',len(c)==8 and all(r['tilted_total_outer_width']<.05 and r['tilted_L_enc_width']<.00139 and r['tilted_U_enc_width']<6.41e-11 and r['bdg_impl_diff']<6.3e-5 for r in c))
    ef=f['block_a_e14_15cells'];keys=[x for x in ef if x['group']!='stress'];rough=[x for x in ef if x['group']=='stress' and x['H']<=.1]
    for name,printed,rs in [('base and curve',['0.008','0.105'],keys),('rough stress',['0.146','0.152'],rough)]:
        hull('T2 JMM low gap '+name,v5,printed,[x['best_U']-x['jmm_lo'][0] for x in rs])
    check('T2B curve formula, all maturities',all(x['curve']=='quad(0.234^2*(1+t)^2)' and x['T']==1/12 and x['Delta']==30/365 for x in ef if x['group']=='curve'))
    az=f['block_b_bdg_sweeps']['fig1_a_sweep']
    check('sampled thresholds',min(x['a'] for x in az if x['L_minus_LN']>.05)==1 and [x['a'] for x in az if x['bdg3_minus_U']>.05]==[1.35,1.5])
    pure=o['bdg2023_fig6'];stress=o['e14_stress_smile'];scale=o['lab2026_smile'];mixed=o['mixed_rbergomi_smile'];rr=ref['grid']
    for name,z,fp,edges,widths in [
        ('pure',pure,['19.9092','19.9152'],[['113.48','114.05'],['114.83','115.01']],['0.00409','0.00877']),
        ('rescaled',scale,['20.3328','20.3390'],[['113.66','114.13'],['114.66','114.86']],['0.00520','0.00873']),
        ('stress',stress,['15.54225','15.55578'],[['126.58','127.44'],['127.77','128.06']],['0.00676','0.01532']),
        ('mixed',mixed,None,[['92.95','93.33'],['99.55','99.73']],['0.00476','0.00706'])]:
        opts=rr['options'] if name=='stress' else z['strikes']
        for tag,printed,r in zip(['left','right'],edges,[opts[0],opts[-1]],strict=True):hull(name+'/'+tag,v6,printed,iv_bounds(r),',')
        hull(name+'/call widths',v6,widths,[r['call_width_volpts'] for r in opts])
        if fp:hull(name+'/forward',v6,fp,forwards(rr) if name=='stress' else fields(z,'condvar_future_L condvar_future_U'),',')
    ps=pure['strikes'];ss=stress['strikes'];rs=rr['options'];ms=mixed['strikes'];lk=lab[-2]['lab_arb_iv']
    check('pure overlaps/strict containment',sum(r['call_L_volpts']<=r['bdg_C3_volpts'][0]<=r['bdg_C3_volpts'][1]<=r['call_U_volpts'] for r in ps)==6)
    check('pure endpoint IV minimum slope',ps[-1]['iv_L_pct']-ps[0]['iv_U_pct']>=.78)
    check('mixed slope all adjacent strikes',all(b['iv_L_pct']>a['iv_U_pct'] for a,b in zip(ms,ms[1:])) and ms[-1]['iv_L_pct']-ms[0]['iv_U_pct']>=6.22)
    hull('BDG pure edge gap',v6,['0.21','0.28'],[r['iv_L_pct']-r['bdg_iv3_pct'][1] for r in ps])
    hull('LAB/BDG refined edge gap',v6,['0.39','0.45'],[v['iv_L_pct']-b for s,v,l in zip(ss,rs,lk) for b in (s['bdg_iv3_pct'][1],l[1])])
    check('stress mesh/ATM 200x420 to 320x620',rr['M']==320 and rr['Nz']==620 and .643<ss[2]['iv_width_pct']<=.644 and .551<rs[2]['iv_width_pct']<=.552 and ss[2]['K_over_F']==1,list(fields(rr,'M Nz'))+[ss[2]['iv_width_pct'],rs[2]['iv_width_pct']])
    check('stress 5/6 call excess and future gap',sum(r['bdg_C3_volpts'][0]>t['call_U_volpts'] for r,t in zip(ss,rs))==5 and ss[0]['bdg_C3_volpts'][0]-rs[0]['call_U_volpts']>=.01802 and ss[2]['bdg_C3_volpts'][0]-rs[2]['call_U_volpts']>=.00901 and stress['bdg_F3_volpts'][0]-rr['future_U_volpts']>=.05836)
    bl=o['bd2022_mlmc_call']
    for tag,fb,cb in [('continuum_arb',('22.17970','22.18063'),('2.98837','2.99007')),('grid500_arb',('22.18090','22.18174'),('2.98798','2.98953'))]:
        model=bl[tag];hull(tag+' forward',v6,fb,forwards(model),',');hull(tag+' call',v6,cb,prices(model['options'][0]),',')
    mm=data('mixed_split')['rows']
    for i,sp,tr in [(1,('93.8671','93.8672'),('93.9407','94.2487')),(4,('97.0817','97.0818'),('97.1870','97.3764'))]:
        hull('split IV '+str(i),v7,sp,mm[i]['split_iv'],',');hull('continuum IV '+str(i),v7,tr,mm[i]['target_iv'],',')
    check('mixed seven edge separations',all(.056<=x['target_iv'][0]-x['split_iv'][1]<=.112 for x in mm))
    d=data('factor_decisions')['.05'];hull('N13 boundary',v7,('−0.0671'.replace('−','-'),'-0.0611'),d['tick_evidence'][12]['futures_error'],',');hull('N14 boundary',v7,('-0.0496','-0.0453'),data('factor_boundary')['error'],',')
    s=scoreboard();b=s[0]
    def diff(x,y):return [x[0]-y[1],x[1]-y[0]]
    hull('500-grid futures above',v7,('0.00098','0.00999'),diff(s[2][0],b[0]))
    hull('LN IV above',v7,('1.26','1.76'),diff(s[5][2],b[2]))
    hull('BCJ future below',v7,('0.425','0.438'),diff(b[0],s[6][0]));hull('BCJ IV below',v7,('5.51','5.86'),diff(b[2],s[6][2]))
    def overlaps(a,b):return min(a[1],b[1])>=max(a[0],b[0])
    check('T6 full-precision overlap/containment',all(overlaps(a,t) and not(t[0]<=a[0] and a[1]<=t[1]) for a,t in zip(s[1],b)) and all(overlaps(a,t) for a,t in zip(s[3],b)) and overlaps(s[4][2],b[2]) and diff(b[0],s[5][0])[0]>.05 and s[7][0][0]<=b[0][0]<b[0][1]<=s[7][0][1])
    mp.mp.dps=65
    for i in (2,6):
        F,C,IV=s[i]
        def price(f,iv):
            v=mp.mpf(iv)/200;f=mp.mpf(f);d1=mp.log(f/27)/v+v/2
            return f*mp.ncdf(d1)-27*mp.ncdf(d1-v)
        check('independent Black76 rectangular corner signs '+str(i),price(F[1],IV[0])<=C[0] and price(F[0],IV[1])>=C[1])
    counts=[sum(z[1]<r['iv_L_pct'] for z,r in zip(l['lab_arb_iv'],l['arb']['options'])) for l in lab[:6]]
    check('52/60, separated at 30+22 author strikes',counts==[10,10,10,10,8,4],counts)
    for name in ['qmc_check','high_precision_qmc']:
        r=data(name);check(name+' diagnostics',sum(len(z['contained']) for z in r)==sum(sum(z['contained']) for z in r)==35)
    tilt=ref['tilt']
    for row,bound in zip(tilt,[.01968,.01447,.03003,.02092]):check('tilt width '+row['a'][:8]+'/'+row['c'],row['U'][1]-row['L'][0]<=bound and f'{bound:.5f}' in v7)
    shares=legacy('E14_vix_forward_variance','e14_results');check('spot vs forward residual share',max(100*z['share'] for z in shares if z['group']=='base')<.062 and '0.062' in v8+v9 and max(100*z['P']/z['sd']**2 for z in legacy('E6c_rough_vol_condvar','e6c_results'))<19.4)
    for name in ('tab_e13','tab_e6c','tab_e5c'):
        t=(M/'tables'/(name+'.tex')).read_text()
        check('statistical risk layer '+name,r'\RoughCertCV{}' in t and ('statistical' in t.lower() or 'replications' in t.lower()) and 'Arb-certified' not in t)
    check('statistical risk label in text',r'\RoughCertCV{}' in v8 and 'statistical risk brackets' in v8)

    expected=json.loads((ROOT/'verification/used_macros.json').read_text())
    old_macros=dict(re.findall(r'\\newcommand\{\\(\w+)\}\{(.*)\}',(M/'tables/numbers.tex').read_text(encoding='utf8')))
    check('44 active legacy macro values agree with independent regeneration',
          len(expected)==44 and all(old_macros.get(k)==v for k,v in expected.items()))

def secondary_checks():
    v4,v5,v6,v7,v8=map(prose,['04_','05_','06_','06b','07_'])
    f=data('vix_futures_full_suite');o=data('vix_options_smile_suite')
    cells=data('week1_gates_results')['gate2']['cells'];z=f['block_b_bdg_sweeps']['fig1_a_sweep'];hs=f['block_b_bdg_sweeps']['fig2_H_sweep']
    for row,printed in zip(z[-3:],['0.0397','0.0584','0.0811']):
        check('sampled a-sweep BDG '+printed,rnd(row['bdg3_minus_U'],4,'down')==printed and printed in v5)
    for number,raw,dp,mode in [('0.06597',hs[0]['bdg3_minus_U'],5,'down'),('0.2263',hs[0]['L_minus_LN'],4,'down'),
                               ('0.0251',hs[0]['tilted_width'],4,'up')]:
        check('H=.05 sweep '+number,rnd(raw,dp,mode)==number and number in v5)
    check('H sweep widths above H=.10',all(z['tilted_width']<.0066 for z in hs if z['H']>=.1))
    check('13-week fixed contract fits finite kernels',all(len(z['nodes'])==len(z['weights'])==z['N'] for z in data('v3_factor_extension')))
    for c in cells:
        check('headline proxy sign '+c['id'],c['bdg_F3_arb'][0]>c['tilted_U_outer'] and c['ln_fut'][1]<c['tilted_L_outer'])
    for k,digits in [('bdg2023_fig6',4),('e14_stress_smile',4)]:
        a=o[k]; num=rnd(a['tilted_lipschitz_opt_bound_volpts'],digits,'up');check('absolute Lipschitz bound '+k,num in v6,num)
    r=o['bdg2023_fig6']['strikes']
    for t in (r[0],r[1],r[-1]):
        v=t['bdg_call_cert_sep_volpts'];num=('+' if v>0 else '-')+rnd(abs(v),5,'zero')
        check('signed call deviation pure '+str(t['K_decimal']),num in v6,num)
    for k in ['bdg2023_fig6','e14_stress_smile','mixed_rbergomi_smile','lab2026_smile']:
        opt=o[k]['strikes'];check('strike/IV verification '+k,all(t['iv_arb_verified'] and close(t['K_volpts'],100*t['K_decimal']) for t in opt))
    q=data('quantize_suite');q30=data('bcj_calendar');check('calendar sizes all budgets',len(q)==76 and len(q30)==72 and all(math.prod(z['allocation'])==z['actual_paths'] for z in q+q30))
    for k,rows in [('1/12',[r for r in q if r['curve']!='stress' and r['budget']==1000]),('30/365',[r for r in q30 if r['budget']==1000])]:
        elow=min(-z['signed_future_error'][1] for z in rows);emax=max(-z['signed_future_error'][0] for z in rows)
        check(k+' BCJ range',len(rows)==18 and (elow>=.208 if k=='1/12' else elow>=.209 and emax<=1.412),[elow,emax])
    curve=next(z for z in data('lab_suite') if z['name']=='curved');bench=curve['arb']['options'][1]['iv_L_pct']
    for k,d in [('bdg_iv','0.14'),('refined_iv','0.13')]:
        ref=curve[k][1];check('curved diagnostic '+k,f'{ref:.3f}' in v7 and f'{bench-ref:.2f}'==d)
    for r in data('qmc_check'):
        check('QMC design '+r['name'],r['scrambles']==12 and [(z['nodes'],z['power']) for z in r['runs']]==[(48,14),(96,17)])
    check('high precision sampler design',all(z['dps']==130 and z['gaussian_dimensions']==40 for z in data('high_precision_qmc')))
    r=legacy('E13_rough_stein_stein','e13_results')
    check('KL mean and tail',max(abs(z['mean_kl']-z['mean']) for z in r)<1e-9 and f"{r[0]['rest_var_share']*1e6:.1f}" in v8)
    z=legacy('E6c_rough_vol_condvar','e6c_results')
    check('median Scarf-Lo MC diagnostics and units',all(p['scarf_median'][0]<=p['fine_mc_e6b']<=p['scarf_median'][1] for v in z for p in v.get('prices',{}).values()))
    check('Gaussian-tail envelope over full BCJ curve suite',100*.234*(2+1/12)*math.sqrt(math.erfc(10/math.sqrt(2)))<2e-10,100*.234*(2+1/12)*math.sqrt(math.erfc(10/math.sqrt(2))))
    cc=legacy('C03_sandwich_bound','c03_results_closed')+legacy('C03_sandwich_bound','c03_results_generic')
    check('24 two-sided sandwich tests, finite and infinite upper bounds',len(cc)==24 and all(0<=r['lower_over_P']<=1<=r['upper_over_P'] for r in cc) and sum(math.isinf(r['upper_over_P']) for r in cc)==6)
    check('legacy E9 five-grid fits',all({x['n'] for x in legacy('E9_measure_axis_rates','e9_results') if x['f']==func and x['H']==.1 and x.get('n',0)>=64}=={64,128,256,512,1024} for func in ['WickExp(eta=1.0)','PositivePart(c=0.5)','Indicator(c=0.5)']))
    check('adversarial arithmetic unit checks',data('adversarial_checks')['bits']==169 and all(data('adversarial_checks')['checks'].values()))
    text=prose('06b')
    check('strict fixed strikes / point thresholds',all(t in text for t in (r'N\in\{5,6\}',r'N\in\{1,2\}',r'K=18',r'K=22',r'K=27')))
    with fitz.open(M/'main.pdf') as doc:
        def pos(s):return [(i+1,r.y0) for i,p in enumerate(doc) for r in p.search_for(s)][-1]
        check('last bibliography entry before appendix A',pos('References')<pos('18M1221837')<pos('Proofs'))
def macro_checks():
    """Every generated narrative macro is recomputed from the JSONs under its declared rounding rule."""
    file_macros=dict(re.findall(r'\\newcommand\{\\(\w+)\}\{([^{}]*)\}',(M/'tables/v4_numbers.tex').read_text(encoding='utf8')))
    src=''.join(q.read_text(encoding='utf8') for q in [*S.glob('*.tex'),M/'main.tex',M/'supplement/supplement.tex'])
    spec=v4spec()
    for name,value,n,mode,meaning in spec:
        check('macro '+name,file_macros[name]==rnd(value,n,mode),{name:file_macros[name],'rule':mode})
        check('macro used '+name,re.search(r'\\'+name+r'(?![A-Za-z])',src) is not None)
    check('no stale generated macros',set(file_macros)=={s[0] for s in spec},sorted(set(file_macros)^{s[0] for s in spec}))
def main():
    prose_checks()
    secondary_checks()
    macro_checks()
    ns={};hard=[]
    for folder,stem in [(M,'main'),(M/'supplement','supplement')]:
        aux=(folder/(stem+'.aux'));labels=dict(proofids(aux));ns[stem]=labels
        paths=([M/'main.tex',*S.glob('*.tex'),*(M/'tables').glob('*.tex')] if stem=='main' else [folder/'supplement.tex',M/'tables/tab_e5c.tex',M/'tables/tab_bcj_full.tex'])
        for p in paths:
            t=p.read_text(encoding='utf8')
            for name in re.findall(r'\\(?:eqref|ref|pageref)\{([^}]+)\}|\\hyperref\[([^]]+)\]',t):
                key=next(v for v in name if v);assert key in labels,(p,key)
            for pat in re.finditer(r'(?:Section|Figure|Table|Proposition|Theorem)s?~(?:S?\d+(?:\.\d+)?)',t):hard.append((stem,p.name,pat.group(),t[max(0,pat.start()-27):pat.end()+50]))
        log=(folder/(stem+'.log')).read_text(errors='replace')
        check(stem+' TeX log',not re.search(r'(?:Overfull \\[hv]box|Underfull \\hbox)|undefined (references|citations)|LaTeX Warning: (Reference|Citation)|multiply defined|Label\(s\) may have changed|^!',log,re.I|re.M))
        with fitz.open(folder/(stem+'.pdf')) as doc:
            check(stem+' PDF unresolved references',all('??' not in page.get_text() for page in doc))
            check(stem+' all internal PDF destinations valid',all(0<=l['page']<len(doc) for page in doc for l in page.get_links() if l['kind']==fitz.LINK_GOTO))
    check('main numbering',[ns['main'][k] for k in ['thm:master','thm:sandwich','thm:rate','thm:tilted','prop:condvar','prop:root','prop:arb','sec:methods_vtwo','sec:variance_risk']]==['1','2','3','4','2','3','4','7','8'])
    check('three method subsections',[ns['main'][k] for k in ['sec:direct_expansion','sec:factor_counts','sec:curved_score']]==['7.1','7.2','7.3'])
    check('supplement numbering',[ns['supplement'][k] for k in ['tab:bcj_full','tab:e5c','fig:supp_sandwich','fig:supp_coverage','fig:supp_e5b']]==['S1','S2','S1','S2','S3'])
    proof=S/'A_proofs.tex'; frozen=ROOT/'verification/proof_baseline.txt'
    recorded=json.loads((ROOT/'verification/proof_edits.json').read_text(encoding='utf8'))['edits']
    cur=frozen.read_bytes()
    for e in recorded:
        o=e['old'].encode();check('proof edit applies once: '+e['why'][:48],cur.count(o)==1);cur=cur.replace(o,e['new'].encode())
    for e in json.loads((ROOT/'verification/proof_v5_edits.json').read_text()):
        if e['file'].endswith('/A_proofs.tex'):
            old=e['old'].encode();check('v5 proof edit applies once: '+e['old'][:55],cur.count(old)==1);cur=cur.replace(old,e['new'].encode())
    check('proof appendix is the frozen v2 text plus the recorded v4 and v5 edits',cur==proof.read_bytes(),
          {'frozen_v2':digest(frozen),'audited_v4':digest(proof),'recorded_edits':len(recorded)})

    tc=(M/'two_columns.log').read_text(errors='replace')
    check('two-column edition TeX log',not re.search(r'(?:Overfull \\[hv]box|Underfull \\hbox)|undefined (references|citations)|LaTeX Warning: (Reference|Citation)|multiply defined|Label\(s\) may have changed|^!',tc,re.I|re.M))
    with fitz.open(M/'two_columns.pdf') as doc:check('two-column edition PDF resolved',all('??' not in page.get_text() for page in doc))
    qa=data('build_qa');check('four release gates and both documents',all(r['exit']==0 for r in qa) and len(set(r['name'] for r in qa)&{'check_norms','check_technical','check_tex_hygiene','scan_leaks'})==4)
    macros={}
    for p in (M/'tables').glob('*numbers.tex'):macros.update(dict(re.findall(r'\\newcommand\{\\(\w+)\}\{(.*)\}',p.read_text(encoding='utf8'))))
    inv=[]
    for p in [*sorted(S.glob('*.tex')),M/'supplement/supplement.tex']:
        for j,line in enumerate(p.read_text(encoding='utf8').splitlines(),1):
            raw=re.sub(r'\\(?:label|ref|eqref|input|includegraphics|cite\w*)\{[^{}]*\}','',line.split('% ')[0])
            num=re.findall(r'(?<![A-Za-z])[-+]?\d+(?:\.\d+)?',raw); expansions={n:macros[n] for n in re.findall(r'\\([A-Za-z]+)',raw) if n in macros}
            if num or expansions:inv.append(dict(file=p.name,line=j,literals=num,macros=expansions))
    (A/'v3_text_inventory.json').write_text(json.dumps({'numerical_lines':inv,'hardcoded_references':hard},indent=1))
    path=RES/'v3_red_team_checks.json';path.write_text(json.dumps(dict(tests=tests,table_rules=json.loads((A/'v3_table_cells.json').read_text())['rules'],figure_tests=len(json.loads((A/'v3_figure_checks.json').read_text())),pdf_pages={s:len(fitz.open(p)) for s,p in [('main',M/'main.pdf'),('supplement',M/'supplement/supplement.pdf')]},pdf_sha256={p.name:digest(p) for p in [M/'main.pdf',M/'supplement/supplement.pdf']},numeric_text_lines=len(inv)),indent=2))
    print(len(tests),'release checks;',len(inv),'numeric source lines')
if __name__=='__main__':main()

