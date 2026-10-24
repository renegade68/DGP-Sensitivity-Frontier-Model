"""Compare raw public-package parameters with frozen production predictions.
No refitting, no result replacement, and no truth-based selection.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
import sys,csv,json,hashlib,collections,datetime
from pathlib import Path
HERE=Path(__file__).resolve().parent;REV=HERE.parent;PRIOR=REV/'sfa_independent_validation_20261024'
sys.path.insert(0,str(REV/'efficiency_benchmark_20261007/environment/python_packages'))
import numpy as np
from scipy.special import log_ndtr, roots_laguerre
from scipy.stats import spearmanr,kendalltau
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,obj):
    (HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvout(name,rows):
    rows=list(rows)
    if not rows:return
    with (HERE/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def table(p):return list(csv.DictReader(Path(p).open()))
def design(x,form):
    lx=np.log(x);m=x.shape[1]
    return np.column_stack([np.ones(len(x)),lx]+([.5*lx**2]+[lx[:,i]*lx[:,j] for i in range(m) for j in range(i+1,m)] if form=='TL' else []))
nodes,weights=roots_laguerre(64)
def scores(e,su,sv,dist,order=64):
    if su==0:return np.ones(len(e)),np.ones(len(e))
    if sv==0:
        assert np.max(e)<=1e-7,'Zero-noise result outside support'
        return np.exp(e),np.exp(e)
    if dist=='HN':
        total=np.hypot(su,sv);mu=-e*(su/total)**2;sd=su*(sv/total)
    else:mu=-e-sv*(sv/su);sd=sv
    a=mu/sd;tail=a < -8
    bc=np.empty(len(e));jl=np.empty(len(e))
    mid=a[~tail]
    logcdf=log_ndtr(mid);mills=np.exp(-.5*mid**2-.5*np.log(2*np.pi)-logcdf)
    jl[~tail]=np.exp(-sd*(mid+mills))
    bc[~tail]=np.exp(-mu[~tail]+.5*sd*sd+log_ndtr(mid-sd)-logcdf)
    # Independently evaluate truncated-normal moment integrals in the difficult
    # tail using scaled Gauss-Laguerre quadrature, not production Mills code.
    if np.any(tail):
        nd,wt=(nodes,weights) if order==64 else roots_laguerre(order)
        z=-a[tail];t=nd[None,:]/z[:,None];w=wt[None,:]*np.exp(-.5*t*t);den=w.sum(axis=1)
        eu=sd*(w*t).sum(axis=1)/den
        jl[tail]=np.exp(-eu);bc[tail]=(w*np.exp(-sd*t)).sum(axis=1)/den
    return bc,jl
def prepare():
    sources={}
    for name,h in read(PRIOR/'MANIFEST.json')['files'].items():
        p=PRIOR/name;assert sha(p)==h;sources[str(p)]=h
    inputs={r['task']:Path(r['directory']) for r in table(PRIOR/'INPUTS.csv')}
    for p in inputs.values():
        complete=read(p/'COMPLETE.json')
        for name in ['observed.npz','train.csv']+[f'SFA_{form}_{dist}/'+name for form in ['CD','TL'] for dist in ['HN','EXP'] for name in ['PREDICTIONS.npz','STATUS.json']]:
            assert sha(p/name)==complete['files'][name];sources[str(p/name)]=sha(p/name)
    raw=read(PRIOR/'R_RESULTS.json');lookup={(r['task'],r['model'],r['engine'],r['start']):i for i,r in enumerate(raw)}
    selected=[]
    for r in table(PRIOR/'R_BEST_BY_CASE.csv'):
        selected.append(dict(index0=lookup[(r['task'],r['model'],r['engine'],r['start'])],task=r['task'],model=r['model'],engine=r['engine'],start=r['start'],directory=str(inputs[r['task']])))
    assert len(selected)==144
    csvout('SELECTED_R.csv',selected);write('SOURCE_HASHES.json',sources)
    write('PROTOCOL.json',dict(scope='24 saved datasets; 96 sfaR fits plus 48 frontier HN fits from the preceding fixed sample',selection='Same per-case best likelihood among three starts used previously; no reselection for scores or rankings',likelihood_match_per_observation=1e-7,score_range_tolerance=1e-8,pairwise_tie_tolerance=1e-8,comparison='Frontier on training and 5000 fixed evaluation points; BC=E(exp(-u)|epsilon), JLMS=exp(-E(u|epsilon)); ranks of observations, not estimator performance ranks',scores='Same stable conditional distribution formulas at each public package parameter vector; sfaR native score routine also evaluated without refitting',descriptive_thresholds=dict(frontier_relative=[1e-6,1e-4,1e-3,1e-2],score_absolute=[1e-6,1e-4,1e-3,1e-2]),production_modified=False))
    print('Prepared 144 selected public-package fits',flush=True)
def run():
    selected=table(HERE/'SELECTED_R.csv');raw=read(PRIOR/'R_RESULTS.json');index=read(PRIOR/'R_PARAMS_INDEX.json');params=np.fromfile(PRIOR/'R_PARAMS.f64','<f8')
    matched={(r['task'],r['model'],r['engine']):r for r in table(PRIOR/'R_BEST_BY_CASE.csv')}
    rows=[];score_rows=[];checks=[];saved={}
    for j,r in enumerate(selected):
        p=Path(r['directory']);task=r['task'];model=r['model'];engine=r['engine'];form,dist=model.split('_')[1:];ix=int(r['index0']);off=index[ix];pars=params[off['offset']:off['offset']+off['length']]
        obs=np.load(p/'observed.npz');X=design(obs['x'],form);XE=design(obs['x_eval'],form);ly=np.log(obs['y']);n,K=X.shape
        b=pars[:K];su=float(pars[K]);sv=float(pars[K+1]);old=read(p/model/'STATUS.json');pred=np.load(p/model/'PREDICTIONS.npz')
        lp=np.sum(X*b,axis=1);lpe=np.sum(XE*b,axis=1);e=ly-lp;bc,jl=scores(e,su,sv,dist)
        assert np.all(np.isfinite(np.r_[bc,jl])) and min(bc.min(),jl.min())>=0 and max(bc.max(),jl.max())<=1+1e-7
        # Quadrature-order and frozen-score identities, independent of production score code.
        bc2,jl2=scores(e,su,sv,dist,128);qe=max(float(np.max(abs(bc-bc2))),float(np.max(abs(jl-jl2))))
        ob=np.array(old['selected']['beta']);oe=ly-np.sum(X*ob,axis=1);obc,ojl=scores(oe,old['sigma_u'],old['sigma_v'],dist)
        olderr=max(float(np.max(abs(obc-pred['te_BC']))),float(np.max(abs(ojl-pred['te_JLMS']))))
        assert qe<1e-10 and olderr<1e-10,(task,model,qe,olderr)
        checks.append(dict(index0=ix,task=task,model=model,engine=engine,quadrature64_vs128_max=qe,independent_vs_frozen_scores_max=olderr))
        gain=float(matched[(task,model,engine)]['canonical_minus_frozen']);same=abs(gain)<=1e-7*n
        row=dict(index0=ix,task=task,model=model,engine=engine,start=r['start'],n=n,K=K,old_status=old['status'],loglik_difference=gain,likelihood_matched=same,sigma_u=su,sigma_v=sv,old_sigma_u=old['sigma_u'],old_sigma_v=old['sigma_v'],beta_max_abs_difference=float(np.max(abs(b-ob))))
        for label,newlp,oldf in [('train',lp,pred['frontier_train']),('eval',lpe,pred['frontier_eval'])]:
            err=np.abs(np.expm1(newlp-np.log(oldf)))
            row[label+'_frontier_mean_relative_difference']=float(np.mean(err));row[label+'_frontier_max_relative_difference']=float(np.max(err));row[label+'_frontier_log_RMSE']=float(np.sqrt(np.mean((newlp-np.log(oldf))**2)))
        rows.append(row)
        for name,a,z in [('BC',bc,pred['te_BC']),('JLMS',jl,pred['te_JLMS'])]:
            rngA=float(np.ptp(a));rngZ=float(np.ptp(z));why='defined' if min(rngA,rngZ)>1e-8 else 'constant_or_nearly_constant_score'
            sp=kt=None;discord=0;comparable=0;ambiguous=0;top=None;topreason=why
            if why=='defined':
                sp=float(spearmanr(a,z).statistic);kt=float(kendalltau(a,z).statistic)
                ii,jj=np.triu_indices(n,1);da=a[ii]-a[jj];dz=z[ii]-z[jj];ok=(abs(da)>1e-8)&(abs(dz)>1e-8)
                comparable=int(ok.sum());discord=int(((da[ok]*dz[ok])<0).sum());ambiguous=int((~ok).sum())
                k=int(np.ceil(.2*n));aa=np.sort(a)[::-1];zz=np.sort(z)[::-1]
                if aa[k-1]-aa[k]>1e-8 and zz[k-1]-zz[k]>1e-8:
                    top=len(set(np.argsort(a)[-k:])&set(np.argsort(z)[-k:]))/k;topreason='defined'
                else:topreason='ties_at_top20_cutoff'
            score_rows.append(dict(index0=ix,task=task,model=model,engine=engine,score=name,old_status=old['status'],likelihood_matched=same,mean_absolute_difference=float(np.mean(abs(a-z))),max_absolute_difference=float(np.max(abs(a-z))),old_mean=float(z.mean()),new_mean=float(a.mean()),old_range=rngZ,new_range=rngA,rank_status=why,spearman=sp,kendall_tau=kt,comparable_pairs=comparable,discordant_pairs=discord,ambiguous_pairs=ambiguous,top20_status=topreason,top20_overlap=top))
        saved[str(ix)+'_BC']=bc;saved[str(ix)+'_JLMS']=jl
    csvout('PREDICTION_COMPARISON.csv',rows);csvout('SCORE_COMPARISON.csv',score_rows);csvout('SCORING_CHECKS.csv',checks);np.savez(HERE/'STANDARDIZED_SCORES.npz',**saved)
    write('SCORING_VALIDATION.json',dict(status='PASS',package_fits=len(rows),score_comparisons=len(score_rows),max_quadrature_order_error=max(r['quadrature64_vs128_max'] for r in checks),max_frozen_score_recomputation_error=max(r['independent_vs_frozen_scores_max'] for r in checks)))
    print(read(HERE/'SCORING_VALIDATION.json'))
if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()

