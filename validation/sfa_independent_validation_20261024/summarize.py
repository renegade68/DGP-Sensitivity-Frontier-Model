from validate import *
assert read(HERE/'R_DONE.json')['fits']==432
cases=read(HERE/'PYTHON_RESULTS.json');old={(r['task'],r['model']):r for r in cases};inputs={r['task']:Path(r['directory']) for r in csv.DictReader((HERE/'INPUTS.csv').open())}
raw=read(HERE/'R_RESULTS.json');offset=read(HERE/'R_PARAMS_INDEX.json');binary=np.fromfile(HERE/'R_PARAMS.f64','<f8');rrows=[];rbest={};observed={}
for task,p in inputs.items():
    obs=np.load(p/'observed.npz');x=obs['x'];y=obs['y'];n,m=x.shape
    exact=np.fromfile(HERE/'R_inputs'/f'{task}.f64','<f8').reshape((n,m+1),order='F')
    observed[task]=(x,np.log(y),exact[:,:m],np.log(exact[:,m]))
for i,r in enumerate(raw):
    task,model=r['task'],r['model'];o=old[(task,model)];form,dist=model.split('_')[1:];x,y,xr,yr=observed[task];X=design(x,form);XR=design(xr,form);K=X.shape[1]
    z=offset[i];p=binary[z['offset']:z['offset']+z['length']]
    ll=llr=delta=None;su=sv=None;zero=False;reported=None
    if r['status']=='returned' and len(p)==K+3 and np.all(np.isfinite(p)):
        b=p[:K];su=float(p[K]);sv=float(p[K+1]);reported=float(p[K+2]);zero=(su==0 or sv==0)
        def evaluated(A,response):
            e=response-np.sum(A*b,axis=1)
            if su>0 and sv>0:return float(ll_res(e,su,sv,dist).sum())
            if su==0 and sv>0:return float(np.sum(-np.log(sv)-.5*np.log(2*np.pi)-.5*(e/sv)**2))
            if sv==0 and su>0 and max(e)<=1e-7:
                return float(np.sum(np.log(2)-np.log(su)-.5*np.log(2*np.pi)-.5*(e/su)**2)) if dist=='HN' else float(np.sum(-np.log(su)+e/su))
            return None
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            ll=evaluated(X,y);llr=evaluated(XR,yr)
        if ll is not None and not np.isfinite(ll):ll=None
        if llr is not None and not np.isfinite(llr):llr=None
        delta=ll-o['old_loglik'] if ll is not None else None
    row=dict(task=task,model=model,engine=r['engine'],start=r['start'],status=r['status'],code=r.get('code'),seconds=r['seconds'],sigma_u=su,sigma_v=sv,exact_zero_component=zero,reported_loglik=reported,independent_R_input_loglik=llr,independent_canonical_input_loglik=ll,reported_minus_independent_R_input=reported-llr if llr is not None else None,canonical_minus_frozen=delta,warnings=json.dumps(r.get('warnings',[])),error=r.get('error'))
    rrows.append(row)
    if ll is not None:
        key=(task,model,r['engine'])
        if key not in rbest or ll>rbest[key]['independent_canonical_input_loglik']:rbest[key]=row
csvout('R_REEVALUATED.csv',rrows);csvout('R_BEST_BY_CASE.csv',rbest.values())

profiles=list(csv.DictReader((HERE/'BOUNDARY_PATHS.csv').open()));attempts=list(csv.DictReader((HERE/'OPTIMIZER_ATTEMPTS.csv').open()))
sensitivity=[];comparison=[]
for key,o in old.items():
    task,model=key;n=o['n'];alts=[r for r in rrows if (r['task'],r['model'])==key and r['independent_canonical_input_loglik'] is not None]
    interior=[dict(loglik=o['best_independent_interior_loglik'],su=o['sigma_u'],sv=o['sigma_v'],origin='independent_python')]+[dict(loglik=r['independent_canonical_input_loglik'],su=r['sigma_u'],sv=r['sigma_v'],origin=r['engine']) for r in alts if not r['exact_zero_component']]
    path=next(r for r in profiles if (r['task'],r['model'])==key and r['path']=='inward_intercept' and float(r['sigma_v_over_sigma_u'])==1e-10)
    # All paths have positive scales; keep path candidates separate from optimized fits.
    ip=max(interior,key=lambda r:r['loglik']);best_positive=max(ip['loglik'],float(path['loglik']))
    v=o['stored_boundary_loglik'];u=o['old_u_zero_loglik'];bestll=max(v,u,best_positive)
    for tie in [0.,1e-10,1e-8,1e-6]:
        eligible=[(v,'QUALIFIED_SIGMA_V_ZERO'),(u,'QUALIFIED_SIGMA_U_ZERO')];eligible=[x for x in eligible if x[0]>=bestll-tie*n]
        selected=max(eligible)[1] if eligible else 'QUALIFIED_INTERIOR'
        sensitivity.append(dict(task=task,model=model,tie_per_observation=tie,frozen_status=o['old_status'],selected_status=selected))
    starts=[float(r['loglik']) for r in attempts if (r['task'],r['model'])==key and r['solver']=='independent_complex_step_LBFGSB' and r['start'].startswith('OLS_')]
    comparison.append(dict(task=task,model=model,n=n,K=o['K'],frozen_status=o['old_status'],frozen_loglik=o['old_loglik'],best_optimized_positive_loglik=ip['loglik'],best_optimized_origin=ip['origin'],best_optimized_sigma_u=ip['su'],best_optimized_sigma_v=ip['sv'],best_positive_including_path_loglik=best_positive,best_positive_minus_frozen=best_positive-o['old_loglik'],endpoint_path_gap=v-float(path['loglik']),endpoint_path_frontier_relative_shift=float(np.expm1(float(path['intercept_shift']))),OLS_start_loglik_spread=max(starts)-min(starts)))
csvout('LABEL_SENSITIVITY.csv',sensitivity);csvout('CASE_COMPARISON.csv',comparison)
engine_summaries=[]
for engine in ['sfaR_nlminb','frontier']:
    rr=[r for r in rrows if r['engine']==engine];bb=[r for r in rbest.values() if r['engine']==engine]
    # Comparison is likelihood consistency at returned parameters, not package convergence certification.
    by_status={s:dict(cases=sum(old[(r['task'],r['model'])]['old_status']==s for r in bb),within_tolerance=sum(old[(r['task'],r['model'])]['old_status']==s and abs(r['canonical_minus_frozen'])<=1e-7*old[(r['task'],r['model'])]['n'] for r in bb)) for s in ['QUALIFIED_SIGMA_V_ZERO','QUALIFIED_SIGMA_U_ZERO','QUALIFIED_INTERIOR']}
    engine_summaries.append(dict(engine=engine,attempts=len(rr),returned=sum(r['status']=='returned' for r in rr),errors=sum(r['status']=='error' for r in rr),finite_recomputed=sum(r['independent_canonical_input_loglik'] is not None for r in rr),returned_but_not_finite_recomputable=sum(r['status']=='returned' and r['independent_canonical_input_loglik'] is None for r in rr),cases=len(bb),better_than_frozen=sum(r['canonical_minus_frozen']>1e-7*old[(r['task'],r['model'])]['n'] for r in bb),best_within_tolerance=sum(abs(r['canonical_minus_frozen'])<=1e-7*old[(r['task'],r['model'])]['n'] for r in bb),by_frozen_status=by_status,return_codes=dict(collections.Counter(str(r['code']) for r in rr)),max_seconds=max(r['seconds'] for r in rr),native_loglik_discrepancy_over_tolerance=sum(r['reported_minus_independent_R_input'] is not None and abs(r['reported_minus_independent_R_input'])>1e-7*old[(r['task'],r['model'])]['n'] for r in rr)))
improved=[r for r in comparison if r['best_positive_minus_frozen']>r['n']*1e-7]
vonly=[r for r in comparison if r['frozen_status']=='QUALIFIED_SIGMA_V_ZERO']
out=dict(sample_fits=len(cases),frozen_status_counts=dict(collections.Counter(r['old_status'] for r in cases)),python_optimization_attempts=len(attempts),R_attempts=len(raw),
    meaningful_improvement_per_observation=1e-7,meaningfully_improved_cases=improved,
    maximum_positive_gain=max(r['best_positive_minus_frozen'] for r in comparison),
    independent_boundary=read(HERE/'INDEPENDENT_BOUNDARY_CHECK.json'),production_gradient=read(HERE/'PRODUCTION_GRADIENT_CHECK.json'),
    R_engines=engine_summaries,
    vzero_cases=len(vonly),vzero_path_max_likelihood_gap=max(r['endpoint_path_gap'] for r in vonly),vzero_path_max_frontier_relative_shift=max(r['endpoint_path_frontier_relative_shift'] for r in vonly),
    label_changes_by_tie={str(t):sum(r['frozen_status']!=r['selected_status'] for r in sensitivity if r['tie_per_observation']==t) for t in [0.,1e-10,1e-8,1e-6]},
    OLS_start_sensitive_cases=sum(r['OLS_start_loglik_spread']>r['n']*1e-7 for r in comparison),
    OLS_start_sensitive_vzero_cases=sum(r['OLS_start_loglik_spread']>r['n']*1e-7 for r in vonly),
    limitations=['Fixed diagnostic sample, not all 108000 SFA fits','Finite multistart search does not prove a global maximum','Raw package return/convergence codes are retained; returned values are not automatically qualified','No performance ranking or production adoption'])
write('SUMMARY.json',out);print(json.dumps(out,ensure_ascii=False,indent=2))
