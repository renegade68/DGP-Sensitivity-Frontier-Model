from compare import *
pred=table(HERE/'PREDICTION_COMPARISON.csv');sc=table(HERE/'SCORE_COMPARISON.csv');pmap={int(r['index0']):r for r in pred}
native=np.fromfile(HERE/'NATIVE_SFAR_SCORES.f64','<f8');native_rows=[];native_frontiers=[];standard=np.load(HERE/'STANDARDIZED_SCORES.npz');selected={int(r['index0']):r for r in table(HERE/'SELECTED_R.csv')}
par=np.fromfile(PRIOR/'R_PARAMS.f64','<f8');offsets=read(PRIOR/'R_PARAMS_INDEX.json')
sources=read(HERE/'SOURCE_HASHES.json')
for rec in read(HERE/'NATIVE_SFAR_INDEX.json'):
    ix=rec['index0'];n=rec['rows'];a=native[rec['offset']:rec['offset']+n*3].reshape((n,3),order='F');r=selected[ix];p=Path(r['directory'])
    template=p/'sfaR_baseline'/f"{r['model']}.rds";h=sha(template);assert h==read(p/'COMPLETE.json')['files'][str(template.relative_to(p))];sources[str(template)]=h
    obs=np.load(p/'observed.npz');X=design(obs['x'],r['model'].split('_')[1]);off=offsets[ix];b=par[off['offset']:off['offset']+X.shape[1]]
    fp=np.exp(np.sum(X*b,axis=1));ferr=float(np.max(abs(a[:,2]/fp-1)))
    native_frontiers.append(dict(index0=ix,task=r['task'],model=r['model'],native_vs_common_frontier_max_relative_error=ferr))
    for k,name in enumerate(['BC','JLMS']):
        z=standard[str(ix)+'_'+name];ok=np.isfinite(a[:,k]);error=float(np.max(abs(a[ok,k]-z[ok]))) if ok.any() else None
        native_rows.append(dict(index0=ix,task=r['task'],model=r['model'],score=name,finite_points=int(ok.sum()),total_points=n,native_vs_stable_max_abs_error=error))
csvout('NATIVE_SCORE_COMPARISON.csv',native_rows);csvout('NATIVE_FRONTIER_COMPARISON.csv',native_frontiers);write('SOURCE_HASHES.json',sources)

def group_summary(engine,matched,state):
    rr=[r for r in pred if r['engine']==engine and (matched=='ALL' or r['likelihood_matched']==matched) and (state=='ALL' or r['old_status']==state)]
    if not rr:return None
    ids={r['index0'] for r in rr};out=dict(engine=engine,likelihood_matched=matched,old_status=state,cases=len(rr))
    for field in ['beta_max_abs_difference','train_frontier_max_relative_difference','eval_frontier_max_relative_difference']:
        a=[float(r[field]) for r in rr];out[field+'_median']=float(np.median(a));out[field+'_max']=max(a)
    for name in ['BC','JLMS']:
        ss=[r for r in sc if r['index0'] in ids and r['score']==name];ranked=[r for r in ss if r['rank_status']=='defined'];top=[r for r in ss if r['top20_status']=='defined']
        a=[float(r['max_absolute_difference']) for r in ss]
        out[name+'_max_abs_diff_median']=float(np.median(a));out[name+'_max_abs_diff_max']=max(a)
        out[name+'_rank_defined_cases']=len(ranked);out[name+'_rank_constant_cases']=len(ss)-len(ranked)
        out[name+'_spearman_min']=min(float(r['spearman']) for r in ranked) if ranked else None
        out[name+'_discordant_pairs']=sum(int(r['discordant_pairs']) for r in ranked);out[name+'_comparable_pairs']=sum(int(r['comparable_pairs']) for r in ranked);out[name+'_ambiguous_pairs']=sum(int(r['ambiguous_pairs']) for r in ranked)
        out[name+'_top20_defined_cases']=len(top);out[name+'_top20_overlap_min']=min(float(r['top20_overlap']) for r in top) if top else None
    return out
groups=[]
for engine in ['sfaR_nlminb','frontier']:
    for same in ['ALL','True','False']:
        for status in ['ALL','QUALIFIED_INTERIOR','QUALIFIED_SIGMA_U_ZERO','QUALIFIED_SIGMA_V_ZERO']:
            r=group_summary(engine,same,status)
            if r:groups.append(r)
csvout('GROUP_COMPARISON.csv',groups)
thresholds=[]
for engine in ['sfaR_nlminb','frontier']:
    for same in ['True','False']:
        rr=[r for r in pred if r['engine']==engine and r['likelihood_matched']==same];ids={r['index0'] for r in rr}
        for eps in [1e-6,1e-4,1e-3,1e-2]:
            thresholds.append(dict(engine=engine,likelihood_matched=same,threshold=eps,cases=len(rr),frontier_eval_max_relative_within=sum(float(r['eval_frontier_max_relative_difference'])<=eps for r in rr),BC_max_absolute_within=sum(float(r['max_absolute_difference'])<=eps for r in sc if r['index0'] in ids and r['score']=='BC'),JLMS_max_absolute_within=sum(float(r['max_absolute_difference'])<=eps for r in sc if r['index0'] in ids and r['score']=='JLMS')))
csvout('DESCRIPTIVE_THRESHOLDS.csv',thresholds)
finite_errors=[r['native_vs_stable_max_abs_error'] for r in native_rows if r['native_vs_stable_max_abs_error'] is not None]
summary=dict(package_comparisons=len(pred),sfaR_comparisons=96,frontier_comparisons=48,training_score_comparisons=len(sc),
    scoring_validation=read(HERE/'SCORING_VALIDATION.json'),native_sfaR=dict(cases=len(native_rows)//2,all_points_finite=all(r['finite_points']==r['total_points'] for r in native_rows),max_abs_error=max(finite_errors),max_frontier_relative_error=max(r['native_vs_common_frontier_max_relative_error'] for r in native_frontiers)),
    groups=groups,interpretation_limits=['Fixed 96-model diagnostic sample, not the full experiment','Best of three saved starts; not single default package runs','Likelihood-matched and unmatched cases both retained','Standardized BC/JLMS formulas at package parameters; native sfaR routine checked separately','Rank comparisons concern units within each dataset, not rankings of estimation methods','All-constant or near-constant scores have undefined ranks; ties are not broken by row order','Tolerance-based pairwise diagnostic does not replace the frozen aggregate ranking metric'])
write('SUMMARY.json',summary)
for r in groups:
    if r['likelihood_matched']=='True':print(json.dumps(r))
print('native',summary['native_sfaR'])
