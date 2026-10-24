"""Recompute only affected model/method/factor results, retaining parent references."""
from pathlib import Path
import argparse,ast,collections,csv,functools,hashlib,json,math,time
import numpy as np
from scipy.stats import t as student_t
from fdh_recompute import H,clean,dump,loss_values,sha

src=H/'runtime/tools/aggregate.py';names={'summary','paired_summary','welch_summary','metric_key','value'}
nodes=[n for n in ast.parse(src.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
env={'np':np,'student_t':student_t,'math':math};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(src),'exec'),env)
summary,paired,welch,mkey=[env[n] for n in ['summary','paired_summary','welch_summary','metric_key']]
FACTOR_KEYS=['frontier|FDH_box|NMSE','frontier|FDH_box|MAE','TE_common_raw|training|MAE','TE_common_raw|training|MSE','TE_common_raw|training|Spearman','TE_common_raw|training|Kendall','TE_native_BC|training|MAE','TE_native_JLMS|training|MAE']
def val(values,model,key):
    v=values.get(mkey(model,key));return None if v is None else 1-v if key.endswith(('|Spearman','|Kendall')) else v
def rowwrite(f,r):f.write(json.dumps(clean(r),allow_nan=False)+'\n')
def compare(a,b):
    for k in ['n','mean','MCSE','CI95','attempted','available_A','available_B','availability_2x2']:
        if k not in a or k not in b:continue
        x,y=a[k],b[k]
        if x is None or y is None:assert x==y,(k,x,y)
        elif isinstance(x,(int,float,list)):
            assert np.allclose(x,y,rtol=1e-10,atol=1e-12,equal_nan=True),(k,x,y)
        else:assert x==y,(k,x,y)

def run(include_stoned=False):
    assert json.loads((H/'fdh/full/SUMMARY.json').read_text())['datasets']==27000
    out=H/('reports/combined' if include_stoned else 'reports/fdh_only');out.mkdir(exist_ok=False)
    tasks=json.loads((H/'INPUT_MANIFEST.json').read_text());parents={r['task']:Path(r['directory']) for r in tasks}
    cells=json.loads((H/'runtime/design/EFFECTIVE_DGP.json').read_text())['cells'];bycell={c['cell_id']:c for c in cells}
    fdhr=({r['task']:{'cell':r['cell']} for r in tasks} if include_stoned else {r['task']:r for r in map(json.loads,(H/'fdh/full/METRICS.jsonl').open())})
    originals={task:({} if include_stoned else {'FDH':r['FDH_before'],'EAT':r['EAT']}) for task,r in fdhr.items()}
    changes={task:({} if include_stoned else {'FDH':r['FDH_after']}) for task,r in fdhr.items()}
    repair_models=set() if include_stoned else {'FDH'};sto_tasks=set()
    if include_stoned:
        for r in map(json.loads,(H/'stoned/REPLACEMENT_METRICS.jsonl').open()):
            if r['qualified']:changes[r['task']]['StoNED_QLE']=r['values'];sto_tasks.add(r['task'])
        if sto_tasks:repair_models.add('StoNED_QLE')
    contract=json.loads((H/'runtime/design/CONTRASTS.json').read_text())
    pairs=[pair for pair in contract['primary_method_pairs'] if set(pair)&repair_models]
    relevant=repair_models|{m for pair in pairs for m in pair}
    touched_cells={fdhr[t]['cell'] for t in sto_tasks}
    needed_cells=set(touched_cells)
    for c in contract['factor_contrasts']:
        if {c['A_cell'],c['B_cell']}&touched_cells:needed_cells.update([c['A_cell'],c['B_cell']])
    if include_stoned:
        for task,p in parents.items():
            if fdhr[task]['cell'] not in needed_cells:continue
            raw=json.loads((p/'METRICS.json').read_text())
            for m in relevant:originals[task][m]=raw['values'][m]
    groups=collections.defaultdict(list)
    for r in tasks:groups[r['cell']].append(r)
    aggregate=Path(json.loads((H.parent/'frontier_full_run_v4_20261012/RESULTS.json').read_text())['aggregate']).parent
    old_methods={};old_models={};old_factors={}
    for filename,dest,predicate,key in [
        ('METHOD_CONTRASTS.jsonl',old_methods,lambda r:(r['A'],r['B']) in map(tuple,pairs),lambda r:(r['cell'],r['window'],r['A'],r['B'],r['metric_id'])),
        ('MODEL_SUMMARIES.jsonl',old_models,lambda r:r['model'] in repair_models,lambda r:(r['cell'],r['window'],r['model'],r['metric_id']))]:
        for line in (aggregate/filename).open():
            r=json.loads(line)
            if predicate(r):dest[key(r)]=r
    def changed_factor(c,models):
        if 'FDH' in models:return True
        return 'StoNED_QLE' in models and any(fdhr[t]['cell'] in [c['A_cell'],c['B_cell']] for t in sto_tasks)
    print('Reading preserved factor summaries',flush=True)
    for line in (aggregate/'FACTOR_CONTRASTS.jsonl').open():
        r=json.loads(line)
        if not set(r['models'])&repair_models or not changed_factor(r['contrast'],r['models']):continue
        records=r.pop('domain_records',[])
        r['domain_records_count']=len(records)
        old_factors[(r['contrast']['contrast_id'],r['window'],tuple(r['models']),r['metric_id'])]=r
    new_methods={};new_models={};checks=0
    with (out/'MODEL_SUMMARIES.jsonl').open('w') as fm,(out/'METHOD_CONTRASTS.jsonl').open('w') as fp:
        if include_stoned:
            for filename,stream,mapping,keys in [('MODEL_SUMMARIES.jsonl',fm,new_models,['cell','window','model','metric_id']),('METHOD_CONTRASTS.jsonl',fp,new_methods,['cell','window','A','B','metric_id'])]:
                for r in map(json.loads,(H/'reports/fdh_only'/filename).open()):
                    mapping[tuple(r[k] for k in keys)]=r;rowwrite(stream,r)
        for cell,rs in sorted(groups.items()):
            if include_stoned and cell not in touched_cells:continue
            rs=sorted(rs,key=lambda r:r['rep'])
            for window,limit in [('base_200',200)]+([('precision_500',500)] if len(rs)==500 else []):
                chosen=[r for r in rs if r['rep']<=limit]
                for model in sorted(repair_models):
                    keys=sorted(set.union(*(set(originals[r['task']][model]) for r in chosen)))
                    for key in keys:
                        old=[originals[r['task']][model].get(key) for r in chosen]
                        new=[changes[r['task']].get(model,originals[r['task']][model]).get(key) for r in chosen]
                        before=summary([v for v in old if v is not None]);after=summary([v for v in new if v is not None])
                        k=(cell,window,model,key);compare(before,old_models[k]);checks+=1
                        row=dict(cell=cell,window=window,model=model,metric_id=key,attempted=len(chosen),available=after['n'],**after)
                        new_models[k]=row;rowwrite(fm,dict(**row,before=before))
                for A,B in pairs:
                    keys=sorted(set.union(*(set(originals[r['task']][A])|set(originals[r['task']][B]) for r in chosen)))
                    for key in keys:
                        if key.startswith('TE_limit_'):continue
                        if key.startswith('frontier|DEA_') and any(m in ('EAT','FDH') for m in (A,B)):continue
                        if key.startswith('frontier|full_') and any(m in ('EAT','FDH','DEA_VRS','CEAT','StoNED_QLE') for m in (A,B)):continue
                        results=[]
                        for changed in [False,True]:
                            av=[];bv=[]
                            for r in chosen:
                                task=r['task']
                                for m,dest in [(A,av),(B,bv)]:dest.append(val(changes[task].get(m,originals[task][m]) if changed else originals[task][m],m,key))
                            results.append(paired(av,bv))
                        before,after=results;k=(cell,window,A,B,key);compare(before,old_methods[k]);checks+=1
                        row=dict(cell=cell,window=window,A=A,B=B,metric_id=key,**after);new_methods[k]=row
                        rowwrite(fp,dict(**row,before=before))
    print('Model and method summaries complete; validated',checks,'parent summaries',flush=True)
    fdhpred=None if include_stoned else np.load(H/'fdh/full/PREDICTIONS.npz')
    @functools.lru_cache(maxsize=1000)
    def arrays(task):
        p=parents[task]
        with np.load(p/'analysis_truth.npz') as z:
            n=len(z['y']);truth=z['true_frontier_eval'][:5000];xe=z['x_eval'][:5000]
        with np.load(p/'MASKS.npz') as z:mask=z['FDH_box'][n:][:5000]
        with np.load(p/'ALL_PREDICTIONS.npz') as z:old={m:z[m][n:][:5000] for m in relevant}
        new={} if include_stoned else {'FDH':fdhpred[task][n:][:5000]}
        if task in sto_tasks:
            with np.load(H/'stoned'/task/'models/StoNED_QLE/PREDICTIONS.npz') as z:new['StoNED_QLE']=z['frontier_eval'][:5000]
        return truth,xe,mask,old,new
    factor_rows=[]
    with (out/'FACTOR_CONTRASTS.jsonl').open('w') as ff:
        if include_stoned:
            for r in map(json.loads,(H/'reports/fdh_only/FACTOR_CONTRASTS.jsonl').open()):factor_rows.append(r);rowwrite(ff,r)
        for idx,c in enumerate(contract['factor_contrasts'],1):
            targets=[m for m in [[m] for m in sorted(repair_models)]+pairs if changed_factor(c,m)]
            if not targets:continue
            ca,cb=c['A_cell'],c['B_cell'];aa={r['rep']:r['task'] for r in groups[ca]};bb={r['rep']:r['task'] for r in groups[cb]}
            cached={False:{},True:{}};domains={}
            required={m for models in targets for m in models}
            for rep in sorted(r for r in set(aa)|set(bb) if r <= (500 if c['precision_reps'] else 200)):
                common=c['paired'] and 'intersection' in c['frontier_domain']
                mask=None
                if common:
                    za,zb=arrays(aa[rep]),arrays(bb[rep]);assert np.array_equal(za[1],zb[1]);mask=za[2]&zb[2];domains[rep]=int(mask.sum())
                for changed in [False,True]:
                    vals=[]
                    for task in [aa.get(rep),bb.get(rep)]:
                        one={}
                        for model in required:
                            raw=changes[task].get(model,originals[task][model]) if changed else originals[task][model]
                            one[model]={k:val(raw,model,k) for k in FACTOR_KEYS}
                            if common:
                                truth,xe,ownmask,op,npred=arrays(task)
                                available=raw.get('frontier|FDH_box|NMSE') is not None and mask.any()
                                pred=npred.get(model,op[model]) if changed else op[model]
                                ls=loss_values(pred[mask],truth[mask]) if available else None
                                for k in FACTOR_KEYS:
                                    if k.startswith('frontier|'):one[model][k]=ls[k.rsplit('|',1)[1]] if ls else None
                        vals.append(one)
                    cached[changed][rep]=vals
            for window,limit in [('base_200',200)]+([('precision_500',500)] if c['precision_reps'] else []):
                for models in targets:
                    for key in FACTOR_KEYS:
                        def target(v):
                            a=[v[m][key] for m in models];return None if any(x is None for x in a) else a[0] if len(a)==1 else a[0]-a[1]
                        ans=[]
                        for changed in [False,True]:
                            av=[target(v[0]) for rep,v in sorted(cached[changed].items()) if rep<=limit and rep in aa]
                            bv=[target(v[1]) for rep,v in sorted(cached[changed].items()) if rep<=limit and rep in bb]
                            ans.append(paired(av,bv) if c['paired'] else welch(av,bv))
                        before,after=ans;k=(c['contrast_id'],window,tuple(models),key)
                        compare(before,old_factors[k]);checks+=1
                        dr=[v for rep,v in domains.items() if rep<=limit]
                        row=dict(contrast=c,window=window,models=models,metric_id=key,domain_points_mean=float(np.mean(dr)) if dr else None,domain_records_count=len(dr),before=before,**after)
                        factor_rows.append(row);rowwrite(ff,row)
            if idx%10==0:print('FACTORS',idx,'/',len(contract['factor_contrasts']),flush=True)
    if fdhpred is not None:fdhpred.close()
    # Recompute both existing Holm families after replacing affected contrasts.
    source=H.parent/'replication_archive_v7/current/data/METHOD_CONTRASTS.csv'
    rows=[r for r in csv.DictReader(source.open()) if r['window']=='base_200' and r['metric_id']=='frontier|FDH_box|NMSE']
    for r in rows:
        k=(r['cell'],r['window'],r['A'],r['B'],r['metric_id'])
        if k in new_methods:
            z=new_methods[k];r.update(mean_difference=z['mean'],MCSE=z['MCSE'],both=z['n'],CI95_low=z['CI95'][0],CI95_high=z['CI95'][1])
        r['p']=float(2*student_t.sf(abs(float(r['mean_difference'])/float(r['MCSE'])),int(r['both'])-1))
    def holm(rs,key):
        p=np.array([r['p'] for r in rs]);ix=np.argsort(p,kind='stable');q=np.empty(len(p));q[ix]=np.minimum(1,np.maximum.accumulate(p[ix]*(len(p)-np.arange(len(p)))))
        for r,v in zip(rs,q):r[key]=float(v)
    core=[r for r in rows if r['panel']=='P1_core' and r['regime']=='mixed'];assert len(rows)==2457 and len(core)==756
    holm(rows,'holm_all');holm(core,'holm_core');counts=[]
    for A,B in contract['primary_method_pairs']:
        rr=[r for r in core if (r['A'],r['B'])==(A,B)];d=dict(A=A,B=B)
        for k,label in [('p','pointwise'),('holm_core','holm_core'),('holm_all','holm_all')]:
            neg=sum(r[k]<.05 and float(r['mean_difference'])<0 for r in rr);pos=sum(r[k]<.05 and float(r['mean_difference'])>0 for r in rr);d[label]=[neg,pos,len(rr)-neg-pos]
        counts.append(d)
    dump(out/'MULTIPLICITY.json',dict(families={'core':756,'all':2457},core_counts=counts,core_total_rejected={k:sum(r[k]<.05 for r in core) for k in ['p','holm_core','holm_all']}))
    dump(out/'FRONTIER_PRIMARY_TESTS.json',rows)
    dump(out/'SUMMARY.json',dict(status='PASS',models=sorted(repair_models|{'FDH'}),recovered_stoned_tasks=sorted(sto_tasks),validated_parent_summary_count=checks,inherited_FDH_validation_count=json.loads((H/'reports/fdh_only/SUMMARY.json').read_text())['validated_parent_summary_count'] if include_stoned else 0,model_rows=len(new_models),method_rows=len(new_methods),factor_rows=len(factor_rows),unaffected_results='Inherited unchanged from frozen aggregate; this is a replacement overlay, not an overwrite',parent_manifest_sha256=sha(aggregate/'MANIFEST.json'),files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--include-stoned',action='store_true');a=ap.parse_args();run(a.include_stoned)
