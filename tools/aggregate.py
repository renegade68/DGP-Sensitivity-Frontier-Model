"""Stream condition and factor summaries from full precision values, never tables."""
from context import *
from metrics import loss_values
from scipy.stats import t as student_t
import argparse,collections,itertools,math

def summary(v):
    v=np.asarray(v,dtype=float);assert np.isfinite(v).all();n=len(v)
    if not n:return dict(n=0,mean=None,MCSE=None,CI95=None)
    mean=float(v.mean());se=float(v.std(ddof=1)/np.sqrt(n)) if n>1 else None
    return dict(n=n,mean=mean,MCSE=se,CI95=[mean-student_t.ppf(.975,n-1)*se,mean+student_t.ppf(.975,n-1)*se] if se is not None else None,
        median=float(np.median(v)),q90=float(np.quantile(v,.9)),q95=float(np.quantile(v,.95)))
def paired_summary(a,b):
    assert len(a)==len(b);counts={'both':0,'A_only':0,'B_only':0,'neither':0};d=[]
    for x,y in zip(a,b):
        key='both' if x is not None and y is not None else ('A_only' if x is not None else ('B_only' if y is not None else 'neither'));counts[key]+=1
        if key=='both':d.append(x-y)
    ans=summary(d);ans.update(attempted=len(a),availability_2x2=counts)
    if d:
        n=len(d);wins=sum(x<0 for x in d);ties=sum(x==0 for x in d);p=wins/n;z=1.959963984540054;den=1+z*z/n;mid=(p+z*z/(2*n))/den;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        ans.update(negative_fraction=p,exact_ties=ties,positive_fraction=sum(x>0 for x in d)/n,negative_fraction_Wilson95=[mid-h,mid+h])
    return ans
def welch_summary(a,b):
    av=[x for x in a if x is not None];bv=[x for x in b if x is not None];sa,sb=summary(av),summary(bv)
    ans=dict(paired=False,attempted_A=len(a),attempted_B=len(b),available_A=len(av),available_B=len(bv),mean=None,MCSE=None,CI95=None)
    if not av or not bv:return ans
    ans['mean']=sa['mean']-sb['mean']
    if len(av)>1 and len(bv)>1:
        va=np.var(av,ddof=1)/len(av);vb=np.var(bv,ddof=1)/len(bv);se=np.sqrt(va+vb);df=(va+vb)**2/(va*va/(len(av)-1)+vb*vb/(len(bv)-1)) if va+vb>0 else math.inf
        delta=0. if se==0 else student_t.ppf(.975,df)*se;ans.update(MCSE=float(se),degrees_of_freedom=None if not np.isfinite(df) else float(df),CI95=[ans['mean']-delta,ans['mean']+delta])
    return ans
def metric_key(model,key):
    if not model.startswith('SFA_') and model!='StoNED_QLE':return key.replace('TE_native_BC|','TE_common_raw|').replace('TE_native_JLMS|','TE_common_raw|')
    return key
def value(row,model,key):
    v=row['values'][model].get(metric_key(model,key));return None if v is None else (1-v if key.endswith(('|Spearman','|Kendall')) else v)
def write_line(f,row):f.write(json.dumps(clean(row),ensure_ascii=False,allow_nan=False)+'\n')
def index(run,include_engineering=False):
    idx={};excluded=[]
    for p in sorted(Path(run).glob('*/attempt_*/COMPLETE.json')):
        task=p.parent;cfg=read(task/'TASK.json')
        if cfg['cohort']=='cost_only' or (cfg['cohort']!='production' and not include_engineering):excluded.append(cfg['task_id']);continue
        key=(cfg['cell_id'],cfg['replication_id']);assert key not in idx;done=read(p)
        for name in ('METRICS.json','AUDIT.json','ALL_PREDICTIONS.npz','MASKS.npz','analysis_truth.npz'):
            assert sha(task/name)==done['files'][name]
        idx[key]=task
    return idx,excluded

def factor_values(pa,pb,models,key,common_domain):
    ra,rb=read(pa/'METRICS.json'),read(pb/'METRICS.json')
    if key.startswith('frontier|') and common_domain:
        _,domain,metric=key.split('|');za,zb=np.load(pa/'analysis_truth.npz'),np.load(pb/'analysis_truth.npz');na,nb=len(za['y']),len(zb['y']);ma=np.load(pa/'MASKS.npz')[domain][na:][:5000];mb=np.load(pb/'MASKS.npz')[domain][nb:][:5000];mask=ma&mb
        assert np.array_equal(za['x_eval'][:5000],zb['x_eval'][:5000]);ap,bp=np.load(pa/'ALL_PREDICTIONS.npz'),np.load(pb/'ALL_PREDICTIONS.npz')
        def one(row,p,z,n):
            vv=[]
            for model in models:
                # Existing qualified/available status is required, never shrink a failure's mask.
                if value(row,model,key) is None or not mask.any():return None
                vv.append(loss_values(p[model][n:][:5000][mask],z['true_frontier_eval'][:5000][mask])[metric])
            return vv[0] if len(vv)==1 else vv[0]-vv[1]
        return one(ra,ap,za,na),one(rb,bp,zb,nb),dict(points=int(mask.sum()),mask_sha256=arrays_hash({'mask':mask}))
    def one(r):
        vals=[value(r,m,key) for m in models]
        return None if any(v is None for v in vals) else vals[0] if len(vals)==1 else vals[0]-vals[1]
    return one(ra),one(rb),None

def factor_cache(pa,pb,keys,common_domain):
    """Read each raw array once per matched dataset pair, not per contrast target."""
    ra,rb=read(pa/'METRICS.json'),read(pb/'METRICS.json')
    va={m:{k:value(ra,m,k) for k in keys} for m in MODELS};vb={m:{k:value(rb,m,k) for k in keys} for m in MODELS};domain=None
    if common_domain:
        za,zb=np.load(pa/'analysis_truth.npz'),np.load(pb/'analysis_truth.npz');na,nb=len(za['y']),len(zb['y'])
        assert np.array_equal(za['x_eval'][:5000],zb['x_eval'][:5000])
        mask=np.load(pa/'MASKS.npz')['FDH_box'][na:][:5000]&np.load(pb/'MASKS.npz')['FDH_box'][nb:][:5000]
        domain=dict(points=int(mask.sum()),mask_sha256=arrays_hash({'mask':mask}))
        for p,z,n,row,dest in ((pa,za,na,ra,va),(pb,zb,nb,rb,vb)):
            pred=np.load(p/'ALL_PREDICTIONS.npz')
            for model in MODELS:
                available=value(row,model,'frontier|FDH_box|NMSE') is not None and mask.any()
                ls=loss_values(pred[model][n:][:5000][mask],z['true_frontier_eval'][:5000][mask]) if available else None
                for key in keys:
                    if key.startswith('frontier|FDH_box|'):dest[model][key]=ls[key.rsplit('|',1)[1]] if ls else None
    return va,vb,domain

def main(run,out,engineering=False):
    run,out=Path(run),Path(out);out.mkdir(parents=True,exist_ok=False);idx,excluded=index(run,engineering);contract=read(ROOT/'design/CONTRASTS.json');pairs=contract['primary_method_pairs'];groups=collections.defaultdict(list)
    for (cell,rep),p in idx.items():groups[cell].append((rep,p))
    counts=collections.Counter();parent_records=[]
    with (out/'MODEL_SUMMARIES.jsonl').open('w') as fm,(out/'METHOD_CONTRASTS.jsonl').open('w') as fc,(out/'DIAGNOSTICS.jsonl').open('w') as fd:
        for cell,items in sorted(groups.items()):
            raw=[read(p/'METRICS.json') for rep,p in sorted(items)];parent_records.extend({'task':r['task_id'],'metrics_sha256':sha(p/'METRICS.json')} for r,(_,p) in zip(raw,sorted(items)))
            windows=[('base_200',[r for r in raw if r['replication_id']<=200])]
            if next(c for c in read(ROOT/'design/EFFECTIVE_DGP.json')['cells'] if c['cell_id']==cell)['target_replications']==500:windows.append(('precision_500',raw))
            for window,rows in windows:
                if not rows:continue
                for model in MODELS:
                    for key in sorted(set.union(*(set(r['values'][model]) for r in rows))):
                        vals=[r['values'][model].get(key) for r in rows];ans=summary([v for v in vals if v is not None]);write_line(fm,dict(cell=cell,window=window,model=model,metric_id=key,attempted=len(rows),available=ans['n'],**ans));counts['model_summary_rows']+=1
                for A,B in pairs:
                    keys=set.union(*(set(r['values'][A])|set(r['values'][B]) for r in rows))
                    for key in sorted(keys):
                        if '|limit_' in key or key.startswith('TE_limit_'):continue
                        if key.startswith('frontier|DEA_') and any(m in ('EAT','FDH') for m in (A,B)):continue
                        if key.startswith('frontier|full_') and any(m in ('EAT','FDH','DEA_VRS','CEAT','StoNED_QLE') for m in (A,B)):continue
                        av=[value(r,A,key) for r in rows];bv=[value(r,B,key) for r in rows]
                        write_line(fc,dict(cell=cell,window=window,A=A,B=B,metric_id=key,source_metric_A=metric_key(A,key),source_metric_B=metric_key(B,key),rank_direction='1-correlation' if key.endswith(('|Spearman','|Kendall')) else None,pointwise_not_multiplicity_adjusted=True,**paired_summary(av,bv)));counts['method_contrast_rows']+=1
            for form in ('CD','TL'):
                ds=[r['OLS'][form] for r in raw];write_line(fd,dict(cell=cell,diagnostic='OLS_'+form,attempted=len(ds),positive_skewness=sum(d['positive_skewness'] is True for d in ds),undefined=sum(d['skewness'] is None for d in ds)))
            audits=[read(p/'AUDIT.json') for rep,p in items]
            for model in MODELS:
                states=[a['model_status'][model] for a in audits]
                write_line(fd,dict(cell=cell,diagnostic='attempt_availability_and_boundaries',model=model,attempted=len(states),finite=sum(s['finite'] for s in states),package_converged_true=sum(s['package_converged'] is True for s in states),package_converged_unknown=sum(s['package_converged'] is None for s in states),numerical_qualified=sum(s['numerical_qualified'] for s in states),identified=sum(s['decomposition_identified'] is True for s in states),degenerate_limits=sum(s['degenerate_limit'] for s in states),statuses=dict(collections.Counter(s['status'] for s in states))))
            write_line(fd,dict(cell=cell,diagnostic='support',datasets=len(audits),point_count_summaries={k:summary([a['support_counts'][k] for a in audits if k in a['support_counts']]) for k in sorted(set.union(*(set(a['support_counts']) for a in audits)))}))
            for A,B in itertools.combinations(MODELS,2):
                ds=[d for r in raw for d in r['agreement'] if d['A']==A and d['B']==B]
                write_line(fd,dict(cell=cell,diagnostic='common_raw_agreement',A=A,B=B,attempted=len(ds),MAD=summary([d['mean_absolute_difference'] for d in ds if d['mean_absolute_difference'] is not None]),Spearman=summary([d['Spearman'] for d in ds if d['Spearman'] is not None])))
    # Factor results use predeclared primary loss keys. All model and method-gap targets.
    factor_keys=['frontier|FDH_box|NMSE','frontier|FDH_box|MAE','TE_common_raw|training|MAE','TE_common_raw|training|MSE','TE_common_raw|training|Spearman','TE_common_raw|training|Kendall','TE_native_BC|training|MAE','TE_native_JLMS|training|MAE']
    with (out/'FACTOR_CONTRASTS.jsonl').open('w') as ff:
        for c in contract['factor_contrasts']:
            aa={r:p for (cell,r),p in idx.items() if cell==c['A_cell']};bb={r:p for (cell,r),p in idx.items() if cell==c['B_cell']}
            if not aa or not bb:continue
            cached={};domains={}
            if c['paired']:
                for rep in sorted(set(aa)&set(bb)):
                    a,b,d=factor_cache(aa[rep],bb[rep],factor_keys,'intersection' in c['frontier_domain']);cached[rep]=(a,b)
                    if d:domains[rep]=dict(replication_id=rep,**d)
            else:
                independent=[]
                for collection in (aa,bb):
                    rr={}
                    for rep,p in collection.items():
                        row=read(p/'METRICS.json');rr[rep]={m:{k:value(row,m,k) for k in factor_keys} for m in MODELS}
                    independent.append(rr)
            windows=[('base_200',200)]+([('precision_500',500)] if c['precision_reps'] else [])
            for window,limit in windows:
                for models in [[m] for m in MODELS]+pairs:
                    for key in factor_keys:
                        av=[];bv=[];domain_records=[d for rep,d in sorted(domains.items()) if rep<=limit]
                        def target(v):
                            vals=[v[m][key] for m in models]
                            return None if any(a is None for a in vals) else vals[0] if len(vals)==1 else vals[0]-vals[1]
                        if c['paired']:
                            # Count missing counterpart attempts separately; never pair by different IDs.
                            for rep in sorted(set(aa)|set(bb)):
                                if rep>limit:continue
                                if rep not in aa or rep not in bb:continue
                                a,b=cached[rep];av.append(target(a));bv.append(target(b))
                            ans=paired_summary(av,bv)
                            ans.update(unattempted_counterpart_A=[r for r in sorted(set(bb)-set(aa)) if r<=limit],unattempted_counterpart_B=[r for r in sorted(set(aa)-set(bb)) if r<=limit])
                        else:
                            for collection,dest in zip(independent,(av,bv)):
                                for rep,vals in sorted(collection.items()):
                                    if rep>limit:continue
                                    dest.append(target(vals))
                            ans=welch_summary(av,bv)
                        write_line(ff,dict(contrast=c,window=window,models=models,metric_id=key,domain_records=domain_records,**ans));counts['factor_contrast_rows']+=1
    write(out/'MANIFEST.json',dict(status='COMPLETE',engineering_only=engineering,full_production_scope_complete=not engineering and len(idx)==27000,datasets=len(idx),excluded_tasks=excluded,counts=dict(counts),parents=parent_records,contract_sha256=sha(ROOT/'design/CONTRASTS.json'),files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    return dict(counts)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run');ap.add_argument('out');ap.add_argument('--engineering',action='store_true');a=ap.parse_args();print(main(a.run,a.out,a.engineering))
