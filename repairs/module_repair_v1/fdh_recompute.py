"""Uniform FDH direct enumeration on exact R-read inputs, in a separate result layer."""
from pathlib import Path
import argparse,ast,collections,hashlib,io,json,time,zipfile
import numpy as np
from scipy.stats import spearmanr,kendalltau
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,(np.integer,np.bool_)):return v.item()
    if isinstance(v,(np.floating,float)):return float(v) if np.isfinite(v) else None
    return v
def dump(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(clean(v),indent=2,allow_nan=False)+'\n');tmp.replace(p)
source=H/'runtime/tools/metrics.py'
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='loss_values')
env={'np':np,'spearmanr':spearmanr,'kendalltau':kendalltau}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),env)
loss_values=env['loss_values']

def direct_reference(x,y,q):
    # Independent reference: descending-y first eligible reference.
    out=np.full(len(q),np.nan)
    for i in np.argsort(-y,kind='stable'):
        hit=np.isnan(out)&np.all(q>=x[i],axis=1);out[hit]=y[i]
    return out

def run(pilot=False):
    manifest=json.loads((H/'INPUT_MANIFEST.json').read_text());exports=H/'input_exports'
    assert json.loads((exports/'COMPLETE.json').read_text())['datasets']==27000
    ridx={r['task']:r for r in map(json.loads,(exports/'R_INDEX.jsonl').read_text().splitlines())}
    bad={r['task'] for r in json.loads((H/'ORIGINAL_UNAVAILABLE.json').read_text()) if r['model']=='FDH'}
    selected=manifest
    if pilot:
        ids={r['task'] for r in manifest[:117]}|bad;selected=[r for r in manifest if r['task'] in ids]
    out=H/('fdh/pilot' if pilot else 'fdh/full');out.mkdir(exist_ok=False)
    mem=np.memmap(exports/'R_INPUTS.bin',dtype='<f8',mode='r');panels={};eligible_cache=collections.OrderedDict()
    start=time.monotonic();failures=0;max_old=0.;checks=0;changed_qualified=0;hashrecords=[]
    with (out/'METRICS.jsonl').open('w') as rows,zipfile.ZipFile(out/'PREDICTIONS.npz','w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as predzip:
        for i,r in enumerate(selected,1):
            p=Path(r['directory']);cfg=ridx[r['task']];n,m=r['n'],r['m']
            assert sha(p/'COMPLETE.json')==r['complete_sha256']
            used=['TASK.json','train.csv','eval.csv','FDH_crosscheck.rds','analysis_truth.npz','METRICS.json','MASKS.npz','ALL_PREDICTIONS.npz','R_predictions.f64','R_predictions_shape.json']
            for name in used:assert sha(p/name)==r['files'][name],(r['task'],name)
            offset=cfg['offset']//8;x=mem[offset:offset+n*m].reshape((n,m),order='F');y=mem[offset+n*m:offset+n*m+n]
            panel=cfg['eval_sha256']
            if panel not in panels:panels[panel]=np.fromfile(exports/(panel+'.f64'),dtype='<f8').reshape((cfg['eval_rows'],m),order='F')
            xe=panels[panel];q=np.vstack([x,xe]);key=(hashlib.sha256(x.tobytes()).hexdigest(),panel)
            if key in eligible_cache:
                eligible=eligible_cache.pop(key)
            else:eligible=np.all(x[None,:,:]<=q[:,None,:],axis=2)
            eligible_cache[key]=eligible
            if len(eligible_cache)>24:eligible_cache.popitem(last=False)
            direct=np.where(eligible,y[None,:],-np.inf).max(axis=1);direct[~np.isfinite(direct)]=np.nan
            canonical=np.nan_to_num(direct,nan=0).astype('<f8').tobytes()
            assert hashlib.sha256(canonical).hexdigest()==cfg['saved_direct_sha256'],('R saved direct differs',r['task'])
            if pilot:
                assert np.array_equal(direct,direct_reference(x,y,q),equal_nan=True);checks+=1
            with np.load(p/'analysis_truth.npz') as z:
                truth=np.r_[z['true_frontier'],z['true_frontier_eval']];true_te=z['true_te'];observed_y=z['y']
            raw=json.loads((p/'METRICS.json').read_text());old=raw['values']['FDH'];new={k:None for k in old}
            with np.load(p/'MASKS.npz') as masks:
                assert np.array_equal(np.isfinite(direct[n:]),masks['FDH_box'][n:] | (masks['FDH_copula'][n:] if 'FDH_copula' in masks.files else False))
                for domain in masks.files:
                    if domain.startswith(('DEA_','full_')):continue
                    mask=masks[domain]
                    assert np.isfinite(direct[mask]).all()
                    values=loss_values(direct[mask],truth[mask]) if mask.any() else {}
                    for k,v in values.items():new['frontier|'+domain+'|'+k]=v
            score=observed_y/direct[:n]
            for k,v in loss_values(score,true_te,rank=True).items():new['TE_common_raw|training|'+k]=v
            qualified_before=old['frontier|FDH_box|NMSE'] is not None
            failures+=int(not qualified_before)
            with np.load(p/'ALL_PREDICTIONS.npz') as z:
                oldpred=z['FDH'];mask=np.isfinite(direct)&np.isfinite(oldpred)
                err=float(np.max(np.abs(oldpred[mask]-direct[mask]),initial=0.))
                max_old=max(max_old,err)
                changed_qualified+=int(qualified_before and not np.array_equal(oldpred,direct,equal_nan=True))
            entry=dict(task=r['task'],cell=r['cell'],rep=r['rep'],FDH_before=old,FDH_after=new,EAT=raw['values']['EAT'],old_available=qualified_before,old_max_finite_prediction_difference=err,direct_reference_sha256=cfg['saved_direct_sha256'],parent_complete_sha256=r['complete_sha256'])
            rows.write(json.dumps(clean(entry),allow_nan=False)+'\n')
            stream=io.BytesIO();np.save(stream,direct,allow_pickle=False);predzip.writestr(r['task']+'.npy',stream.getvalue())
            hashrecords.append({'task':r['task'],'complete_sha256':r['complete_sha256'],'read_file_count':len(used)})
            if i%100==0 or i==len(selected):
                rows.flush();elapsed=time.monotonic()-start
                dump(out/'PROGRESS.json',dict(status='RUNNING',completed=i,total=len(selected),seconds=elapsed,projected_total_seconds=elapsed*len(selected)/i,recovered_original_exclusions=failures))
                print('FDH',i,'/',len(selected),'seconds',round(elapsed,1),flush=True)
    dump(out/'INPUT_CHECKS.json',hashrecords)
    dump(out/'PROGRESS.json',dict(status='COMPLETE',completed=len(selected),total=len(selected),seconds=time.monotonic()-start,recovered_original_exclusions=failures))
    dump(out/'SUMMARY.json',dict(status='PASS',datasets=len(selected),recovered_original_exclusions=failures,all_new_primary_frontier_losses_available=True,all_archived_R_direct_hashes_match=True,independent_algorithm_pilot_checks=checks,changed_originally_qualified_prediction_arrays=changed_qualified,max_old_finite_prediction_difference=max_old,seconds=time.monotonic()-start,scientific_loss_function_sha256=sha(source),estimator_scope='FDH only',files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--pilot',action='store_true');args=a.parse_args();run(args.pilot)
