"""Qualify retried outputs and recompute their original metric views in isolation."""
from pathlib import Path
import json,hashlib
import numpy as np
from fdh_recompute import H,loss_values,clean,dump,sha
records=json.loads((H/'stoned/SUMMARY.json').read_text())['cases']
manifest={r['task']:r for r in json.loads((H/'INPUT_MANIFEST.json').read_text())}
out=[];checks=[]
for rec in records:
    task=rec['task'];p=H/'stoned'/task;parent=Path(manifest[task]['directory'])
    s=json.loads((p/'models/StoNED_QLE/STATUS.json').read_text());meta=s['raw']
    assert rec['exit_code']==0 and s['numerical_qualified'] and s['native_score_qualified']
    assert meta['envelope']['qualified'] and meta['scores_valid'] and meta['score_check']['qualified']
    a=dict(np.load(p/'models/StoNED_QLE/PREDICTIONS.npz'));z=dict(np.load(p/'analysis_truth.npz'));n=len(z['y'])
    ft=np.r_[z['true_frontier'],z['true_frontier_eval']];pred=np.r_[a['frontier_train'],a['frontier_eval']]
    old=json.loads((parent/'METRICS.json').read_text())['values']['StoNED_QLE'];values={k:None for k in old}
    with np.load(p/'MASKS.npz') as masks:
        for domain in masks.files:
            if domain.startswith('full_'):continue
            mask=masks[domain];assert mask.any() and np.isfinite(pred[mask]).all()
            for k,v in loss_values(pred[mask],ft[mask]).items():values['frontier|'+domain+'|'+k]=v
    for view,scores in [('common_raw',z['y']/a['frontier_train']),('native_BC',a['te_BC']),('native_JLMS',a['te_JLMS'])]:
        for k,v in loss_values(scores,z['true_te'],rank=True).items():values['TE_'+view+'|training|'+k]=v
    assert set(values)==set(old)
    oldfirst=np.load(parent/'StoNED_QLE/cnls_tol1e-10_start0.npz');newfirst=np.load(p/'StoNED_QLE/cnls_tol1e-10_start0.npz')
    differences={k:float(np.max(abs(oldfirst[k]-newfirst[k])/(1+abs(oldfirst[k])))) for k in oldfirst.files}
    checks.append(dict(task=task,status=s['status'],qualified=True,seconds=rec['seconds'],CNLS_attempts=meta['attempts'],score_check=meta['score_check'],envelope_qualified=meta['envelope']['qualified'],QLE_profile_agreement=meta['profile_objective_discrepancy_per_obs'],first_CNLS_saved_array_scaled_differences=differences,all_output_hashes_unchanged=all(sha(p/name)==v for name,v in rec['files'].items())))
    assert checks[-1]['all_output_hashes_unchanged']
    out.append(dict(task=task,qualified=True,values=values,before=old,parent_complete_sha256=manifest[task]['complete_sha256']))
with (H/'stoned/REPLACEMENT_METRICS.jsonl').open('w') as f:
    for r in out:f.write(json.dumps(clean(r),allow_nan=False)+'\n')
dump(H/'stoned/VERIFICATION.json',dict(status='PASS',checks=checks,interpretation='Both retries completed under 600 seconds despite a common 1800-second outer cap. This does not establish that extending the cap caused recovery or identify the original wall-time bottleneck. Same scientific algorithm and internal stopping rules retained.'))
print(json.dumps(clean(checks),indent=2))
