"""Prepare an isolated repair study; production sources and results are read-only."""
from pathlib import Path
import csv,hashlib,json,shutil,time
H=Path(__file__).resolve().parent;V=H.parent
P=V/'frontier_production_v4_20261012';RUN=P/'runs/production001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
for n in ['input_exports','fdh','stoned','reports','runtime']: (H/n).mkdir(exist_ok=True)
tasks=json.loads((P/'design/PRODUCTION_TASKS.json').read_text())
rows=[];sources={};start=time.monotonic()
for i,t in enumerate(tasks,1):
    matches=list((RUN/t['task']).glob('attempt_*/COMPLETE.json'));assert len(matches)==1,t
    complete=matches[0];p=complete.parent;d=json.loads(complete.read_text())
    names=['TASK.json','observed.npz','analysis_truth.npz','METRICS.json','MASKS.npz','ALL_PREDICTIONS.npz','train.csv','eval.csv','FDH_crosscheck.rds','R_predictions.f64','R_predictions_shape.json']
    required={n:d['files'][n] for n in names}
    # Hashes are checked when the dependent stages consume the files.
    cfg=json.loads((p/'TASK.json').read_text());assert sha(p/'TASK.json')==required['TASK.json']
    rows.append(dict(task=t['task'],cell=t['cell_id'],rep=t['replication_id'],n=cfg['n'],m=cfg['m'],directory=str(p),eval_sha256=required['eval.csv'],complete_sha256=sha(complete),files=required))
    if i%5000==0:print('indexed',i,flush=True)
save(H/'INPUT_MANIFEST.json',rows)
with (H/'INPUTS.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['task','cell','rep','n','m','directory','eval_sha256']);w.writeheader()
    w.writerows({k:r[k] for k in w.fieldnames} for r in rows)
for folder in ['tools','design','environment']:
    dest=H/'runtime'/folder
    assert not dest.exists(),'Preparation already exists'
    shutil.copytree(P/folder,dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for p in (H/'runtime/tools').iterdir():
    if p.is_file():assert sha(p)==sha(P/'tools'/p.name)
for p in [P/'EXECUTION_LOCK.json',P/'CHECKPOINT_SHA256.json',V/'frontier_full_run_v4_20261012/RESULTS.json',V/'replication_package_v7_20261025/opsearch_replication_v7_20261025.zip']:
    sources[str(p)]=sha(p)
for p in (V/'submission_package_v6_20261025/upload').iterdir():sources[str(p)]=sha(p)
save(H/'PROTECTED_FILES.json',sources)
result=json.loads((V/'frontier_full_run_v4_20261012/RESULTS.json').read_text());a=Path(result['audit']).parent
bad=[json.loads(x) for x in (a/'CLASSIFIED_UNAVAILABLE.jsonl').read_text().splitlines()]
save(H/'ORIGINAL_UNAVAILABLE.json',bad)
save(H/'PROTOCOL.json',dict(status='PREPARED_NOT_ADOPTED',design='All existing inputs, evaluation points, masks, seeds and other-model predictions preserved',FDH='Apply direct enumeration to all 27000 datasets on exact original R-read inputs; independently verify against archived direct check; recompute FDH-dependent summaries and contrasts',StoNED='Only two original timeouts; same scientific algorithm and per-start CPU/iteration settings; raise outer stage wall limit from 600 to 1800 seconds for both cases; retain all attempts and failures; no selective successful-only retry',SFA='No refitting or change',adoption='Separate supplement/sensitivity results; no overwrite or automatic replacement of manuscript/replication archive',counts={'datasets':len(rows),'FDH_original_exclusions':52,'StoNED_original_timeouts':2},runtime_note='Copied scientific source and local runtime references; original readiness/execution approvals are not copied',seconds=time.monotonic()-start))
print('Prepared',len(rows),'datasets',time.monotonic()-start,flush=True)
