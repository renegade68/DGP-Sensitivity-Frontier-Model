"""Same-science, extended outer-wall-time retry of the two frozen timeouts."""
from pathlib import Path
import hashlib,json,os,shutil,signal,subprocess,sys,time
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
manifest={r['task']:r for r in json.loads((H/'INPUT_MANIFEST.json').read_text())}
bad=[r for r in json.loads((H/'ORIGINAL_UNAVAILABLE.json').read_text()) if r['model']=='StoNED_QLE']
assert len(bad)==2
records=[]
for b in bad:
    task=b['task'];parent=Path(manifest[task]['directory']);dest=H/'stoned'/task
    assert not dest.exists(),'Never overwrite a retry attempt'
    dest.mkdir()
    for name in ['TASK.json','observed.npz','analysis_truth.npz','MASKS.npz']:
        assert sha(parent/name)==manifest[task]['files'][name]
        shutil.copy2(parent/name,dest/name)
    save(dest/'PARENT.json',dict(directory=str(parent),complete_sha256=manifest[task]['complete_sha256'],policy='Same algorithm, tolerance, starts, solver iteration and CPU limits; outer wall limit 1800 instead of 600 seconds'))
    env=os.environ.copy()
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:env[k]='1'
    start=time.monotonic()
    with (dest/'retry.log').open('w') as log:
        p=subprocess.Popen(['/usr/bin/nice','-n','10',sys.executable,'-B',str(H/'runtime/tools/worker.py'),'StoNED_QLE',str(dest)],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        save(dest/'PROCESS.json',dict(pid=p.pid,started=time.time(),outer_wall_limit_seconds=1800))
        reason=None
        try:
            while p.poll() is None:
                elapsed=time.monotonic()-start
                save(H/'stoned/PROGRESS.json',dict(task=task,seconds=elapsed,pid=p.pid,completed=len(records)))
                if elapsed>1800 or shutil.disk_usage(H).free<100*1024**3:
                    reason='TIMEOUT' if elapsed>1800 else 'RESOURCE_DISK';os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                    break
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:pass
        except BaseException:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10)
            raise
    statusfile=dest/'models/StoNED_QLE/STATUS.json'
    status=json.loads(statusfile.read_text()) if statusfile.exists() else None
    rec=dict(task=task,seconds=time.monotonic()-start,exit_code=p.returncode,stop_reason=reason,status=status['status'] if status else None,numerical_qualified=status['numerical_qualified'] if status else False,files={str(f.relative_to(dest)):sha(f) for f in dest.rglob('*') if f.is_file()})
    save(dest/'RETRY_COMPLETE.json',rec);records.append(rec);print(task,rec['status'],rec['seconds'],flush=True)
save(H/'stoned/SUMMARY.json',dict(status='RETRIES_COMPLETED',cases=records,original_outcomes_preserved=True,outer_wall_seconds=1800,scientific_policy_changed=False))
save(H/'stoned/PROGRESS.json',dict(status='COMPLETE',completed=len(records),total=len(bad)))
