"""One-worker monitored runner. Production requires the final readiness lock."""
from context import *
from generator import save_task
from schema import classify_failure
import argparse,fcntl,subprocess,signal,time,re,shutil,traceback

def verify_complete(task,source):
    d=read(task/'COMPLETE.json');assert d['source_sha256']==source,'Source mismatch'
    for k,h in d['files'].items():assert sha(task/k)==h,('Corrupt completed file',str(task),k)
    return d

def environment_check():
    assert sys.version==read(ROOT/'environment/PARENT_LOCK.json')['python'],'Python interpreter version changed'
    manifest=read(ROOT/'environment/INSTALLED_FILES_SHA256.json')
    for name,h in manifest.items():assert sha(Path(name))==h,('Installed library changed',name)
    return dict(files=len(manifest),manifest_sha256=sha(ROOT/'environment/INSTALLED_FILES_SHA256.json'))

def production_preflight(run,source,tasks,limit):
    gate=read(ROOT/'validation/READINESS.json');assert gate['status']=='READY_FOR_PRODUCTION' and gate['source_sha256']==source
    lock=read(ROOT/'EXECUTION_LOCK.json');assert lock['source_sha256']==source
    if limit is None or limit>117:
        first=read(run/'FIRST_BATCH_AUDIT.json');assert first['status']=='PASS' and first['source_sha256']==source,'First 117-task audit required'
        assert first['tasks']==117 and first['model_attempts']==1170 and len(first['completed_manifest_sha256'])==117,'Incomplete first-batch audit'
        for key,h in first['completed_manifest_sha256'].items():assert sha(run/key)==h,'First-batch evidence changed'
    budget=read(ROOT/'validation/RESOURCE_BUDGET.json');cells={c['cell_id']:c for c in read(ROOT/'design/EFFECTIVE_DGP.json')['cells']};remaining=0.
    for t in tasks:
        if list((run/t['task']).glob('attempt_*/COMPLETE.json')):continue
        c=cells[t['cell_id']];g=f"n{c['n']}_m{c['m']}_eval{10000 if c['input']['latent_rho'] else 5000}"
        remaining+=1.25*budget['profiles'][g]['max_bytes']/1024**3
    assert shutil.disk_usage(ROOT).free/1024**3>=remaining+100,'Insufficient remaining working allocation plus OS reserve'
    return remaining

def system_state(pgid=None):
    out=subprocess.check_output(['/usr/sbin/sysctl','-n','kern.memorystatus_vm_pressure_level','vm.swapusage'],text=True).splitlines()
    pressure=int(out[0]);swap=float(re.search(r'used = ([0-9.]+)M',out[1]).group(1));rss=0
    if pgid is not None:
        data=subprocess.check_output(['/bin/ps','-axo','pgid=,rss='],text=True)
        rss=sum(int(parts[1])*1024 for line in data.splitlines() if len(parts:=line.split())==2 and int(parts[0])==pgid)
    return dict(memory_pressure=pressure,swap_MiB=swap,group_RSS_bytes=rss,free_GiB=shutil.disk_usage(ROOT).free/1024**3)

def stop_group(p):
    if p.poll() is not None:return
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()

def invoke(cmd,task,stage,timeout,policy):
    start=time.monotonic();initial=system_state();reason=None;peak=0
    assert initial['memory_pressure']==1,'Non-normal memory pressure before stage'
    if initial['free_GiB']<policy['minimum_free_GiB']:raise RuntimeError('Disk reserve reached')
    with (task/f'{stage}.log').open('w') as f,(task/f'{stage}_samples.jsonl').open('w') as samples:
        p=subprocess.Popen(['/usr/bin/time','-l','/usr/bin/nice','-n','10']+cmd,cwd=task,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        write(task/'ACTIVE_PROCESS.json',dict(pid=p.pid,stage=stage,started_utc=now(),command=cmd))
        try:
            while p.poll() is None:
                state=system_state(p.pid);peak=max(peak,state['group_RSS_bytes']);elapsed=time.monotonic()-start
                samples.write(json.dumps(dict(elapsed_seconds=elapsed,**state))+'\n');samples.flush()
                if elapsed>timeout:reason='TIMEOUT'
                elif state['group_RSS_bytes']>policy['maximum_owned_group_RSS_GiB']*1024**3:reason='RESOURCE_RSS'
                elif state['memory_pressure']!=1:reason='RESOURCE_PRESSURE'
                elif state['swap_MiB']-initial['swap_MiB']>policy['maximum_swap_growth_MiB']:reason='RESOURCE_SWAP'
                elif state['free_GiB']<policy['minimum_free_GiB']:reason='RESOURCE_DISK'
                if reason:stop_group(p);break
                try:p.wait(timeout=.5)
                except subprocess.TimeoutExpired:pass
        except BaseException:
            stop_group(p);write(task/f'{stage}_INTERRUPTED.json',dict(stage=stage,pid=p.pid,utc=now(),partial_preserved=True));raise
    match=re.findall(r'^\s*(\d+)\s+maximum resident set size\s*$',(task/f'{stage}.log').read_text(),re.MULTILINE)
    os_peak=int(match[-1]) if match else None
    rec=dict(stage=stage,seconds=time.monotonic()-start,peak_group_RSS_bytes=max(peak,os_peak or 0),OS_peak_process_RSS_bytes=os_peak,exit_code=p.returncode,reason=reason,initial=initial,finished_utc=now())
    write(task/f'{stage}_resources.json',rec)
    if reason and reason!='TIMEOUT':raise RuntimeError(reason)
    if not reason and p.returncode!=0:raise RuntimeError(f'Unexpected {stage} exit {p.returncode}; inspect preserved log')
    return rec

def run_task(task,policy):
    exe=sys.executable;worker=str(ROOT/'tools/worker.py');R=RUNTIME['Rscript'];root=str(ROOT);td=str(task)
    specs=[('domain',[exe,'-B',worker,'domain',td],[]),('R_models',[R,'--vanilla',str(ROOT/'tools/r_models.R'),td,root],['FDH','EAT','CEAT']),
        ('dea',[exe,'-B',worker,'dea',td],['DEA_VRS']),('ceat',[exe,'-B',worker,'ceat',td],['CEAT']),
        ('svf',[exe,'-B',worker,'svf',td],['SVF_PAPER']),('R_baseline',[R,'--vanilla',str(ROOT/'tools/r_baseline.R'),td,root],[])]
    specs += [(m,[exe,'-B',worker,m,td],[m]) for m in MODELS if m.startswith('SFA_')]
    specs += [('R_check',[R,'--vanilla',str(ROOT/'tools/r_check.R'),td,root],[]),('StoNED_QLE',[exe,'-B',worker,'StoNED_QLE',td],['StoNED_QLE']),('normalize',[exe,'-B',worker,'normalize',td],[]),('audit',[exe,'-B',str(ROOT/'tools/metrics.py'),td],[])]
    z=np.load(task/'observed.npz');n,ne=len(z['y']),len(z['x_eval']);resources=[]
    for stage,cmd,models in specs:
        rec=invoke(cmd,task,stage,policy['stage_timeout_seconds'][stage],policy);resources.append(rec)
        if rec['reason']=='TIMEOUT':
            if not models:raise RuntimeError(stage+' timeout requires engineering review')
            completed_R=read(task/'r_status.json') if (task/'r_status.json').exists() else {}
            for m in models:
                if stage=='R_models' and m in completed_R:continue
                classify_failure(task,m,'TIMEOUT',n,ne,rec)
                if stage=='R_models':completed_R[m]=dict(status='TIMEOUT',strict_qualified=False)
            if stage=='R_models':
                write(task/'r_status.json',completed_R)
                if not (task/'R_predictions.f64').exists():
                    np.arange(n+ne,dtype='<f8').tofile(task/'R_predictions.f64');write(task/'R_predictions_shape.json',dict(rows=n+ne,columns=['row']))
        write(task/'RESOURCES.json',dict(stages=resources,total_stage_seconds=sum(r['seconds'] for r in resources),peak_group_RSS_bytes=max(r['peak_group_RSS_bytes'] for r in resources)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True);ap.add_argument('--mode',choices=['smoke','production','example'],default='smoke');ap.add_argument('--limit',type=int);args=ap.parse_args()
    assert re.fullmatch('[a-zA-Z0-9_]+',args.run_id)
    run=ROOT/'runs'/args.run_id;run.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'evidence/runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=environment_check();source=sources();policy=read(ROOT/'design/OPERATIONS.json')
    tasks=read(ROOT/'design'/('PRODUCTION_TASKS.json' if args.mode=='production' else 'SMOKE_TASKS.json'))
    if args.mode=='example':tasks=[t for t in tasks if t['task']=='U1_D04_r0001']
    if args.mode=='production':
        production_preflight(run,source,tasks,args.limit)
    if (run/'RUN_LOCK.json').exists():assert read(run/'RUN_LOCK.json')['source_sha256']==source and read(run/'RUN_LOCK.json')['tasks']==tasks
    else:
        write(run/'RUN_LOCK.json',dict(source_sha256=source,tasks=tasks,environment=env,mode=args.mode,created_utc=now()))
        for k in source:
            p=run/'source'/k;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/k,p)
    selected=tasks if args.limit is None else tasks[:args.limit]
    cells={c['cell_id']:c for c in read(ROOT/'design/EFFECTIVE_DGP.json')['cells']};seeds=read(ROOT/'design/RNG.json')['eat_seeds']
    state=dict(status='RUNNING',planned=len(selected),full_scope_tasks=len(tasks),completed=0,mode=args.mode,pid=os.getpid(),started_utc=now())
    def interrupt(signum,frame):raise KeyboardInterrupt('Received signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        for t in selected:
            assert sources()==source,'Source changed during execution'
            folder=run/t['task'];folder.mkdir(exist_ok=True);done=list(folder.glob('attempt_*/COMPLETE.json'))
            assert len(done)<=1,'Duplicate completed attempts'
            if done:verify_complete(done[0].parent,source)
            else:
                task=folder/f'attempt_{len(list(folder.glob("attempt_*")))+1:03d}'
                save_task(task,cells[t['cell_id']],t['replication_id'],seeds[t['task']]);cfg=read(task/'TASK.json');cfg.update(task_id=t['task'],cohort=t['cohort']);write(task/'TASK.json',cfg)
                state.update(current_task=t['task'],current_attempt=task.name);write(run/'STATUS.json',state)
                print('START',t['task'],task.name,flush=True);run_task(task,policy)
                assert sources()==source,'Source changed during task'
                assert read(task/'AUDIT.json')['model_attempts']==10
                write(task/'COMPLETE.json',dict(task_id=t['task'],source_sha256=source,completed_utc=now(),files={str(p.relative_to(task)):sha(p) for p in task.rglob('*') if p.is_file()}))
            state['completed']+=1;state['updated_utc']=now();write(run/'STATUS.json',state);print('COMPLETE',state['completed'],'/',len(selected),flush=True)
        state['status']='COMPUTED_PENDING_AUDIT'
    except BaseException:
        state.update(status='STOPPED_REVIEW',traceback=traceback.format_exc());print(state['traceback'],flush=True)
    state['finished_utc']=now();write(run/'STATUS.json',state)
    if state['status']=='STOPPED_REVIEW':raise SystemExit(1)
if __name__=='__main__':main()
