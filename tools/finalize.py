"""Close exact-domain correction gates without dispatching models."""
from context import *
import shutil

def main():
    sc=sources()
    gates=['DEA_DOMAIN_FIXTURES.json','DEA_DOMAIN_REVALIDATION.json','FIXTURES.json','RECOVERY.json','INTEGRATION_SMOKE.json','PORTABLE.json','PRODUCTION_RAW_REPLAY.json']
    for name in gates:
        r=read(ROOT/'validation'/name);assert r['status']=='PASS' and r['source_sha256']==sc,name
    saved=read(ROOT/'validation/DEA_DOMAIN_REVALIDATION.json')
    assert saved['production_datasets']==3464 and saved['engineering_datasets']==44 and saved['changed_saved_masks']==0
    raw=read(ROOT/'validation/PRODUCTION_RAW_REPLAY.json')
    assert raw['datasets']==3464 and raw['checks']['attempted']==34640 and raw['checks']['classified_unavailable']==1
    assert raw['checks']['numerical_qualified']==34639
    sm=read(ROOT/'validation/INTEGRATION_SMOKE.json');assert sm['datasets']==15 and sm['all_qualified_or_limit']
    first=read(ROOT/'runs/production001/FIRST_BATCH_AUDIT.json');assert first['status']=='PASS' and first['source_sha256']==sc and not first['classified_unavailable']
    for name,h in first['completed_manifest_sha256'].items():assert sha(ROOT/'runs/production001'/name)==h
    budget=read(ROOT/'validation/RESOURCE_BUDGET.json');assert budget['source_sha256']==sc
    assert shutil.disk_usage(ROOT).free/1024**3>budget['working_allocation_GiB']+100
    tasks=read(ROOT/'design/PRODUCTION_TASKS.json');assert len(tasks)==27000
    lock=dict(status='FROZEN_REVALIDATED_3464_PASSED',source_sha256=sc,scope=dict(conditions=117,datasets=27000,models=10,model_attempts=270000),first_batch_task_ids=[t['task'] for t in tasks[:117]],adopted_production_datasets=3464,pilot_fit_reuse=False,production_auto_launch=False,initial_workers=1,threads=1,working_allocation_GiB=budget['working_allocation_GiB'],OS_reserve_GiB=100,numerical_amendment='Exact rational sign of near-boundary supporting hyperplane; same VRS technology and original primal/dual qualification tolerances',evidence={n:sha(ROOT/'validation'/n) for n in gates},created_utc=now())
    write(ROOT/'EXECUTION_LOCK.json',lock)
    write(ROOT/'validation/READINESS.json',dict(status='READY_FOR_PRODUCTION',source_sha256=sc,execution_lock_sha256=sha(ROOT/'EXECUTION_LOCK.json'),first_batch_passed=True,adopted_production_datasets=3464,full_repetition_run_started=False,utc=now()))
    write(ROOT/'STATUS.json',dict(status='DEA_DOMAIN_REVALIDATION_COMPLETE_READY_TO_RESUME',production_completed_datasets=3464,production_scope_datasets=27000,full_repetition_run_started=False,utc=now()))
    print('V4_ALL_GATES_PASS_READY_TO_RESUME',flush=True)

if __name__=='__main__':main()
