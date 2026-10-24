"""Audit the first 117 scientific tasks before allowing the next dispatch."""
from context import *
from runner import verify_complete

def audit(run):
    run=Path(run);lock=read(ROOT/'EXECUTION_LOCK.json');sc=sources();assert lock['source_sha256']==sc
    records=[];manifests={};failed=[]
    for name in lock['first_batch_task_ids']:
        done=list((run/name).glob('attempt_*/COMPLETE.json'));assert len(done)==1,('Missing or duplicate completion',name)
        task=done[0].parent;verify_complete(task,sc);a=read(task/'AUDIT.json');cfg=read(task/'TASK.json');assert cfg['cohort']=='production' and cfg['replication_id']==1 and a['model_attempts']==10
        assert a['metrics_sha256']==sha(task/'METRICS.json')
        # Qualified estimates are not required for all random samples, but every
        # attempted procedure and every unavailable outcome must be classified.
        for m,s in a['model_status'].items():
            assert s['attempted'] and s['status'] not in ('UNCLASSIFIED','UNQUALIFIED_REVIEW'),(name,m,'unresolved classification')
            if not s['numerical_qualified']:failed.append(dict(task=name,model=m,status=s['status']))
        res=read(task/'RESOURCES.json');assert all(r['reason'] in (None,'TIMEOUT') for r in res['stages'])
        records.append(dict(task=name,numerical_qualified=sum(s['numerical_qualified'] for s in a['model_status'].values()),seconds=res['total_stage_seconds'],bytes=sum(p.stat().st_size for p in task.rglob('*') if p.is_file())))
        manifests[str(done[0].relative_to(run))]=sha(done[0])
    assert len(records)==117
    write(run/'FIRST_BATCH_AUDIT.json',dict(status='PASS',tasks=117,model_attempts=1170,source_sha256=sc,completed_manifest_sha256=manifests,classified_unavailable=failed,records=records,performance_rankings_inspected=False,automatic_promotion=False,utc=now()))
    print('First batch PASS; full continuation remains an explicit command.')
if __name__=='__main__':audit(sys.argv[1])
