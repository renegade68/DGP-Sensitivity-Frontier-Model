"""Rehash all consumed production inputs after the isolated repair calculations."""
from pathlib import Path
import hashlib,json,time
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=json.loads((H/'INPUT_MANIFEST.json').read_text());start=time.monotonic();count=0
for i,r in enumerate(rows,1):
    p=Path(r['directory']);assert sha(p/'COMPLETE.json')==r['complete_sha256'],r['task'];count+=1
    for name,value in r['files'].items():assert sha(p/name)==value,(r['task'],name);count+=1
    if i%5000==0:print('PRESERVED',i,flush=True)
protected=json.loads((H/'PROTECTED_FILES.json').read_text())
for name,value in protected.items():assert sha(Path(name))==value,name
for r in json.loads((H/'ORIGINAL_UNAVAILABLE.json').read_text()):
    if r['model']!='StoNED_QLE':continue
    parent=Path(next(x['directory'] for x in rows if x['task']==r['task']))
    d=json.loads((parent/'COMPLETE.json').read_text())
    for name,value in d['files'].items():assert sha(parent/name)==value,(r['task'],name)
    assert json.loads((parent/'models/StoNED_QLE/STATUS.json').read_text())['status']=='TIMEOUT'
runtime=json.loads((H/'reports/RUNTIME_VERIFICATION.json').read_text());assert runtime['status']=='PASS'
result=dict(status='PASS',production_datasets=len(rows),consumed_input_and_completion_hash_checks=count,protected_release_files=len(protected),original_StoNED_timeout_records_retained=True,recorded_runtime_files_verified=runtime['files'],seconds=time.monotonic()-start,scope='All consumed inputs and COMPLETE records, full file manifests for two original StoNED attempts, and protected manuscript/archive releases; not a rehash of every unrelated production file')
(H/'reports/PRESERVATION_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
