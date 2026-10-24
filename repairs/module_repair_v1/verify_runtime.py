"""Verify the frozen runtime references without importing estimator modules."""
from pathlib import Path
import hashlib,json,sys,time
H=Path(__file__).resolve().parent;start=time.monotonic()
records=json.loads((H/'runtime/environment/INSTALLED_FILES_SHA256.json').read_text())
bad=[]
for name,value in records.items():
    p=Path(name)
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=value:bad.append(name)
same_python=sys.version==json.loads((H/'runtime/environment/PARENT_LOCK.json').read_text())['python']
result=dict(status='PASS' if not bad and same_python else 'FAIL',files=len(records),mismatches=bad,python_version_match=same_python,seconds=time.monotonic()-start)
(H/'reports/RUNTIME_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2));assert result['status']=='PASS'
