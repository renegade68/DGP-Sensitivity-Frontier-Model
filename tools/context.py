"""Portable execution context; external paths are isolated in LOCAL.json."""
import os,sys,json,hashlib,datetime,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REV=ROOT.parent
sys.dont_write_bytecode=True
RUNTIME=json.loads((ROOT/'environment/LOCAL.json').read_text())
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='1'
for d in ('tmp','cache'):(ROOT/d).mkdir(exist_ok=True)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'),MPLCONFIGDIR=str(ROOT/'cache'),R_ENVIRON_USER='/dev/null',R_PROFILE_USER='/dev/null',FRONTIER_R_LIBS=os.pathsep.join(RUNTIME['R_libraries']))
sys.path.insert(0,RUNTIME['python_overlay'])
if RUNTIME.get('python_base'):sys.path.insert(1,RUNTIME['python_base'])
import numpy as np
MODELS=['EAT','CEAT','SVF_PAPER','FDH','DEA_VRS','SFA_CD_HN','SFA_TL_HN','SFA_CD_EXP','SFA_TL_EXP','StoNED_QLE']
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(np.integer,np.bool_)):return x.item()
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def object_hash(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def arrays_hash(a):
    h=hashlib.sha256()
    for k,v in sorted(a.items()):
        v=np.ascontiguousarray(v);h.update(k.encode());h.update(str((v.dtype.str,v.shape)).encode());h.update(v.tobytes())
    return h.hexdigest()
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def sources():
    files=[p for d in ('tools','design','vendor') for p in (ROOT/d).rglob('*') if p.is_file() and p.suffix not in ('.pyc',)]
    files += [ROOT/'environment/LOCAL.json',ROOT/'environment/INSTALLED_FILES_SHA256.json']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}
