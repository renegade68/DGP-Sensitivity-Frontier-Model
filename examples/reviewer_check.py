"""Bounded reviewer check; requires the separate reviewer fixture packet.

Run without -O. No production results are overwritten. The original R baseline
fits are not repeated: their archived parameter vectors are starting candidates.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import types

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def require(condition, message):
    if not condition: raise ValueError(message)

def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'tools'/(name+'.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def main():
    start = time.perf_counter()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fixtures', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    require(__debug__, 'Run without -O: preserved scientific modules use assertions.')
    fixtures, out = args.fixtures.resolve(), args.output.resolve()
    require(not out.exists(), 'Output must be a new directory.')
    require(not out.is_relative_to(ROOT) and not out.is_relative_to(fixtures),
            'Output must be outside the source checkout and fixture directory.')
    protocol = json.loads((ROOT/'examples/PROTOCOL.json').read_text())
    manifest = json.loads((fixtures/'MANIFEST.json').read_text())
    require(protocol == manifest['protocol'], 'Protocol differs from the fixture packet.')
    for name,digest in manifest['files'].items():
        require(sha(fixtures/name)==digest, 'Fixture hash mismatch: '+name)
    for name,digest in manifest['source_hashes'].items():
        require(sha(ROOT/name)==digest, 'Scientific source hash mismatch: '+name)
    import numpy as np
    import scipy
    import clarabel
    # Inject only the original IO/array helpers. Do not execute the original
    # machine-specific context initialization or rewrite any scientific module.
    context = types.ModuleType('context')
    context.__dict__.update(ROOT=ROOT,np=np,Path=Path,json=json,hashlib=hashlib,sys=sys)
    tree = ast.parse((ROOT/'tools/context.py').read_text())
    names = {'read','sha','clean','write','arrays_hash'}
    helpers = ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
    exec(compile(helpers,'original-context-helpers','exec'),context.__dict__)
    context.__all__ = ['ROOT','np','Path','json','hashlib','sys']+sorted(names)
    sys.modules['context'] = context
    generator = load('generator'); load('stable_math')
    adapter = load('sfa_adapter'); checks = load('independent_checks'); metrics = load('metrics')
    tree = ast.parse((ROOT/'repairs/module_repair_v1/fdh_recompute.py').read_text())
    ns = {'np':np}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='direct_reference'],type_ignores=[]),'original-fdh-reference','exec'),ns)
    cells = {c['cell_id']:c for c in context.read(ROOT/'design/EFFECTIVE_DGP.json')['cells']}
    out.mkdir(parents=True)
    rows=[]; generation=[]
    for task in protocol['cases']:
        z=dict(np.load(fixtures/(task+'.npz'),allow_pickle=False))
        cellid,rep=task.rsplit('_r',1)
        data=generator.generate(cells[cellid],int(rep))
        exact={k:bool(np.array_equal(v,z['generated__'+k])) for k,v in data.items()}
        generation.append({'task':task,'exact_arrays':exact,'pass':all(exact.values())})
        # Estimators receive observed x, y and x_eval only; latent truth is used
        # below for post-estimation losses, never candidate selection or fitting.
        x,y,xe=data['x'],data['y'],data['x_eval']; n=len(y)
        for name in protocol['models']:
            tick=time.perf_counter(); row={'task':task,'model':name}
            try:
                require(all(exact.values()),'Generated arrays differ from archived inputs.')
                if name=='FDH':
                    rx,ry=z['r_x'],z['r_y']; q=np.vstack([rx,z['r_x_eval']])
                    eligible=np.all(rx[None,:,:]<=q[:,None,:],axis=2)
                    pred=np.where(eligible,ry[None,:],-np.inf).max(axis=1)
                    pred[~np.isfinite(pred)]=np.nan
                    canonical=np.nan_to_num(pred,nan=0).astype('<f8').tobytes()
                    digest=hashlib.sha256(canonical).hexdigest()
                    row.update(direct_sha256=digest,original_hash_equal=digest==manifest['cases'][task]['fdh_direct_sha256'],
                               sorted_reference_equal=bool(np.array_equal(pred,ns['direct_reference'](rx,ry,q),equal_nan=True)))
                    row['pass']=row['original_hash_equal'] and row['sorted_reference_equal']
                    score=y/pred[:n]
                    np.savez_compressed(out/(task+'_FDH.npz'),frontier=pred,te_native=score)
                else:
                    _,form,dist=name.split('_')
                    meta,a=adapter.fit(x,y,xe,form,dist,out/task/name,z[name+'__baseline'])
                    independent=checks.score_check(meta,a,np.log(y),name,adapter.design(x,form))
                    qualified=bool(meta['qualified'] and independent['qualified'])
                    ref=manifest['cases'][task]['models'][name]
                    frontier_error=max(float(np.max(np.abs(a[k]-z[name+'__'+k])/np.maximum(np.abs(z[name+'__'+k]),np.finfo(float).tiny))) for k in ('frontier_train','frontier_eval'))
                    score_error=max(float(np.max(np.abs(a[k]-z[name+'__'+k]))) for k in ('te_BC','te_JLMS'))
                    ll_error=abs(meta['loglik']-ref['loglik'])/n
                    row.update(status=meta['status'],reference_status=ref['status'],qualified=qualified,
                               reference_qualified=ref['qualified'],independent_score_check=independent,
                               loglik_per_observation_abs=ll_error,frontier_max_relative=frontier_error,
                               native_score_max_abs=score_error)
                    tol=protocol['tolerances']
                    row['pass']=bool(qualified and ref['qualified'] and meta['status']==ref['status'] and
                                     ll_error<=tol['loglik_per_observation_abs'] and frontier_error<=tol['frontier_max_relative'] and score_error<=tol['native_score_max_abs'])
                    pred=np.r_[a['frontier_train'],a['frontier_eval']];score=a['te_BC']
                mask=z['fdh_mask']
                truth=np.r_[data['true_frontier'],data['true_frontier_eval']]
                row['frontier_loss_FDH_box']=metrics.loss_values(pred[mask],truth[mask])
                row['native_TE_loss']=metrics.loss_values(score,data['true_te'],rank=True)
            except Exception as exc:
                row.update(pass_=False,error=str(exc),traceback=traceback.format_exc())
                row['pass']=False
            row['wall_seconds']=time.perf_counter()-tick
            rows.append(row)
            context.write(out/'PROGRESS.json',rows)
            print(task,name,'PASS' if row['pass'] else 'REVIEW',f"{row['wall_seconds']:.3f}s",flush=True)
    preserved=all(sha(ROOT/name)==digest for name,digest in manifest['source_hashes'].items())
    fixtures_preserved=all(sha(fixtures/name)==digest for name,digest in manifest['files'].items())
    passed=all(r['pass'] for r in rows) and all(g['pass'] for g in generation) and preserved and fixtures_preserved
    summary={'status':'PASS' if passed else 'REVIEW_REQUIRED','scope':protocol['scope'],
             'wall_seconds':time.perf_counter()-start,'datasets':len(generation),'checks':len(rows),
             'fresh_SFA_optimizations':sum(r['model']!='FDH' for r in rows),'FDH_direct_evaluations':sum(r['model']=='FDH' for r in rows),
             'passed_checks':sum(r['pass'] for r in rows),'generation':generation,'results':rows,
             'scientific_sources_preserved':preserved,'fixtures_preserved':fixtures_preserved,
             'protocol_sha256':sha(ROOT/'examples/PROTOCOL.json'),'fixture_manifest_sha256':sha(fixtures/'MANIFEST.json'),
             'environment':{'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),
                            'numpy':np.__version__,'scipy':scipy.__version__,'clarabel':clarabel.__version__,'numerical_threads':1},
             'limits':['Setup time excluded. Hardware and numerical libraries affect time and floating-point results.',
                       'Archived R baseline starts reused; no fresh R baseline fitting.',
                       'Does not establish a global optimum or validate all methods or Monte Carlo rankings.',
                       'No cases substituted and no tolerances relaxed after observing outcomes.']}
    context.write(out/'SUMMARY.json',summary)
    print(json.dumps({k:summary[k] for k in ('status','wall_seconds','checks','passed_checks')},indent=2))
    return 0 if passed else 1

if __name__=='__main__':
    raise SystemExit(main())
