from context import *
import time,traceback
def svf(z,task):
    pub=module('current_public_svf',ROOT/'tools/public_svf.py');grid=read(ROOT/'design/SVF_GRID.json')
    x,y,xe=z['x'],z['y'],z['x_eval'];ds=grid['d'][str(len(x))];cs=grid['C'];es=grid['epsilon']
    ids=np.array(read(ROOT/'design/RNG.json')['svf_fold_id'][str(len(x))]);shape=(10,10,8,5)
    loss=np.full(shape,np.nan);diag=np.full(shape+(3,),np.nan);codes=np.full(shape,-1,dtype=np.int8)
    rows=[];started=time.perf_counter();final={};success=False
    try:
        with (task/'svf_candidates.jsonl').open('w') as stream:
            for di,d in enumerate(ds):
                for k in range(5):
                    w=pub.PublicSVFWorkspace(x[ids!=k],y[ids!=k],d,paper_intercepts=True)
                    np.savez_compressed(task/f'svf_fold_d{d}_k{k}.npz',train_rows=np.flatnonzero(ids!=k),test_rows=np.flatnonzero(ids==k),ranges=w.ranges)
                    for ci,C in enumerate(cs):
                        for ei,eps in enumerate(es):
                            key=dict(d=d,C=C,epsilon=eps,fold=k)
                            try:
                                b,dd=w.solve(C,eps);yp=w.predict(b,x[ids==k]);value=float(np.mean((yp-y[ids==k])**2))
                                loss[di,ci,ei,k]=value;diag[di,ci,ei,k]=[dd['primal'],dd['dual'],dd['relative_gap']];codes[di,ci,ei,k]=int(dd['solver']=='CLARABEL')
                                stream.write(json.dumps(clean(dict(**key,status='QUALIFIED',CV_loss=value,**dd)),allow_nan=False)+'\n')
                            except Exception:
                                stream.write(json.dumps(dict(**key,status='FAILED',traceback=traceback.format_exc()))+'\n');stream.flush();raise
                    stream.flush()
            assert np.isfinite(loss).all()
            candidates=[(float(loss[a,b,c].mean()),-C,eps,d,a,b,c) for a,d in enumerate(ds) for b,C in enumerate(cs) for c,eps in enumerate(es)]
            _,_,_,_,di,ci,ei=min(candidates)
            cv_seconds=time.perf_counter()-started;final_start=time.perf_counter()
            w=pub.PublicSVFWorkspace(x,y,ds[di],paper_intercepts=True);b,dd=w.solve(cs[ci],es[ei]);fit_seconds=time.perf_counter()-final_start
            pred_start=time.perf_counter();yp=w.predict(b,np.vstack([x,xe]));predict_seconds=time.perf_counter()-pred_start
            final=dict(frontier_train=yp[:len(x)],frontier_eval=yp[len(x):],te=y/yp[:len(x)],coefficients=b,ranges=w.ranges)
            assert yp[:len(x)].min()>0 and min(yp[:len(x)]-y)>=-1e-6
            (task/'svf_final_author_problem.lp').write_text(w.author_lp)
            write(task/'svf_final_parameters.json',dict(C=cs[ci],epsilon=es[ei],d=ds[di],additional_paper_bounds='all feature-block constants >=0',diagnostics=dd))
            status=dict(status='QUALIFIED',strict_qualified=True,package_converged=True,C=cs[ci],epsilon=es[ei],d=ds[di],final_diagnostics=dd,intercepts='nonnegative',CV_and_checkpoint_IO_seconds=cv_seconds,final_fit_seconds=fit_seconds,predict_seconds=predict_seconds)
            success=True
    finally:
        np.savez_compressed(task/'SVF_PAPER.npz',cv_losses=loss,cv_diagnostics=diag,solver_code=codes,fold_id=ids,d_grid=ds,c_grid=cs,epsilon_grid=es,**final)
        if not success:status=dict(status='FAILED_FULL_CV_REQUIRED',strict_qualified=False,package_converged=False,traceback=traceback.format_exc())
        status.update(model='SVF_PAPER',elapsed_seconds=time.perf_counter()-started,planned_candidate_folds=4000,qualified_candidate_folds=int(np.isfinite(loss).sum()))
        write(task/'SVF_PAPER_status.json',status)

def dea(z,task,leaf=False):
    a=module('current_dea',ROOT/'tools/dea_adapter.py')
    # Its legacy module sets caches on import; redirect them before any fitting.
    os.environ.update(MPLCONFIGDIR=str(ROOT/'cache/matplotlib'),XDG_CACHE_HOME=str(ROOT/'cache'))
    if not leaf:
        arrays,meta=a.fit_dea(z['x'],z['y'],z['x_eval'],task/'DEA_certificates')
        np.savez_compressed(task/'DEA_VRS.npz',**arrays)
        write(task/'DEA_VRS_status.json',{**meta,'status':'QUALIFIED','strict_qualified':True,'package_converged':None})
    else:
        p=task/'CEAT_leaves.tsv'
        if not p.exists():raise RuntimeError('No public CEAT leaf matrix')
        shape=read(task/'CEAT_leaves_shape.json');leaves=np.fromfile(task/'CEAT_leaves.f64',dtype='<f8').reshape((shape['rows'],shape['columns']),order='F')
        arrays,meta=a.fit_dea(leaves[:,:-1],leaves[:,-1],np.vstack([z['x'],z['x_eval']]),task/'CEAT_certificates')
        shape=read(task/'R_predictions_shape.json');table=np.fromfile(task/'R_predictions.f64',dtype='<f8').reshape((shape['rows'],len(shape['columns'])),order='F')
        raw=table[:,shape['columns'].index('CEAT_frontier')]
        mask=arrays['support_mask'];ref=arrays['frontier_eval'];err=float(np.max(abs(raw[mask]-ref[mask]))) if mask.any() else 0.
        qualified=bool(np.isfinite(raw[mask]).all() and err<=1e-6)
        np.savez_compressed(task/'CEAT_check.npz',mask=mask,public=raw,reference=ref)
        write(task/'CEAT_status.json',dict(status='QUALIFIED' if qualified else 'UNQUALIFIED_PUBLIC_LP',strict_qualified=qualified,
            max_error=err,certificate=meta,shared_tree_sha256=sha(task/'EAT_shared_tree.rds')))
