from context import *

def save_model(task,name,meta,arrays,n,ne,extra=None):
    folder=Path(task)/'models'/name;folder.mkdir(parents=True,exist_ok=True)
    qualified=bool(meta.get('qualified',meta.get('strict_qualified',False)))
    state=meta.get('status','UNCLASSIFIED');assert state!='UNCLASSIFIED'
    stochastic=name.startswith('SFA_') or name=='StoNED_QLE'
    limit=state=='DEGENERATE_LIMIT'
    identified=bool(meta.get('stochastic_decomposition_identified',not limit)) if stochastic else None
    for key,size in [('frontier_train',n),('frontier_eval',ne),('te_BC',n),('te_JLMS',n)]:
        if key in arrays:assert np.asarray(arrays[key]).shape==(size,),(name,key)
    finite=bool('frontier_train' in arrays and np.isfinite(arrays['frontier_train']).all())
    positive=bool(finite and np.all(arrays['frontier_train']>0))
    status=dict(schema_version=1,model=name,attempted=True,status=state,finite=finite,
        positive_training_frontier=positive,package_converged=meta.get('package_converged'),
        numerical_qualified=qualified,strict_qualified=qualified,
        decomposition_identified=identified,degenerate_limit=limit,
        native_score_qualified=qualified and positive and identified is True,
        raw=meta,**(extra or {}))
    np.savez_compressed(folder/'PREDICTIONS.npz',**arrays)
    write(folder/'STATUS.json',status)
    return status

def classify_failure(task,name,reason,n,ne,details=None):
    return save_model(task,name,dict(status=reason,qualified=False,details=details),{},n,ne)

def r_predictions(task):
    s=read(task/'R_predictions_shape.json');a=np.fromfile(task/'R_predictions.f64',dtype='<f8').reshape((s['rows'],len(s['columns'])),order='F')
    return {k:a[:,i] for i,k in enumerate(s['columns'])}

def normalize_deterministic(task):
    task=Path(task);z=np.load(task/'observed.npz');n=len(z['y']);ne=len(z['x_eval']);rs=read(task/'r_status.json');table=r_predictions(task)
    for name in ('EAT','FDH'):
        
        if (task/'models'/name/'STATUS.json').exists():continue
        meta=rs[name];a={}
        if name+'_frontier' in table:a=dict(frontier_train=table[name+'_frontier'][:n],frontier_eval=table[name+'_frontier'][n:])
        save_model(task,name,meta,a,n,ne)
    for name in ('CEAT','DEA_VRS','SVF_PAPER'):
        target=task/'models'/name/'STATUS.json'
        if target.exists():continue
        meta=read(task/f'{name}_status.json');a={}
        if name=='CEAT':
            if 'CEAT_frontier' in table:a=dict(frontier_train=table['CEAT_frontier'][:n],frontier_eval=table['CEAT_frontier'][n:])
        elif (task/f'{name}.npz').exists():
            raw=np.load(task/f'{name}.npz');a={k:raw[k] for k in ['frontier_train','frontier_eval'] if k in raw}
        save_model(task,name,meta,a,n,ne)
