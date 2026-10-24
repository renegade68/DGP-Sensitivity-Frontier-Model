from context import *
from scipy.stats import spearmanr,kendalltau

def loss_values(pred,true,rank=False):
    assert len(pred)>0 and np.isfinite(pred).all() and np.isfinite(true).all()
    e=pred-true;mse=float(np.mean(e*e));under=float(np.mean(np.maximum(-e,0)));over=float(np.mean(np.maximum(e,0)))
    out=dict(MSE=mse,RMSE=float(np.sqrt(mse)),MAE=float(np.mean(abs(e))),bias=float(np.mean(e)),under_frequency=float(np.mean(e<0)),over_frequency=float(np.mean(e>0)),tie_frequency=float(np.mean(e==0)),under_magnitude=under,over_magnitude=over)
    assert abs(out['MAE']-under-over)<=1e-12*(1+out['MAE'])
    assert abs(out['bias']-over+under)<=1e-12*(1+out['MAE'])
    scalar=sum((float(a)-float(b))**2 for a,b in zip(pred,true))/len(true)
    assert abs(scalar-mse)<=1e-12*(1+abs(mse))
    if rank:
        constant=np.ptp(pred)==0 or np.ptp(true)==0
        out['Spearman']=None if constant else float(spearmanr(pred,true).statistic)
        out['Kendall']=None if constant else float(kendalltau(pred,true).statistic)
    else:out['NMSE']=mse/float(np.mean(true*true))
    return out

def ols_diagnostics(x,y):
    from sfa_adapter import design
    result={};arrays={}
    for form in ('CD','TL'):
        X=design(x,form);ly=np.log(y);b,_,rank,_=np.linalg.lstsq(X,ly,rcond=None);res=ly-np.einsum('ij,j->i',X,b);center=res-res.mean();s2=float(np.mean(center**2));threshold=128*np.finfo(float).eps*max(X.shape)*max(1.,float(np.max(abs(ly))))
        reason='rank_deficient' if rank<X.shape[1] else ('numerically_zero_residual' if np.sqrt(s2)<=threshold else None)
        g1=None if reason else float(np.mean(center**3)/s2**1.5)
        result[form]=dict(rank=int(rank),columns=X.shape[1],residual_sd=np.sqrt(s2),zero_threshold=threshold,skewness=g1,positive_skewness=None if g1 is None else g1>0,unavailable_reason=reason)
        arrays[form+'_residual']=res;arrays[form+'_coefficients']=b
    return result,arrays

def audit(task):
    task=Path(task);z=dict(np.load(task/'analysis_truth.npz'));cfg=read(task/'TASK.json');n=len(z['y']);ne=len(z['x_eval']);x=z['x'];xx=np.vstack([x,z['x_eval']]);ft=np.r_[z['true_frontier'],z['true_frontier_eval']]
    support=np.load(task/'INPUT_SUPPORT.npz');dea=np.r_[np.ones(n,dtype=bool),support['mask']]
    fdh=np.array([np.any(np.all(x<=q,axis=1)) for q in xx]);assert np.all(dea[fdh])
    masks={'training':np.arange(n+ne)<n}
    for v,measure in ((0,'box'),(1,'copula')):
        ev=z['evaluation_measure']==v
        if not ev.any():continue
        for name,mask in [('FDH',fdh[n:]),('DEA',dea[n:]),('full',np.ones(ne,dtype=bool))]:masks[name+'_'+measure]=np.r_[np.zeros(n,dtype=bool),ev&mask]
    np.savez_compressed(task/'MASKS.npz',**masks)
    values={};records=[];status={};preds={};scores={};checks=0
    rc=read(task/'R_INDEPENDENT_CHECK.json') if (task/'R_INDEPENDENT_CHECK.json').exists() else {}
    for name in MODELS:
        folder=task/'models'/name;s=read(folder/'STATUS.json');assert s['schema_version']==1 and s['model']==name and s['attempted']
        if name in rc and not rc[name]['qualified']:
            s.update(status='UNQUALIFIED_INDEPENDENT_R',numerical_qualified=False,strict_qualified=False,native_score_qualified=False);s['independent_R']=rc[name];write(folder/'STATUS.json',s)
        status[name]=s;a=dict(np.load(folder/'PREDICTIONS.npz'));yp=np.r_[a.get('frontier_train',np.full(n,np.nan)),a.get('frontier_eval',np.full(ne,np.nan))];preds[name]=yp;values[name]={}
        for domain,mask in masks.items():
            if domain.startswith('DEA_') and name in ('EAT','FDH'):continue
            if domain.startswith('full_') and name in ('EAT','FDH','DEA_VRS','CEAT','StoNED_QLE'):continue
            reason='unqualified' if not s['numerical_qualified'] else ('empty_domain' if not mask.any() else ('nonfinite_prediction_on_fixed_domain' if not np.isfinite(yp[mask]).all() else None))
            ls=None if reason else loss_values(yp[mask],ft[mask]);checks+=int(ls is not None)
            keys=ls.keys() if ls else ['MSE','RMSE','NMSE','MAE','bias','under_frequency','over_frequency','tie_frequency','under_magnitude','over_magnitude']
            for k in keys:values[name]['frontier|'+domain+'|'+k]=None if ls is None else ls[k]
            records.append(dict(model=name,estimand='frontier',domain=domain,available=reason is None,reason=reason,points=int(mask.sum()),mask_sha256=arrays_hash({'mask':mask})))
        positive=bool(np.isfinite(yp[:n]).all() and np.all(yp[:n]>0));raw=z['y']/yp[:n] if positive else np.full(n,np.nan)
        score_views={'common_raw':(raw,s['numerical_qualified'] and positive)}
        for k in ('BC','JLMS'):
            if name.startswith('SFA_') or name=='StoNED_QLE':
                score=a.get('te_'+k,np.full(n,np.nan));score_views['native_'+k]=(score,s['native_score_qualified'])
                score_views['limit_'+k]=(score,s['numerical_qualified'] and s['degenerate_limit'])
        for view,(score,ok) in score_views.items():
            reason='unqualified_or_unavailable_view' if not ok else ('nonfinite_score' if not np.isfinite(score).all() else None)
            ls=None if reason else loss_values(score,z['true_te'],rank=True);checks+=int(ls is not None)
            keys=ls.keys() if ls else ['MSE','RMSE','MAE','bias','under_frequency','over_frequency','tie_frequency','under_magnitude','over_magnitude','Spearman','Kendall']
            for k in keys:values[name]['TE_'+view+'|training|'+k]=None if ls is None else ls[k]
            records.append(dict(model=name,estimand='TE_'+view,domain='training',available=reason is None,reason=reason,rank_reason='constant_truth_or_score' if ls and ls['Spearman'] is None else None,points=n))
            scores[name+'_'+view]=score
    diag,ols=ols_diagnostics(x,z['y']);np.savez_compressed(task/'OLS_DIAGNOSTICS.npz',**ols)
    agreement=[]
    for i,A in enumerate(MODELS):
        for B in MODELS[i+1:]:
            aa,bb=scores[A+'_common_raw'],scores[B+'_common_raw'];ok=all(status[m]['numerical_qualified'] for m in (A,B)) and np.isfinite(aa).all() and np.isfinite(bb).all()
            agreement.append(dict(A=A,B=B,score_type_A='common_raw',score_type_B='common_raw',available=bool(ok),mean_absolute_difference=float(np.mean(abs(aa-bb))) if ok else None,Spearman=float(spearmanr(aa,bb).statistic) if ok and np.ptp(aa)>0 and np.ptp(bb)>0 else None))
    np.savez_compressed(task/'ALL_PREDICTIONS.npz',**preds,**scores)
    for name in ('DEA_VRS',):
        p=task/f'{name}.npz'
        if p.exists():assert np.array_equal(np.load(p)['support_mask'],support['mask'])
    parents={k:sha(task/k) for k in ['TASK.json','analysis_truth.npz','ALL_PREDICTIONS.npz','MASKS.npz','OLS_DIAGNOSTICS.npz']}
    write(task/'METRICS.json',dict(task_id=cfg['task_id'],cell_id=cfg['cell_id'],replication_id=cfg['replication_id'],cohort=cfg['cohort'],values=values,availability=records,agreement=agreement,OLS=diag,parents=parents))
    write(task/'AUDIT.json',dict(status='COMPLETE_ATTEMPTS_CLASSIFIED',model_attempts=len(status),model_status=status,scalar_identity_checks=checks,all_numerically_qualified=all(s['numerical_qualified'] for s in status.values()),degenerate_limits=sum(s['degenerate_limit'] for s in status.values()),support_counts={k:int(v.sum()) for k,v in masks.items()},metrics_sha256=sha(task/'METRICS.json')))
if __name__=='__main__':audit(sys.argv[1])
