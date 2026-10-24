"""Same StoNED g_min LP, with bounded certified numerical recovery.

No DGP truth, original draw, CNLS, QLE, coefficient or tolerance changes.
Input-support exclusion is certified by a separating affine plane/ray, never
inferred from a NaN or an unknown/infeasible solver status alone.
"""
from context import *
import highspy,time
from scipy.optimize import linprog
TOL=1e-7
RAY_MARGIN=1e-10
LP_TOL=1e-9

def build(x,g,c=None,no_presolve_scaling=False,logfile=None):
    n,m=x.shape;A=np.column_stack([np.ones(n),x]);h=highspy.Highs()
    h.setOptionValue('output_flag',bool(logfile));h.setOptionValue('log_to_console',False);h.setOptionValue('threads',1)
    h.setOptionValue('primal_feasibility_tolerance',LP_TOL);h.setOptionValue('dual_feasibility_tolerance',LP_TOL)
    if logfile:h.setOptionValue('log_file',str(logfile))
    if no_presolve_scaling:
        h.setOptionValue('presolve','off');h.setOptionValue('simplex_scale_strategy',0);h.setOptionValue('solver','simplex')
    h.addVars(m+1,np.r_[-highspy.kHighsInf,np.zeros(m)],np.full(m+1,highspy.kHighsInf))
    h.addRows(n,np.asarray(g),np.full(n,highspy.kHighsInf),n*(m+1),np.arange(0,n*(m+1)+1,m+1,dtype=np.int32),np.tile(np.arange(m+1,dtype=np.int32),n),A.reshape(-1))
    if c is not None:h.changeColsCost(m+1,np.arange(m+1,dtype=np.int32),c)
    return h

def certify_optimal(x,g,q,b,d):
    A=np.column_stack([np.ones(len(x)),x]);c=np.r_[1.,q];b=np.asarray(b);d=np.asarray(d)
    if not np.all(np.isfinite(np.r_[b,d])):return dict(qualified=False,violation=np.inf)
    val=float(np.sum(c*b));slack=np.einsum('ij,j->i',A,b)-g;rd=c-np.einsum('ij,i->j',A,d)
    # Identical acceptance rule to the sealed pilot, with components exposed.
    components=dict(primal_row=max(0.,-float(slack.min())),dual_nonnegative=max(0.,-float(d.min())),slope_nonnegative=max(0.,-float(b[1:].min())),dual_input_slack=max(0.,-float(rd[1:].min())),dual_mass_error=abs(float(rd[0])),relative_gap=abs(float(np.sum(d*g))-val)/(1+abs(val)))
    violation=max(components.values())
    return dict(qualified=violation<=TOL,classification='FINITE',value=val,violation=violation,components=components,coefficients=b,dual=d)

def certify_ray(x,q,ray):
    ray=np.asarray(ray);ray=ray/max(1.,float(np.max(abs(ray))));A=np.column_stack([np.ones(len(x)),x]);c=np.r_[1.,q]
    if not np.all(np.isfinite(ray)):return dict(qualified=False,violation=np.inf)
    original=ray.copy();slack=np.einsum('ij,j->i',A,ray)
    original_violation=max(0.,-float(slack.min()),-float(ray[1:].min()))
    # Construct a strictly separating certificate, not a clipped estimate.
    # Project only ray slopes and lift its free intercept by its feasibility
    # residual plus a rounding guard; then retest its negative query value.
    ray[1:]=np.maximum(0.,ray[1:]);slack=np.einsum('ij,j->i',A,ray)
    guard=64*np.finfo(float).eps*(1+abs(ray[0])+float(np.max(abs(np.einsum('ij,j->i',x,ray[1:])))))
    lift=max(0.,-float(slack.min()))+guard;ray[0]+=lift
    slack=np.einsum('ij,j->i',A,ray);value=float(np.sum(c*ray));violation=max(0.,-float(slack.min()),-float(ray[1:].min()))
    return dict(qualified=original_violation<=TOL and violation<=TOL and value<-RAY_MARGIN,classification='OUTSIDE_INPUT_SUPPORT',value=None,violation=violation,ray=ray,query_ray_value=value,minimum_training_ray_value=float(slack.min()),minimum_ray_slope=float(ray[1:].min()),original_ray=original,original_ray_violation=original_violation,certificate_intercept_lift=lift)

def inspect(h,x,g,q,run_return=None):
    status=h.getModelStatus();r=dict(model_status=str(status),run_return=str(run_return),qualified=False)
    if status==highspy.HighsModelStatus.kOptimal:
        s=h.getSolution();r.update(certify_optimal(x,g,q,s.col_value,s.row_dual))
    elif status==highspy.HighsModelStatus.kUnbounded:
        rs,exists,ray=h.getPrimalRay();r.update(ray_return=str(rs),ray_available=bool(exists))
        if exists:r.update(certify_ray(x,q,ray))
    return r

def dual_or_separator(x,g,q):
    n,m=x.shape
    dual=linprog(-g,A_ub=x.T,b_ub=q,A_eq=np.ones((1,n)),b_eq=[1.],bounds=[(0,None)]*n,method='highs',options={'primal_feasibility_tolerance':LP_TOL,'dual_feasibility_tolerance':LP_TOL})
    if dual.success:
        b=np.r_[-dual.eqlin.marginals[0],-dual.ineqlin.marginals];r=certify_optimal(x,g,q,b,dual.x);r['solver_status']=dual.message;return r
    # Bounded normalization of the recession cone: sum of nonnegative slopes<=1.
    A=np.column_stack([np.ones(n),x]);sep=linprog(np.r_[1.,q],A_ub=np.vstack([-A,np.r_[0.,np.ones(m)]]),b_ub=np.r_[np.zeros(n),1.],bounds=[(None,None)]+[(0,None)]*m,method='highs',options={'primal_feasibility_tolerance':LP_TOL,'dual_feasibility_tolerance':LP_TOL})
    r=certify_ray(x,q,sep.x) if sep.success else dict(qualified=False)
    r.update(dual_solver_status=dual.message,separation_solver_status=sep.message)
    return r

def recover(x,g,q,initial):
    attempts=[dict(stage='persistent_primary',**initial)]
    for flag,name in ((False,'fresh_same_LP'),(True,'fresh_no_presolve_or_scaling')):
        h=build(x,g,np.r_[1.,q],no_presolve_scaling=flag);ret=h.run();r=inspect(h,x,g,q,ret);attempts.append(dict(stage=name,**r))
        if r['qualified']:return r,attempts
    r=dual_or_separator(x,g,q);attempts.append(dict(stage='independent_dual_or_separation_LP',**r));return r,attempts

def predict_envelope(x,g,xe,out):
    started=time.perf_counter();out=Path(out);out.mkdir(parents=True,exist_ok=True);n,m=x.shape;h=build(x,g)
    vals=np.full(len(xe),np.nan);coeff=np.full((len(xe),m+1),np.nan);duals=np.full((len(xe),n),np.nan);rays=np.full((len(xe),m+1),np.nan)
    violations=np.full(len(xe),np.inf);ray_values=np.full(len(xe),np.nan);classes=[];rawstatuses=[];rawreturns=[];fallbacks=[]
    for i,q in enumerate(xe):
        h.changeColsCost(m+1,np.arange(m+1,dtype=np.int32),np.r_[1.,q]);ret=h.run();r=inspect(h,x,g,q,ret);rawstatuses.append(r['model_status']);rawreturns.append(r['run_return'])
        if not r['qualified']:
            r,attempts=recover(x,g,q,r);fallbacks.append(dict(query_index=i,query=q,attempts=attempts))
        if r['qualified']:
            violations[i]=r['violation'];classes.append(r['classification'])
            if r['classification']=='FINITE':vals[i]=r['value'];coeff[i]=r['coefficients'];duals[i]=r['dual']
            else:rays[i]=r['ray'];ray_values[i]=r['query_ray_value']
        else:classes.append('UNRESOLVED')
    np.savez(out/'ENVELOPE_CERTIFICATES.npz',coefficients=coeff,dual=duals,unbounded_ray=rays,violation=violations,prediction=vals,ray_objective=ray_values)
    write(out/'FALLBACKS.json',fallbacks);write(out/'QUERY_STATUS.json',dict(raw_model_status=rawstatuses,raw_run_return=rawreturns,certified_classification=classes))
    meta=dict(queries=len(xe),finite=int(np.isfinite(vals).sum()),outside=classes.count('OUTSIDE_INPUT_SUPPORT'),unresolved=classes.count('UNRESOLVED'),fallback_count=len(fallbacks),maximum_certificate_violation=float(max(violations)),qualified=bool(all(c!='UNRESOLVED' for c in classes)),elapsed_seconds=time.perf_counter()-started,
        adapter='same_LP_certified_fresh_solver_fallback_v1',acceptance_tolerance=TOL,primal_dual_solver_tolerance=LP_TOL,negative_ray_margin=RAY_MARGIN)
    write(out/'ENVELOPE_STATUS.json',meta);return vals,meta
