"""Frozen public CNLS with explicit, nonnegative, profiled QLE correction."""
from context import *
from stable_math import te
from scipy.special import log_ndtr
from scipy.optimize import minimize_scalar,linprog
from pystoned import CNLS,StoNED
from pyomo.environ import SolverFactory,Constraint,value
from unittest.mock import patch
import scipy.optimize as scipy_opt
import highspy,warnings,time
P=read(ROOT/'design/STOCHASTIC_POLICY.json')['stoned']

from qle_profile_v4 import qle_profile

def cnls(x,y,out,tol):
    exe=Path(RUNTIME['ipopt']);attempts=[];valid=[]
    for idx,mult in enumerate(P['CNLS_starts']):
        t=time.perf_counter();model=CNLS.CNLS(100*y,x,cet='mult',fun='prod',rts='vrs');pm=model.__model__;a=max(2.,float(np.median(100*y))*mult)
        for i in pm.I:
            pm.alpha[i].value=a
            for j in pm.J:pm.beta[i,j].value=0.
            pm.frontier[i].value=a-1.;pm.epsilon[i].value=float(np.log(100*y[i]/a))
        opt=SolverFactory('ipopt',executable=str(exe))
        r=opt.solve(pm,logfile=str(out/f'ipopt_tol{tol}_start{idx}.log'),options={'tol':tol,'max_iter':P['CNLS_max_iterations'],'max_cpu_time':P['CNLS_max_cpu_seconds_per_start'],'print_level':3})
        violation=0.
        for c in pm.component_data_objects(Constraint,active=True):
            b=value(c.body)
            if c.lower is not None:violation=max(violation,value(c.lower)-b)
            if c.upper is not None:violation=max(violation,b-value(c.upper))
        model.problem_status=r;model.optimization_status=1;res=np.asarray(model.get_residual());obj=float(value(pm.objective));floor=float(min(model.get_frontier()));term=str(r.solver.termination_condition)
        ok=term=='optimal' and violation<=P['CNLS_constraint_tolerance'] and floor>1.01
        attempts.append(dict(tolerance=tol,start=idx,termination=term,objective=obj,max_constraint_violation=violation,scaled_frontier_min=floor,accepted=ok,residual_rms=np.sqrt(np.mean(res**2)),seconds=time.perf_counter()-t))
        np.savez(out/f'cnls_tol{tol}_start{idx}.npz',residual=res,frontier=model.get_frontier()/100,alpha=model.get_alpha()/100,beta=model.get_beta()/100)
        if ok:valid.append((obj,idx,model))
    write(out/f'CNLS_tol{tol}.json',attempts)
    if not valid:return None,attempts
    return min(valid,key=lambda x:(x[0],x[1]))[2],attempts

from envelope_adapter import predict_envelope


def fit(x,y,xe,out):
    started=time.perf_counter();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    model,attempts=cnls(x,y,out,P['CNLS_tolerance'])
    if model is None:
        meta=dict(status='UNQUALIFIED_CNLS',qualified=False,attempts=attempts);write(out/'STATUS.json',meta);return meta,{}
    res=np.asarray(model.get_residual());base_rms=np.sqrt(np.mean(res**2));threshold=np.sqrt(np.finfo(float).eps)*max(1.,float(np.std(np.log(y))));degenerate=False;unresolved=False
    if base_rms<=threshold:
        tight,tight_attempts=cnls(x,y,out,P['CNLS_tight_tolerance']);attempts+=tight_attempts
        if tight is not None:
            model=tight;res=np.asarray(model.get_residual());rms=np.sqrt(np.mean(res**2));degenerate=bool(rms<=threshold and rms<=.1*base_rms);unresolved=not degenerate
        else:unresolved=True
    # Preserve an unmodified-package decomposition for the selected CNLS fit.
    rd=StoNED.StoNED(model);records=[];original=scipy_opt.minimize
    def traced(*args,**kwargs):
        r=original(*args,**kwargs);records.append(dict(success=bool(r.success),status=int(r.status),message=str(r.message),fun=float(r.fun),x=np.asarray(r.x)));return r
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        try:
            with patch.object(scipy_opt,'minimize',traced):mu=rd.get_unconditional_expected_inefficiency('QLE');native=np.asarray(rd.get_technical_inefficiency('QLE'))
            raw=dict(sigma_u=rd.sigma_u,sigma_v=rd.sigma_v,mu=mu,optimizer_records=records)
        except (ValueError,FloatingPointError,ZeroDivisionError) as exc:raw=dict(error=repr(exc),optimizer_records=records);native=np.full(len(y),np.nan)
        raw['warnings']=[str(w.message) for w in ws]
    write(out/'PUBLIC_QLE_RAW.json',raw)
    g=np.asarray(model.get_frontier())/100.;ge,env=predict_envelope(x,g,xe,out)
    if degenerate:
        q=dict(rho=None,sigma_u=0.,sigma_v=0.,mu=0.,residual_rms=np.sqrt(np.mean(res**2)));bc=jl=np.ones(len(y));status='DEGENERATE_LIMIT';profile_ok=True;diff=0.
    else:
        q,grid=qle_profile(res,P['QLE_profile_grid']);check,checkgrid=qle_profile(res,P['QLE_check_grid']);diff=abs(q['objective']-check['objective'])/len(y)
        np.savez(out/'QLE_PROFILE.npz',**grid);np.savez(out/'QLE_PROFILE_CHECK.npz',**checkgrid);write(out/'QLE_CHECK.json',check)
        profile_ok=diff<=P['QLE_profile_agreement_per_observation'] and q['selected_success'] and check['selected_success'];su,sv=q['sigma_u'],q['sigma_v'];eps=res-q['mu']
        if su==0:bc=jl=np.ones(len(y));status='QUALIFIED_SIGMA_U_ZERO'
        elif sv==0:bc=jl=np.exp(eps);status='QUALIFIED_SIGMA_V_ZERO'
        else:ss=su*sv/np.hypot(su,sv);a=-eps*su/(sv*np.hypot(su,sv));bc,jl=te(a,ss);status='QUALIFIED_INTERIOR'
    scores_ok=bool(np.all(np.isfinite(np.r_[bc,jl])) and min(np.r_[bc,jl])>=0 and max(np.r_[bc,jl])<=1+1e-8)
    ok=profile_ok and env['qualified'] and scores_ok and not unresolved
    arrays=dict(frontier_train=g*np.exp(q['mu']),frontier_eval=ge*np.exp(q['mu']),te_BC=bc,te_JLMS=jl,cnls_mean_train=g,cnls_mean_eval=ge,cnls_residual=res,public_JLMS=native,alpha=model.get_alpha()/100,beta=model.get_beta()/100)
    meta=dict(status=status if ok else 'UNQUALIFIED_REVIEW',qualified=bool(ok),regular_mle=False,stochastic_decomposition_identified=not degenerate,score_convention='TE=1 unresolved-scale limit, separate category' if degenerate else 'stable conditional HN expectation',decomposition=q,
        raw_decomposition=raw,attempts=attempts,base_residual_rms=base_rms,residual_threshold=threshold,residual_scale_unresolved=unresolved,profile_objective_discrepancy_per_obs=diff,local_CNLS_only=True,envelope=env,scores_valid=scores_ok,fit_seconds=time.perf_counter()-started,adapter='pyStoNED0.7.5_log_ratio_profile_stable_scores_certified_envelope_v4')
    write(out/'STATUS.json',meta);np.savez(out/'PREDICTIONS.npz',**arrays);return meta,arrays
