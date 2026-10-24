"""Observed-only, explicitly labeled public-likelihood numerical adapter.

Raw sfaR objects are retained separately. Boundary certificates apply only to
the boundary subproblem. The multi-start search makes no global-MLE claim.
"""
from context import *
from stable_math import evaluate
from scipy.optimize import minimize,linprog
from scipy.optimize._numdiff import approx_derivative
from scipy.special import expit
from scipy import sparse
import clarabel,time
POLICY=read(ROOT/'design/STOCHASTIC_POLICY.json')['sfa']

def design(x,form):
    lx=np.log(x);parts=[np.ones((len(x),1)),lx]
    if form=='TL':parts+=[.5*lx**2]+[(lx[:,i]*lx[:,j])[:,None] for i in range(lx.shape[1]) for j in range(i+1,lx.shape[1])]
    return np.column_stack(parts)
def mv(X,b):return np.einsum('ij,j->i',X,b)
def boundary(X,y,dist):
    n,K=X.shape;Q,R=np.linalg.qr(X,mode='reduced')
    if dist=='HN':
        s=clarabel.DefaultSettings();s.verbose=False;s.max_iter=500;s.tol_gap_abs=s.tol_gap_rel=s.tol_feas=1e-11
        # Equivalent QP in centered, dimensionless residual coordinates. The
        # uncentered objective is dominated by the fitted level when residuals
        # are small, so absolute solver tolerances can distort likelihood ties.
        center=np.einsum('ij,i->j',Q,y);residual=y-mv(Q,center)
        scale=float(np.sqrt(np.mean(residual**2)))
        assert np.isfinite(scale) and scale>0,'Degenerate HN boundary must be handled by the OLS-limit branch'
        response=residual/scale
        sol=clarabel.DefaultSolver(sparse.eye(K,format='csc'),-np.einsum('ij,i->j',Q,response),sparse.csc_matrix(-Q),-response,[clarabel.NonnegativeConeT(n)],s).solve()
        theta=center+scale*np.array(sol.x);dual=scale*np.array(sol.z);status=str(sol.status)
    else:
        sol=linprog(np.sum(Q,axis=0),A_ub=-Q,b_ub=-y,bounds=[(None,None)]*K,method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        if not sol.success:raise RuntimeError('EXP boundary LP: '+sol.message)
        theta=sol.x;dual=-sol.ineqlin.marginals;status=sol.message
    b=np.linalg.solve(R,theta);r=mv(Q,theta)-y
    if dist=='HN':
        su=np.sqrt(np.mean(r*r));ll=n*np.log(2)-n*np.log(su)-.5*n*np.log(2*np.pi)-.5*n;stationarity=np.einsum('ij,i->j',Q,r-dual)
    else:su=np.mean(r);ll=-n*np.log(su)-n;stationarity=np.einsum('ij,i->j',Q,np.ones(n)-dual)
    cert=dict(primal_violation=max(0.,-float(r.min())),dual_violation=max(0.,-float(dual.min())),stationarity=float(np.max(abs(stationarity))),complementarity=float(np.max(abs(r*dual))))
    return dict(kind='SIGMA_V_ZERO',loglik=ll,beta=b,sigma_u=su,sigma_v=0.,boundary_certificate=cert,boundary_solver_status=status,boundary_certified=max(cert.values())<=POLICY['boundary_KKT_tolerance'],dual=dual)

def fit(x,y,xe,form,dist,out,baseline=None):
    started=time.perf_counter();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    X=design(x,form);Xe=design(xe,form);ly=np.log(y);n,K=X.shape
    rank=np.linalg.matrix_rank(X)
    if rank<K:return dict(status='RANK_DEFICIENT',qualified=False,rank=rank,columns=K),{}
    Q,R=np.linalg.qr(X,mode='reduced');Q*=np.sqrt(n)
    gamma=np.einsum('ij,i->j',Q,ly)/n;olsb=np.linalg.solve(R,np.sqrt(n)*gamma);olsr=ly-mv(Q,gamma);sd=np.sqrt(np.mean(olsr**2))
    threshold=128*np.finfo(float).eps*max(n,K)*max(1.,float(np.max(abs(ly))))
    if sd<=threshold:
        arrays=dict(frontier_train=np.exp(mv(X,olsb)),frontier_eval=np.exp(mv(Xe,olsb)),te_BC=np.ones(n),te_JLMS=np.ones(n),coefficients=olsb)
        meta=dict(status='DEGENERATE_LIMIT',qualified=True,regular_mle=False,stochastic_decomposition_identified=False,score_convention='TE=1 deterministic limit; separate category',sigma_u=0.,sigma_v=0.,ols_residual_rms=sd,threshold=threshold,fit_seconds=time.perf_counter()-started)
        np.savez(out/'PREDICTIONS.npz',**arrays);write(out/'STATUS.json',meta);return meta,arrays
    gaussian=dict(kind='SIGMA_U_ZERO',loglik=-.5*n*(np.log(2*np.pi)+1+2*np.log(sd)),beta=olsb,sigma_u=0.,sigma_v=sd,boundary_certified=True)
    pureu=boundary(X,ly,dist);c=1. if dist=='EXP' else np.sqrt(2/np.pi)
    candidates=[gaussian,pureu];profile=[];optimizers=[];logsd=np.log(sd)
    def unpack(z):
        qu=expit(2*z[-1]);qv=expit(-2*z[-1]);s=np.exp(z[-2]);su=s*np.sqrt(qu);sv=s*np.sqrt(qv)
        return su,sv,qu,qv
    def fun(z):
        su,sv,qu,qv=unpack(z);p=np.r_[z[:K],2*np.log(su),2*np.log(sv)]
        nl,g,_=evaluate(p,Q,ly-c*su,dist);gi=g[0]/Q[0,0]
        grad=np.r_[g[:K],2*(g[-2]+g[-1])+gi*c*su,2*qv*g[-2]-2*qu*g[-1]+gi*c*su*qv]/n
        return nl/n,grad
    def candidate(z,label,success,message):
        su,sv,qu,qv=unpack(z);b=np.linalg.solve(R,np.sqrt(n)*z[:K]);b[0]+=c*su;nl,g=fun(z)
        return dict(kind='INTERIOR',label=label,loglik=-nl*n,beta=b,sigma_u=su,sigma_v=sv,canonical=z,optimizer_success=bool(success),optimizer_message=str(message),scaled_gradient_max=np.max(abs(g)),search_bound_hit=bool(abs(z[-1])>24.99 or abs(z[-2]-logsd)>11.99))
    bounds=[(None,None)]*K+[(logsd-12,logsd+12),(-25.,25.)]
    optkw=dict(ftol=1e-14,gtol=1e-9,maxiter=POLICY['max_iterations'],maxls=50)
    for rho in POLICY['rho_grid']:
        eta=np.log(rho/np.sqrt(1-rho*rho))
        def fn(v):
            val,grad=fun(np.r_[v,eta]);return val,grad[:-1]
        sol=minimize(fn,np.r_[gamma,logsd],jac=True,method='L-BFGS-B',bounds=bounds[:-1],options=optkw)
        p=candidate(np.r_[sol.x,eta],f'fixed_rho_{rho}',sol.success,sol.message);profile.append(p);candidates.append(p)
        optimizers.append(dict(stage='fixed_rho',rho=rho,iterations=sol.nit,evaluations=sol.nfev,**p))
    selected_starts={0,len(profile)-1}
    for i in range(1,len(profile)-1):
        if profile[i]['loglik']>=max(profile[i-1]['loglik'],profile[i+1]['loglik']):selected_starts.add(i)
    starts=[(profile[i]['canonical'],profile[i]['label']) for i in sorted(selected_starts)]
    if baseline is not None and np.all(np.isfinite(baseline)):
        su,sv=np.exp(baseline[-2:]/2);b=baseline[:K].copy();b[0]-=c*su;z=np.r_[mv(R,b)/np.sqrt(n),np.log(np.hypot(su,sv)),np.log(su/sv)]
        raw=candidate(z,'fresh_sfaR_baseline',True,'recorded separately, not automatically qualified');candidates.append(raw)
        if logsd-12<z[-2]<logsd+12 and -25<z[-1]<25:starts.append((z,'fresh_sfaR_baseline'))
    for z,label in starts:
        sol=minimize(fun,z,jac=True,method='L-BFGS-B',bounds=bounds,options=optkw)
        p=candidate(sol.x,'joint_from_'+label,sol.success,sol.message);candidates.append(p)
        optimizers.append(dict(stage='joint',iterations=sol.nit,evaluations=sol.nfev,**p))
        # Fixed alternate public optimizer, triggered by the same gradient rule.
        if p['scaled_gradient_max']>1e-5 and not p['search_bound_hit']:
            def guarded(v):
                if abs(v[-1])>25 or abs(v[-2]-logsd)>12:return 1e100,np.zeros_like(v)
                return fun(v)
            alt=minimize(guarded,sol.x,jac=True,method='BFGS',options={'gtol':1e-9,'maxiter':2000})
            ap=candidate(alt.x,'BFGS_from_'+label,alt.success,alt.message);candidates.append(ap);optimizers.append(dict(stage='fixed_BFGS_fallback',iterations=alt.nit,evaluations=alt.nfev,**ap))
    best=max(candidates,key=lambda p:p['loglik']);tie=POLICY['likelihood_tie_per_observation']*n
    eligible=[p for p in candidates[:2] if p['loglik']>=best['loglik']-tie and p['boundary_certified']]
    if eligible:best=max(eligible,key=lambda p:p['loglik'])
    b=np.array(best['beta']);su,sv=best['sigma_u'],best['sigma_v'];lp=mv(X,b);lpe=mv(Xe,b)
    qual=dict(candidate_selection_gap=max(p['loglik'] for p in candidates)-best['loglik'],candidate_search_global_claim=False)
    if best['kind']=='INTERIOR':
        p=np.r_[b,2*np.log(su),2*np.log(sv)];nl,g,(bc,jl)=evaluate(p,X,ly,dist)
        H=approx_derivative(lambda z:fun(z)[1],best['canonical'],method='3-point',rel_step=1e-4);H=(H+H.T)/2;eig=np.linalg.eigvalsh(H)
        qual.update(original_parameter_gradient_per_obs=np.max(abs(g))/n,scaled_gradient=best['scaled_gradient_max'],hessian_min=float(eig.min()),hessian_max_abs=float(np.max(abs(eig))),search_bound_hit=best['search_bound_hit'])
        ok=qual['original_parameter_gradient_per_obs']<=1e-4 and qual['scaled_gradient']<=1e-5 and eig.min()>=-1e-6*(1+np.max(abs(eig))) and not best['search_bound_hit']
    elif best['kind']=='SIGMA_U_ZERO':bc=jl=np.ones(n);ok=True;qual['OLS_normal_equation_max']=float(np.max(abs(np.einsum('ij,i->j',Q,olsr)))/n)
    else:
        bc=jl=np.exp(ly-lp);ok=best['boundary_certified'];qual['boundary_KKT']=best['boundary_certificate']
    arrays=dict(frontier_train=np.exp(lp),frontier_eval=np.exp(lpe),te_BC=bc,te_JLMS=jl,coefficients=b)
    scores_ok=bool(np.all(np.isfinite(np.r_[bc,jl])) and np.min(np.r_[bc,jl])>=0 and np.max(np.r_[bc,jl])<=1+1e-8)
    pred_ok=bool(np.all(np.isfinite(np.r_[arrays['frontier_train'],arrays['frontier_eval']])) and np.min(np.r_[arrays['frontier_train'],arrays['frontier_eval']])>0)
    ok=bool(ok and scores_ok and pred_ok)
    meta=dict(status='QUALIFIED_'+best['kind'] if ok else 'UNQUALIFIED_REVIEW',qualified=ok,regular_mle=best['kind']=='INTERIOR',stochastic_decomposition_identified=True,selected=best,
        sigma_u=su,sigma_v=sv,loglik=best['loglik'],qualification=qual,scores_valid=scores_ok,predictions_valid=pred_ok,candidate_count=len(candidates),fit_seconds=time.perf_counter()-started,
        adapter='sfaR_likelihood_SciPy_centered_HN_boundary_v3' if dist=='HN' else 'sfaR_likelihood_SciPy_boundary_profile_v2',source_formula='frozen sfaR1.0.1; stable equivalent arithmetic; public optimization backends',condition_number=np.linalg.cond(X))
    write(out/'CANDIDATES.json',candidates);write(out/'OPTIMIZERS.json',optimizers);write(out/'STATUS.json',meta);np.savez(out/'PREDICTIONS.npz',**arrays)
    return meta,arrays
