"""Solve convex boundary problems from OLS-enveloping starts, not stored optima."""
from validate import *
from scipy.optimize import nnls
rows=[]
for inp in csv.DictReader((HERE/'INPUTS.csv').open()):
    p=Path(inp['directory']);obs=np.load(p/'observed.npz');y=np.log(obs['y']);n=len(y)
    for form in ['CD','TL']:
        X=design(obs['x'],form);Q,R=np.linalg.qr(X);ols=np.linalg.lstsq(X,y,rcond=None)[0];res=y-np.sum(X*ols,axis=1);scale=float(np.sqrt(np.mean(res*res)));res=res/scale
        start=Q.sum(axis=0)*(float(res.max())+.1)
        for dist in ['HN','EXP']:
            model=f'SFA_{form}_{dist}'
            def f(t):
                r=np.sum(Q*t,axis=1)-res
                return (.5*float(np.sum(r*r)),np.sum(Q*r[:,None],axis=0)) if dist=='HN' else (float(r.sum()),Q.sum(axis=0))
            sol=minimize(f,start,method='SLSQP',jac=True,constraints=[LinearConstraint(Q,res,np.inf)],options=dict(maxiter=2000,ftol=1e-12))
            residual=(np.sum(Q*sol.x,axis=1)-res)*scale;su=float(np.sqrt(np.mean(residual**2)) if dist=='HN' else np.mean(residual))
            ll=n*(np.log(2)-np.log(su)-.5*np.log(2*np.pi)-.5) if dist=='HN' else -n*np.log(su)-n
            old=next(c for c in read(p/model/'CANDIDATES.json') if c['kind']=='SIGMA_V_ZERO')
            b=ols+np.linalg.solve(R,sol.x*scale);frontier_rel=float(np.max(abs(np.expm1(np.sum(X*(b-np.array(old['beta'])),axis=1)))))
            active=residual<1e-7;target=np.sum(Q*residual[:,None],axis=0) if dist=='HN' else Q.sum(axis=0)
            dual,stationarity_l2=nnls(Q[active].T,target,maxiter=10000)
            comp=float(np.max(abs(residual[active]*dual)))
            rows.append(dict(task=inp['task'],model=model,success=bool(sol.success),solver_message=str(sol.message),iterations=int(sol.nit),primal_violation=max(0.,-float(residual.min())),stationarity_l2=float(stationarity_l2),complementarity_max=comp,loglik_difference=ll-old['loglik'],max_train_frontier_relative_difference=frontier_rel))
csvout('INDEPENDENT_BOUNDARY_CHECKS.csv',rows)
ok=all(max(r['primal_violation'],r['stationarity_l2'],r['complementarity_max'])<1e-7 and abs(r['loglik_difference'])<1e-6 for r in rows)
write('INDEPENDENT_BOUNDARY_CHECK.json',dict(status='PASS_KKT_AND_LIKELIHOOD' if ok else 'REVIEW',fits=len(rows),solver_success_count=sum(r['success'] for r in rows),max_primal_violation=max(r['primal_violation'] for r in rows),max_stationarity_l2=max(r['stationarity_l2'] for r in rows),max_complementarity=max(r['complementarity_max'] for r in rows),max_loglik_abs_difference=max(abs(r['loglik_difference']) for r in rows),max_train_frontier_relative_difference=max(r['max_train_frontier_relative_difference'] for r in rows)))
print(read(HERE/'INDEPENDENT_BOUNDARY_CHECK.json'))
