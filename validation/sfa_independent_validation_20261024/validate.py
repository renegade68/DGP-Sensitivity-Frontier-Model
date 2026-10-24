"""Frozen-input SFA validation. Does not import the production estimator/math."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,csv,hashlib,time,collections
from pathlib import Path
HERE=Path(__file__).resolve().parent;REV=HERE.parent
ROOT=REV/'frontier_production_v4_20261012';RUN=ROOT/'runs/production001'
sys.path.insert(0,str(REV/'efficiency_benchmark_20261007/environment/python_packages'))
import numpy as np
from scipy.special import log_ndtr,erfcx
from scipy.optimize import minimize,LinearConstraint
from scipy.optimize._numdiff import approx_derivative
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,data):
    (HERE/name).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csvout(name,rows):
    rows=list(rows)
    if not rows:return
    with (HERE/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def design(x,form):
    lx=np.log(x);m=x.shape[1]
    return np.column_stack([np.ones(len(x)),lx]+([.5*lx**2]+[lx[:,i]*lx[:,j] for i in range(m) for j in range(i+1,m)] if form=='TL' else []))
def ll_res(e,su,sv,dist):
    """Independent vectorized convolution density; supports complex-step AD."""
    if dist=='HN':
        s=np.sqrt(su*su+sv*sv)
        return np.log(2)-np.log(s)-.5*np.log(2*np.pi)-.5*(e/s)**2+log_ndtr(-su*e/(sv*s))
    t=-e/sv-sv/su;tail=t.real < -8
    # np.where would evaluate an overflowing unused expression; index explicitly.
    su=np.broadcast_to(su,e.shape);sv=np.broadcast_to(sv,e.shape)
    ans=np.empty_like(e)
    ans[tail]=-np.log(su[tail])-.5*(e[tail]/sv[tail])**2+np.log(erfcx(-t[tail]/np.sqrt(2)))-np.log(2)
    ans[~tail]=-np.log(su[~tail])+e[~tail]/su[~tail]+.5*(sv[~tail]/su[~tail])**2+log_ndtr(t[~tail])
    return ans
def nll(z,Q,res,dist):
    # z may be a batch of independent complex perturbations (rows).
    z=np.atleast_2d(z);pred=np.sum(Q[None,:,:]*z[:,None,:-2],axis=2)
    e=res[None,:]-pred;su=np.exp(z[:,-2,None]);sv=np.exp(z[:,-1,None])
    return -ll_res(e,su,sv,dist).mean(axis=1)
def objective(z,Q,res,dist):
    h=1e-25;zz=np.tile(z.astype(complex),(len(z),1));zz[np.diag_indices(len(z))]+=1j*h
    val=float(nll(z,Q,res,dist)[0]);grad=nll(zz,Q,res,dist).imag/h
    return val,grad
def prepare():
    cells=['U1_D04','U1_D06','U1_D10','U1_D16','U1_D20','U1_D36','U5_m3_n100_D04','U5_m3_n100_D10','U5_m5_n50_D04','U5_m5_n50_D10','U5_m5_n100_D04','U5_m5_n100_D10']
    models=['SFA_CD_HN','SFA_CD_EXP','SFA_TL_HN','SFA_TL_EXP'];sources={};inputs=[];states=collections.Counter()
    for cell in cells:
        for rep in [1,100]:
            task=f'{cell}_r{rep:04d}';p=next((RUN/task).glob('attempt_*/COMPLETE.json')).parent;c=read(p/'COMPLETE.json')
            for name in ['observed.npz','train.csv','TASK.json','AUDIT.json']+[m+'/'+f for m in models for f in ['STATUS.json','CANDIDATES.json']]:
                h=sha(p/name);assert h==c['files'][name];sources[str(p/name)]=h
            for model in models:states[read(p/model/'STATUS.json')['status']]+=1
            inputs.append(dict(task=task,directory=str(p)))
    csvout('INPUTS.csv',inputs)
    write('PROTOCOL.json',dict(cells=cells,replications=[1,100],models=models,datasets=24,fits=96,
        selection='Fixed named DGP and replication sample, not a random sample or outcome-selected sample',frozen_states=dict(states),
        optimizer='Independent likelihood with complex-step gradient, L-BFGS-B multistart and Powell restart; SLSQP boundary checks',
        starts=['OLS lambda 0.05,0.3,1,3,10','boundary sigma_v/sigma_u 1e-2,1e-4,1e-6 with inward intercept shift','best saved interior'],
        R_comparison='sfaR nlminb multiple starts (HN and EXP); frontier package (HN only)',
        meaningful_improvement_per_observation=1e-7,likelihood_tie_per_observation=[0,1e-10,1e-8,1e-6],
        no_global_optimum_claim=True,production_modified=False))
    write('SOURCE_HASHES.json',sources);print('Prepared',dict(states),flush=True)
def run():
    cases=[];profiles=[];checks=[];attempts=[];hashes=read(HERE/'SOURCE_HASHES.json')
    for inp in csv.DictReader((HERE/'INPUTS.csv').open()):
        task=inp['task'];p=Path(inp['directory']);obs=np.load(p/'observed.npz');x=obs['x'];y=np.log(obs['y']);n=len(y)
        for form in ['CD','TL']:
            X=design(x,form);Q,R=np.linalg.qr(X);K=X.shape[1];ols=np.linalg.lstsq(X,y,rcond=None)[0];res=y-np.sum(X*ols,axis=1);scale=float(np.sqrt(np.mean(res**2)));res=res/scale
            for dist in ['HN','EXP']:
                started=time.perf_counter();model=f'SFA_{form}_{dist}';old=read(p/model/'STATUS.json');cand=read(p/model/'CANDIDATES.json');boundary=next(c for c in cand if c['kind']=='SIGMA_V_ZERO')
                b=np.array(boundary['beta']);theta=np.sum(X*(b-ols),axis=1)/scale;bt=np.sum(Q*theta[:,None],axis=0);bsu=boundary['sigma_u']/scale
                meanu=np.sqrt(2/np.pi) if dist=='HN' else 1.;varu=1-2/np.pi if dist=='HN' else 1.
                starts=[]
                for lam in [.05,.3,1.,3.,10.]:
                    sv=1/np.sqrt(1+varu*lam*lam);su=lam*sv;t=np.sum(Q,axis=0)*meanu*su
                    starts.append((f'OLS_lambda_{lam}',np.r_[t,np.log(su),np.log(sv)]))
                for ratio in [1e-2,1e-4,1e-6]:
                    sv=bsu*ratio;delta=sv*np.sqrt(2*np.log(1/ratio))+1e-12
                    starts.append((f'boundary_ratio_{ratio}',np.r_[bt+np.sum(Q,axis=0)*delta,np.log(bsu),np.log(sv)]))
                bi=max((c for c in cand if c['kind']=='INTERIOR'),key=lambda c:c['loglik'])
                t=np.sum(Q*((np.sum(X*(np.array(bi['beta'])-ols),axis=1)/scale)[:,None]),axis=0)
                starts.append(('saved_best_interior',np.r_[t,np.log(bi['sigma_u']/scale),np.log(bi['sigma_v']/scale)]))
                fun=lambda z:objective(z,Q,res,dist)
                g=fun(starts[2][1])[1];gn=approx_derivative(lambda z:nll(z,Q,res,dist),starts[2][1],method='3-point').ravel()
                ge=float(np.max(abs(g-gn)));assert ge<1e-6,(task,model,ge)
                checks.append(dict(task=task,model=model,complex_step_vs_finite_diff_max=ge))
                solutions=[];bounds=[(None,None)]*K+[(-24,8),(-24,8)]
                for label,z in starts:
                    z[-2:]=np.clip(z[-2:],-23.99,7.99)
                    sol=minimize(fun,z,method='L-BFGS-B',jac=True,bounds=bounds,options=dict(maxiter=1500,ftol=1e-14,gtol=1e-8,maxls=50))
                    ll=-float(sol.fun)*n-n*np.log(scale);solutions.append((ll,sol.x,label))
                    attempts.append(dict(task=task,model=model,start=label,solver='independent_complex_step_LBFGSB',loglik=ll,success=bool(sol.success),iterations=int(sol.nit),gradient_max=float(np.max(abs(fun(sol.x)[1]))),message=str(sol.message)))
                winner=max(solutions,key=lambda t:t[0])
                # A different, derivative-free algorithm from a moderate OLS start.
                sol=minimize(lambda z:float(nll(z,Q,res,dist)[0]),starts[2][1],method='Powell',bounds=bounds,options=dict(maxiter=100,maxfev=12000,xtol=1e-8,ftol=1e-11))
                ll=-float(sol.fun)*n-n*np.log(scale);solutions.append((ll,sol.x,'Powell_OLS_lambda1'))
                attempts.append(dict(task=task,model=model,start='OLS_lambda_1.0',solver='independent_Powell',loglik=ll,success=bool(sol.success),iterations=int(sol.nit),gradient_max=float(np.max(abs(fun(sol.x)[1]))),message=str(sol.message)))
                winner=max(solutions,key=lambda t:t[0]);z=winner[1];newb=ols+np.linalg.solve(R,z[:-2]*scale);su=float(np.exp(z[-2])*scale);sv=float(np.exp(z[-1])*scale)
                # Convex boundary subproblem, distinct solver from production QP/LP.
                def bf(t):
                    r=np.sum(Q*t,axis=1)-res
                    return (.5*float(np.sum(r*r)),np.sum(Q*r[:,None],axis=0)) if dist=='HN' else (float(r.sum()),Q.sum(axis=0))
                bsol=minimize(bf,bt,method='SLSQP',jac=True,constraints=[LinearConstraint(Q,res,np.inf)],options=dict(maxiter=1000,ftol=1e-12))
                br=np.sum(Q*bsol.x,axis=1)-res;bs=float(np.sqrt(np.mean(br*br)) if dist=='HN' else np.mean(br))*scale
                bll=n*(np.log(2)-np.log(bs)-.5*np.log(2*np.pi)-.5) if dist=='HN' else -n*np.log(bs)-n
                # Both fixed-beta and inward-shift paths; a fixed-beta endpoint alone is misleading at contacting observations.
                e=y-np.sum(X*b,axis=1);bsu0=boundary['sigma_u']
                for ratio in [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-8,1e-10]:
                    tiny=bsu0*ratio
                    for path,shift in [('fixed_beta',0.),('inward_intercept',tiny*np.sqrt(2*np.log(1/ratio))+max(0.,float(e.max()))+1e-14)]:
                        pll=float(ll_res(e-shift,bsu0,tiny,dist).sum())
                        profiles.append(dict(task=task,model=model,path=path,sigma_v_over_sigma_u=ratio,intercept_shift=shift,loglik=pll,boundary_minus_path=boundary['loglik']-pll))
                result=dict(task=task,model=model,n=n,K=K,old_status=old['status'],old_loglik=old['loglik'],old_sigma_u=old['sigma_u'],old_sigma_v=old['sigma_v'],
                    best_independent_interior_loglik=winner[0],interior_minus_frozen=winner[0]-old['loglik'],best_start=winner[2],sigma_u=su,sigma_v=sv,beta=newb.tolist(),
                    independent_gradient_max=float(np.max(abs(fun(z)[1]))),search_logscale_bound_hit=bool(np.any(z[-2:]< -23.99) or np.any(z[-2:]>7.99)),
                    stored_boundary_loglik=boundary['loglik'],independent_boundary_loglik=bll,boundary_difference=bll-boundary['loglik'],boundary_success=bool(bsol.success),boundary_primal_violation=max(0.,-float(br.min()))*scale,
                    old_u_zero_loglik=next(c['loglik'] for c in cand if c['kind']=='SIGMA_U_ZERO'),seconds=time.perf_counter()-started)
                cases.append(result);write('PYTHON_RESULTS.json',cases);csvout('OPTIMIZER_ATTEMPTS.csv',attempts);csvout('BOUNDARY_PATHS.csv',profiles);csvout('GRADIENT_CHECKS.csv',checks)
                print(len(cases),task,model,'deltaLL',round(result['interior_minus_frozen'],9),'sv',sv,'sec',round(result['seconds'],2),flush=True)
    assert all(sha(k)==h for k,h in hashes.items());write('PYTHON_DONE.json',dict(fits=len(cases),sources_unchanged=True))
if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
