"""Equivalent normal-half-normal QLE with explicit boundary-neighborhood search.

The scalar coordinate is w=log(sigma_v/sigma_u). Exact zero-variance endpoints
are evaluated separately. Tail agreement/monotonicity is a numerical screen,
not a certificate of a global minimum for every possible sample.
"""
from context import np, read, ROOT
from scipy.special import log_ndtr
from scipy.optimize import minimize_scalar

P = read(ROOT/'design/STOCHASTIC_POLICY.json')['stoned']

def qle_profile(res, points):
    res = np.asarray(res, dtype=float)
    assert res.ndim == 1 and np.isfinite(res).all() and len(res) > 0
    n = len(res)
    rms = float(np.sqrt(np.mean(res*res)))
    assert rms > 0, 'Zero residuals require the separate degenerate-limit policy'
    z = res/rms
    c = 1-2/np.pi
    k = np.sqrt(2/np.pi)
    offset = n*np.log(rms)

    def evaluate(w):
        w = np.atleast_1d(w)
        t = np.exp(w)[:, None]
        u = 1/np.sqrt(c+t*t)
        sig = u*np.hypot(1.,t)
        e = z[None,:]-k*u
        return n*np.log(sig[:,0])-np.sum(log_ndtr(-e/(sig*t)),axis=1)+.5*np.sum((e/sig)**2,axis=1)

    def objective(w):
        return float(evaluate(w)[0])

    # Same exact endpoints and tie convention as the preceding implementation.
    u1 = 1/np.sqrt(c)
    e1 = z-k*u1
    endpoints = [dict(objective=float(n*np.log(2.)+.5*np.sum(z*z)),rho=0.,
                      sigma_u=0.,sigma_v=rms,mu=0.,success=True,kind='sigma_u_zero')]
    boundary_v = float(n*np.log(u1)+.5*np.sum((e1/u1)**2)) if np.all(e1<=0) else np.inf
    endpoints.append(dict(objective=boundary_v,rho=1.,sigma_u=rms*u1,sigma_v=0.,
                          mu=k*rms*u1,success=True,kind='sigma_v_zero'))

    limit = P['QLE_log_ratio_initial_limit']
    tail_tol = P['QLE_tail_agreement_per_observation']*n
    tail_history = []
    while True:
        vals = evaluate([-limit,-limit+1,limit-1,limit])
        left_ok = (abs(vals[0]-boundary_v)<=tail_tol if np.isfinite(boundary_v)
                   else vals[0]>vals[1])
        right_ok = abs(vals[-1]-endpoints[0]['objective'])<=tail_tol
        tail_history.append(dict(log_ratio_limit=limit,objectives=vals.tolist(),
                                 left_ok=bool(left_ok),right_ok=bool(right_ok),
                                 left_exact_endpoint_feasible=bool(np.isfinite(boundary_v))))
        if left_ok and right_ok or limit>=P['QLE_log_ratio_max_limit']:
            break
        limit += P['QLE_tail_extension']
    tails_ok = bool(left_ok and right_ok)

    grid = np.linspace(-limit,limit,points)
    values = evaluate(grid)
    assert np.isfinite(values).all()
    candidates = endpoints.copy()
    local = np.flatnonzero((values[1:-1]<values[:-2]) & (values[1:-1]<values[2:]))+1
    brackets = [(float(grid[i-1]),float(grid[i+1]),'grid_local_minimum') for i in local]
    best_index = int(np.argmin(values))
    if 0<best_index<points-1:
        brackets.append((float(grid[best_index-1]),float(grid[best_index+1]),'best_sample_neighborhood'))
    # Explicitly search both old rho-grid endpoint neighborhoods even when no
    # sampled interior node beats the endpoint. This fixes the reported case.
    near_zero = np.sin(np.pi/(2*(points-1)))
    near_one = np.cos(np.pi/(2*(points-1)))
    w_right = float(.5*np.log1p(-near_zero*near_zero)-np.log(near_zero))
    w_left = float(.5*np.log1p(-near_one*near_one)-np.log(near_one))
    brackets.extend([(-limit,w_left,'mandatory_sigma_v_zero_neighborhood'),
                     (w_right,limit,'mandatory_sigma_u_zero_neighborhood'),
                     (float(grid[0]),float(grid[1]),'left_outer_interval'),
                     (float(grid[-2]),float(grid[-1]),'right_outer_interval')])

    def candidate(w, value, success, kind, **extra):
        t=np.exp(w);u=rms/np.sqrt(c+t*t);v=t*u
        return dict(objective=float(value),rho=float(1/np.hypot(1.,t)),sigma_u=float(u),
                    sigma_v=float(v),mu=float(k*u),success=bool(success),kind=kind,
                    log_ratio=float(w),**extra)

    candidates.append(candidate(float(grid[best_index]),values[best_index],True,'best_sample'))
    seen=set()
    for lo,hi,kind in brackets:
        if (lo,hi) in seen:continue
        seen.add((lo,hi))
        opt=minimize_scalar(objective,bounds=(lo,hi),method='bounded',
                            options={'xatol':P['QLE_optimizer_xatol'],'maxiter':1000})
        candidates.append(candidate(opt.x,opt.fun,opt.success,kind,bracket=[lo,hi],evaluations=int(opt.nfev)))
    best=min(candidates,key=lambda q:q['objective'])
    tie=P['QLE_tie_per_observation']*n
    for endpoint in endpoints:
        if endpoint['objective']<=best['objective']+tie:
            best=endpoint
            break
    selected={k:best[k] for k in ('rho','sigma_u','sigma_v','mu')}
    selected.update(objective=best['objective']+offset,selected_success=bool(best['success'] and tails_ok),
                    candidates=candidates,residual_rms=rms,coordinate='log(sigma_v/sigma_u)',
                    selected_kind=best['kind'],tail_checks=tail_history,
                    boundary_neighborhoods_searched=True,global_minimum_certified=False)
    return selected,dict(log_ratio=grid,rho=1/np.hypot(1.,np.exp(grid)),normalized_objective=values)
