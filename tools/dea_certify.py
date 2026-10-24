from context import *
from scipy.optimize import linprog
from dea_common import exact_separation_margin
def residuals(x,y,q,w,alpha,beta):
    wx=np.einsum('qi,im->qm',w,x,optimize=False)
    lower=np.einsum('qi,i->q',w,y,optimize=False)
    upper=alpha+np.einsum('qm,qm->q',beta,q,optimize=False)
    plane=alpha[:,None]+np.einsum('qm,im->qi',beta,x,optimize=False)
    primal=np.maximum.reduce([np.max(wx-q,axis=1),abs(w.sum(axis=1)-1),np.max(-w,axis=1),np.zeros(len(q))])
    dual=np.maximum.reduce([np.max(y[None,:]-plane,axis=1),np.max(-beta,axis=1),np.zeros(len(q))])
    good=np.isfinite(primal)&np.isfinite(dual)&np.isfinite(lower)&np.isfinite(upper)&(primal<=1e-8)&(dual<=1e-8)&(abs(upper-lower)<=1e-7)
    return good,lower,upper,primal,dual

def verify_support(x,ev,s):
    w=s['weights'];ab=s['separators'];mask=s['mask'];alpha=ab[:,0];beta=ab[:,1:]
    pv=np.maximum.reduce([np.max(np.einsum('qi,im->qm',w,x,optimize=False)-ev,axis=1),abs(w.sum(axis=1)-1),np.max(-w,axis=1),np.zeros(len(ev))])
    dv=np.maximum.reduce([np.max(alpha[:,None]-np.einsum('qm,im->qi',beta,x,optimize=False),axis=1),np.max(-beta,axis=1),beta.sum(axis=1)-1,np.zeros(len(ev))])
    margin=alpha-np.einsum('qm,qm->q',beta,ev,optimize=False)
    assert np.all(pv[mask]<=1e-8)
    assert np.all(dv[~mask]<=1e-9)
    # Recheck near-boundary exterior signs, including inherited certificates.
    for i in np.flatnonzero((~mask)&(margin<=1e-8)):
        assert exact_separation_margin(x,ev[i],beta[i])>0,'Unproven outside point'
    for i in np.flatnonzero(mask&np.any(beta>0,axis=1)&(margin>=-1e-8)):
        assert exact_separation_margin(x,ev[i],beta[i])<=0,'Exact separator contradicts inside label'

def independent_fallback(x,y,point):
    options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9}
    p=linprog(-y,A_ub=x.T,b_ub=point,A_eq=np.ones((1,len(y))),b_eq=[1],bounds=(0,None),method='highs-ds',options=options)
    d=linprog(np.r_[1.,point],A_ub=-np.column_stack([np.ones(len(y)),x]),b_ub=-y,bounds=[(None,None)]+[(0,None)]*x.shape[1],method='highs-ds',options=options)
    assert p.success and d.success,'Bounded independent fallback failed; preserve and stop'
    return p.x,d.x[0],d.x[1:],float(d.fun)
