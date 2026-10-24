from context import *
from scipy.integrate import quad

def quad_scores(a,ss):
    if a<=0:
        k=1/(1+abs(a));s=ss*k
        def w(v):return np.exp(a*k*v-.5*(k*v)**2)
        den=quad(w,0,np.inf,epsabs=1e-12,epsrel=1e-12)[0];bc=quad(lambda v:np.exp(-s*v)*w(v),0,np.inf,epsabs=1e-12,epsrel=1e-12)[0]/den
        eu=s*quad(lambda v:v*w(v),0,np.inf,epsabs=1e-12,epsrel=1e-12)[0]/den
    else:
        lo=max(-a,-12.);hi=12.
        def w(v):return np.exp(-.5*v*v)
        den=quad(w,lo,hi,epsabs=1e-12,epsrel=1e-12)[0];bc=quad(lambda v:np.exp(-ss*(v+a))*w(v),lo,hi,epsabs=1e-12,epsrel=1e-12)[0]/den
        eu=ss*quad(lambda v:(v+a)*w(v),lo,hi,epsabs=1e-12,epsrel=1e-12)[0]/den
    return bc,np.exp(-eu)

def score_check(meta,arr,ly,model,X=None):
    if meta['status']=='DEGENERATE_LIMIT':return dict(qualified=bool(np.all(arr['te_BC']==1) and np.all(arr['te_JLMS']==1)),kind='explicit limit convention',checks=len(ly)*2)
    if model=='StoNED_QLE':q=meta['decomposition'];su,sv=q['sigma_u'],q['sigma_v'];eps=arr['cnls_residual']-q['mu'];dist='HN'
    else:su,sv=meta['sigma_u'],meta['sigma_v'];eps=ly-np.einsum('ij,j->i',X,arr['coefficients']);dist=model.split('_')[-1]
    if su==0:
        err=max(np.max(abs(arr['te_BC']-1)),np.max(abs(arr['te_JLMS']-1)));return dict(qualified=bool(err<=1e-7),error=err,kind='zero-u identity',checks=len(ly)*2)
    if sv==0:
        ref=np.exp(eps);err=max(np.max(abs(arr['te_BC']-ref)),np.max(abs(arr['te_JLMS']-ref)));return dict(qualified=bool(err<=1e-7),error=err,kind='zero-v identity',checks=len(ly)*2)
    if dist=='HN':ss=su*sv/np.hypot(su,sv);a=-eps*su/(sv*np.hypot(su,sv))
    else:ss=sv;a=-eps/sv-sv/su
    ids=sorted(set([int(np.argmin(a)),int(np.argmax(a)),int(np.argsort(a)[len(a)//2])]+np.linspace(0,len(a)-1,min(7,len(a)),dtype=int).tolist()));records=[]
    for i in ids:
        bc,jl=quad_scores(a[i],ss);records.append(dict(index=i,BC_absolute_error=abs(bc-arr['te_BC'][i]),JLMS_absolute_error=abs(jl-arr['te_JLMS'][i])))
    error=max(max(r['BC_absolute_error'],r['JLMS_absolute_error']) for r in records)
    return dict(qualified=bool(error<=1e-7),error=error,kind='independent posterior quadrature',checks=2*len(ids),records=records)
