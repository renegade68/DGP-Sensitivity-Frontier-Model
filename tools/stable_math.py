"""Equivalent production HN/EXP formulas, for diagnosis only (no truth input).

Tail identities use scipy.special.erfcx. Mills excess uses Laplace's continued
fraction in the far tail to avoid subtracting two nearly equal large numbers.
Parameters match sfaR: beta, log(sigma_u^2), log(sigma_v^2).
"""
from context import np
from scipy.special import log_ndtr, erfcx
LOG2PI=np.log(2*np.pi)
def mills(a):
    a=np.asarray(a,dtype=float);r=np.empty_like(a);delta=np.empty_like(a)
    low=a < -8
    x=-a[low];den=np.zeros_like(x)
    for k in range(200,1,-1):den=k/(x+den)
    delta[low]=1/(x+den);r[low]=x+delta[low]
    mid=(a<=0)&~low
    r[mid]=np.sqrt(2/np.pi)/erfcx(-a[mid]/np.sqrt(2));delta[mid]=a[mid]+r[mid]
    hi=a>0
    r[hi]=np.exp(-.5*a[hi]**2-.5*LOG2PI-log_ndtr(a[hi]));delta[hi]=a[hi]+r[hi]
    return r,delta
def te(a,ss):
    a=np.asarray(a);r,delta=mills(a);lo=a<=0;logbc=np.empty_like(a)
    logbc[lo]=np.log(erfcx((ss-a[lo])/np.sqrt(2)))-np.log(erfcx(-a[lo]/np.sqrt(2)))
    logbc[~lo]=-ss*a[~lo]+.5*ss**2+log_ndtr(a[~lo]-ss)-log_ndtr(a[~lo])
    return np.exp(logbc),np.exp(-ss*delta)
def evaluate(p,X,y,dist,gradient=True,residual=None):
    K=X.shape[1];e=y-np.einsum('ij,j->i',X,p[:K]) if residual is None else residual;su=np.exp(p[K]/2);sv=np.exp(p[K+1]/2)
    if dist=='HN':
        sig=np.hypot(su,sv);lam=su/sv;t=e/sig;a=-lam*t;r,delta=mills(a);q=(su/sig)**2;v=(sv/sig)**2
        ll=np.log(2)-np.log(sig)-.5*LOG2PI-.5*t*t+log_ndtr(a)
        beta=np.einsum('ij,i->j',X,(t+lam*r)/sig)
        gu=.5*np.sum(q*(t*t-1)+v*a*r)
        gv=.5*np.sum(v*(t*t-1)-(1+v)*a*r)
        ss=su*sv/sig
    else:
        t=e/sv;h=sv/su;a=-t-h;r,delta=mills(a);lo=a<0;ll=np.empty_like(a)
        ll[lo]=-np.log(su)-.5*t[lo]**2+np.log(.5*erfcx(-a[lo]/np.sqrt(2)))
        ll[~lo]=-np.log(su)+e[~lo]/su+.5*h*h+log_ndtr(a[~lo])
        beta_score=np.empty_like(t);variance_score=np.empty_like(t)
        beta_score[lo]=(t[lo]+delta[lo])/sv
        variance_score[lo]=t[lo]*t[lo]+delta[lo]*(t[lo]-h)
        # In the opposite tail, t and delta nearly cancel when sigma_v is
        # tiny. Use the original algebra there, not the small-u rearrangement.
        beta_score[~lo]=(r[~lo]-h)/sv
        variance_score[~lo]=h*h+r[~lo]*(t[~lo]-h)
        beta=np.einsum('ij,i->j',X,beta_score)
        gu=.5*np.sum(-1+h*delta)
        gv=.5*np.sum(variance_score)
        ss=sv
    return -float(np.sum(ll)),-np.r_[beta,gu,gv],te(a,ss)
