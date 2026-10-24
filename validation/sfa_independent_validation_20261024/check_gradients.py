"""Compare frozen production derivatives with independent complex-step ones.
The production math source is loaded without its filesystem context import.
"""
from validate import *
import ast
p=ROOT/'tools/stable_math.py';tree=ast.parse(p.read_text(),filename=str(p))
tree.body=[n for n in tree.body if not(isinstance(n,ast.ImportFrom) and n.module=='context')]
ns={'np':np};exec(compile(tree,str(p),'exec'),ns)
sources=read(HERE/'SOURCE_HASHES.json');sources[str(p)]=sha(p);write('SOURCE_HASHES.json',sources)
rows=[]
for inp in csv.DictReader((HERE/'INPUTS.csv').open()):
    obs=np.load(Path(inp['directory'])/'observed.npz');y=np.log(obs['y'])
    for form in ['CD','TL']:
        X=design(obs['x'],form);K=X.shape[1];n=len(y);ols=np.linalg.lstsq(X,y,rcond=None)[0];sd=float(np.sqrt(np.mean((y-np.sum(X*ols,axis=1))**2)))
        for dist in ['HN','EXP']:
            c=np.sqrt(2/np.pi) if dist=='HN' else 1.;var=1-2/np.pi if dist=='HN' else 1.
            for lam in [.1,1,10]:
                sv=sd/np.sqrt(1+var*lam*lam);su=sv*lam;b=ols.copy();b[0]+=c*su
                p=np.r_[b,2*np.log(su),2*np.log(sv)];h=1e-25;pp=np.tile(p.astype(complex),(len(p),1));pp[np.diag_indices(len(p))]+=1j*h
                e=y[None,:]-np.sum(X[None,:,:]*pp[:,None,:K],axis=2)
                gi=-ll_res(e,np.exp(pp[:,-2,None]/2),np.exp(pp[:,-1,None]/2),dist).sum(axis=1).imag/h
                ll,gr,_=ns['evaluate'](p,X,y,dist);ind=-float(ll_res(y-np.sum(X*b,axis=1),su,sv,dist).sum())
                ge=float(np.max(abs(gr-gi)))/n;le=abs(ll-ind)
                assert ge<1e-7 and le<1e-7,(inp['task'],form,dist,lam,ge,le)
                rows.append(dict(task=inp['task'],model=f'SFA_{form}_{dist}',lambda_start=lam,loglik_abs_difference=le,gradient_per_observation_max_difference=ge))
csvout('PRODUCTION_GRADIENT_CHECKS.csv',rows)
write('PRODUCTION_GRADIENT_CHECK.json',dict(status='PASS',points=len(rows),max_likelihood_error=max(r['loglik_abs_difference'] for r in rows),max_gradient_per_obs_error=max(r['gradient_per_observation_max_difference'] for r in rows)))
print(read(HERE/'PRODUCTION_GRADIENT_CHECK.json'))
