"""Executable DGP with explicit transformations, legacy RNG and no estimators."""
from context import *
from scipy.special import ndtr, ndtri

def frontier(x, spec):
    family=spec['family']
    if family in ('CD','CD_interaction_transform'):
        p=np.array(spec['powers']);cd=np.prod(x**p,axis=1)
        if family=='CD':return cd
        mu=(10.**(p+1)-1)/(9*(p+1));M=np.prod(mu)
        additive=M*(np.sum(x**p/mu,axis=1)-(len(p)-1))
        return additive+spec['alpha']*(cd-additive)
    if family=='CES':return np.sum(np.array(spec['weights'])*x**spec['rho'],axis=1)**(spec['returns']/spec['rho'])
    if family=='PWL':
        a=np.array(spec['facets']);return np.min(a[:,0,None]+np.einsum('ij,kj->ik',a[:,1:],x,optimize=False),axis=0)
    raise ValueError(family)

def copula(x,rho):
    q=(x-1)/9
    endpoints=int(np.sum((q<=0)|(q>=1)))
    q=np.clip(q,np.nextafter(0.,1.),np.nextafter(1.,0.))
    z=ndtri(q);m=x.shape[1];L=np.linalg.cholesky((1-rho)*np.eye(m)+rho*np.ones((m,m)))
    transformed=np.einsum('ij,kj->ik',z,L,optimize=False)
    assert np.isfinite(transformed).all()
    return 1+9*ndtr(transformed),endpoints

def generate(cell,rep):
    n,m=cell['n'],cell['m'];s=cell['seed_rule'];e=cell['error'];extra={}
    if 'data_seed_sequence' in s:
        r=np.random.default_rng(np.random.SeedSequence([20261002,2024,m,n,rep]))
        x=r.uniform(1,10,(n,m));base_u=r.exponential(scale=1/3,size=n)
        u=base_u*(e['mean_u']/(1/3));extra['base_u']=base_u
        z=np.random.default_rng(np.random.SeedSequence([20261002,2024,402,m,n,rep])).normal(size=n)
    else:
        def rng(k):return np.random.default_rng(np.random.SeedSequence([20261008,30,m,rep,k]))
        x=rng(1).uniform(1,10,(100,m))[:n];q=rng(2).uniform(size=100)[:n];z=rng(3).standard_normal(100)[:n]
        if e['u_distribution']=='exponential':u=-e['sd_u']*np.log1p(-q)
        elif e['u_distribution']=='half_normal':u=e['halfnormal_underlying_sd']*ndtri((1+q)/2)
        elif e['u_distribution']=='zero':u=np.zeros(n)
        else:raise ValueError(e['u_distribution'])
        extra['base_uniform']=q
    extra['base_normal']=z.copy();extra['base_x']=x.copy()
    rho=cell['input']['latent_rho'];endpoints=0
    if rho:x,endpoints=copula(x,rho)
    kind=e['v_law'];sv=e['sd_v_total']
    if kind=='normal':v=sv*z+e['log_noise_mean']
    elif kind=='heteroskedastic_normal':v=sv*z*(1+e['kappa']*(2*x[:,0]-11)/9)/np.sqrt(e['normalizer_squared'])
    elif kind=='gaussian_scale_mixture':
        B=np.random.default_rng(np.random.SeedSequence([20261008,82,m,rep,11])).uniform(size=100)[:n]<e['mixture_probability']
        extra['mixture_indicator']=B
        v=sv*z*(1+(e['sd_multiplier']-1)*B)/np.sqrt(e['normalizer_squared'])
    else:raise ValueError(kind)
    xe=np.random.default_rng(np.random.SeedSequence([20261008,71,m,0,4])).uniform(1,10,(5000,m))
    eval_measure=np.zeros(5000,dtype=np.int8)
    if rho:
        ce=np.random.default_rng(np.random.SeedSequence([20261008,81,m,0,4])).uniform(1,10,(5000,m))
        ce,k=copula(ce,rho);endpoints+=k
        xe=np.vstack([xe,ce]);eval_measure=np.r_[eval_measure,np.ones(5000,dtype=np.int8)]
    f=frontier(x,cell['frontier'])
    # Preserve historical U3/U4 evaluation order, including the final centering
    # addition; mathematically equivalent regrouping changes low binary bits.
    y=f*np.exp(-u+sv*z+e['log_noise_mean']) if 'data_seed_sequence' in s else f*np.exp(v-u)
    return dict(x=x,y=y,u=u,v=v,true_frontier=f,true_te=np.exp(-u),x_eval=xe,
        true_frontier_eval=frontier(xe,cell['frontier']),evaluation_measure=eval_measure,
        endpoint_events=np.array(endpoints),**extra)

def materialize(source):
    import copy
    c=copy.deepcopy(source);c['error']['v_law']='normal'
    t=c.pop('transformation',None)
    if t:
        if 'rho_latent' in t:c['input']['latent_rho']=t['rho_latent']
        elif 'kappa' in t:c['error'].update(v_law='heteroskedastic_normal',kappa=t['kappa'],normalizer_squared=t['normalizer_squared'])
        elif 'mixture_probability' in t:c['error'].update(v_law='gaussian_scale_mixture',**{k:t[k] for k in ['mixture_probability','sd_multiplier','normalizer_squared']})
        else:raise ValueError('Unmaterialized transformation')
    c['source_record_sha256']=object_hash(source)
    c['source_transform']=t
    c['effective_dgp_sha256']=object_hash({k:c[k] for k in ['n','m','input','frontier','error','seed_rule','integrated_evaluation']})
    return c

def save_task(task,cell,rep,eat_seed):
    task.mkdir(parents=True,exist_ok=False);z=generate(cell,rep)
    np.savez_compressed(task/'analysis_truth.npz',**z)
    observed={k:z[k] for k in ['x','y','x_eval']};np.savez_compressed(task/'observed.npz',**observed)
    m=cell['m'];names=[f'x{j+1}' for j in range(m)]
    np.savetxt(task/'train.csv',np.column_stack([z['x'],z['y']]),delimiter=',',header=','.join(names+['y']),comments='',fmt='%.17g')
    np.savetxt(task/'eval.csv',z['x_eval'],delimiter=',',header=','.join(names),comments='',fmt='%.17g')
    write(task/'TASK.json',dict(cell_id=cell['cell_id'],replication_id=rep,eat_seed=eat_seed,n=cell['n'],m=m,
        effective_dgp_sha256=cell['effective_dgp_sha256'],dataset_sha256=arrays_hash({k:z[k] for k in ['x','y','u','v','true_frontier','true_te']}),
        observed_sha256=arrays_hash(observed),evaluation_counts={'uniform_box':5000,'copula':5000 if cell['input']['latent_rho'] else 0},
        files={p.name:sha(p) for p in task.iterdir() if p.is_file()}))
    return z
