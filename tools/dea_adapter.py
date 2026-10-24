from context import *
from dea_common import support_certificate
from dea_certify import residuals,verify_support,independent_fallback
import subprocess,time,traceback
BASE=ROOT

def implementation_hashes():
    return {str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/"tools/dea_worker.R",ROOT/"tools/dea_public.R",ROOT/"tools/dea_common.py",ROOT/"tools/dea_certify.py"]}

def fit_dea(x,y,x_eval,output_dir):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);ev=np.asarray(x_eval,dtype=float)
    assert x.ndim==2 and y.shape==(len(x),) and ev.ndim==2 and ev.shape[1]==x.shape[1]
    assert len(y)>0 and len(ev)>0 and x.shape[1]>0
    assert np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(ev).all() and (y>0).all()
    inputs=hashlib.sha256(str((x.shape,y.shape,ev.shape)).encode()+x.tobytes()+y.tobytes()+ev.tobytes()).hexdigest()
    sources=implementation_hashes();folder=Path(output_dir).resolve();done=folder/'COMPLETE.json'
    if done.exists():
        rec=read(done);assert rec['status']=='CERTIFIED' and rec['inputs_sha256']==inputs and rec['implementation_sha256']==sources
        for name,h in rec['files'].items():assert sha(folder/name)==h
        return dict(np.load(folder/'DEA.npz',allow_pickle=False)),rec
    folder.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    try:
        support=support_certificate(x,ev);verify_support(x,ev,support)
        np.savez_compressed(folder/'SUPPORT.npz',**support)
        mask=support['mask'];q=np.vstack([x,ev[mask]]);n,m=x.shape
        np.savetxt(folder/'train.csv',np.column_stack([x,y]),delimiter=',',header=','.join([f'x{i+1}' for i in range(m)]+['y']),comments='',fmt='%.17g')
        np.savetxt(folder/'query.csv',q,delimiter=',',header=','.join(f'x{i+1}' for i in range(m)),comments='',fmt='%.17g')
        cfg={'original_root':str(ROOT),'source':str(ROOT/'tools/dea_public.R'),
             'train':str(folder/'train.csv'),'query':str(folder/'query.csv'),'output':str(folder)}
        write(folder/'R_input.json',cfg)
        with (folder/'R_stdout.log').open('w') as log:
            subprocess.run([RUNTIME['Rscript'],'--vanilla',str(BASE/'tools/dea_worker.R'),str(folder/'R_input.json'),str(ROOT)],stdout=log,stderr=log,check=True,timeout=300)
        meta=read(folder/'R_metadata.json');a=np.fromfile(folder/'R_certificates.f64',dtype='<f8').reshape((meta['rows'],meta['columns']),order='F')
        assert a.shape==(len(q),2+n+1+m)
        value=a[:,0].copy();src=a[:,1].astype(int);w=a[:,2:2+n].copy();alpha=a[:,2+n].copy();beta=a[:,3+n:].copy()
        good,lo,hi,pr,du=residuals(x,y,q,w,alpha,beta)
        retry=np.flatnonzero((src==0)|~good|~np.isfinite(value));fallback=[]
        for i in retry:
            w[i],alpha[i],beta[i],value[i]=independent_fallback(x,y,q[i]);src[i]=3
            fallback.append({'query_row_1based':int(i+1),'method':'HiGHS independent primal and dual','value':float(value[i])})
        good,lo,hi,pr,du=residuals(x,y,q,w,alpha,beta)
        assert good.all() and np.isfinite(value).all() and (value>0).all()
        assert np.max(abs(value-lo))<=1e-6 and np.max(abs(value-hi))<=1e-6
        assert np.min(value[:n]-y)>=-1e-6
        ft=value[:n];fe=np.full(len(ev),-np.inf);fe[mask]=value[n:]
        arrays={'frontier_train':ft,'frontier_eval':fe,'te':y/ft,'support_mask':mask,
                'certificate_weights':w,'certificate_alpha':alpha,'certificate_beta':beta,
                'certificate_source':src,'primal_residual':pr,'dual_residual':du,'duality_gap':abs(hi-lo)}
        np.savez_compressed(folder/'DEA.npz',**arrays);write(folder/'FALLBACK.json',fallback)
        rec={'status':'CERTIFIED','package_status_record':'R_attempts.rds','numerical_qualification':'all_points_primal_dual',
             'domain_classification':'input_only_witnesses_and_separating_planes','n':n,'m':m,
             'evaluation_candidates':len(ev),'support_count':int(mask.sum()),'outside_count':int((~mask).sum()),
             'final_unresolved':0,'source_counts':{str(k):int(np.sum(src==k)) for k in (1,2,3)},
             'inputs_sha256':inputs,'implementation_sha256':sources,'elapsed_seconds':time.perf_counter()-started,
             'completed_utc':now(),'files':{p.name:sha(p) for p in sorted(folder.iterdir()) if p.is_file()}}
        write(done,rec);return arrays,rec
    except Exception:
        write(folder/'FAILED.json',{'status':'UNQUALIFIED','inputs_sha256':inputs,'implementation_sha256':sources,
            'traceback':traceback.format_exc(),'elapsed_seconds':time.perf_counter()-started,'created_utc':now()})
        raise
