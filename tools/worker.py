"""One observed-only stage; numerical failures are records, schema defects stop."""
from context import *
import argparse,time,traceback,subprocess
from schema import save_model,classify_failure,normalize_deterministic

def main(stage,task):
    task=Path(task);z=dict(np.load(task/'observed.npz'));x,y,xe=z['x'],z['y'],z['x_eval'];n,ne=len(y),len(xe)
    if stage=='domain':
        from dea_common import support_certificate
        from dea_certify import verify_support
        s=support_certificate(x,xe);verify_support(x,xe,s);np.savez_compressed(task/'INPUT_SUPPORT.npz',**s);return
    if stage=='normalize':normalize_deterministic(task);return
    if stage in ('svf','dea','ceat'):
        import deterministic
        name={'svf':'SVF_PAPER','dea':'DEA_VRS','ceat':'CEAT'}[stage]
        if stage=='ceat' and not (task/'CEAT_leaves.f64').exists():
            meta=dict(status='DEPENDENCY_UNAVAILABLE',qualified=False);write(task/'CEAT_status.json',meta);save_model(task,name,meta,{},n,ne);return
        try:
            if stage=='svf':deterministic.svf(z,task)
            else:deterministic.dea(z,task,leaf=stage=='ceat')
        except (FloatingPointError,OverflowError) as exc:
            write(task/f'{name}_status.json',dict(status='NUMERICAL_NONFINITE',qualified=False,error=repr(exc),traceback=traceback.format_exc()))
        except subprocess.TimeoutExpired as exc:
            meta=dict(status='TIMEOUT',qualified=False,error=repr(exc));write(task/f'{name}_status.json',meta);save_model(task,name,meta,{},n,ne)
        return
    name=stage
    if name.startswith('SFA_'):
        import sfa_adapter
        from independent_checks import score_check
        _,form,dist=name.split('_');f=task/'sfaR_baseline'/f'{name}_params.f64'
        baseline=np.fromfile(f,dtype='<f8') if f.exists() else None
        try:meta,a=sfa_adapter.fit(x,y,xe,form,dist,task/name,baseline)
        except (FloatingPointError,OverflowError) as exc:
            classify_failure(task,name,'NUMERICAL_NONFINITE',n,ne,repr(exc));return
        if a:
            sc=score_check(meta,a,np.log(y),name,sfa_adapter.design(x,form));meta['qualified']=bool(meta['qualified'] and sc['qualified']);meta['score_check']=sc
            np.r_[a['coefficients'],meta['sigma_u'],meta['sigma_v']].astype('<f8').tofile(task/name/'SELECTED.f64')
            np.r_[a['frontier_train'],a['frontier_eval']].astype('<f8').tofile(task/name/'FRONTIER.f64')
        raw=read(task/'sfaR_baseline/STATUS.json').get(name,{})
        meta['package_converged']='successful convergence' in str(raw.get('optStatus','')) if raw.get('status')=='COMPUTED_RAW_NOT_AUTO_QUALIFIED' else None
        save_model(task,name,meta,a,n,ne);return
    if name=='StoNED_QLE':
        import stoned_adapter
        from independent_checks import score_check
        from pyomo.common.errors import ApplicationError
        try:meta,a=stoned_adapter.fit(x,y,xe,task/name)
        except (ApplicationError,FloatingPointError,OverflowError) as exc:
            classify_failure(task,name,'CNLS_BACKEND_FAILURE' if isinstance(exc,ApplicationError) else 'NUMERICAL_NONFINITE',n,ne,dict(error=repr(exc),traceback=traceback.format_exc()));return
        if a:
            sc=score_check(meta,a,np.log(y),name);meta['qualified']=bool(meta['qualified'] and sc['qualified']);meta['score_check']=sc
        save_model(task,name,meta,a,n,ne);return
    raise ValueError('Unknown stage '+stage)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage');ap.add_argument('task');args=ap.parse_args();main(args.stage,args.task)
