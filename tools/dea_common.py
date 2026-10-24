from context import *
from scipy import sparse
import highspy
from fractions import Fraction
def exact_separation_margin(x,point,beta):
    """Exact Farkas sign for stored binary floats; independent of LP tolerances."""
    assert np.isfinite(x).all() and np.isfinite(point).all() and np.isfinite(beta).all()
    assert np.all(beta>=0)
    normal=[Fraction(float(v)) for v in beta]
    def dot(row):
        return sum((Fraction(float(v))*b for v,b in zip(row,normal)),Fraction(0))
    return min(dot(row) for row in x)-dot(point)

def key_for(x,ev):return hashlib.sha256(str((x.shape,ev.shape)).encode()+x.tobytes()+ev.tobytes()).hexdigest()

def model(cost,lower,upper,a,rowlo,rowhi):
    mat=sparse.csr_matrix(a);lp=highspy.HighsLp()
    lp.num_col_=mat.shape[1];lp.num_row_=mat.shape[0]
    lp.col_cost_=np.asarray(cost);lp.col_lower_=np.asarray(lower);lp.col_upper_=np.asarray(upper)
    lp.row_lower_=np.asarray(rowlo);lp.row_upper_=np.asarray(rowhi)
    lp.a_matrix_.format_=highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_=mat.indptr.astype(np.int32);lp.a_matrix_.index_=mat.indices.astype(np.int32);lp.a_matrix_.value_=mat.data
    h=highspy.Highs()
    for k,v in [('output_flag',False),('threads',1),('solver','simplex'),('primal_feasibility_tolerance',1e-9),('dual_feasibility_tolerance',1e-9),('random_seed',0),('time_limit',30.)]:
        assert h.setOptionValue(k,v)==highspy.HighsStatus.kOk
    assert h.passModel(lp)==highspy.HighsStatus.kOk
    return h

def support_certificate(x,ev):
    # min t: X'w - t <= query, sum(w)=1, w>=0, t>=0.
    n,m=x.shape;a=np.vstack([np.column_stack([-np.ones(m),x.T]),np.r_[0.,np.ones(n)]])
    h=model(np.r_[1.,np.zeros(n)],np.zeros(n+1),np.full(n+1,np.inf),a,np.r_[np.full(m,-np.inf),1.],np.r_[ev[0],1.])
    weights=[];planes=[];inside=[];margins=[]
    rid=np.arange(m,dtype=np.int32)
    for point in ev:
        assert h.changeRowsBounds(m,rid,np.full(m,-np.inf),point)==highspy.HighsStatus.kOk
        for attempt in range(3):
            if attempt<2:
                if attempt:h.clearSolver()
                h.run()
                assert h.getModelStatus()==highspy.HighsModelStatus.kOptimal
                sol=h.getSolution();w=np.asarray(sol.col_value)[1:];du=np.asarray(sol.row_dual)
            else:
                from scipy.optimize import linprog
                sol=linprog(np.r_[1.,np.zeros(n)],A_ub=a[:m],b_ub=point,A_eq=a[m:],b_eq=[1.],bounds=(0,None),method='highs-ds',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
                assert sol.success
                w=sol.x[1:];du=np.r_[sol.ineqlin.marginals,sol.eqlin.marginals]
            # Construct an actually supporting input-only hyperplane from the
            # proposed normal, rather than trusting a solver's approximate alpha.
            beta=np.maximum(-du[:m],0);beta/=max(1.,float(beta.sum()))
            alpha=float(np.nextafter(np.min(x@beta),-np.inf));gap=float(alpha-beta@point)
            primal=max(float(np.max(np.maximum(x.T@w-point,0))),abs(float(w.sum()-1)),float(np.max(np.maximum(-w,0))))
            dual=max(float(np.max(np.maximum(alpha-x@beta,0))),float(np.max(np.maximum(-beta,0))),max(0.,float(beta.sum()-1)))
            # An exterior certificate takes priority over a tolerance-feasible
            # primal witness. Near the boundary, establish its sign exactly.
            if dual<=1e-9 and gap>1e-8:ok=False;break
            if dual<=1e-9 and np.any(beta>0) and gap>=-1e-8:
                if exact_separation_margin(x,point,beta)>0:ok=False;break
            if primal<=1e-8:ok=True;break
        else:raise RuntimeError('Unresolved input-only support certificate')
        weights.append(w);planes.append(np.r_[alpha,beta]);inside.append(ok);margins.append(gap)
    return dict(mask=np.array(inside),weights=np.array(weights),separators=np.array(planes),margin=np.array(margins))
