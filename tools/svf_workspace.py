from context import *
import time,highspy,clarabel
from scipy import sparse
from svf_math import _qp_matrices,_pass_sparse_qp,spline_design,spline_ranges
PRIMAL_GATE=1e-6
DUAL_GATE=1e-6
GAP_GATE=1e-5
class Workspace:
    """Reuse matrix construction, but cold-start every QP candidate."""
    def __init__(self, x, y, d):
        self.x, self.y, self.d = np.asarray(x), np.asarray(y), int(d)
        self.ranges = spline_ranges(x)
        self.phi = spline_design(x, self.d, self.ranges)
        self.n, self.q = self.phi.shape
        self.a, self.lo, self.hi, self.cost, self.clo, self.chi = _qp_matrices(
            self.phi, self.y, x.shape[1], self.d, 1., 0.)
        self.h = _pass_sparse_qp(self.a,self.lo,self.hi,self.cost,
                                  self.clo,self.chi,self.q,threads=1)
        for key,val in [('primal_feasibility_tolerance',1e-8),
                        ('dual_feasibility_tolerance',1e-8),
                        ('qp_regularization_value',1e-9),('time_limit',120.)]:
            if self.h.setOptionValue(key,val) != highspy.HighsStatus.kOk:
                raise RuntimeError(f'Unsupported solver setting: {key}')
        self.slack_idx = np.arange(self.q,self.q+self.n,dtype=np.int32)
        self.upper_idx = np.arange(self.n,2*self.n,dtype=np.int32)
        self.fallback_solver = None

    def solve(self, c, eps):
        self.cost[self.q:] = c
        self.hi[self.n:2*self.n] = self.y+eps
        if self.h.changeColsCost(self.n,self.slack_idx,self.cost[self.q:]) != highspy.HighsStatus.kOk:
            raise RuntimeError('Cost update failed')
        if self.h.changeRowsBounds(self.n,self.upper_idx,self.lo[self.n:2*self.n],
                                   self.hi[self.n:2*self.n]) != highspy.HighsStatus.kOk:
            raise RuntimeError('Bound update failed')
        self.h.clearSolver()
        started = time.perf_counter()
        self.h.run()
        elapsed = time.perf_counter()-started
        status = self.h.modelStatusToString(self.h.getModelStatus())
        if self.h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            return self.solve_fallback(f'HiGHS status: {status}',elapsed)
        solution = self.h.getSolution()
        z = np.asarray(solution.col_value)
        rd,cd = np.asarray(solution.row_dual),np.asarray(solution.col_dual)
        activity = self.a@z
        primal = max(float(np.max(np.maximum(self.lo-activity,0))),
                     float(np.max(np.maximum(activity-self.hi,0))),
                     float(np.max(np.maximum(self.clo-z,0))),
                     float(np.max(np.maximum(z-self.chi,0))))
        beta = z[:self.q]
        grad = self.cost.copy()
        grad[:self.q] += 2*beta
        stationarity = float(np.max(abs(grad-self.a.T@rd-cd))/(1+np.max(abs(grad))))
        dual_obj = -float(beta@beta)
        sign_error = 0.
        for multipliers, lower, upper in [(rd,self.lo,self.hi),(cd,self.clo,self.chi)]:
            pos,neg = np.maximum(multipliers,0),np.minimum(multipliers,0)
            lf,uf = np.isfinite(lower),np.isfinite(upper)
            dual_obj += float(pos[lf]@lower[lf]+neg[uf]@upper[uf])
            sign_error = max(sign_error,float(np.max(pos[~lf],initial=0)),
                             float(np.max(-neg[~uf],initial=0)))
        objective = float(beta@beta+self.cost@z)
        gap = abs(objective-dual_obj)/(1+abs(objective))
        dual = max(stationarity,sign_error/(1+c))
        diag = {'primal':primal,'dual':dual,'relative_gap':gap,
                'objective':objective,'solve_seconds':elapsed,'solver':'HiGHS'}
        if primal>PRIMAL_GATE or dual>DUAL_GATE or gap>GAP_GATE:
            return self.solve_fallback(f'HiGHS KKT: primal={primal}, dual={dual}, gap={gap}',elapsed)
        return beta.copy(), diag

    def solve_fallback(self,reason,previous_seconds):
        """Solve the identical QP by an interior-point method; never skip a grid cell."""
        started=time.perf_counter()
        lf,uf=np.isfinite(self.lo),np.isfinite(self.hi)
        clf,cuf=np.isfinite(self.clo),np.isfinite(self.chi)
        rhs=np.r_[-self.lo[lf],self.hi[uf],-self.clo[clf],self.chi[cuf]]
        if self.fallback_solver is None:
            eye=sparse.eye(self.q+self.n,format='csc')
            self.g=sparse.vstack([-self.a[lf],self.a[uf],-eye[clf],eye[cuf]],format='csc')
            p=sparse.diags(np.r_[np.full(self.q,2.),np.zeros(self.n)],format='csc')
            settings=clarabel.DefaultSettings()
            settings.verbose=False
            settings.max_threads=1
            settings.max_iter=300
            settings.time_limit=120.
            settings.tol_feas=settings.tol_gap_abs=settings.tol_gap_rel=1e-9
            self.fallback_solver=clarabel.DefaultSolver(p,self.cost,self.g,rhs,
                [clarabel.NonnegativeConeT(len(rhs))],settings)
        else:
            self.fallback_solver.update(q=self.cost,b=rhs)
        sol=self.fallback_solver.solve()
        if str(sol.status)!='Solved':
            raise RuntimeError(f'Both solvers failed: {reason}; CLARABEL {sol.status}')
        z,lam=np.asarray(sol.x),np.asarray(sol.z)
        beta=z[:self.q]
        activity=self.a@z
        primal=max(float(np.max(np.maximum(self.lo-activity,0))),
                   float(np.max(np.maximum(activity-self.hi,0))),
                   float(np.max(np.maximum(self.clo-z,0))),
                   float(np.max(np.maximum(z-self.chi,0))))
        grad=self.cost.copy();grad[:self.q]+=2*beta
        dual=max(float(np.max(abs(grad+self.g.T@lam))/(1+np.max(abs(grad)))),
                 float(np.max(np.maximum(-lam,0))/(1+np.max(abs(self.cost)))))
        objective=float(beta@beta+self.cost@z)
        dual_objective=-float(beta@beta+rhs@lam)
        gap=abs(objective-dual_objective)/(1+abs(objective))
        diag={'primal':primal,'dual':dual,'relative_gap':gap,'objective':objective,
              'solve_seconds':previous_seconds+time.perf_counter()-started,
              'solver':'CLARABEL','fallback_reason':reason}
        if primal>PRIMAL_GATE or dual>DUAL_GATE or gap>GAP_GATE:
            raise RuntimeError(f'CLARABEL KKT gate failure: {diag}')
        return beta.copy(),diag

    def predict(self, beta, x):
        return spline_design(x,self.d,self.ranges)@beta
