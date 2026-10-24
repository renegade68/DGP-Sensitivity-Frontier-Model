from context import *
import pandas as pd,highspy,contextlib,io,time,importlib.util
from scipy import sparse
from svf_math import _pass_sparse_qp
from svf_workspace import Workspace as AuditedQP
SVF_SOURCE=ROOT/'vendor/SVF_Package'
spec=importlib.util.spec_from_file_location('svf_package',SVF_SOURCE/'__init__.py',submodule_search_locations=[str(SVF_SOURCE)])
mod=importlib.util.module_from_spec(spec);sys.modules['svf_package']=mod;spec.loader.exec_module(mod)
from svf_package.methods.svf_splines import SVFSplines
class PublicSVFWorkspace(AuditedQP):
    """Model coefficients come from the unmodified authors' train() method."""
    def __init__(self, x, y, d, paper_intercepts=False):
        self.x, self.y, self.d = np.asarray(x), np.asarray(y), int(d)
        columns = [f'x{i+1}' for i in range(x.shape[1])]
        data = pd.DataFrame(x, columns=columns)
        data['y'] = y
        build_output = io.StringIO()
        with contextlib.redirect_stdout(build_output):
            author = SVFSplines('SVF-Splines', columns, ['y'], data, 1., 0., self.d)
            author.train()
        self.source_build_output = build_output.getvalue()
        model = author.model
        self.grid = author.grid
        self.n = len(y)
        variables = list(model.iter_variables())
        self.q = len(variables) - self.n
        assert all(v.name.startswith('w_') for v in variables[:self.q])
        assert all(v.name.startswith('xi_') for v in variables[self.q:])
        self.phi = self.design(x)
        self.ranges = np.column_stack([x.min(axis=0), x.max(axis=0)])
        rr, cc, vv, lo, hi = [], [], [], [], []
        eps_rows = []
        for i, ct in enumerate(model.iter_constraints()):
            coefs = {}
            for sign, expr in ((1, ct.left_expr), (-1, ct.right_expr)):
                for var, coef in expr.iter_terms():
                    coefs[var.index] = coefs.get(var.index, 0.) + sign * float(coef)
            for col, coef in coefs.items():
                if coef:
                    rr.append(i); cc.append(col); vv.append(coef)
            rhs = float(getattr(ct.right_expr, 'constant', 0.) - getattr(ct.left_expr, 'constant', 0.))
            sense = ct.sense.name
            lo.append(rhs if sense in ('GE', 'EQ') else -np.inf)
            hi.append(rhs if sense in ('LE', 'EQ') else np.inf)
            if ct.name.startswith('c1_o'):
                eps_rows.append(i)
        self.a = sparse.csc_matrix((vv, (rr, cc)), shape=(len(lo), len(variables)))
        tiny = np.abs(self.a.data) <= 1e-12
        self.coefficient_cleanup_count = int(np.sum(tiny))
        self.coefficient_cleanup_max = float(np.max(np.abs(self.a.data[tiny]), initial=0.))
        self.a.data[tiny] = 0.
        self.a.eliminate_zeros()
        self.lo, self.hi = np.array(lo), np.array(hi)
        self.base_hi = self.hi.copy()
        self.clo = np.array([v.lb if v.lb > -1e30 else -np.inf for v in variables])
        self.chi = np.array([v.ub if v.ub < 1e30 else np.inf for v in variables])
        if paper_intercepts:
            self.clo[np.arange(x.shape[1]) * (self.d + 3)] = 0.
        self.cost = np.zeros(len(variables))
        for var, coef in model.objective_expr.iter_terms():
            self.cost[var.index] = float(coef)
        diagonal = np.zeros(len(variables))
        for v1, v2, coef in model.objective_expr.iter_quad_triplets():
            assert v1.index == v2.index
            diagonal[v1.index] += float(coef)
        assert np.array_equal(diagonal, np.r_[np.ones(self.q), np.zeros(self.n)])
        assert np.array_equal(self.cost, np.r_[np.zeros(self.q), np.ones(self.n)])
        self.h = _pass_sparse_qp(self.a, self.lo, self.hi, self.cost, self.clo, self.chi, self.q, 1)
        for key, val in [('primal_feasibility_tolerance', 1e-8),
                         ('dual_feasibility_tolerance', 1e-8),
                         ('qp_regularization_value', 1e-9), ('time_limit', 120.)]:
            assert self.h.setOptionValue(key, val) == highspy.HighsStatus.kOk
        self.slack_idx = np.arange(self.q, self.q+self.n, dtype=np.int32)
        self.upper_idx = np.array(eps_rows, dtype=np.int32)
        self.fallback_solver = None
        self.author_lp = model.export_as_lp_string()
        model.end()

    def design(self, x):
        return np.array([np.concatenate(self.grid.calculate_dmu_phi(row)[0]) for row in x])

    def predict(self, beta, x):
        phi=self.design(x)
        # Explicit sum avoids macOS BLAS propagating stale floating-point flags
        # left by external solvers. Check operands and output, never mask NaNs.
        if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(phi)):
            raise FloatingPointError('Nonfinite SVF prediction operands')
        out=np.einsum('ij,j->i',phi,beta,optimize=False)
        if not np.all(np.isfinite(out)): raise FloatingPointError('Nonfinite SVF prediction')
        return out

    def solve_fallback(self, reason, previous_seconds):
        # Public QP presolve may eliminate different rows. A fresh Clarabel
        # instance avoids unsupported updates after presolve; same QP/gates.
        self.fallback_solver = None
        return super().solve_fallback(reason, previous_seconds)

    def solve(self, c, eps):
        # Public rows alternate c1/c2; use their actual positions, not a local
        # formula assuming all c1 constraints follow all c2 constraints.
        self.cost[self.q:] = c
        self.hi[self.upper_idx] = self.base_hi[self.upper_idx] + eps
        assert self.h.changeColsCost(self.n, self.slack_idx, self.cost[self.q:]) == highspy.HighsStatus.kOk
        assert self.h.changeRowsBounds(self.n, self.upper_idx, self.lo[self.upper_idx],
                                       self.hi[self.upper_idx]) == highspy.HighsStatus.kOk
        self.h.clearSolver()
        started = time.perf_counter()
        self.h.run()
        elapsed = time.perf_counter()-started
        status = self.h.modelStatusToString(self.h.getModelStatus())
        if self.h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            return self.solve_fallback(f'HiGHS {status}', elapsed)
        sol = self.h.getSolution()
        z, rd, cd = map(np.asarray, (sol.col_value, sol.row_dual, sol.col_dual))
        activity = self.a @ z
        primal = max(np.max(np.maximum(self.lo-activity, 0)), np.max(np.maximum(activity-self.hi, 0)),
                     np.max(np.maximum(self.clo-z, 0)), np.max(np.maximum(z-self.chi, 0)))
        beta = z[:self.q]
        grad = self.cost.copy(); grad[:self.q] += 2*beta
        dual = float(np.max(abs(grad-self.a.T@rd-cd))/(1+np.max(abs(grad))))
        dual_obj = -float(beta@beta)
        for multipliers, lower, upper in [(rd,self.lo,self.hi),(cd,self.clo,self.chi)]:
            pos,neg=np.maximum(multipliers,0),np.minimum(multipliers,0)
            lf,uf=np.isfinite(lower),np.isfinite(upper)
            dual_obj += float(pos[lf]@lower[lf]+neg[uf]@upper[uf])
            dual=max(dual,float(np.max(pos[~lf],initial=0))/(1+c),float(np.max(-neg[~uf],initial=0))/(1+c))
        objective=float(beta@beta+self.cost@z)
        gap=abs(objective-dual_obj)/(1+abs(objective))
        diag=dict(primal=float(primal),dual=dual,relative_gap=gap,objective=objective,
                  solve_seconds=elapsed,solver='HiGHS')
        if primal>1e-6 or dual>1e-6 or gap>1e-5:
            return self.solve_fallback(f'HiGHS KKT {diag}',elapsed)
        return beta.copy(),diag
