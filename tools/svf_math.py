"""Sparse HiGHS implementation of the audited SVF-Spline convex QP.

The formulation matches the v0.2.1-audit Stata contract.  It intentionally
does not copy the authors' Python CV or prediction paths, which leak validation
data and omit later input blocks in the pinned public source.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable

import highspy
import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class SVFSplineDiagnostics:
    status: str
    objective: float
    max_primal_infeasibility: float
    max_dual_infeasibility: float
    max_constraint_violation: float
    solve_seconds: float


@dataclass(frozen=True)
class SVFSplineFit:
    coefficients: np.ndarray
    ranges: np.ndarray
    frontier: np.ndarray
    efficiency: np.ndarray
    n_intervals: int
    c_penalty: float
    epsilon: float
    objective: float
    diagnostics: tuple[SVFSplineDiagnostics, ...]


@dataclass(frozen=True)
class SVFSplineCVFit:
    fit: SVFSplineFit
    cv_results: tuple[dict[str, float], ...]
    cv_predictions: tuple[dict[str, float], ...]
    fold_id: np.ndarray


def _as_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("x and y must be two-dimensional with equal row counts")
    if x.shape[0] < 2 or x.shape[1] < 1 or y.shape[1] < 1:
        raise ValueError("at least two rows, one input, and one output are required")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must be finite")
    if np.any(x < 0):
        raise ValueError("inputs must be nonnegative")
    if np.any(y <= 0):
        raise ValueError("outputs must be strictly positive")
    return x, y


def spline_ranges(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.column_stack((np.min(x, axis=0), np.max(x, axis=0)))


def spline_design(x: np.ndarray, n_intervals: int, ranges: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    ranges = np.asarray(ranges, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if n_intervals < 1:
        raise ValueError("n_intervals must be at least one")
    if ranges.shape != (x.shape[1], 2):
        raise ValueError("ranges must contain one min/max row per input")
    blocks = []
    for j in range(x.shape[1]):
        nodes = np.linspace(ranges[j, 0], ranges[j, 1], n_intervals + 1)
        blocks.append(
            np.column_stack(
                [np.ones(x.shape[0]), x[:, j]]
                + [np.maximum(x[:, j] - node, 0.0) for node in nodes]
            )
        )
    return np.column_stack(blocks)


def _qp_matrices(
    phi: np.ndarray,
    y: np.ndarray,
    n_inputs: int,
    n_intervals: int,
    c_penalty: float,
    epsilon: float,
) -> tuple[sparse.csc_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n, q = phi.shape
    nv = q + n
    inf = highspy.kHighsInf

    col_cost = np.r_[np.zeros(q), np.full(n, c_penalty)]
    col_lower = np.full(nv, -inf)
    col_upper = np.full(nv, inf)
    block = n_intervals + 3
    for j in range(n_inputs):
        off = j * block
        col_lower[off] = 0.0
        col_lower[off + 1] = 0.0
        col_upper[off + 2 : off + block] = 0.0
    col_lower[q:] = 0.0

    # y <= Phi*beta and Phi*beta-xi <= y+epsilon.
    zero = sparse.csr_matrix((n, n))
    eye = sparse.eye(n, format="csr")
    rows = [sparse.hstack((sparse.csr_matrix(phi), zero), format="csr")]
    lower = [y.copy()]
    upper = [np.full(n, inf)]
    rows.append(sparse.hstack((sparse.csr_matrix(phi), -eye), format="csr"))
    lower.append(np.full(n, -inf))
    upper.append(y + epsilon)

    # Cumulative segment slopes, including the linear slope itself.
    shape_rows = []
    for j in range(n_inputs):
        off = j * block
        for stop in range(off + 2, off + block + 1):
            row = sparse.lil_matrix((1, nv))
            row[0, off + 1 : stop] = 1.0
            shape_rows.append(row.tocsr())
    if shape_rows:
        shape = sparse.vstack(shape_rows, format="csr")
        rows.append(shape)
        lower.append(np.zeros(shape.shape[0]))
        upper.append(np.full(shape.shape[0], inf))

    a = sparse.vstack(rows, format="csc")
    a.data[np.abs(a.data) <= 1e-12] = 0.0
    a.eliminate_zeros()
    return a, np.concatenate(lower), np.concatenate(upper), col_cost, col_lower, col_upper


def _pass_sparse_qp(
    a: sparse.csc_matrix,
    row_lower: np.ndarray,
    row_upper: np.ndarray,
    col_cost: np.ndarray,
    col_lower: np.ndarray,
    col_upper: np.ndarray,
    n_quadratic: int,
    threads: int,
) -> highspy.Highs:
    nv = len(col_cost)
    model = highspy.HighsModel()
    lp = model.lp_
    lp.num_col_ = nv
    lp.num_row_ = a.shape[0]
    lp.col_cost_ = col_cost
    lp.col_lower_ = col_lower
    lp.col_upper_ = col_upper
    lp.row_lower_ = row_lower
    lp.row_upper_ = row_upper
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.num_col_ = nv
    lp.a_matrix_.num_row_ = a.shape[0]
    lp.a_matrix_.start_ = a.indptr.astype(np.int32)
    lp.a_matrix_.index_ = a.indices.astype(np.int32)
    lp.a_matrix_.value_ = a.data.astype(float)

    # HiGHS uses 0.5*x'Q*x+c'x; Q=2I on beta gives beta'beta.
    hessian = model.hessian_
    hessian.dim_ = nv
    hessian.format_ = highspy.HessianFormat.kTriangular
    hessian.start_ = np.r_[np.arange(n_quadratic + 1), np.full(nv - n_quadratic, n_quadratic)].astype(np.int32)
    hessian.index_ = np.arange(n_quadratic, dtype=np.int32)
    hessian.value_ = np.full(n_quadratic, 2.0)

    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.setOptionValue("threads", int(threads))
    highs.setOptionValue("random_seed", 0)
    status = highs.passModel(model)
    if status != highspy.HighsStatus.kOk:
        raise RuntimeError(f"HiGHS rejected model: {status}")
    return highs


def _solve_output(
    phi: np.ndarray,
    y: np.ndarray,
    n_inputs: int,
    n_intervals: int,
    c_penalty: float,
    epsilon: float,
    threads: int,
) -> tuple[np.ndarray, np.ndarray, SVFSplineDiagnostics]:
    a, row_lower, row_upper, cost, col_lower, col_upper = _qp_matrices(
        phi, y, n_inputs, n_intervals, c_penalty, epsilon
    )
    highs = _pass_sparse_qp(a, row_lower, row_upper, cost, col_lower, col_upper, phi.shape[1], threads)
    started = time.perf_counter()
    highs.run()
    elapsed = time.perf_counter() - started
    status = highs.modelStatusToString(highs.getModelStatus())
    if highs.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"HiGHS QP status: {status}")
    solution = np.asarray(highs.getSolution().col_value, dtype=float)
    beta = solution[: phi.shape[1]]
    fitted = phi @ beta
    activity = a @ solution
    violation = max(
        float(np.max(np.maximum(row_lower - activity, 0.0))),
        float(np.max(np.maximum(activity - row_upper, 0.0))),
        float(np.max(np.maximum(col_lower - solution, 0.0))),
        float(np.max(np.maximum(solution - col_upper, 0.0))),
    )
    info = highs.getInfo()
    diagnostic = SVFSplineDiagnostics(
        status=status,
        objective=float(info.objective_function_value),
        max_primal_infeasibility=float(info.max_primal_infeasibility),
        max_dual_infeasibility=float(info.max_dual_infeasibility),
        max_constraint_violation=violation,
        solve_seconds=elapsed,
    )
    return beta, fitted, diagnostic


def fit_svfspline_highs(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_intervals: int = 3,
    c_penalty: float = 10.0,
    epsilon: float = 0.0,
    threads: int = 1,
) -> SVFSplineFit:
    x, y = _as_xy(x, y)
    if n_intervals < 1 or c_penalty <= 0 or epsilon < 0 or threads < 1:
        raise ValueError("invalid n_intervals, c_penalty, epsilon, or threads")
    ranges = spline_ranges(x)
    phi = spline_design(x, n_intervals, ranges)
    coefficients = np.empty((phi.shape[1], y.shape[1]))
    frontier = np.empty_like(y)
    diagnostics = []
    for output in range(y.shape[1]):
        coefficients[:, output], frontier[:, output], diagnostic = _solve_output(
            phi, y[:, output], x.shape[1], n_intervals, c_penalty, epsilon, threads
        )
        diagnostics.append(diagnostic)
    efficiency = np.maximum(1.0, np.min(frontier / y, axis=1))
    return SVFSplineFit(
        coefficients=coefficients,
        ranges=ranges,
        frontier=frontier,
        efficiency=efficiency,
        n_intervals=n_intervals,
        c_penalty=c_penalty,
        epsilon=epsilon,
        objective=float(sum(item.objective for item in diagnostics)),
        diagnostics=tuple(diagnostics),
    )


def predict_svfspline(fit: SVFSplineFit, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[1] != fit.ranges.shape[0] or not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError("prediction inputs violate the fitted input contract")
    return spline_design(x, fit.n_intervals, fit.ranges) @ fit.coefficients


def fit_svfspline_cv_highs(
    x: np.ndarray,
    y: np.ndarray,
    *,
    c_grid: Iterable[float],
    epsilon_grid: Iterable[float],
    interval_grid: Iterable[int],
    folds: int = 5,
    seed: int = 12345,
    threads: int = 1,
    tie_tolerance: float = 1e-14,
) -> SVFSplineCVFit:
    x, y = _as_xy(x, y)
    c_values = tuple(float(v) for v in c_grid)
    epsilon_values = tuple(float(v) for v in epsilon_grid)
    interval_values = tuple(int(v) for v in interval_grid)
    if not c_values or not epsilon_values or not interval_values:
        raise ValueError("all CV grids must be nonempty")
    if any(v <= 0 for v in c_values) or any(v < 0 for v in epsilon_values) or any(v < 1 for v in interval_values):
        raise ValueError("invalid CV grid")
    if folds < 2 or folds > x.shape[0]:
        raise ValueError("folds must be between two and the sample size")

    rng = np.random.default_rng(seed)
    order = rng.permutation(x.shape[0])
    fold_id = np.empty(x.shape[0], dtype=int)
    fold_id[order] = np.arange(x.shape[0]) % folds
    results = []
    prediction_records = []
    best = None
    # Deliberately match the audited deterministic d--C--epsilon scan order.
    for d in interval_values:
        for c_penalty in c_values:
            for epsilon in epsilon_values:
                squared_error = 0.0
                count = 0
                solve_seconds = 0.0
                for fold in range(folds):
                    train = fold_id != fold
                    validation = ~train
                    candidate = fit_svfspline_highs(
                        x[train], y[train], n_intervals=d,
                        c_penalty=c_penalty, epsilon=epsilon, threads=threads,
                    )
                    prediction = predict_svfspline(candidate, x[validation])
                    validation_rows = np.flatnonzero(validation)
                    for local_row, observation in enumerate(validation_rows):
                        for output in range(y.shape[1]):
                            prediction_records.append({
                                "C": c_penalty,
                                "epsilon": epsilon,
                                "d": float(d),
                                "fold": float(fold + 1),
                                "observation": float(observation),
                                "output": float(output + 1),
                                "prediction": float(prediction[local_row, output]),
                            })
                    squared_error += float(np.sum((y[validation] - prediction) ** 2))
                    count += int(prediction.size)
                    solve_seconds += sum(item.solve_seconds for item in candidate.diagnostics)
                row = {
                    "C": c_penalty,
                    "epsilon": epsilon,
                    "d": float(d),
                    "mse": squared_error / count,
                    "solve_seconds": solve_seconds,
                }
                results.append(row)
                if best is None or row["mse"] < best["mse"] - tie_tolerance:
                    best = row
    assert best is not None
    final = fit_svfspline_highs(
        x, y, n_intervals=int(best["d"]), c_penalty=best["C"],
        epsilon=best["epsilon"], threads=threads,
    )
    return SVFSplineCVFit(
        fit=final,
        cv_results=tuple(results),
        cv_predictions=tuple(prediction_records),
        fold_id=fold_id,
    )
