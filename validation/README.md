# Independent SFA diagnostics

These source-preserved scripts document the checks applied to a fixed sample of
24 saved datasets: 12 named DGP settings at replications 1 and 100, with four SFA
specifications each. This diagnostic sample is not a random sample of the full
experiment. The scripts retain optimizer return codes, numerical failures, and
likelihood-matched and unmatched cases.

| Code | Purpose |
| --- | --- |
| [validate.py](sfa_independent_validation_20261024/validate.py) | Independent likelihood and complex-step derivatives; multistart interior searches and boundary paths |
| [check_gradients.py](sfa_independent_validation_20261024/check_gradients.py) | Compare production derivatives with the independent calculation |
| [check_boundary.py](sfa_independent_validation_20261024/check_boundary.py) | Solve boundary subproblems from independent starts and inspect KKT residuals |
| [compare_R.R](sfa_independent_validation_20261024/compare_R.R) | Compare with sfaR (HN and EXP) and frontier (HN), using three starts per case |
| [extract_R.R](sfa_independent_validation_20261024/extract_R.R) and [summarize.py](sfa_independent_validation_20261024/summarize.py) | Retain full-precision R parameters, independently evaluate likelihoods, and summarize sensitivity |
| [compare.py](sfa_prediction_validation_20261024/compare.py) | Compare frontier predictions, BC/JLMS scores, and within-dataset rankings at saved package parameters |
| [native_scores.R](sfa_prediction_validation_20261024/native_scores.R) and [summarize.py](sfa_prediction_validation_20261024/summarize.py) | Separately evaluate native sfaR scores and summarize prediction/score differences |

Close likelihoods need not imply identical coefficients, predictions, or rankings.
Finite multistart searches do not prove global optimality. Constant or nearly
constant efficiency scores require explicit treatment of undefined ranks.
These diagnostics do not replace production fits or choose models using truth.

## Required workspace

The original scripts retain their research-workspace paths and write into their
own directories. **Do not execute them in this checkout or an existing evidence
directory.** Copy the two source folders into a new, separate validation workspace
with this layout, supplying read-only access to the saved production tree and the
recorded environment:

```text
WORKSPACE/
  frontier_production_v4_20261012/      # saved production tools and runs/production001
  efficiency_benchmark_20261007/
    environment/python_packages/     # recorded Python package overlay
    environment/R-library/           # recorded R package library
  sfa_independent_validation_20261024/ # fresh copy of the six scripts
  sfa_prediction_validation_20261024/  # fresh copy of the three scripts
```

Saved inputs include completion manifests, observed arrays, exact training CSVs,
SFA statuses, candidates, predictions, and the `sfaR_baseline/*.rds` objects used
for native-score metadata. These inputs and installed packages are not included.
Use compatible Python/R interpreters and the
[recorded versions](../environment/RECORDED_VERSIONS.json); do not use Python `-O`.

## Sequence after staging

From `WORKSPACE/sfa_independent_validation_20261024`:

```sh
python3 -B validate.py prepare
python3 -B validate.py run
python3 -B check_gradients.py
python3 -B check_boundary.py
Rscript --vanilla compare_R.R .
Rscript --vanilla extract_R.R .
python3 -B summarize.py
python3 -B /path/to/checkout/validation/seal_validation.py .
```

`validate.py run`, `check_boundary.py`, and `compare_R.R` perform optimization;
they are not lightweight source checks. The final command creates the integrity
manifest consumed by the prediction stage. It replaces the manifest-writing
portion of the local report workflow, checks recorded inputs, and refuses to
overwrite an existing manifest. It does not certify numerical success.

Then, from `WORKSPACE/sfa_prediction_validation_20261024`:

```sh
python3 -B compare.py prepare
python3 -B compare.py run
Rscript --vanilla native_scores.R .
python3 -B summarize.py
```

This second stage uses saved parameter vectors without refitting. Reports,
arrays, and extracted third-party function text stay in the separate workspace.
The scripts and manifest helper are included for inspection and controlled
reproduction with saved inputs; a clone alone cannot execute this sequence.
