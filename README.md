# DGP Sensitivity in Frontier Estimation

Simulation code for studying how data-generating processes (DGPs) affect
production-frontier estimation and technical-efficiency measurement. The design
compares methods under specified frontier shapes, noise and inefficiency
distributions, sample sizes, input dimensions, and stress conditions. It examines
when conclusions from particular simulation settings persist or change; it is
not intended to identify a universally best estimator.

## Methods and design

The ten specifications are EAT, CEAT, SVF-Splines, FDH, VRS DEA, four SFA
specifications (Cobb–Douglas or translog, with half-normal or exponential
inefficiency), and StoNED-QLE. Package versions and disclosed implementation
variants are recorded in the [method register](design/MODELS.json). The SFA
routines use a stable likelihood/optimization adapter; they are not simply
default calls to the comparison packages.

The [frozen design](design/EFFECTIVE_DGP.json) contains 117 condition-by-cohort
settings and 27,000 simulated datasets. Each setting has 200 replications;
12 settings extend to 500, including their first 200 replications.
Primary frontier NMSE is evaluated on independent points within the common
evaluation domain. Efficiency levels and within-dataset rankings are evaluated
against latent truth on training observations. Common output/frontier ratios and
method-native efficiency scores are kept distinct, as are BC and JLMS scores.
See the [analysis specification](design/ANALYSIS.json) and
[paired contrasts](design/CONTRASTS.json).

Availability, numerical exclusions, boundary solutions, and Monte Carlo
uncertainty accompany comparisons. Agreement with another implementation is a
diagnostic check; it does not establish identification, global optimality, or
equivalent predictions in every case.

## Repository contents

| Directory | Contents |
| --- | --- |
| [tools/](tools/) | DGP generator, estimator adapters, scoring, aggregation, and original monitored runner |
| [design/](design/) | Frozen DGPs, model policies, contrasts, and deterministic schedule metadata |
| [environment/](environment/) | Recorded dependency versions, local-path template, and external SVF snapshot reference |
| [scripts/](scripts/) | Source verification and preparation of a fresh workspace |
| [validation/](validation/README.md) | Independent SFA likelihood, boundary, derivative, prediction, and score comparisons |
| [repairs/](repairs/module_repair_v1/README.md) | Isolated uniform FDH recomputation and controlled StoNED timeout retries |
| [analysis/](analysis/README.md) | Repair-aware aggregate merging and recalculation of fixed Holm families |
| [examples/](examples/README.md) | Bounded reviewer fitting check and saved-summary replay instructions; requires a separate input packet |

Only source and small design/environment metadata are included. Manuscripts,
reviewer correspondence, fitted results, raw simulation arrays, publication
tables and figures, logs, and installed third-party software are not included.

## Start here

From the repository root, using Python 3:

```sh
python3 -B scripts/verify_code.py
python3 -B scripts/prepare_workspace.py --output /path/to/a/new/workspace
```

These commands require only the Python standard library. Verification checks the
file inventory, recorded source hashes, Python syntax, and frozen schedules.
Preparation copies the production source/design to a new directory outside this
checkout and regenerates all task records and EAT seeds with exact hash checks.
Neither command installs software or runs an estimator.

For execution requirements, read [Running the code](docs/RUNNING.md). Full
simulation requires compatible Python/R packages, solvers, the external SVF
snapshot, and fresh runtime qualification. The original monitoring interface is
macOS-specific, and this checkout does not automate every production qualification
step. Validation and repair scripts require saved run inputs. Numerical replay
requires saved aggregate inputs; the repository alone cannot reproduce the
study's result tables.

For a review that does not require rerunning the entire experiment, see
[Reviewer checks](examples/README.md). The separate review packet supports
saved-summary reconstruction and a fixed six-dataset FDH/SFA example. The latter
performs 24 fresh SFA optimizations using archived baseline starting values and
six direct FDH calculations. Its scope and tolerances are explicit; it is not
a new Monte Carlo ranking or an all-method replication.

## Versions and attribution

[Source provenance](SOURCE_PROVENANCE.json) records preserved files and explicit
adaptations. Production code and post-run repairs remain separate: adopting a
repair does not rewrite the original execution record. See the
[maintenance policy](docs/MAINTENANCE.md) for versioning and file scope.

When referring to this code, identify Choonjoo Lee, this repository's title and
URL, and the exact commit used. Author-written software is covered by the MIT
License; project-generated documentation and design records use CC BY 4.0.
See [LICENSE](LICENSE) for the full terms. Third-party packages retain their own
licenses.
