# Reviewer checks without a full production rerun

The full design has 27,000 datasets and ten procedures. Reviewers can inspect
the code and frozen design, reconstruct numerical tables from saved summaries,
and execute a bounded fitting example as separate checks. These activities
provide different evidence; none replaces a fresh full Monte Carlo replication.

## Saved-summary replay

Use the separately supplied replication packet, extract its immutable version-8
numerical base, and follow that archive's `README` and `replay_requirements.txt`:

```sh
python3 -B replay_current.py --output /path/to/new/replay-output
```

This reconstructs 19 numerical table fragments, two numeric macro files and
seven full-precision extracts, and recomputes 2,457 primary tests with their
fixed Holm families. It checks their agreement with the stored references.
It does not reaggregate raw predictions or refit estimators. Figure/PDF rendering
is optional and is not part of this minimal command. The saved inputs and
numerical tables are **not in this source-only repository**.

## Six-dataset fitting example

Requires Python 3.12 or later for the recorded dependency versions, NumPy, SciPy and Clarabel. Recorded versions are
in [requirements.txt](requirements.txt). Install them in a separate environment;
the script does not install packages or need the production `LOCAL.json`.
Obtain the separate `reviewer_check` packet supplied with the review materials.
From this checkout's root:

```sh
python3 -B examples/reviewer_check.py \
  --fixtures /path/to/reviewer_check/fixtures \
  --output /path/to/new/quickcheck-output
```

The output directory must not exist and must be outside the checkout and the
fixture directory. Do not use Python's `-O` option. The packet also includes a
matching source snapshot for offline execution after dependency installation.
The packet is supplied separately; no public download of result data is implied.

The fixed [protocol](PROTOCOL.json) uses the first replication of six settings
chosen by frontier/error features before inspecting the check outcomes. It
regenerates the DGP arrays, freshly optimizes four SFA specifications on each
dataset (24 optimizations), and evaluates FDH directly (six calculations).
It retains interior and boundary solutions and independently checks native SFA
scores by posterior quadrature or boundary identities. Inputs and scientific
source hashes are checked. Original adapters and likelihoods are not rewritten.

SFA uses **archived raw sfaR baseline parameter vectors as starting candidates**;
it does not repeat the original R baseline fits. FDH uses archived R-read doubles
to preserve the production CSV reader's rounding and compares against the
archived direct-result hash and a sorted reference calculation. Latent truth is
used only to check the generator and calculate post-estimation losses. The
losses are diagnostic outputs, not a new Monte Carlo comparison.

`SUMMARY.json` records every case, status, discrepancy, environment and elapsed
time. A nonzero exit and `REVIEW_REQUIRED` indicate a discrepancy or failed
qualification. Retain the report; do not substitute cases or loosen tolerances.
The predeclared tolerances are 1e-7 for log likelihood per observation, 1e-5
for relative frontier predictions, and 1e-6 for absolute native scores. Numerical
libraries and platforms may affect agreement; these bounds do not imply global
optimality. The example does not refit EAT, CEAT, SVF, DEA or StoNED and does not
validate the full study's rankings. Full rerun requirements remain in
[Running the code](../docs/RUNNING.md).
