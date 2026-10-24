# Maintenance policy

Keep this repository for source, fixed design, and small configuration metadata.
Keep manuscripts, reviewer correspondence, outputs, large data, and local secrets
outside the checkout. `.gitignore` allows only the current explicit file list;
forcing a Git add can bypass it, so run `scripts/verify_code.py` before committing.

For scientific changes, record the affected estimator, score, domain, random
stream, or numerical rule in the commit. Recheck the relevant independent
validation. A new code version is not a retroactive correction to frozen results.
Never copy historical READY or EXECUTION_LOCK files to approve a new run.

The repository, frozen replication archive, and manuscript are separate versions.
At submission record the exact source commit used, the applicable replication
archive hash, and the manuscript version. Decide repository visibility then;
do not assert public availability or a DOI before either actually exists.
Code-only access does not supply result data or reproduce paper tables by itself.

The initial scientific files are preserved copies. Some fixed design fields name
historical source records or describe pre-run status; those fields are provenance,
not claims that the completed production run is pending. The explicit scope
contains all 117 adopted condition-by-cohort settings, including six stress cells
whose inherited optional-preflight label is historical.
