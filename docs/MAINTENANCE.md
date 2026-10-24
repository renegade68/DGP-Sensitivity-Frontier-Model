# Maintaining the code

Keep source, fixed design, and small configuration metadata in this repository.
Keep manuscripts, reviewer correspondence, outputs, raw data, installed packages,
and machine-specific configuration outside the checkout. The default-deny
[.gitignore](../.gitignore) and [TRACKED_FILES.txt](../TRACKED_FILES.txt) enumerate
the intended files. Update both when adding source, then run:

```sh
python3 -B scripts/verify_code.py
git diff --check
```

[SOURCE_PROVENANCE.json](../SOURCE_PROVENANCE.json) distinguishes byte-preserved
copies from adapted files. Its source paths identify records in the originating
research workspace, not files expected in a clone. Hashes make these relationships
checkable without committing the original results or publication archive.

For a scientific change, record the affected estimator, score, domain, random
stream, or numerical rule and run the relevant independent validation. Update
provenance deliberately; changing a hash alone does not validate a new algorithm.
Keep the original run immutable and use a separately versioned result layer.
Report implementation changes and their effects, including availability and
auxiliary score/rank differences. Do not copy historical readiness or execution
locks to approve a new run.

Record the exact code commit and input-archive identity for each analysis.
Code, saved numerical evidence, and publication documents have separate version
histories. A source-only check does not certify result reproduction or a newly
configured numerical environment.

Some frozen design fields describe pre-run status or refer to historical source
records. They are retained as provenance. All 117 adopted condition-by-cohort
settings are included, including six stress cells with an inherited optional
preflight label. The readable repository overview describes the current scope;
historical identifiers are not rewritten to imply a new experiment.
