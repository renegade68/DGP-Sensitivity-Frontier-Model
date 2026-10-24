# Execution boundary

Source verification and workspace preparation run from a clone without saved
results or third-party dependencies. [Numerical replay](../analysis/README.md)
also has a command-line entry point, but needs saved aggregate inputs and SciPy.
Scientific routines in `tools/` preserve their original runtime interface; do not
invoke them against the checkout itself.

1. Run `scripts/prepare_workspace.py --output NEW_DIRECTORY` from this checkout.
2. Reconstruct a compatible recorded environment using
   `environment/RECORDED_VERSIONS.json` and the Python version list. These are
   historical records, not a tested cross-platform installer. Paths in the local
   template must be configured on the target computer.
3. Obtain the external SVF source by an authorized route and verify its snapshot
   using `environment/SVF_EXTERNAL_SOURCE.json`. No SVF source is redistributed:
   the frozen record did not contain a license file. FEAR is not needed for the
   current ten-method code and is not bundled; it belongs to historical work.
4. Establish fresh runtime file hashes, interpreter checks, and scientific
   qualification before any example/smoke fit. The workspace intentionally has
   no active `environment/LOCAL.json`, installed-file manifest, or readiness lock.
   Source-only preparation must fail closed rather than imply a runnable system.
5. Full production additionally requires fresh readiness, source-lock,
   first-batch, and resource-budget records. Do not reuse the old run's approvals.

The runner checks these inputs before production. The historical qualification
controllers and their raw fixtures are local and outside this minimal repository;
this checkout does not automate all qualification steps. The full local
replication archive has the more extensive preparation and validation material.
`tools/first_batch.py`, `finalize.py`, and `aggregate.py` consume actual run records;
they do not supply missing records or certify a fresh environment on their own.

The [repair workflow](../repairs/module_repair_v1/README.md) applies direct FDH
evaluation uniformly and retries the two StoNED timeouts with a disclosed outer
time cap. Original exclusions and timeouts remain recorded; repairs are a separate
layer. The [SFA diagnostics](../validation/README.md) compare likelihoods,
derivatives, boundary subproblems, predictions, and scores with independent
calculations and package results. They require saved inputs and a separately
staged workspace. Qualified boundaries do not imply statistical identification
or global optimality of every fitted result.
