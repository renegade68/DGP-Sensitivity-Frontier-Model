# Execution boundary

Only source verification and workspace preparation are ready-to-run entry points
for this minimal checkout. Scientific routines in `tools/` preserve their original
runtime interface; do not invoke them against the checkout itself.

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

FDH fast-route exclusions and StoNED timeouts are recorded outcomes, not empty
slots to patch selectively. A changed estimator requires a new implementation
version and uniformly applied comparison. Qualified SFA boundaries do not imply
statistical identification or global optimality of every fitted result.
