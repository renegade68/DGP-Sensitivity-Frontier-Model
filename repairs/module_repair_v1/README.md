# Isolated FDH / StoNED repair workflow

This pipeline operates on the preserved local research workspace. It is not a
standalone simulation or a replacement for the original run. Only code is tracked
here; inputs, predictions, diagnostics, reports, and manuscript files remain local.

Copy this directory into a NEW direct child of the original workspace's `revision/`
directory. Its parent must contain the named frozen production and replication
components used by `prepare.py`. Run the copied scripts there with the recorded
Python/R environment. Never run the copy inside this repository as a results
workspace. Existing output directories are rejected rather than overwritten.

Sequence:

1. `python -B prepare.py` indexes the frozen completed attempts and stages a
   separate copy of their scientific source/runtime references.
2. `Rscript --vanilla export_inputs.R .` recreates the exact R-read CSV inputs and
   records hashes of archived direct FDH checks. No estimator fits are performed.
3. `python -B fdh_recompute.py --pilot` compares two direct algorithms on all
   conditions' first repetitions and the known FDH exceptions.
4. `python -B fdh_recompute.py` applies the same direct FDH rule to every dataset,
   verifies archived direct predictions, and writes isolated metrics/predictions.
5. `python -B retry_stoned.py` retries both original timeouts once, with unchanged
   scientific and solver-internal rules and a common 1800-second outer wall cap.
6. `python -B verify_stoned.py` checks qualification and computes replacement
   metrics. Successful completion does not identify the original timeout cause.
7. `python -B aggregate_repair.py`, then
   `python -B aggregate_repair.py --include-stoned`, creates replacement overlays
   for affected model, method, and factor summaries and the existing Holm families.
8. `python -B analyze_impact.py` inventories changed interval classifications and
   isolates score/rank sensitivity on the original available sample. Its eight
   decimal rounding is a diagnostic only and is not adopted as the rank metric.
9. Run `python -B verify_runtime.py` to verify the recorded runtime files, then
   `python -B verify_preservation.py`.
   This final check rehashes consumed inputs and protected original releases.

The new results are a post-run repair/sensitivity layer. Original numerical
exclusions and timeouts remain in their historical records. A paper must describe
implementation and retry changes explicitly if adopting the new results. Preserve
all differences, including auxiliary rank/equality effects; agreement of main
frontier comparisons does not establish that every statistic is unchanged.
