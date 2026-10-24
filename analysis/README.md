# Replay repair-aware aggregates

[merge_repair.py](merge_repair.py) combines original aggregate extracts with the
separate FDH/StoNED repair layer. It checks original means, Monte Carlo standard
errors, and intervals against each overlay's `before` fields; merges by complete
scientific keys; retains original availability alongside derived availability;
and recomputes the fixed 2,457-comparison and 756-comparison Holm families.
This is the frozen study's replay, not a general-purpose merger for other designs.

The numerical merge and multiplicity calculations are preserved from the
repair-aware replay. Unused document-writing helpers were removed and explicit
paths, output isolation, and input-hash checks were added. No manuscript text or
table-generation templates are included.

Required saved inputs (not distributed here):

```text
INPUTS/
  original_data/
    MODEL_METRICS.csv
    METHOD_CONTRASTS.csv
    FACTOR_CONTRASTS.csv
    AVAILABILITY_ALL_PLANNED.csv
  repair/
    ORIGINAL_UNAVAILABLE.json
    reports/combined/
      MODEL_SUMMARIES.jsonl
      METHOD_CONTRASTS.jsonl
      FACTOR_CONTRASTS.jsonl
      FRONTIER_PRIMARY_TESTS.json
      MULTIPLICITY.json
```

With SciPy available, run from the repository root:

```sh
python3 -B analysis/merge_repair.py --inputs /path/to/INPUTS --output /path/to/NEW_OUTPUT
```

Use the original full-precision extracts and validated combined repair reports.
The program checks the study's expected counts and consistency with the repair
diagnostics. It does not independently revalidate the underlying fits.
The destination must be new, outside both the inputs and this checkout, with an
existing parent. Do not use Python's `-O` option: scientific assertions are required.

The output contains merged CSVs and copied repair overlays in `data/`,
`REPAIR_MERGE_CHECK.json`, and `INPUT_HASHES.json`. Inputs are rehashed at the end;
no estimator is fitted. A failed run may leave an incomplete output directory;
only successful completion with both check files is a completed replay.
