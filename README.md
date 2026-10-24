# DGP Sensitivity in Frontier Estimation

Minimal maintenance repository for the current ten-method simulation code and
its frozen design. Repository visibility is currently private; the author will
decide public availability at submission. This is not a public data deposit or a
complete replication release, and no DOI is claimed.

## Scope

- `tools/`: preserved scientific implementation, estimator adapters, metrics,
  aggregation, and monitored runner from the completed production study.
- `design/`: full-precision DGP definitions, model/analysis policies, contrasts,
  and compact deterministic scheduling metadata.
- `environment/`: recorded versions, a local-path template, and the external SVF
  source reference. Software distributions are not bundled.
- `scripts/`: code-inventory checks and preparation of a fresh workspace.

Manuscripts, supplements, reviewer responses, publication tables/figures, saved
results, raw arrays, logs, caches, installed packages, and replication ZIPs remain
local. Historical bootstrap/selector code is outside this initial repository.
The separately preserved replication archive supplies numerical replay and
historical evidence; this repository alone cannot regenerate publication tables.

## Safe local checks (no estimation)

```sh
python3 scripts/verify_code.py
python3 scripts/prepare_workspace.py --output /path/to/a/new/workspace
```

The preparation command regenerates all 27,000 task records and EAT seeds and
checks their exact hashes against the frozen schedules. It copies source/design
into a **new directory outside this checkout**, without copying old approvals,
runtime paths, outcomes, or launching any fits. The schedule includes 117
condition-by-cohort settings (200 repetitions each, with 12 extended to 500).
The 500-repetition windows include the first 200; they are not additional studies.

## Running estimators later

The original runtime needs compatible Python/R packages, solvers, and the
authorized external SVF snapshot. Follow `docs/RUNNING.md` before invoking the
runner. Monitoring currently uses macOS interfaces. Full production also needs
fresh source/environment qualification and resource gates; preparation is not
certification. No automatic simulation, dependency install, or production resume
is configured in this repository.

## Maintenance and release

Scientific source and retained design files initially match the local frozen
production code byte-for-byte; see `SOURCE_PROVENANCE.json`. Generated schedules
are reconstructed instead of committed as multi-megabyte repeated records.
`TRACKED_FILES.txt` and the default-deny `.gitignore` define the intended tracked
inventory. Before adding a file, update that inventory deliberately and run the
verification command. Original results must never be silently overwritten after
an estimator, seed, scoring rule, or numerical policy changes.

See `docs/MAINTENANCE.md` for versioning and `LICENSE` for the existing project
terms. A software license does not imply current public availability. No
manuscript or author-response material should be added here.
