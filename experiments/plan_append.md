# Crash-Safe, Resumable Experiment Results

## Summary

- Add transactional checkpoints for `run_imputation.py`, `run_local_feature_importance.py`, `run_outlier_detection.py`, `run_prediction_identity.py`, and `run_prototypes.py`.
- Persist each completed scientific unit immediately from its worker. Existing raw CSVs and compiled tables remain deterministic exports generated after a successful invocation.
- Resume by default. Reuse a checkpoint only when its scientific configuration, prepared data, source code, and relevant environment exactly match the current run.
- Leave `run_mds_visualization.py` outside this change and preserve the current script selection in `run_all.py`.

## Shared persistence and interfaces

- Implement a shared SQLite checkpoint store in `experiments/common.py` at `results/raw/<experiment>/checkpoints.sqlite3`. Use one connection per worker, WAL mode, a busy timeout, schema versioning, and transactional bundle upserts.
- Store a complete manifest, summary rows, optional detail rows/artifacts, statuses, and a completion marker per atomic unit. Only all-`ok`/`unsupported`/`excluded` bundles are reusable; persist errors as retryable attempts.
- Hash the unit dimensions, result-affecting arguments, exact prepared arrays and application inputs, relevant source contents, Python/package versions, and checkpoint schema. Exclude scheduling and selection controls such as worker counts and dataset limits.
- Add `--resume/--no-resume` (resume by default), deterministic per-unit seeding, completion-order future collection, and atomic replacement of canonical CSVs. Fully cached runs must still rebuild outputs for exactly the requested scope.
- Keep existing output names and scientific columns, adding checkpoint schema/key/provenance columns. Preserve cached runtime values as measurements of the original computation.

## Runner changes

- Prediction identity: checkpoint one terminal row per dataset/task, seed, model, and derived proximity.
- Local feature importance: checkpoint per proximity and ensure every terminal row includes `prox_method`.
- Imputation: checkpoint model/proximity/variant/mechanism/rate batches containing both downstream-model rows; protect mask generation and lazily compute oracle results only for pending work.
- Outlier detection: checkpoint each model/fraction/proximity batch containing all score/label rows. Store the canonical first fit's native-score vector as an internal dependency so later proximity batches preserve current native-agreement semantics and emit only one native result row.
- Prototypes: transactionally bundle learned and baseline summaries with all detail rows per model/proximity. Cache rows before stability enrichment and recompute cross-seed stability after cached and new payloads are combined.

## Tests and acceptance

- Test manifest stability/invalidation, concurrent writes, rollback, serialization, idempotent commits, retryable errors, and preservation of a prior valid bundle after a failed forced attempt.
- Simulate interruption after several committed units; restart and verify only pending units run and final scientific results match an uninterrupted run apart from runtimes.
- Verify exact-scope, fully cached CSV regeneration with stable ordering and no duplicates.
- Cover imputation batch completeness/lazy oracles, local-importance proximity identities, outlier native dependency/cardinality, prototype summary/detail atomicity and cross-seed stability, and prediction fit skipping.

## Defaults

- Existing raw CSVs are legacy outputs and are not imported for resume because they lack sufficient configuration, source, environment, and split provenance. Preserve them until a successful fingerprinted export.
- Keep old checkpoint fingerprints for audit but exclude them from current summaries.
- Do not change MDS behavior, experiment metrics, launcher scheduling, or the current `run_all.py` script list.
