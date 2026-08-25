# Experiments

This directory contains reproducible benchmarks for `forestgap` proximity
relations. The current study protocol is
[`notes/experimental_designs.md`](notes/experimental_designs.md); the
historical requests in `experiments/experiments_prompt.md` are background, not
the governing specification. Application results are written to
`results/raw/<experiment>/` and `results/<experiment>/<task-kind>/` unless a
runner exposes an explicit output argument.

## Runnable experiments

| Runner | What it evaluates | Main output |
| --- | --- | --- |
| `run_method_matrix.py` | A fast, local synthetic matrix over estimator family, relation, and application. It measures kNN prediction accuracy, proximity clustering ARI, and retrieval quality across axis-aligned and oblique regimes. Use this for capability and integration smoke tests. | `results/raw/method_matrix.jsonl` by default, or `--output` |
| `run_protocol_matrix.py` | The official-OpenML-fold supervised baseline used by the 1.1 protocol. Random forest, Extra Trees, and gradient boosting parents are compared through the declared prediction relations on official train/test folds. Classification reports balanced accuracy and log loss; regression reports RMSE, MAE, and R². | `results/raw/protocol_matrix.jsonl` by default, or `--output` |
| `run_prediction_identity.py` | Whether an RF-GAP proximity prediction reproduces the fitted forest's ordinary and OOB predictions. It reports probability/prediction differences, agreement, valid OOB coverage, accuracy or R², and runtimes. | `results/raw/prediction_identity/{classification,regression}_raw.csv` and comparison tables/plots |
| `run_imputation.py` | Proximity imputation under MCAR, MAR, and MNAR masking at configurable rates. Each relation runs single-pass and iterative variants, scores masked numeric/categorical cells, then measures downstream classification/regression against an uncorrupted oracle. | `results/raw/imputation/` and per-task rate tables/plots |
| `run_outlier_detection.py` | Detection of synthetically corrupted training observations with supervised forests and unsupervised anomaly models. It compares native and proximity-based scores using ROC AUC, average precision, precision/recall/F1@k, rank correlation, and top-k overlap. | `results/raw/outlier_detection/` and per-task score summaries/plots |
| `run_prototypes.py` | Prototype selection on both designed multimodal classification scenarios and OpenML classification data. It measures prototype accuracy, coverage, class-neighbour purity, centrality, mode coverage, random-center baselines, and cross-seed stability. | `results/raw/prototypes/` plus summaries, details, baseline comparisons, and plots |
| `run_local_feature_importance.py` | Local feature importance for classification. It compares scores based on predicted versus true labels, checks the probability reduction after perturbing the selected top feature versus a random feature, and records runtime and misclassification rate. | `results/raw/local_feature_importance/` and per-task summary/plot |
| `run_mds_visualization.py` | MDS visualizations of sampled training-set proximity geometry for each supported model/relation. It is a qualitative diagnostic and saves one PNG per successful configuration. | `results/raw/mds_visualization/` and `results/mds_visualization/<task-kind>/` |
| `run_all.py` | Sequential launcher that records child-process logs and a runtime/status CSV. Its current `SCRIPTS` configuration runs `run_prototypes.py`; the other application runners remain available individually. | `results/raw/run_all_logs*/` and `results/raw/run_all_process_runtimes*.csv` |

`--help` on any runner lists its parameters. `run_method_matrix.py` and
`run_protocol_matrix.py` have `--smoke` modes; the application runners can be
kept small with `--dataset-limit`, `--random-states`, `--max-rows`, and
`--workers`.

## Shared experiment infrastructure

- `common.py` defines the OpenML suite access, preprocessing, model registry,
  missingness and corruption generators, metrics, plotting/export helpers,
  worker scheduler, and crash-safe SQLite checkpoint store.
- `harness.py` executes declared estimator × relation × application cells and
  records capabilities, manifests, timings, memory, unsupported relations, and
  per-cell errors.
- `program.py` declares capability-aware protocol cells, frozen trial manifests,
  and relation-use policy.
- `analysis.py` aggregates successful outer folds within datasets and provides
  paired effects, bootstrap intervals, rankings, Friedman tests, and Holm
  adjustment.
- `__init__.py` marks the package. `experiments_prompt.md` and `plan_append.md`
  are historical/planning notes rather than runners.

## Failure and resume behavior

The checkpoint-enabled runners (`run_prediction_identity.py`,
`run_imputation.py`, `run_outlier_detection.py`,
`run_local_feature_importance.py`, and `run_prototypes.py`) store terminal
units in `results/raw/<experiment>/checkpoints.sqlite3` and resume compatible
completed units by default; pass `--no-resume` to recompute them. Application
runners record expected data/model failures as `status=error` rows with an
exception type, message, and traceback; harness-based runners record the
status, reason, and error type. Unsupported models and relations are recorded
as `unsupported` rather than silently dropped.

The shared parallel scheduler permits only one active setup per OpenML dataset.
If the operating system terminates a worker, it requeues the unfinished setup
groups and retries them with fewer workers, while retaining already committed
checkpoint results. Empty or fully filtered result sets now compile cleanly,
so an unsuccessful slice does not turn into a secondary export crash.
