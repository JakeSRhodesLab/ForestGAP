# `forestgap` 1.1 Experiment Protocol

**Protocol version:** `forestgap-experiment-program/1.1`
**Status:** normative for new benchmark results; legacy exploratory notebooks and
pre-1.1 result files are not evidence for this protocol.

This document defines the executable study program for `forestgap` 1.1. A
result is in scope only when its declared estimator, relation, data split,
tuning trial, preprocessing, and status are captured in a manifest row. The
shared harness is the source of truth for those rows. A failed, unsupported, or
excluded requested comparison is retained as data; it is never silently
dropped.

## Scope and capability matrix

Cells are generated from `capabilities_for(fitted_estimator)`, not from a
legacy hard-coded relation list. The static registry remains the roadmap, while
the fitted capability snapshot is the executable contract. Do not schedule
gated or unregistered families. Survival is explicitly deferred in this phase.

| Family | Task scope | Relations eligible after the runtime contract | Study status |
| --- | --- | --- | --- |
| RF, ExtraTrees, RotationForest, treeple oblique/patch/extra-oblique, CCF/CCF-Bag | classification, regression | `original`, `oob`, `rfgap`, `kerf`, `path`, `englund`, `random_height` | executable when installed and bootstrap/OOB metadata satisfy the relation contract |
| HonestForest classifier | classification | `original`, `honest_gap`, `kerf`, `path`, `englund`, `random_height` | executable when honesty-role indices are available |
| GradientBoosting, XGBoost, LightGBM, CatBoost | classification, regression | `original`, `kerf`, `boosted`, `path`, `englund`, `random_height` | executable; CatBoost path-derived relations require `grow_policy="SymmetricTree"` |
| Isolation and extended-isolation forest | anomaly | `collision` | executable; native path scores remain separate comparators |
| Synthetic-null forest | anomaly, geometry | `original`, `oob`, `kerf`, `path`, `englund`, `random_height` | executable; RF-GAP is intentionally absent |
| Random survival forest | survival | `survival_leaf` | deferred: no benchmark runner, dataset dependency, or result claim |
| RF-AE / autoencoding, causal, generalized, Bayesian, structured, multiview, graph, and other gated families | varies | none | excluded until their adapter and data-specific protocol exist |

`oob` and `rfgap` require the fitted bootstrap, in-bag, and OOB contracts.
Boosting settings never create those contracts. Cells requested for a relation
outside a fitted snapshot are recorded as `unsupported`; deliberately deferred,
gated, or unregistered cells are recorded as `excluded` without fitting the
factory.

### Directionality

`rfgap` and `honest_gap` are directed query-to-training weights. Use them
directly for prediction, outlier scoring, imputation donors, retrieval, active
acquisition, and attribution.
Symmetrize a directed relation only in a declared geometry application
(embedding, clustering, or graph diagnostic), and record that policy in the
manifest. `original`, `oob`, `kerf`, `boosted`, `path`, `englund`,
`random_height`, and `collision` are symmetric relations under their documented
contracts. Prediction identity is evaluated only for RF-GAP and Honest-GAP.

### Application × model × relation declaration

The family matrix above says what a fitted model can expose. This table says
which of those fitted capabilities each proximity-based application requests.
“All fitted relations” means exactly the runtime `capabilities_for` snapshot
for the listed family; it does not mean a legacy, hard-coded relation list.
Thus unavailable optional backends and missing bootstrap/OOB/honesty metadata
produce retained `unsupported` rows rather than a substituted relation.

| Application / runner | Models included | Relations requested and consumption rule |
| --- | --- | --- |
| Kernel/kNN prediction and retrieval (`run_protocol_matrix.py`, `run_method_matrix.py`) | each executable family exposed by the capability-aware runner (bagged, HonestForest, and boosted as installed) | all fitted relations; `rfgap` and `honest_gap` are used directly as query-to-training weights, all other relations as their symmetric weights |
| Prediction identity (`run_prediction_identity.py`) | RF-GAP-capable bagged families and HonestForest | `rfgap` for bagged families; `honest_gap` for HonestForest; no other relation is an identity comparison |
| Imputation (`run_imputation.py`) | executable bagged and HonestForest families in the legacy runner; all executable families in capability-aware studies where donor prediction is supported | all fitted relations for one-pass and iterative donors; directed relations remain directed donor weights |
| Local feature importance and feature selection | each supervised family supported by the runner | all fitted relations; perturbation recomputes the same selected relation against the fixed training reference |
| Prototypes and retrieval (`run_prototypes.py`) | each supervised classification family supported by the runner | all fitted relations; any directed matrix is symmetrized explicitly for center-to-center geometry, while query retrieval stays directed |
| Visualization, embedding, and clustering (`run_mds_visualization.py`, `run_method_matrix.py`) | each executable family supported by the runner; the capability-aware matrix is the cross-family coverage runner | all fitted relations; `rfgap` and `honest_gap` are symmetrized with the recorded geometry policy before affinity/distance conversion |
| Anomaly detection (`run_outlier_detection.py`) | isolation/extended-isolation forests and synthetic-null forests; supervised forests only as declared comparators | isolation families: `collision`; synthetic-null: its fitted symmetric relations; bagged supervised comparators: `original`, `oob`, `rfgap`, `kerf`, `path`, `englund`, and `random_height`; HonestForest comparators: `original`, `honest_gap`, `kerf`, `path`, `englund`, and `random_height`. RF-GAP and Honest-GAP are included for their respectively capable forests. |
| Active learning | executable supervised predictive families | all fitted relations, with directed relations used directly for predictive/acquisition weights |
| Training-instance attribution | RF-GAP-capable bagged families and HonestForest | `rfgap` and `honest_gap`, respectively; no symmetric relation is treated as a directed attribution weight |

The capability-aware runners enforce these scopes through
`experiments.program.APPLICATION_RELATION_SCOPES`. Legacy runners retain their
model-specific candidate rows so that unavailable optional models and invalid
runtime contracts remain visible in their outputs.

## Dataset and split manifests

The package exposes typed `OpenMLTask` and `OpenMLTaskFold` objects. These hold
the task target, repeat/fold/sample dimensions, official train/test indices,
and a stable `r{repeat}-f{fold}-s{sample}` fold identifier.

| Study family | Dataset source | Outer evaluation protocol |
| --- | --- | --- |
| Supervised, imputation, explanations, selection, prototypes, retrieval, active learning, attribution | OpenML-CC18 classification study 99 and OpenML study 455 regression | every eligible task and every official OpenML repeat/fold/sample; never regenerate an outer `train_test_split` |
| Geometry and clustering | eligible ClustBench v1.1 datasets plus controlled simulations | label-free fitting/tuning; labels are outer evaluation only |
| Anomaly | all 57 eligible ADBench datasets plus controlled simulations | labels are outer scoring only for label-free runs |

The prior regression suite, OpenML study 353 / CTR23, is not part of this
protocol. Existing generic `load_openml_suite` support is retained for backwards
compatibility; study runners use typed task/fold access instead.

Every cell records at least: protocol version; suite, task, dataset, repeat,
fold, sample, and fold IDs; split and artifact hashes; parent and relation
trials; preprocessing/calibration roles; runtime capability snapshot; relation
settings and directionality policy; configuration; metrics; dependency
versions; code revision; fit/relation/application timing; peak memory; and a
terminal `ok`, `unsupported`, `excluded`, or `error` status with a reason when
not `ok`.

## Study portfolio

| Track | Required comparisons | Primary outcomes |
| --- | --- | --- |
| General prediction benchmarking | native parent, proximity kNN/kernel prediction, and identity where valid | balanced accuracy and log loss for classification; RMSE, MAE, and R² for regression |
| Imputation | median/mode, kNN, iterative imputation, one-pass and iterative proximity donors at MCAR/MAR/MNAR 5/10/25/50/75% | normalized RMSE/MAE, categorical accuracy, downstream retention |
| Explanations and selection | local importance versus permutation, SHAP, and LIME; feature selection versus mutual information and embedded baselines | deletion/insertion fidelity, synthetic truth recovery, stability |
| Prototypes and retrieval | learned prototypes and non-proximity comparators | prototype coverage/purity/stability; Recall@k, mAP, nDCG, latency |
| Geometry and clustering | RF-PHATE/Isomap/UMAP-style embeddings; spectral, HDBSCAN, and kernel-PCA clustering | ARI/NMI/AMI, stability, trustworthiness, continuity, neighborhood preservation |
| Anomaly detection | native isolation scores, collision relation, synthetic-null forests, proximity outlier score, LOF, and kNN distance | ROC-AUC, PR-AUC/AP, Precision@contamination, runtime, stability |
| Active learning and attribution | random, uncertainty, core-set, and proximity-diverse acquisition; directed predictive attribution only | learning curves from 10% to 50% labels in 5% batches; deletion/retraining fidelity |

Cluster count, embedding hyperparameters, and all geometry tuning use only
training covariates, graph/reconstruction diagnostics, and stability. Labels
may colour plots or score the held-out result, but may not choose a label-free
configuration. The method-matrix smoke runner uses a data-only eigengap
criterion and records the selected count.

## Tuning and refitting

For every application × outer fold:

1. Freeze a seeded 50-trial family-specific parent manifest and a seeded
   25-trial relation/application manifest before fitting. The provided helpers
   use Sobol or Latin-hypercube draws and persist the exact values.
2. Use five inner folds that are subsets of the outer-training indices. Select
   the parent independently for each application.
3. Refit that selected parent once on the application outer-training partition
   and reuse that refit across eligible relation comparisons only.
4. Fit calibration roles, encoders, scalers, feature selection, missingness
   generators, view construction, and anchors inside the outer-training
   partition. The CatBoost-native categorical branch is declared separately;
   it must not receive generic ordinal preprocessing.

The required search spaces are:

| Family or component | Search space |
| --- | --- |
| Bagged forests | 256–1,024 trees; depth `{None, 8, 16, 32}`; leaf size `{1, 2, 5, 10, 20}`; valid feature/sample fractions; enforce bootstrap for OOB/RF-GAP candidates |
| Boosters | 100–1,000 stages; learning rate 0.01–0.3; depth/leaves; minimum child/leaf size; row/column sampling; CatBoost symmetric trees for path studies |
| Relation/application settings | Englund weight `{0.25, 0.5, 1, 2, 4}`; neighbours `{3, 5, 10, 20, 40}`; kernel regularization; imputation iterations `{1, 3, 5, 10}`; label-free geometry grids |

## Leakage controls

- Official outer test rows are never used in preprocessing, fitting, tuning,
  missingness generation, calibration, anchor selection, or model selection.
- Inner train/validation partitions and calibration partitions are disjoint and
  are expressed as original outer-training row IDs in the manifest.
- Missingness is induced only after the outer split; its mechanism and seed are
  recorded.
- Label-free tracks never inspect labels until outer evaluation. Label-informed
  geometry is a separately labelled analysis, never mixed with label-free
  results.
- Directed predictive relations are not silently symmetrized. Geometry
  symmetrization is explicit and recorded.

## Analysis and acceptance criteria

Aggregate outer folds within each dataset before cross-dataset comparisons; do
not pool raw fold scores. Report paired effects, bootstrap confidence intervals,
average ranks, an omnibus test, and Holm-adjusted paired comparisons. Report
runtime and peak memory alongside predictive or structural metrics.

A study is publishable under this protocol only when it has:

- one complete manifest row per requested cell, including clean optional-dependency skips;
- deterministic smoke coverage for every installed family/relation/application contract;
- synthetic checks for RF-GAP/Honest-GAP identity, imputation recovery, and
  feature-truth and cluster/anomaly recovery;
- tests for official-fold loading, runtime capability generation, status
  preservation, manifest completeness, inner/outer isolation, calibration,
  directionality and label-free tuning; and
- an explicit statement of any deferred survival, RF-AE, single-cell, gated,
  or unavailable optional-dependency work.

This protocol incorporates the prior request for general prediction
benchmarking while replacing the previous exploratory outline.
