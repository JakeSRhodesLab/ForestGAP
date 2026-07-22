# Capability-Driven `forestgap` 1.1 Experiment Program

## Summary

- Make [`experimental_designs.md`](/yunity/rhodes8/ObliqueProximities/ObliqueProximitiesCode/notes/experimental_designs.md) the normative protocol: versioned scope, capability matrix, dataset manifests, search spaces, leakage rules, metrics, and acceptance criteria.
- Target the checked-out `forestgap` 1.1 API. Generate cells from runtime `capabilities_for(fitted_estimator)`, not legacy hard-coded lists.
- Cover every registry-supported family with an applicable relation except survival, which is explicitly deferred. Retain `unsupported`/`excluded` rows; do not schedule gated or unregistered families.

## Capability Matrix and Infrastructure

- Freeze the model × relation matrix in the design note:
  - Bagged C/R families—RF, ExtraTrees, RotationForest, treeple oblique/patch/extra-oblique, CCF/CCF-Bag: `original`, `oob`, `rfgap`, `kerf`, `path`, `englund`, `random_height`, subject to runtime bootstrap/OOB contracts.
  - HonestForest classification: `original`, `honest_gap`, `kerf`, `path`, `englund`, `random_height`.
  - GradientBoosting, XGBoost, LightGBM, CatBoost: `original`, `kerf`, `boosted`, `path`, `englund`, `random_height`; restrict CatBoost path relations to symmetric trees.
  - Quantile forest: `conditional`, `original`, `kerf`, `path`, `englund`, `random_height`; isolation families: `collision`; synthetic-null forest: `original`, `oob`, `kerf`, `path`, `englund`, `random_height`.
  - Random survival forest / `survival_leaf`: documented as deferred, with no benchmark runner, dataset dependency, or result claim in this phase.
- Record directionality: use `rfgap`, `honest_gap`, and `conditional` directly for predictive weights and attribution; symmetrize only for geometry applications. Prediction identity is limited to RF-GAP and Honest-GAP.
- Make the application-level selection unambiguous (and mirror it in the normative protocol): prediction, imputation, local importance/selection, prototypes/retrieval, active learning, and geometry request every relation in the fitted capability snapshot; geometry alone symmetrizes directed relations. Quantiles/conformal use quantile-forest `conditional`; attribution uses only directed `rfgap`/`honest_gap`/`conditional`; identity uses only `rfgap` or `honest_gap`; anomaly uses `collision` for isolation and fitted symmetric relations for synthetic-null/supervised comparator forests. The legacy runners retain the corresponding model-specific candidate rows, while `experiments.program.APPLICATION_RELATION_SCOPES` is the executable filter for capability-aware runners.
- Extend `forestgap.datasets` with typed OpenML task/fold access while retaining its suite loaders. Replace legacy regression suite 353 with requested study 455; use all eligible CC18 and study-455 tasks and their official folds ([OpenML benchmark documentation](https://docs.openml.org/benchmark/)).
- Centralize capability-aware cell generation on the shared harness. Persist suite/task/fold IDs, capability snapshot, relation settings, search trial, preprocessing, metrics, artifact hashes, dependency versions, timings, memory, and status/reason.

## Study Portfolio

| Track | Data and protocol | Outcomes and comparators |
|---|---|---|
| Supervised prediction, kernel/kNN, quantiles, conformal | CC18 classification and study-455 regression; official outer folds; train/calibration/test roles where needed | Prediction requests all fitted relations; quantile/conformal request only quantile-forest `conditional`; identity is RF-GAP/Honest-GAP only. Compare native parent, proximity kNN/kernel prediction, conditional quantiles, and split conformal. Primary metrics: balanced accuracy/log loss; RMSE/MAE/R²; pinball loss, coverage, interval/set size. |
| Imputation | Same OpenML folds; induce MCAR/MAR/MNAR only after splitting at 5/10/25/50/75% missingness | Every fitted relation is used as direct donor weights for both one-pass and iterative proximity donors. Report numeric normalized RMSE/MAE, categorical accuracy, and downstream retention against median/mode, kNN, and iterative baselines. |
| Explanations, selection, prototypes, retrieval | Controlled synthetic truth plus CC18/455 | Local importance/selection and prototypes/retrieval request all fitted relations; directed relations are symmetrized only for prototype geometry, not retrieval. Compare local importance with permutation, SHAP, and LIME; selection with mutual-information and embedded baselines; report fidelity, truth recovery, stability, coverage/purity, Recall@k, mAP, nDCG, and latency. |
| Geometry and clustering | All eligible ClustBench v1.1 datasets plus controlled simulations; labels are evaluation-only in label-free runs ([ClustBench documentation](https://clustering-benchmarks.gagolewski.com/clustbench-documentation.html)) | Every fitted relation is included; directed relations are explicitly symmetrized before RF-PHATE/Isomap/UMAP-style embeddings or spectral/HDBSCAN/kernel-PCA clustering. Report ARI/NMI/AMI, stability, trustworthiness, continuity, and neighborhood preservation. Select cluster count only with label-free criteria. RF-AE remains deferred pending adapter/OOS semantics. |
| Anomaly detection | All 57 eligible ADBench datasets; supplied labels are outer-scoring-only in label-free runs ([official ADBench repository](https://github.com/Minqi824/ADBench)) | Isolation uses `collision`; synthetic-null and supervised comparators use all fitted symmetric relations, not directed RF-GAP/Honest-GAP. Compare native isolation scores, synthetic-null forests, proximity outlier scores, LOF, and kNN-distance baselines using ROC-AUC, PR-AUC, AP, Precision@contamination, runtime, and stability. |
| Active learning and attribution | CC18/455 pool-based folds | Start with a stratified 10% labeled set; acquire 5% batches through 50%. Compare random, uncertainty, core-set, and proximity-diverse acquisition. Restrict attribution to directed predictive relations and assess deletion/retraining fidelity only. |
| Alignment | Faithful RF-MA/FoSTA classification reference studies on their UCI view-split protocols; FoSTA is UCI-only | RF-MA: 16 datasets, five view constructions, 30% anchors, three repetitions. FoSTA: 15 datasets, six view constructions, 50% target-label masking, five seeds. Report label transfer, FOSCTTM, alignment score, runtime, and memory; separate transductive from inductive/OOS results. See [RF-MA](https://arxiv.org/pdf/2411.15179) and [FoSTA](https://arxiv.org/html/2602.00974). |
| Alignment regression extension | Explicitly label as a new RF-MA-style study, not a FoSTA reproduction; create paired feature views from study-455 training data | Restrict to RF-GAP-capable families; use 30% training-only anchors; hide target labels from fitting/tuning; report target RMSE/MAE/R² and FOSCTTM separately for transductive and OOS evaluation. |

## Tuning, Leakage Control, and Analysis

- Tune the parent separately for each application. Within an application × outer fold, reuse its selected refit across all eligible relation comparisons.
- Use a fixed seeded 50-trial family-specific parent search and 25-trial relation/application search, both with five inner folds. Store frozen Sobol/Latin-hypercube trial manifests.
- Specify conditional spaces in the note:
  - Forests: 256–1,024 trees; depth `{None,8,16,32}`; leaf size `{1,2,5,10,20}`; valid feature/sample fractions; force bootstrap for OOB/RF-GAP candidates.
  - Boosters: 100–1,000 stages, learning rate 0.01–0.3, depth/leaves, minimum child/leaf size, row/column sampling; CatBoost uses `SymmetricTree` for path studies.
  - Relation/application settings include Englund weight `{0.25,0.5,1,2,4}`, neighbors `{3,5,10,20,40}`, kernel regularization, imputation iterations `{1,3,5,10}`, and label-free clustering/embedding grids.
- Label-free tracks tune only on training covariates, graph diagnostics, reconstruction, and stability; labels remain outer-evaluation-only. Report label-informed geometry separately.
- Fit encoders, scalers, feature selection, missingness generators, calibration splits, anchors, and all tuning inside training partitions. Preserve a separately declared CatBoost-native categorical branch.
- Aggregate folds within dataset before cross-dataset comparison. Report paired effects, bootstrap confidence intervals, average ranks, omnibus tests, and Holm-adjusted paired comparisons; do not pool raw scores.

## Verification and Defaults

- Add mocked tests for study-455/task-fold loading, capability-driven matrix generation, status preservation, and manifest completeness.
- Add deterministic smoke cells for each installed family/relation/application contract; optional dependencies must skip cleanly while emitting declared rows.
- Add leakage tests for inner/outer separation, label-free tuning, directed/symmetrized relation use, calibration separation, and paired-view anchors.
- Add synthetic validity checks for RF-GAP/Honest-GAP identity, imputation recovery, feature-truth recovery, cluster/anomaly recovery, and alignment label transfer.
- Preserve current user changes. The note will explicitly distinguish executable studies, deferred survival/RF-AE/single-cell work, and invalid gated families.
