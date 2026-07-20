# Forest-method and proximity-application roadmap

**Decision ledger.** Evidence and repository state frozen on **2026-07-14**.
This is the implementation-facing companion to the [literature ledger](proximity_applications_literature.md), not a claim that its 50 direct proximity works are an exhaustive final corpus.  The ledger is a strong, checked snapshot; its counts, bibliography, and local-PDF inventory still need a reconciliation pass.

The unit of comparison is a scientifically valid cell:

```text
estimator family × relation definition × application × data regime
```

An estimator family is not a relation definition.  For example, an RF can
produce original co-leaf affinity, an OOB approximation, RF-GAP prediction
weights, or a future KeRF normalisation; an isolation forest's native
path-length score is not thereby an RF-GAP proximity.

## Reading this ledger

**Decision labels:** **reuse** = supported here and stabilize; **integrate** =
use an existing dependency behind an adapter; **modernize/port** = rehabilitate
an existing implementation with an oracle; **build** = implement a missing
relation/application; **specialty** = bounded, data-specific comparison;
**reference-only** = document but do not make a required implementation.

“Available” below means available to this checkout, not merely installable.
`treeple==0.10.3` is an optional research dependency in the
[`treeple` extra](../pyproject.toml); it is noncommercial and must not be
vendored.  An unavailable optional dependency must remain an explicit
unsupported registry row, rather than silently shrinking a study.

The 1.0 public composition surfaces are `Proximity`, `get_proximities`, and
`prox_extend`. Current general relations are `original`, `oob`, `rfgap`,
`honest_gap`, `kerf`, `boosted`, `path`, and `random_height`; method-specific relations
are named `collision`, `conditional`, and `survival_leaf`. `honest_gap` is
restricted to an honest forest.
`rfgap` is asymmetric prediction-weight geometry, while the original relation
is a symmetric co-leaf affinity.  See the [current application semantics](proximity_applications_summary.md)
and the [manifold conversion tests](../tests/proximity/test_manifold_conversion.py).

## Method decision matrix

Task support is C=classification, R=regression, U=unsupervised/anomaly,
S=survival, and CA=causal. “Leaf/OOB contract” names the information an adapter
must expose: leaf assignment (L), tree inventory (T), in-bag multiplicities
(IB), OOB/query masks (O), and per-tree contribution weights (W).

| Foundational family | Task / partition geometry | Python/sklearn status; other-language oracle | Leaf/OOB contract; license & maintenance | Scientifically valid relation(s) | Decision and rationale |
|---|---|---|---|---|---|
| [CART](https://doi.org/10.1201/9781315139470) | C/R; axis-aligned recursive regions | sklearn trees; R `rpart` | L/T only; BSD-3-Clause sklearn, maintained | Original co-leaf / leaf incidence only | **reference-only** ancestor; sklearn RF supplies the relevant ensemble baseline. |
| [ID3](https://doi.org/10.1007/BF00116251) | C; categorical information-gain regions | No project adapter; Weka ID3 | L/T only; Weka GPL, maintained suite | Original co-leaf only | **reference-only** historical contrast. |
| [C4.5](https://shop.elsevier.com/books/c45/quinlan/978-0-08-050058-4) | C; pruned axis-aligned regions with missing/categorical routing | No project adapter; Weka J48/C5.0 | L/T only; external implementation; maintained descendants | Original co-leaf only | **reference-only**; unique missing-value routing is not currently a hypothesis. |
| [Random subspace](https://doi.org/10.1109/34.709601) | C/R; feature-subsampled trees | Covered as RF component; Weka oracle | L/T; implementation dependent | Original leaf incidence | **reference-only** precursor; redundant with RF ablations. |
| [Bagging](https://doi.org/10.1007/BF00058655) | C/R; bootstrap-aggregated partitions | sklearn Bagging; no generic project adapter | L/T/IB/O required; BSD-3-Clause sklearn, maintained | Original/OOB only if bootstrap roles are recoverable | **reference-only** ancestor; needed conceptually for OOB/RF-GAP validity. |
| [Random Forest](https://doi.org/10.1023/A:1010933404324) | C/R; axis-aligned bootstrap forest | sklearn RF available | L/T/IB/O/W; BSD-3-Clause, maintained | Original, OOB, RF-GAP, KeRF, path, random-height | **reuse** and stabilize; baseline implementation and prediction-identity anchor. |
| [Extra-Trees](https://doi.org/10.1007/s10994-006-6226-1) | C/R; randomized axis-aligned cuts | sklearn ExtraTrees available | L/T; IB/O only under an explicitly valid bootstrap configuration; BSD-3-Clause, maintained | Original, KeRF, path, random-height; OOB/RF-GAP only when bootstrap contract holds | **reuse** as randomized-partition comparator. |
| [AdaBoost](https://doi.org/10.1006/jcss.1997.1504) | C/R; sequential weighted trees | sklearn available | L/T/W; no exchangeable bootstrap/OOB contract | Explicit boosted-tree relation only | **reference-only**; ordinary RF-GAP is invalid. |
| [Gradient boosting](https://doi.org/10.1214/aos/1013203451) | C/R; additive stagewise trees | sklearn adapter available | L/T/path/W; no RF bootstrap/OOB | Original, KeRF, contribution-weighted `boosted`, path, Englund, random-height | **integrated.** Stage/class trees are flattened and empirical fitted-contribution variances supply normalized tree weights; not an RF-GAP target. |
| [Stochastic gradient boosting](https://doi.org/10.1016/S0167-9473(01)00065-2) | C/R; subsampled stagewise trees | sklearn available | L/T/W; subsampling does not make RF-GAP valid | Explicit boosted relation only | **reference-only**; historical bridge only. |
| [XGBoost](https://github.com/dmlc/xgboost) | C/R; scalable boosted partitions | Optional `boosted`-extra adapter | L/T/path/W; Apache-2.0, actively maintained | Original, KeRF, contribution-weighted `boosted`, path, Englund, random-height; never ordinary RF-GAP | **integrated** with native leaf inference and cached JSON tree geometry. |
| [LightGBM](https://github.com/lightgbm-org/LightGBM) | C/R; leaf-wise histogram GBDT | Optional `boosted`-extra adapter | L/T/path/W; MIT, maintained | Original, KeRF, contribution-weighted `boosted`, path, Englund, random-height | **integrated** with native leaf inference and cached model-dump geometry. |
| [CatBoost](https://github.com/catboost/catboost) | C/R; ordered categorical/oblivious trees | Optional `boosted`-extra adapter | L/T/W for every grow policy; symmetric-tree paths; Apache-2.0, maintained | Original, KeRF, `boosted`; path/Englund/random-height for `SymmetricTree` only | **integrated** with native DataFrame/Pool typing and explicit non-symmetric path gating. |
| [Early oblique induction](https://dblp.org/rec/conf/ijcai/HeathKS93) | C/R; learned hyperplane cuts | No maintained Python implementation found | L/T if supplied; historical implementation unavailable | Original co-leaf only | **reference-only**; OC1/modern oblique forests cover the comparison. |
| [OC1](https://doi.org/10.1613/jair.63) | C/R; hill-climbed hyperplanes | Archived/third-party wrapper only | L/T; GPL wrapper, not a production dependency | Original co-leaf only | **reference-only** unless its particular optimization becomes a hypothesis. |
| [Menze obliqueRF](https://doi.org/10.1007/978-3-642-23783-6_29) | C/R; regularized linear node splits | Archived R package | L/T; archived, avoid vendoring | Original co-leaf only | **reference-only**; retain as an R conformance/reference source. |
| [Rotation Forest](https://doi.org/10.1109/TPAMI.2006.211) | C/R; PCA-rotated coordinate trees | aeon adapter available when installed | L/T plus synchronized aeon training-estimate reconstruction for IB/O/W; aeon BSD-3-Clause, maintained | Original, OOB, RF-GAP, KeRF, path, random-height | **reuse** through the aeon-compatible adapter; external RF-GAP uses fitted rotation trees and training OOB uses matching refitted trees. |
| [Canonical Correlation Forest](https://arxiv.org/abs/1507.05444) | C/R; supervised CCA hyperplanes | Native sklearn-compatible C/R estimators; deterministic fixture tied to the author [MATLAB oracle](https://github.com/twgr/ccfs) | L/T/path established; CCF-Bag persists IB/O/W samples | Original, KeRF, path, random-height; OOB/RF-GAP for CCF-Bag only | **modernized.** Packaged C/R oracle checks pass and CCF-Bag preserves native test/OOB prediction identity. |
| [SPORF](https://www.jmlr.org/papers/v21/18-664.html) / treeple ObliqueRF | C/R; sparse random projection hyperplanes | Optional treeple adapter available | L/T/IB/O/W; treeple noncommercial, maintained research dependency | Original/OOB/RF-GAP if contract exposes bootstrap state | **reuse** and stabilize; optional/unavailable rows stay visible. |
| [Isolation Forest](https://doi.org/10.1109/ICDM.2008.17) | U; random axis-aligned isolation paths | sklearn `IsolationForest` available | Path scores native; L/T only for a separately defined collision relation; BSD-3-Clause | Native `score_samples`; any co-leaf relation must be labeled distinct | **reuse** native scores; **restricted** to native path-score baselines for anomaly claims. |
| [Extended Isolation Forest](https://doi.org/10.1109/TKDE.2019.2947676) | U; random oblique isolation paths | Optional treeple family / external EIF | Path scores native; L/T distinct from RF bootstrap | Native path score; explicit partition relation only | **reuse** native score when installed; no RF-GAP semantics. |
| Shi–Horvath unsupervised RF ([paper](https://doi.org/10.1198/106186006X94072)) | U; observed-vs-synthetic classifier partitions | Native `SyntheticNullForest` meta-estimator | L/T/IB/O with persisted generator and seed | Original/OOB and native observed-class anomaly score | **integrated.** Relations index observed rows only; the synthetic construction is retained as provenance. |
| [Quantile Regression Forest](https://www.jmlr.org/papers/v7/meinshausen06a.html) | R; conditional leaf distributions | Optional [`quantile-forest`](https://zillow.github.io/quantile-forest/) adapter | L/T and leaf-mass conditional weights; BSD-3-Clause, maintained | `conditional` weights plus native quantiles | **integrated.** Held-out tutorial reports pinball loss and interval coverage against native predictions. |
| [Random Survival Forest](https://doi.org/10.1214/08-AOAS169) | S; censored-outcome forest | Optional [`scikit-survival`](https://scikit-survival.readthedocs.io/en/stable/user_guide/random-survival-forest.html) adapter; R `randomForestSRC` oracle | L/T plus native survival/risk/hazard outputs | Separately named `survival_leaf` relation | **integrated.** Evaluation uses training-derived IPCW concordance and integrated Brier score inside follow-up support. |
| [Causal Tree](https://doi.org/10.1073/pnas.1510489113) | CA; honest treatment-effect partition | R `causalTree`; no current Python adapter | Honest structure/estimation masks essential | Honest neighborhood only with declared causal protocol | **specialty**; never infer causality from generic proximity. |
| [Causal Forest](https://doi.org/10.1080/01621459.2017.1319839) | CA; honest adaptive treatment neighborhoods | R [`grf`](https://github.com/grf-labs/grf) and Python causal comparators | Honest masks, treatment/outcome roles, W; GPL-3.0 R oracle | Matching/balancing and causal forest weights | **specialty** after leakage and estimand protocol. |
| [Generalized Random Forest](https://doi.org/10.1214/18-AOS1709) | CA/general local moments | R `grf` oracle | Same honest/role contract as causal forests | Local-moment forest weights | **specialty**; compare with Python causal tools, not generic RF-GAP. |
| [Bayesian CART](https://doi.org/10.1080/01621459.1998.10473750) | C/R; posterior over partitions | No maintained exact adapter found | Posterior tree draws required | Posterior co-clustering only after semantics defined | **reference-only**. |
| [BART](https://doi.org/10.1214/09-AOAS285) / PyMC-BART | C/R; posterior additive trees | Optional PyMC-BART; R BART oracle | Posterior draws and tree contributions, not RF OOB | Posterior co-clustering/uncertainty relation | **specialty** only after posterior relation semantics and invariants are specified. |

### Relation constructors, independent of the estimator choice

| Relation | Current status | Validity gate | Next action |
|---|---|---|---|
| Original co-leaf affinity | Available | L/T | Keep as symmetric baseline. |
| OOB co-leaf approximation | Available | Recoverable bootstrap O masks | Keep labeled as the current separable approximation, not a generic OOB formula. |
| RF-GAP | Available | Exchangeable bootstrap forest with L/IB/O/W; unsymmetrized for prediction identity | Keep as directed prediction weights; reject it for boosted/Bayesian/isolation ensembles. |
| Honest-GAP | Available for HonestForest classifier | Honest structure/estimation roles plus compatible OOB output | Re-run and report it; mark older ordinary-RF-GAP mismatch as legacy. |
| KeRF / leaf-mass normalization | Available as `kerf` | L/T and leaf mass | Symmetry/PSD and sparse/dense invariants are tested. |
| Boosted contribution kernel | Available as `boosted` for sklearn GBT, XGBoost, LightGBM, and CatBoost | L/T plus fitted leaf contributions on attached training data | Normalize empirical contribution variances, fall back to uniform when all are zero, and keep prediction identity false. |
| Path/branch similarity | Available as `path` | Decision paths and explicit normalization | Shared-path feature construction is tested as a PSD kernel. |
| Random-height partition kernel | Available as `random_height` | Decision paths plus a seeded cached cut protocol | Seed determinism, cache behavior, and PSD invariants are tested. |
| PF-GAP / structured-object relation | Missing | Object metric, split semantics and a structure-preserving evaluator | **Specialty** after vector-valued core benchmarks. |

## Application decision matrix

An application is a reusable use case, not one row per domain paper. Evidence
links are representative primary sources; the literature ledger holds the full
work-level evidence. “Current” records the actual package surface, runner, and
test state at the freeze; a dash means not yet implemented, not “not useful.”

| Tier / reusable use case | Representative evidence; required relation | Leakage/supervision risk and data | Baselines and metrics | Current API / runner / tests | Priority and acceptance criterion |
|---|---|---|---|---|---|
| 0 — prediction identity | [RF-GAP](https://doi.org/10.1109/TPAMI.2023.3263774); directed RF-GAP/Honest-GAP | Train/OOB/query roles must not mix; C/R labels | Native prediction/OOB output; max absolute error, row-weight mass | `prox_predict`, `prox_extend`; `run_prediction_identity.py`; `test_prox.py`, `test_honest_gap.py` | **Correct/complete.** Every valid adapter matches native test and valid OOB predictions to tolerance. |
| 0 — imputation | [yaImpute](https://doi.org/10.18637/jss.v023.i10); donor weights | Fit preprocessing/forest on training data only; MCAR/MAR/MNAR and feature types | Simple/iterative imputation; numeric RMSE, categorical accuracy, downstream C/R utility | `impute`; `run_imputation.py`; `test_missingness.py` | **Correct/complete.** Rerun redesigned smoke suite and preserve mask/split provenance. |
| 0 — anomaly, novelty, mislabel audit | [Isolation Forest](https://doi.org/10.1109/ICDM.2008.17), [Shi–Horvath](https://doi.org/10.1198/106186006X94072); native paths or declared relation | Do not use labels to score nominal unsupervised anomalies; contamination protocol | Native IF/EIF scores, supervised baselines; ROC-AUC, AP, rank/top-k agreement | `outlier_scores`; `run_outlier_detection.py`; runner tests are indirect | **Correct/complete.** Native isolation scores remain required baselines and unsupported methods report explicitly. |
| 0 — local feature importance | [Whitmore et al.](https://arxiv.org/abs/1807.06572); local relation perturbation | Evaluation labels cannot choose the explanation neighborhood | Permutation/conditional alternatives; local prediction drop, stability, runtime | `local_feature_importance`; `run_local_feature_importance.py`; no dedicated unit suite | **Correct/complete.** Add deterministic local-neighborhood and leakage smoke tests when refactoring. |
| 0 — prototypes and retrieval | [Tree Space Prototypes](https://doi.org/10.1145/3412815.3416893); affinity or directed query weights | Prototype selection must not consume held-out labels; class constraints explicit | Random/class medoids/kNN; coverage, class purity, retrieval precision, stability | `get_prototypes`; `run_prototypes.py`; runner coverage | **Correct/complete.** Rerun prototype suite with legacy outputs retained. |
| 0 — manifold visualisation | [RF-PHATE](https://doi.org/10.1038/s43588-026-00999-7); affinity→distance or PHATE affinity | Labels only for colouring/evaluation; directed affinities need declared symmetrisation | PCA/UMAP/t-SNE/PHATE; trustworthiness, neighborhood preservation, label-free structure | `embed`, `embed_transform`, `plot_mds`; `run_mds_visualization.py`; `test_manifold.py` | **Correct/complete.** Quarantine legacy MDS figures, route MDS through shared conversion, then rerun. |
| 1 — kernel SVM/KRR/kNN prediction | [Scornet](https://doi.org/10.1109/TIT.2016.2514489); symmetric PSD kernel or directed kNN weights | Split before kernel construction; train–test rectangular extension | Native forest, RBF/linear kernels, kNN; C/R score, calibration, PSD diagnostics | `ProximityKernelPredictor`, `knn_predict` / method-matrix runner / application tests | **Built.** The shared runner reports parent and downstream scores on identical splits. |
| 1 — clustering | [Shi–Horvath](https://doi.org/10.1198/106186006X94072); symmetric affinity/dissimilarity | Synthetic-null and cluster labels never share tuning roles | k-means/spectral/hierarchical; ARI/NMI/silhouette and stability | `proximity_clustering` / method-matrix runner / application tests | **Built.** Symmetrization is explicit and labels are post-hoc evaluation only. |
| 1 — conditional distributions and quantiles | [Meinshausen](https://www.jmlr.org/papers/v7/meinshausen06a.html); leaf distribution weights | Conditional calibration evaluated on held-out data | Quantile forest/native quantile models; pinball loss, coverage, CRPS | `conditional_distribution_weights`, `conditional_quantiles` / tutorial / adapter tests | **Integrated.** Conditional leaf-mass and held-out calibration checks pass. |
| 1 — conformal/selective prediction | [Yang et al. 2009](https://link.springer.com/article/10.1186/1471-2105-10-S1-S22); calibration scores plus declared neighborhood | Strict train/calibration/test separation; no label leakage through proximity tuning | Split conformal, RF intervals; marginal/conditional coverage, set size, risk–coverage | `split_conformal_classification`, `split_conformal_regression` / application tests | **Built as a split-conformal utility.** Callers retain explicit train/calibration/test roles. |
| 1 — batch active learning | [Kee et al.](https://www.sciencedirect.com/science/article/abs/pii/S0020025518303700); diversity/uncertainty relation | Pool labels unavailable until acquisition; batch redundancy control | Random/uncertainty/core-set; learning curve, area under curve, diversity | `select_active_batch` / application tests | **Built as a batch-selection primitive.** Labels are not accepted by the selector. |
| 1 — training-instance attribution | [AXIL](https://arxiv.org/abs/2301.01864) / RF-GAP; directed query-to-training weights | Attribution target and retraining counterfactual cannot be reused for selection | Influence/leave-one-out; deletion correlation, fidelity, sign/mass diagnostics | `instance_attributions` / application tests | **Built with predictive-weight semantics only;** it makes no causal influence claim. |
| 1 — retrieval | [Corporate-bond retrieval](https://doi.org/10.1145/3533271.3561736); query-to-index relation | Relevance judgments and labels held out by query/group/time | Cosine, Euclidean, kNN; Recall@k, mAP, nDCG, latency | `retrieve_neighbors`, `retrieval_metrics` / method-matrix runner / application tests | **Built as generic query-to-index primitives.** Domain relevance protocols remain caller-owned. |
| 1 — proximity-neighborhood feature selection | [Shi–Horvath](https://doi.org/10.1198/106186006X94072); local stability/conditional relation | Feature selection nested within CV; no outcome leakage into neighbor graph | Filter/wrapper/embedded methods; selected-feature stability and C/R utility | `neighborhood_feature_scores`, `select_features` / application tests | **Built as nested-workflow primitives.** Selection must still occur inside caller CV. |
| 2 — personalized local models | [Lee 2017](https://doi.org/10.2196/medinform.6690); query weights | Patient/group/time leakage; local sample-size floor | Global RF, kNN, local GLM; per-case loss, calibration, subgroup error | — / — / — | **Specialty.** Open cohort and transport protocol required. |
| 2 — causal matching and balancing | [Zhao et al.](https://doi.org/10.1016/j.cct.2015.12.012); honest/causal neighborhood | Treatment, outcome, and post-treatment variables strictly separated | Propensity/nearest-neighbor/causal forest; balance, ATT/ATE bias, coverage | — / — / — | **Specialty.** Causal estimand and negative controls gate implementation. |
| 2 — survival stratification | [RSF clustering](https://www.nature.com/articles/s41598-021-92072-8); survival relation | Censoring and time-dependent split leakage | RSF risk, Cox, clinical strata; C-index, IBS, time-dependent AUC, log-rank | — / — / — | **Integrate then specialty.** scikit-survival adapter plus censoring-safe evaluation. |
| 2 — process/change monitoring | [Auret–Aldrich 2011](https://skoge.folk.ntnu.no/prost/proceedings/ifac11-proceedings/data/html/papers/1928.pdf); temporal neighborhood/dissimilarity | Respect time order, delayed labels, and drift windows | Control charts, PCA, IF; detection delay, false-alarm rate, ARL | — / — / — | **Specialty.** Prequential protocol must be written first. |
| 2 — multiview fusion/alignment | [Cao et al.](https://doi.org/10.1016/j.patcog.2018.11.011); per-view/cross-view relation | View pairing, batch and label leakage | Early/late fusion, CCA; task score, alignment, view ablation | — / — / — | **Specialty.** Define missing-view and pairing policy before code. |
| 2 — time series | [Forest Proximities for Time Series](https://doi.org/10.1007/978-3-032-07109-5_40); PF-GAP/structured relation | Temporal blocking and no future observations in features | DTW/distance forest, forecasting baselines; forecast error, anomaly delay | — / — / — | **Specialty.** Object relation and temporal split protocol gate the track. |
| 2 — arbitrary structured objects | [generalized proximity forest](https://arxiv.org/abs/2511.19487); object-aware partition relation | Representation/metric is part of the estimator and evaluation | Object kernels/metric learners; task score, compute, invariance | — / — / — | **Specialty.** No vector-only API may imply support. |
| 3 — proximity graphs/GNNs | [RF-GNN](https://arxiv.org/abs/2602.24224); sparse symmetric graph | Transductive edges and test labels/features must be controlled | kNN graph/GNN/MLP; C/R score, graph sparsity, inductive transfer | — / — / — | **Capstone.** Graph construction and inductive/transductive protocol required. |
| 3 — autoencoding/denoising/compression | [RF autoencoders](https://arxiv.org/abs/2502.13257); relation or path representation | Reconstruction target must not leak evaluation samples | PCA/AE/VAE; reconstruction, downstream utility, compression, robustness | — / — / — | **Capstone.** Define posterior/geometry semantics and OOS encoder before study. |
| 3 — open domain case studies | Biology, medicine, finance, astronomy, remote sensing; use only a representative open set | Domain-specific group/time/censoring leakage | Domain-standard plus generic baselines; preregistered utility and failure modes | Existing runners use OpenML only | **Capstone.** Reusable primitive first, then a small open, reproducible case set—not a reproduction per paper. |
| Deferred — label propagation | Evidence gap for direct forest-induced relation use | Graph labels can leak through edges | Graph SSL baselines; accuracy/calibration | — / — / — | **Defer** until direct evidence and Tier 1 graph benchmarks strengthen. |
| Deferred — proximity resampling | Evidence gap | Synthetic labels/rare outcomes and resampling must be nested | SMOTE/class weighting; minority utility/calibration | — / — / — | **Defer** pending direct evidence. |
| Deferred — fairness-neighborhood audit | Evidence gap | Protected attributes, group membership and proxy leakage | Group calibration/error/balance; disparity confidence intervals | — / — / — | **Defer** pending direct evidence and a fairness protocol. |

## Repository and experimental sequence

| Phase | Dependencies / blockers | Status | Measurable acceptance criterion |
|---|---|---|---|
| 1. Evidence freeze | Targeted backward/forward searches from conformal, active-learning, monitoring, survival, causal, personalized, and selection seeds; URL/version/license/oracle checks | In progress: literature snapshot is strong | Two snowball passes add no reusable theme; every method/application/theme has exactly one decision and rationale. |
| 2. Correctness and provenance | Shared `proximity_to_distance`; HonestForest optional dependency; legacy artifact location | Complete for 1.0 | Sparse zeros map to maximal distance; honest identity uses `honest_gap`; pre-1.0 MDS artifacts are under `results/legacy`; the shared harness records provenance and failures. |
| 3. Adapter architecture | Composition API and formal capability contract | Complete for registered 1.1 families | Adapters declare L/T/task/IB/O/W; invalid pairs fail explicitly; relation parity/invariant tests pass. |
| 4. Core expansion | CCF oracle validation, relation specifications | Complete for the scoped 1.1 relations | Packaged C/R CCF fixture checks, CCF-Bag OOB/RF-GAP identity, RotationForest OOB adapter identity, boosted adapters, and KeRF/path/random-height invariants pass. |
| 5. Tier 1 applications | Stable relations, deterministic shared splits, benchmark protocols | Utilities complete; broader benchmark studies remain future work | All listed utilities have focused tests; prediction/clustering/retrieval share the method-matrix runner and quantile has a held-out native comparison. |
| 6. Specialty tracks | Data-specific protocol written and reviewed first | Deferred | Causal, survival, time-series, multiview, structured, graph, and autoencoding tracks individually satisfy their table acceptance criteria. |
| 7. Cross-method study | Validity matrix and shared experiment harness | Harness and local matrix complete; full-scale estimation pending | The runner retains every valid/unsupported cell and records parent/downstream scores, diagnostics, revision/config/split, runtime, memory, and failures. |

### Required protocol for the cross-method study

Reuse the identical split and fitted parent model when relation constructors or
applications are compared.  Stratify datasets by axis-aligned versus
rotated/oblique signal, `p/n`, feature correlation, mixed/categorical values,
imbalance, missingness, cluster structure, censoring, temporal dependence,
multiview structure, and non-vector objects.  Do not pool raw scores across
datasets: estimate within-dataset utility changes and method–data interactions.

Every run must record code revision, configuration, data/split identifiers,
wall-clock runtime, peak memory, relation diagnostics, and failure/unsupported
status.  Future gates are adapter conformance, sparse/dense parity,
relation-specific invariants, deterministic shared splits, absent-dependency
behavior, CLI smoke tests, and application-specific leakage checks.

## Immediate TODOs

1. Reconcile the literature ledger's work counts, bibliography keys, URLs, and
   lawful PDF inventory; retain “not found” as a bounded search result.
2. Create a capability registry that exposes L/T/task/IB/O/W and prevents
   invalid RF-GAP use on boosted, Bayesian, and isolation ensembles.
3. Replace `plot_mds`'s legacy sparse conversion with the tested shared
   conversion, archive current MDS conclusions as legacy, and rerun the Tier 0
   smoke suites with provenance capture.
4. Re-run HonestForest only with `honest_gap` and label the older ordinary
   RF-GAP result as legacy.
5. Modernize CCF against its MATLAB oracle; then add KeRF, path/branch, and
   random-height constructors before broader application work.
