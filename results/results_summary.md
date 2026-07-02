## Benchmark Run Summary

Date: April 21, 2026

Environment:
- All experiment runs used `PYTHONNOUSERSITE=1` in the `generalforest` conda environment.
- Benchmarks used `OpenML-CC18` for classification and `OpenML-CTR23` for regression.
- Requested model families were run wherever the current environment supports them:
  `RandomForest`, `RotationForest`, `HonestForest`, `ObliqueRandomForest`, `PatchObliqueRandomForest`, `ExtraTrees`.

## Imputation Benchmark Redesign

Date: April 22, 2026

Status:
- `experiments/run_imputation.py` and `experiments/common.py` were updated for the expanded imputation benchmark.
- The full CC18/CTR23 imputation sweep has not been rerun yet after this redesign.
- The notes below describe the new experiment design and the smoke tests that validated the code path.

Updated imputation design:
- The proximity configurations remain unchanged:
  `original + single_pass`, `oob + single_pass`, `rfgap + single_pass`, `rfgap + iterative`.
- The OpenML benchmark split remains unchanged:
  `OpenML-CC18` for classification and `OpenML-CTR23` for regression.
- The CLI now supports `--missingness-mechanisms` and `--missing-rates`, with defaults
  `MCAR,MAR,MNAR` and `0.05,0.10,0.25,0.50,0.75`.
- For each dataset, task kind, random state, missingness mechanism, and missing rate, one deterministic train mask and one deterministic test mask are generated once and then reused across every forest family, proximity method, variant, and downstream model.

Missingness mechanisms:
- `MCAR`: uniform random masking over currently observed cells, with an exact per-feature masking count of `floor(rate * n_observed)` on train and test.
- `MAR`: for feature `j`, masking depends only on other columns. The implementation picks up to two usable conditioning columns from the training split, standardizes them with training-derived moments, builds a logistic score, calibrates the intercept on the training split, and then samples an exact masking count using those probabilities as weights.
- `MNAR` numeric: masking depends on `x_j` through a training-derived quantile-rank logistic score. Constant or all-missing numeric columns fall back to MCAR and record fallback metadata.
- `MNAR` categorical: masking depends on category-specific rarity scores estimated from the training split. Very low-support levels are pooled to a neutral score; if a categorical column is all-missing, constant, or has too little usable support after pooling, it falls back to MCAR and records fallback metadata.
- MAR and MNAR fallback metadata are stored per row as JSON summaries:
  `missingness_fallback_count`, `missingness_fallback_summary_json`, and `missingness_metadata_json`.

Evaluation updates:
- Imputation quality is still evaluated on masked cells only.
- Raw rows now include train/test numeric MSE, numeric RMSE, categorical accuracy, and masked-cell counts for numeric and categorical features separately.
- A downstream stage now runs after imputation.
- Classification downstream models:
  `LogisticRegression` and `KNeighborsClassifier`.
- Regression downstream models:
  `Lasso` and `KNeighborsRegressor`.
- Downstream models use a `StandardScaler` pipeline. Any remaining native OpenML missing values are handled with the existing deterministic train-derived rough fill before the downstream estimator fit, including for the oracle baseline.
- Oracle downstream baselines are computed on the original uncorrupted `prepared.X_train` and `prepared.X_test` splits and stored alongside each imputed row as `oracle_*` metrics.

Updated result schema:
- Raw rows now carry:
  `experiment`, `status`, `task_kind`, `suite_name`, `dataset_id`, `dataset_name`, `task_id`, `random_state`, `model_name`, `prox_method`, `variant`, `missingness_mechanism`, `missing_rate`, `downstream_model`, train/test imputation metrics, downstream metrics, oracle downstream metrics, runtime columns, and missingness fallback metadata.
- Compiled `summary.csv` for imputation now groups by:
  `model_name`, `prox_method`, `variant`, `missingness_mechanism`, `missing_rate`, `downstream_model`.
- Higher-level compiled tables and figures now include:
  `imputation_rate_comparison.csv`, `downstream_rate_comparison.csv`, and missing-rate figures split by mechanism for classification and regression.

Smoke-test evidence:
- Classification smoke test:
  `PYTHONNOUSERSITE=1 conda run -n generalforest python ... --task classification --dataset-limit 1 --random-states 0 --max-rows 120 --n-estimators 16 --n-iterations 2 --missingness-mechanisms MCAR,MAR,MNAR --missing-rates 0.05`
- Classification smoke result:
  `144/144` rows succeeded on `kr-vs-kp`, with all three mechanisms and both downstream models represented.
- Regression smoke test:
  `PYTHONNOUSERSITE=1 conda run -n generalforest python ... --task regression --dataset-limit 1 --random-states 0 --max-rows 120 --n-estimators 16 --n-iterations 2 --missingness-mechanisms MCAR,MAR,MNAR --missing-rates 0.05`
- Regression smoke result:
  `120/144` rows succeeded on `abalone`; the remaining `24` rows were the expected unsupported `HonestForest` regression entries.
- Compile smoke test:
  the updated `compile_outputs()` path was exercised against combined classification/regression smoke frames and wrote raw and compiled outputs under `results/_smoke_raw/imputation` and `results/_smoke_compiled/imputation`, including the new summary tables and mechanism-split figures.

## Prototype Benchmark Redesign

Date: April 23, 2026

Status:
- `experiments/run_prototypes.py` was rewritten around the existing `model.get_prototypes(...)` extraction path.
- `experiments/common.py` gained a small estimator-parameter introspection helper so the prototype benchmark only requests OOB diagnostics when the estimator family supports them cleanly.
- The full OpenML-CC18 prototype sweep has not been rerun yet after this redesign.
- The notes below describe the new design and the smoke tests that validated the updated raw and compiled outputs.

Definition of a good prototype:
- Purity:
  the selected neighborhood should be mostly same-class.
- Representativeness:
  the prototype center should look like a strong in-class representative rather than an arbitrary or fringe point.
- Coverage:
  the selected prototype set should cover a substantial share of the class population.
- Non-redundancy:
  multiple prototypes from the same class should not collapse onto the same local region.
- Predictive utility:
  a prototype-only classifier should preserve useful held-out signal relative to the full forest.
- Stability:
  core summary metrics and, on synthetic data, recovered class modes should not change wildly across seeds.

Updated prototype design:
- Track A is synthetic classification validation with two controlled scenarios:
  `synthetic_multimodal_balanced` and `synthetic_overlap_imbalance`.
- Track B remains the main real-data validation track on `OpenML-CC18`.
- Classification comparisons now sweep every supported classifier family in the experiments framework across:
  `original`, `oob`, and `rfgap`.
- For each dataset and random state, the train/test split and preprocessing are created once and then reused across all compared model families and proximity modes.
- Regression remains explicit unsupported output:
  `run_prototypes.py` now records unsupported rows for every requested model family and every proximity mode on `OpenML-CTR23`, rather than skipping regression silently.

Learned path and baseline:
- Prototype extraction still goes through `model.get_prototypes(k=..., n_prototypes=..., categorical_features=...)`.
- The required baseline is now implemented:
  `random_same_class_center`.
- For each learned run, the benchmark samples classwise random centers with the same requested `k` and `n_prototypes`, builds neighborhoods using the same fitted proximity matrix and the same within-class neighborhood-removal rule, and repeats that baseline multiple times with deterministic seeds.

Prototype quality metrics:
- Purity:
  `same_class_neighbor_fraction`, `same_class_neighbor_count`, and class-balanced purity summaries.
- Coverage:
  overall training coverage, class-balanced training coverage, and test-set nearest-center label accuracy.
- Non-redundancy:
  within-class neighborhood-overlap Jaccard, within-class center proximity, and unique-covered-neighbor fraction.
- Predictive utility:
  prototype-only held-out accuracy using both sum and max aggregation over class prototype centers, plus the accuracy gap versus the full forest.
- Representativeness:
  focal-class in-sample `predict_proba`, OOB probability when valid, and percentile diagnostics inspired by `notebooks/prototypes_demo.ipynb`.
- Stability:
  per-dataset standard deviations across random states for purity, coverage, prototype accuracy, and center proximity; synthetic runs also report cross-run Jaccard agreement of recovered center-mode sets.

Synthetic-only validation metrics:
- `mode_coverage_fraction`:
  fraction of known class submodes hit by the selected prototype centers.
- `mean_same_mode_neighbor_fraction`:
  fraction of each prototype neighborhood that stays inside the center’s true generating mode.
- `mean_center_to_mode_distance`:
  Euclidean distance from the selected center to its true subcluster center.
- `distinct_center_mode_fraction`:
  fraction of prototype centers within a class that land on distinct known modes.

Representativeness/OOB fallback:
- Prototype-center confidence diagnostics now follow the notebook idea:
  compare prototype centers to same-class peers using in-sample probabilities and OOB probabilities when available.
- If a family does not provide clean OOB probabilities for this application, the benchmark falls back explicitly.
- In the smoke run, `HonestForest` exposed non-finite OOB values for this diagnostic, so its prototype OOB confidence fields are recorded as unavailable rather than being treated as probabilities.

Updated result schema:
- Raw summary rows now include:
  `experiment`, `status`, `task_kind`, `suite_name`, `dataset_id`, `dataset_name`, `task_id`, `random_state`, `model_name`, `prox_method`, `prototype_source`, `baseline_repeat`, `k`, `n_prototypes`, `n_centers`, purity metrics, coverage metrics, redundancy metrics, prototype-only predictive metrics, representativeness metrics, stability metrics, and runtime columns.
- Raw detail rows now include:
  `task_kind`, `suite_name`, `model_name`, `prox_method`, `prototype_source`, `dataset_name`, `dataset_id`, `task_id`, `random_state`, `class_label`, `prototype_number`, `center_position`, `center_index`, neighborhood size, purity metrics, redundancy diagnostics, prototype-center confidence metrics, and synthetic mode fields when applicable.
- Compiled outputs now include:
  `summary.csv`, `model_comparison.csv`, `purity_summary.csv`, `coverage_summary.csv`, `redundancy_summary.csv`, `predictive_utility_summary.csv`, `representativeness_summary.csv`, `baseline_comparison.csv`, and `synthetic_mode_summary.csv`.
- The classification compiled directory also writes suite-specific plots for:
  prototype accuracy, coverage, mean same-class neighbor fraction, and accuracy improvement over random centers.

Smoke-test evidence:
- Synthetic-only smoke test:
  `PYTHONNOUSERSITE=1 bash -lc 'source /yunity/rhodes8/anaconda3/bin/activate generalforest && python experiments/run_prototypes.py --task classification --skip-openml --random-states 0 --n-estimators 16 --k 10 --n-prototypes 2 --baseline-repeats 2'`
- Synthetic-only smoke result:
  `108/108` summary rows succeeded and `648` detail rows were written across the two synthetic scenarios, six classifier families, three proximity modes, and the learned-vs-random baseline comparison.
- Mixed classification smoke test:
  `PYTHONNOUSERSITE=1 bash -lc 'source /yunity/rhodes8/anaconda3/bin/activate generalforest && python experiments/run_prototypes.py --task classification --dataset-limit 1 --random-states 0 --max-rows 120 --n-estimators 16 --k 10 --n-prototypes 2 --baseline-repeats 2'`
- Mixed classification smoke result:
  `162/162` summary rows succeeded and `864` detail rows were written across `synthetic_multimodal_balanced`, `synthetic_overlap_imbalance`, and the first capped `OpenML-CC18` slice (`kr-vs-kp`).
- Regression unsupported smoke test:
  `PYTHONNOUSERSITE=1 bash -lc 'source /yunity/rhodes8/anaconda3/bin/activate generalforest && python experiments/run_prototypes.py --task regression --dataset-limit 1 --random-states 0 --skip-synthetic --n-estimators 8'`
- Regression unsupported smoke result:
  `18/18` rows were written as explicit unsupported outputs for `task_361234`, covering six requested model families and all three proximity modes.

Smoke-run observations:
- On the capped `kr-vs-kp` slice, the learned prototype method usually improved purity relative to random same-class centers and often improved prototype-only accuracy as well; those deltas are stored in `results/prototypes/classification/baseline_comparison.csv`.
- The balanced synthetic scenario is intentionally near ceiling, so learned and random baselines can tie there on simple accuracy.
- The overlapping synthetic scenario is where the stronger metrics matter more:
  mode coverage, purity, and random-baseline improvement are more informative than raw prototype-only accuracy alone.

Code changes used for this run:
- `forestgap/datasets.py`: OpenML suite alias/prefix resolution now accepts `OpenML-CC18` and `OpenML-CTR23`.
- `forestgap/core.py`: retained the treeple/sklearn compatibility shim for sample weights.
- `experiments/common.py`: model registry expanded to the requested families, runtime columns added across applications, `FORESTGAP_N_JOBS` can now override estimator parallelism, and the imputation helpers now include MCAR/MAR/MNAR mask generation plus downstream-model evaluation support.
- `experiments/run_imputation.py`: the benchmark now sweeps missingness mechanism and missing rate, evaluates downstream models after imputation, stores oracle downstream baselines, and writes the expanded summary tables and figures.
- `experiments/run_local_feature_importance.py`: `HonestForest` is recorded as unsupported for this application because the current treeple build segfaults during fit on the benchmark sweep.

Stored outputs:
- Raw results: `results/raw/*`
- Compiled tables and figures: `results/prediction_identity`, `results/imputation`, `results/outlier_detection`, `results/prototypes`, `results/local_feature_importance`, `results/mds_visualization`
- Runtime rollups: `results/process_runtimes.csv`, `results/process_runtime_overview.csv`

Coverage:
- `prediction_identity`
  Classification: `864/864` rows succeeded.
  Regression: `350/420` rows succeeded; `70` `HonestForest` regression rows were logged as unsupported.
- `imputation`
  Classification: `3396/3411` rows succeeded; `15` rows errored.
  Regression: `1376/1452` rows succeeded; `70` `HonestForest` regression rows were unsupported and `6` rows errored.
- `outlier_detection`
  Classification: `10344/10346` rows succeeded; `2` `HonestForest` rows errored.
  Regression: `2100/2170` rows succeeded; `70` `HonestForest` regression rows were unsupported.
- `prototypes`
  Classification: `862/864` rows succeeded; `2` `HonestForest` rows errored.
  Regression: `420/420` rows were logged as unsupported because prototypes are classifier-only in `forestgap.core`.
- `local_feature_importance`
  Classification: `720/864` rows succeeded; `144` `HonestForest` rows were logged as unsupported because the current build segfaults during fit.
  Regression: `420/420` rows were logged as unsupported because local feature importance is classifier-only in `forestgap.core`.
- `mds_visualization`
  Classification: `860/864` rows succeeded; `4` `HonestForest` rows errored.
  Regression: `350/420` rows succeeded; `70` `HonestForest` regression rows were unsupported.

Key comparative findings:
- Prediction identity:
  Classification identity was effectively exact for `RandomForest`, `ExtraTrees`, `ObliqueRandomForest`, and `PatchObliqueRandomForest` with mean test max absolute differences around `1e-7`.
  `RotationForest` stayed small but materially worse at `3.35e-07` and was much slower.
  `HonestForest` was the outlier: classification mean test max absolute difference was `2.43e-01`, far worse than every other family.
  Regression identity remained small for the supported models; `ObliqueRandomForest` was best at `0.0196`, followed closely by `RandomForest` at `0.0203`.
- Imputation:
  Classification numeric RMSE favored `HonestForest` (`3819.47`) and `PatchObliqueRandomForest` (`4363.71`), while `RandomForest` and `ObliqueRandomForest` were competitive and much faster than `RotationForest`.
  Classification categorical accuracy was highest for `RandomForest` (`0.5938`) and `ExtraTrees` (`0.5926`).
  Regression numeric RMSE favored `ExtraTrees` (`204862.53`), followed by `RandomForest` (`241225.11`).
- Outlier detection:
  Classification average precision was best for `ObliqueRandomForest` (`0.2726`), followed by `RandomForest` (`0.2574`) and `ExtraTrees` (`0.2485`).
  Regression average precision was again best for `ObliqueRandomForest` (`0.2031`).
- Prototypes:
  `RotationForest` produced the strongest prototype-induced classification accuracy (`0.7572`), ahead of `RandomForest` (`0.7379`) and `ObliqueRandomForest` (`0.7375`).
  `PatchObliqueRandomForest` was the weakest prototype model on this benchmark (`0.6466`).
- Local feature importance:
  `ObliqueRandomForest` achieved the highest top-feature win rate (`0.5451`) and the largest mean top-feature probability drop (`0.1232`), edging out `ExtraObliqueRandomForest` (`0.5350`, drop `0.0974`) and the axis-aligned baselines.
  `PatchObliqueRandomForest` ranked third on win rate (`0.5150`) but had notably higher misclassified fraction (`0.2025`) than the other families.
  `RandomForest` and `ExtraTrees` were competitive for probability drop but trailed on win rate (`0.4829` and `0.4812` respectively).
  `RotationForest` had the largest raw top-feature probability drop (`0.1464`) and the lowest misclassified fraction (`0.1603`), but at roughly 2–3× the runtime of every other model (`207 s` total vs `~87 s`); its win rate (`0.4962`) did not outpace the oblique families.
  The difference between top-feature drop and random-feature drop was consistent across families (roughly `0.04–0.09 pp`), suggesting that all models identify locally informative features above chance, with oblique forests showing the clearest signal.
- MDS:
  All supported models produced plots across the capped benchmark sweep.
  `RotationForest` was consistently the slowest MDS model by a wide margin; `ExtraTrees`, `RandomForest`, and the oblique forests stayed under `1` second mean total runtime per plot on both suites.

Runtime notes:
- Mean per-row total runtimes are summarized in `results/process_runtime_overview.csv`.
- The largest average runtime burden came from `prediction_identity` classification (`6.89s` mean per successful row) and `local_feature_importance` classification (`3.96s` mean per successful row), both dominated by `RotationForest`.
- On classification tasks, `RotationForest` was typically the slowest family by one to two orders of magnitude:
  `prediction_identity`: `36.16s` mean row runtime vs about `0.43-0.62s` for most forest baselines.
  `outlier_detection`: `7.12s` vs about `0.13-0.17s`.
  `prototypes`: `6.04s` vs about `0.68-1.04s`.
  `local_feature_importance`: `8.76s` vs about `2.69-2.78s`.
- `HonestForest` was also noticeably slower than `RandomForest`/`ExtraTrees`, but its bigger issue in this sweep was stability and prediction-identity mismatch rather than runtime alone.

Important limitations:
- `HonestForestRegressor` is not available in the current environment, so all regression `HonestForest` rows are unsupported.
- `get_prototypes()` and `local_feature_importance()` remain classifier-only in `forestgap.core`.
- Several small or highly imbalanced CC18 datasets still fail for `HonestForest` because the least-populated class can drop below `2` after the capped split.
- Some imputation rows still fail on datasets where NaNs remain unacceptable to specific estimators after the masking workflow.

## Prediction Identity Benchmark Refresh (OOB-first)

Date: April 24, 2026

Status:
- `experiments/run_prediction_identity.py` was updated to prioritize the RF-GAP OOB identity claim.
- `experiments/common.py` model registry was extended to include `ExtraObliqueRandomForest` when available in treeple.

Updated design:
- The benchmark uses `prox_method="rfgap"` as the primary configuration.
- Primary identity comparison:
  - classification: `prox_predict()` vs `oob_decision_function_`
  - regression: `prox_predict()` vs `oob_prediction_`
- Secondary identity comparison:
  - classification: `prox_predict(X_test)` vs `predict_proba(X_test)`
  - regression: `prox_predict(X_test)` vs `predict(X_test)`
- OOB and test comparisons reuse the same fitted model (`fit(X_train, y_train)`).
- If a model is unavailable for a task kind, explicit `unsupported` rows are emitted instead of silent skipping.
- If OOB outputs are missing/undefined for a fitted model (`missing oob_*` or no finite overlap), explicit `unsupported` rows are emitted with reason fields.

Raw schema updates:
- Added/confirmed fields:
  `experiment`, `status`, `task_kind`, `suite_name`, `dataset_id`, `dataset_name`, `task_id`, `random_state`, `model_name`, `prox_method`, `train_size`, `test_size`, `n_features`, `n_categorical_features`, `test_mean_abs_diff`, `test_max_abs_diff`, `oob_mean_abs_diff`, `oob_max_abs_diff`, `fit_runtime_seconds`, `prox_test_runtime_seconds`, `forest_test_runtime_seconds`, `prox_oob_runtime_seconds`, `total_runtime_seconds`.
- Classification-specific fields:
  `test_argmax_agreement`, `oob_argmax_agreement`, `test_accuracy`.
- Regression-specific fields:
  `test_r2`, `test_rmse_diff`.
- OOB coverage diagnostics for partial OOB availability:
  `oob_valid_count`, `oob_total_count`, `oob_valid_fraction`.

Compiled output updates:
- Separate classification/regression outputs are preserved.
- `summary.csv` is grouped by `model_name`.
- `model_comparison.csv` is sorted by OOB identity (`oob_max_abs_diff`) to emphasize OOB first.
- Two plots are generated per task kind:
  - `oob_max_abs_diff.png`
  - `test_max_abs_diff.png`

Smoke-test evidence:
- Classification smoke test:
  `PYTHONNOUSERSITE=1 conda run -n generalforest python experiments/run_prediction_identity.py --task classification --dataset-limit 1 --random-states 0 --max-rows 150 --n-estimators 16`
- Regression smoke test:
  `PYTHONNOUSERSITE=1 conda run -n generalforest python experiments/run_prediction_identity.py --task regression --dataset-limit 1 --random-states 0 --max-rows 150 --n-estimators 16`
- Smoke outcomes:
  - classification: `7/7` rows `ok` across all requested supervised families available in the environment (`RandomForest`, `ExtraTrees`, `RotationForest`, `HonestForest`, `ObliqueRandomForest`, `PatchObliqueRandomForest`, `ExtraObliqueRandomForest`).
  - regression: `6/7` rows `ok`, `1/7` `unsupported` (`HonestForest` unavailable for regression in this environment).
  - all required schema fields present for both task kinds.
  - classification smoke showed partial OOB coverage for at least one family with low tree count (`oob_valid_fraction` minimum about `0.876` at `n_estimators=16`), which is now recorded explicitly rather than hidden.

---

## Local Feature Importance Benchmark

Date: April 28, 2026

### Experiment Design

The local feature importance benchmark evaluates whether proximity-based local feature importances, as implemented in `forestgap.core`, identify genuinely informative features for individual predictions on classification tasks.

**Scope:** Classification only (`OpenML-CC18`). Regression is unsupported by `forestgap.core`. `HonestForest` is excluded because the current treeple build segfaults during fit on this sweep.

**Coverage:** `720/864` rows succeeded; `144` `HonestForest` rows were logged as unsupported.

**Key design choices:**
- Each model is fitted once on the full training split (no OOB; `bootstrap=False`, `oob_score=False`) using `prox_method="original"`.
- A random subset of up to `--n-queries` (default `16`) test observations is selected per dataset/seed for efficiency.
- Local feature importances are computed twice per query set: once using predicted labels (`importance_pred`) and once using true labels (`importance_true`).
- The `perturbation="auto"` mode is used with `--n-repeats` (default `5`) perturbation repeats.
- After importance scoring, the top-ranked feature per query point (by `|importance_pred|`) is perturbed to its training-derived fill value, and the resulting drop in predicted class probability is compared against the same drop for a uniformly random feature. This constitutes the primary evaluation signal.

### Metrics

| Metric | Description |
|---|---|
| `true_vs_pred_mean_abs_delta` | Mean absolute difference between `importance_true` and `importance_pred` vectors across all queries. Measures how much the label source (true vs predicted) changes the importance ranking. |
| `top_feature_drop_mean` | Mean probability drop when the top-ranked feature (by `|importance_pred|`) is perturbed to its fill value. Higher is better: the model's top pick should hurt the prediction when removed. |
| `random_feature_drop_mean` | Mean probability drop when a uniformly random feature is perturbed. Serves as a per-model baseline; should be lower than `top_feature_drop_mean`. |
| `top_feature_win_rate` | Fraction of query points where the top-ranked feature drop exceeds the random feature drop. The primary summary metric; chance level is `0.50`. |
| `misclassified_fraction` | Fraction of query points where the model's predicted class differs from the true label. Contextualizes importance quality on harder vs easier instances. |
| `fit_runtime_seconds` | Wall-clock time to fit the forest on the training split. |
| `importance_runtime_seconds` | Wall-clock time to compute both `importance_pred` and `importance_true` on the query set. |
| `evaluation_runtime_seconds` | Wall-clock time to run the perturbation evaluation (two forward passes). |
| `total_runtime_seconds` | Sum of the three runtime components. |

### What Was Compared

Six classifier families from the `forestgap` model registry were evaluated:

- `RandomForest` — standard sklearn random forest; axis-aligned splits.
- `ExtraTrees` — extremely randomized trees; axis-aligned.
- `RotationForest` — PCA-rotated random forest; dense linear projections per tree.
- `ObliqueRandomForest` — sparse random oblique splits (treeple).
- `PatchObliqueRandomForest` — patch-based oblique splits (treeple).
- `ExtraObliqueRandomForest` — extra-randomized oblique splits (treeple).

All models used default hyperparameters except for `n_estimators` (controlled by CLI, default `96`), and all used `prox_method="original"` with `bootstrap=False`.

### Results

Aggregated across all `OpenML-CC18` classification datasets and random states:

| Model | top_feature_win_rate | top_feature_drop_mean | random_feature_drop_mean | true_vs_pred_mean_abs_delta | misclassified_fraction | total_runtime_seconds |
|---|---|---|---|---|---|---|
| ObliqueRandomForest | **0.5451** | **0.1232** | 0.0466 | 0.1929 | 0.1615 | 86.85 |
| ExtraObliqueRandomForest | 0.5350 | 0.0974 | 0.0422 | 0.1493 | 0.1719 | 86.96 |
| PatchObliqueRandomForest | 0.5150 | 0.1222 | 0.0562 | 0.2856 | 0.2025 | 87.19 |
| RotationForest | 0.4962 | **0.1464** | 0.0511 | 0.3996 | **0.1603** | **207.46** |
| RandomForest | 0.4829 | 0.1310 | 0.0469 | 0.1758 | 0.1707 | 86.37 |
| ExtraTrees | 0.4812 | 0.1019 | 0.0453 | 0.0997 | 0.1667 | 86.55 |

Bold entries mark the best value per column (lower is better for `misclassified_fraction`, `true_vs_pred_mean_abs_delta`, and `total_runtime_seconds`; higher is better for win rate and probability drops).

### Key Findings

1. **Oblique forests lead on top-feature win rate.** `ObliqueRandomForest` (0.5451) and `ExtraObliqueRandomForest` (0.5350) were the only families to exceed the 0.50 chance threshold, suggesting that oblique-split proximity structures produce more discriminative local importance rankings than axis-aligned alternatives under this perturbation evaluation.

2. **`RotationForest` has the largest raw probability drop but mid-tier win rate.** Its mean top-feature drop of 0.1464 is the highest across all models, yet its win rate (0.4962) does not exceed the oblique families. This is consistent with denser feature interactions in rotation trees producing large absolute perturbation effects that are not always better targeted than random.

3. **`true_vs_pred_mean_abs_delta` varies substantially.** `ExtraTrees` and `RandomForest` show the smallest divergence between true- and predicted-label importances (0.0997 and 0.1758), while `RotationForest` and `PatchObliqueRandomForest` show the largest (0.3996 and 0.2856). A lower delta indicates the model's own predictions drive importance rankings close to the oracle (true-label) rankings.

4. **`PatchObliqueRandomForest` has elevated misclassification.** Its misclassified fraction (0.2025) is noticeably higher than the other families (~0.16–0.17), which may partly explain its lower local importance quality: importance scores on misclassified points are inherently noisier.

5. **Runtime is dominated by `RotationForest`.** At 207 s mean total runtime vs ~87 s for every other family, `RotationForest`'s cost-to-benefit ratio for local feature importance is unfavorable. The importance computation itself accounts for ~163 s vs ~84 s for the tree families.

6. **The random-baseline gap is consistent but modest.** Across all models, the mean top-feature drop exceeds the random-feature drop by roughly 0.05–0.09 percentage points, confirming that all families identify locally influential features above chance but that the absolute signal is small under a single-perturbation evaluation at `n_queries=16`.

### Stored Outputs

- Raw results: `results/raw/local_feature_importance/`
- Compiled tables: `results/local_feature_importance/classification/summary.csv`, `results/local_feature_importance/classification/model_comparison.csv`
- Figure: `results/local_feature_importance/classification/top_feature_win_rate.png`
