# Experiment Prompts


## CLI Setup
codex --model gpt-5.4 --config model_reasoning_effort='"high"' --yolo

## Original Prompt
Please review the codebase in ObliqueProximitiesCode/forestgap. Review the proximity-based applications in ObliqueProximitiesCode/notes/proximity_applications_summary.md. For any code runs, please use the conda environment generalforest. The
  ObliqueProximitiesCode/forestgap/core.py file contains the main code base for running the proximity-based applications. The ObliqueProximitiesCode/forestgap/datasets.py file contains code for loading in OpenML Benchmarks. Use the OpenML-CC18
  benchmark for classification, and the OpenML-CTR23 benchmark for regression tasks. Now, devise experiments based on both of these benchmarks (separate for regression and classification) to test the applications spelled out in
  ObliqueProximitiesCode/notes/proximity_applications_summary.md. Most of these task involve supervision, including imputation, outlier detection, prototypes, and local feature importance. In the ObliqueProximitiesCode/experiments directory, create
  scripts to test each application found. Each experiment should be reproducible and run with multiple random states. Raw results should be stored in ObliqueProximitiesCode/results/raw. Compiled results (e.g., figures, tables, plots) should be
  stored in respective directories of ObliqueProximitiesCode/results. Use as many threads as possible. Continue refining experiments until all run on all benchmark datasets. If any applications in ObliqueProximitiesCode/forestgap/core.py need to be
  updated, please do so. Only work on the current branch (create_experiments). Any critical results or continued questions should be recorded in respectively ObliqueProximitiesCode/results/results_summary.md and ObliqueProximitiesCode/results/
  questions.md. Reiterate through your processes as often as necessary.


## Imputation Prompt:

codex --model gpt-5.4 --config model_reasoning_effort='"high"' --yolo

Please update the imputation benchmark in ObliqueProximitiesCode/experiments/run_imputation.py and any shared helpers in ObliqueProximitiesCode/experiments/common.py as needed. For any code runs, use the conda environment generalforest. Only work on the current branch.

First review:
- ObliqueProximitiesCode/experiments/run_imputation.py
- ObliqueProximitiesCode/experiments/common.py
- ObliqueProximitiesCode/notes/proximity_applications_summary.md

Current-state constraints to preserve:
- Keep the existing proximity configurations:
  - original + single_pass
  - oob + single_pass
  - rfgap + single_pass
  - rfgap + iterative
- Keep the same OpenML benchmark split:
  - OpenML-CC18 for classification
  - OpenML-CTR23 for regression
- Keep reproducibility through random states and the existing results directory layout.

Required experiment expansion:

1. Missingness mechanisms and rates
- Expand the experiment to evaluate three missingness mechanisms:
  - MCAR
  - MAR
  - MNAR
- Expand the experiment to evaluate the following missingness fractions:
  - 0.05
  - 0.10
  - 0.25
  - 0.50
  - 0.75
- Add CLI options such as --missingness-mechanisms and --missing-rates, with the defaults above.
- For a given dataset / task / random_state / missingness_mechanism / missing_rate, generate one train mask and one test mask and reuse those exact masks across all compared proximity methods and variants.

2. Explicit mechanism definitions
Implement the mechanisms in shared helpers in experiments/common.py and document them clearly in code and results_summary.md.

Use these definitions:
- MCAR:
  uniform random masking among currently observed entries, calibrated to the requested rate per feature.
- MAR:
  missingness probability for feature j must depend only on other observed features, not on x_j itself.
  Use training-derived parameters only.
  A reasonable implementation is a per-feature logistic missingness score based on one or a few other standardized columns, then calibrate probabilities so the realized missingness rate matches the target among originally observed cells.
- MNAR:
  missingness probability for feature j must depend on x_j itself.
  For numeric features, use x_j directly through a standardized value or quantile-rank-based logistic score.
  For categorical features, use category-specific missingness probabilities.
  Calibrate to the requested masking rate among originally observed cells.

Handle degenerate cases gracefully:
- all-missing columns
- constant columns
- no usable conditioning column for MAR
- very low-support categorical levels
If fallback behavior is needed, use a documented fallback such as MCAR and record that in the result metadata.

3. Imputation-quality evaluation
Keep evaluation on masked cells only.

Report at least:
- numeric_mse as the primary numeric imputation metric
- numeric_rmse if helpful
- categorical_accuracy for categorical masked cells
- n_train_numeric_masked
- n_test_numeric_masked
- n_train_categorical_masked
- n_test_categorical_masked

Store train and test metrics separately.

4. Downstream-model evaluation
Add a second evaluation stage after imputation.

For each imputed train/test pair:
- fit simple downstream models that are agnostic to the forest method
- train on imputed X_train and original y_train
- evaluate on imputed X_test and original y_test

Use:
- classification:
  - LogisticRegression
  - KNeighborsClassifier
- regression:
  - Lasso
  - KNeighborsRegressor

Use sklearn pipelines with scaling where appropriate so the downstream comparison is stable.

Report at least:
- classification:
  - test_accuracy
  - optionally roc_auc when feasible
- regression:
  - test_mse
  - test_r2

Also include an oracle downstream baseline:
- fit the same downstream models on the original uncorrupted prepared.X_train / prepared.X_test
- store those oracle metrics so we can quantify the loss induced by missingness plus imputation

5. Result schema and summaries
Update raw results so each row includes at least:
- experiment
- status
- task_kind
- suite_name
- dataset_id
- dataset_name
- task_id
- random_state
- model_name
- prox_method
- variant
- missingness_mechanism
- missing_rate
- downstream_model
- all imputation metrics
- all downstream metrics
- fit_runtime_seconds
- impute_runtime_seconds
- downstream_runtime_seconds
- total_runtime_seconds
- any fallback metadata for MAR/MNAR construction if applicable

Update compile_outputs so summaries are grouped by:
- model_name
- prox_method
- variant
- missingness_mechanism
- missing_rate
- downstream_model

Also create higher-level comparison tables and figures separately for classification and regression:
- imputation metric vs missingness rate, split by mechanism
- downstream metric vs missingness rate, split by mechanism

6. Implementation guidance
- Keep helper logic in experiments/common.py rather than embedding all new logic in run_imputation.py.
- Preserve current coding style and result conventions.
- Make sure the updated experiment still runs for both task kinds.
- Smoke-test on a small dataset_limit for both classification and regression.
- Record any unresolved issues or benchmark blockers in ObliqueProximitiesCode/results/questions.md.
- Record the exact experimental design, especially the MAR and MNAR constructions, in ObliqueProximitiesCode/results/results_summary.md.

Deliverables:
- updated run_imputation.py
- any new or updated helper functions in common.py
- updated raw/compiled output schema
- smoke-test evidence
- concise documentation of the new benchmark design


## Outlier Detection

codex --model gpt-5.4 --config model_reasoning_effort='"high"' --yolo

Please update the outlier detection benchmark in ObliqueProximitiesCode/experiments/run_outlier_detection.py and any shared helpers in ObliqueProximitiesCode/experiments/common.py as needed. For any code runs, use the conda environment generalforest. Only work on the current branch.

First review:
- ObliqueProximitiesCode/experiments/run_outlier_detection.py
- ObliqueProximitiesCode/experiments/common.py
- ObliqueProximitiesCode/forestgap/core.py, especially the outlier scoring path
- ObliqueProximitiesCode/notebooks/rotation_forest_proximities.ipynb
- ObliqueProximitiesCode/notebooks/isolation_forest_outlier_comparison.ipynb

Important notebook-alignment requirements:
- Mirror the import / wrapping patterns already used in the notebooks.
- Use `with_sparse_proximity(...)` from `forestgap`.
- For sklearn isolation models, native anomaly scores should be `-score_samples(X)` so that larger means “more anomalous”.
- For proximity-based unsupervised scores, use `model.outlier_scores()` with no labels.
- For supervised forests, preserve the option to use label-aware proximity outlier scores where appropriate.
- Follow the notebook fitting conventions:
  - supervised models fit with `fit(X, y)`
  - unsupervised models fit with `fit(X)`
  - IsolationForest / ExtendedIsolationForest use explicit bootstrap handling
  - UnsupervisedRandomForest / UnsupervisedObliqueRandomForest enable bootstrap for `oob` / `rfgap` proximity modes when needed

Current-state goals:
- Keep the synthetic corruption benchmark because it gives known outlier labels and allows fair comparison across supervised and unsupervised methods.
- Do not assume supervised and unsupervised methods need different detection metrics unless the code or benchmark structure really requires it.
- If different metric sets are justified, explain why in results_summary.md.

Required experiment expansion:

1. Model families
Expand the outlier benchmark so it cleanly separates:
- supervised forest models:
  - keep the currently benchmarked supervised models already supported in the experiments codebase, such as RandomForest, RotationForest, HonestForest, ObliqueRandomForest, PatchObliqueRandomForest, ExtraTrees, etc.
- unsupervised forest / anomaly models:
  - IsolationForest
  - ExtendedIsolationForest
  - UnsupervisedRandomForest
  - UnsupervisedObliqueRandomForest
Use notebook import patterns and skip gracefully with unsupported rows if optional dependencies are unavailable.

2. Score sources to compare
For each model, distinguish the source of the anomaly score.

For supervised forest models:
- proximity-based scores only
- evaluate both:
  - label-aware proximity outlier scores where meaningful
  - label-free proximity outlier scores where meaningful

For unsupervised models:
- proximity-based scores
- native model scores when the estimator exposes them

Specifically:
- IsolationForest:
  - native score: `-score_samples(X)`
  - proximity score(s): `outlier_scores()` for each proximity type
- ExtendedIsolationForest:
  - native score: `-score_samples(X)`
  - proximity score(s): `outlier_scores()` for each proximity type
- UnsupervisedRandomForest / UnsupervisedObliqueRandomForest:
  - proximity score(s) at minimum
  - only add a native score path if the estimator actually exposes one cleanly and it is meaningful

For proximity-based scoring, continue comparing:
- `original`
- `oob`
- `rfgap`

For proximity score methods, continue supporting:
- `breiman`
- `ene`

Do not duplicate native-score rows across proximity modes if the native score is invariant to that choice. If separate fits are needed because bootstrap or estimator settings differ, document that clearly.

3. Metrics
Use corruption labels as ground truth for the main benchmark.

Primary detection metrics for all methods:
- `roc_auc`
- `average_precision`

Secondary detection metrics:
- `precision_at_k`, where `k` equals the number of injected outliers
- optionally `recall_at_k` and `f1_at_k` if helpful

Additional comparison metrics for models that have both native and proximity scores:
- `spearman_rank_corr`
- `topk_overlap_at_k`, where `k` equals the number of injected outliers
- optionally overlap at several fixed `k` values

Use the same ground-truth detection metrics for supervised and unsupervised methods unless there is a compelling reason not to. If you decide the metric sets should differ, justify that decision in ObliqueProximitiesCode/results/results_summary.md.

4. Supervised vs unsupervised scoring rules
Classification tasks:
- supervised forest models may evaluate both label-aware and label-free proximity outlier scores
- unsupervised models evaluate label-free scores
- native isolation scores should be compared against proximity-based scores for the same underlying model family when available

Regression tasks:
- do not force label-aware outlier scoring with continuous targets unless there is a strong code-level justification
- default regression to label-free scoring only unless you confirm a meaningful supervised formulation in the existing code

5. Corruption benchmark
Keep the current corruption-based benchmark structure, but make it more explicit and reproducible.

Requirements:
- for each dataset / random_state / outlier_fraction, generate one corrupted dataset and reuse it across all compared score sources
- preserve the current corruption-label evaluation path
- keep `--outlier-fraction`
- optionally add `--outlier-fractions` to support sweeps if that is easy to integrate cleanly

If you refine the corruption procedure, document exactly what changed and why.

6. Result schema
Update raw output rows so they include at least:
- experiment
- status
- task_kind
- suite_name
- dataset_id
- dataset_name
- task_id
- random_state
- model_name
- model_family
- score_source
- prox_method
- score_method
- label_mode
- outlier_fraction
- roc_auc
- average_precision
- precision_at_k
- recall_at_k if added
- f1_at_k if added
- spearman_rank_corr when applicable
- topk_overlap_at_k when applicable
- fit_runtime_seconds
- score_runtime_seconds
- total_runtime_seconds
- any metadata needed to explain native-score vs proximity-score comparability

Use a clean schema such as:
- `score_source` in `{native, proximity}`
- `score_method` in `{score_samples, breiman, ene}` or similar
- `label_mode` in `{supervised, unsupervised, not_applicable}`

7. Summaries and plots
Update compiled outputs so they do not collapse together incomparable score types.

Create separate summaries for:
- supervised forest models
- unsupervised models
- IsolationForest / ExtendedIsolationForest native-vs-proximity comparisons

Group summaries by at least:
- model_name
- score_source
- prox_method
- score_method
- label_mode

Create figures / tables that show:
- average precision by configuration
- roc auc by configuration
- IsolationForest native vs proximity comparison
- ExtendedIsolationForest native vs proximity comparison
- rank-agreement or top-k-overlap summaries where applicable

8. Implementation notes
- Keep helper logic in `experiments/common.py` rather than embedding everything into `run_outlier_detection.py`.
- Extend the current model registry or create a clean outlier-specific registry if that is the simplest way to support unsupervised models without breaking the other experiments.
- Preserve the current unsupported-model handling style.
- Use notebook-consistent class names and imports rather than inventing new wrappers.
- Continue using reproducible random states and the existing results directory conventions.

9. Validation and documentation
- Smoke-test the updated outlier benchmark on at least one classification dataset and one regression dataset.
- Confirm that supervised and unsupervised models both run when available.
- Record any blockers or unresolved issues in ObliqueProximitiesCode/results/questions.md.
- Record the final metric choices, including the reasoning for supervised vs unsupervised comparability, in ObliqueProximitiesCode/results/results_summary.md.

Deliverables:
- updated run_outlier_detection.py
- any new or updated helpers in common.py
- updated raw/compiled result schema
- smoke-test evidence
- concise documentation of the scoring design


## Local Feature Importance

codex --model gpt-5.4 --config model_reasoning_effort='"high"' --yolo

Please update the local feature importance benchmark in ObliqueProximitiesCode/experiments/run_local_feature_importance.py and any shared helpers in ObliqueProximitiesCode/experiments/common.py as needed. For any code runs, use the conda environment generalforest. Only work on the current branch.

First review:
- ObliqueProximitiesCode/experiments/run_local_feature_importance.py
- ObliqueProximitiesCode/experiments/common.py
- ObliqueProximitiesCode/forestgap/core.py, especially `local_feature_importance`
- ObliqueProximitiesCode/notes/proximity_applications_summary.md, especially Section 6
- ObliqueProximitiesCode/tests/test_prox.py, especially the local feature importance tests
- ObliqueProximitiesCode/papers/ExplicatingFeature.pdf

Current limitations to address:
- the benchmark is currently classification-only because `local_feature_importance()` is classifier-only in `forestgap.core`
- it only uses `prox_method="original"`
- it only uses `perturbation="auto"`
- it only samples a small test-query subset and evaluates mostly whether perturbing the top-ranked feature drops predicted probability more than perturbing a random feature
- it does not validate signal recovery on synthetic data, focal-class sensitivity, directionality, perturbation choice, or stability
- HonestForest is currently unstable here; keep unsupported/error rows explicit if that remains true

Goal:
Design a stronger validation suite for proximity-based local feature importance that is faithful to the Whitmore / ExplicatingFeature methodology, but also validates the generalized implementation actually present in `forestgap.core`. If the paper does not fully specify benchmark methodology, devise additional validation methods and document the reasoning.

Scope:
- Keep classification as the default supported scope unless you intentionally and safely extend `forestgap.core` to regression.
- If regression is not extended, continue logging explicit unsupported rows for regression rather than silently skipping it.

Methodology requirements:
1. Read the paper and the repo notes, then align the experiment with the implemented score:
   - importance is based on perturbing one feature at a time, recomputing proximity structure relative to a fixed training set, and measuring in-group vs out-group proximity-space shift
   - positive scores mean the observed feature value supports the focal class
   - negative scores mean it pushes away from the focal class
2. Treat `return_closeness=True` and `return_directionality=True` as first-class outputs for evaluation, not just optional extras.
3. Compare proximity constructions where supported:
   - `original`
   - `oob`
   - `rfgap`
4. Compare perturbation modes where meaningful:
   - `auto`
   - `flip`
   - `permute`
   - `quantile`
5. Compare focal-label choices:
   - predicted-label mode
   - true-label mode
   - for binary synthetic tasks, explicitly test alternate-label sign reversal when feasible

Use a two-track benchmark design.

Track A: synthetic validation with known local ground truth
Create controlled synthetic classification problems where the locally relevant features are known by construction. Include at least:
- a binary-signal dataset where one binary feature dominates class membership and `flip` perturbation should recover it
- a numeric monotone or threshold dataset where one ordered feature drives the focal class locally and `quantile` perturbation plus directionality should be informative
- a heterogeneous local-signal dataset where different regions of feature space depend on different features, so local explanations should vary by query rather than collapse to one global ranking

For the synthetic track, evaluate at least:
- top-1 true-signal hit rate
- top-k true-signal hit rate
- mean rank of the true signal feature(s)
- mean absolute importance on signal features vs noise features
- predicted-label vs true-label agreement when the prediction is correct
- focal-class sign reversal or near-sign-reversal in binary cases when the focal label is flipped
- directionality agreement on monotone numeric features:
  - verify that upward vs downward perturbation summaries reflect the designed direction of effect
- stability across seeds / repeats:
  - Spearman rank correlation or similar between feature rankings across repeated runs

Track B: OpenML benchmark faithfulness and stability
Keep the OpenML-CC18 classification benchmark, but make the evaluation stronger and more local-explanation-specific.

For each fitted model and query set:
- reuse the same fitted forest across perturbation modes and focal-label choices whenever possible
- reuse the same query subset across compared methods so differences are attributable to the explanation method rather than sample selection
- stratify query summaries by:
  - correctly classified queries
  - misclassified queries

For the benchmark track, evaluate at least:
- top-feature probability drop vs random-feature probability drop
- top-feature probability drop vs bottom-feature probability drop
- optionally top-k deletion curves, e.g. perturb top-1 / top-3 features vs random / bottom features
- class-switch rate after perturbing top-ranked features
- closeness-based faithfulness using `return_closeness=True`
  - top-ranked features should generally reduce in-group closeness and/or increase out-group closeness more than random or bottom-ranked features
- sensitivity to focal-class choice
  - compare predicted-label and true-label local importance
  - report this separately for correctly classified and misclassified queries
- stability across perturbation repeats and random seeds
  - use rank correlation, score correlation, or coefficient-of-variation style summaries
- perturbation-mode comparison
  - `flip` vs `permute` vs `quantile` vs `auto` where applicable
- optional secondary sanity checks
  - compare dataset-level mean absolute local importance against split-based feature importance and/or permutation importance
  - do not let these secondary comparisons replace the primary local validation metrics

Result schema:
Update raw results so they can represent both synthetic and benchmark validations. Include at least:
- experiment
- status
- evaluation_track in `{synthetic, openml}`
- synthetic_scenario when applicable
- task_kind
- suite_name
- dataset_id
- dataset_name
- task_id
- random_state
- model_name
- prox_method
- perturbation
- label_mode in `{predicted, true, alternate, not_applicable}`
- query_group in `{all, correct, misclassified}` where applicable
- n_queries
- n_features
- top1_hit_rate when applicable
- topk_hit_rate when applicable
- mean_rank_true_signal when applicable
- signal_vs_noise_gap when applicable
- sign_flip_error or sign_flip_agreement when applicable
- directionality_agreement when applicable
- top_feature_drop_mean
- random_feature_drop_mean
- bottom_feature_drop_mean if added
- top_feature_win_rate
- class_switch_rate if added
- in_group_closeness_change metrics
- out_group_closeness_change metrics
- predicted_vs_true_delta metrics
- stability metrics
- fit_runtime_seconds
- importance_runtime_seconds
- evaluation_runtime_seconds
- total_runtime_seconds

Compiled outputs:
Create separate summaries and figures for:
- synthetic signal-recovery validation
- OpenML faithfulness validation
- focal-label sensitivity
- perturbation-mode comparison
- directionality validation
- stability across repeats / seeds

Implementation guidance:
- Keep shared evaluation logic in `experiments/common.py` when appropriate.
- Preserve the existing result directory structure and CSV/plot conventions.
- Keep HonestForest explicitly unsupported if it still segfaults or fails unreliably.
- Do not hide failures; log them as unsupported/error rows.
- Use the existing tests as the minimum behavioral floor, then add experiment-level validation beyond them.

Documentation:
- Record the final evaluation design in ObliqueProximitiesCode/results/results_summary.md.
- If the paper leaves evaluation ambiguous, explain what additional validation criteria you introduced and why.
- Record unresolved issues or blockers in ObliqueProximitiesCode/results/questions.md.

Validation:
- Smoke-test the updated benchmark on at least one synthetic scenario and a small OpenML classification slice.
- Confirm that the new result schema, summaries, and plots are generated successfully.

Deliverables:
- updated run_local_feature_importance.py
- any new or updated helpers in common.py
- updated raw/compiled result schema
- smoke-test evidence
- concise documentation of the revised validation methodology
