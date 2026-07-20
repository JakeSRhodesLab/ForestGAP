# Add validated boosted-tree proximity support

## Summary

Add sklearn GradientBoosting, XGBoost, LightGBM, and CatBoost classifier/regressor adapters. All expose `original`, `kerf`, and a new `boosted` relation; path-capable backends additionally expose `path`, `englund`, and `random_height`. `oob` and `rfgap` remain explicitly invalid for every boosted model.

## Adapter and relation implementation

- Use the reviewed ForestGeom adapters as the semantic reference for `boosted`: calculate each tree’s fitted leaf contribution across attached training data, set its weight proportional to empirical contribution variance, normalize weights to sum to one, and fall back to uniform weights when all variances are zero. Build the kernel as `Σ_t weight_t · 1[same terminal leaf]` via `sqrt(weight_t)` on each sparse factor.
- Add a common boosted-adapter contract for native leaf IDs, tree count, node paths/depths, and contribution weights.
  - Flatten sklearn GradientBoosting’s stage/class tree layout.
  - Use XGBoost/LightGBM native leaf inference plus cached model dumps to map terminal leaves to root paths and leaf values.
  - Use CatBoost `calc_leaf_indexes`, which returns a native two-dimensional per-tree leaf-index matrix, preserving DataFrame/Pool categorical input rather than coercing it to NumPy. For scalar models, use leaf-value variance; for multiclass CatBoost, use the sum of per-class variances (the covariance trace) as the tree’s scalar weight. [CatBoost API](https://catboost.ai/docs/en/concepts/python-reference_catboost_calc_leaf_indexes)
- Add `boosted` to `Proximity`’s public relation type, sparse-factor construction, extension path, diagnostics, and metadata. Publish normalized `tree_weights_` and `tree_weighting="training_leaf_contribution_variance"`; keep `prediction_identity=False`.
- Generalize current path/depth code to consume the adapter contract instead of assuming sklearn `estimators_` and `decision_path`.
- CatBoost supports `original`, `kerf`, and `boosted` for all fitted classifier/regressor models. Enable `path`, `englund`, and `random_height` only for `grow_policy="SymmetricTree"`; reconstruct paths from the native leaf code and symmetric tree depth. Reject these path-derived relations for `Depthwise`/`Lossguide` with a clear capability reason. CatBoost’s default policy is `SymmetricTree`. [CatBoost grow-policy reference](https://catboost.ai/docs/en/references/training-parameters/common)
- Keep `oob`/`rfgap` unavailable even if a booster uses subsampling or a CatBoost bootstrap type: those do not supply the exchangeable bootstrap membership and prediction-weight contract required by forestgap’s relations.

## Registry, packaging, and docs

- Replace the single supported-model gap with distinct `gradient_boosting`, `xgboost`, `lightgbm`, and `catboost` registry rows; retain unrelated boosted families as gated.
- Add `boosted = ["lightgbm>=4.3", "xgboost>=2.0", "catboost>=1.2"]`, use lazy imports, update the lockfile, install the extra in CI, and bump the package version to `1.1.0`.
- Update the README relation list, installation instructions, applicability table, and limitations text to include all four backends and the new `boosted` kernel.
- Update roadmap/migration notes to show the four implemented boosted families and preserve the explicit prohibition on ordinary RF-GAP for boosting.
- Register the adapters in the experiment harness, record all three optional package versions, and make optional backends selectable without requiring them in the base-install default matrix.

## Test plan

- Test binary classification, multiclass classification, and regression for all four backend families.
- Verify capability discovery, lazy optional imports, and rejection of `oob`, `rfgap`, and CatBoost non-symmetric path relations.
- Compare `original`, `kerf`, and `boosted` matrices to direct leaf-collision calculations; verify finite, non-negative, normalized weights and the all-zero-variance fallback.
- Verify `path`, `englund`, and `random_height` for supported backends: native leaf/path consistency, sparse/dense equality, symmetry/PSD where applicable, seeded random-height behavior, and query-to-training extension.
- Add a CatBoost categorical DataFrame/Pool case to ensure adapter indexing preserves native feature typing, and a non-symmetric CatBoost case to validate relation gating.
- Run the full suite, static checks, method-matrix boosted smoke cells, and distribution build.

## Assumptions

- “All possible” means relations with defensible boosted-tree semantics, not pseudo-OOB or pseudo-RF-GAP variants.
- CatBoost path semantics are deliberately limited to its symmetric-tree contract; its leaf-based relations remain available for other CatBoost growth policies.
- Preserve the existing untracked `README.pdf`; it is unrelated to this work.
