# Preserve RotationForest RF-GAP and extend path relations

## Summary

Keep RotationForest OOB and RF-GAP support as a first-class 1.0 feature, using the established aeon-compatible OOB adapter. Its contract is: reconstruct aeon’s per-tree resampling/OOB roles, use the matching refitted OOB trees for training-row RF-GAP, and use the fitted RotationForest trees for test-row prediction weights. Both OOB and test-set proximity predictions must match the corresponding forest output.

Enable CCF-Bag OOB/RF-GAP and add path-based relations wherever an adapter exposes a stable decision-path and node-depth contract.

## Capability and implementation changes

- Retain and harden RotationForest `oob`/`rfgap` support:
  - Keep the per-tree OOB mask, in-bag multiplicities, surrogate leaves, and original-tree query-weight path.
  - Derive resampling/OOB state from aeon’s training-estimate machinery using synchronized seeds; fail explicitly when that private contract is unavailable.
  - Publish the reconstructed `oob_decision_function_` / `oob_prediction_` on both the `Proximity` wrapper and wrapped estimator, with valid-row metadata for samples lacking OOB trees.
  - Document this as the project’s RotationForest-compatible RF-GAP adapter, not as an assumption that aeon persists `estimators_samples_`.

- Extend CCF:
  - Add `oob_score=False` to native CCF estimators; for `bootstrap=True, oob_score=True`, calculate standard classifier/regressor OOB outputs from persisted `estimators_samples_`.
  - Advertise `oob` and `rfgap` only for CCF-Bag; default non-bagged CCF remains ineligible.
  - Add CCF node-depth support for its internal node representation so `random_height` can operate without assuming sklearn’s `tree_` layout.

- Add a RotationForest decision-path adapter:
  - Reapply each stored rotation, obtain each base tree’s decision-path indicator, concatenate indicators, and return sklearn-style tree offsets.
  - Enable `path` and `random_height` for RotationForest once this adapter is available.

- Apply a uniform path contract in the registry:
  - Enable `path` when an estimator exposes a stable `(indicator, node_ptr)` decision-path representation.
  - Enable `random_height` only when per-tree node depths can be obtained from either sklearn trees or an adapter-specific depth provider.
  - Cover RandomForest, bootstrap ExtraTrees, RotationForest, CCF, treeple oblique/honest forests, quantile forests, and synthetic-null forests when their runtime interfaces meet those gates.
  - Retain native-only restrictions for isolation forests and method-specific restrictions for survival, causal, Bayesian, boosted, and structured-object methods.

## Roadmap updates

Revise the method matrix and relation table to state:

- RandomForest: original, OOB, RF-GAP, KeRF, path, random-height.
- ExtraTrees: OOB/RF-GAP only with bootstrap; path relations when the fitted path interface is present.
- RotationForest: original, OOB, RF-GAP, KeRF, path, random-height through the aeon adapter.
- CCF: original/KeRF/path/random-height generally; OOB/RF-GAP only for CCF-Bag.
- treeple oblique forests: original/OOB/RF-GAP and path relations when fitted bootstrap/path contracts are present; HonestForest uses `honest_gap` for prediction identity.
- Quantile, survival, isolation, synthetic-null, boosted, causal, Bayesian, and structured methods: retain their existing method-specific limitations, with explicit reasons that ordinary RF-GAP cannot reconstruct their native targets or lacks the required OOB state.

Update the prediction-identity acceptance criterion: every enabled RF-GAP adapter must match native test prediction for external queries and native/reconstructed OOB prediction on rows with at least one OOB tree.

## Test plan

- RotationForest optional tests, for both classifier and regressor:
  - `prox_predict(X_test)` equals `predict_proba(X_test)` / `predict(X_test)`.
  - `prox_predict()` equals `oob_decision_function_` / `oob_prediction_` on valid OOB rows.
  - OOB-only rows, seed determinism, and unavailable aeon internals are handled explicitly.
  - `path` and `random_height` are symmetric, PSD where applicable, deterministic, and extend to test rows.

- CCF tests:
  - Default CCF rejects OOB/RF-GAP.
  - CCF-Bag accepts them and matches test and OOB predictions for classification and regression.
  - CCF path/random-height kernels pass sparse/dense, PSD, deterministic-cut, and extension checks.

- Shared contract tests:
  - Preserve RandomForest, bootstrap ExtraTrees, and treeple oblique RF-GAP identity coverage.
  - Parameterize path/random-height tests over every available adapter that advertises those relations.
  - Update method-matrix and CLI tests so CCF-Bag RF-GAP cells are successful and RotationForest RF-GAP cells remain supported rather than downgraded.

## Assumptions

- RotationForest compatibility follows the prior project behavior and aeon’s training-estimate interface; no public API rename is introduced.
- Existing unrelated notebook edits and the untracked roadmap PDF remain untouched.
