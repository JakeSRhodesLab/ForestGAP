## Open Questions

1. `HonestForest` is the main unstable family in this environment.
It is classification-only here, throws class-balance `ValueError`s on several capped CC18 tasks, and segfaults during the local feature importance sweep.
Should `HonestForest` stay in the benchmark matrix with explicit unsupported/error rows, or should it be dropped from the applications it cannot run reliably?

2. `HonestForest` also breaks the expected prediction-identity behavior.
Its classification mean test max absolute difference in `prediction_identity` is `2.43e-01`, while every other supported family is effectively at machine precision.
Is that a known property of the honest variant under the proximity wrapper, or does `prox_predict()` need a model-specific fix there?

3. Imputation still has a small number of hard failures on `task_3021` and `task_361268`.
Those failures come from NaNs reaching estimators that do not accept them, and from a few `HonestForest` class-balance failures on tiny splits.
Should the imputation pipeline explicitly drop fully missing columns or add another deterministic fill step before estimator fitting?

4. `get_prototypes()` and `local_feature_importance()` are still classifier-only in `forestgap.core`.
Do you want those APIs extended to regression, or is classification-only the intended scope?

5. `RotationForest` is often competitive in quality but dominates runtime across multiple applications.
Should I add a second, higher-budget but smaller benchmark slice for quality-focused comparisons and keep the current capped full sweep as the throughput/stability baseline?

6. `plot_mds()` still emits sklearn `FutureWarning`s because `MDS(n_init)` will change in sklearn `1.9`.
Should I patch `forestgap/core.py` to set `n_init` explicitly and optionally record MDS stress in the compiled outputs?

7. The imputation benchmark code has been expanded and smoke-tested, but the full CC18/CTR23 imputation sweep has not been rerun yet with the new MCAR/MAR/MNAR grid.
Do you want the repository’s main `results/imputation/*` outputs regenerated now, or should the redesign stay code-only until the broader benchmark rerun is scheduled?

8. The prototype benchmark redesign is now smoke-tested, but the full `OpenML-CC18` prototype sweep has not been rerun yet with the expanded metrics and random-center baselines.
Do you want the repository’s main `results/prototypes/*` outputs regenerated now, or should the redesign stay code-only until the broader rerun is scheduled?

9. `HonestForest` does not currently provide clean OOB probabilities for the prototype representativeness diagnostic in this environment; those rows now fall back to unavailable OOB confidence.
Should that family get a model-specific confidence fallback, or is the explicit unavailability marker sufficient?

10. For the refreshed prediction-identity benchmark, should the full benchmark run keep `n_estimators` high enough (for example 128+) to reduce partial OOB rows?
The smoke run at `n_estimators=16` produced valid but partial classification OOB coverage for at least one model family (`oob_valid_fraction` down to about `0.876`).

11. `ExtraObliqueRandomForest` is now included in the supervised model registry when available.
Do you want it added to historical comparisons in older compiled result snapshots as well, or only in newly generated benchmark outputs?

12. Regression `HonestForest` remains unavailable in the current environment and is logged as explicit `unsupported` rows.
Should we keep that behavior for all future sweeps, or gate `HonestForest` out of regression requests entirely to reduce unsupported-row volume?
