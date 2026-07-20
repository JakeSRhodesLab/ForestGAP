# Migrating to forestgap 1.0

Version 1.0 removes dynamic estimator subclasses completely. There is no
compatibility shim: create the estimator normally, then give its instance to
`Proximity`.

| Removed 0.x surface | 1.0 equivalent |
|---|---|
| `with_sparse_proximity(ForestClass)(..., prox_method=name)` | `Proximity(ForestClass(...), relation=name)` |
| `SparseProximityMixin` | `Proximity` |
| `forestgap.core` imports | `forestgap.proximity`, `forestgap.adapters`, or `forestgap.applications` |
| `model.fit(X, y)` on a generated wrapper | `Proximity(forest, relation=...).fit(X, y)` |
| `model.prox_fit(X, y)` | `Proximity(fitted_forest, relation=...).attach(X, y)` |
| `model.predict(X)` | `proximity.estimator.predict(X)` (or `proximity.predict(X)` delegation) |
| `model.apply(X)` | `proximity.estimator.apply(X)` (or adapter-aware `proximity.apply(X)`) |
| `model.validate_data(...)` | Estimator-native validation; this accidental wrapper surface is not replaced |
| `model.get_proximities()` | `proximity.get_proximities()` / `get_relation()` |
| `model.prox_extend(X)` | `proximity.prox_extend(X)` / `extend_relation()` |
| `model.prox_predict(X)` | `proximity.prox_predict(X)` / `prediction_identity(proximity, X)` |
| `model.impute(...)` | `impute(proximity, ...)` |
| `model.outlier_scores(...)` | `outlier_scores(proximity, ...)`; isolation models use `native_anomaly_scores(forest, X)` |
| `model.local_feature_importance(...)` | `local_feature_importance(proximity, ...)` |
| `model.local_variable_importance(...)` | `local_feature_importance(proximity, ...)` |
| `model.get_prototypes(...)` | `get_prototypes(proximity, ...)` |
| `model.plot_mds(...)` | `plot_mds(proximity, ...)` |
| `model.embed(...)` / `model.phate(...)` | `embed(proximity, method=...)` |
| `model.proximity_embedding(...)` | `embed(proximity, method=...)` |
| `model.embed_transform(...)` / `model.phate_transform(...)` | `embed_transform(proximity, X, method=...)` |

Estimator predictions and attributes now remain on the estimator:

```python
forest = RandomForestClassifier(random_state=0)
proximity = Proximity(forest, relation="original").fit(X_train, y_train)
predictions = proximity.estimator.predict(X_test)
```

Invalid relation requests now raise `UnsupportedRelationError` instead of
attempting a best-effort bootstrap interpretation. In particular, isolation,
boosted, Bayesian, causal, and structured forests never inherit generic RF-GAP
semantics. Automated runners should catch that exception and record an
`unsupported` row rather than drop the cell.

## 1.1 boosted-tree additions

Version 1.1 adds explicit adapters for sklearn GradientBoosting, XGBoost,
LightGBM, and CatBoost. Use `relation="boosted"` for the normalized empirical
leaf-contribution-variance kernel; `original` and `kerf` remain available, and
path-derived relations are available when the backend exposes stable paths.
CatBoost restricts those path-derived relations to `grow_policy="SymmetricTree"`.
Optional backends are installed with `pip install '.[boosted]'`.

This does not relax the RF-GAP validity gate. Booster subsampling and CatBoost
bootstrap modes are not the exchangeable bootstrap membership/prediction-weight
contract required by `oob` and `rfgap`, so both relations remain explicitly
invalid for all boosted families.
