# Proximity manifold applications (implemented in 1.0)

This historical plan is superseded by the composition implementation in
`forestgap.proximity`, the shared conversions in `forestgap.manifold`, and the
standalone visualization functions in `forestgap.applications.visualization`.

The implemented pattern is:

```python
from sklearn.ensemble import RandomForestClassifier
from forestgap import Proximity, embed, embed_transform

forest = RandomForestClassifier(n_estimators=300, random_state=0)
proximity = Proximity(forest, relation="original").fit(X_train, y_train)

embedding = embed(proximity, method="isomap", n_components=2)
projected = embed_transform(proximity, X_test, method="isomap")
```

All affinity consumers route through `prepare_proximity_affinity`; all distance
consumers route through `proximity_to_distance`. Dense conversion maps implicit
sparse zero affinity to maximal distance. Directed relations require declared
symmetrization before use by undirected manifold methods. PHATE, UMAP, and
other optional dependencies remain behind extras and executable skip paths.

The maintained acceptance tests are in
`tests/proximity/test_manifold_conversion.py`, and all notebook examples are
validated by `scripts/validate_notebooks.py`.
