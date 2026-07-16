# forestgap

`forestgap` 1.0 builds sparse forest relations by composition and provides
standalone applications for those relations. It never changes an estimator's
class and rejects estimator–relation pairs whose bootstrap, OOB, honesty, or
path contract is absent.

## Installation

Python 3.11 or newer is required.

```bash
pip install .
```

Optional integrations are separate extras:

```bash
pip install '.[aeon]'
pip install '.[treeple]'
pip install '.[quantile]'
pip install '.[survival]'
pip install '.[manifold]'
```

## Composition API

```python
from sklearn.ensemble import RandomForestClassifier
from forestgap import Proximity, outlier_scores

forest = RandomForestClassifier(n_estimators=300, random_state=0)
proximity = Proximity(forest, relation="rfgap").fit(X_train, y_train)

P_train = proximity.get_proximities()
P_test = proximity.prox_extend(X_test)
scores = outlier_scores(proximity)
```

Use `attach(X, y)` instead of `fit(X, y)` to index an already fitted parent
forest. The unchanged parent is always available as `proximity.estimator`.

Supported relation names include `original`, `oob`, `rfgap`, `honest_gap`,
`kerf`, `path`, and `random_height`; isolation collision, quantile conditional,
and survival-leaf relations are separately named. Call `capabilities_for` or
inspect `proximity.capabilities_` before constructing automated study cells.
Invalid cells raise `UnsupportedRelationError`.

Native `CanonicalCorrelationForestClassifier` and
`CanonicalCorrelationForestRegressor` estimators expose the `original`
relation. Their deterministic MATLAB-source conformance fixture is packaged;
CCF-Bag OOB/RF-GAP remains deliberately gated.

RF-GAP, Honest-GAP, and conditional weights are directed. KeRF, path,
random-height, original, and collision relations are symmetric kernels or
affinities. Symmetrization must be explicit before an application requiring an
undirected geometry.

## Standalone applications

The `forestgap.applications` namespace includes imputation, native and
relation-based anomaly scoring, local importance, prototypes, visualization,
kernel/kNN prediction, clustering, conditional quantiles, split conformal
prediction, batch active learning, instance attribution, retrieval, and
proximity-neighborhood feature selection.

```python
from forestgap import embed, proximity_clustering

coordinates = embed(proximity, method="isomap", symmetrize="arithmetic")
clusters = proximity_clustering(proximity, n_clusters=3)
```

See [the migration guide](notes/migration_1_0.md) and the deterministic
notebooks under `notebooks/`.

## Verification

```bash
pytest -q
python scripts/validate_notebooks.py --execute
python -m experiments.run_method_matrix --smoke --output results/raw/method_matrix.jsonl
```

The experiment harness in `experiments/harness.py` runs declared
`estimator × relation × application × data regime` cells, reuses fitted parent
models and splits, and records provenance, diagnostics, runtime, peak memory,
parent/downstream metrics, and unsupported statuses. The CI workflow also
rejects executable imports of the removed wrapper API and builds the wheel.
