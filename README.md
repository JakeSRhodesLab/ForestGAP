# forestgap

`forestgap` 1.1 builds sparse forest relations by composition and provides
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
pip install '.[boosted]'
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
`kerf`, `boosted`, `path`, `englund`, and `random_height`; isolation collision, quantile
conditional, and survival-leaf relations are separately named. `englund` is
the Englund–Verikas terminal-node branch-distance kernel,
`exp(-englund_weight * g)`, averaged over trees; its default
`englund_weight` is `1.0`. Call `capabilities_for` or
inspect `proximity.capabilities_` before constructing automated study cells.
Invalid cells raise `UnsupportedRelationError`.

`boosted` is a symmetric kernel for supported additive boosted trees. Each
tree is weighted by the empirical variance of its fitted leaf contribution on
the attached training data, the weights are normalized to one, and an
all-zero-variance ensemble falls back to uniform weights. The fitted index
publishes `tree_weights_`, `tree_weighting`, and the same provenance in
`relation_metadata_`; it does not claim prediction identity.

### Relation applicability

The table below summarizes the forest-family contracts implemented by the
adapter registry. Relation names are the values accepted by `Proximity`.

| Forest type | Applicable relations | Availability and important conditions |
| --- | --- | --- |
| scikit-learn random forest | `original`, `kerf`, `path`, `englund`, `random_height`; `oob`, `rfgap` | `oob` and `rfgap` require bootstrap/OOB information. |
| scikit-learn extra-trees forest | `original`, `kerf`, `path`, `englund`, `random_height`; `oob`, `rfgap` | `oob` and `rfgap` require `bootstrap=True` and recoverable in-bag samples. |
| aeon RotationForest | `original`, `kerf`, `path`, `englund`, `random_height`; `oob`, `rfgap` | Requires the `aeon` extra; OOB/RF-GAP use aeon’s synchronized training-estimate contract. |
| treeple oblique/patch forest | `original`, `kerf`, `path`, `englund`, `random_height`; `oob`, `rfgap` | Requires the `treeple` extra and bootstrap/in-bag support for `oob`/`rfgap`. |
| treeple HonestForest classifier | `original`, `kerf`, `path`, `englund`, `random_height`, `honest_gap` | Requires the `treeple` extra; `honest_gap` requires fitted honesty-role indices. |
| Canonical Correlation Forest (CCF) | `original`, `kerf`, `path`, `englund`, `random_height`; `oob`, `rfgap` | `oob`/`rfgap` apply to CCF-Bag (`bootstrap=True`) only. |
| sklearn GradientBoosting classifier/regressor | `original`, `kerf`, `boosted`, `path`, `englund`, `random_height` | Included in the base install; stage/class trees are flattened. `oob` and `rfgap` are invalid. |
| XGBoost classifier/regressor | `original`, `kerf`, `boosted`, `path`, `englund`, `random_height` | Requires the `boosted` extra. `oob` and `rfgap` are invalid. |
| LightGBM classifier/regressor | `original`, `kerf`, `boosted`, `path`, `englund`, `random_height` | Requires the `boosted` extra. `oob` and `rfgap` are invalid. |
| CatBoost classifier/regressor | `original`, `kerf`, `boosted`; `path`, `englund`, `random_height` | Requires the `boosted` extra. Path-derived relations require `grow_policy="SymmetricTree"`; categorical DataFrame/Pool inputs retain native typing. `oob` and `rfgap` are invalid. |
| quantile forest | `original`, `kerf`, `path`, `englund`, `random_height`, `conditional` | Requires the `quantile` extra; `conditional` is the quantile-specific relation. |
| random survival forest | `survival_leaf` | Requires the `survival` extra. |
| isolation forest | `collision` | Uses the isolation partition/collision relation; native anomaly scoring remains separate. |
| extended isolation forest | `collision` | Requires the `treeple` extra; native path scoring remains separate from `collision`. |
| synthetic-null URF | `original`, `oob`, `kerf`, `path`, `random_height` | `rfgap` is intentionally not enabled for this unsupervised forest. |

These are family-level capabilities. `capabilities_for(estimator)` reports the
relations available for a particular fitted configuration: `path` and
`random_height` are removed when decision-path or node-depth metadata is absent
(including non-symmetric CatBoost growth policies),
and `oob`/`rfgap` are removed when bootstrap, in-bag, or contribution-weight
requirements are absent. Forest families marked as gated in the registry
(other boosted families, causal, generalized, Bayesian, structured, multiview, graph, and
autoencoding forests) currently have no applicable `Proximity` relations.

Native `CanonicalCorrelationForestClassifier` and
`CanonicalCorrelationForestRegressor` expose `original`, `kerf`, `path`, and
`random_height`. Set `bootstrap=True, oob_score=True` for CCF-Bag to enable
OOB/RF-GAP and native OOB outputs. Their deterministic MATLAB-source
conformance fixture is packaged.

aeon RotationForest is supported through a compatibility adapter: it
reconstructs aeon's training-estimate resampling roles for OOB/RF-GAP and
uses the fitted rotation trees for external-query weights. This does not rely
on an `estimators_samples_` attribute that aeon does not publish.

RF-GAP, Honest-GAP, and conditional weights are directed. KeRF, boosted, path,
Englund, random-height, original, and collision relations are symmetric kernels or
affinities. Symmetrization must be explicit before an application requiring an
undirected geometry.

Booster subsampling and CatBoost bootstrap settings do not enable `oob` or
`rfgap`: those relations require exchangeable bootstrap membership and the
forest prediction-weight contract, which additive boosting does not provide.

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

`proximity_clustering` supports `method="spectral"` (the default),
`method="hdbscan"`, and `method="kmeans"`. Spectral clustering consumes the
prepared proximity affinity directly; HDBSCAN receives its dense precomputed
distance conversion; and K-means runs on a KernelPCA embedding of the
precomputed affinity. HDBSCAN infers its number of clusters, so use its
density controls directly, for example:

```python
labels = proximity_clustering(proximity, method="hdbscan", min_cluster_size=10)
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
