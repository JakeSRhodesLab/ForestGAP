# Proximity Manifold Applications Plan

## Goal

Add RF-PHATE-style manifold learning capabilities to `GeneralForestProximities` as applications of generalized forest proximities. The implementation should support PHATE directly from forest proximity affinities and provide an optional path for using proximities with other manifold learning methods such as t-SNE, Isomap, and UMAP.

## Design Principles

- Keep `forestgap.core` focused on proximity generation.
- Add manifold learning as a separate application layer.
- Reuse `with_sparse_proximity` and `SparseProximityMixin` instead of duplicating RF proximity logic from `RF-PHATE`.
- Make manifold embeddings extractable from an already-trained proximity-enabled forest object.
- Avoid forcing users into a separate estimator class when a method on the trained forest is the cleaner user-facing API.
- Preserve sparse matrices as long as possible.
- Route optional dependencies through extras so the base proximity package stays lightweight.
- Be explicit about whether each downstream method expects an affinity matrix or a distance matrix.

## Important Note: RF-GAP Asymmetry

RF-GAP proximities are asymmetric by default. This is meaningful for supervised proximity weighting and for prediction-style extensions, but it can be problematic for manifold methods that assume symmetric pairwise distances or undirected neighbor relationships.

For t-SNE, UMAP, and Isomap, the application layer should symmetrize RF-GAP proximities before converting them to distances. Recommended defaults:

- `symm_mode="arithmetic"` for a conservative symmetric affinity.
- `non_zero_diagonal=True` when the downstream method needs self-neighbors or when affinity-vs-distance detection depends on diagonal values.
- Optional `symm_mode="geometric"` for stricter mutual-proximity structure.

PHATE is less clear-cut because RF-PHATE intentionally uses proximities as precomputed affinities and PHATE/graphtools can apply its own graph construction and kernel symmetrization options. The PHATE wrapper should therefore expose `kernel_symm` or equivalent PHATE options, document the asymmetry behavior, and test both asymmetric and explicitly symmetrized RF-GAP inputs.

## Proposed Package Layout

Add a new manifold/application namespace:

```text
forestgap/
  manifold/
    __init__.py
    proximity.py
    phate.py
    embedding.py
```

Responsibilities:

- `proximity.py`: matrix validation, affinity preparation, symmetrization, diagonal handling, affinity-to-distance conversion.
- `phate.py`: RF-PHATE-style functions and optional transformer wrapper using forestgap proximities.
- `embedding.py`: generic t-SNE, Isomap, and UMAP functions and optional wrappers using proximity-derived distances.
- `__init__.py`: public exports.

## Preferred User-Facing API

The primary API should allow embeddings to be extracted from a forest object that has already been trained and has proximity structures available:

```python
from forestgap import with_sparse_proximity
from sklearn.ensemble import RandomForestClassifier

ProxRF = with_sparse_proximity(RandomForestClassifier)
forest = ProxRF(
    n_estimators=500,
    random_state=0,
    prox_method="rfgap",
    symm_mode="arithmetic",
    non_zero_diagonal=True,
)
forest.fit(X, y)

embedding = forest.embed(method="phate", n_components=2)
```

Candidate methods on `SparseProximityMixin`:

```python
forest.embed(method="phate", **kwargs)
forest.embed(method="tsne", **kwargs)
forest.embed(method="isomap", **kwargs)
forest.embed(method="umap", **kwargs)
forest.phate(**kwargs)
forest.proximity_embedding(method="tsne", **kwargs)
```

Recommended direction:

- Add a general `embed(method=..., **kwargs)` method to the mixin for a compact public API.
- Optionally add `phate(**kwargs)` as a convenience alias because RF-PHATE is a first-class application.
- Keep standalone classes such as `ForestPHATE` or `ProximityEmbedding` only if they materially simplify sklearn pipeline integration, parameter management, or testing.
- Implement the actual work in functions under `forestgap.manifold` so both trained-forest methods and optional wrapper classes can reuse the same code.
- If the forest is not fitted, `embed()` should raise the same style of fitted-state error as other proximity methods rather than silently fitting.

## Matrix Preparation Utilities

Add reusable helpers such as:

```python
prepare_proximity_affinity(
    P,
    *,
    symmetrize="auto",
    diagonal="one",
    normalize=None,
)

proximity_to_distance(
    P,
    *,
    method="sqrt_one_minus",
    dense=False,
)
```

Implementation requirements:

- Accept dense NumPy arrays and SciPy sparse matrices.
- Validate square matrices for training embeddings.
- Support rectangular extension matrices for out-of-sample projection where relevant.
- Clip proximity values into a safe range before distance conversion.
- Treat missing sparse entries correctly: zero proximity should become maximal distance, not zero distance.
- Preserve sparsity when the downstream estimator supports sparse precomputed input.
- Provide clear errors when a method requires dense distances.

Recommended distance conversions:

- `sqrt_one_minus`: `D = sqrt(max(1 - P, 0))`
- `one_minus`: `D = max(1 - P, 0)`
- Optional future methods: angular/cosine-like conversions if useful for normalized proximities.

## PHATE Application

Add PHATE support that can be called from a trained proximity-enabled forest:

```python
embedding = forest.embed(
    method="phate",
    n_components=2,
    t="auto",
)

embedding = forest.phate(
    n_components=2,
    t="auto",
)
```

Optionally add a sklearn-style transformer, tentatively named `ForestPHATE`, if this is useful for pipelines:

```python
from forestgap.manifold import ForestPHATE
from sklearn.ensemble import RandomForestClassifier

embedding = ForestPHATE(
    forest=RandomForestClassifier(n_estimators=500, random_state=0),
    prox_method="rfgap",
    symm_mode=None,
    non_zero_diagonal=True,
    phate_kwargs={"n_components": 2, "t": "auto"},
).fit_transform(X, y)
```

Behavior for an optional forest-owning transformer API:

1. Wrap the supplied forest with `with_sparse_proximity`.
2. Fit the proximity-enabled forest.
3. Build the training proximity affinity with `get_proximities()`.
4. Prepare the affinity matrix for PHATE.
5. Fit PHATE using `knn_dist="precomputed_affinity"`.
6. Store the fitted forest, PHATE operator, training embedding, and matrix preparation settings.

For the trained-forest API, steps 1 and 2 are already complete. The method should start from `get_proximities()` and should store the fitted PHATE operator on the forest, for example in `forest.manifold_models_["phate"]`, so later extension can reuse it.

Out-of-sample support:

- Use `prox_extend(X_new)` to compute affinities from new observations to training observations.
- For landmark PHATE, map new observations through the landmark clusters as RF-PHATE currently does.
- Use PHATE graph interpolation to project new observations into the fitted embedding.

## PageRank-PHATE Option

Port RF-PHATE's `PageRankPHATE` idea as an optional mode:

```python
ForestPHATE(..., pagerank_beta=0.9)
```

Recommended behavior:

- `pagerank_beta=None`: use standard `phate.PHATE`.
- `pagerank_beta` in `(0, 1]`: use a local `PageRankPHATE` subclass.
- Keep this dependency lazy and local to the PHATE application module.

## Generic Proximity Embedding

Add a trained-forest embedding API for non-PHATE methods:

```python
tsne_embedding = forest.embed(method="tsne", n_components=2)
isomap_embedding = forest.embed(method="isomap", n_components=2)
umap_embedding = forest.embed(method="umap", n_components=2)
```

Optionally add a wrapper for sklearn-style workflows, tentatively named `ProximityEmbedding`:

```python
from forestgap.manifold import ProximityEmbedding

tsne_embedding = ProximityEmbedding(method="tsne", forest=forest).fit_transform(X, y)
isomap_embedding = ProximityEmbedding(method="isomap", forest=forest).fit_transform(X, y)
umap_embedding = ProximityEmbedding(method="umap", forest=forest).fit_transform(X, y)
```

Behavior for an optional forest-owning wrapper:

1. Fit a proximity-enabled forest.
2. Compute training proximities.
3. Symmetrize if needed, especially for `prox_method="rfgap"`.
4. Convert proximities to distances.
5. Dispatch to the selected manifold learner with precomputed distance settings.

For the trained-forest API, fitting the forest is not part of this method. The method should require fitted proximity structures, call `get_proximities()`, and store fitted manifold operators on the forest when the downstream estimator supports reuse or transform.

Method-specific defaults:

- t-SNE: `sklearn.manifold.TSNE(metric="precomputed", init="random")`
- Isomap: `sklearn.manifold.Isomap(metric="precomputed")`
- UMAP: `umap.UMAP(metric="precomputed")`

## Transform Support by Method

The API should be honest about out-of-sample behavior:

- PHATE: support `transform(X_new)` using proximity extension and graph interpolation.
- Isomap: likely support `transform(X_new)` using distances from new points to training points; verify behavior with sklearn's precomputed path.
- UMAP: support only after testing. UMAP can fit with `metric="precomputed"`, but transform support may be limited depending on how precomputed neighbors are supplied.
- t-SNE: expose `fit_transform` only. sklearn t-SNE does not provide a standard `transform`.

For trained-forest methods, consider:

```python
forest.embed(method="phate")
forest.embed_transform(X_new, method="phate")
```

or method-specific aliases:

```python
forest.phate()
forest.phate_transform(X_new)
```

Only add transform aliases for methods where out-of-sample behavior is well-defined.

## Optional Dependencies

Update `pyproject.toml` extras:

```toml
[project.optional-dependencies]
phate = ["phate", "graphtools"]
umap = ["umap-learn"]
manifold = ["phate", "graphtools", "umap-learn"]
```

Keep imports lazy:

- Import `phate` and `graphtools` only inside `forestgap.manifold.phate`.
- Import `umap` only when `method="umap"` is requested.
- Raise helpful installation errors that mention the appropriate extra.

## Testing Plan

Add tests under `tests/`, likely `test_manifold.py`.

Core tests:

- `embed()` raises when called before forest fitting.
- `embed(method=...)` works from an already-fitted proximity-enabled forest.
- Dense and sparse affinity preparation.
- RF-GAP asymmetry detection and symmetrization.
- Diagonal handling for precomputed affinities.
- Sparse zero handling during affinity-to-distance conversion.
- Shape checks for `ForestPHATE.fit_transform()`.
- Shape checks for `ForestPHATE.transform()` on held-out data.
- If wrapper classes are deferred, replace `ForestPHATE` checks with direct `forest.embed()` and method-specific transform checks.
- Smoke tests for `ProximityEmbedding(method="tsne")`.
- Smoke tests for `ProximityEmbedding(method="isomap")`.
- Optional UMAP smoke tests guarded by `pytest.importorskip("umap")`.
- Optional PHATE tests guarded by `pytest.importorskip("phate")`.

Regression tests:

- Existing proximity tests must continue to pass.
- Existing `plot_mds()` behavior should either remain unchanged or be refactored to use the new distance conversion helper with equivalent output shape.

## Documentation Plan

Update `README.md` with examples:

```python
from forestgap import with_sparse_proximity
from sklearn.ensemble import RandomForestClassifier

ProxRF = with_sparse_proximity(RandomForestClassifier)
forest = ProxRF(
    n_estimators=300,
    random_state=0,
    prox_method="rfgap",
    non_zero_diagonal=True,
)

forest.fit(X, y)
embedding = forest.embed(method="phate", n_components=2)
```

```python
embedding = forest.embed(
    method="umap",
    symm_mode="arithmetic",
    n_components=2,
)
```

Documentation should explain:

- PHATE consumes proximity affinities directly.
- t-SNE, Isomap, and UMAP consume proximity-derived distances.
- RF-GAP is asymmetric by default.
- Symmetrization is recommended for t-SNE, Isomap, and UMAP.
- Asymmetric RF-GAP may still be useful for PHATE-style graph construction, but this should be exposed as a deliberate choice.

## Implementation Order

1. Add matrix preparation utilities and tests.
2. Add a trained-forest `embed()` method on `SparseProximityMixin` that delegates to `forestgap.manifold`.
3. Add PHATE support with lazy PHATE imports.
4. Add optional PageRank-PHATE mode.
5. Add t-SNE and Isomap support.
6. Add optional UMAP support.
7. Add optional `ForestPHATE` / `ProximityEmbedding` wrapper classes only if pipeline integration benefits from them.
8. Refactor `plot_mds()` to use shared matrix conversion utilities if practical.
9. Add README examples and optional dependency notes.
10. Run the existing test suite plus new manifold tests.

## Open Questions

- Should the default PHATE path preserve asymmetric RF-GAP affinities, or should it default to symmetric affinities for consistency with other methods?
- Should `embed()` be the only public forest-level method, or should `phate()` be added as a convenience alias?
- Should `ForestPHATE` be added for sklearn pipeline integration, or should the trained-forest API be the only supported interface at first?
- Should there also be a lower-level `ProximityPHATE` function/class that accepts an already-computed proximity matrix?
- Should generic embedding wrappers support already-fitted proximity models, precomputed proximity matrices, or only raw `(X, y)` plus a forest estimator?
- Which distance conversion should be the package default for non-PHATE methods: `sqrt_one_minus` or `one_minus`?
