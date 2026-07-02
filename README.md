# GeneralForestProximities

## Outlier Detection Runner

Use the wrapper script below to run outlier detection in the `generalforest` conda environment while isolating the run from user site-packages:

```bash
bash scripts/run_outlier_detection.sh --task both --dataset-limit 1 --random-states 0 --n-estimators 16 --max-rows 300
```

The script sets `PYTHONNOUSERSITE=1` automatically and forwards all CLI arguments to `experiments/run_outlier_detection.py`.
It also clears `PYTHONPATH` before execution, preventing `~/.local/.../site-packages` from overriding conda packages.

To use a different conda environment, set `GENERALFOREST_ENV`:

```bash
GENERALFOREST_ENV=myenv bash scripts/run_outlier_detection.sh --task classification
```

If you previously saw errors like `ImportError: numpy.core.multiarray failed to import` or matplotlib imports from `~/.local/lib/python3.11/site-packages`, use this wrapper instead of calling `python experiments/run_outlier_detection.py` directly.
Sparse proximity computations for random forest-like ensembles.

## Installation

### Install from local clone

```bash
pip install .
```

### Install directly from GitHub

```bash
pip install "git+https://github.com/JakeSRhodesLab/GeneralForestProximities.git"
```

### Install optional extras

```bash
pip install .[openml]
pip install .[treeple]
pip install .[phate]
pip install .[umap]
pip install .[manifold]
```

## Manifold embeddings from proximities

Proximity-enabled forests can produce manifold embeddings after fitting:

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

PHATE consumes the forest proximities as precomputed affinities. t-SNE, Isomap,
and UMAP use proximity-derived precomputed distances:

```python
embedding = forest.embed(
    method="isomap",
    n_components=2,
    symm_mode="arithmetic",
)
```

RF-GAP proximities are asymmetric by default. Symmetrization is recommended for
t-SNE, Isomap, and UMAP because these methods generally assume symmetric
distances or undirected neighbor structure. PHATE can be run with the original
asymmetric RF-GAP affinities, or with explicit symmetrization when desired.
