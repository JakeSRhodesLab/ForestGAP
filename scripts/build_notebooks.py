"""Build deterministic composition-API tutorial notebooks.

Run this script after changing a tutorial template, then execute
``scripts/smoke_notebooks.py`` to refresh committed outputs and metadata.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"

PARAMETERS = """\
SMOKE = True
RANDOM_STATE = 0
N_SAMPLES = 80 if SMOKE else 800
N_ESTIMATORS = 20 if SMOKE else 300
"""

SEMANTICS = """
All examples use local synthetic data, fixed seeds, and a reduced smoke
configuration by default. `original`, `kerf`, `path`, `random_height`, and the
separately named isolation `collision` relation are symmetric affinities.
`rfgap`, `honest_gap`, and conditional weights are directed query-to-training
weights; downstream visualization, clustering, or attribution must state any
symmetrization. Prediction identity is claimed only for unsymmetrized `rfgap`
or `honest_gap` under their declared bootstrap/honesty contracts.
"""

SETUP_CLASSIFICATION = """\
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=N_SAMPLES, n_features=8, n_informative=5,
    n_redundant=1, random_state=RANDOM_STATE,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE,
)
"""


def markdown(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str, *, parameters: bool = False):
    cell = nbf.v4.new_code_cell(dedent(text).strip())
    if parameters:
        cell.metadata["tags"] = ["parameters"]
    return cell


def notebook(title: str, purpose: str, cells: list):
    nb = nbf.v4.new_notebook()
    nb.cells = [
        markdown(f"# {title}\n\n{purpose}\n\n{SEMANTICS}"),
        code(PARAMETERS, parameters=True),
        *cells,
    ]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "forestgap": {"smoke": True, "network": False, "seed": 0},
    }
    return nb


TUTORIALS = {
    "honest_gap_prediction_identity.ipynb": notebook(
        "Honest-GAP prediction identity",
        "Use `honest_gap` only when an HonestForest exposes structure and estimation roles.",
        [
            code(SETUP_CLASSIFICATION),
            code("""
import inspect
import warnings
from forestgap import Proximity, prediction_identity

try:
    from treeple import HonestForestClassifier
except ImportError:
    HonestForestClassifier = None

if HonestForestClassifier is None:
    result = {"status": "skipped", "reason": "install forestgap[treeple]"}
else:
    signature = inspect.signature(HonestForestClassifier.__init__)
    kwargs = {"n_estimators": N_ESTIMATORS, "random_state": RANDOM_STATE}
    for name, value in (("bootstrap", True), ("honest_fraction", 0.5), ("max_depth", 5), ("n_jobs", 1), ("kernel_method", True)):
        if name in signature.parameters:
            kwargs[name] = value
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forest = HonestForestClassifier(**kwargs).fit(X_train, y_train)
        proximity = Proximity(forest, relation="honest_gap").attach(X_train, y_train)
    reconstructed = prediction_identity(proximity, X_test)
    result = {
        "status": "ok",
        "relation": proximity.relation,
        "max_abs_error": float(np.max(np.abs(reconstructed - forest.predict_proba(X_test)))),
    }
result
"""),
        ],
    ),
    "honest_forest_outlier_demo_labeled.ipynb": notebook(
        "HonestForest labeled outlier audit",
        "For class-relative anomaly geometry use an explicitly symmetric relation; reserve `honest_gap` for prediction identity.",
        [
            code(SETUP_CLASSIFICATION),
            code("""
import inspect
from forestgap import Proximity, outlier_scores

try:
    from treeple import HonestForestClassifier
except ImportError:
    HonestForestClassifier = None

if HonestForestClassifier is None:
    result = {"status": "skipped", "reason": "install forestgap[treeple]"}
else:
    signature = inspect.signature(HonestForestClassifier.__init__)
    kwargs = {"n_estimators": N_ESTIMATORS, "random_state": RANDOM_STATE}
    if "n_jobs" in signature.parameters:
        kwargs["n_jobs"] = 1
    forest = HonestForestClassifier(**kwargs).fit(X_train, y_train)
    proximity = Proximity(forest, relation="original").attach(X_train, y_train)
    scores = outlier_scores(proximity, y_train, method="ene", scaling="normalize")
    result = {"status": "ok", "relation": "original", "largest_scores": np.sort(scores)[-5:].round(3).tolist()}
result
"""),
        ],
    ),
    "isolation_forest_demo.ipynb": notebook(
        "Isolation Forest native anomaly scores",
        "Isolation forests use native path-length `score_samples`; collision affinity is optional and never called RF-GAP or OOB.",
        [
            code(SETUP_CLASSIFICATION),
            code("""
from sklearn.ensemble import IsolationForest
from forestgap import Proximity, native_anomaly_scores

forest = IsolationForest(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE).fit(X_train)
native_scores = native_anomaly_scores(forest, X_test)
collision = Proximity(forest, relation="collision").attach(X_train)
result = {
    "native_score_range": [float(native_scores.min()), float(native_scores.max())],
    "collision_relation": collision.relation,
    "collision_shape": collision.get_proximities().shape,
}
result
"""),
        ],
    ),
    "isolation_forest_outlier_comparison.ipynb": notebook(
        "Isolation score comparison",
        "Compare native isolation path scores across valid parent configurations without relabeling leaf collisions as RF-GAP.",
        [
            code(SETUP_CLASSIFICATION),
            code("""
from sklearn.ensemble import IsolationForest
from forestgap import native_anomaly_scores

rows = []
for bootstrap in (False, True):
    forest = IsolationForest(
        n_estimators=N_ESTIMATORS, bootstrap=bootstrap, random_state=RANDOM_STATE,
    ).fit(X_train)
    scores = native_anomaly_scores(forest, X_test)
    rows.append({"bootstrap": bootstrap, "mean_native_anomaly": float(scores.mean())})
rows
"""),
        ],
    ),
    "imputation_and_outlier_detection.ipynb": notebook(
        "Proximity imputation and outlier audit",
        "Use standalone donor imputation and symmetric relation-based anomaly scoring.",
        [
            code(SETUP_CLASSIFICATION),
            code("""
from sklearn.ensemble import RandomForestClassifier
from forestgap import Proximity, impute, outlier_scores

forest = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
proximity = Proximity(forest, relation="original").fit(X_train, y_train)
X_missing = X_train.copy()
X_missing[::7, 0] = np.nan
X_imputed = impute(proximity, X_missing)
scores = outlier_scores(proximity, y_train, method="ene")
{"remaining_missing": int(np.isnan(X_imputed).sum()), "score_shape": scores.shape}
"""),
        ],
    ),
    "local_feature_importance_demo.ipynb": notebook(
        "Local feature importance",
        "Perturb query features against a fixed indexed training relation via the standalone application.",
        [
            code(SETUP_CLASSIFICATION),
            code("""
from sklearn.ensemble import RandomForestClassifier
from forestgap import Proximity, local_feature_importance

proximity = Proximity(
    RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE),
    relation="original",
).fit(X_train, y_train)
importance = local_feature_importance(
    proximity, X_test[:3], perturbation="permute", n_repeats=3,
    random_state=RANDOM_STATE,
)
{"shape": importance.shape, "top_features": np.argmax(np.abs(importance), axis=1).tolist()}
"""),
        ],
    ),
    "manifold.ipynb": notebook(
        "Forest proximity manifold",
        "Route affinity preparation and distance conversion through the shared standalone embedding application.",
        [
            code(SETUP_CLASSIFICATION),
            code("""
from sklearn.ensemble import RandomForestClassifier
from forestgap import Proximity, embed

proximity = Proximity(
    RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE),
    relation="path", random_state=RANDOM_STATE,
).fit(X_train, y_train)
coordinates = embed(proximity, method="isomap", n_components=2, n_neighbors=5)
{"shape": coordinates.shape, "relation": proximity.relation}
"""),
        ],
    ),
    "manifold_mwe.ipynb": notebook(
        "Minimal manifold example",
        "A minimal composition example with an explicit symmetric relation and Isomap distance conversion.",
        [
            code(SETUP_CLASSIFICATION),
            code("""
from sklearn.ensemble import RandomForestClassifier
from forestgap import Proximity, embed

forest = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
proximity = Proximity(forest, relation="original").fit(X_train, y_train)
embedding = embed(proximity, method="isomap", n_neighbors=5)
embedding.shape
"""),
        ],
    ),
    "oblique_random_forest.ipynb": notebook(
        "Oblique random forest relations",
        "treeple remains optional and non-vendored; only relations proven by its runtime capability contract are exposed.",
        [
            code(SETUP_CLASSIFICATION),
            code("""
from forestgap import Proximity

try:
    from treeple import ObliqueRandomForestClassifier
except ImportError:
    ObliqueRandomForestClassifier = None

if ObliqueRandomForestClassifier is None:
    result = {"status": "skipped", "reason": "install forestgap[treeple]"}
else:
    forest = ObliqueRandomForestClassifier(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=1,
    )
    proximity = Proximity(forest, relation="original").fit(X_train, y_train)
    result = {"status": "ok", "family": proximity.capabilities_.family, "shape": proximity.get_proximities().shape}
result
"""),
        ],
    ),
    "oblique_rf_simulation_comparison.ipynb": notebook(
        "Axis-aligned and oblique forest comparison",
        "Fit each parent once on a deterministic rotated-signal simulation and compare the same declared original relation.",
        [
            code(SETUP_CLASSIFICATION),
            code("""
from sklearn.ensemble import RandomForestClassifier
from forestgap import Proximity

families = [("axis", RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE))]
try:
    from treeple import ObliqueRandomForestClassifier
    families.append(("oblique", ObliqueRandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=1)))
except ImportError:
    pass

results = []
for name, forest in families:
    proximity = Proximity(forest, relation="original").fit(X_train, y_train)
    results.append({"family": name, "accuracy": float((forest.predict(X_test) == y_test).mean()), "density": proximity.diagnostics()["density"]})
results
"""),
        ],
    ),
    "prototypes_demo.ipynb": notebook(
        "Forest-space prototypes",
        "Select class-constrained representatives with the standalone prototype application.",
        [
            code(SETUP_CLASSIFICATION),
            code("""
from sklearn.ensemble import RandomForestClassifier
from forestgap import Proximity, get_prototypes

proximity = Proximity(
    RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE), relation="original",
).fit(X_train, y_train)
prototypes = get_prototypes(proximity, k=5, n_prototypes=1)
{int(label): int(records[0]["center_index"]) for label, records in prototypes.items()}
"""),
        ],
    ),
    "rotation_forest_proximities.ipynb": notebook(
        "Rotation Forest proximities",
        "aeon is optional; its adapter reconstructs leaf assignments, and bootstrap relations remain capability-gated.",
        [
            code(SETUP_CLASSIFICATION),
            code("""
from forestgap import Proximity

try:
    from aeon.classification.sklearn import RotationForestClassifier
except ImportError:
    RotationForestClassifier = None

if RotationForestClassifier is None:
    result = {"status": "skipped", "reason": "install forestgap[aeon]"}
else:
    forest = RotationForestClassifier(
        n_estimators=max(3, N_ESTIMATORS // 4), random_state=RANDOM_STATE, n_jobs=1,
    )
    proximity = Proximity(forest, relation="original").fit(X_train, y_train)
    result = {"status": "ok", "valid_relations": proximity.capabilities_.valid_relations, "shape": proximity.get_proximities().shape}
result
"""),
        ],
    ),
    "spectral_cluster.ipynb": notebook(
        "Spectral clustering from a forest relation",
        "Clustering uses a symmetric affinity and never consumes evaluation labels while fitting the graph.",
        [
            code(SETUP_CLASSIFICATION),
            code("""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_rand_score
from forestgap import Proximity, proximity_clustering

proximity = Proximity(
    RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE), relation="kerf",
).fit(X_train, y_train)
clusters = proximity_clustering(proximity, n_clusters=2, random_state=RANDOM_STATE)
{"shape": clusters.shape, "ari_for_posthoc_evaluation_only": float(adjusted_rand_score(y_train, clusters))}
"""),
        ],
    ),
    "canonical_correlation_forest.ipynb": notebook(
        "Canonical Correlation Forest and MATLAB conformance",
        "Fit the native sklearn-compatible CCF, validate its deterministic MATLAB-source fixture, and construct only its supported original co-leaf relation. CCF-Bag OOB/RF-GAP remain gated.",
        [
            code("""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from forestgap import CanonicalCorrelationForestClassifier, Proximity
from forestgap.adapters import (
    UnsupportedRelationError, load_ccf_oracle_fixture,
    method_capabilities, validate_ccf_oracle,
)

X, y = make_classification(
    n_samples=N_SAMPLES, n_features=6, n_informative=4,
    n_redundant=0, random_state=RANDOM_STATE,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.25, stratify=y, random_state=RANDOM_STATE,
)
forest = CanonicalCorrelationForestClassifier(
    n_estimators=N_ESTIMATORS, max_depth=8, random_state=RANDOM_STATE,
).fit(X_train, y_train)
proximity = Proximity(forest, relation="original").attach(X_train, y_train)
P_test = proximity.prox_extend(X_test)
{"accuracy": forest.score(X_test, y_test), "original_relation_shape": P_test.shape}
"""),
            code("""
fixture = load_ccf_oracle_fixture()
oracle = validate_ccf_oracle(
    CanonicalCorrelationForestClassifier(**fixture["estimator_parameters"]),
    np.asarray(fixture["X_train"]),
    np.asarray(fixture["classification"]["y_train"]),
    np.asarray(fixture["X_test"]),
    np.asarray(fixture["classification"]["expected_predictions"]),
)
capability = method_capabilities("canonical_correlation_forest")
gates = {}
for relation in ("oob", "rfgap"):
    try:
        Proximity(forest, relation=relation).attach(X_train, y_train)
    except UnsupportedRelationError as error:
        gates[relation] = {"status": "gated", "reason": error.reason}
{
    "oracle_source_commit": fixture["source"]["commit"],
    "oracle_passed": oracle.passed,
    "oracle_max_absolute_error": oracle.max_absolute_error,
    "valid_relations": capability.valid_relations,
    "gated_relations": gates,
}
"""),
        ],
    ),
    "quantile_forest_proximities.ipynb": notebook(
        "Quantile forest conditional weights",
        "Use native quantile prediction and separately declared conditional-distribution weights; calibration is evaluated on held-out rows.",
        [
            code("""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_pinball_loss
from forestgap import (
    Proximity, conditional_distribution_weights, conditional_quantiles,
)

X, y = make_regression(n_samples=N_SAMPLES, n_features=6, noise=8, random_state=RANDOM_STATE)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=RANDOM_STATE)
try:
    from quantile_forest import RandomForestQuantileRegressor
except ImportError:
    RandomForestQuantileRegressor = None

if RandomForestQuantileRegressor is None:
    result = {"status": "skipped", "reason": "install forestgap[quantile]"}
else:
    forest = RandomForestQuantileRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE).fit(X_train, y_train)
    proximity = Proximity(forest, relation="conditional").attach(X_train, y_train)
    quantiles = [0.1, 0.5, 0.9]
    weights = conditional_distribution_weights(proximity, X_test)
    weighted = conditional_quantiles(proximity, y_train, quantiles, X=X_test)
    native = forest.predict(X_test, quantiles=quantiles)
    result = {
        "status": "ok",
        "weight_row_mass": np.asarray(weights.sum(axis=1)).ravel().round(8).tolist()[:5],
        "weighted_80pct_coverage": float(np.mean((y_test >= weighted[:, 0]) & (y_test <= weighted[:, 2]))),
        "native_80pct_coverage": float(np.mean((y_test >= native[:, 0]) & (y_test <= native[:, 2]))),
        "weighted_pinball": float(np.mean([
            mean_pinball_loss(y_test, weighted[:, i], alpha=q) for i, q in enumerate(quantiles)
        ])),
        "native_pinball": float(np.mean([
            mean_pinball_loss(y_test, native[:, i], alpha=q) for i, q in enumerate(quantiles)
        ])),
        "median_mean_absolute_difference": float(np.mean(np.abs(weighted[:, 1] - native[:, 1]))),
    }
result
"""),
        ],
    ),
    "random_survival_forest_proximities.ipynb": notebook(
        "Random survival forest outputs and leaf relation",
        "Survival risk, survival, and hazard outputs stay native; `survival_leaf` is separately named and evaluation must use censoring-safe metrics.",
        [
            code("""
import numpy as np
from sklearn.model_selection import train_test_split
from forestgap import Proximity
from forestgap.adapters import (
    censoring_safe_survival_metrics, native_survival_outputs,
)

rng = np.random.default_rng(RANDOM_STATE)
X = rng.normal(size=(N_SAMPLES, 5))
event_time = rng.exponential(np.exp(-.7 * X[:, 0]) * 8.0)
censor_time = rng.exponential(11.0, size=N_SAMPLES)
event = event_time <= censor_time
observed_time = np.minimum(event_time, censor_time) + .05
y = np.array(list(zip(event, observed_time)), dtype=[("event", "?"), ("time", "<f8")])
train_ids, test_ids = train_test_split(
    np.arange(N_SAMPLES), test_size=.3, stratify=event,
    random_state=RANDOM_STATE,
)
X_train, X_test, y_train, y_test = X[train_ids], X[test_ids], y[train_ids], y[test_ids]
try:
    from sksurv.ensemble import RandomSurvivalForest
except ImportError:
    RandomSurvivalForest = None

if RandomSurvivalForest is None:
    result = {"status": "skipped", "reason": "install forestgap[survival]"}
else:
    forest = RandomSurvivalForest(
        n_estimators=N_ESTIMATORS, min_samples_leaf=3,
        random_state=RANDOM_STATE, n_jobs=1,
    ).fit(X_train, y_train)
    proximity = Proximity(forest, relation="survival_leaf").attach(X_train, y_train)
    outputs = native_survival_outputs(forest, X_test[:3])
    metrics = censoring_safe_survival_metrics(forest, X_test, y_train, y_test)
    result = {
        "status": "ok",
        "train_events": int(np.count_nonzero(y_train["event"])),
        "test_censored": int(np.count_nonzero(~y_test["event"])),
        "risk_shape": outputs["risk"].shape,
        "survival_functions": len(outputs["survival"]),
        "hazard_functions": len(outputs["cumulative_hazard"]),
        "relation_shape": proximity.prox_extend(X_test).shape,
        "ipcw_concordance": metrics["ipcw_concordance"],
        "integrated_brier_score": metrics["integrated_brier_score"],
        "excluded_outside_train_support": metrics["n_excluded_outside_train_support"],
    }
result
"""),
        ],
    ),
    "synthetic_null_urf.ipynb": notebook(
        "Shi–Horvath synthetic-null unsupervised forest",
        "Persist the synthetic generator and seed, then index observed rows only for declared original/OOB relations.",
        [
            code("""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from forestgap import Proximity, SyntheticNullForest, proximity_clustering

X, labels = make_blobs(n_samples=N_SAMPLES, centers=3, n_features=5, random_state=RANDOM_STATE)
forest = SyntheticNullForest(
    estimator=RandomForestClassifier(
        n_estimators=N_ESTIMATORS, bootstrap=True, random_state=RANDOM_STATE,
    ),
    random_state=RANDOM_STATE,
).fit(X)
original = Proximity(forest, relation="original").attach(X)
oob = Proximity(forest, relation="oob").attach(X)
clusters = proximity_clustering(original, n_clusters=3, random_state=RANDOM_STATE)
native_observed_scores = forest.score_samples(X)
{
    "generator": forest.generator_,
    "synthetic_seed": forest.synthetic_seed_,
    "synthetic_rows": forest.synthetic_X_.shape[0],
    "original_relation_shape": original.get_proximities().shape,
    "oob_relation_shape": oob.get_proximities().shape,
    "native_observed_score_range": [float(native_observed_scores.min()), float(native_observed_scores.max())],
    "cluster_count": int(np.unique(clusters).size),
}
"""),
        ],
    ),
}


def main() -> None:
    NOTEBOOKS.mkdir(exist_ok=True)
    for filename, nb in TUTORIALS.items():
        nbf.write(nb, NOTEBOOKS / filename)
    print(f"wrote {len(TUTORIALS)} notebooks")


if __name__ == "__main__":
    main()
