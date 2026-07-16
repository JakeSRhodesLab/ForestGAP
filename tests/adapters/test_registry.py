import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest, RandomForestClassifier

from forestgap.adapters import (
    METHOD_REGISTRY,
    SyntheticNullForest,
    UnsupportedRelationError,
    capabilities_for,
    method_registry_rows,
    validate_relation,
)


def test_all_non_reference_roadmap_families_have_visible_rows():
    expected = {
        "random_forest", "extra_trees", "rotation_forest", "treeple_oblique_forest",
        "honest_forest", "isolation_forest", "extended_isolation_forest",
        "canonical_correlation_forest", "quantile_forest", "random_survival_forest",
        "synthetic_null_urf", "boosted_trees", "causal_forest",
        "generalized_random_forest", "bayesian_trees", "structured_pf_gap",
        "multiview_forest", "graph_forest", "autoencoding_forest",
    }
    assert expected <= set(METHOD_REGISTRY)
    assert all("available" in row for row in method_registry_rows())


def test_runtime_contract_gates_bootstrap_relations():
    X, y = make_classification(n_samples=60, n_features=6, random_state=1)
    forest = RandomForestClassifier(n_estimators=10, random_state=2).fit(X, y)
    caps = capabilities_for(forest)
    assert {"original", "oob", "rfgap", "path"} <= set(caps.valid_relations)
    extra = ExtraTreesClassifier(n_estimators=10, bootstrap=False, random_state=2).fit(X, y)
    assert "rfgap" not in capabilities_for(extra).valid_relations


def test_isolation_forest_only_exposes_distinct_collision_relation():
    X, _ = make_classification(n_samples=50, n_features=5, random_state=3)
    estimator = IsolationForest(random_state=4).fit(X)
    assert capabilities_for(estimator).valid_relations == ("collision",)
    with pytest.raises(UnsupportedRelationError):
        validate_relation(estimator, "oob")


def test_synthetic_null_persists_generator_and_seed():
    X, _ = make_classification(n_samples=50, n_features=5, random_state=5)
    model = SyntheticNullForest(random_state=91).fit(X)
    assert model.generator_ == "independent_marginals"
    assert model.synthetic_seed_ == 91
    assert model.synthetic_X_.shape == X.shape
    assert model.score_samples(X).shape == (len(X),)
