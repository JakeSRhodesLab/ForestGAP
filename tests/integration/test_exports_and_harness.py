from pathlib import Path

from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

import forestgap
from experiments.harness import ExperimentCell, run_cells


def test_removed_wrapper_surface_and_public_composition_exports():
    assert forestgap.__version__ == "1.1.0"
    assert hasattr(forestgap, "Proximity")
    assert not hasattr(forestgap, "with_sparse_proximity")
    assert not hasattr(forestgap, "SparseProximityMixin")
    assert not (Path(forestgap.__file__).parent / "core.py").exists()


def test_harness_reuses_parent_and_records_unsupported_cells():
    X, y = make_classification(n_samples=60, n_features=6, random_state=40)
    fits = {"count": 0}

    class CountingForest(RandomForestClassifier):
        def fit(self, X, y, **kwargs):
            fits["count"] += 1
            return super().fit(X, y, **kwargs)

    def metrics(proximity, cell):
        return {"matrix_rows": proximity.get_proximities().shape[0]}

    cells = [
        ExperimentCell(
            "rf", relation, "smoke", "synthetic", lambda: CountingForest(n_estimators=12, random_state=41),
            metrics, X, y, seed=41, split_id="shared",
        )
        for relation in ("original", "kerf")
    ]
    rows = run_cells(cells)
    assert fits["count"] == 1
    assert [row["status"] for row in rows] == ["ok", "ok"]
    assert all("relation_diagnostics" in row and "versions" in row for row in rows)

    unsupported = run_cells([
        ExperimentCell(
            "boosted", "rfgap", "smoke", "synthetic",
            lambda: HistGradientBoostingClassifier(random_state=42), metrics,
            X, y, seed=42,
        )
    ])[0]
    assert unsupported["status"] == "unsupported"
    assert unsupported["runtime_seconds"] >= 0
    assert unsupported["peak_memory_bytes"] >= 0
    assert unsupported["relation_runtime_seconds"] is not None
    assert unsupported["relation_diagnostics"] is None


def test_harness_does_not_reuse_different_parent_configurations():
    X, y = make_classification(n_samples=50, n_features=5, random_state=43)
    fits = {"count": 0}

    class CountingForest(RandomForestClassifier):
        def fit(self, X, y, **kwargs):
            fits["count"] += 1
            return super().fit(X, y, **kwargs)

    def metrics(proximity, cell):
        return {"rows": proximity.get_proximities().shape[0]}

    rows = run_cells([
        ExperimentCell(
            "rf", "original", "smoke", "synthetic",
            lambda depth=depth: CountingForest(
                n_estimators=5, max_depth=depth, random_state=44,
            ),
            metrics, X, y, seed=44, split_id="same-split",
            configuration={"max_depth": depth},
        )
        for depth in (2, 4)
    ])
    assert fits["count"] == 2
    assert all(row["status"] == "ok" for row in rows)


def test_harness_error_rows_keep_timing_memory_and_relation_diagnostics():
    X, y = make_classification(n_samples=45, n_features=5, random_state=45)

    def fail_after_relation(proximity, cell):
        raise RuntimeError("deliberate downstream failure")

    row = run_cells([
        ExperimentCell(
            "rf", "original", "failure", "synthetic",
            lambda: RandomForestClassifier(n_estimators=5, random_state=46),
            fail_after_relation, X, y, seed=46,
        )
    ])[0]
    assert row["status"] == "error"
    assert row["application_runtime_seconds"] is not None
    assert row["peak_memory_bytes"] > 0
    assert row["relation_diagnostics"] is not None
