from pathlib import Path

from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

import forestgap
from experiments.harness import ExperimentCell, run_cells


def test_removed_wrapper_surface_and_public_composition_exports():
    assert forestgap.__version__ == "1.0.0"
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
