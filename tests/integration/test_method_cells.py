from experiments.run_method_matrix import build_cells
from experiments.harness import run_cells


def test_cross_method_cells_cover_cartesian_matrix_and_keep_unsupported_rows():
    cells = build_cells(
        estimators=["random_forest", "ccf", "ccf_bag"],
        relations=["original", "rfgap"],
        applications=["prediction", "retrieval"],
        data_regimes=["axis_aligned", "rotated_oblique"],
        seeds=[0],
        n_samples=70,
        n_estimators=5,
    )
    assert len(cells) == 24
    rows = run_cells(cells)
    assert len(rows) == len(cells)
    assert {row["status"] for row in rows} == {"ok", "unsupported"}
    required = {
        "revision", "versions", "configuration", "seed", "split_hash",
        "parent_score", "runtime_seconds", "peak_memory_bytes",
    }
    assert all(required <= set(row) for row in rows if row["status"] == "ok")
    assert all(
        row["status"] == "unsupported"
        for row in rows
        if row["estimator"] == "ccf" and row["relation"] == "rfgap"
    )
    assert all(
        row["status"] == "ok"
        for row in rows
        if row["estimator"] == "ccf_bag" and row["relation"] == "rfgap"
    )
