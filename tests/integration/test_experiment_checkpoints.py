from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import (
    CheckpointManifest,
    ExperimentCheckpointStore,
    PreparedDataset,
    prepared_dataset_hash,
    save_dataframe,
)


def _manifest(index: int = 0, *, config_value: int = 1) -> CheckpointManifest:
    return CheckpointManifest(
        experiment="checkpoint_test",
        unit={"task_id": 7, "seed": index, "model": "RandomForest"},
        scientific_config={"n_estimators": config_value},
        data_hash="data-hash",
        source_hash="source-hash",
        environment={"python": "test", "packages": {"numpy": "test"}},
    )


def _commit_in_subprocess(arguments: tuple[str, int]) -> str:
    root, index = arguments
    store = ExperimentCheckpointStore(Path(root) / "checkpoints.sqlite3")
    manifest = _manifest(index)
    store.save(
        manifest,
        [{"status": "ok", "seed": index, "value": np.float64(index)}],
    )
    return manifest.checkpoint_key


def test_checkpoint_manifest_is_canonical_and_scientific_changes_invalidate_it():
    left = _manifest(3, config_value=8)
    reordered = CheckpointManifest(
        experiment=left.experiment,
        unit={"model": "RandomForest", "seed": 3, "task_id": 7},
        scientific_config={"n_estimators": 8},
        data_hash=left.data_hash,
        source_hash=left.source_hash,
        environment=left.environment,
    )

    assert left.checkpoint_key == reordered.checkpoint_key
    assert left.checkpoint_key != _manifest(3, config_value=9).checkpoint_key
    assert left.checkpoint_key != _manifest(4, config_value=8).checkpoint_key


def test_checkpoint_round_trip_is_atomic_and_preserves_special_values(tmp_path):
    store = ExperimentCheckpointStore(tmp_path / "checkpoints.sqlite3")
    manifest = _manifest()
    bundle = store.save(
        manifest,
        [{"status": "ok", "nan": np.nan, "positive": np.inf, "count": np.int64(4)}],
        detail_rows=[{"center": 2, "score": np.float32(0.25)}],
        artifacts={"native_scores": np.asarray([1.0, np.nan, -np.inf])},
    )

    loaded = store.load(manifest)
    assert loaded is not None
    assert loaded.summary_rows[0]["checkpoint_key"] == bundle.summary_rows[0]["checkpoint_key"]
    assert loaded.summary_rows[0]["count"] == bundle.summary_rows[0]["count"]
    assert np.isnan(loaded.summary_rows[0]["nan"])
    assert loaded.summary_rows[0]["positive"] == np.inf
    assert loaded.detail_rows[0]["score"] == np.float32(0.25)
    np.testing.assert_equal(
        loaded.artifacts["native_scores"],
        np.asarray([1.0, np.nan, -np.inf]),
    )


def test_error_attempt_is_retryable_and_does_not_replace_last_good_bundle(tmp_path):
    store = ExperimentCheckpointStore(tmp_path / "checkpoints.sqlite3")
    manifest = _manifest()

    store.save(manifest, [{"status": "error", "error_message": "first"}])
    assert store.load(manifest) is None

    store.save(manifest, [{"status": "ok", "value": 17}])
    store.save(manifest, [{"status": "error", "error_message": "forced failure"}])
    loaded = store.load(manifest)
    assert loaded is not None
    assert loaded.summary_rows[0]["status"] == "ok"
    assert loaded.summary_rows[0]["value"] == 17


def test_checkpoint_store_accepts_concurrent_process_commits(tmp_path):
    arguments = [(str(tmp_path), index) for index in range(12)]
    with ProcessPoolExecutor(max_workers=4) as executor:
        keys = list(executor.map(_commit_in_subprocess, arguments))

    assert len(set(keys)) == len(arguments)
    store = ExperimentCheckpointStore(tmp_path / "checkpoints.sqlite3")
    for index in range(len(arguments)):
        loaded = store.load(_manifest(index))
        assert loaded is not None
        assert loaded.summary_rows[0]["seed"] == index


def test_prepared_dataset_hash_uses_object_values_not_object_addresses():
    def prepared() -> PreparedDataset:
        return PreparedDataset(
            task_kind="classification",
            suite_id=99,
            suite_name="suite",
            task_id=3,
            dataset_id=4,
            dataset_name="dataset",
            feature_names=["numeric", "category"],
            categorical_indices=[1],
            X_train=np.asarray([[1.0, 0.0], [2.0, 1.0]]),
            X_test=np.asarray([[3.0, 0.0]]),
            y_train=np.asarray(["a", "b"], dtype=object),
            y_test=np.asarray(["a"], dtype=object),
            original_n_samples=3,
            original_n_features=2,
            train_size=2,
            test_size=1,
        )

    assert prepared_dataset_hash(prepared()) == prepared_dataset_hash(prepared())
    changed = prepared()
    changed.y_train[0] = "different"
    assert prepared_dataset_hash(prepared()) != prepared_dataset_hash(changed)


def test_dataframe_export_replaces_existing_file(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text("old\nvalue\n", encoding="utf-8")
    save_dataframe(pd.DataFrame([{"new": 1}]), path)

    assert pd.read_csv(path).to_dict(orient="records") == [{"new": 1}]
    assert not list(tmp_path.glob(".raw.csv.*.tmp"))
