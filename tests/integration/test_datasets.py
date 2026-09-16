import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from forestgap.datasets import (
    OpenMLBenchmarkSuite,
    OpenMLDataset,
    OpenMLTask,
    OpenMLTaskFold,
    iter_openml_task_folds,
    load_openml_dataset,
    load_openml_suite,
    load_openml_task_fold,
)


class TestOpenMLDatasetLoading(unittest.TestCase):
    def test_load_openml_dataset_requires_single_identifier(self):
        with self.assertRaises(ValueError):
            load_openml_dataset()

        with self.assertRaises(ValueError):
            load_openml_dataset(name="iris", dataset_id=61)

    def test_load_openml_suite_requires_single_identifier(self):
        with self.assertRaises(ValueError):
            load_openml_suite()

        with self.assertRaises(ValueError):
            load_openml_suite(suite_id=1, suite_name="OpenML-CC18")

    def test_load_openml_dataset_returns_normalized_dataclass(self):
        fake_dataset = SimpleNamespace(
            dataset_id=61,
            name="iris",
            version=1,
            default_target_attribute="class",
            description="classic iris dataset",
            url="https://www.openml.org/d/61",
            row_id_attribute=None,
            ignore_attribute=None,
            citation="citation",
            creator="Fisher",
            collection_date="1936",
            language="en",
            licence="Public",
        )
        fake_dataset.get_data = lambda target, dataset_format: (
            [["1.0", "2.0"]],
            ["setosa"],
            [False, False],
            ["sepal_length", "sepal_width"],
        )
        fake_openml = SimpleNamespace(
            datasets=SimpleNamespace(
                list_datasets=lambda **kwargs: {
                    61: {"name": "iris", "version": 1},
                },
                get_dataset=lambda **kwargs: fake_dataset,
            ),
        )

        with patch("forestgap.datasets._require_openml", return_value=fake_openml):
            loaded = load_openml_dataset(name="iris")

        self.assertIsInstance(loaded, OpenMLDataset)
        self.assertEqual(loaded.name, "iris")
        self.assertEqual(loaded.dataset_id, 61)
        self.assertEqual(loaded.target_name, "class")
        self.assertEqual(loaded.feature_names, ["sepal_length", "sepal_width"])
        self.assertEqual(loaded.categorical_indicator, [False, False])

    def test_load_openml_dataset_splits_multi_target_defaults(self):
        fake_dataset = SimpleNamespace(
            dataset_id=42871,
            name="iris",
            version=2,
            default_target_attribute="L1,L2,L3",
            description="iris with one-hot labels",
            url="https://www.openml.org/d/42871",
            row_id_attribute=None,
            ignore_attribute=None,
            citation=None,
            creator=None,
            collection_date=None,
            language=None,
            licence=None,
        )
        frame = pd.DataFrame(
            {
                "A1": [5.1, 4.9],
                "A2": [3.5, 3.0],
                "L1": [1, 1],
                "L2": [0, 0],
                "L3": [0, 0],
            }
        )
        fake_dataset.get_data = lambda target, dataset_format: (
            frame,
            None,
            [False, False, False, False, False],
            ["A1", "A2", "L1", "L2", "L3"],
        )
        fake_openml = SimpleNamespace(
            datasets=SimpleNamespace(
                list_datasets=lambda **kwargs: {
                    42871: {"name": "iris", "version": 2},
                },
                get_dataset=lambda **kwargs: fake_dataset,
            ),
        )

        with patch("forestgap.datasets._require_openml", return_value=fake_openml):
            loaded = load_openml_dataset(name="iris")

        self.assertEqual(loaded.target_name, ["L1", "L2", "L3"])
        self.assertEqual(list(loaded.X.columns), ["A1", "A2"])
        self.assertEqual(list(loaded.y.columns), ["L1", "L2", "L3"])
        self.assertEqual(loaded.feature_names, ["A1", "A2"])
        self.assertEqual(loaded.categorical_indicator, [False, False])

    def test_load_openml_dataset_falls_back_to_frame_split_on_openml_indexerror(self):
        fake_dataset = SimpleNamespace(
            dataset_id=42585,
            name="iris",
            version=1,
            default_target_attribute="class",
            description="iris",
            url="https://www.openml.org/d/42585",
            row_id_attribute=None,
            ignore_attribute=None,
            citation=None,
            creator=None,
            collection_date=None,
            language=None,
            licence=None,
        )
        frame = pd.DataFrame(
            {
                "A1": [5.1, 4.9],
                "A2": [3.5, 3.0],
                "class": ["setosa", "setosa"],
            }
        )

        def fake_get_data(target, dataset_format):
            if target == "class":
                raise IndexError("list index out of range")
            return frame, None, [False, False, True], ["A1", "A2", "class"]

        fake_dataset.get_data = fake_get_data
        fake_openml = SimpleNamespace(
            datasets=SimpleNamespace(
                list_datasets=lambda **kwargs: {
                    42585: {"name": "iris", "version": 1},
                },
                get_dataset=lambda **kwargs: fake_dataset,
            ),
        )

        with patch("forestgap.datasets._require_openml", return_value=fake_openml):
            loaded = load_openml_dataset(name="iris")

        self.assertEqual(loaded.target_name, "class")
        self.assertEqual(list(loaded.X.columns), ["A1", "A2"])
        self.assertEqual(list(loaded.y.astype(str)), ["setosa", "setosa"])
        self.assertEqual(loaded.feature_names, ["A1", "A2"])

    def test_load_openml_dataset_retries_parquet_extension_error_with_arff(self):
        fake_dataset = SimpleNamespace(
            dataset_id=28,
            name="optdigits",
            version=1,
            default_target_attribute="class",
            description="digits",
            url="https://www.openml.org/d/28",
            row_id_attribute=None,
            ignore_attribute=None,
            citation=None,
            creator=None,
            collection_date=None,
            language=None,
            licence=None,
            parquet_file="dataset_28.pq",
        )
        calls = []

        def fake_get_data(target, dataset_format):
            calls.append((target, dataset_format))
            if len(calls) == 1:
                try:
                    raise Exception(
                        "No type extension with name arrow.py_extension_type found"
                    )
                except Exception as arrow_error:
                    raise Exception("File: dataset_28.pq") from arrow_error
            self.assertIsNone(fake_dataset.parquet_file)
            self.assertEqual(os.environ["OPENML_SKIP_PARQUET"], "true")
            return [[1.0]], [0], [False], ["pixel_0"]

        fake_dataset.get_data = fake_get_data
        fake_openml = SimpleNamespace(
            datasets=SimpleNamespace(
                get_dataset=lambda **kwargs: fake_dataset,
            ),
        )

        with patch.dict("forestgap.datasets.os.environ", {"OPENML_SKIP_PARQUET": "false"}):
            with patch("forestgap.datasets._require_openml", return_value=fake_openml):
                loaded = load_openml_dataset(dataset_id=28, as_frame=False)

            self.assertEqual(os.environ["OPENML_SKIP_PARQUET"], "false")

        self.assertEqual(calls, [("class", "array"), ("class", "array")])
        self.assertEqual(loaded.X, [[1.0]])
        self.assertEqual(loaded.y, [0])

    def test_load_openml_suite_loads_each_task_dataset(self):
        fake_suite = SimpleNamespace(
            name="Toy Suite",
            description="benchmark",
            tasks=[11, 12],
            alias="toy",
            main_entity_type="task",
        )
        fake_tasks = {
            11: SimpleNamespace(dataset_id=101, target_name="target"),
            12: SimpleNamespace(dataset_id=102, target_name="label"),
        }
        fake_openml = SimpleNamespace(
            study=SimpleNamespace(
                get_suite=lambda suite_id: fake_suite,
                list_suites=lambda output_format="dict": {7: {"name": "Toy Suite"}},
            ),
            tasks=SimpleNamespace(get_task=lambda task_id: fake_tasks[task_id]),
        )

        loaded_datasets = [
            OpenMLDataset(
                name="ds1",
                dataset_id=101,
                version=1,
                target_name="target",
                X=[[1.0]],
                y=[0],
                feature_names=["x1"],
                categorical_indicator=[False],
            ),
            OpenMLDataset(
                name="ds2",
                dataset_id=102,
                version=1,
                target_name="label",
                X=[[2.0]],
                y=[1],
                feature_names=["x2"],
                categorical_indicator=[False],
            ),
        ]

        with patch("forestgap.datasets._require_openml", return_value=fake_openml):
            with patch("forestgap.datasets.load_openml_dataset", side_effect=loaded_datasets) as mocked_loader:
                suite = load_openml_suite(suite_name="Toy Suite")

        self.assertIsInstance(suite, OpenMLBenchmarkSuite)
        self.assertEqual(suite.suite_id, 7)
        self.assertEqual(suite.task_ids, [11, 12])
        self.assertEqual([dataset.dataset_id for dataset in suite.datasets], [101, 102])
        self.assertEqual(mocked_loader.call_count, 2)

    def test_load_openml_suite_matches_alias_and_prefix(self):
        fake_suite = SimpleNamespace(
            name="OpenML-CTR23 - A curated tabular regression benchmarking suite",
            description="benchmark",
            tasks=[],
            alias="8f0ea660163b436bbd4abd49665c7b1d",
            main_entity_type="task",
        )
        fake_suites = pd.DataFrame(
            [
                {
                    "id": 99,
                    "name": "OpenML-CC18 Curated Classification benchmark",
                    "alias": "OpenML-CC18",
                },
                {
                    "id": 353,
                    "name": "OpenML-CTR23 - A curated tabular regression benchmarking suite",
                    "alias": "8f0ea660163b436bbd4abd49665c7b1d",
                },
            ]
        )
        fake_openml = SimpleNamespace(
            study=SimpleNamespace(
                get_suite=lambda suite_id: fake_suite,
                list_suites=lambda output_format="dataframe": fake_suites,
            ),
            tasks=SimpleNamespace(get_task=lambda task_id: None),
        )

        with patch("forestgap.datasets._require_openml", return_value=fake_openml):
            with patch("forestgap.datasets.load_openml_dataset", return_value=None):
                cc18 = load_openml_suite(suite_name="OpenML-CC18")
                ctr23 = load_openml_suite(suite_name="OpenML-CTR23")

        self.assertEqual(cc18.suite_id, 99)
        self.assertEqual(ctr23.suite_id, 353)

    def test_typed_openml_task_fold_preserves_official_indices(self):
        loaded_dataset = OpenMLDataset(
            name="study-455-toy",
            dataset_id=4550,
            version=1,
            target_name="target",
            X=pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}),
            y=pd.Series([0.0, 1.0, 2.0, 3.0]),
            feature_names=["x"],
            categorical_indicator=[False],
        )
        fake_task = SimpleNamespace(
            dataset_id=4550,
            target_name="target",
            task_type="Supervised Regression",
            task_type_id=2,
            get_split_dimensions=lambda: (1, 2, 1),
            get_train_test_split_indices=lambda *, repeat, fold, sample: (
                [0, 2] if fold == 0 else [1, 3],
                [1, 3] if fold == 0 else [0, 2],
            ),
        )
        fake_openml = SimpleNamespace(
            tasks=SimpleNamespace(get_task=lambda task_id, **kwargs: fake_task),
        )

        with patch("forestgap.datasets._require_openml", return_value=fake_openml):
            with patch("forestgap.datasets.load_openml_dataset", return_value=loaded_dataset):
                loaded = load_openml_task_fold(task_id=361085, fold=1)
                all_folds = list(iter_openml_task_folds(task_id=361085))

        self.assertIsInstance(loaded, OpenMLTaskFold)
        self.assertIsInstance(loaded.task, OpenMLTask)
        self.assertEqual(loaded.task_id, 361085)
        self.assertEqual(loaded.dataset_id, 4550)
        self.assertEqual(loaded.fold_id, "r0-f1-s0")
        self.assertListEqual(loaded.train_indices.tolist(), [1, 3])
        self.assertListEqual(loaded.test_indices.tolist(), [0, 2])
        self.assertEqual([fold.fold_id for fold in all_folds], ["r0-f0-s0", "r0-f1-s0"])

    def test_typed_openml_task_fold_rejects_overlapping_indices(self):
        loaded_dataset = OpenMLDataset(
            name="bad-fold",
            dataset_id=4551,
            version=1,
            target_name="target",
            X=[[0.0], [1.0], [2.0]],
            y=[0, 1, 2],
            feature_names=["x"],
            categorical_indicator=[False],
        )
        fake_task = SimpleNamespace(
            dataset_id=4551,
            target_name="target",
            get_split_dimensions=lambda: (1, 1, 1),
            get_train_test_split_indices=lambda **kwargs: ([0, 1], [1, 2]),
        )
        fake_openml = SimpleNamespace(
            tasks=SimpleNamespace(get_task=lambda task_id, **kwargs: fake_task),
        )

        with patch("forestgap.datasets._require_openml", return_value=fake_openml):
            with patch("forestgap.datasets.load_openml_dataset", return_value=loaded_dataset):
                with self.assertRaisesRegex(ValueError, "disjoint"):
                    load_openml_task_fold(task_id=361086)


if __name__ == "__main__":
    unittest.main()
