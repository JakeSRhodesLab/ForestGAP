import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from forestgap.datasets import (
    OpenMLBenchmarkSuite,
    OpenMLDataset,
    load_openml_dataset,
    load_openml_suite,
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


if __name__ == "__main__":
    unittest.main()
