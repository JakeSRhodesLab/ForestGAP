import unittest

import numpy as np

from forestgap.datasets import (
    SIMULATION_DATASET_GENERATORS,
    SyntheticDataset,
    load_simulated_dataset,
)


class TestSimulatedDatasets(unittest.TestCase):
    def test_all_registered_generators_return_binary_classification_datasets(self):
        for i, (name, generator) in enumerate(SIMULATION_DATASET_GENERATORS.items()):
            with self.subTest(dataset=name):
                dataset = generator(random_state=100 + i)
                self.assertIsInstance(dataset, SyntheticDataset)
                self.assertEqual(dataset.name, name)
                self.assertEqual(dataset.X.shape[0], dataset.y.shape[0])
                self.assertEqual(len(dataset.feature_names), dataset.X.shape[1])
                self.assertTrue(np.isfinite(dataset.X).all())
                self.assertSetEqual(set(np.unique(dataset.y)), {0, 1})

    def test_load_simulated_dataset_dispatches_registry(self):
        dataset = load_simulated_dataset(
            "rotated_subspace_signal",
            n_samples=120,
            n_features=8,
            random_state=7,
        )
        self.assertEqual(dataset.name, "rotated_subspace_signal")
        self.assertEqual(dataset.X.shape, (120, 8))
        self.assertEqual(dataset.y.shape, (120,))

    def test_load_simulated_dataset_rejects_unknown_name(self):
        with self.assertRaises(ValueError):
            load_simulated_dataset("not-a-real-simulation")


if __name__ == "__main__":
    unittest.main()
