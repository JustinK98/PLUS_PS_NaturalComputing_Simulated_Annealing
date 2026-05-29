from __future__ import annotations

import unittest

import numpy as np

from activations import parse_layout_spec
from benchmarks import load_benchmark
from configs import DatasetConfig, default_epochs, default_hidden_sizes
from model import ModularMLP


class BenchmarkSuiteTests(unittest.TestCase):
    def test_official_benchmark_defaults_match_shared_spec(self) -> None:
        self.assertEqual(default_hidden_sizes("two_moons"), (8,))
        self.assertEqual(default_hidden_sizes("concentric_circles"), (8, 8))
        self.assertEqual(default_hidden_sizes("crossing_spirals"), (16, 16))
        self.assertEqual(default_epochs("two_moons"), 100)
        self.assertEqual(default_epochs("concentric_circles"), 150)
        self.assertEqual(default_epochs("crossing_spirals"), 250)

    def test_two_moons_bundle_uses_binary_output_metadata(self) -> None:
        dataset = load_benchmark(DatasetConfig(name="two_moons", random_state=5))

        self.assertEqual(dataset.input_size, 2)
        self.assertEqual(dataset.output_size, 2)
        self.assertEqual(dataset.model_output_size, 1)
        self.assertEqual(len(dataset.feature_names), 2)
        self.assertEqual(len(dataset.target_names), 2)

    def test_official_csv_benchmarks_have_expected_shapes(self) -> None:
        expected = {
            "two_moons": (2, 2, 1),
            "concentric_circles": (2, 2, 1),
            "crossing_spirals": (6, 2, 1),
        }
        for benchmark, (inputs, classes, model_outputs) in expected.items():
            with self.subTest(benchmark=benchmark):
                dataset = load_benchmark(DatasetConfig(name=benchmark, random_state=11))
                self.assertEqual(dataset.input_size, inputs)
                self.assertEqual(dataset.output_size, classes)
                self.assertEqual(dataset.model_output_size, model_outputs)
                self.assertGreater(dataset.train_size, 0)
                self.assertGreater(dataset.validation_size, 0)
                self.assertGreater(dataset.test_size, 0)

    def test_iris_is_not_an_official_benchmark_anymore(self) -> None:
        with self.assertRaises(ValueError):
            load_benchmark(DatasetConfig(name="iris", random_state=11))

    def test_binary_output_model_still_returns_two_class_probabilities(self) -> None:
        dataset = load_benchmark(DatasetConfig(name="two_moons", random_state=7))
        layout = parse_layout_spec("swish*8", (8,))
        model = ModularMLP(
            input_size=dataset.input_size,
            hidden_sizes=(8,),
            output_size=dataset.model_output_size,
            layout=layout,
            num_classes=dataset.output_size,
            random_state=7,
        )

        probabilities = model.predict_proba(dataset.X_val[:6])
        predictions = model.predict(dataset.X_val[:6])
        loss, gradients = model.loss_and_gradients(dataset.X_train[:12], dataset.y_train[:12])

        self.assertEqual(probabilities.shape, (6, 2))
        self.assertEqual(predictions.shape, (6,))
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6))
        self.assertGreaterEqual(loss, 0.0)
        self.assertEqual(gradients["weights"][-1].shape[1], 1)

    def test_xavier_initialization_and_zero_biases(self) -> None:
        dataset = load_benchmark(DatasetConfig(name="two_moons", random_state=7))
        layout = parse_layout_spec("relu", (8,))
        model = ModularMLP(
            input_size=dataset.input_size,
            hidden_sizes=(8,),
            output_size=dataset.model_output_size,
            layout=layout,
            num_classes=dataset.output_size,
            random_state=7,
        )

        for weight in model.weights:
            self.assertTrue(np.all(np.isfinite(weight)))
            self.assertGreater(float(np.max(weight)), 0.0)
            self.assertLess(float(np.min(weight)), 0.0)
        for bias in model.biases:
            np.testing.assert_allclose(bias, np.zeros_like(bias))

    def test_extended_activation_set_parses_cleanly(self) -> None:
        layout = parse_layout_spec("identity*8|swish*4", (8, 4))
        self.assertEqual(layout.layers[0][0], "identity")
        self.assertEqual(layout.layers[1][0], "swish")


if __name__ == "__main__":
    unittest.main()
