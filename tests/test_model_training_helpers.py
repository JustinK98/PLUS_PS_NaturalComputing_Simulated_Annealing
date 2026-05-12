from __future__ import annotations

import unittest

import numpy as np

from activations import parse_layout_spec
from benchmarks import load_benchmark
from configs import DatasetConfig
from model import ModularMLP
from trainer import evaluate_batch, train_one_batch


class ModelTrainingHelperTests(unittest.TestCase):
    def _model(self) -> ModularMLP:
        dataset = load_benchmark(DatasetConfig(name="concentric_circles", random_state=5))
        layout = parse_layout_spec("relu", (8,))
        return ModularMLP(
            input_size=dataset.input_size,
            hidden_sizes=(8,),
            output_size=dataset.model_output_size,
            layout=layout,
            num_classes=dataset.output_size,
            random_state=5,
        )

    def test_set_layout_preserves_weights_and_rejects_wrong_shape(self) -> None:
        model = self._model()
        before_weights = [weight.copy() for weight in model.weights]
        before_biases = [bias.copy() for bias in model.biases]

        model.set_layout(parse_layout_spec("tanh", (8,)))

        self.assertEqual(model.layout.to_compact_spec(), "tanh*8")
        for before, after in zip(before_weights, model.weights, strict=True):
            np.testing.assert_allclose(before, after)
        for before, after in zip(before_biases, model.biases, strict=True):
            np.testing.assert_allclose(before, after)
        with self.assertRaises(ValueError):
            model.set_layout(parse_layout_spec("relu|tanh", (8, 4)))

    def test_batch_helpers_evaluate_and_train_one_batch(self) -> None:
        dataset = load_benchmark(DatasetConfig(name="concentric_circles", random_state=6))
        model = self._model()
        X_batch = dataset.X_train[:16]
        y_batch = dataset.y_train[:16]

        helper_loss, helper_acc = evaluate_batch(model, X_batch, y_batch)
        direct_loss, direct_acc = model.evaluate(X_batch, y_batch)
        self.assertAlmostEqual(helper_loss, direct_loss)
        self.assertAlmostEqual(helper_acc, direct_acc)

        before_weights = [weight.copy() for weight in model.weights]
        batch_loss = train_one_batch(model, X_batch, y_batch, learning_rate=0.03)
        self.assertTrue(np.isfinite(batch_loss))
        self.assertTrue(
            any(
                not np.allclose(before, after)
                for before, after in zip(before_weights, model.weights, strict=True)
            )
        )


if __name__ == "__main__":
    unittest.main()
