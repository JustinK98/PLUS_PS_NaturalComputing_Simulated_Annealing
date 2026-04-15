from __future__ import annotations

import unittest

from activations import parse_layout_spec
from benchmarks import load_benchmark
from configs import DatasetConfig, TrainingConfig
from model import ModularMLP
from services.training_session_service import (
    create_training_session,
    model_from_training_session,
    run_training_session_epochs,
)


class TrainingSessionServiceTests(unittest.TestCase):
    def test_training_session_accumulates_history_across_chunks(self) -> None:
        dataset = load_benchmark(DatasetConfig(name="test_activation", random_state=11))
        model = ModularMLP(
            input_size=dataset.input_size,
            hidden_sizes=(4, 3),
            output_size=dataset.output_size,
            layout=parse_layout_spec("relu|tanh", (4, 3)),
            random_state=11,
        )
        session = create_training_session(model)

        config = TrainingConfig(
            epochs=1,
            learning_rate=0.03,
            batch_size=4,
            random_state=11,
            shuffle=True,
        )
        session = run_training_session_epochs(session, dataset, config)
        self.assertEqual(session.completed_epochs, 1)
        self.assertEqual(len(session.history["train_loss"]), 1)

        session = run_training_session_epochs(session, dataset, config)
        self.assertEqual(session.completed_epochs, 2)
        self.assertEqual(len(session.history["train_loss"]), 2)

        reconstructed = model_from_training_session(session)
        self.assertEqual(reconstructed.hidden_sizes, model.hidden_sizes)
        self.assertIsNotNone(session.test_metrics)


if __name__ == "__main__":
    unittest.main()
