from __future__ import annotations

import unittest

from annealing import AnnealingConfig
from annealing_objectives import ObjectiveConfig
from configs import DatasetConfig
from services.annealing_service import AnnealingRunRequest
from services.annealing_session_service import (
    create_session,
    evaluate_start,
    reset,
    run_to_completion,
    step_once,
)


class AnnealingSessionServiceTests(unittest.TestCase):
    def _request(self) -> AnnealingRunRequest:
        return AnnealingRunRequest(
            dataset_config=DatasetConfig(name="test_activation", random_state=9),
            hidden_sizes=(4, 3),
            layout_spec="relu|tanh",
            objective_config=ObjectiveConfig(
                objective_name="validation_loss",
                candidate_epochs=2,
                learning_rate=0.03,
                batch_size=4,
                weight_scale=0.05,
                random_state=9,
                shuffle=True,
            ),
            annealing_config=AnnealingConfig(
                start_temperature=1.0,
                cooling_schedule="geometric",
                cooling_parameter=0.9,
                iterations_per_temperature=2,
                max_steps=4,
                min_temperature=0.01,
                neighborhood_operations=("set_neuron",),
            ),
            random_state=9,
        )

    def test_session_can_evaluate_step_reset_and_complete(self) -> None:
        session = create_session(self._request())
        start_snapshot = evaluate_start(session, "en")
        self.assertTrue(start_snapshot.is_initialized)
        self.assertIsNotNone(start_snapshot.start_evaluation)

        step_snapshot = step_once(session, "en")
        self.assertGreaterEqual(len(step_snapshot.history), 1)
        self.assertIsNotNone(step_snapshot.current_evaluation)

        reset_snapshot = reset(session, "en")
        self.assertFalse(reset_snapshot.is_initialized)

        complete_snapshot = run_to_completion(session, "en")
        self.assertTrue(complete_snapshot.is_initialized)
        self.assertTrue(complete_snapshot.is_complete)
        self.assertIsNotNone(complete_snapshot.best_evaluation)


if __name__ == "__main__":
    unittest.main()
