from __future__ import annotations

import unittest

import numpy as np

from annealing import AnnealingConfig
from configs import DatasetConfig, TrainingConfig
from services.online_annealing_training_service import (
    OnlineAnnealingRequest,
    create_online_session,
    evaluate_online_start,
    online_step_once,
)


class OnlineAnnealingTrainingServiceTests(unittest.TestCase):
    def _request(
        self,
        *,
        seed: int = 12,
        temperature: float = 1.0,
        max_steps: int = 4,
        train_policy: str = "accepted",
        batch_size: int = 16,
    ) -> OnlineAnnealingRequest:
        return OnlineAnnealingRequest(
            dataset_config=DatasetConfig(name="concentric_circles", random_state=seed),
            hidden_sizes=(8,),
            layout_spec="relu",
            training_config=TrainingConfig(
                epochs=2,
                learning_rate=0.03,
                batch_size=batch_size,
                random_state=seed,
                shuffle=True,
            ),
            annealing_config=AnnealingConfig(
                start_temperature=temperature,
                cooling_schedule="geometric",
                cooling_parameter=0.9,
                iterations_per_temperature=2,
                max_steps=max_steps,
                min_temperature=0.0,
                neighborhood_operations=("set_neuron",),
            ),
            weight_scale=0.05,
            random_state=seed,
            train_policy=train_policy,
        )

    def test_start_evaluation_uses_one_batch_and_does_not_train(self) -> None:
        session = create_online_session(self._request(seed=4))
        before_weights = [weight.copy() for weight in session.model.weights]

        snapshot = evaluate_online_start(session, "en")

        self.assertTrue(snapshot.is_initialized)
        self.assertEqual(snapshot.sa_evaluation_mode, "online_delta")
        self.assertEqual(snapshot.batch_index, 0)
        self.assertIsNotNone(snapshot.current_evaluation)
        for before, after in zip(before_weights, session.model.weights, strict=True):
            np.testing.assert_allclose(before, after)

    def test_candidate_evaluation_does_not_include_training_update(self) -> None:
        session = create_online_session(self._request(seed=7, temperature=1_000_000.0))
        evaluate_online_start(session, "en")
        before_weights = [weight.copy() for weight in session.model.weights]

        snapshot = online_step_once(session, "en")
        assert snapshot.last_step is not None
        step = snapshot.last_step

        for before, candidate_weight in zip(
            before_weights,
            step.candidate_evaluation.trained_model.weights,
            strict=True,
        ):
            np.testing.assert_allclose(before, candidate_weight)
        self.assertTrue(step.accepted)
        self.assertTrue(step.trained_after_accept)
        self.assertAlmostEqual(step.delta, step.candidate_loss_after - step.batch_loss_before)
        if step.delta <= 0:
            self.assertEqual(step.acceptance_probability, 1.0)
        self.assertTrue(
            any(
                not np.allclose(before, after)
                for before, after in zip(before_weights, session.model.weights, strict=True)
            )
        )

    def test_rejected_candidate_restores_previous_layout(self) -> None:
        rejected_snapshot = None
        rejected_session = None
        for seed in range(1, 80):
            session = create_online_session(
                self._request(seed=seed, temperature=1e-12, train_policy="none")
            )
            evaluate_online_start(session, "en")
            previous_layout = session.model.layout.to_compact_spec()
            snapshot = online_step_once(session, "en")
            if snapshot.last_step is not None and not snapshot.last_step.accepted:
                rejected_snapshot = snapshot
                rejected_session = session
                self.assertEqual(session.model.layout.to_compact_spec(), previous_layout)
                break
        self.assertIsNotNone(rejected_snapshot)
        self.assertIsNotNone(rejected_session)

    def test_batch_cursor_advances_and_epoch_increments(self) -> None:
        session = create_online_session(
            self._request(seed=8, temperature=1_000_000.0, max_steps=20, batch_size=64)
        )
        evaluate_online_start(session, "en")
        first_epoch = session.batch_cursor.epoch_index

        snapshot = None
        for _ in range(8):
            snapshot = online_step_once(session, "en")
        assert snapshot is not None

        self.assertGreaterEqual(session.batch_cursor.epoch_index, first_epoch)
        self.assertGreater(len(snapshot.history), 0)
        self.assertIsNotNone(snapshot.best_evaluation)
        assert snapshot.best_evaluation is not None
        self.assertTrue(np.isfinite(snapshot.best_evaluation.val_loss))


if __name__ == "__main__":
    unittest.main()
