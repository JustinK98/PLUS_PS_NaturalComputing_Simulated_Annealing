from __future__ import annotations

import unittest

from services.presentation_service import (
    PRESENTATION_BENCHMARK,
    PRESENTATION_HIDDEN_SIZES,
    create_presentation_annealing_session,
    create_presentation_state,
    presentation_training_config,
)


class PresentationServiceTests(unittest.TestCase):
    def test_presentation_state_uses_reproducible_breast_cancer_defaults(self) -> None:
        state = create_presentation_state("en")
        self.assertEqual(state.dataset.name, PRESENTATION_BENCHMARK)
        self.assertEqual(state.model.hidden_sizes, PRESENTATION_HIDDEN_SIZES)
        self.assertEqual(state.analysis_sample.split_name, "val")
        self.assertEqual(state.selected_hidden, (0, 0))

    def test_presentation_training_and_annealing_defaults_are_small(self) -> None:
        training_config = presentation_training_config(10)
        self.assertEqual(training_config.epochs, 10)
        session = create_presentation_annealing_session()
        self.assertEqual(session.request.dataset_config.name, "breast_cancer")
        self.assertEqual(session.request.annealing_config.max_steps, 8)
        self.assertEqual(session.request.objective_config.objective_name, "validation_loss")


if __name__ == "__main__":
    unittest.main()
