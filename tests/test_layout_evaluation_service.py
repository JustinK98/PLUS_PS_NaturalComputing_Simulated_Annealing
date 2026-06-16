from __future__ import annotations

import unittest

from configs import DatasetConfig, TrainingConfig
from services.layout_evaluation_service import (
    InheritedModelCandidate,
    LayoutCandidate,
    LayoutEvaluationRequest,
    best_layout_run,
    build_online_delta_layout_candidates,
    evaluate_inherited_models,
    run_layout_evaluation,
    with_inherited_model_evaluations,
)


class LayoutEvaluationServiceTests(unittest.TestCase):
    def test_active_online_delta_candidates_only_compare_random_start_and_end(self) -> None:
        candidates = build_online_delta_layout_candidates(
            start_layout_spec="relu,gelu",
            end_layout_spec="tanh,swish",
        )

        self.assertEqual(
            [candidate.label for candidate in candidates],
            ["random_start_layout", "end_sa_layout"],
        )

    def test_active_candidates_are_deduplicated(self) -> None:
        candidates = build_online_delta_layout_candidates(
            start_layout_spec="relu",
            end_layout_spec="relu",
        )

        self.assertEqual([candidate.label for candidate in candidates], ["random_start_layout"])

    def test_layout_evaluation_ranks_layouts_under_identical_conditions(self) -> None:
        candidates = (
            LayoutCandidate("random_start_layout", "relu"),
            LayoutCandidate("end_sa_layout", "tanh"),
        )
        result = run_layout_evaluation(
            LayoutEvaluationRequest(
                dataset_config=DatasetConfig(name="concentric_circles", random_state=1),
                hidden_sizes=(8,),
                candidates=candidates,
                seeds=(1,),
                training_config=TrainingConfig(
                    epochs=2,
                    learning_rate=0.03,
                    batch_size=32,
                    random_state=1,
                ),
                weight_scale=0.05,
                primary_metric="validation_loss",
            )
        )

        self.assertEqual(len(result.runs), 2)
        self.assertEqual(len(result.aggregated), 2)
        self.assertEqual(result.ranking[0].ranking_score, min(item.ranking_score for item in result.aggregated))
        self.assertIsNotNone(best_layout_run(result, "validation_loss").model_state)

    def test_inherited_model_evaluation_is_combined_with_retrained_ranking(self) -> None:
        candidates = (
            LayoutCandidate("random_start_layout", "relu"),
            LayoutCandidate("end_sa_layout", "tanh"),
        )
        request = LayoutEvaluationRequest(
            dataset_config=DatasetConfig(name="concentric_circles", random_state=2),
            hidden_sizes=(8,),
            candidates=candidates,
            seeds=(2,),
            training_config=TrainingConfig(
                epochs=1,
                learning_rate=0.03,
                batch_size=32,
                random_state=2,
            ),
            weight_scale=0.05,
            primary_metric="validation_loss",
        )
        result = run_layout_evaluation(request)
        inherited_runs = evaluate_inherited_models(
            request.dataset_config,
            (
                InheritedModelCandidate(
                    "diagnostic_best_inherited_model_from_sa",
                    result.runs[0].model_state,
                    2,
                ),
            ),
            primary_metric="validation_loss",
        )
        combined = with_inherited_model_evaluations(result, inherited_runs, "validation_loss")

        self.assertEqual(inherited_runs[0].evaluation_type, "inherited")
        self.assertEqual(inherited_runs[0].history, {})
        self.assertIn("test_loss", inherited_runs[0].metrics)
        self.assertTrue(all(item.evaluation_type in {"retrained", "inherited"} for item in combined.combined_ranking))
        self.assertEqual(
            combined.combined_ranking[0].ranking_score,
            min(item.ranking_score for item in combined.combined_ranking),
        )

    def test_layout_evaluation_can_skip_test_metrics(self) -> None:
        result = run_layout_evaluation(
            LayoutEvaluationRequest(
                dataset_config=DatasetConfig(name="two_moons", random_state=3),
                hidden_sizes=(8,),
                candidates=(LayoutCandidate("random_start_layout", "relu"),),
                seeds=(3,),
                training_config=TrainingConfig(
                    epochs=1,
                    learning_rate=0.03,
                    batch_size=32,
                    random_state=3,
                ),
                weight_scale=1.0,
                include_test_metrics=False,
            )
        )

        self.assertNotIn("test_loss", result.runs[0].metrics)
        self.assertNotIn("test_accuracy", result.runs[0].metrics)

if __name__ == "__main__":
    unittest.main()
