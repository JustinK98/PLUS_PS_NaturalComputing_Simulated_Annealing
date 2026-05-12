from __future__ import annotations

import unittest

from configs import DatasetConfig, TrainingConfig
from services.layout_evaluation_service import (
    InheritedModelCandidate,
    LayoutEvaluationRequest,
    best_layout_run,
    build_layout_grid_candidates,
    build_standard_layout_candidates,
    evaluate_inherited_models,
    run_layout_evaluation,
    with_inherited_model_evaluations,
)


class LayoutEvaluationServiceTests(unittest.TestCase):
    def test_standard_candidates_include_sa_and_baselines_without_duplicates(self) -> None:
        candidates = build_standard_layout_candidates(
            (8,),
            start_layout_spec="relu",
            best_layout_spec="tanh",
            end_layout_spec="gelu",
            random_state=4,
        )

        labels = [candidate.label for candidate in candidates]

        self.assertIn("start_layout", labels)
        self.assertIn("best_layout_from_sa", labels)
        self.assertIn("end_layout_from_sa", labels)
        self.assertIn("random_layout", labels)
        self.assertIn("all_swish", labels)
        self.assertEqual(len(labels), len(set(labels)))

    def test_layout_evaluation_ranks_layouts_under_identical_conditions(self) -> None:
        candidates = build_standard_layout_candidates(
            (8,),
            start_layout_spec="relu",
            best_layout_spec="tanh",
            random_state=1,
        )[:3]
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

        self.assertEqual(len(result.runs), 3)
        self.assertEqual(len(result.aggregated), 3)
        self.assertEqual(result.ranking[0].ranking_score, min(item.ranking_score for item in result.aggregated))
        self.assertIsNotNone(best_layout_run(result, "validation_loss").model_state)

    def test_inherited_model_evaluation_is_combined_with_retrained_ranking(self) -> None:
        candidates = build_standard_layout_candidates(
            (8,),
            start_layout_spec="relu",
            best_layout_spec="tanh",
            random_state=1,
        )[:2]
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
                    "best_inherited_model_from_sa",
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

    def test_layout_grid_candidates_are_reproducible_and_bounded(self) -> None:
        candidates = build_layout_grid_candidates((8,), max_candidates=5)

        self.assertEqual(len(candidates), 5)
        self.assertEqual(candidates[0].label, "all_relu")
        self.assertEqual(candidates[0].layout_spec, "relu")


if __name__ == "__main__":
    unittest.main()
