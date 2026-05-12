from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from experiment_builder import ExperimentDefinition
from search_spaces import SearchSpaceDefinition
from services.experiment_service import (
    default_experiment_definition,
    load_experiment_definition,
    run_experiment_definition,
    save_experiment_definition,
    save_experiment_template,
)


class ExperimentServiceTests(unittest.TestCase):
    def test_template_save_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "template.json"
            save_experiment_template(output_path)
            definition = load_experiment_definition(output_path)
            self.assertIsInstance(definition, ExperimentDefinition)
            self.assertEqual(definition.experiment_id, "example_concentric_circles_manual")

    def test_definition_json_roundtrip(self) -> None:
        definition = default_experiment_definition()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "experiment.json"
            save_experiment_definition(definition, output_path)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            reloaded = ExperimentDefinition.from_dict(payload)
            self.assertEqual(reloaded.to_dict(), definition.to_dict())
            self.assertEqual(reloaded.sa_evaluation_mode, "online_delta")
            self.assertEqual(reloaded.online_train_policy, "accepted")
            self.assertEqual(reloaded.primary_metric, "validation_loss")

    def test_old_definition_without_sa_mode_loads_as_short_retrain(self) -> None:
        payload = default_experiment_definition().to_dict()
        payload.pop("sa_evaluation_mode")
        payload.pop("online_train_policy")
        reloaded = ExperimentDefinition.from_dict(payload)
        self.assertEqual(reloaded.sa_evaluation_mode, "short_retrain")
        self.assertEqual(reloaded.online_train_policy, "accepted")

    def test_run_experiment_definition_manual_smoke(self) -> None:
        definition = replace(
            default_experiment_definition(),
            experiment_id="concentric_manual_smoke",
            benchmark="concentric_circles",
            hidden_sizes=(8,),
            layout_spec="relu",
            seeds=(11,),
            save_json=False,
            epochs=2,
            search_space=SearchSpaceDefinition(search_type="none"),
        )
        payload = run_experiment_definition(definition)
        self.assertEqual(payload["summary"].number_of_runs, 1)
        self.assertEqual(len(payload["run_results"]), 1)
        self.assertIn("val_accuracy", payload["run_results"][0].metrics)

    def test_run_experiment_definition_online_delta_sa_smoke(self) -> None:
        definition = replace(
            default_experiment_definition(),
            experiment_id="concentric_online_sa_smoke",
            benchmark="concentric_circles",
            hidden_sizes=(8,),
            layout_spec="relu",
            run_mode="simulated_annealing",
            seeds=(13,),
            save_json=False,
            epochs=2,
            max_steps=2,
            search_space=SearchSpaceDefinition(search_type="none"),
        )
        payload = run_experiment_definition(definition)
        result = payload["run_results"][0]
        self.assertEqual(result.extra["sa_evaluation_mode"], "online_delta")
        self.assertGreaterEqual(len(result.extra["annealing_history"]), 1)
        self.assertIn("final_layout_evaluation", result.extra)
        self.assertIn("final_retrained_layout_evaluation", result.extra)
        self.assertIn("best_inherited_model_evaluation", result.extra)
        self.assertIn("end_inherited_model_evaluation", result.extra)
        self.assertIn("combined_final_comparison", result.extra)
        self.assertTrue(
            any(
                entry["evaluation_type"] == "inherited"
                for entry in result.extra["combined_final_comparison"]
            )
        )

    def test_run_experiment_definition_short_retrain_sa_still_works(self) -> None:
        definition = replace(
            default_experiment_definition(),
            experiment_id="concentric_short_retrain_sa_smoke",
            benchmark="concentric_circles",
            hidden_sizes=(8,),
            layout_spec="relu",
            run_mode="simulated_annealing",
            sa_evaluation_mode="short_retrain",
            seeds=(14,),
            save_json=False,
            epochs=2,
            candidate_epochs=1,
            max_steps=1,
            search_space=SearchSpaceDefinition(search_type="none"),
        )
        payload = run_experiment_definition(definition)
        result = payload["run_results"][0]
        self.assertEqual(result.extra["effective_parameters"]["sa_evaluation_mode"], "short_retrain")
        self.assertGreaterEqual(len(result.extra["annealing_history"]), 1)


if __name__ == "__main__":
    unittest.main()
