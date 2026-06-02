from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from services.evaluation_profile_service import load_evaluation_benchmark_profile
from services.experiment_suite_service import ExperimentSuiteRequest, run_experiment_suite


class EvaluationProfileServiceTests(unittest.TestCase):
    def test_promoted_profile_contains_benchmark_specific_training_and_sa_values(self) -> None:
        profile = load_evaluation_benchmark_profile("tuned_20260530", "crossing_spirals")

        self.assertEqual(profile["training"]["epochs"], 1500)
        self.assertEqual(profile["training"]["batch_size"], 8)
        self.assertEqual(profile["online_delta"]["max_steps"], 480)
        self.assertEqual(profile["online_delta"]["online_learning_rate"], 0.025)

    def test_suite_uses_promoted_profile_values_and_keeps_explicit_epoch_override(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_experiment_suite(
                ExperimentSuiteRequest(
                    exp="random-baseline",
                    benchmark="two_moons",
                    learning_rate_preset=1,
                    evaluation_profile="tuned_20260530",
                    epochs=1,
                    run_count=1,
                    output_root=Path(temp_dir),
                    no_plots=True,
                )
            )

            config = json.loads(
                (result.output_dir / "learning_rate_0.200" / "config.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(result.learning_rates, (0.2,))
            self.assertEqual(config["epochs"], 1)
            self.assertEqual(config["batch_size"], 64)
            self.assertEqual(config["weight_scale"], 2.0)
            self.assertEqual(config["evaluation_profile"]["name"], "tuned_20260530")
            self.assertTrue(config["include_test_metrics"])
            summary = (result.output_dir / "aggregate" / "summary.csv").read_text(
                encoding="utf-8"
            )
            run = next(
                (result.output_dir / "learning_rate_0.200" / "runs").glob("*.json")
            ).read_text(encoding="utf-8")
            self.assertIn("mean_test_loss", summary)
            self.assertIn("test_loss", run)

    def test_unknown_profile_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unbekanntes Evaluationsprofil"):
            load_evaluation_benchmark_profile("missing", "two_moons")


if __name__ == "__main__":
    unittest.main()
