from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from annealing_schedules import temperature_for_step
from services.hyperparameter_tuning_service import (
    HyperparameterTuningRequest,
    _cooling_candidates,
    _load_profile,
    _stratified_sample_configs,
    run_hyperparameter_tuning,
    temperatures_for_target_acceptance,
)


class HyperparameterTuningServiceTests(unittest.TestCase):
    def test_temperature_derivation_matches_target_acceptance(self) -> None:
        temperatures = temperatures_for_target_acceptance(0.02, (0.2, 0.4, 0.8))

        self.assertEqual(len(temperatures), 3)
        for temperature, target in zip(temperatures, (0.2, 0.4, 0.8), strict=True):
            self.assertAlmostEqual(float(__import__("math").exp(-0.02 / temperature)), target)

    def test_methodical_profile_normalizes_all_cooling_schedules(self) -> None:
        profile = _load_profile("methodical", smoke=True)
        candidates = _cooling_candidates(
            profile,
            start_temperature=0.03,
            iterations_per_temperature=5,
            max_steps=101,
        )

        self.assertEqual(
            {item["cooling_schedule"] for item in candidates},
            {"geometric", "linear", "logarithmic"},
        )
        for candidate in candidates:
            final_temperature = temperature_for_step(
                str(candidate["cooling_schedule"]),
                0.03,
                float(candidate["cooling_parameter"]),
                20,
            )
            self.assertAlmostEqual(final_temperature / 0.03, 0.01)

    def test_methodical_confirmation_profile_expands_layout_blocks(self) -> None:
        profile = _load_profile("methodical-confirmation-30", smoke=False)

        self.assertEqual(profile["confirmation_layout_count"], 30)
        self.assertEqual(profile["confirmation_replicate_count"], 3)
        self.assertEqual(profile["confirmation_runs"], 90)
        self.assertEqual(
            profile["cooling_schedules"],
            ["geometric", "linear", "logarithmic"],
        )

    def test_stratified_cooling_sample_keeps_every_schedule(self) -> None:
        configs = [
            {"cooling_schedule": schedule, "candidate": candidate}
            for schedule in ("geometric", "linear", "logarithmic")
            for candidate in range(10)
        ]

        selected = _stratified_sample_configs(
            configs,
            6,
            group_key="cooling_schedule",
            rng=np.random.default_rng(42),
        )

        self.assertEqual(
            {item["cooling_schedule"] for item in selected},
            {"geometric", "linear", "logarithmic"},
        )

    def test_full_smoke_writes_validation_only_stages_and_confirmation_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_hyperparameter_tuning(
                HyperparameterTuningRequest(
                    benchmark="two_moons",
                    phase="full",
                    workers=2,
                    output_root=Path(temp_dir),
                    smoke=True,
                )
            )

            root = result.output_dir
            self.assertTrue((root / "aggregate" / "tuning_summary.csv").exists())
            self.assertTrue((root / "selected" / "two_moons.json").exists())
            for path in (
                root / "training",
                root / "delta_probe",
                root / "online_budget_probe",
                root / "sa",
            ):
                self.assertNotIn("test_loss", "\n".join(item.read_text(encoding="utf-8") for item in path.rglob("*.csv")))
            confirmation = root / "confirmation" / "two_moons"
            self.assertIn("test_loss", (confirmation / "summary.csv").read_text(encoding="utf-8"))
            self.assertTrue((confirmation / "aggregate" / "online_delta_fitness_mean.png").exists())
            self.assertTrue((confirmation / "aggregate" / "online_delta_validation_progress_mean.png").exists())
            self.assertFalse((confirmation / "online_delta" / "layout_frames").exists())
            self.assertTrue(list((confirmation / "online_delta" / "plots_single").glob("*_layout_summary.png")))
            run_payload = json.loads(
                next((confirmation / "online_delta" / "runs").glob("*.json")).read_text(encoding="utf-8")
            )
            comparison_labels = {item["label"] for item in run_payload["final_comparisons"]}
            self.assertEqual(
                comparison_labels,
                {
                    "same_random_start_retrained",
                    "end_layout_from_sa_retrained",
                    "best_online_delta_value",
                },
            )
            self.assertEqual(run_payload["layout_index"], 0)
            self.assertEqual(run_payload["replicate_index"], 0)
            self.assertNotEqual(
                run_payload["seeds"]["sa_proposal_seed"],
                run_payload["seeds"]["sa_acceptance_seed"],
            )
            confirmation_trials = (
                confirmation / "online_delta" / "trials.csv"
            ).read_text(encoding="utf-8")
            self.assertIn("layout_index", confirmation_trials)
            self.assertIn("sa_proposal_seed", confirmation_trials)
            selected = json.loads(
                (root / "selected" / "two_moons.json").read_text(encoding="utf-8")
            )
            self.assertIn("online_budget", selected)

            trials_path = root / "training" / "two_moons" / "screen" / "trials.csv"
            before = trials_path.read_text(encoding="utf-8")
            run_hyperparameter_tuning(
                HyperparameterTuningRequest(
                    benchmark="two_moons",
                    phase="training-screen",
                    workers=2,
                    resume=root,
                    smoke=True,
                )
            )
            self.assertEqual(trials_path.read_text(encoding="utf-8"), before)

    def test_crossing_spirals_stop_rule_reduces_sa_to_diagnostic_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_hyperparameter_tuning(
                HyperparameterTuningRequest(
                    benchmark="crossing_spirals",
                    phase="full",
                    workers=2,
                    output_root=Path(temp_dir),
                    smoke=True,
                )
            )

            root = result.output_dir
            diagnostic_path = root / "sa" / "crossing_spirals" / "screen" / "diagnostic_only.json"
            diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
            selected = json.loads((root / "selected" / "crossing_spirals.json").read_text(encoding="utf-8"))

            self.assertEqual(diagnostic["screened_sa_configurations"], 1)
            self.assertTrue(selected["confirmation"]["diagnostic_only"])


if __name__ == "__main__":
    unittest.main()
