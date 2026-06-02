from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from services.hyperparameter_tuning_service import (
    HyperparameterTuningRequest,
    run_hyperparameter_tuning,
    temperatures_for_target_acceptance,
)


class HyperparameterTuningServiceTests(unittest.TestCase):
    def test_temperature_derivation_matches_target_acceptance(self) -> None:
        temperatures = temperatures_for_target_acceptance(0.02, (0.2, 0.4, 0.8))

        self.assertEqual(len(temperatures), 3)
        for temperature, target in zip(temperatures, (0.2, 0.4, 0.8), strict=True):
            self.assertAlmostEqual(float(__import__("math").exp(-0.02 / temperature)), target)

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
                root / "sa",
            ):
                self.assertNotIn("test_loss", "\n".join(item.read_text(encoding="utf-8") for item in path.rglob("*.csv")))
            confirmation = root / "confirmation" / "two_moons"
            self.assertIn("test_loss", (confirmation / "summary.csv").read_text(encoding="utf-8"))
            self.assertTrue((confirmation / "aggregate" / "online_delta_progress_mean.png").exists())
            self.assertFalse((confirmation / "online_delta" / "layout_frames").exists())
            self.assertTrue(list((confirmation / "online_delta" / "plots_single").glob("*_layout_summary.png")))
            run_payload = json.loads(
                next((confirmation / "online_delta" / "runs").glob("*.json")).read_text(encoding="utf-8")
            )
            comparison_labels = {item["label"] for item in run_payload["final_comparisons"]}
            self.assertIn("best_inherited_model_from_sa", comparison_labels)
            self.assertIn("end_inherited_model_from_sa", comparison_labels)

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
