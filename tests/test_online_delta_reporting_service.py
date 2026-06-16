from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from services.online_delta_reporting_service import build_online_delta_report


class OnlineDeltaReportingServiceTests(unittest.TestCase):
    def test_report_normalizes_suite_histories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            suite_run = root / "suite" / "learning_rate_0.010" / "runs" / "run.json"
            suite_run.parent.mkdir(parents=True)

            suite_run.write_text(
                json.dumps(
                    {
                        "kind": "experiment_suite_online_delta_run",
                        "schema_version": 2,
                        "experiment_id": "E3_online_delta_from_random",
                        "benchmark": "two_moons",
                        "run_index": 0,
                        "run_id": "layout_0000_replicate_00",
                        "layout_index": 0,
                        "replicate_index": 0,
                        "seeds": {
                            "layout_seed": 11,
                            "data_split_seed": 12,
                            "online_weight_seed": 13,
                            "online_batch_seed": 14,
                            "sa_proposal_seed": 15,
                            "sa_acceptance_seed": 16,
                            "retraining_weight_seed": 17,
                            "retraining_batch_seed": 18,
                        },
                        "training_seed": 0,
                        "layout_seed": 0,
                        "learning_rate": 0.01,
                        "start_layout": "relu,gelu",
                        "diagnostic_best_layout": "relu,swish",
                        "end_layout": "tanh,swish",
                        "trained_batch_updates": 1,
                        "effective_online_epochs": 0.05,
                        "final_comparisons": [
                            {
                                "label": "same_random_start_retrained",
                                "val_loss": 0.4,
                                "val_accuracy": 0.8,
                            },
                            {
                                "label": "end_layout_from_sa_retrained",
                                "val_loss": 0.35,
                                "val_accuracy": 0.82,
                            },
                            {
                                "label": "best_online_delta_value",
                                "val_loss": 0.38,
                                "val_accuracy": 0.81,
                            },
                        ],
                        "online_history": [
                            {
                                "step_index": 0,
                                "temperature": 1.0,
                                "validation_loss_after_update": 0.72,
                                "accepted": True,
                                "delta": 0.0,
                                "neighbor_label": "start",
                                "reason_code": "start",
                                "post_training_batch_loss": 0.72,
                                "trained_batch_updates_after_step": 0,
                                "effective_online_epochs_after_step": 0.0,
                                "diagnostics_refreshed": True,
                            },
                            {
                                "step_index": 1,
                                "temperature": 1.0,
                                "validation_loss_after_update": 0.68,
                                "accepted": True,
                                "delta": -0.02,
                                "neighbor_label": "set:L1:1:swish",
                                "reason_code": "improved",
                                "post_training_batch_loss": 0.66,
                                "trained_batch_updates_after_step": 1,
                                "effective_online_epochs_after_step": 0.05,
                                "diagnostics_refreshed": True,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            result = build_online_delta_report(root)

            self.assertEqual(result.run_count, 1)
            self.assertTrue((root / "online_delta_report" / "online_delta_steps.csv").exists())
            self.assertTrue((root / "online_delta_report" / "online_delta_runs.csv").exists())
            self.assertTrue((root / "online_delta_report" / "online_delta_fitness_mean.png").exists())
            self.assertTrue((root / "online_delta_report" / "online_delta_validation_progress_mean.png").exists())
            self.assertTrue((root / "online_delta_report" / "activation_counts_end.csv").exists())
            self.assertTrue((root / "online_delta_report" / "layout_similarity_end.csv").exists())
            self.assertTrue((root / "online_delta_report" / "paired_runs.csv").exists())
            self.assertTrue((root / "online_delta_report" / "layout_level_summary.csv").exists())
            self.assertTrue((root / "online_delta_report" / "benchmark_summary.csv").exists())
            self.assertTrue((root / "online_delta_report" / "seed_manifest.csv").exists())
            self.assertTrue(
                (root / "online_delta_report" / "activation_layout_features.csv").exists()
            )
            self.assertTrue(
                (root / "online_delta_report" / "activation_effect_summary.csv").exists()
            )
            self.assertTrue(
                (root / "online_delta_report" / "activation_effect_association.png").exists()
            )
            run_rows = (root / "online_delta_report" / "online_delta_runs.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("online_validation_loss_delta", run_rows)
            counts = (root / "online_delta_report" / "activation_counts_end.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("swish", counts)
            paired = (root / "online_delta_report" / "paired_runs.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("paired_val_loss_improvement", paired)
            self.assertIn("layout_0000_replicate_00", paired)
            benchmark_summary = (
                root / "online_delta_report" / "benchmark_summary.csv"
            ).read_text(encoding="utf-8")
            self.assertIn("ci_unit", benchmark_summary)
            self.assertIn("layout_mean", benchmark_summary)
            activation_effects = (
                root / "online_delta_report" / "activation_effect_summary.csv"
            ).read_text(encoding="utf-8")
            self.assertIn("correlation_share_change_with_improvement", activation_effects)
            self.assertIn("swish", activation_effects)
            self.assertFalse(result.warnings)


if __name__ == "__main__":
    unittest.main()
