from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from services.online_delta_reporting_service import build_online_delta_report


class OnlineDeltaReportingServiceTests(unittest.TestCase):
    def test_report_normalizes_suite_and_builder_histories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            suite_run = root / "suite" / "learning_rate_0.010" / "runs" / "run.json"
            builder_run = root / "builder" / "runs" / "builder_run.json"
            suite_run.parent.mkdir(parents=True)
            builder_run.parent.mkdir(parents=True)

            suite_run.write_text(
                json.dumps(
                    {
                        "kind": "experiment_suite_online_delta_run",
                        "experiment_id": "E3_online_delta_from_random",
                        "benchmark": "two_moons",
                        "run_index": 0,
                        "training_seed": 0,
                        "layout_seed": 0,
                        "learning_rate": 0.01,
                        "start_layout": "relu,gelu",
                        "best_layout": "relu,swish",
                        "end_layout": "tanh,swish",
                        "online_history": [
                            {
                                "step_index": 0,
                                "temperature": 1.0,
                                "validation_loss_after_update": 0.72,
                                "accepted": True,
                                "delta": 0.0,
                                "neighbor_label": "start",
                                "reason_code": "start",
                            },
                            {
                                "step_index": 1,
                                "temperature": 1.0,
                                "validation_loss_after_update": 0.68,
                                "accepted": True,
                                "delta": -0.02,
                                "neighbor_label": "set:L1:1:swish",
                                "reason_code": "improved",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            builder_run.write_text(
                json.dumps(
                    {
                        "run_definition": {
                            "run_id": "builder_run",
                            "experiment_id": "builder_exp",
                            "seed": 3,
                            "benchmark": "two_moons",
                            "config_values": {"learning_rate": 0.01},
                        },
                        "metrics": {},
                        "history": {},
                        "layout_spec": "gelu,identity",
                        "created_at": "2026-05-29T00:00:00Z",
                        "extra": {
                            "sa_evaluation_mode": "online_delta",
                            "start_layout_spec": "sigmoid,identity",
                            "best_layout_spec": "gelu,identity",
                            "end_layout_spec": "gelu,tanh",
                            "annealing_history": [
                                {
                                    "step_index": 1,
                                    "temperature": 0.5,
                                    "validation_loss_after_update": 0.7,
                                    "accepted": False,
                                    "loss_delta": 0.03,
                                    "neighbor_label": "set:L1:0:gelu",
                                    "reason_code": "rejected",
                                }
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = build_online_delta_report(root)

            self.assertEqual(result.run_count, 2)
            self.assertTrue((root / "online_delta_report" / "online_delta_steps.csv").exists())
            self.assertTrue((root / "online_delta_report" / "online_delta_runs.csv").exists())
            self.assertTrue((root / "online_delta_report" / "online_delta_progress_mean.png").exists())
            self.assertTrue((root / "online_delta_report" / "activation_counts_best.csv").exists())
            self.assertTrue((root / "online_delta_report" / "layout_similarity_best.csv").exists())
            run_rows = (root / "online_delta_report" / "online_delta_runs.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("online_validation_loss_delta", run_rows)
            counts = (root / "online_delta_report" / "activation_counts_best.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("swish", counts)
            self.assertIn("identity", counts)
            self.assertTrue(any("history starts after step 0" in warning for warning in result.warnings))


if __name__ == "__main__":
    unittest.main()
