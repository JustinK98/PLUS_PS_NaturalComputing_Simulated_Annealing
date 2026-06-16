from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CliSmokeTests(unittest.TestCase):
    def test_run_subcommand_smoke(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "main.py",
                "run",
                "--benchmark",
                "two_moons",
                "--hidden-sizes",
                "8",
                "--layout",
                "relu",
                "--epochs",
                "1",
                "--show-neighbors",
                "0",
                "--no-plot",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("Training", completed.stdout)

    def test_experiment_suite_subcommand_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            completed = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "experiment",
                    "suite",
                    "--exp",
                    "online-delta",
                    "--benchmark",
                    "two_moons",
                    "--epochs",
                    "1",
                    "--layouts",
                    "1",
                    "--replicates",
                    "1",
                    "--max-steps",
                    "2",
                    "--no-plots",
                    "--output-root",
                    temp_dir,
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("suite:", completed.stdout)
            self.assertIn("summary:", completed.stdout)
            self.assertIn("test_metrics:   validation-only", completed.stdout)

            suite_dir = next(Path(temp_dir).iterdir())
            learning_rate_dir = suite_dir / "learning_rate_0.010"
            summary_text = (learning_rate_dir / "summary.csv").read_text(encoding="utf-8")
            run_text = next((learning_rate_dir / "runs").glob("*.json")).read_text(
                encoding="utf-8"
            )
            self.assertNotIn("test_loss", summary_text)
            self.assertNotIn("test_accuracy", summary_text)
            self.assertNotIn("test_loss", run_text)
            self.assertNotIn("test_accuracy", run_text)
            self.assertTrue((learning_rate_dir / "seed_manifest.csv").exists())
            self.assertIn('"layout_index": 0', run_text)
            self.assertIn('"replicate_index": 0', run_text)
            self.assertTrue((suite_dir / "aggregate" / "paired_runs.csv").exists())
            self.assertTrue((suite_dir / "aggregate" / "benchmark_summary.csv").exists())
            self.assertFalse(
                (suite_dir / "aggregate" / "paired_random_vs_sa_end.png").exists()
            )
            report = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "experiment",
                    "report-online-delta",
                    "--path",
                    str(suite_dir),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("runs:       1", report.stdout)
            self.assertIn("online_delta_fitness_mean.png", report.stdout)
            self.assertIn("online_delta_validation_progress_mean.png", report.stdout)

    def test_experiment_suite_online_delta_visual_artifacts_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "experiment",
                    "suite",
                    "--exp",
                    "online-delta",
                    "--benchmark",
                    "two_moons",
                    "--epochs",
                    "1",
                    "--runs",
                    "1",
                    "--max-steps",
                    "2",
                    "--start-temperature",
                    "1000000",
                    "--export-layout-frames",
                    "--output-root",
                    temp_dir,
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            )

            suite_dir = next(Path(temp_dir).iterdir())
            learning_rate_dir = suite_dir / "learning_rate_0.010"
            self.assertTrue(list((learning_rate_dir / "plots_single").glob("*_layout_summary.png")))
            self.assertTrue(list((learning_rate_dir / "plots_single").glob("*_accepted_timeline.png")))
            frame_dirs = list((learning_rate_dir / "layout_frames").iterdir())
            self.assertEqual(len(frame_dirs), 1)
            self.assertTrue((frame_dirs[0] / "frame_000_start.png").exists())
            self.assertTrue((frame_dirs[0] / "accepted_changes.csv").exists())
            self.assertTrue(list(frame_dirs[0].glob("frame_*_step_*.png")))

    def test_experiment_suite_skips_layout_frames_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "experiment",
                    "suite",
                    "--exp",
                    "online-delta",
                    "--benchmark",
                    "two_moons",
                    "--epochs",
                    "1",
                    "--runs",
                    "1",
                    "--max-steps",
                    "2",
                    "--start-temperature",
                    "1000000",
                    "--output-root",
                    temp_dir,
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            )

            suite_dir = next(Path(temp_dir).iterdir())
            learning_rate_dir = suite_dir / "learning_rate_0.010"
            self.assertTrue(list((learning_rate_dir / "plots_single").glob("*_layout_summary.png")))
            self.assertTrue(list((learning_rate_dir / "plots_single").glob("*_accepted_timeline.png")))
            self.assertFalse((learning_rate_dir / "layout_frames").exists())
            self.assertFalse((suite_dir / "aggregate" / "test_loss_by_layout.png").exists())


if __name__ == "__main__":
    unittest.main()
