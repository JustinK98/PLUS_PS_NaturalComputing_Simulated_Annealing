from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from search_spaces import SearchSpaceDefinition
from services.experiment_service import default_experiment_definition, save_experiment_definition


ROOT = Path(__file__).resolve().parents[1]


class CliSmokeTests(unittest.TestCase):
    def test_run_subcommand_smoke(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "main.py",
                "run",
                "--benchmark",
                "concentric_circles",
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

    def test_experiment_template_and_analyze_subcommands_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            template_path = Path(temp_dir) / "template.json"
            completed = subprocess.run(
                [sys.executable, "main.py", "experiment", "template", "--output", str(template_path)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertTrue(template_path.exists())
            self.assertIn("template:", completed.stdout)

            analyze = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "experiment",
                    "analyze",
                    "--path",
                    "outputs/backend_regression/backend_regression_grid",
                    "--max-ranking",
                    "2",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("Top ranking entries", analyze.stdout)

    def test_experiment_run_subcommand_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "experiment.json"
            definition = replace(
                default_experiment_definition(),
                experiment_id="cli_smoke_experiment",
                benchmark="concentric_circles",
                hidden_sizes=(8,),
                layout_spec="relu",
                epochs=2,
                seeds=(5,),
                save_json=False,
                search_space=SearchSpaceDefinition(search_type="none"),
            )
            save_experiment_definition(definition, config_path)
            completed = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "experiment",
                    "run",
                    "--config",
                    str(config_path),
                    "--max-ranking",
                    "1",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("Top ranking entries", completed.stdout)

    def test_layout_grid_subcommand_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "grid.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "experiment",
                    "layout-grid",
                    "--benchmark",
                    "concentric_circles",
                    "--epochs",
                    "1",
                    "--max-candidates",
                    "3",
                    "--output",
                    str(output_path),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertTrue(output_path.exists())
            self.assertIn("layout_grid:", completed.stdout)
            self.assertIn("top layouts:", completed.stdout)


if __name__ == "__main__":
    unittest.main()
