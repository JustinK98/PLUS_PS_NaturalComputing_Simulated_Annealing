from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
import unittest

from services.reporting_service import build_report_assets


class ReportingServiceTests(unittest.TestCase):
    def test_build_report_assets_writes_tables_and_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            layout_dir = root / "layout_grids"
            demo_sa_dir = root / "demo_sa"
            neighborhood_dir = root / "neighborhood_grids"
            output_dir = root / "report_assets"
            layout_dir.mkdir()
            neighborhood_dir.mkdir()
            _write_layout_grid(layout_dir / "concentric_circles_demo_grid.json")
            _write_sa_result(demo_sa_dir / "demo_sa_concentric_circles")
            _write_neighborhood_summary(neighborhood_dir / "summary.csv")

            result = build_report_assets(
                output_dir=output_dir,
                layout_grid_dir=layout_dir,
                demo_sa_dir=demo_sa_dir,
                neighborhood_summary_path=neighborhood_dir / "summary.csv",
            )

            expected_files = {
                "layout_grid_summary.csv",
                "sa_summary.csv",
                "neighborhood_summary.csv",
                "benchmark_overview.csv",
                "layout_grid_best_accuracy.png",
                "sa_vs_grid_accuracy.png",
                "neighborhood_comparison.png",
                "training_curves_best_layouts.png",
            }
            self.assertTrue(expected_files.issubset({path.name for path in result.files}))
            for path in result.files:
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0, path)

            layout_rows = _read_csv(output_dir / "layout_grid_summary.csv")
            self.assertEqual(layout_rows[0]["benchmark"], "concentric_circles")
            self.assertEqual(layout_rows[0]["best_label"], "all_relu")

            sa_rows = _read_csv(output_dir / "sa_summary.csv")
            self.assertEqual(sa_rows[0]["final_mean_test_accuracy"], "0.9")


def _write_layout_grid(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "kind": "layout_grid",
                "benchmark": "concentric_circles",
                "candidate_count": 1,
                "seeds": [1],
                "training": {
                    "epochs": 2,
                    "learning_rate": 0.03,
                    "batch_size": 32,
                    "weight_scale": 0.05,
                },
                "best_run": {
                    "label": "all_relu",
                    "layout_spec": "relu*8",
                    "seed": 1,
                    "metrics": {
                        "val_loss": 0.3,
                        "val_accuracy": 0.8,
                        "test_accuracy": 0.7,
                    },
                    "history": {
                        "train_acc": [0.5, 0.7],
                        "val_acc": [0.4, 0.8],
                    },
                },
                "result": {
                    "ranking": [
                        {
                            "label": "all_relu",
                            "layout_spec": "relu*8",
                            "ranking_score": 0.3,
                            "mean_metrics": {
                                "val_loss": 0.3,
                                "val_accuracy": 0.8,
                                "test_loss": 0.4,
                                "test_accuracy": 0.7,
                            },
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )


def _write_sa_result(experiment_dir: Path) -> None:
    runs_dir = experiment_dir / "runs"
    runs_dir.mkdir(parents=True)
    (experiment_dir / "summary.json").write_text(
        json.dumps(
            {
                "number_of_seeds": 1,
                "ranking": [
                    {
                        "ranking_score": 0.4,
                        "mean_val_loss": 0.4,
                        "mean_val_accuracy": 0.75,
                        "mean_test_accuracy": 0.74,
                        "best_seed": 1,
                        "worst_seed": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (runs_dir / "cfg_0001_seed_0001.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "final_val_loss": 0.2,
                    "final_val_accuracy": 0.85,
                    "final_test_loss": 0.25,
                    "final_test_accuracy": 0.9,
                    "step_count": 2.0,
                    "acceptance_rate": 0.5,
                }
            }
        ),
        encoding="utf-8",
    )


def _write_neighborhood_summary(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "benchmark",
                "neighborhood",
                "experiment_id",
                "path",
                "ranking_score",
                "mean_val_loss",
                "mean_val_accuracy",
                "mean_test_accuracy",
                "best_seed",
                "worst_seed",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "benchmark": "concentric_circles",
                "neighborhood": "set_neuron",
                "experiment_id": "ng",
                "path": "outputs/example",
                "ranking_score": "0.3",
                "mean_val_loss": "0.3",
                "mean_val_accuracy": "0.8",
                "mean_test_accuracy": "0.7",
                "best_seed": "1",
                "worst_seed": "1",
            }
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
