from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from activations import random_layout_spec
from annealing import AnnealingConfig
from configs import (
    DEFAULT_ANNEALING_COOLING_SCHEDULE,
    DEFAULT_ANNEALING_ITERATIONS_PER_TEMPERATURE,
    DEFAULT_ANNEALING_NEIGHBORHOODS,
    DEFAULT_ANNEALING_MIN_TEMPERATURE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_WEIGHT_SCALE,
    DatasetConfig,
    TrainingConfig,
)
from services.layout_visualization_service import (
    ACCEPTED_CHANGES_FIELDS,
    write_online_delta_layout_artifacts,
)
from services.online_annealing_training_service import (
    OnlineAnnealingRequest,
    create_online_session,
    online_run_to_completion,
)


class LayoutVisualizationServiceTests(unittest.TestCase):
    def _snapshot(self):
        request = OnlineAnnealingRequest(
            dataset_config=DatasetConfig(name="two_moons", random_state=3),
            hidden_sizes=(8,),
            layout_spec=random_layout_spec((8,), 3),
            training_config=TrainingConfig(
                epochs=1,
                learning_rate=0.01,
                batch_size=DEFAULT_BATCH_SIZE,
                random_state=3,
            ),
            annealing_config=AnnealingConfig(
                start_temperature=1_000_000.0,
                cooling_schedule=DEFAULT_ANNEALING_COOLING_SCHEDULE,
                cooling_parameter=0.95,
                iterations_per_temperature=DEFAULT_ANNEALING_ITERATIONS_PER_TEMPERATURE,
                max_steps=2,
                min_temperature=DEFAULT_ANNEALING_MIN_TEMPERATURE,
                neighborhood_operations=DEFAULT_ANNEALING_NEIGHBORHOODS,
            ),
            weight_scale=DEFAULT_WEIGHT_SCALE,
            random_state=3,
        )
        return online_run_to_completion(create_online_session(request), "en")

    def test_online_delta_layout_artifacts_skip_frames_by_default(self) -> None:
        snapshot = self._snapshot()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifacts = write_online_delta_layout_artifacts(
                snapshot,
                plots_dir=root / "plots_single",
                frames_root=root / "layout_frames",
                run_id="smoke_run",
            )

            self.assertEqual(len(artifacts), 2)
            self.assertTrue((root / "plots_single" / "smoke_run_layout_summary.png").exists())
            self.assertTrue((root / "plots_single" / "smoke_run_accepted_timeline.png").exists())
            self.assertFalse((root / "layout_frames").exists())

    def test_online_delta_layout_frames_are_written_when_requested(self) -> None:
        snapshot = self._snapshot()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifacts = write_online_delta_layout_artifacts(
                snapshot,
                plots_dir=root / "plots_single",
                frames_root=root / "layout_frames",
                run_id="smoke_run",
                export_frames=True,
            )

            self.assertTrue(artifacts)
            self.assertTrue((root / "plots_single" / "smoke_run_layout_summary.png").exists())
            self.assertTrue((root / "plots_single" / "smoke_run_accepted_timeline.png").exists())
            frame_dir = root / "layout_frames" / "smoke_run"
            self.assertTrue((frame_dir / "frame_000_start.png").exists())
            self.assertTrue((frame_dir / "accepted_changes.csv").exists())
            self.assertTrue(any(frame_dir.glob("frame_*_step_*.png")))

            with (frame_dir / "accepted_changes.csv").open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(tuple(reader.fieldnames or ()), ACCEPTED_CHANGES_FIELDS)
                rows = list(reader)
            self.assertGreaterEqual(len(rows), 1)
            self.assertTrue(all(row["neighbor_label"].startswith("set:") for row in rows))


if __name__ == "__main__":
    unittest.main()
