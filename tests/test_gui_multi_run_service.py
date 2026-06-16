from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from configs import DatasetConfig, TrainingConfig
from services.gui_multi_run_service import (
    aggregate_histories,
    build_gui_seed_schedule,
    create_gui_multi_run_session,
    generate_unique_gui_runs,
    load_gui_multi_run_session,
    save_gui_multi_run_session,
)
from services.training_service import TrainingRunRequest, run_single_training_experiment


class GuiMultiRunServiceTests(unittest.TestCase):
    def test_gui_seed_schedules_are_deterministic_and_independent(self) -> None:
        first = build_gui_seed_schedule("two_moons", 0, 42)
        repeated = build_gui_seed_schedule("two_moons", 0, 42)
        second = build_gui_seed_schedule("two_moons", 1, 42)

        self.assertEqual(first, repeated)
        self.assertNotEqual(first.layout_seed, second.layout_seed)
        self.assertNotEqual(first.data_split_seed, second.data_split_seed)
        self.assertNotEqual(first.online_weight_seed, first.online_batch_seed)
        self.assertNotEqual(first.sa_proposal_seed, first.sa_acceptance_seed)

    def test_generates_requested_number_of_unique_layouts(self) -> None:
        records = generate_unique_gui_runs("two_moons", (8,), 10, 42)

        self.assertEqual(len(records), 10)
        self.assertEqual(len({record.start_layout_spec for record in records}), 10)

    def test_blocked_layout_runs_share_nuisance_randomness_but_not_layouts(self) -> None:
        records = generate_unique_gui_runs(
            "two_moons",
            (8,),
            10,
            42,
            comparison_mode="blocked_layout",
        )

        schedules = [record.schedule for record in records]
        self.assertEqual(len({schedule.layout_seed for schedule in schedules}), 10)
        self.assertEqual(len({schedule.data_split_seed for schedule in schedules}), 1)
        self.assertEqual(len({schedule.online_weight_seed for schedule in schedules}), 1)
        self.assertEqual(len({schedule.online_batch_seed for schedule in schedules}), 1)
        self.assertEqual(len({schedule.sa_proposal_seed for schedule in schedules}), 1)
        self.assertEqual(len({schedule.sa_acceptance_seed for schedule in schedules}), 1)
        self.assertEqual(len({schedule.retraining_weight_seed for schedule in schedules}), 1)
        self.assertEqual(len({schedule.retraining_batch_seed for schedule in schedules}), 1)

    def test_robustness_runs_use_independent_nuisance_randomness(self) -> None:
        records = generate_unique_gui_runs(
            "two_moons",
            (8,),
            10,
            42,
            comparison_mode="robustness",
        )

        schedules = [record.schedule for record in records]
        self.assertEqual(len({schedule.data_split_seed for schedule in schedules}), 10)
        self.assertEqual(len({schedule.online_weight_seed for schedule in schedules}), 10)
        self.assertEqual(len({schedule.retraining_weight_seed for schedule in schedules}), 10)

    def test_history_aggregation_uses_median_iqr_and_nan_tail(self) -> None:
        series = aggregate_histories(
            [
                {"val_loss": [3.0, 2.0, 1.0]},
                {"val_loss": [5.0, 4.0]},
            ],
            "val_loss",
        )

        assert series is not None
        self.assertEqual(series.median, (4.0, 3.0, 1.0))
        self.assertEqual(series.count, (2, 2, 1))
        self.assertEqual(series.q25[-1], 1.0)
        self.assertEqual(series.q75[-1], 1.0)

    def test_session_roundtrip_reconstructs_training_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = create_gui_multi_run_session(
                benchmark="two_moons",
                profile_name="demo",
                master_seed=7,
                hidden_sizes=(8,),
                config_snapshot={"epochs": 1},
                run_count=2,
                comparison_mode="blocked_layout",
                output_root=Path(temp_dir),
            )
            record = session.runs[0]
            record.training_artifacts = run_single_training_experiment(
                TrainingRunRequest(
                    dataset_config=DatasetConfig(
                        name="two_moons",
                        random_state=record.schedule.data_split_seed,
                    ),
                    hidden_sizes=(8,),
                    layout_spec=record.start_layout_spec,
                    training_config=TrainingConfig(
                        epochs=1,
                        learning_rate=0.01,
                        batch_size=16,
                        random_state=record.schedule.retraining_batch_seed,
                    ),
                    weight_scale=0.5,
                    random_state=record.schedule.retraining_weight_seed,
                )
            )
            path = save_gui_multi_run_session(session)

            loaded = load_gui_multi_run_session(path)

            self.assertEqual(len(loaded.runs), 2)
            self.assertEqual(loaded.comparison_mode, "blocked_layout")
            self.assertEqual(loaded.runs[0].start_layout_spec, record.start_layout_spec)
            self.assertIsNotNone(loaded.runs[0].training_artifacts)
            self.assertEqual(
                loaded.runs[0].training_artifacts.training_result.history,
                record.training_artifacts.training_result.history,
            )

    def test_interrupted_run_loads_at_last_safe_run_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = create_gui_multi_run_session(
                benchmark="two_moons",
                profile_name="demo",
                master_seed=7,
                hidden_sizes=(8,),
                config_snapshot={},
                run_count=2,
                output_root=Path(temp_dir),
            )
            session.runs[0].status = "running"
            path = save_gui_multi_run_session(session)

            loaded = load_gui_multi_run_session(path)

            self.assertEqual(loaded.runs[0].status, "pending")
            self.assertIsNone(loaded.runs[0].annealing_snapshot)


if __name__ == "__main__":
    unittest.main()
