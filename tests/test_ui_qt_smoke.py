from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", ".mplconfig")

from PySide6 import QtWidgets

from annealing import AnnealingConfig
from annealing_objectives import ObjectiveConfig
from configs import DatasetConfig, GuiExperimentConfig, TrainingConfig
from services.annealing_service import AnnealingRunRequest
from services.annealing_session_service import create_session, run_to_completion
from services.experiment_service import load_saved_experiment
from services.training_service import TrainingRunRequest, run_single_training_experiment
from ui_qt.shell.main_window import MainWindow
from ui_qt.state import WorkspacePreferences
from ui_qt.workspaces.demo_workspace import DemoWorkspace
from ui_qt.workspaces.experiment_builder_workspace import ExperimentBuilderWorkspace
from ui_qt.workspaces.playground_workspace import PlaygroundWorkspace


def _app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class QtSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _app()

    def test_main_window_boots_with_three_workspaces(self) -> None:
        window = MainWindow(
            GuiExperimentConfig(
                benchmark="wine",
                hidden_sizes=(16, 8),
                app_mode="demo",
                layout_spec="relu|tanh",
                mode="expert",
                language="en",
            )
        )
        self.assertEqual(len(window.workspaces), 3)
        window.close()

    def test_demo_workspace_training_smoke(self) -> None:
        workspace = DemoWorkspace(
            GuiExperimentConfig(
                benchmark="test_activation",
                hidden_sizes=(4, 3),
                app_mode="demo",
                layout_spec="relu|tanh",
                mode="expert",
                language="en",
            ),
            WorkspacePreferences(language="en", detail_mode="expert"),
        )
        artifacts = run_single_training_experiment(
            TrainingRunRequest(
                dataset_config=DatasetConfig(name="test_activation", random_state=7),
                hidden_sizes=(4, 3),
                layout_spec="relu|tanh",
                training_config=TrainingConfig(epochs=2, learning_rate=0.03, batch_size=4, random_state=7),
                weight_scale=0.05,
                random_state=7,
            )
        )
        workspace._on_training_finished(artifacts)
        self.assertIsNotNone(workspace.training_result)
        self.assertIsNotNone(workspace.trained_model)
        workspace.close()

    def test_playground_workspace_sa_smoke(self) -> None:
        workspace = PlaygroundWorkspace(
            GuiExperimentConfig(
                benchmark="test_activation",
                hidden_sizes=(4, 3),
                app_mode="playground",
                layout_spec="relu|tanh",
                mode="expert",
                language="en",
            ),
            WorkspacePreferences(language="en", detail_mode="expert"),
        )
        session = create_session(
            AnnealingRunRequest(
                dataset_config=DatasetConfig(name="test_activation", random_state=9),
                hidden_sizes=(4, 3),
                layout_spec="relu|tanh",
                objective_config=ObjectiveConfig(
                    objective_name="validation_loss",
                    candidate_epochs=2,
                    learning_rate=0.03,
                    batch_size=4,
                    weight_scale=0.05,
                    random_state=9,
                    shuffle=True,
                ),
                annealing_config=AnnealingConfig(
                    start_temperature=1.0,
                    cooling_schedule="geometric",
                    cooling_parameter=0.9,
                    iterations_per_temperature=2,
                    max_steps=4,
                    min_temperature=0.01,
                    neighborhood_operations=("set_neuron",),
                ),
                random_state=9,
            )
        )
        snapshot = run_to_completion(session, "en")
        workspace.session = session
        workspace._on_session_snapshot_ready(snapshot)
        self.assertIsNotNone(workspace.session_snapshot)
        workspace.close()

    def test_builder_workspace_loads_saved_results(self) -> None:
        workspace = ExperimentBuilderWorkspace(
            GuiExperimentConfig(
                benchmark="wine",
                hidden_sizes=(16, 8),
                app_mode="experiment_builder",
                layout_spec="relu|relu",
                mode="expert",
                language="en",
            ),
            WorkspacePreferences(language="en", detail_mode="expert"),
        )
        payload = load_saved_experiment("outputs/backend_regression/backend_regression_grid")
        workspace._on_payload_ready(payload)
        self.assertGreater(workspace.run_model.rowCount(), 0)
        workspace.close()


if __name__ == "__main__":
    unittest.main()
