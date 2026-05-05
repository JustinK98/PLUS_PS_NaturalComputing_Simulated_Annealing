from __future__ import annotations

import os
import time
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
from ui_qt.workspaces.activation_workflow_workspace import ActivationWorkflowWorkspace
from ui_qt.workspaces.demo_workspace import DemoWorkspace
from ui_qt.workspaces.experiment_builder_workspace import ExperimentBuilderWorkspace
from ui_qt.workspaces.playground_workspace import PlaygroundWorkspace
from ui_qt.workspaces.presentation_workspace import PresentationWorkspace


def _app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _wait_until(app: QtWidgets.QApplication, predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()


class QtSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _app()

    def test_main_window_boots_with_four_workspaces(self) -> None:
        window = MainWindow(
            GuiExperimentConfig(
                benchmark="iris",
                hidden_sizes=(8,),
                app_mode="activation_workflow",
                layout_spec="relu",
                mode="expert",
                language="en",
            )
        )
        self.assertEqual(len(window.workspaces), 5)
        self.assertIn("activation_workflow", window.workspaces)
        self.assertIn("presentation", window.workspaces)
        window.close()

    def test_activation_workflow_workspace_boots(self) -> None:
        workspace = ActivationWorkflowWorkspace(
            GuiExperimentConfig(
                benchmark="concentric_circles",
                hidden_sizes=(8,),
                app_mode="activation_workflow",
                layout_spec="relu",
                epochs=2,
                mode="expert",
                language="en",
            ),
            WorkspacePreferences(language="en", detail_mode="expert"),
        )
        self.assertIsNotNone(workspace.dataset)
        self.assertIsNotNone(workspace.current_model)
        self.assertEqual(workspace.benchmark_combo.currentText(), "concentric_circles")
        workspace.close()

    def test_presentation_workspace_tracks_can_advance(self) -> None:
        workspace = PresentationWorkspace(
            GuiExperimentConfig(
                benchmark="concentric_circles",
                hidden_sizes=(8,),
                app_mode="presentation",
                layout_spec="relu",
                mode="beginner",
                language="en",
            ),
            WorkspacePreferences(language="en", detail_mode="beginner"),
        )
        workspace._start_track("neural_network")
        self.assertEqual(workspace.track_id, "neural_network")
        for _ in range(6):
            workspace._next_slide()
        self.assertEqual(workspace.slide_index, 6)
        workspace.slide_index = 3
        before_activation = workspace.runtime_state.model.layout.layers[0][0]
        workspace._run_slide_action()
        self.assertNotEqual(workspace.runtime_state.model.layout.layers[0][0], before_activation)
        workspace._on_activation_lab_selection_changed(0, 2)
        workspace._on_activation_lab_changed(0, 2, "sigmoid")
        self.assertEqual(workspace.runtime_state.selected_hidden, (0, 2))
        self.assertEqual(workspace.runtime_state.model.layout.layers[0][2], "sigmoid")
        workspace._start_track("simulated_annealing")
        workspace._run_annealing_action("evaluate_start")
        _wait_until(
            self.app,
            lambda: workspace.annealing_snapshot is not None
            and workspace.annealing_snapshot.is_initialized,
        )
        self.assertIsNotNone(workspace.annealing_snapshot)
        workspace._run_annealing_action("step_10")
        _wait_until(
            self.app,
            lambda: workspace.annealing_snapshot is not None
            and len(workspace.annealing_snapshot.history) >= 1,
        )
        self.assertIsNotNone(workspace.annealing_snapshot)
        workspace._run_annealing_action("run_to_completion")
        _wait_until(
            self.app,
            lambda: workspace.annealing_snapshot is not None
            and workspace.annealing_snapshot.is_complete,
        )
        self.assertIsNotNone(workspace.annealing_snapshot)
        assert workspace.annealing_snapshot is not None
        self.assertTrue(workspace.annealing_snapshot.is_complete)
        workspace.slide_index = 6
        workspace._refresh_slide()
        workspace._reset_current_track()
        self.assertEqual(workspace.track_id, "simulated_annealing")
        self.assertEqual(workspace.slide_index, 0)
        self.assertIsNone(workspace.annealing_snapshot)
        workspace.close()

    def test_demo_workspace_training_smoke(self) -> None:
        workspace = DemoWorkspace(
            GuiExperimentConfig(
                benchmark="concentric_circles",
                hidden_sizes=(8,),
                app_mode="demo",
                layout_spec="relu",
                mode="expert",
                language="en",
            ),
            WorkspacePreferences(language="en", detail_mode="expert"),
        )
        artifacts = run_single_training_experiment(
            TrainingRunRequest(
                dataset_config=DatasetConfig(name="concentric_circles", random_state=7),
                hidden_sizes=(8,),
                layout_spec="relu",
                training_config=TrainingConfig(epochs=2, learning_rate=0.03, batch_size=16, random_state=7),
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
                benchmark="concentric_circles",
                hidden_sizes=(8,),
                app_mode="playground",
                layout_spec="relu",
                mode="expert",
                language="en",
            ),
            WorkspacePreferences(language="en", detail_mode="expert"),
        )
        session = create_session(
            AnnealingRunRequest(
                dataset_config=DatasetConfig(name="concentric_circles", random_state=9),
                hidden_sizes=(8,),
                layout_spec="relu",
                objective_config=ObjectiveConfig(
                    objective_name="validation_loss",
                    candidate_epochs=2,
                    learning_rate=0.03,
                    batch_size=16,
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
                benchmark="iris",
                hidden_sizes=(8,),
                app_mode="experiment_builder",
                layout_spec="relu",
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
