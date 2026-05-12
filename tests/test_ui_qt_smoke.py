from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", ".mplconfig")

from PySide6 import QtWidgets

from configs import DatasetConfig, GuiExperimentConfig, TrainingConfig
from services.experiment_service import load_saved_experiment
from services.training_service import TrainingRunRequest, run_single_training_experiment
from ui_qt.shell.main_window import MainWindow
from ui_qt.state import WorkspacePreferences
from ui_qt.workspaces.activation_workflow_workspace import ActivationWorkflowWorkspace
from ui_qt.workspaces.experiment_builder_workspace import ExperimentBuilderWorkspace


def _app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class QtSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _app()

    def test_main_window_boots_with_two_workspaces(self) -> None:
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
        self.assertEqual(len(window.workspaces), 2)
        self.assertIn("activation_workflow", window.workspaces)
        self.assertIn("experiment_builder", window.workspaces)
        self.assertNotIn("demo", window.workspaces)
        self.assertNotIn("playground", window.workspaces)
        self.assertNotIn("presentation", window.workspaces)
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

    def test_activation_workflow_training_smoke(self) -> None:
        workspace = ActivationWorkflowWorkspace(
            GuiExperimentConfig(
                benchmark="concentric_circles",
                hidden_sizes=(8,),
                app_mode="activation_workflow",
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
        self.assertIsNotNone(workspace.training_artifacts)
        self.assertIsNotNone(workspace.current_model)
        workspace.close()

    def test_activation_workflow_online_sa_smoke(self) -> None:
        workspace = ActivationWorkflowWorkspace(
            GuiExperimentConfig(
                benchmark="concentric_circles",
                hidden_sizes=(8,),
                app_mode="activation_workflow",
                layout_spec="relu",
                mode="expert",
                language="en",
            ),
            WorkspacePreferences(language="en", detail_mode="expert"),
        )
        workspace.max_steps_spin.setValue(2)
        session = workspace._ensure_annealing_session()
        from services.online_annealing_training_service import online_run_to_completion

        snapshot = online_run_to_completion(session, "en")
        workspace._on_annealing_snapshot(snapshot)
        self.assertIsNotNone(workspace.annealing_snapshot)
        self.assertEqual(workspace.sa_mode_combo.currentText(), "online_delta")
        self.assertEqual(workspace.final_table.columnCount(), 8)
        self.assertEqual(workspace.final_table.horizontalHeaderItem(1).text(), "Type")
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
        workspace.run_mode_combo.setCurrentText("simulated_annealing")
        workspace.sa_mode_combo.setCurrentText("online_delta")
        self.assertEqual(workspace.primary_metric_combo.currentText(), "validation_loss")
        self.assertFalse(workspace.primary_metric_combo.isEnabled())
        payload = load_saved_experiment("outputs/backend_regression/backend_regression_grid")
        workspace._on_payload_ready(payload)
        self.assertGreater(workspace.run_model.rowCount(), 0)
        workspace.close()


if __name__ == "__main__":
    unittest.main()
