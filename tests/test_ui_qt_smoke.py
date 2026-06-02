from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", ".mplconfig")

from PySide6 import QtWidgets

from activations import parse_layout_spec, random_layout_spec
from configs import DatasetConfig, GuiExperimentConfig, TrainingConfig
from services.online_annealing_training_service import evaluate_online_start
from services.training_service import TrainingRunRequest, run_single_training_experiment
from ui_qt.shell.main_window import MainWindow
from ui_qt.state import WorkspacePreferences
from ui_qt.workspaces.activation_workflow_workspace import ActivationWorkflowWorkspace


def _app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class QtSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _app()

    def test_main_window_boots_with_online_delta_demo_workspace(self) -> None:
        window = MainWindow(
            GuiExperimentConfig(
                benchmark="two_moons",
                mode="expert",
                language="en",
            )
        )
        self.assertIsInstance(window.workspace, ActivationWorkflowWorkspace)
        self.assertEqual(window.findChildren(QtWidgets.QToolBar), [])
        window.close()

    def test_activation_workflow_workspace_boots(self) -> None:
        workspace = ActivationWorkflowWorkspace(
            GuiExperimentConfig(
                benchmark="concentric_circles",
                epochs=2,
                mode="expert",
                language="en",
            ),
            WorkspacePreferences(language="en", detail_mode="expert"),
        )
        self.assertIsNotNone(workspace.dataset)
        self.assertIsNotNone(workspace.current_model)
        self.assertEqual(workspace.benchmark_combo.currentText(), "concentric_circles")
        self.assertEqual(workspace.layout_editor.hidden_sizes(), (8, 8))
        self.assertEqual(
            workspace.layout_editor.layout_spec(),
            parse_layout_spec(random_layout_spec((8, 8), 42), (8, 8)).to_compact_spec(),
        )
        workspace.benchmark_combo.setCurrentText("crossing_spirals")
        self.assertEqual(workspace.layout_editor.hidden_sizes(), (16, 16))
        workspace.close()

    def test_random_layout_button_draws_a_new_layout_on_every_click(self) -> None:
        workspace = ActivationWorkflowWorkspace(
            GuiExperimentConfig(
                benchmark="two_moons",
                mode="expert",
                language="en",
            ),
            WorkspacePreferences(language="en", detail_mode="expert"),
        )
        initial_layout = workspace.layout_editor.layout_spec()
        first_clicked_layout = workspace._generate_random_layout()
        second_clicked_layout = workspace._generate_random_layout()

        self.assertNotEqual(first_clicked_layout, initial_layout)
        self.assertNotEqual(second_clicked_layout, first_clicked_layout)
        workspace.close()

    def test_activation_workflow_loads_tuned_gui_profile(self) -> None:
        workspace = ActivationWorkflowWorkspace(
            GuiExperimentConfig(
                benchmark="concentric_circles",
                mode="expert",
                language="en",
                gui_profile="tuned_20260530",
            ),
            WorkspacePreferences(language="en", detail_mode="expert"),
        )
        self.assertEqual(workspace.epochs_spin.value(), 400)
        self.assertEqual(workspace.batch_spin.value(), 16)
        self.assertEqual(workspace.max_steps_spin.value(), 480)
        self.assertEqual(workspace.online_batch_size_value.text(), "128")
        workspace.close()

    def test_activation_workflow_training_smoke(self) -> None:
        workspace = ActivationWorkflowWorkspace(
            GuiExperimentConfig(
                benchmark="concentric_circles",
                mode="expert",
                language="en",
            ),
            WorkspacePreferences(language="en", detail_mode="expert"),
        )
        artifacts = run_single_training_experiment(
            TrainingRunRequest(
                dataset_config=DatasetConfig(name="concentric_circles", random_state=7),
                hidden_sizes=(8, 8),
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
                mode="expert",
                language="en",
            ),
            WorkspacePreferences(language="en", detail_mode="expert"),
        )
        workspace.max_steps_spin.setValue(2)
        initial_layout = workspace.layout_editor.layout_spec()
        session = workspace._ensure_annealing_session()
        self.assertEqual(workspace.layout_editor.layout_spec(), initial_layout)
        self.assertIs(workspace._ensure_annealing_session(), session)
        self.assertEqual(workspace.layout_editor.layout_spec(), initial_layout)
        from services.online_annealing_training_service import online_run_to_completion

        snapshot = online_run_to_completion(session, "en")
        workspace._on_annealing_snapshot(snapshot)
        self.assertIsNotNone(workspace.annealing_snapshot)
        self.assertEqual(workspace.neighborhood_value.text(), "set_neuron")
        self.assertEqual(workspace.online_delta_timeline_panel.slider.maximum(), len(snapshot.history))
        workspace.online_delta_timeline_panel.slider.setValue(0)
        self.assertIn("Schritt 0", workspace.online_delta_timeline_panel.position_label.text())
        workspace.online_delta_timeline_panel.slider.setValue(len(snapshot.history))
        self.assertIn("Schritt 2", workspace.online_delta_timeline_panel.position_label.text())
        self.assertIsNotNone(workspace.online_delta_timeline_panel.network_view._model)
        self.assertFalse(workspace.network_comparison_panel.reference_box.isHidden())
        self.assertEqual(workspace.final_table.columnCount(), 8)
        self.assertEqual(workspace.final_table.horizontalHeaderItem(1).text(), "Typ")
        visible_tabs = [
            workspace.tabs.tabText(index)
            for index in range(workspace.tabs.count())
        ]
        self.assertNotIn("SA Delta", visible_tabs)
        self.assertNotIn("SA Entscheidung", visible_tabs)
        self.assertNotIn("Ablauf", visible_tabs)
        workspace.close()

    def test_starting_online_delta_preserves_random_baseline_training_plot(self) -> None:
        workspace = ActivationWorkflowWorkspace(
            GuiExperimentConfig(
                benchmark="two_moons",
                mode="expert",
                language="en",
            ),
            WorkspacePreferences(language="en", detail_mode="expert"),
        )
        artifacts = run_single_training_experiment(
            TrainingRunRequest(
                dataset_config=DatasetConfig(name="two_moons", random_state=42),
                hidden_sizes=(8,),
                layout_spec=workspace.layout_editor.layout_spec(),
                training_config=TrainingConfig(epochs=2, learning_rate=0.03, batch_size=16, random_state=42),
                weight_scale=0.05,
                random_state=42,
            )
        )
        workspace._on_training_finished(artifacts)
        training_artifacts = workspace.training_artifacts
        loss_line_count = len(workspace.training_plot.figure.axes[0].lines)

        snapshot = evaluate_online_start(workspace._ensure_annealing_session(), "en")
        workspace._on_annealing_snapshot(snapshot)

        self.assertIs(workspace.training_artifacts, training_artifacts)
        self.assertEqual(len(workspace.training_plot.figure.axes[0].lines), loss_line_count)
        workspace.close()

    def test_final_comparison_heatmap_maps_best_to_green_and_worst_to_red(self) -> None:
        best = ActivationWorkflowWorkspace._comparison_heatmap_color(0.0)
        worst = ActivationWorkflowWorkspace._comparison_heatmap_color(1.0)

        self.assertGreater(best.green(), best.red())
        self.assertGreater(worst.red(), worst.green())

if __name__ == "__main__":
    unittest.main()
