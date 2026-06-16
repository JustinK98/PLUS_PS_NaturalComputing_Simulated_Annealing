from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", ".mplconfig")

from PySide6 import QtWidgets

from activations import parse_layout_spec, random_layout_spec
from configs import DatasetConfig, GuiExperimentConfig, TrainingConfig
from services.layout_evaluation_service import AggregatedLayoutEvaluation, LayoutEvaluationResult
from services.online_annealing_training_service import evaluate_online_start
from services.training_service import TrainingRunRequest, run_single_training_experiment
from ui_qt.shell.main_window import MainWindow
from ui_qt.state import WorkspacePreferences
from ui_qt.widgets.plot_widgets import TrainingPlotWidget
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

    def test_activation_workflow_defaults_to_methodical_profile_and_exposes_alternatives(self) -> None:
        workspace = ActivationWorkflowWorkspace(
            GuiExperimentConfig(
                benchmark="concentric_circles",
                mode="expert",
                language="en",
            ),
            WorkspacePreferences(language="en", detail_mode="expert"),
        )
        self.assertEqual(
            [workspace.gui_profile_combo.itemText(index) for index in range(workspace.gui_profile_combo.count())],
            ["methodical_selected_20260613", "demo", "mayer_corrected_20260604"],
        )
        self.assertEqual(workspace.gui_profile_combo.currentText(), "methodical_selected_20260613")
        self.assertEqual(workspace.max_steps_spin.value(), 25000)
        self.assertEqual(workspace.cooling_schedule_value.text(), "logarithmic")
        self.assertIn("kein robuster allgemeiner SA-Vorteil", workspace.gui_profile_note.text())
        workspace.close()

    def test_methodical_gui_profile_builds_session_with_selected_cooling_schedule(self) -> None:
        workspace = ActivationWorkflowWorkspace(
            GuiExperimentConfig(
                benchmark="crossing_spirals",
                mode="expert",
                language="en",
            ),
            WorkspacePreferences(language="en", detail_mode="expert"),
        )

        session = workspace._ensure_annealing_session()

        self.assertEqual(session.request.annealing_config.cooling_schedule, "linear")
        self.assertAlmostEqual(session.request.annealing_config.cooling_parameter, 3.602576692057094e-05)
        self.assertEqual(session.request.annealing_config.iterations_per_temperature, 25)
        self.assertEqual(session.request.annealing_config.max_steps, 2500)
        self.assertAlmostEqual(
            session.request.annealing_config.start_temperature,
            0.004458188656420654,
            places=11,
        )
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

    def test_baseline_training_button_submits_a_valid_training_request(self) -> None:
        workspace = ActivationWorkflowWorkspace(
            GuiExperimentConfig(
                benchmark="two_moons",
                epochs=1,
                mode="expert",
                language="en",
            ),
            WorkspacePreferences(language="en", detail_mode="expert"),
        )
        with patch.object(workspace.task_controller, "submit") as submit:
            workspace.train_button.click()

        submit.assert_called_once()
        callable_arg, request = submit.call_args.args[:2]
        self.assertIs(callable_arg, run_single_training_experiment)
        self.assertIsInstance(request, TrainingRunRequest)
        self.assertEqual(request.layout_spec, workspace.layout_editor.layout_spec())
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

    def test_training_plot_can_overlay_final_layout_histories(self) -> None:
        widget = TrainingPlotWidget()
        history = {
            "train_loss": [0.7, 0.5],
            "val_loss": [0.72, 0.55],
            "train_acc": [0.5, 0.8],
            "val_acc": [0.48, 0.78],
        }
        widget.set_history(
            None,
            [
                ("Random Start Layout", history),
                ("Best SA-Layout", history),
                ("SA-Endlayout", history),
            ],
            [("Best Online Delta Value", {"val_loss": 0.6, "val_accuracy": 0.7})],
        )

        self.assertEqual(len(widget.figure.axes[0].lines), 4)
        self.assertEqual(len(widget.figure.axes[1].lines), 4)
        self.assertIn("Best SA-Layout Val", widget.figure.axes[0].get_legend_handles_labels()[1])
        self.assertIn("Best Online Delta Value Val", widget.figure.axes[0].get_legend_handles_labels()[1])
        widget.close()

    def test_final_comparison_heatmap_maps_best_to_green_and_worst_to_red(self) -> None:
        best = ActivationWorkflowWorkspace._comparison_heatmap_color(0.0)
        worst = ActivationWorkflowWorkspace._comparison_heatmap_color(1.0)

        self.assertGreater(best.green(), best.red())
        self.assertGreater(worst.red(), worst.green())

    def test_final_comparison_hides_legacy_all_baselines(self) -> None:
        workspace = ActivationWorkflowWorkspace(
            GuiExperimentConfig(
                benchmark="two_moons",
                mode="expert",
                language="en",
            ),
            WorkspacePreferences(language="en", detail_mode="expert"),
        )

        def item(label: str, val_loss: float) -> AggregatedLayoutEvaluation:
            return AggregatedLayoutEvaluation(
                label=label,
                layout_spec=label,
                num_runs=1,
                mean_metrics={
                    "val_loss": val_loss,
                    "val_accuracy": 0.8,
                    "test_loss": val_loss,
                    "test_accuracy": 0.8,
                },
                std_metrics={},
                min_metrics={},
                max_metrics={},
                ranking_score=val_loss,
                evaluation_type="retrained",
            )

        workspace.layout_evaluation = LayoutEvaluationResult(
            runs=(),
            aggregated=(),
            ranking=(),
            combined_ranking=(
                item("diagnostic_best_layout_from_sa", 0.2),
                item("start_layout", 0.3),
                item("all_relu", 0.4),
                item("all_gelu", 0.5),
            ),
        )
        workspace._refresh_final_table()

        self.assertEqual(workspace.final_table.rowCount(), 2)
        self.assertEqual(workspace.final_table.item(0, 2).text(), "diagnostic_best_layout_from_sa")
        workspace.close()

    def test_multi_seed_mode_creates_unique_runs_and_central_aggregate_tab(self) -> None:
        workspace = ActivationWorkflowWorkspace(
            GuiExperimentConfig(
                benchmark="two_moons",
                epochs=1,
                mode="expert",
                language="en",
                gui_profile="demo",
            ),
            WorkspacePreferences(language="en", detail_mode="expert"),
        )
        workspace.multi_count_combo.setCurrentText("3")
        with patch.object(workspace, "_autosave_multi_session"):
            workspace.run_mode_combo.setCurrentIndex(workspace.run_mode_combo.findData("multi"))

        self.assertEqual(len(workspace.run_records), 3)
        self.assertEqual(len({record.start_layout_spec for record in workspace.run_records}), 3)
        self.assertEqual(workspace.run_selector.count(), 4)
        self.assertIn("Aggregiert", workspace.run_selector.tabText(3))

        workspace.run_selector.setCurrentIndex(1)
        self.assertEqual(workspace.active_run_index, 1)
        self.assertEqual(workspace.layout_editor.layout_spec(), workspace.run_records[1].start_layout_spec)
        workspace.run_records[0].status = "complete"
        workspace.layout_editor.set_layout_spec("relu")
        self.assertEqual(workspace.run_records[0].status, "complete")
        self.assertEqual(workspace.run_records[1].status, "pending")
        self.assertTrue(workspace.run_records[1].manually_edited)

        workspace.run_selector.setCurrentIndex(3)
        self.assertTrue(workspace.aggregate_selected)
        self.assertFalse(workspace.layout_editor.isEnabled())
        self.assertEqual(workspace.training_stack.currentIndex(), 1)
        workspace.close()

    def test_blocked_layout_mode_shares_split_weights_and_random_streams(self) -> None:
        workspace = ActivationWorkflowWorkspace(
            GuiExperimentConfig(
                benchmark="two_moons",
                epochs=1,
                mode="expert",
                language="en",
                gui_profile="demo",
            ),
            WorkspacePreferences(language="en", detail_mode="expert"),
        )
        self.assertEqual(workspace.comparison_mode_combo.currentData(), "blocked_layout")
        workspace.multi_count_combo.setCurrentText("3")
        with patch.object(workspace, "_autosave_multi_session"):
            workspace.run_mode_combo.setCurrentIndex(workspace.run_mode_combo.findData("multi"))

        schedules = [record.schedule for record in workspace.run_records]
        self.assertEqual(len({record.start_layout_spec for record in workspace.run_records}), 3)
        self.assertEqual(len({schedule.online_weight_seed for schedule in schedules}), 1)
        self.assertEqual(len({schedule.data_split_seed for schedule in schedules}), 1)
        self.assertIn("Random-Mixed-Startlayout variiert", workspace.comparison_mode_note.text())
        workspace.close()

    def test_robustness_mode_regenerates_independent_run_streams(self) -> None:
        workspace = ActivationWorkflowWorkspace(
            GuiExperimentConfig(
                benchmark="two_moons",
                epochs=1,
                mode="expert",
                language="en",
                gui_profile="demo",
            ),
            WorkspacePreferences(language="en", detail_mode="expert"),
        )
        workspace.multi_count_combo.setCurrentText("3")
        with patch.object(workspace, "_autosave_multi_session"):
            workspace.run_mode_combo.setCurrentIndex(workspace.run_mode_combo.findData("multi"))
            workspace.comparison_mode_combo.setCurrentIndex(
                workspace.comparison_mode_combo.findData("robustness")
            )

        schedules = [record.schedule for record in workspace.run_records]
        self.assertEqual(len({schedule.online_weight_seed for schedule in schedules}), 3)
        self.assertEqual(len({schedule.data_split_seed for schedule in schedules}), 3)
        workspace.close()

    def test_multi_final_jobs_use_paired_retraining_seed_streams(self) -> None:
        workspace = ActivationWorkflowWorkspace(
            GuiExperimentConfig(
                benchmark="two_moons",
                epochs=1,
                mode="expert",
                language="en",
                gui_profile="demo",
            ),
            WorkspacePreferences(language="en", detail_mode="expert"),
        )
        workspace.multi_count_combo.setCurrentText("2")
        with patch.object(workspace, "_autosave_multi_session"):
            workspace.run_mode_combo.setCurrentIndex(workspace.run_mode_combo.findData("multi"))

        record = workspace.run_records[0]
        job = workspace._final_job_for_record(record)

        self.assertEqual(job.request.data_split_seeds, (record.schedule.data_split_seed,))
        self.assertEqual(job.request.weight_seeds, (record.schedule.retraining_weight_seed,))
        self.assertEqual(job.request.batch_seeds, (record.schedule.retraining_batch_seed,))
        workspace.close()

if __name__ == "__main__":
    unittest.main()
