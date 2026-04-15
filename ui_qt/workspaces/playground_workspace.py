"""Qt-Workspace fuer einzelne Simulated-Annealing-Laeufe."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from annealing import AnnealingConfig
from annealing_objectives import ObjectiveConfig
from benchmarks import DatasetBundle, load_benchmark
from configs import DatasetConfig, GuiExperimentConfig, default_hidden_sizes
from model import ModularMLP
from services.annealing_session_service import (
    AnnealingSession,
    AnnealingSessionSnapshot,
    create_session,
    evaluate_start,
    reset,
    run_to_completion,
    step_once,
)
from services.analysis_sample_service import AnalysisSample, build_analysis_sample, split_arrays
from services.help_service import workspace_help_html
from services.neuron_analysis_service import build_neuron_analysis_payload
from services.network_projection_service import build_input_projection
from services.annealing_service import AnnealingRunRequest
from terminal_viz import render_layout, render_layout_diff
from ui_qt.state import WorkspacePreferences
from ui_qt.tasking import BackgroundTaskController
from ui_qt.texts import text
from ui_qt.utils import parse_hidden_sizes
from ui_qt.widgets.annealing_decision_panel import AnnealingDecisionPanel
from ui_qt.widgets.annealing_history_panel import AnnealingHistoryPanel
from ui_qt.widgets.annealing_live_panel import AnnealingLivePanel
from ui_qt.widgets.info_card import InfoCardWidget
from ui_qt.widgets.layout_editor import LayoutEditorWidget
from ui_qt.widgets.network_view import NetworkViewWidget
from ui_qt.widgets.neuron_detail_panel import NeuronDetailPanel
from ui_qt.widgets.recipe_panel import RecipePanelWidget
from ui_qt.widgets.sample_panel import SampleDisplayPayload, SamplePanelWidget
from .base import BaseWorkspace


class PlaygroundWorkspace(BaseWorkspace):
    workspace_id = "playground"

    def __init__(self, config: GuiExperimentConfig, preferences: WorkspacePreferences, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(preferences, parent)
        self.config = config
        self.dataset: DatasetBundle | None = None
        self.preview_model: ModularMLP | None = None
        self.session: AnnealingSession | None = None
        self.session_snapshot: AnnealingSessionSnapshot | None = None
        self.selected_hidden: tuple[int, int] | None = None
        self.split_name = "val"
        self.sample_index = 0
        self.task_controller = BackgroundTaskController(self)
        self.task_controller.busyChanged.connect(self.busyChanged)
        self.task_controller.messageEmitted.connect(self.statusMessage)
        self._build_ui()
        self._load_dataset()
        self._rebuild_preview_model()
        self._refresh_all_views()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(10)
        root.addWidget(self.splitter)

        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left = QtWidgets.QWidget()
        left_scroll.setWidget(left)
        self.splitter.addWidget(left_scroll)
        left_layout = QtWidgets.QVBoxLayout(left)

        self.problem_group = QtWidgets.QGroupBox()
        problem_layout = QtWidgets.QFormLayout(self.problem_group)
        self.benchmark_combo = QtWidgets.QComboBox()
        self.benchmark_combo.addItems(("breast_cancer", "wine", "digits", "test_activation"))
        self.benchmark_combo.setCurrentText(self.config.benchmark)
        self.benchmark_combo.currentTextChanged.connect(self._on_benchmark_changed)
        self.hidden_sizes_edit = QtWidgets.QLineEdit(", ".join(str(v) for v in self.config.hidden_sizes))
        self.hidden_sizes_edit.editingFinished.connect(self._on_hidden_sizes_changed)
        self.split_combo = QtWidgets.QComboBox()
        self.split_combo.addItems(("train", "val", "test"))
        self.split_combo.setCurrentText(self.split_name)
        self.split_combo.currentTextChanged.connect(self._on_split_changed)
        self.sample_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sample_slider.valueChanged.connect(self._on_sample_index_changed)
        self.sample_spin = QtWidgets.QSpinBox()
        self.sample_spin.valueChanged.connect(self._on_sample_index_changed)
        sample_row = QtWidgets.QHBoxLayout()
        sample_row.addWidget(self.sample_slider)
        sample_row.addWidget(self.sample_spin)
        sample_row_widget = QtWidgets.QWidget()
        sample_row_widget.setLayout(sample_row)
        problem_layout.addRow(self.make_help_label("Benchmark", "benchmark"), self.benchmark_combo)
        problem_layout.addRow(self.make_help_label("Hidden Sizes", "hidden_sizes"), self.hidden_sizes_edit)
        problem_layout.addRow(self.make_help_label("Split", "sample_selection"), self.split_combo)
        problem_layout.addRow(self.make_help_label("Sample", "sample_selection"), sample_row_widget)
        left_layout.addWidget(self.problem_group)

        self.start_group = QtWidgets.QGroupBox()
        start_layout = QtWidgets.QVBoxLayout(self.start_group)
        start_layout.addWidget(self.make_help_strip("layout"))
        self.layout_editor = LayoutEditorWidget(self.config.hidden_sizes, self.config.layout_spec)
        self.layout_editor.layoutSpecChanged.connect(self._on_layout_changed)
        start_layout.addWidget(self.layout_editor)
        left_layout.addWidget(self.start_group)

        self.objective_group = QtWidgets.QGroupBox()
        objective_layout = QtWidgets.QFormLayout(self.objective_group)
        self.objective_combo = QtWidgets.QComboBox()
        self.objective_combo.addItems(("validation_loss", "validation_accuracy"))
        self.candidate_epochs_spin = QtWidgets.QSpinBox()
        self.candidate_epochs_spin.setRange(1, 500)
        self.candidate_epochs_spin.setValue(20)
        objective_layout.addRow(self.make_help_label("Objective", "objective"), self.objective_combo)
        objective_layout.addRow(self.make_help_label("Candidate Epochs", "objective"), self.candidate_epochs_spin)
        left_layout.addWidget(self.objective_group)

        self.annealing_group = QtWidgets.QGroupBox()
        annealing_layout = QtWidgets.QFormLayout(self.annealing_group)
        self.start_temp_spin = QtWidgets.QDoubleSpinBox()
        self.start_temp_spin.setRange(0.001, 100.0)
        self.start_temp_spin.setValue(1.5)
        self.cooling_schedule_combo = QtWidgets.QComboBox()
        self.cooling_schedule_combo.addItems(("geometric", "linear", "logarithmic"))
        self.cooling_parameter_spin = QtWidgets.QDoubleSpinBox()
        self.cooling_parameter_spin.setRange(0.001, 10.0)
        self.cooling_parameter_spin.setValue(0.92)
        self.iter_temp_spin = QtWidgets.QSpinBox()
        self.iter_temp_spin.setRange(1, 1000)
        self.iter_temp_spin.setValue(5)
        self.max_steps_spin = QtWidgets.QSpinBox()
        self.max_steps_spin.setRange(1, 100000)
        self.max_steps_spin.setValue(40)
        self.min_temp_spin = QtWidgets.QDoubleSpinBox()
        self.min_temp_spin.setRange(0.0, 10.0)
        self.min_temp_spin.setValue(0.02)
        annealing_layout.addRow(self.make_help_label("Start Temperature", "annealing_config"), self.start_temp_spin)
        annealing_layout.addRow(self.make_help_label("Cooling", "annealing_config"), self.cooling_schedule_combo)
        annealing_layout.addRow(self.make_help_label("Cooling Parameter", "annealing_config"), self.cooling_parameter_spin)
        annealing_layout.addRow(self.make_help_label("Iter / Temperature", "annealing_config"), self.iter_temp_spin)
        annealing_layout.addRow(self.make_help_label("Max Steps", "annealing_config"), self.max_steps_spin)
        annealing_layout.addRow(self.make_help_label("Min Temperature", "annealing_config"), self.min_temp_spin)
        left_layout.addWidget(self.annealing_group)

        self.neighborhood_group = QtWidgets.QGroupBox("Neighborhood")
        neighborhood_layout = QtWidgets.QVBoxLayout(self.neighborhood_group)
        neighborhood_layout.addWidget(self.make_help_strip("neighborhood"))
        self.neighborhood_checks = {}
        for operation in ("set_neuron", "fill_layer", "swap_neurons"):
            checkbox = QtWidgets.QCheckBox(operation)
            checkbox.setChecked(True)
            neighborhood_layout.addWidget(checkbox)
            self.neighborhood_checks[operation] = checkbox
        left_layout.addWidget(self.neighborhood_group)

        self.run_group = QtWidgets.QGroupBox()
        run_layout = QtWidgets.QGridLayout(self.run_group)
        self.evaluate_button = QtWidgets.QPushButton()
        self.evaluate_button.clicked.connect(self._evaluate_start)
        self.step_button = QtWidgets.QPushButton()
        self.step_button.clicked.connect(self._step_once)
        self.run_button = QtWidgets.QPushButton()
        self.run_button.setProperty("role", "primary")
        self.run_button.clicked.connect(self._run_to_completion)
        self.reset_button = QtWidgets.QPushButton()
        self.reset_button.clicked.connect(self._reset_session)
        run_layout.addWidget(self.evaluate_button, 0, 0)
        run_layout.addWidget(self.step_button, 0, 1)
        run_layout.addWidget(self.run_button, 1, 0)
        run_layout.addWidget(self.reset_button, 1, 1)
        left_layout.addWidget(self.run_group)
        left_layout.addStretch(1)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(1, 1)

        cards_row = QtWidgets.QHBoxLayout()
        self.state_card = InfoCardWidget("State", "-")
        self.objective_card = InfoCardWidget("Objective", "-")
        self.acceptance_card = InfoCardWidget("Acceptance", "-")
        cards_row.addWidget(self.state_card)
        cards_row.addWidget(self.objective_card)
        cards_row.addWidget(self.acceptance_card)
        right_layout.addLayout(cards_row)

        snapshot_row = QtWidgets.QHBoxLayout()
        self.snapshot_combo = QtWidgets.QComboBox()
        self.snapshot_combo.addItems(("preview", "start", "candidate", "current", "best", "end"))
        self.snapshot_combo.currentTextChanged.connect(lambda _value: self._refresh_network_snapshot())
        snapshot_row.addWidget(QtWidgets.QLabel("Snapshot"))
        snapshot_row.addWidget(self.snapshot_combo)
        snapshot_row.addStretch(1)
        right_layout.addLayout(snapshot_row)

        self.tabs = QtWidgets.QTabWidget()
        self.sample_panel = SamplePanelWidget(self.preferences.language)
        self.live_panel = AnnealingLivePanel(self.preferences.language)
        self.decision_panel = AnnealingDecisionPanel(self.preferences.language)
        self.neuron_detail_panel = NeuronDetailPanel(self.preferences.language)
        self.history_panel = AnnealingHistoryPanel(self.preferences.language)
        self.diff_text = QtWidgets.QTextBrowser()
        self.best_layout_text = QtWidgets.QTextBrowser()
        self.help_panel = RecipePanelWidget(self.preferences.language)
        self.help_text = QtWidgets.QTextBrowser()
        self.tabs.addTab(self.sample_panel, "")
        self.tabs.addTab(self.live_panel, "")
        self.tabs.addTab(self.decision_panel, "")
        self.tabs.addTab(self.neuron_detail_panel, "")
        self.tabs.addTab(self.history_panel, "")
        self.tabs.addTab(self.diff_text, "")
        self.tabs.addTab(self.best_layout_text, "")
        self.tabs.addTab(self.help_panel, "")
        self.tabs.addTab(self.help_text, "")
        self.tab_help_button = self.make_info_button("sample_selection")
        self.tabs.setCornerWidget(self.tab_help_button, QtCore.Qt.TopRightCorner)
        self.tabs.currentChanged.connect(self._update_tab_help_topic)

        self.content_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.content_splitter.setChildrenCollapsible(False)
        self.content_splitter.setHandleWidth(10)
        self.network_view = NetworkViewWidget()
        self.network_view.hiddenNeuronSelected.connect(self._on_hidden_selected)
        self.network_container = QtWidgets.QWidget()
        network_container_layout = QtWidgets.QVBoxLayout(self.network_container)
        network_container_layout.setContentsMargins(0, 0, 0, 0)
        network_toolbar = QtWidgets.QHBoxLayout()
        network_toolbar.addStretch(1)
        self.fit_view_button = QtWidgets.QPushButton()
        self.fit_view_button.clicked.connect(self.network_view.reset_view_to_fit)
        self.zoom_in_button = QtWidgets.QPushButton("+")
        self.zoom_in_button.clicked.connect(self.network_view.zoom_in)
        self.zoom_out_button = QtWidgets.QPushButton("-")
        self.zoom_out_button.clicked.connect(self.network_view.zoom_out)
        self.network_info_button = self.make_info_button("network_view")
        self.snapshot_info_button = self.make_info_button("annealing_snapshots")
        network_toolbar.addWidget(self.fit_view_button)
        network_toolbar.addWidget(self.zoom_out_button)
        network_toolbar.addWidget(self.zoom_in_button)
        network_toolbar.addWidget(self.snapshot_info_button)
        network_toolbar.addWidget(self.network_info_button)
        network_container_layout.addLayout(network_toolbar)
        network_container_layout.addWidget(self.network_view, 1)
        self.tabs.setMinimumHeight(220)
        self.content_splitter.addWidget(self.network_container)
        self.content_splitter.addWidget(self.tabs)
        self.content_splitter.setStretchFactor(0, 3)
        self.content_splitter.setStretchFactor(1, 2)
        self.register_secondary_splitter("content", self.content_splitter)
        right_layout.addWidget(self.content_splitter, 1)
        self.retranslate()
        self.apply_detail_mode()

    def retranslate(self) -> None:
        self.problem_group.setTitle("Problem Setup" if self.preferences.language == "en" else "Problem Setup")
        self.start_group.setTitle("Start Layout" if self.preferences.language == "en" else "Startlayout")
        self.objective_group.setTitle("Objective")
        self.annealing_group.setTitle("Annealing Config")
        self.run_group.setTitle("Run Control")
        self.evaluate_button.setText("Evaluate Start" if self.preferences.language == "en" else "Start bewerten")
        self.step_button.setText("Step Once" if self.preferences.language == "en" else "Ein Schritt")
        self.run_button.setText("Run To Completion" if self.preferences.language == "en" else "Bis zum Ende")
        self.reset_button.setText(text(self.preferences.language, "common.reset"))
        self.fit_view_button.setText("Fit")
        self.tabs.setTabText(0, text(self.preferences.language, "common.sample"))
        self.tabs.setTabText(1, "Live Run")
        self.tabs.setTabText(2, "Decision")
        self.tabs.setTabText(3, text(self.preferences.language, "common.neuron_detail"))
        self.tabs.setTabText(4, "History")
        self.tabs.setTabText(5, "Diff")
        self.tabs.setTabText(6, "Best Layout")
        self.tabs.setTabText(7, text(self.preferences.language, "common.recipes"))
        self.tabs.setTabText(8, text(self.preferences.language, "common.help"))
        self.sample_panel.set_language(self.preferences.language)
        self.live_panel.set_language(self.preferences.language)
        self.decision_panel.set_language(self.preferences.language)
        self.neuron_detail_panel.set_language(self.preferences.language)
        self.history_panel.set_language(self.preferences.language)
        self.help_panel.apply_language(self.preferences.language)
        self._refresh_help_tab()
        self._update_tab_help_topic()

    def apply_detail_mode(self) -> None:
        self.tabs.setTabVisible(5, self.preferences.detail_mode == "expert")

    def _on_benchmark_changed(self) -> None:
        benchmark = self.benchmark_combo.currentText()
        defaults = default_hidden_sizes(benchmark)
        self.hidden_sizes_edit.setText(", ".join(str(v) for v in defaults))
        self.layout_editor.set_hidden_sizes(defaults)
        self._load_dataset()
        self._rebuild_preview_model()
        self._refresh_help_tab()
        self._refresh_all_views()

    def _on_hidden_sizes_changed(self) -> None:
        hidden_sizes = parse_hidden_sizes(self.hidden_sizes_edit.text(), self.layout_editor.hidden_sizes())
        self.hidden_sizes_edit.setText(", ".join(str(v) for v in hidden_sizes))
        self.layout_editor.set_hidden_sizes(hidden_sizes)
        self._rebuild_preview_model()
        self._refresh_all_views()

    def _on_layout_changed(self, _layout_spec: str) -> None:
        self._rebuild_preview_model()
        self._refresh_all_views()

    def _on_split_changed(self, split_name: str) -> None:
        self.split_name = split_name
        self._update_sample_bounds()
        self._refresh_live_text()

    def _on_sample_index_changed(self, value: int) -> None:
        self.sample_index = int(value)
        self.sample_slider.blockSignals(True)
        self.sample_spin.blockSignals(True)
        self.sample_slider.setValue(self.sample_index)
        self.sample_spin.setValue(self.sample_index)
        self.sample_slider.blockSignals(False)
        self.sample_spin.blockSignals(False)
        self._refresh_live_text()

    def _on_hidden_selected(self, layer_index: int, neuron_index: int) -> None:
        self.selected_hidden = (layer_index, neuron_index)
        self._refresh_live_text()

    def _load_dataset(self) -> None:
        self.dataset = load_benchmark(DatasetConfig(name=self.benchmark_combo.currentText(), random_state=self.config.random_state))
        self._update_sample_bounds()

    def _update_sample_bounds(self) -> None:
        if self.dataset is None:
            return
        X, _raw, _y = split_arrays(self.dataset, self.split_name)
        maximum = max(0, len(X) - 1)
        self.sample_slider.setRange(0, maximum)
        self.sample_spin.setRange(0, maximum)
        self.sample_slider.setValue(min(self.sample_index, maximum))
        self.sample_spin.setValue(min(self.sample_index, maximum))
        self.sample_index = min(self.sample_index, maximum)

    def _analysis_sample(self) -> AnalysisSample:
        if self.dataset is None:
            raise ValueError("No dataset loaded.")
        return build_analysis_sample(
            self.dataset,
            self.split_name,
            self.sample_index,
            language=self.preferences.language,
        )

    def _rebuild_preview_model(self) -> None:
        if self.dataset is None:
            return
        self.preview_model = ModularMLP(
            input_size=self.dataset.input_size,
            hidden_sizes=self.layout_editor.hidden_sizes(),
            output_size=self.dataset.output_size,
            layout=self.layout_editor.current_layout(),
            weight_scale=self.config.weight_scale,
            random_state=self.config.random_state,
        )
        self.session = None
        self.session_snapshot = None
        self.network_view.set_model(self.preview_model)

    def _current_request(self) -> AnnealingRunRequest:
        operations = tuple(
            operation for operation, checkbox in self.neighborhood_checks.items() if checkbox.isChecked()
        )
        return AnnealingRunRequest(
            dataset_config=DatasetConfig(name=self.benchmark_combo.currentText(), random_state=self.config.random_state),
            hidden_sizes=self.layout_editor.hidden_sizes(),
            layout_spec=self.layout_editor.layout_spec(),
            objective_config=ObjectiveConfig(
                objective_name=self.objective_combo.currentText(),
                candidate_epochs=self.candidate_epochs_spin.value(),
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                weight_scale=self.config.weight_scale,
                random_state=self.config.random_state,
                shuffle=True,
            ),
            annealing_config=AnnealingConfig(
                start_temperature=self.start_temp_spin.value(),
                cooling_schedule=self.cooling_schedule_combo.currentText(),
                cooling_parameter=self.cooling_parameter_spin.value(),
                iterations_per_temperature=self.iter_temp_spin.value(),
                max_steps=self.max_steps_spin.value(),
                min_temperature=self.min_temp_spin.value(),
                neighborhood_operations=operations,
            ),
            random_state=self.config.random_state,
        )

    def _ensure_session(self) -> AnnealingSession:
        if self.session is None:
            self.session = create_session(self._current_request())
            self.dataset = self.session.dataset
        return self.session

    def _evaluate_start(self) -> None:
        self.task_controller.submit(
            evaluate_start,
            self._ensure_session(),
            self.preferences.language,
            on_success=self._on_session_snapshot_ready,
            on_error=self._on_task_error,
            status_message="Evaluating start layout...",
        )

    def _step_once(self) -> None:
        self.task_controller.submit(
            step_once,
            self._ensure_session(),
            self.preferences.language,
            on_success=self._on_session_snapshot_ready,
            on_error=self._on_task_error,
            status_message="Running one annealing step...",
        )

    def _run_to_completion(self) -> None:
        self.task_controller.submit(
            run_to_completion,
            self._ensure_session(),
            self.preferences.language,
            on_success=self._on_session_snapshot_ready,
            on_error=self._on_task_error,
            status_message="Running simulated annealing...",
        )

    def _reset_session(self) -> None:
        if self.session is None:
            self.session = create_session(self._current_request())
        self.session_snapshot = reset(self.session, self.preferences.language)
        self.snapshot_combo.setCurrentText("preview")
        self._refresh_all_views()
        self.statusMessage.emit("Annealing reset." if self.preferences.language == "en" else "Annealing zurueckgesetzt.")

    def _on_session_snapshot_ready(self, snapshot: AnnealingSessionSnapshot) -> None:
        self.session_snapshot = snapshot
        if snapshot.is_complete:
            self.snapshot_combo.setCurrentText("best")
            self.statusMessage.emit("Simulated annealing finished.")
        else:
            self.statusMessage.emit("Annealing session updated.")
        self._refresh_all_views()

    def _on_task_error(self, traceback_text: str) -> None:
        QtWidgets.QMessageBox.critical(self, "Task failed", traceback_text)
        self.statusMessage.emit("Task failed.")

    def _selected_snapshot_model(self) -> ModularMLP | None:
        if self.session_snapshot is None:
            return self.preview_model
        snapshot = self.snapshot_combo.currentText()
        if snapshot == "preview":
            return self.preview_model
        if snapshot == "start":
            return (
                self.session_snapshot.start_evaluation.trained_model
                if self.session_snapshot.start_evaluation is not None
                else self.preview_model
            )
        if snapshot == "candidate":
            return (
                self.session_snapshot.candidate_evaluation.trained_model
                if self.session_snapshot.candidate_evaluation is not None
                else (
                    self.session_snapshot.current_evaluation.trained_model
                    if self.session_snapshot.current_evaluation is not None
                    else self.preview_model
                )
            )
        if snapshot == "current":
            return (
                self.session_snapshot.current_evaluation.trained_model
                if self.session_snapshot.current_evaluation is not None
                else self.preview_model
            )
        if snapshot == "best":
            return (
                self.session_snapshot.best_evaluation.trained_model
                if self.session_snapshot.best_evaluation is not None
                else self.preview_model
            )
        if snapshot == "end":
            return (
                self.session_snapshot.end_evaluation.trained_model
                if self.session_snapshot.end_evaluation is not None
                else (
                    self.session_snapshot.current_evaluation.trained_model
                    if self.session_snapshot.current_evaluation is not None
                    else self.preview_model
                )
            )
        return self.preview_model

    def _refresh_network_snapshot(self) -> None:
        model = self._selected_snapshot_model()
        if self.dataset is not None and model is not None:
            self.network_view.set_input_projection(
                build_input_projection(self.dataset, model, self._analysis_sample(), self.selected_hidden)
            )
        self.network_view.set_model(model)
        self.network_view.set_selected_hidden(self.selected_hidden)
        self._refresh_live_text()

    def _refresh_all_views(self) -> None:
        self._refresh_network_snapshot()
        if self.session_snapshot is None:
            self.state_card.set_content(value="preview", body=self.layout_editor.layout_spec())
            self.objective_card.set_content(value=self.objective_combo.currentText(), body="No SA run yet.")
            self.acceptance_card.set_content(value="-", body="No annealing history yet.")
            self.live_panel.set_placeholder(self.preferences.language)
            self.decision_panel.set_placeholder(self.preferences.language)
            self.history_panel.set_placeholder(self.preferences.language)
            self.diff_text.setPlainText(self.layout_editor.current_layout().to_compact_spec())
            self.best_layout_text.setPlainText(self.layout_editor.current_layout().to_compact_spec())
        else:
            snapshot = self.session_snapshot
            step_index = snapshot.last_step.step_index if snapshot.last_step is not None else 0
            best_eval = snapshot.best_evaluation or snapshot.current_evaluation
            current_eval = snapshot.current_evaluation
            self.state_card.set_content(value=f"step {step_index}", body=f"T={snapshot.current_temperature:.3f}")
            self.objective_card.set_content(
                value=f"{best_eval.objective_value:.4f}" if best_eval is not None else "-",
                body=(
                    f"val_acc={best_eval.val_accuracy:.3f}" if best_eval is not None else "No objective yet."
                ),
            )
            self.acceptance_card.set_content(
                value=f"{snapshot.acceptance_rate:.3f}",
                body=f"accepted={snapshot.accepted_steps}",
            )
            self.live_panel.set_snapshot(snapshot, self.preferences.language)
            self.decision_panel.set_snapshot(snapshot, self.preferences.language)
            self.history_panel.set_snapshot(snapshot, self.preferences.language)
            if snapshot.start_evaluation is not None and snapshot.best_evaluation is not None:
                self.diff_text.setPlainText(
                    render_layout_diff(snapshot.start_evaluation.layout, snapshot.best_evaluation.layout)
                )
                self.best_layout_text.setPlainText(
                    render_layout(snapshot.best_evaluation.layout, title="Best Layout")
                )
            elif current_eval is not None:
                self.diff_text.setPlainText(current_eval.layout.to_compact_spec())
                self.best_layout_text.setPlainText(current_eval.layout.to_compact_spec())
        self._refresh_live_text()

    def _refresh_live_text(self) -> None:
        if self.dataset is None:
            return
        model = self._selected_snapshot_model()
        if model is None:
            return
        analysis_sample = self._analysis_sample()
        sample = analysis_sample.scaled_sample.reshape(1, -1)
        probabilities = model.predict_proba(sample)[0]
        prediction_index = int(model.predict(sample)[0])
        self.sample_panel.set_sample(
            SampleDisplayPayload(
                dataset=self.dataset,
                analysis_sample=analysis_sample,
                prediction_name=self.dataset.target_names[prediction_index],
                probabilities=probabilities,
            )
        )
        if self.session_snapshot is None:
            self.live_panel.set_placeholder(self.preferences.language)
            self.decision_panel.set_placeholder(self.preferences.language)
            self.history_panel.set_placeholder(self.preferences.language)
        else:
            self.live_panel.set_snapshot(self.session_snapshot, self.preferences.language)
            self.decision_panel.set_snapshot(self.session_snapshot, self.preferences.language)
            self.history_panel.set_snapshot(self.session_snapshot, self.preferences.language)
        self._refresh_neuron_detail()

    def _refresh_neuron_detail(self) -> None:
        if self.dataset is None or self.selected_hidden is None:
            self.neuron_detail_panel.set_placeholder(self.preferences.language)
            return
        model = self._selected_snapshot_model()
        if model is None:
            self.neuron_detail_panel.set_placeholder(self.preferences.language)
            return
        layer_index, neuron_index = self.selected_hidden
        payload = build_neuron_analysis_payload(
            model,
            self.dataset,
            self._analysis_sample(),
            layer_index,
            neuron_index,
            self.preferences.language,
        )
        self.neuron_detail_panel.set_payload(payload, self.preferences.language)

    def _refresh_help_tab(self) -> None:
        self.help_text.setHtml(
            workspace_help_html(
                self.workspace_id,
                self.benchmark_combo.currentText(),
                self.preferences.language,
            )
        )

    def _update_tab_help_topic(self) -> None:
        topic_by_index = {
            0: "sample_selection",
            1: "annealing_live",
            2: "annealing_decision",
            3: "neuron_tracker",
            4: "annealing_history",
            5: "annealing_snapshots",
            6: "annealing_snapshots",
            7: "recipes",
            8: "workspace_help",
        }
        self.tab_help_button.set_topic_key(topic_by_index.get(self.tabs.currentIndex(), "workspace_help"))
