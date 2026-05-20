"""Gemeinsamer Workflow fuer Training, SA-Suche und finalen Layout-Vergleich."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from activations import parse_layout_spec
from annealing import AnnealingConfig
from annealing_objectives import ObjectiveConfig
from benchmarks import DatasetBundle, load_benchmark
from configs import (
    DEFAULT_ANNEALING_CANDIDATE_EPOCHS,
    DEFAULT_ANNEALING_COOLING_PARAMETER,
    DEFAULT_ANNEALING_COOLING_SCHEDULE,
    DEFAULT_ANNEALING_ITERATIONS_PER_TEMPERATURE,
    DEFAULT_ANNEALING_MAX_STEPS,
    DEFAULT_ANNEALING_MIN_TEMPERATURE,
    DEFAULT_ANNEALING_NEIGHBORHOODS,
    DEFAULT_ANNEALING_OBJECTIVE,
    DEFAULT_ANNEALING_START_TEMPERATURE,
    DEFAULT_ONLINE_TRAIN_POLICY,
    DEFAULT_SA_EVALUATION_MODE,
    DatasetConfig,
    GuiExperimentConfig,
    SUPPORTED_ONLINE_TRAIN_POLICIES,
    SUPPORTED_SA_EVALUATION_MODES,
    SUPPORTED_BENCHMARKS,
    TrainingConfig,
    default_epochs,
    default_hidden_sizes,
)
from model import ModularMLP
from services.analysis_sample_service import AnalysisSample, build_analysis_sample
from services.annealing_service import AnnealingRunRequest
from services.annealing_session_service import (
    AnnealingSession,
    AnnealingSessionSnapshot,
    create_session,
    evaluate_start,
    reset,
    run_to_completion,
    step_once,
)
from services.layout_evaluation_service import (
    InheritedModelCandidate,
    LayoutEvaluationRequest,
    LayoutEvaluationResult,
    build_standard_layout_candidates,
    evaluate_inherited_models,
    run_layout_evaluation,
    with_inherited_model_evaluations,
)
from services.network_projection_service import build_input_projection
from services.neuron_analysis_service import build_neuron_analysis_payload
from services.online_annealing_training_service import (
    OnlineAnnealingSession,
    OnlineAnnealingSnapshot,
    OnlineAnnealingRequest,
    create_online_session,
    evaluate_online_start,
    online_reset,
    online_run_steps,
    online_run_to_completion,
    online_step_once,
)
from services.stepper_service import build_step_entries
from services.training_service import TrainingRunArtifacts, TrainingRunRequest, run_single_training_experiment
from ui_qt.state import WorkspacePreferences
from ui_qt.tasking import BackgroundTaskController
from ui_qt.utils import parse_hidden_sizes
from ui_qt.widgets.annealing_decision_panel import AnnealingDecisionPanel
from ui_qt.widgets.annealing_history_panel import AnnealingHistoryPanel
from ui_qt.widgets.annealing_live_panel import AnnealingLivePanel
from ui_qt.widgets.info_card import InfoCardWidget
from ui_qt.widgets.layout_editor import LayoutEditorWidget
from ui_qt.widgets.network_view import NetworkViewWidget
from ui_qt.widgets.neuron_detail_panel import NeuronDetailPanel
from ui_qt.widgets.plot_widgets import TrainingPlotWidget
from ui_qt.widgets.sample_panel import SampleDisplayPayload, SamplePanelWidget
from ui_qt.widgets.stepper_panel import StepperPanel
from .base import BaseWorkspace


def _run_short_session_steps(session: AnnealingSession, count: int, language: str):
    snapshot = None
    for _ in range(max(1, count)):
        snapshot = step_once(session, language)
        if snapshot.is_complete:
            break
    return snapshot if snapshot is not None else evaluate_start(session, language)


def _run_layout_evaluation_with_inherited(
    request: LayoutEvaluationRequest,
    inherited_candidates: tuple[InheritedModelCandidate, ...],
) -> LayoutEvaluationResult:
    result = run_layout_evaluation(request)
    if not inherited_candidates:
        return result
    inherited_runs = evaluate_inherited_models(
        request.dataset_config,
        inherited_candidates,
        primary_metric=request.primary_metric,
    )
    return with_inherited_model_evaluations(result, inherited_runs, request.primary_metric)


class ActivationWorkflowWorkspace(BaseWorkspace):
    """Ein Fenster fuer den eigentlichen Projektablauf."""

    workspace_id = "activation_workflow"

    def __init__(
        self,
        config: GuiExperimentConfig,
        preferences: WorkspacePreferences,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(preferences, parent)
        self.config = config
        self.dataset: DatasetBundle | None = None
        self.analysis_sample: AnalysisSample | None = None
        self.current_model: ModularMLP | None = None
        self.training_artifacts: TrainingRunArtifacts | None = None
        self.annealing_session: AnnealingSession | OnlineAnnealingSession | None = None
        self.annealing_snapshot: AnnealingSessionSnapshot | OnlineAnnealingSnapshot | None = None
        self.layout_evaluation: LayoutEvaluationResult | None = None
        self.selected_hidden: tuple[int, int] | None = (0, 0)
        self.task_controller = BackgroundTaskController(self)
        self.task_controller.busyChanged.connect(self.busyChanged)
        self.task_controller.messageEmitted.connect(self.statusMessage)
        self._build_ui()
        self._load_dataset()
        self._rebuild_preview_model()
        self._refresh_views()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(10)
        root.addWidget(self.splitter)

        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_container = QtWidgets.QWidget()
        left_scroll.setWidget(left_container)
        self.splitter.addWidget(left_scroll)

        left_layout = QtWidgets.QVBoxLayout(left_container)
        left_layout.setContentsMargins(8, 8, 8, 8)

        self.setup_group = QtWidgets.QGroupBox("1. Benchmark und Layout")
        setup_layout = QtWidgets.QFormLayout(self.setup_group)
        self.benchmark_combo = QtWidgets.QComboBox()
        self.benchmark_combo.addItems(SUPPORTED_BENCHMARKS)
        self.benchmark_combo.setCurrentText(self.config.benchmark)
        self.benchmark_combo.currentTextChanged.connect(self._on_benchmark_changed)
        self.hidden_sizes_edit = QtWidgets.QLineEdit(", ".join(str(size) for size in self.config.hidden_sizes))
        self.hidden_sizes_edit.editingFinished.connect(self._on_hidden_sizes_changed)
        setup_layout.addRow("Benchmark", self.benchmark_combo)
        setup_layout.addRow("Hidden Sizes", self.hidden_sizes_edit)
        self.reset_workflow_button = QtWidgets.QPushButton("Workflow zurücksetzen")
        self.reset_workflow_button.clicked.connect(self._reset_workflow)
        setup_layout.addRow(self.reset_workflow_button)
        left_layout.addWidget(self.setup_group)

        self.layout_group = QtWidgets.QGroupBox("2. Aktivierungs-Layout")
        layout_group_layout = QtWidgets.QVBoxLayout(self.layout_group)
        self.layout_editor = LayoutEditorWidget(self.config.hidden_sizes, self.config.layout_spec)
        self.layout_editor.layoutSpecChanged.connect(self._on_layout_changed)
        layout_group_layout.addWidget(self.layout_editor)
        left_layout.addWidget(self.layout_group)

        self.training_group = QtWidgets.QGroupBox("3. Manuelles Training")
        training_layout = QtWidgets.QFormLayout(self.training_group)
        self.epochs_spin = QtWidgets.QSpinBox()
        self.epochs_spin.setRange(1, 5000)
        self.epochs_spin.setValue(self.config.epochs)
        self.epochs_spin.valueChanged.connect(lambda _value: self._on_training_config_changed(reload_dataset=False))
        self.lr_spin = QtWidgets.QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 10.0)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setSingleStep(0.003)
        self.lr_spin.setValue(self.config.learning_rate)
        self.lr_spin.valueChanged.connect(lambda _value: self._on_training_config_changed(reload_dataset=False))
        self.batch_spin = QtWidgets.QSpinBox()
        self.batch_spin.setRange(1, 4096)
        self.batch_spin.setValue(self.config.batch_size)
        self.batch_spin.valueChanged.connect(lambda _value: self._on_training_config_changed(reload_dataset=False))
        self.weight_spin = QtWidgets.QDoubleSpinBox()
        self.weight_spin.setRange(0.0001, 10.0)
        self.weight_spin.setDecimals(4)
        self.weight_spin.setSingleStep(0.01)
        self.weight_spin.setValue(self.config.weight_scale)
        self.weight_spin.valueChanged.connect(lambda _value: self._on_training_config_changed(reload_dataset=False))
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 1_000_000)
        self.seed_spin.setValue(self.config.random_state)
        self.seed_spin.valueChanged.connect(lambda _value: self._on_training_config_changed(reload_dataset=True))
        training_layout.addRow("Epochs", self.epochs_spin)
        training_layout.addRow("Learning Rate", self.lr_spin)
        training_layout.addRow("Batch Size", self.batch_spin)
        training_layout.addRow("Weight Scale", self.weight_spin)
        training_layout.addRow("Seed", self.seed_spin)
        self.train_button = QtWidgets.QPushButton("Gewähltes Layout trainieren")
        self.train_button.setProperty("role", "primary")
        self.train_button.clicked.connect(self._run_manual_training)
        training_layout.addRow(self.train_button)
        left_layout.addWidget(self.training_group)

        self.sa_group = QtWidgets.QGroupBox("4. Simulated Annealing")
        sa_layout = QtWidgets.QFormLayout(self.sa_group)
        self.sa_mode_combo = QtWidgets.QComboBox()
        self.sa_mode_combo.addItems(SUPPORTED_SA_EVALUATION_MODES)
        self.sa_mode_combo.setCurrentText(DEFAULT_SA_EVALUATION_MODE)
        self.sa_mode_combo.currentTextChanged.connect(self._on_sa_mode_changed)
        self.online_train_policy_combo = QtWidgets.QComboBox()
        self.online_train_policy_combo.addItems(SUPPORTED_ONLINE_TRAIN_POLICIES)
        self.online_train_policy_combo.setCurrentText(DEFAULT_ONLINE_TRAIN_POLICY)
        self.online_train_policy_combo.currentTextChanged.connect(lambda _value: self._on_sa_config_changed())
        self.candidate_epochs_spin = QtWidgets.QSpinBox()
        self.candidate_epochs_spin.setRange(1, 1000)
        self.candidate_epochs_spin.setValue(DEFAULT_ANNEALING_CANDIDATE_EPOCHS)
        self.candidate_epochs_spin.valueChanged.connect(lambda _value: self._on_sa_config_changed())
        self.max_steps_spin = QtWidgets.QSpinBox()
        self.max_steps_spin.setRange(1, 10000)
        self.max_steps_spin.setValue(DEFAULT_ANNEALING_MAX_STEPS)
        self.max_steps_spin.valueChanged.connect(lambda _value: self._on_sa_config_changed())
        self.temperature_spin = QtWidgets.QDoubleSpinBox()
        self.temperature_spin.setRange(0.001, 100.0)
        self.temperature_spin.setDecimals(3)
        self.temperature_spin.setValue(DEFAULT_ANNEALING_START_TEMPERATURE)
        self.temperature_spin.valueChanged.connect(lambda _value: self._on_sa_config_changed())
        sa_layout.addRow("SA Evaluation Mode", self.sa_mode_combo)
        sa_layout.addRow("Online Train Policy", self.online_train_policy_combo)
        self.online_train_policy_label = sa_layout.labelForField(self.online_train_policy_combo)
        sa_layout.addRow("Candidate Epochs (short-retrain only)", self.candidate_epochs_spin)
        self.candidate_epochs_label = sa_layout.labelForField(self.candidate_epochs_spin)
        sa_layout.addRow("Max Steps", self.max_steps_spin)
        sa_layout.addRow("Start Temperature", self.temperature_spin)
        sa_buttons = QtWidgets.QHBoxLayout()
        self.evaluate_start_button = QtWidgets.QPushButton("Start bewerten")
        self.step_once_button = QtWidgets.QPushButton("1 Schritt")
        self.step_ten_button = QtWidgets.QPushButton("10 Schritte")
        self.run_sa_button = QtWidgets.QPushButton("SA bis Ende")
        self.reset_sa_button = QtWidgets.QPushButton("Reset")
        self.evaluate_start_button.clicked.connect(self._evaluate_start)
        self.step_once_button.clicked.connect(self._step_once)
        self.step_ten_button.clicked.connect(self._step_ten)
        self.run_sa_button.clicked.connect(self._run_sa_to_completion)
        self.reset_sa_button.clicked.connect(self._reset_sa)
        sa_buttons.addWidget(self.evaluate_start_button)
        sa_buttons.addWidget(self.step_once_button)
        sa_buttons.addWidget(self.step_ten_button)
        sa_buttons.addWidget(self.run_sa_button)
        sa_buttons.addWidget(self.reset_sa_button)
        sa_layout.addRow(sa_buttons)
        left_layout.addWidget(self.sa_group)

        self.compare_group = QtWidgets.QGroupBox("5. Finale Layouts vergleichen")
        compare_layout = QtWidgets.QVBoxLayout(self.compare_group)
        self.compare_button = QtWidgets.QPushButton("Retrained vs Inherited Layouts vergleichen")
        self.compare_button.clicked.connect(self._run_final_layout_comparison)
        compare_layout.addWidget(self.compare_button)
        left_layout.addWidget(self.compare_group)
        left_layout.addStretch(1)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(1, 1)

        card_row = QtWidgets.QHBoxLayout()
        self.benchmark_card = InfoCardWidget("Benchmark")
        self.training_card = InfoCardWidget("Training")
        self.sa_card = InfoCardWidget("Annealing")
        card_row.addWidget(self.benchmark_card)
        card_row.addWidget(self.training_card)
        card_row.addWidget(self.sa_card)
        right_layout.addLayout(card_row)

        self.tabs = QtWidgets.QTabWidget()
        self.network_view = NetworkViewWidget()
        self.network_view.hiddenNeuronSelected.connect(self._on_hidden_selected)
        self.sample_panel = SamplePanelWidget(self.preferences.language)
        self.training_plot = TrainingPlotWidget()
        self.neuron_detail_panel = NeuronDetailPanel(self.preferences.language)
        self.stepper_panel = StepperPanel(self.preferences.language)
        self.annealing_live_panel = AnnealingLivePanel(self.preferences.language)
        self.annealing_history_panel = AnnealingHistoryPanel(self.preferences.language)
        self.annealing_decision_panel = AnnealingDecisionPanel(self.preferences.language)
        self.final_table = QtWidgets.QTableWidget(0, 8)
        self.final_table.setHorizontalHeaderLabels(
            ["Rank", "Type", "Layout", "Val Loss", "Val Acc", "Test Loss", "Test Acc", "Seeds"]
        )
        self.final_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.help_text = QtWidgets.QTextBrowser()
        self.help_text.setHtml(self._help_html())
        self.tabs.addTab(self.network_view, "Netz")
        self.tabs.addTab(self.sample_panel, "Sample")
        self.tabs.addTab(self.neuron_detail_panel, "Neuron Detail")
        self.tabs.addTab(self.stepper_panel, "Stepper")
        self.tabs.addTab(self.training_plot, "Training")
        self.tabs.addTab(self.annealing_live_panel, "SA Delta")
        self.tabs.addTab(self.annealing_history_panel, "SA History")
        self.tabs.addTab(self.annealing_decision_panel, "SA Entscheidung")
        self.tabs.addTab(self.final_table, "Finaler Vergleich")
        self.tabs.addTab(self.help_text, "Ablauf")
        right_layout.addWidget(self.tabs, 1)
        self._update_sa_mode_controls()

    def _load_dataset(self) -> None:
        benchmark = self.benchmark_combo.currentText()
        self.dataset = load_benchmark(DatasetConfig(name=benchmark, random_state=self.seed_spin.value()))
        self.analysis_sample = build_analysis_sample(
            self.dataset,
            "val",
            0,
            language=self.preferences.language,
        )
        self.training_artifacts = None
        self.annealing_session = None
        self.annealing_snapshot = None
        self.layout_evaluation = None
        self.selected_hidden = (0, 0)

    def _rebuild_preview_model(self) -> None:
        if self.dataset is None:
            return
        layout = parse_layout_spec(self.layout_editor.layout_spec(), self.layout_editor.hidden_sizes())
        self.current_model = ModularMLP(
            input_size=self.dataset.input_size,
            hidden_sizes=self.layout_editor.hidden_sizes(),
            output_size=self.dataset.model_output_size,
            layout=layout,
            num_classes=self.dataset.output_size,
            weight_scale=self.weight_spin.value(),
            random_state=self.seed_spin.value(),
        )

    def _refresh_views(self) -> None:
        if self.dataset is None or self.current_model is None or self.analysis_sample is None:
            return
        projection = build_input_projection(
            self.dataset,
            self.current_model,
            self.analysis_sample,
            self.selected_hidden,
        )
        self.network_view.set_input_projection(projection)
        self.network_view.set_selected_hidden(self.selected_hidden)
        self.network_view.set_model(self.current_model)
        probabilities = self.current_model.predict_proba(self.analysis_sample.scaled_sample.reshape(1, -1))[0]
        prediction_index = int(self.current_model.predict(self.analysis_sample.scaled_sample.reshape(1, -1))[0])
        self.sample_panel.set_sample(
            SampleDisplayPayload(
                dataset=self.dataset,
                analysis_sample=self.analysis_sample,
                prediction_name=self.dataset.target_names[prediction_index],
                probabilities=probabilities,
            )
        )
        self._refresh_neuron_detail()
        self._refresh_stepper()
        history = self.training_artifacts.training_result.history if self.training_artifacts is not None else None
        self.training_plot.set_history(history)
        self._refresh_cards()
        self._refresh_annealing_panels()
        self._refresh_final_table()

    def _refresh_neuron_detail(self) -> None:
        if self.dataset is None or self.current_model is None or self.analysis_sample is None:
            self.neuron_detail_panel.set_placeholder(self.preferences.language)
            return
        layer_index, neuron_index = self.selected_hidden or (0, 0)
        try:
            payload = build_neuron_analysis_payload(
                self.current_model,
                self.dataset,
                self.analysis_sample,
                layer_index,
                neuron_index,
                self.preferences.language,
            )
            self.neuron_detail_panel.set_payload(payload, self.preferences.language)
        except ValueError:
            self.neuron_detail_panel.set_placeholder(self.preferences.language)

    def _refresh_stepper(self) -> None:
        if self.dataset is None or self.current_model is None or self.analysis_sample is None:
            self.stepper_panel.set_placeholder(self.preferences.language)
            return
        trace = self.current_model.trace_sample(
            self.analysis_sample.scaled_sample.reshape(1, -1),
            target_index=self.analysis_sample.effective_target_index,
        )
        entries = build_step_entries(
            trace,
            self.dataset,
            self.analysis_sample,
            self.preferences.language,
        )
        self.stepper_panel.set_entries(entries)

    def _refresh_cards(self) -> None:
        if self.dataset is None:
            return
        self.benchmark_card.set_content(
            value=self.dataset.name,
            body=(
                f"{self.dataset.input_size} Inputs, {self.dataset.output_size} Klassen, "
                f"Train/Val/Test {self.dataset.train_size}/{self.dataset.validation_size}/{self.dataset.test_size}"
            ),
        )
        if self.training_artifacts is None:
            self.training_card.set_content(value="nicht trainiert", body=self.layout_editor.layout_spec())
        else:
            metrics = self.training_artifacts.training_result.test_metrics
            self.training_card.set_content(
                value=f"test acc {metrics['accuracy']:.3f}",
                body=f"layout {self.training_artifacts.training_layout.to_compact_spec()}",
            )
        if self.annealing_snapshot is None or self.annealing_snapshot.best_evaluation is None:
            self.sa_card.set_content(value="kein SA-Lauf", body="Bestlayout noch offen")
        else:
            best = self.annealing_snapshot.best_evaluation
            self.sa_card.set_content(
                value=f"best {best.objective_value:.4f}",
                body=best.layout.to_compact_spec(),
            )

    def _refresh_annealing_panels(self) -> None:
        if self.annealing_snapshot is None:
            self.annealing_live_panel.set_placeholder(self.preferences.language)
            self.annealing_history_panel.set_placeholder(self.preferences.language)
            self.annealing_decision_panel.set_placeholder(self.preferences.language)
            return
        self.annealing_live_panel.set_snapshot(self.annealing_snapshot, self.preferences.language)
        self.annealing_history_panel.set_snapshot(self.annealing_snapshot, self.preferences.language)
        self.annealing_decision_panel.set_snapshot(self.annealing_snapshot, self.preferences.language)

    def _refresh_final_table(self) -> None:
        if self.layout_evaluation is None:
            self.final_table.setRowCount(0)
            return
        ranking = self.layout_evaluation.combined_ranking or self.layout_evaluation.ranking
        self.final_table.setRowCount(len(ranking))
        for row_index, item in enumerate(ranking):
            values = (
                row_index + 1,
                item.evaluation_type,
                item.label,
                item.mean_metrics["val_loss"],
                item.mean_metrics["val_accuracy"],
                item.mean_metrics["test_loss"],
                item.mean_metrics["test_accuracy"],
                item.num_runs,
            )
            for column_index, value in enumerate(values):
                if isinstance(value, float):
                    text = f"{value:.4f}"
                else:
                    text = str(value)
                table_item = QtWidgets.QTableWidgetItem(text)
                table_item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                self.final_table.setItem(row_index, column_index, table_item)

    def _on_benchmark_changed(self, benchmark: str) -> None:
        hidden_sizes = default_hidden_sizes(benchmark)
        self.hidden_sizes_edit.setText(", ".join(str(size) for size in hidden_sizes))
        self.epochs_spin.setValue(default_epochs(benchmark))
        self.layout_editor.set_hidden_sizes(hidden_sizes)
        self.layout_editor.set_layout_spec("relu")
        self._load_dataset()
        self._rebuild_preview_model()
        self._refresh_views()

    def _on_hidden_sizes_changed(self) -> None:
        try:
            hidden_sizes = parse_hidden_sizes(self.hidden_sizes_edit.text(), self.layout_editor.hidden_sizes())
            self.layout_editor.set_hidden_sizes(hidden_sizes)
            self._rebuild_preview_model()
            self._refresh_views()
        except ValueError as exc:
            self.statusMessage.emit(str(exc))

    def _on_layout_changed(self, _layout_spec: str) -> None:
        self.training_artifacts = None
        self.layout_evaluation = None
        self.annealing_session = None
        self.annealing_snapshot = None
        self._rebuild_preview_model()
        self._refresh_views()

    def _on_training_config_changed(self, *, reload_dataset: bool) -> None:
        self.training_artifacts = None
        self.layout_evaluation = None
        self.annealing_session = None
        self.annealing_snapshot = None
        if reload_dataset:
            self._load_dataset()
        self._rebuild_preview_model()
        self._refresh_views()

    def _on_sa_config_changed(self) -> None:
        self.layout_evaluation = None
        self.annealing_session = None
        self.annealing_snapshot = None
        self._refresh_views()

    def _reset_workflow(self) -> None:
        self.training_artifacts = None
        self.layout_evaluation = None
        self.annealing_session = None
        self.annealing_snapshot = None
        self.selected_hidden = (0, 0)
        self._load_dataset()
        self._rebuild_preview_model()
        self._refresh_views()
        self.statusMessage.emit("Workflow zurückgesetzt.")

    def _on_hidden_selected(self, layer_index: int, neuron_index: int) -> None:
        self.selected_hidden = (layer_index, neuron_index)
        self._refresh_views()

    def _run_manual_training(self) -> None:
        request = TrainingRunRequest(
            dataset_config=DatasetConfig(name=self.benchmark_combo.currentText(), random_state=self.seed_spin.value()),
            hidden_sizes=self.layout_editor.hidden_sizes(),
            layout_spec=self.layout_editor.layout_spec(),
            training_config=TrainingConfig(
                epochs=self.epochs_spin.value(),
                learning_rate=self.lr_spin.value(),
                batch_size=self.batch_spin.value(),
                random_state=self.seed_spin.value(),
                shuffle=True,
            ),
            weight_scale=self.weight_spin.value(),
            random_state=self.seed_spin.value(),
        )
        self.task_controller.submit(
            run_single_training_experiment,
            request,
            on_success=self._on_training_finished,
            on_error=self._on_task_error,
            status_message="Trainiere Layout...",
        )

    def _on_training_finished(self, artifacts: TrainingRunArtifacts) -> None:
        self.dataset = artifacts.dataset
        self.training_artifacts = artifacts
        self.current_model = artifacts.model
        self.analysis_sample = build_analysis_sample(
            artifacts.dataset,
            "val",
            0,
            language=self.preferences.language,
        )
        self.statusMessage.emit("Training abgeschlossen.")
        self._refresh_views()

    def _on_sa_mode_changed(self, _mode: str) -> None:
        self.annealing_session = None
        self.annealing_snapshot = None
        self._update_sa_mode_controls()
        self._refresh_views()

    def _update_sa_mode_controls(self) -> None:
        is_online = self.sa_mode_combo.currentText() == "online_delta"
        self.candidate_epochs_spin.setVisible(not is_online)
        if self.candidate_epochs_label is not None:
            self.candidate_epochs_label.setVisible(not is_online)
        self.online_train_policy_combo.setEnabled(is_online)
        self.online_train_policy_combo.setVisible(is_online)
        if self.online_train_policy_label is not None:
            self.online_train_policy_label.setVisible(is_online)

    def _ensure_annealing_session(self) -> AnnealingSession | OnlineAnnealingSession:
        if self.annealing_session is None:
            annealing_config = AnnealingConfig(
                start_temperature=self.temperature_spin.value(),
                cooling_schedule=DEFAULT_ANNEALING_COOLING_SCHEDULE,
                cooling_parameter=DEFAULT_ANNEALING_COOLING_PARAMETER,
                iterations_per_temperature=DEFAULT_ANNEALING_ITERATIONS_PER_TEMPERATURE,
                max_steps=self.max_steps_spin.value(),
                min_temperature=DEFAULT_ANNEALING_MIN_TEMPERATURE,
                neighborhood_operations=DEFAULT_ANNEALING_NEIGHBORHOODS,
            )
            dataset_config = DatasetConfig(name=self.benchmark_combo.currentText(), random_state=self.seed_spin.value())
            if self.sa_mode_combo.currentText() == "online_delta":
                self.annealing_session = create_online_session(
                    OnlineAnnealingRequest(
                        dataset_config=dataset_config,
                        hidden_sizes=self.layout_editor.hidden_sizes(),
                        layout_spec=self.layout_editor.layout_spec(),
                        training_config=TrainingConfig(
                            epochs=self.epochs_spin.value(),
                            learning_rate=self.lr_spin.value(),
                            batch_size=self.batch_spin.value(),
                            random_state=self.seed_spin.value(),
                            shuffle=True,
                        ),
                        annealing_config=annealing_config,
                        weight_scale=self.weight_spin.value(),
                        random_state=self.seed_spin.value(),
                        train_policy=self.online_train_policy_combo.currentText(),
                    )
                )
            else:
                request = AnnealingRunRequest(
                    dataset_config=dataset_config,
                    hidden_sizes=self.layout_editor.hidden_sizes(),
                    layout_spec=self.layout_editor.layout_spec(),
                    objective_config=ObjectiveConfig(
                        objective_name=DEFAULT_ANNEALING_OBJECTIVE,
                        candidate_epochs=self.candidate_epochs_spin.value(),
                        learning_rate=self.lr_spin.value(),
                        batch_size=self.batch_spin.value(),
                        weight_scale=self.weight_spin.value(),
                        random_state=self.seed_spin.value(),
                        shuffle=True,
                    ),
                    annealing_config=annealing_config,
                    random_state=self.seed_spin.value(),
                )
                self.annealing_session = create_session(request)
        return self.annealing_session

    def _evaluate_start(self) -> None:
        session = self._ensure_annealing_session()
        function = evaluate_online_start if self.sa_mode_combo.currentText() == "online_delta" else evaluate_start
        self.task_controller.submit(
            function,
            session,
            self.preferences.language,
            on_success=self._on_annealing_snapshot,
            on_error=self._on_task_error,
            status_message="Bewerte Startlayout...",
        )

    def _step_once(self) -> None:
        session = self._ensure_annealing_session()
        function = online_step_once if self.sa_mode_combo.currentText() == "online_delta" else step_once
        self.task_controller.submit(
            function,
            session,
            self.preferences.language,
            on_success=self._on_annealing_snapshot,
            on_error=self._on_task_error,
            status_message="Fuehre SA-Schritt aus...",
        )

    def _step_ten(self) -> None:
        session = self._ensure_annealing_session()
        if self.sa_mode_combo.currentText() == "online_delta":
            self.task_controller.submit(
                online_run_steps,
                session,
                10,
                self.preferences.language,
                on_success=self._on_annealing_snapshot,
                on_error=self._on_task_error,
                status_message="Fuehre 10 Online-SA-Schritte aus...",
            )
            return
        self.task_controller.submit(
            _run_short_session_steps,
            session,
            10,
            self.preferences.language,
            on_success=self._on_annealing_snapshot,
            on_error=self._on_task_error,
            status_message="Fuehre 10 SA-Schritte aus...",
        )

    def _run_sa_to_completion(self) -> None:
        session = self._ensure_annealing_session()
        function = online_run_to_completion if self.sa_mode_combo.currentText() == "online_delta" else run_to_completion
        self.task_controller.submit(
            function,
            session,
            self.preferences.language,
            on_success=self._on_annealing_snapshot,
            on_error=self._on_task_error,
            status_message="Fuehre SA bis zum Ende aus...",
        )

    def _reset_sa(self) -> None:
        if self.annealing_session is None:
            return
        if self.sa_mode_combo.currentText() == "online_delta":
            self.annealing_snapshot = online_reset(self.annealing_session, self.preferences.language)  # type: ignore[arg-type]
        else:
            self.annealing_snapshot = reset(self.annealing_session, self.preferences.language)  # type: ignore[arg-type]
        self._refresh_views()

    def _on_annealing_snapshot(self, snapshot: AnnealingSessionSnapshot | OnlineAnnealingSnapshot) -> None:
        self.annealing_snapshot = snapshot
        if snapshot.best_evaluation is not None:
            self.current_model = snapshot.best_evaluation.trained_model
        self.statusMessage.emit("Annealing-Zustand aktualisiert.")
        self._refresh_views()

    def _run_final_layout_comparison(self) -> None:
        inherited_candidates: tuple[InheritedModelCandidate, ...] = ()
        best_layout = (
            self.annealing_snapshot.best_evaluation.layout.to_compact_spec()
            if self.annealing_snapshot is not None and self.annealing_snapshot.best_evaluation is not None
            else None
        )
        end_layout = (
            self.annealing_snapshot.current_evaluation.layout.to_compact_spec()
            if self.annealing_snapshot is not None and self.annealing_snapshot.current_evaluation is not None
            else None
        )
        if (
            self.annealing_snapshot is not None
            and getattr(self.annealing_snapshot, "sa_evaluation_mode", "") == "online_delta"
            and self.annealing_snapshot.best_evaluation is not None
            and self.annealing_snapshot.current_evaluation is not None
        ):
            inherited_candidates = (
                InheritedModelCandidate(
                    "best_inherited_model_from_sa",
                    self.annealing_snapshot.best_evaluation.trained_model.to_state_dict(),
                    self.seed_spin.value(),
                ),
                InheritedModelCandidate(
                    "end_inherited_model_from_sa",
                    self.annealing_snapshot.current_evaluation.trained_model.to_state_dict(),
                    self.seed_spin.value(),
                ),
            )
        request = LayoutEvaluationRequest(
            dataset_config=DatasetConfig(name=self.benchmark_combo.currentText(), random_state=self.seed_spin.value()),
            hidden_sizes=self.layout_editor.hidden_sizes(),
            candidates=build_standard_layout_candidates(
                self.layout_editor.hidden_sizes(),
                start_layout_spec=self.layout_editor.layout_spec(),
                best_layout_spec=best_layout,
                end_layout_spec=end_layout,
                random_state=self.seed_spin.value(),
            ),
            seeds=(self.seed_spin.value(),),
            training_config=TrainingConfig(
                epochs=self.epochs_spin.value(),
                learning_rate=self.lr_spin.value(),
                batch_size=self.batch_spin.value(),
                random_state=self.seed_spin.value(),
                shuffle=True,
            ),
            weight_scale=self.weight_spin.value(),
            primary_metric="validation_loss",
        )
        self.task_controller.submit(
            _run_layout_evaluation_with_inherited,
            request,
            inherited_candidates,
            on_success=self._on_layout_evaluation_finished,
            on_error=self._on_task_error,
            status_message="Vergleiche retrained und inherited Layouts...",
        )

    def _on_layout_evaluation_finished(self, result: LayoutEvaluationResult) -> None:
        self.layout_evaluation = result
        self.tabs.setCurrentWidget(self.final_table)
        self.statusMessage.emit("Finaler Layout-Vergleich abgeschlossen.")
        self._refresh_views()

    def _on_task_error(self, traceback_text: str) -> None:
        self.statusMessage.emit("Fehler im Workflow.")
        QtWidgets.QMessageBox.critical(self, "Workflow Fehler", traceback_text)

    def retranslate(self) -> None:
        self.sample_panel.set_language(self.preferences.language)
        self.neuron_detail_panel.set_language(self.preferences.language)
        self.stepper_panel.set_language(self.preferences.language)
        self.annealing_live_panel.set_language(self.preferences.language)
        self.annealing_history_panel.set_language(self.preferences.language)
        self.annealing_decision_panel.set_language(self.preferences.language)
        self.help_text.setHtml(self._help_html())
        self._refresh_views()

    def apply_detail_mode(self) -> None:
        pass

    def _help_html(self) -> str:
        return (
            "<h2>Activation Workflow</h2>"
            "<p><b>Ablauf:</b> CSV-Benchmark laden, Layout wählen, optional SA ausführen, "
            "danach Start-, Best-, End-, Random- und homogene Baseline-Layouts unter gleichen "
            "Trainingsbedingungen final vergleichen.</p>"
            "<p><b>Wichtig:</b> SA trainiert nicht final das Modell. SA sucht zuerst ein Layout. "
            "Die belastbare Aussage entsteht erst im finalen Layout-Vergleich.</p>"
        )
