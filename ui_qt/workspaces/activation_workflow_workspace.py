"""Fokussierte GUI-Demo fuer Random-Mixed Online-Delta-SA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from activations import parse_layout_spec, random_layout_spec
from annealing import AnnealingConfig
from benchmarks import DatasetBundle, load_benchmark
from configs import (
    DEFAULT_ANNEALING_COOLING_SCHEDULE,
    DEFAULT_ANNEALING_NEIGHBORHOODS,
    DatasetConfig,
    GuiExperimentConfig,
    SUPPORTED_BENCHMARKS,
    TrainingConfig,
    default_hidden_sizes,
)
from model import ModularMLP
from services.analysis_sample_service import AnalysisSample, build_analysis_sample
from services.gui_profile_service import SUPPORTED_GUI_PROFILES, load_gui_benchmark_profile
from services.gui_multi_run_service import (
    GUI_COMPARISON_MODES,
    GuiMultiRunSession,
    GuiRunRecord,
    build_gui_seed_schedule,
    create_gui_multi_run_session,
    latest_gui_multi_run_session,
    load_gui_multi_run_session,
    save_gui_multi_run_session,
)
from services.layout_evaluation_service import (
    InheritedModelCandidate,
    LayoutEvaluationRequest,
    LayoutEvaluationResult,
    build_online_delta_layout_candidates,
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
from ui_qt.widgets.annealing_history_panel import AnnealingHistoryPanel
from ui_qt.widgets.info_card import InfoCardWidget
from ui_qt.widgets.layout_editor import LayoutEditorWidget
from ui_qt.widgets.multi_run_widgets import (
    AggregateAnnealingWidget,
    AggregateFinalComparisonWidget,
    AggregateLayoutHeatmapWidget,
    AggregatePlaceholderWidget,
)
from ui_qt.widgets.network_comparison_panel import NetworkComparisonPanel
from ui_qt.widgets.neuron_detail_panel import NeuronDetailPanel
from ui_qt.widgets.online_delta_timeline_panel import OnlineDeltaTimelinePanel
from ui_qt.widgets.plot_widgets import TrainingPlotWidget
from ui_qt.widgets.sample_panel import SampleDisplayPayload, SamplePanelWidget
from ui_qt.widgets.stepper_panel import StepperPanel
from .base import BaseWorkspace


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


@dataclass(frozen=True)
class _FinalComparisonJob:
    request: LayoutEvaluationRequest
    inherited_candidates: tuple[InheritedModelCandidate, ...]


def _run_online_request_to_completion(
    request: OnlineAnnealingRequest,
) -> tuple[OnlineAnnealingSession, OnlineAnnealingSnapshot]:
    session = create_online_session(request)
    return session, online_run_to_completion(session, "de")


def _run_final_comparison_job(job: _FinalComparisonJob) -> LayoutEvaluationResult:
    return _run_layout_evaluation_with_inherited(job.request, job.inherited_candidates)


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
        self.gui_profile = load_gui_benchmark_profile(config.gui_profile, config.benchmark)
        self.dataset: DatasetBundle | None = None
        self.analysis_sample: AnalysisSample | None = None
        self.current_model: ModularMLP | None = None
        self.training_artifacts: TrainingRunArtifacts | None = None
        self.annealing_session: OnlineAnnealingSession | None = None
        self.annealing_snapshot: OnlineAnnealingSnapshot | None = None
        self.layout_evaluation: LayoutEvaluationResult | None = None
        self.selected_hidden: tuple[int, int] | None = (0, 0)
        self.run_records: list[GuiRunRecord] = []
        self.active_run_index = 0
        self.aggregate_selected = False
        self.multi_session: GuiMultiRunSession | None = None
        self._batch_stage: str | None = None
        self._batch_completed: list[int] = []
        self._batch_failures = 0
        self._batch_records: list[GuiRunRecord] = []
        self._random_layout_rng = np.random.default_rng(config.random_state)
        self._active_random_layout_spec: str | None = None
        self._setting_random_layout = False
        self.task_controller = BackgroundTaskController(self)
        self.task_controller.busyChanged.connect(self.busyChanged)
        self.task_controller.busyChanged.connect(self._set_busy_controls)
        self.task_controller.messageEmitted.connect(self.statusMessage)
        self._build_ui()
        self._load_dataset()
        self._rebuild_preview_model()
        self._initialize_single_record()
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
        self.comparison_mode_combo = QtWidgets.QComboBox()
        self.comparison_mode_combo.addItem("Blocked Layout Comparison", "blocked_layout")
        self.comparison_mode_combo.addItem("Robustheitsanalyse", "robustness")
        self.comparison_mode_combo.currentIndexChanged.connect(self._on_comparison_mode_changed)
        setup_layout.addRow("Methodisches Design", self.comparison_mode_combo)
        self.comparison_mode_note = QtWidgets.QLabel()
        self.comparison_mode_note.setWordWrap(True)
        setup_layout.addRow("Designhinweis", self.comparison_mode_note)
        self.benchmark_combo = QtWidgets.QComboBox()
        self.benchmark_combo.addItems(SUPPORTED_BENCHMARKS)
        self.benchmark_combo.setCurrentText(self.config.benchmark)
        self.benchmark_combo.currentTextChanged.connect(self._on_benchmark_changed)
        setup_layout.addRow("Benchmark", self.benchmark_combo)
        self.run_mode_combo = QtWidgets.QComboBox()
        self.run_mode_combo.addItem("Single", "single")
        self.run_mode_combo.addItem("Multi Seeds", "multi")
        self.run_mode_combo.currentIndexChanged.connect(self._on_run_mode_changed)
        setup_layout.addRow("Run-Modus", self.run_mode_combo)
        self.multi_count_combo = QtWidgets.QComboBox()
        for count in range(2, 11):
            self.multi_count_combo.addItem(str(count), count)
        self.multi_count_combo.setCurrentText("10")
        self.multi_count_combo.currentIndexChanged.connect(self._refresh_multi_controls)
        setup_layout.addRow("Random-Mixed-Layouts", self.multi_count_combo)
        self.gui_profile_combo = QtWidgets.QComboBox()
        self.gui_profile_combo.addItems(SUPPORTED_GUI_PROFILES)
        self.gui_profile_combo.setCurrentText(self.config.gui_profile)
        self.gui_profile_combo.currentTextChanged.connect(self._on_gui_profile_changed)
        setup_layout.addRow("Parameterprofil", self.gui_profile_combo)
        self.gui_profile_note = QtWidgets.QLabel()
        self.gui_profile_note.setWordWrap(True)
        setup_layout.addRow("Profilhinweis", self.gui_profile_note)
        self.reset_workflow_button = QtWidgets.QPushButton("Workflow zurücksetzen")
        self.reset_workflow_button.clicked.connect(self._reset_workflow)
        setup_layout.addRow(self.reset_workflow_button)
        session_buttons = QtWidgets.QHBoxLayout()
        self.load_latest_session_button = QtWidgets.QPushButton("Letzte Multi-Session laden")
        self.open_session_button = QtWidgets.QPushButton("Session öffnen…")
        self.load_latest_session_button.clicked.connect(self._load_latest_multi_session)
        self.open_session_button.clicked.connect(self._open_multi_session)
        session_buttons.addWidget(self.load_latest_session_button)
        session_buttons.addWidget(self.open_session_button)
        setup_layout.addRow(session_buttons)
        left_layout.addWidget(self.setup_group)

        self.layout_group = QtWidgets.QGroupBox("2. Aktivierungs-Layout")
        layout_group_layout = QtWidgets.QVBoxLayout(self.layout_group)
        initial_hidden_sizes = default_hidden_sizes(self.config.benchmark)
        initial_layout_spec = self._next_random_layout_spec(initial_hidden_sizes)
        self._active_random_layout_spec = initial_layout_spec
        self.layout_editor = LayoutEditorWidget(initial_hidden_sizes, initial_layout_spec)
        self.layout_editor.layoutSpecChanged.connect(self._on_layout_changed)
        layout_group_layout.addWidget(self.layout_editor)
        self.random_layout_button = QtWidgets.QPushButton("Neues Random-Mixed-Layout")
        self.random_layout_button.clicked.connect(self._generate_random_layout)
        layout_group_layout.addWidget(self.random_layout_button)
        left_layout.addWidget(self.layout_group)

        self.training_group = QtWidgets.QGroupBox("3. Random-Baseline trainieren")
        training_layout = QtWidgets.QFormLayout(self.training_group)
        self.epochs_spin = QtWidgets.QSpinBox()
        self.epochs_spin.setRange(1, 5000)
        self.epochs_spin.setValue(self._initial_training_value("epochs", self.config.epochs))
        self.epochs_spin.valueChanged.connect(lambda _value: self._on_training_config_changed(reload_dataset=False))
        self.lr_spin = QtWidgets.QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 10.0)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setSingleStep(0.003)
        self.lr_spin.setValue(self._initial_training_value("learning_rate", self.config.learning_rate))
        self.lr_spin.valueChanged.connect(lambda _value: self._on_training_config_changed(reload_dataset=False))
        self.batch_spin = QtWidgets.QSpinBox()
        self.batch_spin.setRange(1, 4096)
        self.batch_spin.setValue(self._initial_training_value("batch_size", self.config.batch_size))
        self.batch_spin.valueChanged.connect(lambda _value: self._on_training_config_changed(reload_dataset=False))
        self.weight_spin = QtWidgets.QDoubleSpinBox()
        self.weight_spin.setRange(0.0001, 10.0)
        self.weight_spin.setDecimals(4)
        self.weight_spin.setSingleStep(0.01)
        self.weight_spin.setValue(self._initial_training_value("weight_scale", self.config.weight_scale))
        self.weight_spin.valueChanged.connect(lambda _value: self._on_training_config_changed(reload_dataset=False))
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 1_000_000)
        self.seed_spin.setValue(self.config.random_state)
        self.seed_spin.valueChanged.connect(lambda _value: self._on_training_config_changed(reload_dataset=True))
        training_layout.addRow("Epochen", self.epochs_spin)
        training_layout.addRow("Lernrate", self.lr_spin)
        training_layout.addRow("Batch-Groesse", self.batch_spin)
        training_layout.addRow("Xavier-Skalierung", self.weight_spin)
        training_layout.addRow("Master-Seed", self.seed_spin)
        self.train_button = QtWidgets.QPushButton("Aktuelles Layout als Baseline trainieren")
        self.train_button.setProperty("role", "primary")
        self.train_button.clicked.connect(self._run_manual_training)
        training_layout.addRow(self.train_button)
        self.train_all_button = QtWidgets.QPushButton("Alle Baselines trainieren")
        self.train_all_button.clicked.connect(self._run_all_baselines)
        training_layout.addRow(self.train_all_button)
        left_layout.addWidget(self.training_group)

        self.sa_group = QtWidgets.QGroupBox("4. Online-Delta-SA")
        sa_layout = QtWidgets.QFormLayout(self.sa_group)
        self.neighborhood_value = QtWidgets.QLabel(", ".join(DEFAULT_ANNEALING_NEIGHBORHOODS))
        self.online_learning_rate_value = QtWidgets.QLabel()
        self.online_batch_size_value = QtWidgets.QLabel()
        self.cooling_schedule_value = QtWidgets.QLabel()
        self.cooling_parameter_value = QtWidgets.QLabel()
        self.iterations_per_temperature_value = QtWidgets.QLabel()
        self.min_temperature_value = QtWidgets.QLabel()
        self.max_steps_spin = QtWidgets.QSpinBox()
        self.max_steps_spin.setRange(1, 100000)
        self.max_steps_spin.setValue(self._online_profile_value("max_steps"))
        self.max_steps_spin.valueChanged.connect(lambda _value: self._on_sa_config_changed())
        self.temperature_spin = QtWidgets.QDoubleSpinBox()
        self.temperature_spin.setRange(0.000000000001, 100.0)
        self.temperature_spin.setDecimals(12)
        self.temperature_spin.setSingleStep(0.001)
        self.temperature_spin.setValue(self._online_profile_value("start_temperature"))
        self.temperature_spin.valueChanged.connect(lambda _value: self._on_sa_config_changed())
        sa_layout.addRow("Nachbarschaft", self.neighborhood_value)
        sa_layout.addRow("Online-Lernrate", self.online_learning_rate_value)
        sa_layout.addRow("Online-Batch-Groesse", self.online_batch_size_value)
        sa_layout.addRow("Cooling-Schedule", self.cooling_schedule_value)
        sa_layout.addRow("Cooling-Parameter", self.cooling_parameter_value)
        sa_layout.addRow("Iterationen / Temperatur", self.iterations_per_temperature_value)
        sa_layout.addRow("Mindesttemperatur", self.min_temperature_value)
        sa_layout.addRow("Maximale Schritte", self.max_steps_spin)
        sa_layout.addRow("Starttemperatur", self.temperature_spin)
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
        self.run_all_sa_button = QtWidgets.QPushButton("SA alle bis Ende")
        self.run_all_sa_button.clicked.connect(self._run_all_sa)
        sa_layout.addRow(self.run_all_sa_button)
        left_layout.addWidget(self.sa_group)

        self.compare_group = QtWidgets.QGroupBox("5. Finale Layouts vergleichen")
        compare_layout = QtWidgets.QVBoxLayout(self.compare_group)
        self.compare_button = QtWidgets.QPushButton("Retrainierte und geerbte Layouts vergleichen")
        self.compare_button.clicked.connect(self._run_final_layout_comparison)
        compare_layout.addWidget(self.compare_button)
        self.compare_all_button = QtWidgets.QPushButton("Alle final vergleichen")
        self.compare_all_button.clicked.connect(self._run_all_final_comparisons)
        compare_layout.addWidget(self.compare_all_button)
        left_layout.addWidget(self.compare_group)

        self.progress_group = QtWidgets.QGroupBox("Multi-Run-Fortschritt")
        progress_layout = QtWidgets.QVBoxLayout(self.progress_group)
        self.multi_progress_bar = QtWidgets.QProgressBar()
        self.multi_progress_label = QtWidgets.QLabel("Kein Sammellauf aktiv.")
        self.multi_progress_label.setWordWrap(True)
        self.cancel_batch_button = QtWidgets.QPushButton("Sammellauf nach aktuellem Run abbrechen")
        self.cancel_batch_button.clicked.connect(self.task_controller.cancel_sequential)
        progress_layout.addWidget(self.multi_progress_bar)
        progress_layout.addWidget(self.multi_progress_label)
        progress_layout.addWidget(self.cancel_batch_button)
        left_layout.addWidget(self.progress_group)
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
        self.run_selector = QtWidgets.QTabBar()
        self.run_selector.setExpanding(False)
        self.run_selector.currentChanged.connect(self._on_run_selector_changed)
        right_layout.addWidget(self.run_selector)
        right_layout.addLayout(card_row)

        self.tabs = QtWidgets.QTabWidget()
        self.network_comparison_panel = NetworkComparisonPanel()
        self.network_comparison_panel.hiddenNeuronSelected.connect(self._on_hidden_selected)
        self.network_view = self.network_comparison_panel.start_view
        self.sample_panel = SamplePanelWidget(self.preferences.language)
        self.training_plot = TrainingPlotWidget()
        self.neuron_detail_panel = NeuronDetailPanel(self.preferences.language)
        self.stepper_panel = StepperPanel(self.preferences.language)
        self.annealing_history_panel = AnnealingHistoryPanel(self.preferences.language)
        self.online_delta_timeline_panel = OnlineDeltaTimelinePanel()

        self.final_comparison_detail = QtWidgets.QWidget()
        final_comparison_layout = QtWidgets.QVBoxLayout(self.final_comparison_detail)
        final_comparison_layout.setContentsMargins(0, 0, 0, 0)
        self.final_table_label = QtWidgets.QLabel("Online-Delta Vergleich")
        self.final_table_label.setStyleSheet("font-weight: 700;")
        self.final_table = QtWidgets.QTableWidget(0, 8)
        self.final_table.setHorizontalHeaderLabels(
            ["Rang", "Typ", "Layout", "Val Loss", "Val Acc", "Test Loss", "Test Acc", "Runs"]
        )
        self.final_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        final_comparison_layout.addWidget(self.final_table_label)
        final_comparison_layout.addWidget(self.final_table, 1)

        self.aggregate_network_panel = AggregateLayoutHeatmapWidget()
        self.aggregate_training_plot = TrainingPlotWidget()
        self.aggregate_annealing_panel = AggregateAnnealingWidget()
        self.aggregate_final_panel = AggregateFinalComparisonWidget()
        self.network_stack = self._stacked_pair(self.network_comparison_panel, self.aggregate_network_panel)
        self.sample_stack = self._stacked_pair(
            self.sample_panel,
            AggregatePlaceholderWidget("Datenpunkt"),
        )
        self.neuron_stack = self._stacked_pair(
            self.neuron_detail_panel,
            AggregatePlaceholderWidget("Neuron-Details"),
        )
        self.stepper_stack = self._stacked_pair(
            self.stepper_panel,
            AggregatePlaceholderWidget("Rechenschritte"),
        )
        self.training_stack = self._stacked_pair(self.training_plot, self.aggregate_training_plot)
        self.annealing_stack = self._stacked_pair(
            self.annealing_history_panel,
            self.aggregate_annealing_panel,
        )
        self.timeline_stack = self._stacked_pair(
            self.online_delta_timeline_panel,
            AggregatePlaceholderWidget("SA-Timeline"),
        )
        self.final_comparison_tab = self._stacked_pair(
            self.final_comparison_detail,
            self.aggregate_final_panel,
        )
        self._view_stacks = (
            self.network_stack,
            self.sample_stack,
            self.neuron_stack,
            self.stepper_stack,
            self.training_stack,
            self.annealing_stack,
            self.timeline_stack,
            self.final_comparison_tab,
        )

        self.tabs.addTab(self.network_stack, "Netzvergleich")
        self.tabs.addTab(self.sample_stack, "Datenpunkt")
        self.tabs.addTab(self.neuron_stack, "Neuron-Details")
        self.tabs.addTab(self.stepper_stack, "Rechenschritte")
        self.tabs.addTab(self.training_stack, "Training")
        self.tabs.addTab(self.annealing_stack, "SA-Verlauf")
        self.tabs.addTab(self.timeline_stack, "SA-Timeline")
        self.tabs.addTab(self.final_comparison_tab, "Finaler Vergleich")
        right_layout.addWidget(self.tabs, 1)
        self._refresh_online_profile_labels()
        self._refresh_multi_controls()

    @staticmethod
    def _stacked_pair(
        run_widget: QtWidgets.QWidget,
        aggregate_widget: QtWidgets.QWidget,
    ) -> QtWidgets.QStackedWidget:
        stack = QtWidgets.QStackedWidget()
        stack.addWidget(run_widget)
        stack.addWidget(aggregate_widget)
        return stack

    def _initial_training_value(self, key: str, fallback: float | int) -> float | int:
        if self.config.gui_profile == "demo":
            return fallback
        return self.gui_profile["training"][key]

    def _online_profile_value(self, key: str) -> float | int:
        return self.gui_profile["online_delta"][key]

    def _refresh_online_profile_labels(self) -> None:
        online_delta = self.gui_profile["online_delta"]
        self.gui_profile_note.setText(str(self.gui_profile.get("description", "")))
        self.online_learning_rate_value.setText(str(self._online_profile_value("online_learning_rate")))
        self.online_batch_size_value.setText(str(self._online_profile_value("online_batch_size")))
        self.cooling_schedule_value.setText(
            str(online_delta.get("cooling_schedule", DEFAULT_ANNEALING_COOLING_SCHEDULE))
        )
        self.cooling_parameter_value.setText(f"{float(online_delta['cooling_parameter']):.8g}")
        self.iterations_per_temperature_value.setText(str(online_delta["iterations_per_temperature"]))
        self.min_temperature_value.setText(f"{float(online_delta['min_temperature']):.8g}")

    def _apply_gui_profile_to_controls(self) -> None:
        training = self.gui_profile["training"]
        online_delta = self.gui_profile["online_delta"]
        self.epochs_spin.setValue(int(training["epochs"]))
        self.lr_spin.setValue(float(training["learning_rate"]))
        self.batch_spin.setValue(int(training["batch_size"]))
        self.weight_spin.setValue(float(training["weight_scale"]))
        self.max_steps_spin.setValue(int(online_delta["max_steps"]))
        self.temperature_spin.setValue(float(online_delta["start_temperature"]))
        self._refresh_online_profile_labels()

    def _initialize_single_record(self) -> None:
        schedule = build_gui_seed_schedule(
            self.benchmark_combo.currentText(),
            0,
            self.seed_spin.value(),
            self._comparison_mode(),
        )
        self.run_records = [
            GuiRunRecord(
                run_index=0,
                schedule=schedule,
                start_layout_spec=self.layout_editor.layout_spec(),
                dataset=self.dataset,
                current_model=self.current_model,
            )
        ]
        self.active_run_index = 0
        self.aggregate_selected = False
        self.multi_session = None
        self._rebuild_run_selector()

    def _active_record(self) -> GuiRunRecord | None:
        if self.aggregate_selected or not self.run_records:
            return None
        return self.run_records[self.active_run_index]

    def _active_schedule(self):
        record = self._active_record()
        if record is not None:
            return record.schedule
        return build_gui_seed_schedule(
            self.benchmark_combo.currentText(),
            0,
            self.seed_spin.value(),
            self._comparison_mode(),
        )

    def _sync_globals_to_active_record(self) -> None:
        record = self._active_record()
        if record is None:
            return
        record.start_layout_spec = self.layout_editor.layout_spec()
        record.dataset = self.dataset
        record.current_model = self.current_model
        record.training_artifacts = self.training_artifacts
        record.annealing_session = self.annealing_session
        record.annealing_snapshot = self.annealing_snapshot
        record.layout_evaluation = self.layout_evaluation

    def _load_active_record(self) -> None:
        record = self._active_record()
        if record is None:
            return
        self._setting_random_layout = True
        try:
            self.layout_editor.set_layout_spec(record.start_layout_spec)
        finally:
            self._setting_random_layout = False
        self._active_random_layout_spec = record.start_layout_spec
        self.dataset = record.dataset or load_benchmark(
            DatasetConfig(
                name=self.benchmark_combo.currentText(),
                random_state=record.schedule.data_split_seed,
            )
        )
        self.analysis_sample = build_analysis_sample(
            self.dataset,
            "val",
            0,
            language=self.preferences.language,
        )
        self.training_artifacts = record.training_artifacts
        self.annealing_session = record.annealing_session
        self.annealing_snapshot = record.annealing_snapshot
        self.layout_evaluation = record.layout_evaluation
        if record.current_model is not None:
            self.current_model = record.current_model
        elif self.annealing_snapshot is not None and self.annealing_snapshot.current_evaluation is not None:
            self.current_model = self.annealing_snapshot.current_evaluation.trained_model
        elif self.training_artifacts is not None:
            self.current_model = self.training_artifacts.model
        else:
            self._rebuild_preview_model()

    def _config_snapshot(self) -> dict[str, object]:
        online_delta = self.gui_profile["online_delta"]
        return {
            "epochs": self.epochs_spin.value(),
            "learning_rate": self.lr_spin.value(),
            "batch_size": self.batch_spin.value(),
            "weight_scale": self.weight_spin.value(),
            "start_temperature": self.temperature_spin.value(),
            "max_steps": self.max_steps_spin.value(),
            "online_learning_rate": online_delta["online_learning_rate"],
            "online_batch_size": online_delta["online_batch_size"],
            "cooling_schedule": online_delta.get(
                "cooling_schedule",
                DEFAULT_ANNEALING_COOLING_SCHEDULE,
            ),
            "cooling_parameter": online_delta["cooling_parameter"],
            "iterations_per_temperature": online_delta["iterations_per_temperature"],
            "min_temperature": online_delta["min_temperature"],
            "methodological_note": (
                self._comparison_mode_description()
            ),
            "comparison_mode": self._comparison_mode(),
        }

    def _load_dataset(self) -> None:
        benchmark = self.benchmark_combo.currentText()
        schedule = self._active_schedule()
        self.dataset = load_benchmark(DatasetConfig(name=benchmark, random_state=schedule.data_split_seed))
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
            random_state=self._active_schedule().retraining_weight_seed,
        )

    def _refresh_views(self) -> None:
        if self.aggregate_selected:
            self._refresh_aggregate_views()
            return
        for stack in self._view_stacks:
            stack.setCurrentIndex(0)
        self.layout_editor.setEnabled(True)
        if self.dataset is None or self.current_model is None or self.analysis_sample is None:
            return
        projection = build_input_projection(
            self.dataset,
            self.current_model,
            self.analysis_sample,
            self.selected_hidden,
        )
        self.network_comparison_panel.set_input_projection(projection)
        self.network_comparison_panel.set_selected_hidden(self.selected_hidden)
        self.network_comparison_panel.set_preview_model(self.current_model)
        self.network_comparison_panel.set_snapshot(self.annealing_snapshot)
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
        self.training_plot.set_history(
            history,
            self._final_training_overlay_histories(),
            self._final_training_reference_metrics(),
        )
        self._refresh_cards()
        self._refresh_annealing_panels()
        self._refresh_final_table()
        self._refresh_run_selector_labels()

    def _refresh_aggregate_views(self) -> None:
        for stack in self._view_stacks:
            stack.setCurrentIndex(1)
        self.layout_editor.setEnabled(False)
        self.aggregate_network_panel.set_runs(self.run_records, self.layout_editor.hidden_sizes())
        baseline_histories = [
            record.training_artifacts.training_result.history
            for record in self.run_records
            if record.training_artifacts is not None
        ]
        start_histories = []
        end_histories = []
        for record in self.run_records:
            if record.layout_evaluation is None:
                continue
            start_histories.extend(
                run.history
                for run in record.layout_evaluation.runs
                if run.label == "random_start_layout" and run.history
            )
            end_histories.extend(
                run.history
                for run in record.layout_evaluation.runs
                if run.label == "end_sa_layout" and run.history
            )
        groups: list[tuple[str, list[dict[str, list[float]]]]] = []
        if baseline_histories:
            groups.append(("Baseline", baseline_histories))
        if start_histories:
            groups.append(("Random Start retrained", start_histories))
        if end_histories:
            groups.append(("Final SA retrained", end_histories))
        self.aggregate_training_plot.set_aggregate_histories(groups)
        self.aggregate_annealing_panel.set_runs(self.run_records)
        self.aggregate_final_panel.set_runs(self.run_records)
        completed = sum(record.status == "complete" for record in self.run_records)
        failed = sum(record.status == "failed" for record in self.run_records)
        self.benchmark_card.set_content(
            value=f"{len(self.run_records)} Multi-Runs",
            body=self._comparison_mode_description(),
        )
        self.training_card.set_content(
            value=f"{completed}/{len(self.run_records)} komplett",
            body=f"Fehlgeschlagen: {failed}",
        )
        self.sa_card.set_content(
            value="Median + IQR",
            body="Aggregationen enthalten nur vorhandene kompatible Run-Ergebnisse.",
        )

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
                f"{self.dataset.input_size} Eingaben, {self.dataset.output_size} Klassen, "
                f"Train/Val/Test {self.dataset.train_size}/{self.dataset.validation_size}/{self.dataset.test_size}"
            ),
        )
        if self.training_artifacts is None:
            self.training_card.set_content(value="nicht trainiert", body=self.layout_editor.layout_spec())
        else:
            metrics = self.training_artifacts.training_result.test_metrics
            self.training_card.set_content(
                value=f"Test-Acc {metrics['accuracy']:.3f}",
                body=f"Layout {self.training_artifacts.training_layout.to_compact_spec()}",
            )
        if self.annealing_snapshot is None or self.annealing_snapshot.current_evaluation is None:
            self.sa_card.set_content(value="kein SA-Lauf", body="Endzustand noch offen")
        else:
            current = self.annealing_snapshot.current_evaluation
            self.sa_card.set_content(
                value=f"aktuell val {current.val_loss:.4f}",
                body=current.layout.to_compact_spec(),
            )

    def _refresh_annealing_panels(self) -> None:
        if self.annealing_snapshot is None:
            self.annealing_history_panel.set_placeholder(self.preferences.language)
            self.online_delta_timeline_panel.set_snapshot(None)
            return
        self.annealing_history_panel.set_snapshot(self.annealing_snapshot, self.preferences.language)
        self.online_delta_timeline_panel.set_snapshot(self.annealing_snapshot)

    def _refresh_final_table(self) -> None:
        if self.layout_evaluation is None:
            self.final_table.setRowCount(0)
            return
        ranking = self.layout_evaluation.combined_ranking or self.layout_evaluation.ranking
        active_ranking = tuple(
            item for item in ranking if not str(item.label).startswith("all_")
        )
        self._populate_comparison_table(self.final_table, active_ranking)

    def _populate_comparison_table(self, table: QtWidgets.QTableWidget, ranking: tuple[object, ...]) -> None:
        """Schreibt eine gerankte Vergleichsliste mit Heatmap-Faerbung in eine Tabelle."""

        table.setRowCount(len(ranking))
        if not ranking:
            return
        losses = [float(item.mean_metrics["val_loss"]) for item in ranking]
        best_loss = min(losses)
        worst_loss = max(losses)
        for row_index, item in enumerate(ranking):
            loss = float(item.mean_metrics["val_loss"])
            relative_loss = (
                (loss - best_loss) / (worst_loss - best_loss)
                if worst_loss > best_loss
                else 0.0
            )
            background = QtGui.QBrush(self._comparison_heatmap_color(relative_loss))
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
                table_item.setBackground(background)
                if row_index in {0, len(ranking) - 1}:
                    font = table_item.font()
                    font.setBold(True)
                    table_item.setFont(font)
                table.setItem(row_index, column_index, table_item)

    def _final_training_overlay_histories(self) -> list[tuple[str, dict[str, list[float]]]] | None:
        """Liefert Retraining-Kurven fuer den Training-Tab nach finalem Vergleich."""

        if self.layout_evaluation is None:
            return None
        labels = (
            ("random_start_layout", "Random Start Layout"),
            ("end_sa_layout", "Final SA-Layout"),
        )
        histories: list[tuple[str, dict[str, list[float]]]] = []
        seen: set[str] = set()
        for run_label, display_label in labels:
            matching_run = next(
                (
                    run
                    for run in self.layout_evaluation.runs
                    if run.label == run_label
                    and run.evaluation_type == "retrained"
                    and run.history
                ),
                None,
            )
            if matching_run is None or matching_run.layout_spec in seen:
                continue
            histories.append((display_label, matching_run.history))
            seen.add(matching_run.layout_spec)
        return histories or None

    def _final_training_reference_metrics(self) -> list[tuple[str, dict[str, float]]] | None:
        """Liefert geerbte SA-Modellwerte als Referenzlinien fuer den Training-Tab."""

        if self.layout_evaluation is None:
            return None
        matching_run = next(
            (
                run
                for run in self.layout_evaluation.inherited_runs
                if run.label == "best_online_delta_value"
            ),
            None,
        )
        if matching_run is None:
            return None
        return [("Best Online Delta Value", matching_run.metrics)]

    @staticmethod
    def _comparison_heatmap_color(relative_loss: float) -> QtGui.QColor:
        """Faerbt gute Validation-Loss-Werte gruen und schlechte rot."""

        clamped = max(0.0, min(1.0, relative_loss))
        if clamped <= 0.5:
            ratio = clamped * 2.0
            start = (220, 252, 231)
            end = (254, 249, 195)
        else:
            ratio = (clamped - 0.5) * 2.0
            start = (254, 249, 195)
            end = (254, 226, 226)
        return QtGui.QColor(
            *(
                round(left + (right - left) * ratio)
                for left, right in zip(start, end, strict=True)
            )
        )

    def _is_multi_mode(self) -> bool:
        return self.run_mode_combo.currentData() == "multi"

    def _comparison_mode(self) -> str:
        mode = str(self.comparison_mode_combo.currentData() or "blocked_layout")
        return mode if mode in GUI_COMPARISON_MODES else "blocked_layout"

    def _comparison_mode_description(self) -> str:
        if self._comparison_mode() == "blocked_layout":
            return (
                "Blocked Layout Comparison: Alle Layouts teilen Datensplit, Startgewichte, "
                "Batch-Reihenfolgen sowie SA-Proposal- und Acceptance-Streams. Nur das "
                "Random-Mixed-Startlayout variiert. Ein GUI-Set ist ein Block; fuer "
                "Inferenz sind mehrere Master-Seeds erforderlich."
            )
        return (
            "Robustheitsanalyse: Layout, Datensplit, Startgewichte, Batch-Reihenfolgen "
            "und SA-Zufallsströme variieren zwischen Runs."
        )

    def _refresh_multi_controls(self) -> None:
        multi = self._is_multi_mode()
        self.multi_count_combo.setEnabled(multi)
        self.train_all_button.setVisible(multi)
        self.run_all_sa_button.setVisible(multi)
        self.compare_all_button.setVisible(multi)
        self.progress_group.setVisible(multi)
        self.run_selector.setVisible(multi)
        self.cancel_batch_button.setEnabled(multi and self.task_controller.is_busy)
        count = int(self.multi_count_combo.currentData() or 10)
        self.random_layout_button.setText(
            f"Neue {count} Random-Mixed-Layouts" if multi else "Neues Random-Mixed-Layout"
        )
        self.comparison_mode_note.setText(self._comparison_mode_description())

    def _on_comparison_mode_changed(self, _index: int) -> None:
        self._refresh_multi_controls()
        if not hasattr(self, "run_records") or not self.run_records:
            return
        if self._is_multi_mode():
            self._generate_multi_layouts()
        else:
            self._reset_workflow()

    def _set_busy_controls(self, busy: bool) -> None:
        for widget in (
            self.benchmark_combo,
            self.comparison_mode_combo,
            self.run_mode_combo,
            self.multi_count_combo,
            self.gui_profile_combo,
            self.reset_workflow_button,
            self.random_layout_button,
            self.epochs_spin,
            self.lr_spin,
            self.batch_spin,
            self.weight_spin,
            self.seed_spin,
            self.train_button,
            self.train_all_button,
            self.max_steps_spin,
            self.temperature_spin,
            self.evaluate_start_button,
            self.step_once_button,
            self.step_ten_button,
            self.run_sa_button,
            self.run_all_sa_button,
            self.reset_sa_button,
            self.compare_button,
            self.compare_all_button,
            self.load_latest_session_button,
            self.open_session_button,
        ):
            widget.setEnabled(not busy)
        self.layout_editor.setEnabled(not busy and not self.aggregate_selected)
        self.cancel_batch_button.setEnabled(busy and self._is_multi_mode())

    def _on_run_mode_changed(self, _index: int) -> None:
        self._refresh_multi_controls()
        if self._is_multi_mode():
            self._generate_multi_layouts()
        else:
            self._sync_globals_to_active_record()
            self._initialize_single_record()
            self._load_active_record()
            self._refresh_views()

    def _generate_multi_layouts(self) -> None:
        has_results = any(
            record.training_artifacts is not None
            or record.annealing_snapshot is not None
            or record.layout_evaluation is not None
            for record in self.run_records
        )
        if self._is_multi_mode() and has_results and self.isVisible():
            answer = QtWidgets.QMessageBox.question(
                self,
                "Multi-Session ersetzen",
                "Neue Random-Mixed-Layouts verwerfen die aktuellen Multi-Run-Ergebnisse. Fortfahren?",
            )
            if answer != QtWidgets.QMessageBox.Yes:
                return
        count = int(self.multi_count_combo.currentData() or 10)
        self.multi_session = create_gui_multi_run_session(
            benchmark=self.benchmark_combo.currentText(),
            profile_name=self.gui_profile_combo.currentText(),
            master_seed=self.seed_spin.value(),
            hidden_sizes=self.layout_editor.hidden_sizes(),
            config_snapshot=self._config_snapshot(),
            run_count=count,
            comparison_mode=self._comparison_mode(),
        )
        self.run_records = self.multi_session.runs
        self.active_run_index = 0
        self.aggregate_selected = False
        self._rebuild_run_selector()
        self._load_active_record()
        self._autosave_multi_session()
        self._refresh_views()
        self.statusMessage.emit(f"{count} eindeutige Random-Mixed-Layouts erzeugt.")

    def _rebuild_run_selector(self) -> None:
        self.run_selector.blockSignals(True)
        while self.run_selector.count():
            self.run_selector.removeTab(0)
        for record in self.run_records:
            self.run_selector.addTab(f"Run {record.run_index + 1}")
        if len(self.run_records) > 1:
            self.run_selector.addTab("Aggregiert")
        self.run_selector.setCurrentIndex(0)
        self.run_selector.blockSignals(False)
        self._refresh_run_selector_labels()

    def _refresh_run_selector_labels(self) -> None:
        markers = {
            "pending": "○",
            "running": "▶",
            "baseline_done": "B",
            "sa_done": "S",
            "complete": "✓",
            "failed": "!",
        }
        colors = {
            "pending": QtGui.QColor("#64748b"),
            "running": QtGui.QColor("#2563eb"),
            "baseline_done": QtGui.QColor("#7c3aed"),
            "sa_done": QtGui.QColor("#d97706"),
            "complete": QtGui.QColor("#15803d"),
            "failed": QtGui.QColor("#dc2626"),
        }
        for index, record in enumerate(self.run_records):
            self.run_selector.setTabText(
                index,
                f"{markers.get(record.status, '○')} Run {record.run_index + 1}",
            )
            self.run_selector.setTabToolTip(
                index,
                record.error or f"Stage: {record.current_stage}",
            )
            self.run_selector.setTabTextColor(index, colors.get(record.status, colors["pending"]))

    def _on_run_selector_changed(self, index: int) -> None:
        if index < 0 or not self.run_records:
            return
        self._sync_globals_to_active_record()
        if index >= len(self.run_records):
            self.aggregate_selected = True
        else:
            self.aggregate_selected = False
            self.active_run_index = index
            self._load_active_record()
        self._refresh_views()

    def _autosave_multi_session(self) -> None:
        if self.multi_session is None:
            return
        self.multi_session.runs = self.run_records
        self.multi_session.config_snapshot = self._config_snapshot()
        save_gui_multi_run_session(self.multi_session)

    def _load_latest_multi_session(self) -> None:
        path = latest_gui_multi_run_session()
        if path is None:
            self.statusMessage.emit("Keine gespeicherte Multi-Session gefunden.")
            return
        self._load_multi_session(path)

    def _open_multi_session(self) -> None:
        filename, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Multi-Session öffnen",
            str(Path("outputs") / "gui_multi_runs"),
            "GUI Multi Session (session.json);;JSON (*.json)",
        )
        if filename:
            self._load_multi_session(Path(filename))

    def _load_multi_session(self, path: Path) -> None:
        session = load_gui_multi_run_session(path)
        benchmark_index = self.benchmark_combo.findText(session.benchmark)
        profile_index = self.gui_profile_combo.findText(session.profile_name)
        self.benchmark_combo.blockSignals(True)
        self.gui_profile_combo.blockSignals(True)
        if benchmark_index >= 0:
            self.benchmark_combo.setCurrentIndex(benchmark_index)
        if profile_index >= 0:
            self.gui_profile_combo.setCurrentIndex(profile_index)
        self.benchmark_combo.blockSignals(False)
        self.gui_profile_combo.blockSignals(False)
        self.gui_profile = load_gui_benchmark_profile(
            self.gui_profile_combo.currentText(),
            self.benchmark_combo.currentText(),
        )
        self.comparison_mode_combo.blockSignals(True)
        comparison_index = self.comparison_mode_combo.findData(session.comparison_mode)
        if comparison_index >= 0:
            self.comparison_mode_combo.setCurrentIndex(comparison_index)
        self.comparison_mode_combo.blockSignals(False)
        for widget in (
            self.epochs_spin,
            self.lr_spin,
            self.batch_spin,
            self.weight_spin,
            self.max_steps_spin,
            self.temperature_spin,
        ):
            widget.blockSignals(True)
        self.epochs_spin.setValue(int(session.config_snapshot.get("epochs", self.epochs_spin.value())))
        self.lr_spin.setValue(float(session.config_snapshot.get("learning_rate", self.lr_spin.value())))
        self.batch_spin.setValue(int(session.config_snapshot.get("batch_size", self.batch_spin.value())))
        self.weight_spin.setValue(float(session.config_snapshot.get("weight_scale", self.weight_spin.value())))
        self.max_steps_spin.setValue(int(session.config_snapshot.get("max_steps", self.max_steps_spin.value())))
        self.temperature_spin.setValue(
            float(session.config_snapshot.get("start_temperature", self.temperature_spin.value()))
        )
        for widget in (
            self.epochs_spin,
            self.lr_spin,
            self.batch_spin,
            self.weight_spin,
            self.max_steps_spin,
            self.temperature_spin,
        ):
            widget.blockSignals(False)
        self._refresh_online_profile_labels()
        self._setting_random_layout = True
        try:
            self.layout_editor.set_hidden_sizes(session.hidden_sizes)
        finally:
            self._setting_random_layout = False
        self.seed_spin.blockSignals(True)
        self.seed_spin.setValue(session.master_seed)
        self.seed_spin.blockSignals(False)
        self.run_mode_combo.blockSignals(True)
        self.run_mode_combo.setCurrentIndex(self.run_mode_combo.findData("multi"))
        self.run_mode_combo.blockSignals(False)
        self.multi_count_combo.setCurrentText(str(len(session.runs)))
        self.multi_session = session
        self.run_records = session.runs
        self.active_run_index = 0
        self.aggregate_selected = False
        self._refresh_multi_controls()
        self._rebuild_run_selector()
        self._load_active_record()
        self._refresh_views()
        self.statusMessage.emit(f"Multi-Session geladen: {path}")

    def _on_benchmark_changed(self, benchmark: str) -> None:
        self.gui_profile = load_gui_benchmark_profile(self.gui_profile_combo.currentText(), benchmark)
        hidden_sizes = default_hidden_sizes(benchmark)
        self._apply_gui_profile_to_controls()
        self.layout_editor.set_hidden_sizes(hidden_sizes)
        self._load_dataset()
        if self._is_multi_mode():
            self._generate_multi_layouts()
        else:
            self._generate_random_layout()
            self._initialize_single_record()

    def _on_gui_profile_changed(self, profile_name: str) -> None:
        self.gui_profile = load_gui_benchmark_profile(profile_name, self.benchmark_combo.currentText())
        self._apply_gui_profile_to_controls()
        self._on_training_config_changed(reload_dataset=False)
        self.statusMessage.emit(f"GUI-Parameterprofil geladen: {profile_name}")

    def _on_layout_changed(self, _layout_spec: str) -> None:
        if self._setting_random_layout:
            return
        self._active_random_layout_spec = None
        record = self._active_record()
        if record is not None:
            record.start_layout_spec = self.layout_editor.layout_spec()
            record.manually_edited = True
            record.invalidate()
        self.training_artifacts = None
        self.layout_evaluation = None
        self.annealing_session = None
        self.annealing_snapshot = None
        self._rebuild_preview_model()
        self._sync_globals_to_active_record()
        self._autosave_multi_session()
        self._refresh_views()

    def _on_training_config_changed(self, *, reload_dataset: bool) -> None:
        if self._is_multi_mode() and self.run_records:
            for record in self.run_records:
                record.invalidate()
        self.training_artifacts = None
        self.layout_evaluation = None
        self.annealing_session = None
        self.annealing_snapshot = None
        if reload_dataset:
            self._reset_random_layout_rng()
            self._load_dataset()
            if self._is_multi_mode():
                self._generate_multi_layouts()
            else:
                self._generate_random_layout()
                self._initialize_single_record()
            return
        self._rebuild_preview_model()
        self._sync_globals_to_active_record()
        self._refresh_views()

    def _on_sa_config_changed(self) -> None:
        if self._is_multi_mode():
            for record in self.run_records:
                record.annealing_session = None
                record.annealing_snapshot = None
                record.layout_evaluation = None
                record.status = "baseline_done" if record.training_artifacts is not None else "pending"
                record.current_stage = "baseline" if record.training_artifacts is not None else "layout"
        self.layout_evaluation = None
        self.annealing_session = None
        self.annealing_snapshot = None
        self._refresh_views()

    def _reset_workflow(self) -> None:
        if self._is_multi_mode():
            self._generate_multi_layouts()
            self.statusMessage.emit("Multi-Workflow zurückgesetzt.")
            return
        self.training_artifacts = None
        self.layout_evaluation = None
        self.annealing_session = None
        self.annealing_snapshot = None
        self.selected_hidden = (0, 0)
        self._reset_random_layout_rng()
        self._load_dataset()
        self._generate_random_layout()
        self._initialize_single_record()
        self.statusMessage.emit("Workflow zurückgesetzt.")

    def _on_hidden_selected(self, layer_index: int, neuron_index: int) -> None:
        self.selected_hidden = (layer_index, neuron_index)
        self._refresh_views()

    def _run_manual_training(self) -> None:
        if self.aggregate_selected:
            return
        schedule = self._active_schedule()
        request = TrainingRunRequest(
            dataset_config=DatasetConfig(
                name=self.benchmark_combo.currentText(),
                random_state=schedule.data_split_seed,
            ),
            hidden_sizes=self.layout_editor.hidden_sizes(),
            layout_spec=self.layout_editor.layout_spec(),
            training_config=TrainingConfig(
                epochs=self.epochs_spin.value(),
                learning_rate=self.lr_spin.value(),
                batch_size=self.batch_spin.value(),
                random_state=schedule.retraining_batch_seed,
                shuffle=True,
            ),
            weight_scale=self.weight_spin.value(),
            random_state=schedule.retraining_weight_seed,
        )
        self.task_controller.submit(
            run_single_training_experiment,
            request,
            on_success=self._on_training_finished,
            on_error=self._on_task_error,
            status_message="Trainiere Layout...",
        )

    def _training_request_for_record(self, record: GuiRunRecord) -> TrainingRunRequest:
        schedule = record.schedule
        return TrainingRunRequest(
            dataset_config=DatasetConfig(
                name=self.benchmark_combo.currentText(),
                random_state=schedule.data_split_seed,
            ),
            hidden_sizes=self.layout_editor.hidden_sizes(),
            layout_spec=record.start_layout_spec,
            training_config=TrainingConfig(
                epochs=self.epochs_spin.value(),
                learning_rate=self.lr_spin.value(),
                batch_size=self.batch_spin.value(),
                random_state=schedule.retraining_batch_seed,
                shuffle=True,
            ),
            weight_scale=self.weight_spin.value(),
            random_state=schedule.retraining_weight_seed,
        )

    def _online_request_for_record(self, record: GuiRunRecord) -> OnlineAnnealingRequest:
        online_delta = self.gui_profile["online_delta"]
        schedule = record.schedule
        return OnlineAnnealingRequest(
            dataset_config=DatasetConfig(
                name=self.benchmark_combo.currentText(),
                random_state=schedule.data_split_seed,
            ),
            hidden_sizes=self.layout_editor.hidden_sizes(),
            layout_spec=record.start_layout_spec,
            training_config=TrainingConfig(
                epochs=self.epochs_spin.value(),
                learning_rate=float(online_delta["online_learning_rate"]),
                batch_size=int(online_delta["online_batch_size"]),
                random_state=schedule.online_batch_seed,
                shuffle=True,
            ),
            annealing_config=AnnealingConfig(
                start_temperature=self.temperature_spin.value(),
                cooling_schedule=str(
                    online_delta.get("cooling_schedule", DEFAULT_ANNEALING_COOLING_SCHEDULE)
                ),
                cooling_parameter=float(online_delta["cooling_parameter"]),
                iterations_per_temperature=int(online_delta["iterations_per_temperature"]),
                max_steps=self.max_steps_spin.value(),
                min_temperature=float(online_delta["min_temperature"]),
                neighborhood_operations=DEFAULT_ANNEALING_NEIGHBORHOODS,
            ),
            weight_scale=self.weight_spin.value(),
            random_state=schedule.online_weight_seed,
            weight_seed=schedule.online_weight_seed,
            batch_seed=schedule.online_batch_seed,
            proposal_seed=schedule.sa_proposal_seed,
            acceptance_seed=schedule.sa_acceptance_seed,
        )

    def _final_job_for_record(self, record: GuiRunRecord) -> _FinalComparisonJob:
        schedule = record.schedule
        snapshot = record.annealing_snapshot
        end_layout = (
            snapshot.current_evaluation.layout.to_compact_spec()
            if snapshot is not None and snapshot.current_evaluation is not None
            else None
        )
        inherited_candidates: tuple[InheritedModelCandidate, ...] = ()
        if snapshot is not None and snapshot.best_evaluation is not None:
            inherited_candidates = (
                InheritedModelCandidate(
                    "best_online_delta_value",
                    snapshot.best_evaluation.trained_model.to_state_dict(),
                    schedule.data_split_seed,
                ),
            )
        request = LayoutEvaluationRequest(
            dataset_config=DatasetConfig(
                name=self.benchmark_combo.currentText(),
                random_state=schedule.data_split_seed,
            ),
            hidden_sizes=self.layout_editor.hidden_sizes(),
            candidates=build_online_delta_layout_candidates(
                start_layout_spec=record.start_layout_spec,
                end_layout_spec=end_layout,
            ),
            seeds=(schedule.retraining_weight_seed,),
            training_config=TrainingConfig(
                epochs=self.epochs_spin.value(),
                learning_rate=self.lr_spin.value(),
                batch_size=self.batch_spin.value(),
                random_state=schedule.retraining_batch_seed,
                shuffle=True,
            ),
            weight_scale=self.weight_spin.value(),
            primary_metric="validation_loss",
            data_split_seeds=(schedule.data_split_seed,),
            weight_seeds=(schedule.retraining_weight_seed,),
            batch_seeds=(schedule.retraining_batch_seed,),
        )
        return _FinalComparisonJob(request, inherited_candidates)

    def _run_all_baselines(self) -> None:
        records = list(self.run_records)
        self._start_batch_stage(
            "Baselines",
            records,
            run_single_training_experiment,
            [self._training_request_for_record(record) for record in records],
            self._on_batch_baseline_success,
        )

    def _run_all_sa(self) -> None:
        records = list(self.run_records)
        self._start_batch_stage(
            "Online-Delta-SA",
            records,
            _run_online_request_to_completion,
            [self._online_request_for_record(record) for record in records],
            self._on_batch_sa_success,
        )

    def _run_all_final_comparisons(self) -> None:
        records = [
            record
            for record in self.run_records
            if record.annealing_snapshot is not None
            and record.annealing_snapshot.current_evaluation is not None
        ]
        if not records:
            self.statusMessage.emit("Noch keine SA-Endlayouts fuer den finalen Sammelvergleich.")
            return
        self._start_batch_stage(
            "Finalvergleich",
            records,
            _run_final_comparison_job,
            [self._final_job_for_record(record) for record in records],
            self._on_batch_final_success,
        )

    def _start_batch_stage(
        self,
        stage: str,
        records: list[GuiRunRecord],
        worker,
        jobs: list[object],
        success_handler,
    ) -> None:
        if not self._is_multi_mode() or not records:
            return
        if self._batch_stage is not None or self.task_controller.is_busy:
            self.statusMessage.emit("Es laeuft bereits ein Sammellauf.")
            return
        self._batch_stage = stage
        self._batch_records = records
        self._batch_completed = []
        self._batch_failures = 0
        self.multi_progress_bar.setRange(0, len(records))
        self.multi_progress_bar.setValue(0)
        self._update_batch_progress()
        self.task_controller.submit_sequential(
            worker,
            jobs,
            on_item_started=self._on_batch_item_started,
            on_item_success=success_handler,
            on_item_error=self._on_batch_item_error,
            on_finished=self._on_batch_finished,
            status_message=f"Starte sequenziellen Sammellauf: {stage}",
        )

    def _on_batch_item_started(self, index: int) -> None:
        record = self._batch_records[index]
        record.status = "running"
        record.current_stage = str(self._batch_stage or "running")
        record.error = None
        self._update_batch_progress(current_run=record.run_index + 1)
        self._refresh_run_selector_labels()

    def _on_batch_baseline_success(self, index: int, artifacts: TrainingRunArtifacts) -> None:
        record = self._batch_records[index]
        record.dataset = artifacts.dataset
        record.training_artifacts = artifacts
        record.current_model = artifacts.model
        record.status = "baseline_done"
        record.current_stage = "baseline"
        self._finish_batch_item(record)

    def _on_batch_sa_success(
        self,
        index: int,
        result: tuple[OnlineAnnealingSession, OnlineAnnealingSnapshot],
    ) -> None:
        record = self._batch_records[index]
        session, snapshot = result
        record.annealing_session = session
        record.annealing_snapshot = snapshot
        record.dataset = session.dataset
        record.current_model = (
            snapshot.current_evaluation.trained_model
            if snapshot.current_evaluation is not None
            else session.model
        )
        record.status = "sa_done"
        record.current_stage = "sa"
        self._finish_batch_item(record)

    def _on_batch_final_success(self, index: int, result: LayoutEvaluationResult) -> None:
        record = self._batch_records[index]
        record.layout_evaluation = result
        record.status = "complete"
        record.current_stage = "final"
        self._finish_batch_item(record)

    def _on_batch_item_error(self, index: int, traceback_text: str) -> None:
        record = self._batch_records[index]
        record.status = "failed"
        record.error = traceback_text
        self._batch_failures += 1
        self._finish_batch_item(record)

    def _finish_batch_item(self, record: GuiRunRecord) -> None:
        self._batch_completed.append(record.run_index + 1)
        self.multi_progress_bar.setValue(len(self._batch_completed))
        self._autosave_multi_session()
        self._update_batch_progress()
        self._refresh_run_selector_labels()
        if not self.aggregate_selected and self.active_run_index == record.run_index:
            self._load_active_record()
            self._refresh_views()
        elif self.aggregate_selected:
            self._refresh_views()

    def _on_batch_finished(self) -> None:
        self._update_batch_progress()
        self.statusMessage.emit(
            f"{self._batch_stage}: {len(self._batch_completed)} Runs bearbeitet, "
            f"{self._batch_failures} fehlgeschlagen."
        )
        self._batch_stage = None
        self._batch_records = []

    def _update_batch_progress(self, current_run: int | None = None) -> None:
        total = len(self._batch_records)
        current = f" | aktuell Run {current_run}" if current_run is not None else ""
        self.multi_progress_label.setText(
            f"{self._batch_stage or 'Kein Sammellauf'}: "
            f"{len(self._batch_completed)}/{total} bearbeitet, "
            f"Fehler {self._batch_failures}{current}<br>"
            f"Fertige Runs: {self._batch_completed}"
        )

    def _generate_random_layout(self) -> str:
        if self._is_multi_mode():
            self._generate_multi_layouts()
            return self.layout_editor.layout_spec()
        layout_spec = self._next_random_layout_spec(self.layout_editor.hidden_sizes())
        self._setting_random_layout = True
        try:
            self.layout_editor.set_layout_spec(layout_spec)
        finally:
            self._setting_random_layout = False
        self._active_random_layout_spec = self.layout_editor.layout_spec()
        self._on_layout_changed(self._active_random_layout_spec)
        self._active_random_layout_spec = self.layout_editor.layout_spec()
        record = self._active_record()
        if record is not None:
            record.start_layout_spec = self._active_random_layout_spec
            record.manually_edited = False
        self.statusMessage.emit(
            "Random-Mixed-Layout erzeugt."
        )
        return self._active_random_layout_spec

    def _next_random_layout_spec(self, hidden_sizes: tuple[int, ...]) -> str:
        current_layout_spec = (
            self.layout_editor.layout_spec()
            if hasattr(self, "layout_editor")
            else None
        )
        while True:
            layout_spec = random_layout_spec(hidden_sizes, self._random_layout_rng)
            normalized_spec = parse_layout_spec(layout_spec, hidden_sizes).to_compact_spec()
            if normalized_spec != current_layout_spec:
                return normalized_spec

    def _reset_random_layout_rng(self) -> None:
        self._random_layout_rng = np.random.default_rng(self.seed_spin.value())

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
        record = self._active_record()
        if record is not None:
            record.status = "baseline_done"
            record.current_stage = "baseline"
            record.error = None
        self._sync_globals_to_active_record()
        self._autosave_multi_session()
        self.statusMessage.emit("Training abgeschlossen.")
        self._refresh_views()

    def _ensure_annealing_session(self) -> OnlineAnnealingSession:
        if self.annealing_session is None:
            record = self._active_record()
            if record is None:
                raise RuntimeError("Waehle einen konkreten Run fuer SA.")
            self.annealing_session = create_online_session(self._online_request_for_record(record))
            record.annealing_session = self.annealing_session
        return self.annealing_session

    def _evaluate_start(self) -> None:
        session = self._ensure_annealing_session()
        self.task_controller.submit(
            evaluate_online_start,
            session,
            self.preferences.language,
            on_success=self._on_annealing_snapshot,
            on_error=self._on_task_error,
            status_message="Bewerte Startlayout...",
        )

    def _step_once(self) -> None:
        session = self._ensure_annealing_session()
        self.task_controller.submit(
            online_step_once,
            session,
            self.preferences.language,
            on_success=self._on_annealing_snapshot,
            on_error=self._on_task_error,
            status_message="Fuehre SA-Schritt aus...",
        )

    def _step_ten(self) -> None:
        session = self._ensure_annealing_session()
        self.task_controller.submit(
            online_run_steps,
            session,
            10,
            self.preferences.language,
            on_success=self._on_annealing_snapshot,
            on_error=self._on_task_error,
            status_message="Fuehre 10 Online-SA-Schritte aus...",
        )

    def _run_sa_to_completion(self) -> None:
        session = self._ensure_annealing_session()
        self.task_controller.submit(
            online_run_to_completion,
            session,
            self.preferences.language,
            on_success=self._on_annealing_snapshot,
            on_error=self._on_task_error,
            status_message="Fuehre SA bis zum Ende aus...",
        )

    def _reset_sa(self) -> None:
        if self.annealing_session is None:
            return
        self.annealing_snapshot = online_reset(self.annealing_session, self.preferences.language)
        self._sync_globals_to_active_record()
        self._refresh_views()

    def _on_annealing_snapshot(self, snapshot: OnlineAnnealingSnapshot) -> None:
        self.annealing_snapshot = snapshot
        if snapshot.current_evaluation is not None:
            self.current_model = snapshot.current_evaluation.trained_model
        record = self._active_record()
        if record is not None:
            record.status = "sa_done" if snapshot.is_complete else "running"
            record.current_stage = "sa"
            record.error = None
        self._sync_globals_to_active_record()
        self._autosave_multi_session()
        self.statusMessage.emit("Annealing-Zustand aktualisiert.")
        self._refresh_views()

    def _run_final_layout_comparison(self) -> None:
        if self.aggregate_selected:
            return
        schedule = self._active_schedule()
        inherited_candidates: tuple[InheritedModelCandidate, ...] = ()
        end_layout = (
            self.annealing_snapshot.current_evaluation.layout.to_compact_spec()
            if self.annealing_snapshot is not None and self.annealing_snapshot.current_evaluation is not None
            else None
        )
        if (
            self.annealing_snapshot is not None
            and self.annealing_snapshot.best_evaluation is not None
        ):
            inherited_candidates = (
                InheritedModelCandidate(
                    "best_online_delta_value",
                    self.annealing_snapshot.best_evaluation.trained_model.to_state_dict(),
                    schedule.data_split_seed,
                ),
            )
        request = LayoutEvaluationRequest(
            dataset_config=DatasetConfig(
                name=self.benchmark_combo.currentText(),
                random_state=schedule.data_split_seed,
            ),
            hidden_sizes=self.layout_editor.hidden_sizes(),
            candidates=build_online_delta_layout_candidates(
                start_layout_spec=self.layout_editor.layout_spec(),
                end_layout_spec=end_layout,
            ),
            seeds=(schedule.retraining_weight_seed,),
            training_config=TrainingConfig(
                epochs=self.epochs_spin.value(),
                learning_rate=self.lr_spin.value(),
                batch_size=self.batch_spin.value(),
                random_state=schedule.retraining_batch_seed,
                shuffle=True,
            ),
            weight_scale=self.weight_spin.value(),
            primary_metric="validation_loss",
            data_split_seeds=(schedule.data_split_seed,),
            weight_seeds=(schedule.retraining_weight_seed,),
            batch_seeds=(schedule.retraining_batch_seed,),
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
        record = self._active_record()
        if record is not None:
            record.status = "complete"
            record.current_stage = "final"
            record.error = None
        self._sync_globals_to_active_record()
        self._autosave_multi_session()
        self.tabs.setCurrentWidget(self.final_comparison_tab)
        self.statusMessage.emit("Finaler Layout-Vergleich abgeschlossen.")
        self._refresh_views()

    def _on_task_error(self, traceback_text: str) -> None:
        self.statusMessage.emit("Fehler im Workflow.")
        QtWidgets.QMessageBox.critical(self, "Workflow Fehler", traceback_text)

    def retranslate(self) -> None:
        self.sample_panel.set_language(self.preferences.language)
        self.neuron_detail_panel.set_language(self.preferences.language)
        self.stepper_panel.set_language(self.preferences.language)
        self.annealing_history_panel.set_language(self.preferences.language)
        self.online_delta_timeline_panel.set_language(self.preferences.language)
        self._refresh_views()

    def apply_detail_mode(self) -> None:
        pass
