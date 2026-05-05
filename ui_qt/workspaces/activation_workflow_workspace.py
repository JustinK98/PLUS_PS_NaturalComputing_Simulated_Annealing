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
    DatasetConfig,
    GuiExperimentConfig,
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
    LayoutEvaluationRequest,
    LayoutEvaluationResult,
    build_standard_layout_candidates,
    run_layout_evaluation,
)
from services.network_projection_service import build_input_projection
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
from ui_qt.widgets.plot_widgets import TrainingPlotWidget
from ui_qt.widgets.sample_panel import SampleDisplayPayload, SamplePanelWidget
from .base import BaseWorkspace


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
        self.annealing_session: AnnealingSession | None = None
        self.annealing_snapshot: AnnealingSessionSnapshot | None = None
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
        self.lr_spin = QtWidgets.QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 10.0)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setSingleStep(0.003)
        self.lr_spin.setValue(self.config.learning_rate)
        self.batch_spin = QtWidgets.QSpinBox()
        self.batch_spin.setRange(1, 4096)
        self.batch_spin.setValue(self.config.batch_size)
        self.weight_spin = QtWidgets.QDoubleSpinBox()
        self.weight_spin.setRange(0.0001, 10.0)
        self.weight_spin.setDecimals(4)
        self.weight_spin.setSingleStep(0.01)
        self.weight_spin.setValue(self.config.weight_scale)
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 1_000_000)
        self.seed_spin.setValue(self.config.random_state)
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
        self.candidate_epochs_spin = QtWidgets.QSpinBox()
        self.candidate_epochs_spin.setRange(1, 1000)
        self.candidate_epochs_spin.setValue(DEFAULT_ANNEALING_CANDIDATE_EPOCHS)
        self.max_steps_spin = QtWidgets.QSpinBox()
        self.max_steps_spin.setRange(1, 10000)
        self.max_steps_spin.setValue(DEFAULT_ANNEALING_MAX_STEPS)
        self.temperature_spin = QtWidgets.QDoubleSpinBox()
        self.temperature_spin.setRange(0.001, 100.0)
        self.temperature_spin.setDecimals(3)
        self.temperature_spin.setValue(DEFAULT_ANNEALING_START_TEMPERATURE)
        sa_layout.addRow("Candidate Epochs", self.candidate_epochs_spin)
        sa_layout.addRow("Max Steps", self.max_steps_spin)
        sa_layout.addRow("Start Temperature", self.temperature_spin)
        sa_buttons = QtWidgets.QHBoxLayout()
        self.evaluate_start_button = QtWidgets.QPushButton("Start bewerten")
        self.step_once_button = QtWidgets.QPushButton("1 Schritt")
        self.run_sa_button = QtWidgets.QPushButton("SA bis Ende")
        self.reset_sa_button = QtWidgets.QPushButton("Reset")
        self.evaluate_start_button.clicked.connect(self._evaluate_start)
        self.step_once_button.clicked.connect(self._step_once)
        self.run_sa_button.clicked.connect(self._run_sa_to_completion)
        self.reset_sa_button.clicked.connect(self._reset_sa)
        sa_buttons.addWidget(self.evaluate_start_button)
        sa_buttons.addWidget(self.step_once_button)
        sa_buttons.addWidget(self.run_sa_button)
        sa_buttons.addWidget(self.reset_sa_button)
        sa_layout.addRow(sa_buttons)
        left_layout.addWidget(self.sa_group)

        self.compare_group = QtWidgets.QGroupBox("5. Finale Layouts vergleichen")
        compare_layout = QtWidgets.QVBoxLayout(self.compare_group)
        self.compare_button = QtWidgets.QPushButton("Start / Best / Random / Baselines final trainieren")
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
        self.annealing_live_panel = AnnealingLivePanel(self.preferences.language)
        self.annealing_history_panel = AnnealingHistoryPanel(self.preferences.language)
        self.annealing_decision_panel = AnnealingDecisionPanel(self.preferences.language)
        self.final_table = QtWidgets.QTableWidget(0, 7)
        self.final_table.setHorizontalHeaderLabels(
            ["Rank", "Layout", "Val Loss", "Val Acc", "Test Loss", "Test Acc", "Seeds"]
        )
        self.final_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.help_text = QtWidgets.QTextBrowser()
        self.help_text.setHtml(self._help_html())
        self.tabs.addTab(self.network_view, "Netz")
        self.tabs.addTab(self.sample_panel, "Sample")
        self.tabs.addTab(self.training_plot, "Training")
        self.tabs.addTab(self.annealing_live_panel, "SA Live")
        self.tabs.addTab(self.annealing_history_panel, "SA History")
        self.tabs.addTab(self.annealing_decision_panel, "SA Entscheidung")
        self.tabs.addTab(self.final_table, "Finaler Vergleich")
        self.tabs.addTab(self.help_text, "Ablauf")
        right_layout.addWidget(self.tabs, 1)

    def _load_dataset(self) -> None:
        benchmark = self.benchmark_combo.currentText()
        self.dataset = load_benchmark(DatasetConfig(name=benchmark, random_state=self.seed_spin.value()))
        self.analysis_sample = build_analysis_sample(
            self.dataset,
            "val",
            0,
            language=self.preferences.language,
        )
        self.annealing_session = None
        self.annealing_snapshot = None
        self.layout_evaluation = None

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
        history = self.training_artifacts.training_result.history if self.training_artifacts is not None else None
        self.training_plot.set_history(history)
        self._refresh_cards()
        self._refresh_annealing_panels()
        self._refresh_final_table()

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
        self.final_table.setRowCount(len(self.layout_evaluation.ranking))
        for row_index, item in enumerate(self.layout_evaluation.ranking):
            values = (
                row_index + 1,
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

    def _ensure_annealing_session(self) -> AnnealingSession:
        if self.annealing_session is None:
            request = AnnealingRunRequest(
                dataset_config=DatasetConfig(name=self.benchmark_combo.currentText(), random_state=self.seed_spin.value()),
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
                annealing_config=AnnealingConfig(
                    start_temperature=self.temperature_spin.value(),
                    cooling_schedule=DEFAULT_ANNEALING_COOLING_SCHEDULE,
                    cooling_parameter=DEFAULT_ANNEALING_COOLING_PARAMETER,
                    iterations_per_temperature=DEFAULT_ANNEALING_ITERATIONS_PER_TEMPERATURE,
                    max_steps=self.max_steps_spin.value(),
                    min_temperature=DEFAULT_ANNEALING_MIN_TEMPERATURE,
                    neighborhood_operations=DEFAULT_ANNEALING_NEIGHBORHOODS,
                ),
                random_state=self.seed_spin.value(),
            )
            self.annealing_session = create_session(request)
        return self.annealing_session

    def _evaluate_start(self) -> None:
        self.task_controller.submit(
            evaluate_start,
            self._ensure_annealing_session(),
            self.preferences.language,
            on_success=self._on_annealing_snapshot,
            on_error=self._on_task_error,
            status_message="Bewerte Startlayout...",
        )

    def _step_once(self) -> None:
        self.task_controller.submit(
            step_once,
            self._ensure_annealing_session(),
            self.preferences.language,
            on_success=self._on_annealing_snapshot,
            on_error=self._on_task_error,
            status_message="Fuehre SA-Schritt aus...",
        )

    def _run_sa_to_completion(self) -> None:
        self.task_controller.submit(
            run_to_completion,
            self._ensure_annealing_session(),
            self.preferences.language,
            on_success=self._on_annealing_snapshot,
            on_error=self._on_task_error,
            status_message="Fuehre SA bis zum Ende aus...",
        )

    def _reset_sa(self) -> None:
        if self.annealing_session is None:
            return
        self.annealing_snapshot = reset(self.annealing_session, self.preferences.language)
        self._refresh_views()

    def _on_annealing_snapshot(self, snapshot: AnnealingSessionSnapshot) -> None:
        self.annealing_snapshot = snapshot
        if snapshot.best_evaluation is not None:
            self.current_model = snapshot.best_evaluation.trained_model
        self.statusMessage.emit("Annealing-Zustand aktualisiert.")
        self._refresh_views()

    def _run_final_layout_comparison(self) -> None:
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
            run_layout_evaluation,
            request,
            on_success=self._on_layout_evaluation_finished,
            on_error=self._on_task_error,
            status_message="Trainiere finale Layout-Vergleiche...",
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
