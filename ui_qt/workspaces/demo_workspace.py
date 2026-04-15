"""Qt-Demo-Workspace fuer didaktische Einzelruns."""

from __future__ import annotations

import copy

from PySide6 import QtCore, QtWidgets

from benchmarks import DatasetBundle, load_benchmark
from configs import DatasetConfig, GuiExperimentConfig, TrainingConfig, default_hidden_sizes
from model import ModularMLP
from services.analysis_sample_service import (
    AnalysisSample,
    build_analysis_sample,
    split_arrays,
)
from services.compare_service import build_compare_payload
from services.help_service import workspace_help_html
from services.network_projection_service import build_input_projection
from services.neuron_analysis_service import build_neuron_analysis_payload
from services.stepper_service import build_step_entries
from services.training_session_service import (
    TrainingSessionSnapshot,
    create_training_session,
    model_from_training_session,
    run_training_session_epochs,
    training_result_from_session,
)
from services.training_service import TrainingRunRequest, run_single_training_experiment
from ui_qt.state import WorkspacePreferences
from ui_qt.tasking import BackgroundTaskController
from ui_qt.texts import RECIPES, text
from ui_qt.utils import parse_hidden_sizes
from ui_qt.widgets.compare_panel import ComparePanel
from ui_qt.widgets.info_card import InfoCardWidget
from ui_qt.widgets.layout_editor import LayoutEditorWidget
from ui_qt.widgets.network_view import NetworkViewWidget
from ui_qt.widgets.neuron_detail_panel import NeuronDetailPanel
from ui_qt.widgets.plot_widgets import TrainingPlotWidget
from ui_qt.widgets.recipe_panel import RecipePanelWidget
from ui_qt.widgets.sample_panel import SampleDisplayPayload, SamplePanelWidget
from ui_qt.widgets.stepper_panel import StepperPanel
from .base import BaseWorkspace


class DemoWorkspace(BaseWorkspace):
    workspace_id = "demo"

    def __init__(self, config: GuiExperimentConfig, preferences: WorkspacePreferences, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(preferences, parent)
        self.config = config
        self.dataset: DatasetBundle | None = None
        self.preview_model: ModularMLP | None = None
        self.trained_model: ModularMLP | None = None
        self.training_result = None
        self.training_session: TrainingSessionSnapshot | None = None
        self.baseline_model: ModularMLP | None = None
        self.baseline_training_result = None
        self.baseline_completed_epochs = 0
        self.baseline_benchmark_name: str | None = None
        self.selected_hidden: tuple[int, int] | None = None
        self.split_name = "train"
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
        left_container = QtWidgets.QWidget()
        left_scroll.setWidget(left_container)
        self.splitter.addWidget(left_scroll)

        left_layout = QtWidgets.QVBoxLayout(left_container)
        left_layout.setContentsMargins(6, 6, 6, 6)

        self.problem_group = QtWidgets.QGroupBox()
        problem_layout = QtWidgets.QFormLayout(self.problem_group)
        self.benchmark_combo = QtWidgets.QComboBox()
        self.benchmark_combo.addItems(("breast_cancer", "wine", "digits", "test_activation"))
        self.benchmark_combo.setCurrentText(self.config.benchmark)
        self.benchmark_combo.currentTextChanged.connect(self._on_benchmark_changed)
        self.hidden_sizes_edit = QtWidgets.QLineEdit(", ".join(str(v) for v in self.config.hidden_sizes))
        self.hidden_sizes_edit.editingFinished.connect(self._on_hidden_sizes_changed)
        problem_layout.addRow(self.make_help_label("Benchmark", "benchmark"), self.benchmark_combo)
        problem_layout.addRow(self.make_help_label("Hidden Sizes", "hidden_sizes"), self.hidden_sizes_edit)
        left_layout.addWidget(self.problem_group)

        self.layout_group = QtWidgets.QGroupBox()
        layout_group_layout = QtWidgets.QVBoxLayout(self.layout_group)
        layout_group_layout.addWidget(self.make_help_strip("layout"))
        self.layout_editor = LayoutEditorWidget(self.config.hidden_sizes, self.config.layout_spec)
        self.layout_editor.layoutSpecChanged.connect(self._on_layout_changed)
        layout_group_layout.addWidget(self.layout_editor)
        left_layout.addWidget(self.layout_group)

        self.sample_group = QtWidgets.QGroupBox()
        sample_layout = QtWidgets.QFormLayout(self.sample_group)
        self.split_combo = QtWidgets.QComboBox()
        self.split_combo.addItems(("train", "val", "test"))
        self.split_combo.currentTextChanged.connect(self._on_split_changed)
        self.sample_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sample_slider.valueChanged.connect(self._on_sample_index_changed)
        self.sample_spin = QtWidgets.QSpinBox()
        self.sample_spin.valueChanged.connect(self._on_sample_index_changed)
        sample_layout.addRow(self.make_help_label("Split", "sample_selection"), self.split_combo)
        sample_row = QtWidgets.QHBoxLayout()
        sample_row.addWidget(self.sample_slider)
        sample_row.addWidget(self.sample_spin)
        sample_row_widget = QtWidgets.QWidget()
        sample_row_widget.setLayout(sample_row)
        sample_layout.addRow(self.make_help_label("Sample", "sample_selection"), sample_row_widget)
        left_layout.addWidget(self.sample_group)

        self.training_group = QtWidgets.QGroupBox()
        training_layout = QtWidgets.QFormLayout(self.training_group)
        self.epochs_spin = QtWidgets.QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(self.config.epochs)
        self.lr_spin = QtWidgets.QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 10.0)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setSingleStep(0.01)
        self.lr_spin.setValue(self.config.learning_rate)
        self.batch_spin = QtWidgets.QSpinBox()
        self.batch_spin.setRange(1, 4096)
        self.batch_spin.setValue(self.config.batch_size)
        self.weight_spin = QtWidgets.QDoubleSpinBox()
        self.weight_spin.setRange(0.0001, 10.0)
        self.weight_spin.setDecimals(4)
        self.weight_spin.setSingleStep(0.01)
        self.weight_spin.setValue(self.config.weight_scale)
        training_layout.addRow(self.make_help_label("Epochs", "training_hyperparameters"), self.epochs_spin)
        training_layout.addRow(self.make_help_label("Learning Rate", "training_hyperparameters"), self.lr_spin)
        training_layout.addRow(self.make_help_label("Batch Size", "training_hyperparameters"), self.batch_spin)
        training_layout.addRow(self.make_help_label("Weight Scale", "training_hyperparameters"), self.weight_spin)
        self.train_one_button = QtWidgets.QPushButton()
        self.train_one_button.clicked.connect(lambda: self._train_for_epochs(1))
        self.train_ten_button = QtWidgets.QPushButton()
        self.train_ten_button.clicked.connect(lambda: self._train_for_epochs(10))
        self.run_button = QtWidgets.QPushButton()
        self.run_button.setProperty("role", "primary")
        self.run_button.clicked.connect(lambda: self._train_for_epochs(self.epochs_spin.value()))
        self.reset_training_button = QtWidgets.QPushButton()
        self.reset_training_button.clicked.connect(self._reset_training_state)
        training_button_row = QtWidgets.QHBoxLayout()
        training_button_row.addWidget(self.train_one_button)
        training_button_row.addWidget(self.train_ten_button)
        training_button_row.addWidget(self.run_button)
        training_layout.addRow(training_button_row)
        training_layout.addRow(self.reset_training_button)
        left_layout.addWidget(self.training_group)
        left_layout.addStretch(1)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(1, 1)

        cards_row = QtWidgets.QHBoxLayout()
        self.dataset_card = InfoCardWidget("Benchmark")
        self.layout_card = InfoCardWidget("Layout")
        self.metrics_card = InfoCardWidget("Validation")
        cards_row.addWidget(self.dataset_card)
        cards_row.addWidget(self.layout_card)
        cards_row.addWidget(self.metrics_card)
        right_layout.addLayout(cards_row)

        self.tabs = QtWidgets.QTabWidget()
        self.sample_panel = SamplePanelWidget(self.preferences.language)
        self.training_plot = TrainingPlotWidget()
        self.stepper_panel = StepperPanel(self.preferences.language)
        self.neuron_detail_panel = NeuronDetailPanel(self.preferences.language)
        self.compare_panel = ComparePanel(self.preferences.language)
        self.compare_panel.storeBaselineRequested.connect(self._store_baseline)
        self.recipe_panel = RecipePanelWidget(self.preferences.language)
        self.help_text = QtWidgets.QTextBrowser()
        self.tabs.addTab(self.sample_panel, "")
        self.tabs.addTab(self.training_plot, "")
        self.tabs.addTab(self.stepper_panel, "")
        self.tabs.addTab(self.neuron_detail_panel, "")
        self.tabs.addTab(self.compare_panel, "")
        self.tabs.addTab(self.recipe_panel, "")
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
        self.activation_info_button = self.make_info_button("activation_curve")
        network_toolbar.addWidget(self.fit_view_button)
        network_toolbar.addWidget(self.zoom_out_button)
        network_toolbar.addWidget(self.zoom_in_button)
        network_toolbar.addWidget(self.activation_info_button)
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
        self.problem_group.setTitle(text(self.preferences.language, "demo.problem"))
        self.layout_group.setTitle(text(self.preferences.language, "demo.layout"))
        self.sample_group.setTitle(text(self.preferences.language, "demo.sample"))
        self.training_group.setTitle(text(self.preferences.language, "demo.training"))
        self.train_one_button.setText("Train 1")
        self.train_ten_button.setText("Train 10")
        self.run_button.setText(
            "Train N" if self.preferences.language == "en" else "Trainiere N"
        )
        self.reset_training_button.setText(
            "Reset model" if self.preferences.language == "en" else "Modell resetten"
        )
        self.fit_view_button.setText("Fit")
        self.tabs.setTabText(0, text(self.preferences.language, "common.sample"))
        self.tabs.setTabText(1, text(self.preferences.language, "common.training"))
        self.tabs.setTabText(2, "Stepper")
        self.tabs.setTabText(3, text(self.preferences.language, "common.neuron_detail"))
        self.tabs.setTabText(4, text(self.preferences.language, "demo.compare"))
        self.tabs.setTabText(5, text(self.preferences.language, "common.recipes"))
        self.tabs.setTabText(6, text(self.preferences.language, "common.help"))
        self.sample_panel.set_language(self.preferences.language)
        self.stepper_panel.set_language(self.preferences.language)
        self.neuron_detail_panel.set_language(self.preferences.language)
        self.compare_panel.set_language(self.preferences.language)
        self.recipe_panel.apply_language(self.preferences.language)
        self._refresh_help_tab()
        self._update_tab_help_topic()

    def apply_detail_mode(self) -> None:
        expert = self.preferences.detail_mode == "expert"
        self.tabs.setTabVisible(2, expert)
        self.tabs.setTabVisible(4, expert)

    def _on_benchmark_changed(self) -> None:
        benchmark = self.benchmark_combo.currentText()
        defaults = default_hidden_sizes(benchmark)
        self.hidden_sizes_edit.setText(", ".join(str(value) for value in defaults))
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
        self._refresh_sample_views()

    def _on_sample_index_changed(self, value: int) -> None:
        self.sample_index = int(value)
        self.sample_slider.blockSignals(True)
        self.sample_spin.blockSignals(True)
        self.sample_slider.setValue(self.sample_index)
        self.sample_spin.setValue(self.sample_index)
        self.sample_slider.blockSignals(False)
        self.sample_spin.blockSignals(False)
        self._refresh_sample_views()

    def _on_hidden_selected(self, layer_index: int, neuron_index: int) -> None:
        self.selected_hidden = (layer_index, neuron_index)
        self._refresh_neuron_detail()

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
        self.sample_index = min(self.sample_index, maximum)
        self.sample_slider.setValue(self.sample_index)
        self.sample_spin.setValue(self.sample_index)

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
        hidden_sizes = self.layout_editor.hidden_sizes()
        layout = self.layout_editor.current_layout()
        self.preview_model = ModularMLP(
            input_size=self.dataset.input_size,
            hidden_sizes=hidden_sizes,
            output_size=self.dataset.output_size,
            layout=layout,
            weight_scale=self.weight_spin.value(),
            random_state=self.config.random_state,
        )
        self.training_session = create_training_session(self.preview_model)
        self.trained_model = None
        self.training_result = None
        if self.trained_model is None:
            self.network_view.set_model(self.preview_model)

    def _current_model(self) -> ModularMLP | None:
        return self.trained_model or self.preview_model

    def _train_for_epochs(self, epochs: int) -> None:
        if self.dataset is None or self.preview_model is None:
            return
        if self.training_session is None:
            self.training_session = create_training_session(self.preview_model)
        training_config = TrainingConfig(
            epochs=int(epochs),
            learning_rate=self.lr_spin.value(),
            batch_size=self.batch_spin.value(),
            random_state=self.config.random_state,
            shuffle=True,
        )
        self.task_controller.submit(
            run_training_session_epochs,
            self.training_session,
            self.dataset,
            training_config,
            on_success=self._on_training_session_updated,
            on_error=self._on_task_error,
            status_message=f"Training for {epochs} epochs...",
        )

    def _run_training(self) -> None:
        hidden_sizes = self.layout_editor.hidden_sizes()
        request = TrainingRunRequest(
            dataset_config=DatasetConfig(name=self.benchmark_combo.currentText(), random_state=self.config.random_state),
            hidden_sizes=hidden_sizes,
            layout_spec=self.layout_editor.layout_spec(),
            training_config=TrainingConfig(
                epochs=self.epochs_spin.value(),
                learning_rate=self.lr_spin.value(),
                batch_size=self.batch_spin.value(),
                random_state=self.config.random_state,
                shuffle=True,
            ),
            weight_scale=self.weight_spin.value(),
            random_state=self.config.random_state,
        )
        self.task_controller.submit(
            run_single_training_experiment,
            request,
            on_success=self._on_training_finished,
            on_error=self._on_task_error,
            status_message="Running demo training...",
        )

    def _on_training_finished(self, artifacts) -> None:
        self.dataset = artifacts.dataset
        self.preview_model = artifacts.model
        self.trained_model = artifacts.model
        self.training_result = artifacts.training_result
        self.training_session = TrainingSessionSnapshot(
            model_state=artifacts.model.to_state_dict(),
            history={key: list(values) for key, values in artifacts.training_result.history.items()},
            completed_epochs=len(artifacts.training_result.history.get("train_loss", [])),
            test_metrics=dict(artifacts.training_result.test_metrics),
        )
        self.network_view.set_model(self.trained_model)
        self.training_plot.set_history(self.training_result.history)
        self.statusMessage.emit("Demo training finished.")
        self._refresh_all_views()

    def _on_training_session_updated(self, session: TrainingSessionSnapshot) -> None:
        self.training_session = session
        self.trained_model = model_from_training_session(session)
        self.training_result = training_result_from_session(session)
        self.network_view.set_model(self.trained_model)
        self.training_plot.set_history(self.training_result.history)
        self.statusMessage.emit(f"Training updated: {session.completed_epochs} epochs.")
        self._refresh_all_views()

    def _reset_training_state(self) -> None:
        if self.preview_model is None:
            return
        self.trained_model = None
        self.training_result = None
        self.training_session = create_training_session(self.preview_model)
        self.training_plot.set_history(None)
        self.network_view.set_model(self.preview_model)
        self._refresh_all_views()

    def _store_baseline(self) -> None:
        model = self._current_model()
        if model is None or self.dataset is None:
            return
        self.baseline_model = model.clone()
        self.baseline_training_result = copy.deepcopy(self.training_result)
        self.baseline_completed_epochs = (
            self.training_session.completed_epochs if self.training_session is not None else 0
        )
        self.baseline_benchmark_name = self.dataset.name
        self.statusMessage.emit(
            "Baseline stored." if self.preferences.language == "en" else "Baseline gespeichert."
        )
        self._refresh_compare_view()

    def _on_task_error(self, traceback_text: str) -> None:
        QtWidgets.QMessageBox.critical(self, "Task failed", traceback_text)
        self.statusMessage.emit("Task failed.")

    def _refresh_all_views(self) -> None:
        if self.dataset is None:
            return
        self.dataset_card.set_content(value=self.dataset.name, body=f"{self.dataset.input_size} -> {self.dataset.output_size}")
        self.layout_card.set_content(value=self.layout_editor.layout_spec(), body=", ".join(str(v) for v in self.layout_editor.hidden_sizes()))
        if self.training_result is None:
            self.metrics_card.set_content(value="-", body="No training run yet.")
        else:
            self.metrics_card.set_content(
                value=f"{self.training_result.history['val_acc'][-1]:.3f}",
                body=(
                    f"val_loss={self.training_result.history['val_loss'][-1]:.3f} | "
                    f"epochs={self.training_session.completed_epochs if self.training_session else len(self.training_result.history['val_loss'])}"
                ),
            )
        model = self._current_model()
        analysis_sample = self._analysis_sample()
        if model is not None:
            self.network_view.set_input_projection(
                build_input_projection(self.dataset, model, analysis_sample, self.selected_hidden)
            )
        self.network_view.set_model(model)
        self.network_view.set_selected_hidden(self.selected_hidden)
        self._refresh_sample_views()
        self._refresh_compare_view()

    def _refresh_sample_views(self) -> None:
        if self.dataset is None:
            return
        X, _raw, _y = split_arrays(self.dataset, self.split_name)
        model = self._current_model()
        if len(X) == 0 or model is None:
            return
        analysis_sample = self._analysis_sample()
        probabilities = model.predict_proba(analysis_sample.scaled_sample.reshape(1, -1))[0]
        prediction_index = int(model.predict(analysis_sample.scaled_sample.reshape(1, -1))[0])
        self.sample_panel.set_sample(
            SampleDisplayPayload(
                dataset=self.dataset,
                analysis_sample=analysis_sample,
                prediction_name=self.dataset.target_names[prediction_index],
                probabilities=probabilities,
            )
        )

        trace = model.trace_sample(
            analysis_sample.scaled_sample.reshape(1, -1),
            target_index=analysis_sample.effective_target_index,
        )
        loss_value = trace.loss if trace.loss is not None else 0.0
        self.stepper_panel.set_entries(
            build_step_entries(trace, self.dataset, analysis_sample, self.preferences.language)
        )
        self._refresh_neuron_detail()

    def _refresh_neuron_detail(self) -> None:
        if self.dataset is None or self.selected_hidden is None:
            self.neuron_detail_panel.set_placeholder(self.preferences.language)
            return
        model = self._current_model()
        if model is None:
            return
        analysis_sample = self._analysis_sample()
        layer_index, neuron_index = self.selected_hidden
        payload = build_neuron_analysis_payload(
            model,
            self.dataset,
            analysis_sample,
            layer_index,
            neuron_index,
            self.preferences.language,
        )
        self.neuron_detail_panel.set_payload(payload, self.preferences.language)

    def _refresh_compare_view(self) -> None:
        if self.dataset is None or self.baseline_model is None:
            self.compare_panel.set_placeholder(self.preferences.language)
            return
        current_model = self._current_model()
        if current_model is None:
            self.compare_panel.set_placeholder(self.preferences.language)
            return
        payload = build_compare_payload(
            current_model,
            self.baseline_model,
            self.dataset,
            self.preferences.language,
            analysis_sample=self._analysis_sample(),
            baseline_training_result=self.baseline_training_result,
            current_training_result=self.training_result,
            baseline_completed_epochs=self.baseline_completed_epochs,
            current_completed_epochs=(
                self.training_session.completed_epochs if self.training_session is not None else 0
            ),
            baseline_benchmark_name=self.baseline_benchmark_name,
        )
        self.compare_panel.set_payload(payload, self.preferences.language)

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
            1: "training_plot",
            2: "stepper",
            3: "neuron_tracker",
            4: "compare",
            5: "recipes",
            6: "workspace_help",
        }
        self.tab_help_button.set_topic_key(topic_by_index.get(self.tabs.currentIndex(), "workspace_help"))
