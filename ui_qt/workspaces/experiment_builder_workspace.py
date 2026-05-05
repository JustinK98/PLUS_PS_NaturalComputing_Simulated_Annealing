"""Qt-Workspace fuer reproduzierbare Experimente und Builder-Analyse."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtWidgets

from activations import parse_layout_spec
from benchmarks import load_benchmark
from configs import DatasetConfig, GuiExperimentConfig, OUTPUT_DIR, SUPPORTED_BENCHMARKS, default_hidden_sizes
from experiment_builder import ExperimentDefinition
from model import ModularMLP
from search_spaces import SearchSpaceDefinition, SearchValueDefinition
from services.help_service import workspace_help_html
from services.experiment_service import (
    default_experiment_definition,
    load_saved_experiment,
    run_experiment_definition,
    save_experiment_definition,
)
from services.preview_service import format_experiment_summary, format_run_summary
from ui_qt.models.run_table_model import RunTableModel
from ui_qt.state import WorkspacePreferences
from ui_qt.tasking import BackgroundTaskController
from ui_qt.texts import text
from ui_qt.utils import normalize_output_path, parse_hidden_sizes
from ui_qt.widgets.info_card import InfoCardWidget
from ui_qt.widgets.layout_editor import LayoutEditorWidget
from ui_qt.widgets.network_view import NetworkViewWidget
from ui_qt.widgets.plot_widgets import BuilderSummaryPlotWidget
from ui_qt.widgets.recipe_panel import RecipePanelWidget
from .base import BaseWorkspace


class ExperimentBuilderWorkspace(BaseWorkspace):
    workspace_id = "experiment_builder"

    def __init__(self, config: GuiExperimentConfig, preferences: WorkspacePreferences, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(preferences, parent)
        self.config = config
        self.task_controller = BackgroundTaskController(self)
        self.task_controller.busyChanged.connect(self.busyChanged)
        self.task_controller.messageEmitted.connect(self.statusMessage)
        self.payload: dict[str, Any] | None = None
        self.selected_run_payload: dict[str, Any] | None = None
        self._build_ui()
        self.retranslate()
        self.apply_detail_mode()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(self.splitter)

        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left = QtWidgets.QWidget()
        left_scroll.setWidget(left)
        left_layout = QtWidgets.QVBoxLayout(left)
        self.splitter.addWidget(left_scroll)

        definition = default_experiment_definition()
        definition = replace(
            definition,
            benchmark=self.config.benchmark,
            hidden_sizes=self.config.hidden_sizes,
            layout_spec=self.config.layout_spec,
        )

        self.setup_group = QtWidgets.QGroupBox()
        setup_layout = QtWidgets.QFormLayout(self.setup_group)
        self.experiment_id_edit = QtWidgets.QLineEdit("qt_experiment")
        self.benchmark_combo = QtWidgets.QComboBox()
        self.benchmark_combo.addItems(SUPPORTED_BENCHMARKS)
        self.benchmark_combo.setCurrentText(definition.benchmark)
        self.benchmark_combo.currentTextChanged.connect(self._on_benchmark_changed)
        self.hidden_sizes_edit = QtWidgets.QLineEdit(", ".join(str(v) for v in definition.hidden_sizes))
        self.hidden_sizes_edit.editingFinished.connect(self._on_hidden_sizes_changed)
        self.run_mode_combo = QtWidgets.QComboBox()
        self.run_mode_combo.addItems(("manual_training", "simulated_annealing"))
        self.run_mode_combo.currentTextChanged.connect(self._update_run_mode_visibility)
        self.primary_metric_combo = QtWidgets.QComboBox()
        self.primary_metric_combo.addItems(("validation_accuracy", "validation_loss"))
        setup_layout.addRow(self.make_help_label("Experiment ID", "experiment_setup"), self.experiment_id_edit)
        setup_layout.addRow(self.make_help_label("Benchmark", "benchmark"), self.benchmark_combo)
        setup_layout.addRow(self.make_help_label("Hidden Sizes", "hidden_sizes"), self.hidden_sizes_edit)
        setup_layout.addRow(self.make_help_label("Run Mode", "experiment_setup"), self.run_mode_combo)
        setup_layout.addRow(self.make_help_label("Primary Metric", "experiment_setup"), self.primary_metric_combo)
        left_layout.addWidget(self.setup_group)

        self.layout_group = QtWidgets.QGroupBox()
        layout_box = QtWidgets.QVBoxLayout(self.layout_group)
        layout_box.addWidget(self.make_help_strip("layout"))
        self.layout_editor = LayoutEditorWidget(definition.hidden_sizes, definition.layout_spec)
        layout_box.addWidget(self.layout_editor)
        left_layout.addWidget(self.layout_group)

        self.seeds_group = QtWidgets.QGroupBox()
        seeds_layout = QtWidgets.QFormLayout(self.seeds_group)
        self.seeds_edit = QtWidgets.QLineEdit("42, 43, 44")
        seeds_layout.addRow(self.make_help_label("Seeds", "seeds"), self.seeds_edit)
        left_layout.addWidget(self.seeds_group)

        self.training_group = QtWidgets.QGroupBox()
        training_layout = QtWidgets.QFormLayout(self.training_group)
        self.lr_spin = QtWidgets.QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 10.0)
        self.lr_spin.setValue(definition.learning_rate)
        self.lr_spin.setDecimals(4)
        self.batch_spin = QtWidgets.QSpinBox()
        self.batch_spin.setRange(1, 4096)
        self.batch_spin.setValue(definition.batch_size)
        self.weight_spin = QtWidgets.QDoubleSpinBox()
        self.weight_spin.setRange(0.0001, 10.0)
        self.weight_spin.setValue(definition.weight_scale)
        self.weight_spin.setDecimals(4)
        self.epochs_spin = QtWidgets.QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(definition.epochs)
        training_layout.addRow(self.make_help_label("Learning Rate", "training_hyperparameters"), self.lr_spin)
        training_layout.addRow(self.make_help_label("Batch Size", "training_hyperparameters"), self.batch_spin)
        training_layout.addRow(self.make_help_label("Weight Scale", "training_hyperparameters"), self.weight_spin)
        training_layout.addRow(self.make_help_label("Epochs", "training_hyperparameters"), self.epochs_spin)
        left_layout.addWidget(self.training_group)

        self.sa_group = QtWidgets.QGroupBox()
        sa_layout = QtWidgets.QFormLayout(self.sa_group)
        self.objective_combo = QtWidgets.QComboBox()
        self.objective_combo.addItems(("validation_loss", "validation_accuracy"))
        self.candidate_epochs_spin = QtWidgets.QSpinBox()
        self.candidate_epochs_spin.setRange(1, 500)
        self.candidate_epochs_spin.setValue(definition.candidate_epochs)
        self.start_temp_spin = QtWidgets.QDoubleSpinBox()
        self.start_temp_spin.setRange(0.001, 100.0)
        self.start_temp_spin.setValue(definition.start_temperature)
        self.cooling_schedule_combo = QtWidgets.QComboBox()
        self.cooling_schedule_combo.addItems(("geometric", "linear", "logarithmic"))
        self.cooling_param_spin = QtWidgets.QDoubleSpinBox()
        self.cooling_param_spin.setRange(0.001, 10.0)
        self.cooling_param_spin.setValue(definition.cooling_parameter)
        self.iter_temp_spin = QtWidgets.QSpinBox()
        self.iter_temp_spin.setRange(1, 1000)
        self.iter_temp_spin.setValue(definition.iterations_per_temperature)
        self.max_steps_spin = QtWidgets.QSpinBox()
        self.max_steps_spin.setRange(1, 100000)
        self.max_steps_spin.setValue(definition.max_steps)
        self.min_temp_spin = QtWidgets.QDoubleSpinBox()
        self.min_temp_spin.setRange(0.0, 10.0)
        self.min_temp_spin.setValue(definition.min_temperature)
        sa_layout.addRow(self.make_help_label("Objective", "objective"), self.objective_combo)
        sa_layout.addRow(self.make_help_label("Candidate Epochs", "objective"), self.candidate_epochs_spin)
        sa_layout.addRow(self.make_help_label("Start Temperature", "annealing_config"), self.start_temp_spin)
        sa_layout.addRow(self.make_help_label("Cooling", "annealing_config"), self.cooling_schedule_combo)
        sa_layout.addRow(self.make_help_label("Cooling Parameter", "annealing_config"), self.cooling_param_spin)
        sa_layout.addRow(self.make_help_label("Iter / Temp", "annealing_config"), self.iter_temp_spin)
        sa_layout.addRow(self.make_help_label("Max Steps", "annealing_config"), self.max_steps_spin)
        sa_layout.addRow(self.make_help_label("Min Temperature", "annealing_config"), self.min_temp_spin)
        left_layout.addWidget(self.sa_group)

        self.search_group = QtWidgets.QGroupBox()
        search_layout = QtWidgets.QFormLayout(self.search_group)
        self.search_type_combo = QtWidgets.QComboBox()
        self.search_type_combo.addItems(("none", "grid_search", "random_search"))
        self.random_samples_spin = QtWidgets.QSpinBox()
        self.random_samples_spin.setRange(1, 1000)
        self.random_samples_spin.setValue(8)
        search_layout.addRow(self.make_help_label("Search Type", "search_space"), self.search_type_combo)
        search_layout.addRow(self.make_help_label("Random Samples", "search_space"), self.random_samples_spin)
        self.search_rows: dict[str, tuple[QtWidgets.QComboBox, QtWidgets.QLineEdit]] = {}
        for parameter_name, placeholder in {
            "learning_rate": "0.001,0.01,0.1",
            "batch_size": "8,16,32",
            "epochs": "20,50,100",
            "candidate_epochs": "5,10,20",
            "start_temperature": "0.5,1.0,2.0",
        }.items():
            row_widget = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            mode_combo = QtWidgets.QComboBox()
            mode_combo.addItems(("fixed", "list", "range"))
            value_edit = QtWidgets.QLineEdit()
            value_edit.setPlaceholderText(placeholder)
            row_layout.addWidget(mode_combo)
            row_layout.addWidget(value_edit)
            search_layout.addRow(self.make_help_label(parameter_name, "search_space"), row_widget)
            self.search_rows[parameter_name] = (mode_combo, value_edit)
        left_layout.addWidget(self.search_group)

        self.storage_group = QtWidgets.QGroupBox()
        storage_layout = QtWidgets.QFormLayout(self.storage_group)
        self.output_dir_edit = QtWidgets.QLineEdit(str(OUTPUT_DIR / "experiments"))
        self.load_path_edit = QtWidgets.QLineEdit(str(OUTPUT_DIR / "backend_regression" / "backend_regression_grid"))
        storage_layout.addRow(self.make_help_label("Output Dir", "builder_storage"), self.output_dir_edit)
        storage_layout.addRow(self.make_help_label("Load Path", "builder_storage"), self.load_path_edit)
        left_layout.addWidget(self.storage_group)

        self.run_group = QtWidgets.QGroupBox()
        run_layout = QtWidgets.QVBoxLayout(self.run_group)
        self.run_button = QtWidgets.QPushButton()
        self.run_button.setProperty("role", "primary")
        self.run_button.clicked.connect(self._run_experiment)
        self.load_button = QtWidgets.QPushButton()
        self.load_button.clicked.connect(self._load_results)
        self.export_button = QtWidgets.QPushButton()
        self.export_button.clicked.connect(self._export_definition)
        run_layout.addWidget(self.run_button)
        run_layout.addWidget(self.load_button)
        run_layout.addWidget(self.export_button)
        left_layout.addWidget(self.run_group)
        left_layout.addStretch(1)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(1, 1)

        cards_row = QtWidgets.QHBoxLayout()
        self.summary_card = InfoCardWidget("Experiment", "-")
        self.runs_card = InfoCardWidget("Runs", "-")
        self.ranking_card = InfoCardWidget("Ranking", "-")
        cards_row.addWidget(self.summary_card)
        cards_row.addWidget(self.runs_card)
        cards_row.addWidget(self.ranking_card)
        right_layout.addLayout(cards_row)

        self.tabs = QtWidgets.QTabWidget()
        self.summary_text = QtWidgets.QTextBrowser()
        self.run_model = RunTableModel(self)
        self.run_table = QtWidgets.QTableView()
        self.run_table.setModel(self.run_model)
        self.run_table.selectionModel().currentChanged.connect(self._on_run_selected)
        self.analysis_plot = BuilderSummaryPlotWidget()
        self.detail_text = QtWidgets.QTextBrowser()
        self.network_preview = NetworkViewWidget()
        self.network_snapshot_combo = QtWidgets.QComboBox()
        self.network_snapshot_combo.addItems(("final", "start", "best", "current"))
        self.network_snapshot_combo.currentTextChanged.connect(lambda _text: self._refresh_selected_run_preview())
        network_tab = QtWidgets.QWidget()
        network_layout = QtWidgets.QVBoxLayout(network_tab)
        network_layout.addWidget(self.network_snapshot_combo)
        network_layout.addWidget(self.network_preview)
        self.recipe_panel = RecipePanelWidget(self.preferences.language)
        self.help_text = QtWidgets.QTextBrowser()
        self.tabs.addTab(self.summary_text, "")
        self.tabs.addTab(self.run_table, "")
        self.tabs.addTab(self.analysis_plot, "")
        self.tabs.addTab(self.detail_text, "")
        self.tabs.addTab(network_tab, "")
        self.tabs.addTab(self.recipe_panel, "")
        self.tabs.addTab(self.help_text, "")
        self.tab_help_button = self.make_info_button("builder_results")
        self.tabs.setCornerWidget(self.tab_help_button, QtCore.Qt.TopRightCorner)
        self.tabs.currentChanged.connect(self._update_tab_help_topic)
        right_layout.addWidget(self.tabs)
        self.retranslate()
        self.apply_detail_mode()
        self._update_run_mode_visibility()

    def retranslate(self) -> None:
        self.setup_group.setTitle("Experiment Setup" if self.preferences.language == "en" else "Experiment Setup")
        self.layout_group.setTitle("Layout")
        self.seeds_group.setTitle("Seeds")
        self.training_group.setTitle("Training")
        self.sa_group.setTitle("Simulated Annealing")
        self.search_group.setTitle("Search Space")
        self.storage_group.setTitle("Storage")
        self.run_group.setTitle("Run Control")
        self.run_button.setText(text(self.preferences.language, "common.run"))
        self.load_button.setText(text(self.preferences.language, "common.load"))
        self.export_button.setText(text(self.preferences.language, "common.export"))
        self.tabs.setTabText(0, text(self.preferences.language, "common.summary"))
        self.tabs.setTabText(1, text(self.preferences.language, "common.runs"))
        self.tabs.setTabText(2, text(self.preferences.language, "common.analysis"))
        self.tabs.setTabText(3, "Per-Run Detail")
        self.tabs.setTabText(4, "Network Preview")
        self.tabs.setTabText(5, text(self.preferences.language, "common.recipes"))
        self.tabs.setTabText(6, text(self.preferences.language, "common.help"))
        self.recipe_panel.apply_language(self.preferences.language)
        self._refresh_help_tab()
        self._update_tab_help_topic()

    def apply_detail_mode(self) -> None:
        expert = self.preferences.detail_mode == "expert"
        self.tabs.setTabVisible(4, expert)
        self.search_group.setVisible(expert)

    def _on_benchmark_changed(self) -> None:
        defaults = default_hidden_sizes(self.benchmark_combo.currentText())
        self.hidden_sizes_edit.setText(", ".join(str(v) for v in defaults))
        self.layout_editor.set_hidden_sizes(defaults)
        self._refresh_help_tab()

    def _on_hidden_sizes_changed(self) -> None:
        hidden_sizes = parse_hidden_sizes(self.hidden_sizes_edit.text(), self.layout_editor.hidden_sizes())
        self.hidden_sizes_edit.setText(", ".join(str(v) for v in hidden_sizes))
        self.layout_editor.set_hidden_sizes(hidden_sizes)

    def _update_run_mode_visibility(self) -> None:
        is_sa = self.run_mode_combo.currentText() == "simulated_annealing"
        self.sa_group.setVisible(is_sa)
        if not is_sa:
            for parameter_name in ("candidate_epochs", "start_temperature"):
                combo, edit = self.search_rows[parameter_name]
                combo.setCurrentText("fixed")
                edit.clear()

    def _build_search_space(self) -> SearchSpaceDefinition:
        search_type = self.search_type_combo.currentText()
        definitions: list[SearchValueDefinition] = []
        if search_type == "none":
            return SearchSpaceDefinition(search_type="none")
        for parameter_name, (mode_combo, value_edit) in self.search_rows.items():
            raw_text = value_edit.text().strip()
            if not raw_text:
                continue
            kind = mode_combo.currentText()
            value_type = "float" if parameter_name in {"learning_rate", "start_temperature"} else "int"
            if parameter_name == "batch_size":
                value_type = "int"
            if kind == "fixed":
                definitions.append(
                    SearchValueDefinition(
                        parameter_name=parameter_name,
                        kind="fixed",
                        value_type=value_type,
                        fixed_value=float(raw_text) if value_type == "float" else int(raw_text),
                    )
                )
            elif kind == "list":
                raw_values = [value.strip() for value in raw_text.split(",") if value.strip()]
                casted = tuple(float(value) if value_type == "float" else int(value) for value in raw_values)
                definitions.append(
                    SearchValueDefinition(
                        parameter_name=parameter_name,
                        kind="list",
                        value_type=value_type,
                        values=casted,
                    )
                )
            else:
                start, stop, step = [value.strip() for value in raw_text.split(":")]
                definitions.append(
                    SearchValueDefinition(
                        parameter_name=parameter_name,
                        kind="range",
                        value_type=value_type,
                        range_start=float(start) if value_type == "float" else int(start),
                        range_stop=float(stop) if value_type == "float" else int(stop),
                        range_step=float(step) if value_type == "float" else int(step),
                    )
                )
        return SearchSpaceDefinition(
            search_type=search_type,
            value_definitions=tuple(definitions),
            random_samples=self.random_samples_spin.value(),
            random_state=self.config.random_state,
        )

    def _build_definition(self) -> ExperimentDefinition:
        seeds = tuple(int(value.strip()) for value in self.seeds_edit.text().split(",") if value.strip())
        hidden_sizes = parse_hidden_sizes(self.hidden_sizes_edit.text(), self.layout_editor.hidden_sizes())
        return ExperimentDefinition(
            experiment_id=self.experiment_id_edit.text().strip() or "qt_experiment",
            benchmark=self.benchmark_combo.currentText(),
            hidden_sizes=hidden_sizes,
            layout_spec=self.layout_editor.layout_spec(),
            run_mode=self.run_mode_combo.currentText(),
            seeds=seeds or (self.config.random_state,),
            primary_metric=self.primary_metric_combo.currentText(),
            language=self.preferences.language,
            save_json=True,
            output_dir=normalize_output_path(self.output_dir_edit.text(), str(OUTPUT_DIR / "experiments")),
            shuffle=True,
            learning_rate=self.lr_spin.value(),
            batch_size=self.batch_spin.value(),
            weight_scale=self.weight_spin.value(),
            epochs=self.epochs_spin.value(),
            objective_name=self.objective_combo.currentText(),
            candidate_epochs=self.candidate_epochs_spin.value(),
            neighborhood_operations=("set_neuron", "fill_layer", "swap_neurons"),
            start_temperature=self.start_temp_spin.value(),
            cooling_schedule=self.cooling_schedule_combo.currentText(),
            cooling_parameter=self.cooling_param_spin.value(),
            iterations_per_temperature=self.iter_temp_spin.value(),
            max_steps=self.max_steps_spin.value(),
            min_temperature=self.min_temp_spin.value(),
            search_space=self._build_search_space(),
        )

    def _run_experiment(self) -> None:
        definition = self._build_definition()
        self.task_controller.submit(
            run_experiment_definition,
            definition,
            on_success=self._on_payload_ready,
            on_error=self._on_task_error,
            status_message="Running builder experiment...",
        )

    def _load_results(self) -> None:
        path = self.load_path_edit.text().strip()
        self.task_controller.submit(
            load_saved_experiment,
            path,
            on_success=self._on_payload_ready,
            on_error=self._on_task_error,
            status_message="Loading stored experiment results...",
        )

    def _export_definition(self) -> None:
        definition = self._build_definition()
        target_path = Path(self.output_dir_edit.text().strip()) / f"{definition.experiment_id}.json"
        save_experiment_definition(definition, target_path)
        self.statusMessage.emit(f"Exported definition to {target_path}")

    def _on_payload_ready(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        run_payloads = payload["runs"] if "runs" in payload else [run_result.to_dict() for run_result in payload["run_results"]]
        summary_payload = payload["summary"] if isinstance(payload["summary"], dict) else payload["summary"].to_dict()
        manifest = payload.get("manifest") or {
            "experiment_id": payload["definition"].experiment_id,
            "benchmark": payload["definition"].benchmark,
            "run_mode": payload["definition"].run_mode,
            "search_type": payload["definition"].search_space.search_type,
            "primary_metric": payload["definition"].primary_metric,
        }
        self.payload = {"manifest": manifest, "summary": summary_payload, "runs": run_payloads}
        self.run_model.set_runs(run_payloads)
        self.analysis_plot.set_summary(summary_payload)
        self.summary_text.setPlainText(format_experiment_summary(self.payload))
        self.summary_card.set_content(value=manifest["experiment_id"], body=manifest["benchmark"])
        self.runs_card.set_content(value=str(summary_payload["number_of_runs"]), body=f"configs={summary_payload['configuration_count']}")
        top_score = summary_payload["ranking"][0]["ranking_score"] if summary_payload.get("ranking") else 0.0
        self.ranking_card.set_content(value=f"{top_score:.4f}", body=manifest["primary_metric"])
        if run_payloads:
            self.run_table.selectRow(0)
            self._set_selected_run(run_payloads[0])
        self.statusMessage.emit("Experiment payload ready.")

    def _on_task_error(self, traceback_text: str) -> None:
        QtWidgets.QMessageBox.critical(self, "Task failed", traceback_text)
        self.statusMessage.emit("Task failed.")

    def _on_run_selected(self, current: QtCore.QModelIndex, _previous: QtCore.QModelIndex) -> None:
        payload = self.run_model.run_payload(current.row())
        if payload is not None:
            self._set_selected_run(payload)

    def _set_selected_run(self, run_payload: dict[str, Any]) -> None:
        self.selected_run_payload = run_payload
        self.detail_text.setPlainText(format_run_summary(run_payload))
        self._refresh_selected_run_preview()

    def _refresh_selected_run_preview(self) -> None:
        if self.selected_run_payload is None:
            self.network_preview.set_model(None)
            return
        extra = self.selected_run_payload.get("extra", {})
        run_definition = self.selected_run_payload["run_definition"]
        snapshot = self.network_snapshot_combo.currentText()
        model_state = None
        if run_definition["run_mode"] == "manual_training":
            model_state = extra.get("model_state")
        else:
            if snapshot == "start":
                model_state = extra.get("start_model_state")
            elif snapshot == "best":
                model_state = extra.get("best_model_state")
            else:
                model_state = extra.get("current_model_state")
        if model_state is None:
            dataset = load_benchmark(
                DatasetConfig(
                    name=run_definition["benchmark"],
                    random_state=int(run_definition["seed"]),
                )
            )
            hidden_sizes = tuple(run_definition["hidden_sizes"])
            layout = parse_layout_spec(self.selected_run_payload["layout_spec"], hidden_sizes)
            model = ModularMLP(
                input_size=dataset.input_size,
                hidden_sizes=hidden_sizes,
                output_size=dataset.model_output_size,
                layout=layout,
                num_classes=dataset.output_size,
                random_state=0,
            )
        else:
            model = ModularMLP.from_state_dict(model_state)
        self.network_preview.set_model(model)

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
            0: "builder_results",
            1: "builder_results",
            2: "builder_results",
            3: "builder_results",
            4: "network_view",
            5: "recipes",
            6: "workspace_help",
        }
        self.tab_help_button.set_topic_key(topic_by_index.get(self.tabs.currentIndex(), "workspace_help"))
