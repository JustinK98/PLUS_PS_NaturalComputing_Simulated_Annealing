"""Gefuehrter Praesentationsmodus fuer Low-Level-Vortraege."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from activations import parse_layout_spec
from configs import GuiExperimentConfig
from model import ModularMLP
from services.analysis_sample_service import build_analysis_sample
from services.annealing_session_service import (
    AnnealingSessionSnapshot,
    evaluate_start,
    run_to_completion,
    step_once,
)
from services.neuron_analysis_service import build_neuron_analysis_payload
from services.network_projection_service import build_input_projection
from services.presentation_service import (
    PRESENTATION_HIDDEN_SIZES,
    PRESENTATION_LAYOUT,
    PRESENTATION_RANDOM_STATE,
    PRESENTATION_WEIGHT_SCALE,
    PresentationRuntimeState,
    create_presentation_annealing_session,
    create_presentation_state,
    presentation_training_config,
)
from services.training_session_service import (
    create_training_session,
    model_from_training_session,
    run_training_session_epochs,
    training_result_from_session,
)
from ui_qt.presentation import TRACKS, PresentationSlide
from ui_qt.state import WorkspacePreferences
from ui_qt.tasking import BackgroundTaskController
from ui_qt.widgets.annealing_decision_panel import AnnealingDecisionPanel
from ui_qt.widgets.annealing_history_panel import AnnealingHistoryPanel
from ui_qt.widgets.activation_curve_widget import ActivationCurveWidget
from ui_qt.widgets.network_view import NetworkViewWidget
from ui_qt.widgets.neuron_detail_panel import NeuronDetailPanel
from ui_qt.widgets.plot_widgets import TrainingPlotWidget
from ui_qt.widgets.presentation_widgets import (
    ActivationLabWidget,
    ConceptChoiceWidget,
    MetricsCardRow,
    NetworkComparisonWidget,
    PresentationSlideWidget,
    ReadingGuideWidget,
)
from ui_qt.widgets.sample_panel import SampleDisplayPayload, SamplePanelWidget
from .base import BaseWorkspace


def _step_annealing_many(session, language: str, count: int) -> AnnealingSessionSnapshot:
    snapshot = evaluate_start(session, language)
    for _ in range(count):
        snapshot = step_once(session, language)
        if snapshot.is_complete:
            break
    return snapshot


class PresentationWorkspace(BaseWorkspace):
    """Slide-basierte, reduzierte Oberflaeche fuer die Live-Praesentation."""

    workspace_id = "presentation"

    def __init__(
        self,
        config: GuiExperimentConfig,
        preferences: WorkspacePreferences,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(preferences, parent)
        self.config = config
        self.runtime_state: PresentationRuntimeState = create_presentation_state(preferences.language)
        self.track_id: str | None = None
        self.slide_index = 0
        self.annealing_snapshot: AnnealingSessionSnapshot | None = None
        self.training_before_prediction = self._prediction_summary()
        self.training_after_prediction: tuple[str, str] | None = None
        self.task_controller = BackgroundTaskController(self)
        self.task_controller.busyChanged.connect(self._on_busy_changed)
        self.task_controller.messageEmitted.connect(self.statusMessage)
        self._build_ui()
        self._show_track_choice()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.stack = QtWidgets.QStackedWidget()
        root.addWidget(self.stack)

        self.choice = ConceptChoiceWidget(self.preferences.language)
        self.choice.trackSelected.connect(self._start_track)
        self.stack.addWidget(self.choice)

        self.slide_view = PresentationSlideWidget()
        self.slide_view.back_button.clicked.connect(self._previous_slide)
        self.slide_view.next_button.clicked.connect(self._next_slide)
        self.slide_view.reset_button.clicked.connect(self._show_track_choice)
        self.slide_view.restart_button.clicked.connect(self._reset_current_track)
        self.slide_view.action_button.clicked.connect(self._run_slide_action)
        self.slide_view.actionRequested.connect(self._run_named_action)
        self.stack.addWidget(self.slide_view)

        self.intro_panel = QtWidgets.QTextBrowser()
        self.sample_panel = SamplePanelWidget(self.preferences.language)
        self.network_panel = self._make_network_panel()
        self.activation_lab_panel = self._make_activation_lab_panel()
        self.neuron_detail_panel = NeuronDetailPanel(self.preferences.language)
        self.training_panel = self._make_training_panel()
        self.layout_comparison_panel = NetworkComparisonWidget(self.preferences.language)
        self.neighbor_comparison_panel = NetworkComparisonWidget(self.preferences.language)
        self.sa_evaluation_panel = self._make_sa_evaluation_panel()
        self.sa_decision_panel = self._make_sa_decision_panel()
        self.sa_history_panel = self._make_sa_history_panel()
        self.sa_summary_panel = self._make_sa_summary_panel()

        self._visuals = {
            "intro_problem": self.intro_panel,
            "sample_with_definitions": self.sample_panel,
            "network_with_curve": self.network_panel,
            "activation_lab": self.activation_lab_panel,
            "neuron_calculation": self.neuron_detail_panel,
            "training_with_prediction": self.training_panel,
            "transition_to_sa": self.intro_panel,
            "sa_intro": self.intro_panel,
            "layout_comparison": self.layout_comparison_panel,
            "neighbor_comparison": self.neighbor_comparison_panel,
            "sa_evaluation": self.sa_evaluation_panel,
            "sa_decision_cards": self.sa_decision_panel,
            "sa_history_explained": self.sa_history_panel,
            "sa_run_summary": self.sa_summary_panel,
            "sa_final_takeaway": self.intro_panel,
        }
        added_widgets: list[QtWidgets.QWidget] = []
        for widget in self._visuals.values():
            if widget not in added_widgets:
                self.slide_view.visual_host.addWidget(widget)
                added_widgets.append(widget)

    def _make_network_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        panel.setChildrenCollapsible(False)
        self.network_view = NetworkViewWidget()
        self.network_view.hiddenNeuronSelected.connect(self._on_network_selection_changed)
        self.network_neuron_detail = NeuronDetailPanel(self.preferences.language)
        self.network_neuron_detail.setMinimumHeight(330)
        panel.addWidget(self.network_view)
        panel.addWidget(self.network_neuron_detail)
        panel.setStretchFactor(0, 2)
        panel.setStretchFactor(1, 3)
        return panel

    def _make_activation_lab_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        panel.setChildrenCollapsible(False)
        top = QtWidgets.QWidget()
        top_layout = QtWidgets.QVBoxLayout(top)
        self.activation_lab = ActivationLabWidget(PRESENTATION_HIDDEN_SIZES, self.preferences.language)
        self.activation_lab.selectionChanged.connect(self._on_activation_lab_selection_changed)
        self.activation_lab.activationChanged.connect(self._on_activation_lab_changed)
        self.activation_lab_network = NetworkViewWidget()
        self.activation_lab_network.hiddenNeuronSelected.connect(self._on_activation_lab_selection_changed)
        top_layout.addWidget(self.activation_lab)
        top_layout.addWidget(self.activation_lab_network, 1)
        self.activation_lab_curve = ActivationCurveWidget()
        self.activation_lab_curve.setMinimumHeight(430)
        panel.addWidget(top)
        panel.addWidget(self.activation_lab_curve)
        panel.setStretchFactor(0, 1)
        panel.setStretchFactor(1, 3)
        return panel

    def _make_training_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        panel.setChildrenCollapsible(False)
        self.training_cards = MetricsCardRow()
        self.training_plot = TrainingPlotWidget()
        self.training_guide = ReadingGuideWidget()
        panel.addWidget(self.training_cards)
        panel.addWidget(self.training_plot)
        panel.addWidget(self.training_guide)
        panel.setStretchFactor(0, 1)
        panel.setStretchFactor(1, 4)
        panel.setStretchFactor(2, 1)
        return panel

    def _make_sa_evaluation_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QVBoxLayout()
        container = QtWidgets.QWidget()
        container.setLayout(panel)
        self.sa_evaluation_cards = MetricsCardRow()
        self.sa_evaluation_text = QtWidgets.QTextBrowser()
        self.sa_evaluation_text.setStyleSheet("font-size: 22px; line-height: 1.45;")
        panel.addWidget(self.sa_evaluation_cards)
        panel.addWidget(self.sa_evaluation_text, 1)
        return container

    def _make_sa_decision_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        panel.setChildrenCollapsible(False)
        self.sa_decision_cards = MetricsCardRow()
        self.sa_decision_detail = AnnealingDecisionPanel(self.preferences.language)
        self.sa_decision_guide = ReadingGuideWidget()
        panel.addWidget(self.sa_decision_cards)
        panel.addWidget(self.sa_decision_detail)
        panel.addWidget(self.sa_decision_guide)
        panel.setStretchFactor(0, 1)
        panel.setStretchFactor(1, 3)
        panel.setStretchFactor(2, 1)
        return panel

    def _make_sa_history_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        panel.setChildrenCollapsible(False)
        self.sa_history_cards = MetricsCardRow()
        self.sa_history_detail = AnnealingHistoryPanel(self.preferences.language)
        self.sa_history_guide = ReadingGuideWidget()
        panel.addWidget(self.sa_history_cards)
        panel.addWidget(self.sa_history_detail)
        panel.addWidget(self.sa_history_guide)
        panel.setStretchFactor(0, 1)
        panel.setStretchFactor(1, 4)
        panel.setStretchFactor(2, 1)
        return panel

    def _make_sa_summary_panel(self) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        self.sa_summary_cards = MetricsCardRow()
        self.sa_summary_layouts = NetworkComparisonWidget(self.preferences.language)
        layout.addWidget(self.sa_summary_cards)
        layout.addWidget(self.sa_summary_layouts, 1)
        return container

    def retranslate(self) -> None:
        self.choice.set_language(self.preferences.language)
        self.sample_panel.set_language(self.preferences.language)
        self.neuron_detail_panel.set_language(self.preferences.language)
        self.network_neuron_detail.set_language(self.preferences.language)
        self.activation_lab.set_language(self.preferences.language)
        self.sa_decision_detail.set_language(self.preferences.language)
        self.sa_history_detail.set_language(self.preferences.language)
        self.layout_comparison_panel.set_language(self.preferences.language)
        self.neighbor_comparison_panel.set_language(self.preferences.language)
        self.sa_summary_layouts.set_language(self.preferences.language)
        if self.track_id is not None:
            self._refresh_slide()

    def apply_detail_mode(self) -> None:
        return

    def _show_track_choice(self) -> None:
        self.track_id = None
        self.slide_index = 0
        self.runtime_state = create_presentation_state(self.preferences.language)
        self.annealing_snapshot = None
        self.training_before_prediction = self._prediction_summary()
        self.training_after_prediction = None
        self.stack.setCurrentWidget(self.choice)
        self.statusMessage.emit(
            "Presentation track selection ready."
            if self.preferences.language == "en"
            else "Praesentationsauswahl bereit."
        )

    def _start_track(self, track_id: str) -> None:
        self.track_id = track_id
        self.slide_index = 0
        self.runtime_state = create_presentation_state(self.preferences.language)
        self.annealing_snapshot = None
        self.training_before_prediction = self._prediction_summary()
        self.training_after_prediction = None
        if track_id == "simulated_annealing":
            self.runtime_state.annealing_session = create_presentation_annealing_session()
        self.stack.setCurrentWidget(self.slide_view)
        self._refresh_slide()

    def _current_slide(self) -> PresentationSlide:
        if self.track_id is None:
            raise ValueError("No presentation track selected.")
        return TRACKS[self.track_id][self.slide_index]

    def _refresh_slide(self) -> None:
        slide = self._current_slide()
        total = len(TRACKS[slide.track_id])
        self.slide_view.set_slide(
            slide,
            slide_index=self.slide_index,
            total_slides=total,
            language=self.preferences.language,
        )
        self.slide_view.back_button.setEnabled(self.slide_index > 0)
        self.slide_view.next_button.setEnabled(self.slide_index < total - 1)
        self.slide_view.set_action(self._action_label(slide.action))
        self.slide_view.set_extra_actions(self._extra_actions(slide))
        self.slide_view.set_actions_enabled(not self.task_controller.is_busy)
        self._render_visual(slide)

    def _render_visual(self, slide: PresentationSlide) -> None:
        if slide.visual in {"intro_problem", "transition_to_sa", "sa_intro", "sa_final_takeaway"}:
            self.intro_panel.setHtml(self._concept_html(slide))
        elif slide.visual == "sample_with_definitions":
            self._refresh_sample()
        elif slide.visual == "network_with_curve":
            self._refresh_network_with_curve()
        elif slide.visual == "activation_lab":
            self._refresh_activation_lab()
        elif slide.visual == "neuron_calculation":
            self._refresh_neuron_detail()
        elif slide.visual == "training_with_prediction":
            self._refresh_training_panel()
        elif slide.visual == "layout_comparison":
            self._refresh_layout_comparison()
        elif slide.visual == "neighbor_comparison":
            self._refresh_neighbor_comparison()
        elif slide.visual == "sa_evaluation":
            self._refresh_sa_evaluation()
        elif slide.visual == "sa_decision_cards":
            self._refresh_sa_decision()
        elif slide.visual == "sa_history_explained":
            self._refresh_sa_history()
        elif slide.visual == "sa_run_summary":
            self._refresh_sa_summary()
        self.slide_view.visual_host.setCurrentWidget(self._visuals[slide.visual])

    def _refresh_sample(self) -> None:
        state = self.runtime_state
        probabilities = state.model.predict_proba(state.analysis_sample.scaled_sample.reshape(1, -1))[0]
        prediction_index = int(state.model.predict(state.analysis_sample.scaled_sample.reshape(1, -1))[0])
        self.sample_panel.set_sample(
            SampleDisplayPayload(
                dataset=state.dataset,
                analysis_sample=state.analysis_sample,
                prediction_name=state.dataset.target_names[prediction_index],
                probabilities=probabilities,
            )
        )

    def _refresh_network_with_curve(self) -> None:
        self._refresh_network(self.network_view)
        payload = self._neuron_payload()
        self.network_neuron_detail.set_payload(payload, self.preferences.language)

    def _refresh_network(self, view: NetworkViewWidget) -> None:
        state = self.runtime_state
        view.set_input_projection(
            build_input_projection(
                state.dataset,
                state.model,
                state.analysis_sample,
                state.selected_hidden,
                max_inputs=10,
            )
        )
        view.set_model(state.model)
        view.set_selected_hidden(state.selected_hidden)

    def _refresh_activation_lab(self) -> None:
        state = self.runtime_state
        layer_index, neuron_index = state.selected_hidden
        self.activation_lab.set_selection(
            layer_index,
            neuron_index,
            state.model.layout.layers[layer_index][neuron_index],
        )
        self.activation_lab_network.set_input_projection(
            build_input_projection(
                state.dataset,
                state.model,
                state.analysis_sample,
                state.selected_hidden,
                max_inputs=10,
            )
        )
        self.activation_lab_network.set_model(state.model)
        self.activation_lab_network.set_selected_hidden(state.selected_hidden)
        payload = build_neuron_analysis_payload(
            state.model,
            state.dataset,
            state.analysis_sample,
            layer_index,
            neuron_index,
            self.preferences.language,
        )
        self.activation_lab_curve.set_payload(payload)

    def _refresh_neuron_detail(self) -> None:
        self.neuron_detail_panel.set_payload(self._neuron_payload(), self.preferences.language)

    def _neuron_payload(self):
        state = self.runtime_state
        layer_index, neuron_index = state.selected_hidden
        return build_neuron_analysis_payload(
            state.model,
            state.dataset,
            state.analysis_sample,
            layer_index,
            neuron_index,
            self.preferences.language,
        )

    def _refresh_training_panel(self) -> None:
        result = training_result_from_session(self.runtime_state.training_session)
        self.training_plot.set_history(result.history)
        before_name, before_probs = self.training_before_prediction
        after_name, after_probs = self.training_after_prediction or self._prediction_summary()
        cards = [
            (
                "Vorher" if self.preferences.language == "de" else "Before",
                before_name,
                before_probs,
            ),
            (
                "Nachher" if self.preferences.language == "de" else "After",
                after_name,
                after_probs,
            ),
            (
                "Echtes Ziel" if self.preferences.language == "de" else "True target",
                self.runtime_state.analysis_sample.actual_target_name,
                "Label aus dem Datensatz" if self.preferences.language == "de" else "Label from dataset",
            ),
            (
                "Epochen" if self.preferences.language == "de" else "Epochs",
                str(self.runtime_state.training_session.completed_epochs),
                "Mini-Batch Gradient Descent",
            ),
        ]
        self.training_cards.set_cards(cards)
        if self.preferences.language == "en":
            self.training_guide.set_content(
                "How to read this",
                (
                    "Loss should go down; it measures prediction error.",
                    "Accuracy should go up; it measures correct classifications.",
                    "Gradient descent updates weights and biases, not the activation layout.",
                    "The sample prediction cards show whether this concrete example changes.",
                ),
            )
        else:
            self.training_guide.set_content(
                "So liest du die Anzeige",
                (
                    "Loss soll sinken; er misst den Vorhersagefehler.",
                    "Accuracy soll steigen; sie misst korrekte Klassifikationen.",
                    "Gradient Descent veraendert Gewichte und Biases, nicht das Aktivierungs-Layout.",
                    "Die Sample-Karten zeigen, ob sich dieses konkrete Beispiel veraendert.",
                ),
            )

    def _refresh_layout_comparison(self) -> None:
        start_model = self._model_for_layout(PRESENTATION_LAYOUT)
        target_layout = "relu*2,tanh*2,gelu*2,swish*2"
        target_model = self._model_for_layout(target_layout)
        projection_left = self._projection_for(start_model, (0, 0))
        projection_right = self._projection_for(target_model, (0, 1))
        description = (
            f"<b>Startlayout:</b> {PRESENTATION_LAYOUT}<br>"
            "<b>Beispielhaftes Ziellayout:</b> gemischte Aktivierungsverteilung.<br>"
            "SA sucht genau in diesem diskreten Raum nach Layouts, die nach kurzem Training besser validieren."
            if self.preferences.language == "de"
            else f"<b>Start layout:</b> {PRESENTATION_LAYOUT}<br>"
            "<b>Illustrative target layout:</b> mixed activation distribution.<br>"
            "SA searches this discrete space for layouts that validate better after short training."
        )
        self.layout_comparison_panel.set_models(
            left_title="Startlayout",
            right_title="Beispiel-Ziellayout" if self.preferences.language == "de" else "Example target layout",
            left_model=start_model,
            right_model=target_model,
            left_projection=projection_left,
            right_projection=projection_right,
            left_selection=(0, 0),
            right_selection=(0, 1),
            description_html=description,
        )

    def _refresh_neighbor_comparison(self) -> None:
        current_model = self._model_for_layout(PRESENTATION_LAYOUT)
        neighbor_layout = current_model.layout.replace_neuron(0, 0, "tanh")
        neighbor_model = self._model_for_layout(neighbor_layout.to_compact_spec())
        description = (
            "<b>Operation:</b> set_neuron(L1:n0, tanh)<br>"
            "Nur eine lokale Entscheidung aendert sich. Genau solche kleinen Schritte definieren die Nachbarschaft."
            if self.preferences.language == "de"
            else "<b>Operation:</b> set_neuron(L1:n0, tanh)<br>"
            "Only one local choice changes. These small steps define the neighborhood."
        )
        self.neighbor_comparison_panel.set_models(
            left_title=f"Current: {PRESENTATION_LAYOUT}",
            right_title="Candidate: set_neuron(L1:n0, tanh)",
            left_model=current_model,
            right_model=neighbor_model,
            left_projection=self._projection_for(current_model, (0, 0)),
            right_projection=self._projection_for(neighbor_model, (0, 0)),
            left_selection=(0, 0),
            right_selection=(0, 0),
            description_html=description,
        )

    def _refresh_sa_evaluation(self) -> None:
        snapshot = self.annealing_snapshot
        if snapshot is None or snapshot.start_evaluation is None:
            self.sa_evaluation_cards.set_cards(
                [
                    (
                        "Start" if self.preferences.language == "de" else "Start",
                        PRESENTATION_LAYOUT,
                        "Klicke Start bewerten." if self.preferences.language == "de" else "Click Evaluate start.",
                    )
                ]
            )
            self.sa_evaluation_text.setHtml(self._evaluation_placeholder())
            return
        evaluation = snapshot.start_evaluation
        self.sa_evaluation_cards.set_cards(
            [
                ("Score", f"{evaluation.comparable_score:.4f}", "validation_loss"),
                ("Val Loss", f"{evaluation.val_loss:.4f}", "kleiner ist besser" if self.preferences.language == "de" else "lower is better"),
                ("Val Acc", f"{evaluation.val_accuracy:.3f}", "Klassifikationsleistung"),
                ("Layout", evaluation.layout.to_compact_spec(), "bewerteter Startzustand"),
            ]
        )
        self.sa_evaluation_text.setHtml(self._evaluation_result_html(evaluation))

    def _refresh_sa_decision(self) -> None:
        snapshot = self.annealing_snapshot
        if snapshot is None:
            self.sa_decision_cards.set_cards([])
            self.sa_decision_detail.set_placeholder(self.preferences.language)
            return
        self.sa_decision_cards.set_cards(self._snapshot_cards(snapshot))
        self.sa_decision_detail.set_snapshot(snapshot, self.preferences.language)
        if self.preferences.language == "en":
            self.sa_decision_guide.set_content(
                "How to read this",
                (
                    "Candidate is the newly proposed layout.",
                    "Current is the accepted layout after the decision.",
                    "Best is the best layout seen so far.",
                    "High temperature can still accept worse candidates.",
                ),
            )
        else:
            self.sa_decision_guide.set_content(
                "So liest du die Entscheidung",
                (
                    "Candidate ist das neu vorgeschlagene Layout.",
                    "Current ist das nach der Entscheidung akzeptierte Layout.",
                    "Best ist das beste bisher gesehene Layout.",
                    "Hohe Temperatur kann schlechtere Kandidaten trotzdem akzeptieren.",
                ),
            )

    def _refresh_sa_history(self) -> None:
        snapshot = self.annealing_snapshot
        if snapshot is None:
            self.sa_history_cards.set_cards([])
            self.sa_history_detail.set_placeholder(self.preferences.language)
            return
        self.sa_history_cards.set_cards(self._snapshot_cards(snapshot))
        self.sa_history_detail.set_snapshot(snapshot, self.preferences.language)
        if self.preferences.language == "en":
            self.sa_history_guide.set_content(
                "How to read the run",
                (
                    "Best score should only improve or stay equal.",
                    "Current score may fluctuate because exploration is allowed.",
                    "Temperature decreases and makes later steps more conservative.",
                    "Acceptance rate shows how often candidates became current.",
                ),
            )
        else:
            self.sa_history_guide.set_content(
                "So liest du den Lauf",
                (
                    "Best Score sollte nur besser werden oder gleich bleiben.",
                    "Current Score darf schwanken, weil Exploration erlaubt ist.",
                    "Temperature sinkt und macht spaetere Schritte konservativer.",
                    "Acceptance Rate zeigt, wie oft Kandidaten zu Current wurden.",
                ),
            )

    def _refresh_sa_summary(self) -> None:
        snapshot = self.annealing_snapshot
        if snapshot is None:
            self.sa_summary_cards.set_cards([])
            self._refresh_summary_comparison(None)
            return
        self.sa_summary_cards.set_cards(self._snapshot_cards(snapshot))
        self._refresh_summary_comparison(snapshot)

    def _refresh_summary_comparison(self, snapshot: AnnealingSessionSnapshot | None) -> None:
        start_model = self._model_for_layout(PRESENTATION_LAYOUT)
        best_eval = snapshot.best_evaluation if snapshot is not None else None
        current_eval = snapshot.current_evaluation if snapshot is not None else None
        right_eval = best_eval or current_eval
        right_model = right_eval.trained_model.clone() if right_eval is not None else self._model_for_layout(PRESENTATION_LAYOUT)
        right_layout = right_eval.layout.to_compact_spec() if right_eval is not None else PRESENTATION_LAYOUT
        status = (
            ", ".join(snapshot.stop_reasons)
            if snapshot is not None and snapshot.stop_reasons
            else ("Lauf noch nicht abgeschlossen" if self.preferences.language == "de" else "Run not complete yet")
        )
        description = (
            f"<b>Start:</b> {PRESENTATION_LAYOUT}<br><b>Current/Best:</b> {right_layout}<br><b>Status:</b> {status}"
            if self.preferences.language == "de"
            else f"<b>Start:</b> {PRESENTATION_LAYOUT}<br><b>Current/Best:</b> {right_layout}<br><b>Status:</b> {status}"
        )
        self.sa_summary_layouts.set_models(
            left_title="Start Layout",
            right_title="Best/Current Layout",
            left_model=start_model,
            right_model=right_model,
            left_projection=self._projection_for(start_model, (0, 0)),
            right_projection=self._projection_for(right_model, (0, 0)),
            left_selection=(0, 0),
            right_selection=(0, 0),
            description_html=description,
        )

    def _snapshot_cards(self, snapshot: AnnealingSessionSnapshot) -> list[tuple[str, str, str]]:
        def score(evaluation) -> str:
            return "-" if evaluation is None else f"{evaluation.comparable_score:.4f}"

        return [
            ("Current", score(snapshot.current_evaluation), self._layout_of(snapshot.current_evaluation)),
            ("Candidate", score(snapshot.candidate_evaluation), self._layout_of(snapshot.candidate_evaluation)),
            ("Best", score(snapshot.best_evaluation), self._layout_of(snapshot.best_evaluation)),
            ("T", f"{snapshot.current_temperature:.3f}", f"accepted={snapshot.accepted_steps}, rate={snapshot.acceptance_rate:.2f}"),
        ]

    def _layout_of(self, evaluation) -> str:
        return "-" if evaluation is None else evaluation.layout.to_compact_spec()

    def _model_for_layout(self, layout_spec: str) -> ModularMLP:
        return ModularMLP(
            input_size=self.runtime_state.dataset.input_size,
            hidden_sizes=PRESENTATION_HIDDEN_SIZES,
            output_size=self.runtime_state.dataset.model_output_size,
            layout=parse_layout_spec(layout_spec, PRESENTATION_HIDDEN_SIZES),
            num_classes=self.runtime_state.dataset.output_size,
            weight_scale=PRESENTATION_WEIGHT_SCALE,
            random_state=PRESENTATION_RANDOM_STATE,
        )

    def _projection_for(self, model: ModularMLP, selection: tuple[int, int] | None):
        return build_input_projection(
            self.runtime_state.dataset,
            model,
            self.runtime_state.analysis_sample,
            selection,
            max_inputs=10,
        )

    def _prediction_summary(self) -> tuple[str, str]:
        state = self.runtime_state
        probabilities = state.model.predict_proba(state.analysis_sample.scaled_sample.reshape(1, -1))[0]
        prediction_index = int(state.model.predict(state.analysis_sample.scaled_sample.reshape(1, -1))[0])
        probability_text = ", ".join(
            f"{state.dataset.target_names[index]}={float(probability):.2f}"
            for index, probability in enumerate(probabilities)
        )
        return state.dataset.target_names[prediction_index], probability_text

    def _previous_slide(self) -> None:
        if self.slide_index > 0:
            self.slide_index -= 1
            self._refresh_slide()

    def _next_slide(self) -> None:
        if self.track_id is None:
            return
        if self.slide_index < len(TRACKS[self.track_id]) - 1:
            self.slide_index += 1
            self._refresh_slide()

    def _reset_current_track(self) -> None:
        if self.track_id is None:
            self._show_track_choice()
            return
        track_id = self.track_id
        self._start_track(track_id)
        self.statusMessage.emit(
            "Track reset to deterministic start."
            if self.preferences.language == "en"
            else "Track auf reproduzierbaren Start zurueckgesetzt."
        )

    def _run_slide_action(self) -> None:
        action = self._current_slide().action
        if action is not None:
            self._run_named_action(action)

    def _run_named_action(self, action: str) -> None:
        if action == "select_neuron":
            self.runtime_state.selected_hidden = (0, 0)
            self._refresh_slide()
            return
        if action == "cycle_activation":
            layer_index, neuron_index = self.runtime_state.selected_hidden
            new_layout = self.runtime_state.model.layout.cycle_neuron(layer_index, neuron_index)
            self._replace_runtime_model_layout(new_layout)
            self._refresh_slide()
            return
        if action == "train_10":
            self.task_controller.submit(
                run_training_session_epochs,
                self.runtime_state.training_session,
                self.runtime_state.dataset,
                presentation_training_config(10),
                on_success=self._on_training_ready,
                on_error=self._on_task_error,
                status_message=(
                    "Training 10 presentation epochs..."
                    if self.preferences.language == "en"
                    else "Trainiere 10 Praesentationsepochen..."
                ),
            )
            return
        if action in {"evaluate_start", "step_once", "step_10", "run_to_completion"}:
            self._run_annealing_action(action)

    def _run_annealing_action(self, action: str) -> None:
        if self.runtime_state.annealing_session is None:
            self.runtime_state.annealing_session = create_presentation_annealing_session()
        if action == "step_10":
            function = _step_annealing_many
            args = (self.runtime_state.annealing_session, self.preferences.language, 10)
        else:
            function = {
                "evaluate_start": evaluate_start,
                "step_once": step_once,
                "run_to_completion": run_to_completion,
            }[action]
            args = (self.runtime_state.annealing_session, self.preferences.language)
        self.task_controller.submit(
            function,
            *args,
            on_success=self._on_annealing_ready,
            on_error=self._on_task_error,
            status_message=(
                "Updating annealing presentation..."
                if self.preferences.language == "en"
                else "Aktualisiere Annealing-Praesentation..."
            ),
        )

    def _on_network_selection_changed(self, layer_index: int, neuron_index: int) -> None:
        self.runtime_state.selected_hidden = (layer_index, neuron_index)
        if self.track_id is not None:
            self._refresh_slide()

    def _on_activation_lab_selection_changed(self, layer_index: int, neuron_index: int) -> None:
        self.runtime_state.selected_hidden = (layer_index, neuron_index)
        if self.track_id is not None and self._current_slide().visual == "activation_lab":
            self._refresh_activation_lab()

    def _on_activation_lab_changed(self, layer_index: int, neuron_index: int, activation_name: str) -> None:
        new_layout = self.runtime_state.model.layout.replace_neuron(layer_index, neuron_index, activation_name)
        self.runtime_state.selected_hidden = (layer_index, neuron_index)
        self._replace_runtime_model_layout(new_layout)
        if self.track_id is not None and self._current_slide().visual == "activation_lab":
            self._refresh_activation_lab()

    def _replace_runtime_model_layout(self, layout) -> None:
        self.runtime_state.model = ModularMLP(
            input_size=self.runtime_state.dataset.input_size,
            hidden_sizes=PRESENTATION_HIDDEN_SIZES,
            output_size=self.runtime_state.dataset.model_output_size,
            layout=layout,
            num_classes=self.runtime_state.dataset.output_size,
            weight_scale=PRESENTATION_WEIGHT_SCALE,
            random_state=PRESENTATION_RANDOM_STATE,
        )
        self.runtime_state.training_session = create_training_session(self.runtime_state.model)
        self.training_before_prediction = self._prediction_summary()
        self.training_after_prediction = None

    def _on_training_ready(self, session) -> None:
        self.runtime_state.training_session = session
        self.runtime_state.model = model_from_training_session(session)
        self.runtime_state.analysis_sample = build_analysis_sample(
            self.runtime_state.dataset,
            "val",
            0,
            language=self.preferences.language,
        )
        self.training_after_prediction = self._prediction_summary()
        self.statusMessage.emit(f"Training updated: {session.completed_epochs} epochs.")
        self._refresh_slide()

    def _on_annealing_ready(self, snapshot: AnnealingSessionSnapshot) -> None:
        self.annealing_snapshot = snapshot
        model_evaluation = snapshot.best_evaluation or snapshot.current_evaluation or snapshot.start_evaluation
        if model_evaluation is not None:
            self.runtime_state.model = model_evaluation.trained_model.clone()
        self.statusMessage.emit(
            "Annealing presentation updated."
            if self.preferences.language == "en"
            else "Annealing-Praesentation aktualisiert."
        )
        self._refresh_slide()

    def _on_task_error(self, traceback_text: str) -> None:
        QtWidgets.QMessageBox.critical(self, "Presentation task failed", traceback_text)
        self.statusMessage.emit("Presentation task failed.")

    def _on_busy_changed(self, busy: bool) -> None:
        self.slide_view.set_actions_enabled(not busy)
        self.busyChanged.emit(busy)

    def _action_label(self, action: str | None) -> str | None:
        if action is None:
            return None
        labels_de = {
            "select_neuron": "Neuron markieren",
            "cycle_activation": "Aktivierung wechseln",
            "train_10": "Train 10",
            "evaluate_start": "Start bewerten",
            "step_once": "Ein SA-Schritt",
            "step_10": "10 SA-Schritte",
            "run_to_completion": "Bis zum Ende laufen",
        }
        labels_en = {
            "select_neuron": "Highlight neuron",
            "cycle_activation": "Change activation",
            "train_10": "Train 10",
            "evaluate_start": "Evaluate start",
            "step_once": "One SA step",
            "step_10": "10 SA steps",
            "run_to_completion": "Run to completion",
        }
        return (labels_en if self.preferences.language == "en" else labels_de)[action]

    def _extra_actions(self, slide: PresentationSlide) -> list[tuple[str, str]]:
        if slide.visual not in {"sa_decision_cards", "sa_history_explained", "sa_run_summary"}:
            return []
        actions = ["step_once", "step_10", "run_to_completion"]
        return [
            (action, self._action_label(action))
            for action in actions
            if action != slide.action
        ]

    def _concept_html(self, slide: PresentationSlide) -> str:
        bullets = "".join(f"<li>{bullet}</li>" for bullet in slide.bullets(self.preferences.language))
        return f"""
        <div style="font-size: 30px; line-height: 1.55; padding: 34px;">
          <h1 style="font-size: 42px; margin-bottom: 28px; color: #0f172a;">
            {slide.question(self.preferences.language)}
          </h1>
          <ul style="margin-top: 24px;">{bullets}</ul>
        </div>
        """

    def _evaluation_placeholder(self) -> str:
        return (
            "<h2>Noch keine Startbewertung</h2><p>Klicke <b>Start bewerten</b>. "
            "Dann wird das Startlayout kurz trainiert und auf Validation-Daten bewertet.</p>"
            if self.preferences.language == "de"
            else "<h2>No start evaluation yet</h2><p>Click <b>Evaluate start</b>. "
            "The start layout will be trained briefly and evaluated on validation data.</p>"
        )

    def _evaluation_result_html(self, evaluation) -> str:
        if self.preferences.language == "en":
            return (
                "<h2>What happened?</h2>"
                "<p>A fresh model was built from this activation layout, trained for a small "
                "candidate budget, and evaluated on validation data.</p>"
                "<p><b>validation_loss</b> is the optimization score here. Lower is better.</p>"
                "<p>If we optimized <b>validation_accuracy</b>, the service would convert it into a comparable "
                "minimization score such as <code>1 - val_accuracy</code>. The displayed accuracy remains the intuitive "
                "classification metric.</p>"
            )
        return (
            "<h2>Was ist passiert?</h2>"
            "<p>Aus diesem Aktivierungs-Layout wurde ein frisches Modell gebaut, kurz "
            "trainiert und auf Validation-Daten bewertet.</p>"
            "<p><b>validation_loss</b> ist hier der Optimierungsscore. Kleiner ist besser.</p>"
            "<p>Wuerden wir <b>validation_accuracy</b> optimieren, wuerde der Service daraus intern "
            "einen vergleichbaren Minimierungswert wie <code>1 - val_accuracy</code> machen. "
            "Die angezeigte Accuracy bleibt trotzdem die intuitive Klassifikationsmetrik.</p>"
        )
