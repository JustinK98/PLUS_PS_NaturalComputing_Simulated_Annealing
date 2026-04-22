"""Kompakte Widgets fuer den gefuehrten Presentation Mode."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from configs import SUPPORTED_ACTIVATIONS

from ui_qt.presentation import PresentationSlide, track_title
from ui_qt.widgets.network_view import NetworkViewWidget


class FormulaCardWidget(QtWidgets.QFrame):
    """Grosse Formelkarte fuer eine einzelne Kernformel."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("CardFrame")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        self.label = QtWidgets.QLabel()
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setTextFormat(QtCore.Qt.PlainText)
        font = self.label.font()
        font.setFamily("Georgia")
        font.setPointSize(font.pointSize() + 9)
        font.setBold(True)
        self.label.setFont(font)
        layout.addWidget(self.label)
        self.set_formula("")

    def set_formula(self, formula: str) -> None:
        self.setVisible(bool(formula))
        self.label.setText(formula)


class ConceptChoiceWidget(QtWidgets.QWidget):
    """Startscreen mit zwei grossen Track-Auswahlen."""

    trackSelected = QtCore.Signal(str)

    def __init__(self, language: str = "de", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._language = language
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(48, 48, 48, 48)
        layout.setSpacing(24)

        self.title_label = QtWidgets.QLabel()
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_font = self.title_label.font()
        title_font.setPointSize(title_font.pointSize() + 14)
        title_font.setBold(True)
        self.title_label.setFont(title_font)

        self.subtitle_label = QtWidgets.QLabel()
        self.subtitle_label.setAlignment(QtCore.Qt.AlignCenter)
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setProperty("role", "muted")

        button_row = QtWidgets.QHBoxLayout()
        button_row.setSpacing(24)
        self.network_button = self._make_track_button()
        self.annealing_button = self._make_track_button()
        self.network_button.clicked.connect(lambda: self.trackSelected.emit("neural_network"))
        self.annealing_button.clicked.connect(lambda: self.trackSelected.emit("simulated_annealing"))
        button_row.addWidget(self.network_button)
        button_row.addWidget(self.annealing_button)

        layout.addStretch(1)
        layout.addWidget(self.title_label)
        layout.addWidget(self.subtitle_label)
        layout.addLayout(button_row)
        layout.addStretch(1)
        self.set_language(language)

    def set_language(self, language: str) -> None:
        self._language = language
        if language == "en":
            self.title_label.setText("Presentation Mode")
            self.subtitle_label.setText(
                "Choose one guided 5-7 minute story. Each step shows one concept with one visual."
            )
            self.network_button.setText("Neural Network Preview\n\nUnderstand one concrete model")
            self.annealing_button.setText("Simulated Annealing Preview\n\nUnderstand the layout search")
        else:
            self.title_label.setText("Presentation Mode")
            self.subtitle_label.setText(
                "Waehle eine gefuehrte 5-7-Minuten-Story. Jeder Schritt zeigt genau ein Konzept mit einem Visual."
            )
            self.network_button.setText("Neural Network Preview\n\nEin konkretes Modell verstehen")
            self.annealing_button.setText("Simulated Annealing Preview\n\nDie Layout-Suche verstehen")

    def _make_track_button(self) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton()
        button.setMinimumHeight(180)
        button.setProperty("role", "primary")
        font = button.font()
        font.setPointSize(font.pointSize() + 4)
        font.setBold(True)
        button.setFont(font)
        return button


class PresentationSlideWidget(QtWidgets.QWidget):
    """Slide-Huelle mit Visual links und Erklaerung rechts."""

    actionRequested = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(16)

        self.track_label = QtWidgets.QLabel()
        self.track_label.setProperty("role", "muted")
        self.title_label = QtWidgets.QLabel()
        title_font = self.title_label.font()
        title_font.setPointSize(title_font.pointSize() + 14)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet("color: #0f172a;")
        self.question_label = QtWidgets.QLabel()
        self.question_label.setWordWrap(True)
        self.question_label.setProperty("role", "muted")

        header = QtWidgets.QVBoxLayout()
        header.addWidget(self.track_label)
        header.addWidget(self.title_label)
        header.addWidget(self.question_label)
        root.addLayout(header)

        content = QtWidgets.QHBoxLayout()
        content.setSpacing(20)
        self.visual_host = QtWidgets.QStackedWidget()
        self.visual_host.setMinimumWidth(780)
        content.addWidget(self.visual_host, 3)

        self.explanation = QtWidgets.QFrame()
        self.explanation.setObjectName("CardFrame")
        explanation_layout = QtWidgets.QVBoxLayout(self.explanation)
        explanation_layout.setContentsMargins(18, 18, 18, 18)
        self.panel_kind_label = QtWidgets.QLabel()
        self.panel_kind_label.setProperty("role", "muted")
        explanation_layout.addWidget(self.panel_kind_label)
        self.bullets_label = QtWidgets.QLabel()
        self.bullets_label.setWordWrap(True)
        self.bullets_label.setTextFormat(QtCore.Qt.RichText)
        bullet_font = self.bullets_label.font()
        bullet_font.setPointSize(bullet_font.pointSize() + 4)
        self.bullets_label.setFont(bullet_font)
        self.formula_card = FormulaCardWidget()
        explanation_layout.addWidget(self.bullets_label)
        explanation_layout.addWidget(self.formula_card)
        explanation_layout.addStretch(1)
        content.addWidget(self.explanation, 1)
        root.addLayout(content, 1)

        navigation = QtWidgets.QHBoxLayout()
        self.back_button = QtWidgets.QPushButton()
        self.next_button = QtWidgets.QPushButton()
        self.reset_button = QtWidgets.QPushButton()
        self.restart_button = QtWidgets.QPushButton()
        self.action_button = QtWidgets.QPushButton()
        self.action_button.setProperty("role", "primary")
        self.extra_actions = QtWidgets.QWidget()
        self.extra_actions_layout = QtWidgets.QHBoxLayout(self.extra_actions)
        self.extra_actions_layout.setContentsMargins(0, 0, 0, 0)
        self.extra_actions_layout.setSpacing(8)
        self._extra_action_buttons: list[QtWidgets.QPushButton] = []
        navigation.addWidget(self.back_button)
        navigation.addWidget(self.next_button)
        navigation.addStretch(1)
        navigation.addWidget(self.restart_button)
        navigation.addWidget(self.reset_button)
        navigation.addWidget(self.extra_actions)
        navigation.addWidget(self.action_button)
        root.addLayout(navigation)

    def set_slide(
        self,
        slide: PresentationSlide,
        *,
        slide_index: int,
        total_slides: int,
        language: str,
    ) -> None:
        self.track_label.setText(
            f"{track_title(slide.track_id, language)} | Step {slide_index + 1}/{total_slides}"
        )
        self.title_label.setText(slide.title(language))
        self.question_label.setText(slide.question(language))
        self.bullets_label.setText(
            "<ul>"
            + "".join(f"<li>{bullet}</li>" for bullet in slide.bullets(language))
            + "</ul>"
        )
        self.panel_kind_label.setText(self._panel_label(slide.right_panel, language))
        self.explanation.setVisible(slide.right_panel != "none")
        self.formula_card.set_formula(slide.formula)
        if language == "en":
            self.back_button.setText("Back")
            self.next_button.setText("Next")
            self.restart_button.setText("Reset track")
            self.reset_button.setText("Back to tracks")
        else:
            self.back_button.setText("Zurueck")
            self.next_button.setText("Weiter")
            self.restart_button.setText("Track resetten")
            self.reset_button.setText("Zur Track-Auswahl")

    def set_action(self, label: str | None) -> None:
        self.action_button.setVisible(label is not None)
        if label is not None:
            self.action_button.setText(label)

    def set_extra_actions(self, actions: list[tuple[str, str]]) -> None:
        for button in self._extra_action_buttons:
            button.setParent(None)
        self._extra_action_buttons.clear()
        self.extra_actions.setVisible(bool(actions))
        for action, label in actions:
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(lambda checked=False, name=action: self.actionRequested.emit(name))
            self.extra_actions_layout.addWidget(button)
            self._extra_action_buttons.append(button)

    def set_actions_enabled(self, enabled: bool) -> None:
        self.action_button.setEnabled(enabled)
        for button in self._extra_action_buttons:
            button.setEnabled(enabled)

    def _panel_label(self, right_panel: str, language: str) -> str:
        labels_de = {
            "intro": "Einordnung",
            "definitions": "Begriffe",
            "network": "Struktur",
            "activation": "Signalformung",
            "calculation": "Lokale Rechnung",
            "training": "Training",
            "transition": "Leitfrage",
            "sa_intro": "Seminarbezug",
            "layout_compare": "Layout als Zustand",
            "neighbor": "Kandidat",
            "evaluation": "Bewertung",
            "decision": "Entscheidung",
            "plot_legend": "Plot lesen",
            "summary": "Ergebnis",
            "final": "Takeaway",
            "none": "",
        }
        labels_en = {
            "intro": "Context",
            "definitions": "Definitions",
            "network": "Structure",
            "activation": "Signal shaping",
            "calculation": "Local calculation",
            "training": "Training",
            "transition": "Guiding question",
            "sa_intro": "Seminar context",
            "layout_compare": "Layout as state",
            "neighbor": "Candidate",
            "evaluation": "Evaluation",
            "decision": "Decision",
            "plot_legend": "Reading the plot",
            "summary": "Result",
            "final": "Takeaway",
            "none": "",
        }
        return (labels_en if language == "en" else labels_de).get(right_panel, right_panel)


class ActivationLabWidget(QtWidgets.QWidget):
    """Kleine Aktivierungsdemo fuer genau ein Beispielneuron."""

    selectionChanged = QtCore.Signal(int, int)
    activationChanged = QtCore.Signal(int, int, str)

    def __init__(
        self,
        hidden_sizes: tuple[int, ...],
        language: str = "de",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._language = language
        self._hidden_sizes = tuple(hidden_sizes)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        controls = QtWidgets.QHBoxLayout()
        self.layer_combo = QtWidgets.QComboBox()
        self.neuron_combo = QtWidgets.QComboBox()
        self.activation_combo = QtWidgets.QComboBox()
        self.activation_combo.addItems(SUPPORTED_ACTIVATIONS)
        self.layer_combo.currentIndexChanged.connect(self._on_layer_changed)
        self.neuron_combo.currentIndexChanged.connect(self._emit_selection)
        self.activation_combo.currentTextChanged.connect(self._emit_activation)
        controls.addWidget(QtWidgets.QLabel("Layer"))
        controls.addWidget(self.layer_combo)
        controls.addWidget(QtWidgets.QLabel("Neuron"))
        controls.addWidget(self.neuron_combo)
        controls.addStretch(1)
        controls.addWidget(QtWidgets.QLabel("Activation"))
        controls.addWidget(self.activation_combo)

        self.explain_label = QtWidgets.QLabel()
        self.explain_label.setWordWrap(True)
        self.explain_label.setProperty("role", "muted")
        layout.addLayout(controls)
        layout.addWidget(self.explain_label)
        self._populate_layers()
        self.set_language(language)

    def set_language(self, language: str) -> None:
        self._language = language
        self.explain_label.setText(
            "Waehle eine Aktivierung fuer das Beispielneuron. Die Kurve und die lokale Rechnung reagieren sofort."
            if language == "de"
            else "Choose an activation for the example neuron. The curve and local calculation update immediately."
        )

    def set_activation(self, activation_name: str) -> None:
        self.activation_combo.blockSignals(True)
        self.activation_combo.setCurrentText(activation_name)
        self.activation_combo.blockSignals(False)

    def set_selection(self, layer_index: int, neuron_index: int, activation_name: str) -> None:
        self.layer_combo.blockSignals(True)
        self.neuron_combo.blockSignals(True)
        self.activation_combo.blockSignals(True)
        self.layer_combo.setCurrentIndex(layer_index)
        self._populate_neurons(layer_index)
        self.neuron_combo.setCurrentIndex(neuron_index)
        self.activation_combo.setCurrentText(activation_name)
        self.activation_combo.blockSignals(False)
        self.neuron_combo.blockSignals(False)
        self.layer_combo.blockSignals(False)

    def _populate_layers(self) -> None:
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        for index in range(len(self._hidden_sizes)):
            self.layer_combo.addItem(f"L{index + 1}", index)
        self.layer_combo.blockSignals(False)
        self._populate_neurons(0)

    def _populate_neurons(self, layer_index: int) -> None:
        self.neuron_combo.clear()
        for neuron_index in range(self._hidden_sizes[layer_index]):
            self.neuron_combo.addItem(f"n{neuron_index}", neuron_index)

    def _on_layer_changed(self, layer_index: int) -> None:
        self.neuron_combo.blockSignals(True)
        self._populate_neurons(max(0, layer_index))
        self.neuron_combo.blockSignals(False)
        self._emit_selection()

    def _emit_selection(self) -> None:
        layer_index = max(0, self.layer_combo.currentIndex())
        neuron_index = max(0, self.neuron_combo.currentIndex())
        self.selectionChanged.emit(layer_index, neuron_index)

    def _emit_activation(self, activation_name: str) -> None:
        layer_index = max(0, self.layer_combo.currentIndex())
        neuron_index = max(0, self.neuron_combo.currentIndex())
        self.activationChanged.emit(layer_index, neuron_index, activation_name)


class MetricsCardRow(QtWidgets.QWidget):
    """Kompakte Kennzahlenreihe fuer Vorher/Nachher und SA-Zustaende."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QtWidgets.QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(10)
        self._cards: list[QtWidgets.QFrame] = []

    def set_cards(self, cards: list[tuple[str, str, str]]) -> None:
        for card in self._cards:
            card.setParent(None)
        self._cards.clear()
        for title, value, body in cards:
            frame = QtWidgets.QFrame()
            frame.setObjectName("CardFrame")
            layout = QtWidgets.QVBoxLayout(frame)
            layout.setContentsMargins(12, 10, 12, 10)
            title_label = QtWidgets.QLabel(title)
            title_label.setProperty("role", "muted")
            value_label = QtWidgets.QLabel(value)
            font = value_label.font()
            font.setBold(True)
            font.setPointSize(font.pointSize() + 3)
            value_label.setFont(font)
            body_label = QtWidgets.QLabel(body)
            body_label.setWordWrap(True)
            body_label.setProperty("role", "muted")
            layout.addWidget(title_label)
            layout.addWidget(value_label)
            layout.addWidget(body_label)
            self._layout.addWidget(frame)
            self._cards.append(frame)


class NetworkComparisonWidget(QtWidgets.QWidget):
    """Zwei Netzwerkansichten nebeneinander fuer Vorher/Nachher-Folien."""

    def __init__(self, language: str = "de", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._language = language
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.title = QtWidgets.QLabel()
        self.title.setStyleSheet("font-size: 22px; font-weight: 700;")
        root.addWidget(self.title)
        body = QtWidgets.QHBoxLayout()
        body.setSpacing(12)
        self.left_view = NetworkViewWidget()
        self.right_view = NetworkViewWidget()
        self.left_label = QtWidgets.QLabel()
        self.right_label = QtWidgets.QLabel()
        body.addWidget(self._wrap_view(self.left_label, self.left_view))
        body.addWidget(self._wrap_view(self.right_label, self.right_view))
        root.addLayout(body, 1)
        self.text = QtWidgets.QTextBrowser()
        self.text.setMaximumHeight(220)
        self.text.setStyleSheet("font-size: 20px; line-height: 1.45;")
        root.addWidget(self.text)
        self.set_language(language)

    def set_language(self, language: str) -> None:
        self._language = language
        self.title.setText("Netzwerkvergleich" if language == "de" else "Network comparison")

    def set_models(
        self,
        *,
        left_title: str,
        right_title: str,
        left_model,
        right_model,
        left_projection,
        right_projection,
        left_selection: tuple[int, int] | None = None,
        right_selection: tuple[int, int] | None = None,
        description_html: str = "",
    ) -> None:
        self.left_label.setText(left_title)
        self.right_label.setText(right_title)
        self.left_view.set_input_projection(left_projection)
        self.left_view.set_model(left_model)
        self.left_view.set_selected_hidden(left_selection)
        self.right_view.set_input_projection(right_projection)
        self.right_view.set_model(right_model)
        self.right_view.set_selected_hidden(right_selection)
        self.text.setHtml(f'<div style="font-size: 20px; line-height: 1.45;">{description_html}</div>')

    def _wrap_view(self, label: QtWidgets.QLabel, view: NetworkViewWidget) -> QtWidgets.QWidget:
        container = QtWidgets.QFrame()
        container.setObjectName("CardFrame")
        layout = QtWidgets.QVBoxLayout(container)
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setStyleSheet("font-size: 18px; font-weight: 700;")
        layout.addWidget(label)
        layout.addWidget(view, 1)
        return container


class ReadingGuideWidget(QtWidgets.QFrame):
    """Kompakte Lesekarte fuer Plots, Score-Karten und Live-Demo-Schritte."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("CardFrame")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        self.title = QtWidgets.QLabel()
        self.title.setStyleSheet("font-weight: 700;")
        self.body = QtWidgets.QLabel()
        self.body.setWordWrap(True)
        self.body.setTextFormat(QtCore.Qt.RichText)
        self.body.setProperty("role", "muted")
        layout.addWidget(self.title)
        layout.addWidget(self.body)

    def set_content(self, title: str, bullets: tuple[str, ...]) -> None:
        self.title.setText(title)
        self.body.setText("<ul>" + "".join(f"<li>{bullet}</li>" for bullet in bullets) + "</ul>")
