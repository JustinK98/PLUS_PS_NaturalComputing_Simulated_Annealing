"""Didaktisches Detailpanel fuer selektierte Hidden-Neuronen."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from services.neuron_analysis_service import NeuronAnalysisPayload

from .activation_curve_widget import ActivationCurveWidget


class NeuronDetailPanel(QtWidgets.QWidget):
    def __init__(self, language: str = "de", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._language = language
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.header_label = QtWidgets.QLabel()
        self.header_label.setStyleSheet("font-size: 15px; font-weight: 600; color: #111827;")
        self.meta_label = QtWidgets.QLabel()
        self.meta_label.setWordWrap(True)
        self.meta_label.setStyleSheet("color: #475569;")
        self.detail_text = QtWidgets.QTextBrowser()
        self.detail_text.setOpenExternalLinks(False)
        self.detail_text.setMinimumHeight(180)
        self.curve_widget = ActivationCurveWidget()
        self.curve_widget.setMinimumHeight(220)

        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        split.setChildrenCollapsible(False)
        split.addWidget(self.detail_text)
        split.addWidget(self.curve_widget)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 2)

        layout.addWidget(self.header_label)
        layout.addWidget(self.meta_label)
        layout.addWidget(split, 1)
        self.set_placeholder(language)

    def set_language(self, language: str) -> None:
        self._language = language

    def set_placeholder(self, language: str | None = None) -> None:
        if language is not None:
            self._language = language
        message = (
            "Select a hidden neuron in the network view."
            if self._language == "en"
            else "Waehle ein Hidden-Neuron in der Netzwerkansicht aus."
        )
        self.header_label.setText(
            "Neuron Tracker" if self._language == "en" else "Neuron Tracker"
        )
        self.meta_label.setText(message)
        self.detail_text.setPlainText(message)
        self.curve_widget.set_payload(None)

    def set_payload(self, payload: NeuronAnalysisPayload, language: str | None = None) -> None:
        if language is not None:
            self._language = language
        inspection = payload.inspection
        self.header_label.setText(payload.title)
        meta = (
            f"{inspection.activation_name} | z={inspection.pre_activation:+.4f} | "
            f"a={inspection.output_value:+.4f} | d={inspection.derivative:+.4f}"
        )
        self.meta_label.setText(meta)

        if self._language == "en":
            sections = [
                f"Activation: {inspection.activation_name}",
                f"Formula: {payload.formula_label}",
                f"Sample source: {payload.source_label}",
                f"True target: {payload.actual_target_label}",
                f"Analysis target: {payload.analysis_target_label}",
                "",
                "Step 1: weighted sum",
                "z = sum_i (input_i * weight_i) + bias",
                f"Compact computation: {payload.equation_text}",
                f"bias = {inspection.bias:+.6f}",
                f"z    = {inspection.pre_activation:+.6f}",
                "",
                "Step 2: activation",
                f"a = {inspection.activation_name}(z) = {inspection.output_value:+.6f}",
                f"Derivative at this point = {inspection.derivative:+.6f}",
                "",
                "Strongest terms in z",
            ]
            sections.extend(
                f"{rank:02d}. {term.source_label}: value={term.source_value:+.6f}, "
                f"weight={term.weight:+.6f}, contribution={term.contribution:+.6f}"
                for rank, term in enumerate(payload.top_terms, start=1)
            )
            sections.extend(["", "Strongest outgoing weights"])
            sections.extend(f"{label}: weight={weight:+.6f}" for label, weight in payload.outgoing_weights)
        else:
            sections = [
                f"Aktivierung: {inspection.activation_name}",
                f"Formel: {payload.formula_label}",
                f"Quelle des Samples: {payload.source_label}",
                f"Echtes Ziel: {payload.actual_target_label}",
                f"Analyse-Ziel: {payload.analysis_target_label}",
                "",
                "Schritt 1: gewichtete Summe",
                "z = sum_i (eingang_i * gewicht_i) + bias",
                f"Kompakte Rechnung: {payload.equation_text}",
                f"bias = {inspection.bias:+.6f}",
                f"z    = {inspection.pre_activation:+.6f}",
                "",
                "Schritt 2: Aktivierung",
                f"a = {inspection.activation_name}(z) = {inspection.output_value:+.6f}",
                f"Ableitung an dieser Stelle = {inspection.derivative:+.6f}",
                "",
                "Staerkste Summanden von z",
            ]
            sections.extend(
                f"{rank:02d}. {term.source_label}: wert={term.source_value:+.6f}, "
                f"gewicht={term.weight:+.6f}, beitrag={term.contribution:+.6f}"
                for rank, term in enumerate(payload.top_terms, start=1)
            )
            sections.extend(["", "Staerkste ausgehende Gewichte"])
            sections.extend(f"{label}: gewicht={weight:+.6f}" for label, weight in payload.outgoing_weights)

        sections.extend(["", "Interpretation"])
        sections.extend(payload.interpretation_lines)
        self.detail_text.setPlainText("\n".join(sections))
        self.curve_widget.set_payload(payload)
