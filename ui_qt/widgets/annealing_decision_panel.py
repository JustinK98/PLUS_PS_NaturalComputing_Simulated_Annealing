"""Didaktische Entscheidungsansicht fuer SA-Schritte."""

from __future__ import annotations

from PySide6 import QtWidgets

from services.annealing_session_service import AnnealingSessionSnapshot


class AnnealingDecisionPanel(QtWidgets.QWidget):
    def __init__(self, language: str = "de", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._language = language
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.text = QtWidgets.QTextBrowser()
        layout.addWidget(self.text, 1)
        self.set_placeholder(language)

    def set_language(self, language: str) -> None:
        self._language = language

    def set_placeholder(self, language: str | None = None) -> None:
        if language is not None:
            self._language = language
        self.text.setPlainText(
            "No annealing decision available yet."
            if self._language == "en"
            else "Noch keine Annealing-Entscheidung verfuegbar."
        )

    def set_snapshot(self, snapshot: AnnealingSessionSnapshot, language: str | None = None) -> None:
        if language is not None:
            self._language = language
        if snapshot.last_step is None:
            self.set_placeholder()
            return
        step = snapshot.last_step
        lines = [
            f"{'Decision' if self._language == 'en' else 'Entscheidung'}: {snapshot.decision_label}",
            f"{'Neighbor operation' if self._language == 'en' else 'Nachbarschaftsoperation'}: {step.neighbor_label}",
            f"{'Delta' if self._language == 'en' else 'Delta'}: {step.delta:+.4f}",
            f"{'Acceptance probability' if self._language == 'en' else 'Akzeptanzwahrscheinlichkeit'}: {step.acceptance_probability:.4f}",
            f"{'Accepted' if self._language == 'en' else 'Akzeptiert'}: {step.accepted}",
            "",
            f"{'Previous layout' if self._language == 'en' else 'Vorheriges Layout'}: {step.previous_layout.to_compact_spec()}",
            f"{'Candidate layout' if self._language == 'en' else 'Kandidaten-Layout'}: {step.candidate_layout.to_compact_spec()}",
            f"{'Previous score' if self._language == 'en' else 'Vorheriger Score'}: {step.previous_evaluation.objective_value:.4f}",
            f"{'Candidate score' if self._language == 'en' else 'Kandidaten-Score'}: {step.candidate_evaluation.objective_value:.4f}",
        ]
        self.text.setPlainText("\n".join(lines))
