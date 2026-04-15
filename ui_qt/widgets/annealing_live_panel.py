"""Live-Zusammenfassung fuer interaktive SA-Sessions."""

from __future__ import annotations

from PySide6 import QtWidgets

from services.annealing_session_service import AnnealingSessionSnapshot


class AnnealingLivePanel(QtWidgets.QWidget):
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
            "No annealing session started yet."
            if self._language == "en"
            else "Noch keine Annealing-Session gestartet."
        )

    def set_snapshot(self, snapshot: AnnealingSessionSnapshot, language: str | None = None) -> None:
        if language is not None:
            self._language = language
        lines = [
            f"{'State' if self._language == 'en' else 'Zustand'}: {snapshot.state_label}",
            f"{'Decision' if self._language == 'en' else 'Entscheidung'}: {snapshot.decision_label}",
            f"{'Reason' if self._language == 'en' else 'Grund'}: {snapshot.reason_label}",
            f"{'Temperature' if self._language == 'en' else 'Temperatur'}: {snapshot.current_temperature:.4f}",
            f"{'Accepted steps' if self._language == 'en' else 'Akzeptierte Schritte'}: {snapshot.accepted_steps}",
            f"{'Acceptance rate' if self._language == 'en' else 'Akzeptanzrate'}: {snapshot.acceptance_rate:.3f}",
        ]
        if snapshot.current_evaluation is not None:
            lines.extend(
                [
                    "",
                    f"{'Current layout' if self._language == 'en' else 'Aktuelles Layout'}: {snapshot.current_evaluation.layout.to_compact_spec()}",
                    f"{'Current objective' if self._language == 'en' else 'Aktuelles Objective'}: {snapshot.current_evaluation.objective_value:.4f}",
                ]
            )
        if snapshot.best_evaluation is not None:
            lines.extend(
                [
                    f"{'Best layout' if self._language == 'en' else 'Bestes Layout'}: {snapshot.best_evaluation.layout.to_compact_spec()}",
                    f"{'Best objective' if self._language == 'en' else 'Bestes Objective'}: {snapshot.best_evaluation.objective_value:.4f}",
                ]
            )
        self.text.setPlainText("\n".join(lines))
