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
            f"Mode: {getattr(snapshot, 'sa_evaluation_mode', 'short_retrain')}",
            f"{'Temperature' if self._language == 'en' else 'Temperatur'}: {snapshot.current_temperature:.4f}",
            f"{'Accepted steps' if self._language == 'en' else 'Akzeptierte Schritte'}: {snapshot.accepted_steps}",
            f"{'Acceptance rate' if self._language == 'en' else 'Akzeptanzrate'}: {snapshot.acceptance_rate:.3f}",
        ]
        if getattr(snapshot, "sa_evaluation_mode", "") == "online_delta":
            lines.extend(
                [
                    f"{'Epoch' if self._language == 'en' else 'Epoche'}: {getattr(snapshot, 'epoch_index', 0)}",
                    f"{'Batch' if self._language == 'en' else 'Batch'}: {getattr(snapshot, 'batch_index', 0)}",
                    f"{'Batch start' if self._language == 'en' else 'Batch-Start'}: {getattr(snapshot, 'batch_start', 0)}",
                ]
            )
        if snapshot.current_evaluation is not None:
            online_delta = getattr(snapshot, "sa_evaluation_mode", "") == "online_delta"
            current_label = (
                "Current batch loss"
                if self._language == "en" and online_delta
                else "Aktueller Batch-Loss"
                if online_delta
                else "Current objective"
                if self._language == "en"
                else "Aktuelles Objective"
            )
            lines.extend(
                [
                    "",
                    f"{'Current layout' if self._language == 'en' else 'Aktuelles Layout'}: {snapshot.current_evaluation.layout.to_compact_spec()}",
                    f"{current_label}: {snapshot.current_evaluation.objective_value:.4f}",
                    f"val_loss: {snapshot.current_evaluation.val_loss:.4f}",
                ]
            )
        if snapshot.best_evaluation is not None:
            online_delta = getattr(snapshot, "sa_evaluation_mode", "") == "online_delta"
            best_label = (
                "Best validation loss"
                if self._language == "en" and online_delta
                else "Bester Validation-Loss"
                if online_delta
                else "Best objective"
                if self._language == "en"
                else "Bestes Objective"
            )
            lines.extend(
                [
                    f"{'Best layout' if self._language == 'en' else 'Bestes Layout'}: {snapshot.best_evaluation.layout.to_compact_spec()}",
                    f"{best_label}: {snapshot.best_evaluation.val_loss:.4f}",
                ]
            )
        self.text.setPlainText("\n".join(lines))
