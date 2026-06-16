"""History-Panel fuer SA inklusive Plot und Schrittprotokoll."""

from __future__ import annotations

from PySide6 import QtWidgets

from services.online_annealing_training_service import OnlineAnnealingSnapshot

from .plot_widgets import AnnealingPlotWidget


class AnnealingHistoryPanel(QtWidgets.QWidget):
    def __init__(self, language: str = "de", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._language = language
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = AnnealingPlotWidget()
        self.text = QtWidgets.QTextBrowser()
        self.text.setMinimumHeight(140)
        layout.addWidget(self.plot, 2)
        layout.addWidget(self.text, 1)
        self.set_placeholder(language)

    def set_language(self, language: str) -> None:
        self._language = language

    def set_placeholder(self, language: str | None = None) -> None:
        if language is not None:
            self._language = language
        self.plot.set_state(None)
        self.text.setPlainText(
            "No annealing history yet." if self._language == "en" else "Noch keine Annealing-History."
        )

    def set_snapshot(self, snapshot: OnlineAnnealingSnapshot, language: str | None = None) -> None:
        if language is not None:
            self._language = language
        if not snapshot.is_initialized:
            self.set_placeholder()
            return

        state_like = type(
            "HistoryState",
            (),
            {
                "history": list(snapshot.history),
                "start_evaluation": snapshot.start_evaluation,
                "start_temperature": (
                    snapshot.history[0].temperature
                    if snapshot.history
                    else snapshot.current_temperature
                ),
            },
        )()
        self.plot.set_state(state_like)
        lines = []
        for step in snapshot.history[-8:]:
            lines.append(
                (
                    f"step={step.step_index} epoch={step.epoch_index} batch={step.batch_index} "
                    f"accepted={step.accepted} delta={step.delta:+.4f} "
                    f"before={step.batch_loss_before:.4f} after={step.candidate_loss_after:.4f} "
                    f"trained={step.trained_after_accept} updates={step.trained_batch_updates_after_step} "
                    f"online_epochs={step.effective_online_epochs_after_step:.3f}"
                )
            )
        self.text.setPlainText(
            "\n".join(lines)
            if lines
            else ("Start evaluated." if self._language == "en" else "Start bewertet.")
        )
