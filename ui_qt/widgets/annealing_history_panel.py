"""History-Panel fuer SA inklusive Plot und Schrittprotokoll."""

from __future__ import annotations

from PySide6 import QtWidgets

from services.annealing_session_service import AnnealingSessionSnapshot

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

    def set_snapshot(self, snapshot: AnnealingSessionSnapshot, language: str | None = None) -> None:
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
            },
        )()
        self.plot.set_state(state_like)
        lines = []
        for step in snapshot.history[-8:]:
            lines.append(
                (
                    f"step={step.step_index} T={step.temperature:.3f} accepted={step.accepted} "
                    f"delta={step.delta:+.4f} p={step.acceptance_probability:.4f}"
                )
            )
        self.text.setPlainText(
            "\n".join(lines)
            if lines
            else ("Start evaluated." if self._language == "en" else "Start bewertet.")
        )
