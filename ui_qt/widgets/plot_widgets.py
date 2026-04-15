"""Matplotlib-basierte Plot-Widgets fuer die Qt-GUI."""

from __future__ import annotations

from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6 import QtWidgets


class _BasePlotWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(6, 4), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)


class TrainingPlotWidget(_BasePlotWidget):
    def set_history(self, history: dict[str, list[float]] | None) -> None:
        self.figure.clear()
        ax_loss = self.figure.add_subplot(211)
        ax_acc = self.figure.add_subplot(212)
        if history:
            ax_loss.plot(history.get("train_loss", []), label="train_loss")
            ax_loss.plot(history.get("val_loss", []), label="val_loss")
            ax_acc.plot(history.get("train_acc", []), label="train_acc")
            ax_acc.plot(history.get("val_acc", []), label="val_acc")
            ax_loss.legend(loc="best")
            ax_acc.legend(loc="best")
        ax_loss.set_title("Loss")
        ax_acc.set_title("Accuracy")
        self.canvas.draw_idle()


class AnnealingPlotWidget(_BasePlotWidget):
    def set_state(self, state: Any | None) -> None:
        self.figure.clear()
        ax_score = self.figure.add_subplot(211)
        ax_temp = self.figure.add_subplot(212)
        if state is not None and getattr(state, "history", None):
            steps = [step.step_index for step in state.history]
            best_scores = [step.best_score_after_step for step in state.history]
            candidate_scores = [step.candidate_evaluation.comparable_score for step in state.history]
            temperatures = [step.temperature for step in state.history]
            ax_score.plot(steps, candidate_scores, label="candidate")
            ax_score.plot(steps, best_scores, label="best")
            ax_score.legend(loc="best")
            ax_temp.plot(steps, temperatures, label="temperature", color="#d97706")
        ax_score.set_title("Annealing Score")
        ax_temp.set_title("Temperature")
        self.canvas.draw_idle()


class BuilderSummaryPlotWidget(_BasePlotWidget):
    def set_summary(self, summary_payload: dict[str, Any] | None) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if summary_payload:
            ranking = summary_payload.get("ranking", [])
            labels = [entry["config_id"] for entry in ranking[:8]]
            scores = [float(entry["ranking_score"]) for entry in ranking[:8]]
            if labels:
                ax.bar(labels, scores, color="#2563eb")
                ax.set_ylabel("Ranking score")
        ax.set_title("Configuration Ranking")
        self.canvas.draw_idle()
