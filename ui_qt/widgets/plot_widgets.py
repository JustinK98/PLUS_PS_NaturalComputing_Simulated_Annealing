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
        history = list(getattr(state, "history", []) if state is not None else [])
        if history and hasattr(history[0], "batch_loss_before"):
            ax_batch = self.figure.add_subplot(311)
            ax_val = self.figure.add_subplot(312)
            ax_temp = self.figure.add_subplot(313)
            steps = [step.step_index for step in history]
            ax_batch.plot(steps, [step.batch_loss_before for step in history], label="before")
            ax_batch.plot(steps, [step.candidate_loss_after for step in history], label="candidate")
            ax_val.plot(steps, [step.validation_loss_after_update for step in history], label="current val")
            ax_val.plot(steps, [step.best_score_after_step for step in history], label="best val")
            ax_temp.plot(steps, [step.temperature for step in history], label="temperature", color="#d97706")
            accepted_steps = [step.step_index for step in history if step.accepted]
            accepted_temps = [step.temperature for step in history if step.accepted]
            if accepted_steps:
                ax_temp.scatter(accepted_steps, accepted_temps, label="accepted", color="#16a34a", s=14)
            ax_batch.set_title("Online Delta Batch Loss")
            ax_val.set_title("Validation Loss During Search")
            ax_temp.set_title("Temperature")
            ax_batch.legend(loc="best")
            ax_val.legend(loc="best")
            ax_temp.legend(loc="best")
            self.canvas.draw_idle()
            return

        ax_score = self.figure.add_subplot(211)
        ax_temp = self.figure.add_subplot(212)
        if history:
            steps = [step.step_index for step in history]
            best_scores = [step.best_score_after_step for step in history]
            candidate_scores = [step.candidate_evaluation.comparable_score for step in history]
            temperatures = [step.temperature for step in history]
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
