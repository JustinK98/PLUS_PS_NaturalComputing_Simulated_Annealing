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
            ax_loss.plot(history.get("train_loss", []), label="Train-Loss")
            ax_loss.plot(history.get("val_loss", []), label="Val-Loss")
            ax_acc.plot(history.get("train_acc", []), label="Train-Accuracy")
            ax_acc.plot(history.get("val_acc", []), label="Val-Accuracy")
            ax_loss.legend(loc="best")
            ax_acc.legend(loc="best")
        ax_loss.set_title("Loss")
        ax_acc.set_title("Accuracy")
        self.canvas.draw_idle()


class AnnealingPlotWidget(_BasePlotWidget):
    def set_state(self, state: Any | None) -> None:
        self.figure.clear()
        history = list(getattr(state, "history", []) if state is not None else [])
        start_evaluation = getattr(state, "start_evaluation", None)
        start_temperature = getattr(state, "start_temperature", None)
        ax_batch = self.figure.add_subplot(311)
        ax_val = self.figure.add_subplot(312)
        ax_temp = self.figure.add_subplot(313)
        steps = [step.step_index for step in history]
        if start_evaluation is not None:
            batch_steps = [0, *steps]
            before_losses = [
                start_evaluation.objective_value,
                *(step.batch_loss_before for step in history),
            ]
            candidate_losses = [
                start_evaluation.objective_value,
                *(step.candidate_loss_after for step in history),
            ]
            validation_steps = [0, *steps]
            current_validation_losses = [
                start_evaluation.val_loss,
                *(step.validation_loss_after_update for step in history),
            ]
            best_validation_losses = [
                start_evaluation.val_loss,
                *(step.best_score_after_step for step in history),
            ]
        else:
            batch_steps = steps
            before_losses = [step.batch_loss_before for step in history]
            candidate_losses = [step.candidate_loss_after for step in history]
            validation_steps = steps
            current_validation_losses = [step.validation_loss_after_update for step in history]
            best_validation_losses = [step.best_score_after_step for step in history]

        if batch_steps:
            ax_batch.plot(batch_steps, before_losses, marker="o", markersize=3, label="vorher")
            ax_batch.plot(batch_steps, candidate_losses, marker="o", markersize=3, label="Kandidat")
            ax_val.plot(
                validation_steps,
                current_validation_losses,
                marker="o",
                markersize=3,
                label="aktueller Val-Loss",
            )
            ax_val.plot(
                validation_steps,
                best_validation_losses,
                marker="o",
                markersize=3,
                label="bester Val-Loss",
            )

        if history:
            ax_temp.plot(steps, [step.temperature for step in history], label="Temperatur", color="#d97706")
            accepted_steps = [step.step_index for step in history if step.accepted]
            accepted_temps = [step.temperature for step in history if step.accepted]
            if accepted_steps:
                ax_temp.scatter(accepted_steps, accepted_temps, label="akzeptiert", color="#16a34a", s=8)
        elif start_temperature is not None:
            ax_temp.plot([0], [start_temperature], marker="o", label="Temperatur", color="#d97706")
        ax_batch.set_title("Online-Delta Batch-Loss")
        ax_val.set_title("Validation-Loss waehrend der Suche")
        ax_temp.set_title("Temperatur")
        for axis in (ax_batch, ax_val, ax_temp):
            if axis.lines or axis.collections:
                axis.legend(loc="best")
        self.canvas.draw_idle()
