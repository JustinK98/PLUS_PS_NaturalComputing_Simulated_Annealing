"""Matplotlib-basierte Plot-Widgets fuer die Qt-GUI."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6 import QtWidgets


def _adaptive_smoothing_window(length: int) -> int:
    """Return an odd rolling window that remains readable for long SA histories."""

    if length < 25:
        return 1
    window = max(11, min(251, length // 50))
    return window if window % 2 == 1 else window + 1


def _rolling_median(values: list[float], window: int) -> np.ndarray:
    """Smooth noisy batch diagnostics without letting large spikes dominate."""

    array = np.asarray(values, dtype=np.float64)
    if window <= 1 or len(array) < 2:
        return array
    radius = window // 2
    padded = np.pad(array, (radius, radius), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(windows, axis=1)


def _sample_raw_points(
    steps: list[int],
    values: list[float],
    *,
    max_points: int = 3000,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep raw diagnostics visible without overwhelming Matplotlib or the reader."""

    if len(steps) <= max_points:
        return np.asarray(steps), np.asarray(values)
    indices = np.linspace(0, len(steps) - 1, max_points, dtype=int)
    return np.asarray(steps)[indices], np.asarray(values)[indices]


class _BasePlotWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(6, 4), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)


class TrainingPlotWidget(_BasePlotWidget):
    def set_history(
        self,
        history: dict[str, list[float]] | None,
        overlay_histories: list[tuple[str, dict[str, list[float]]]] | None = None,
        reference_metrics: list[tuple[str, dict[str, float]]] | None = None,
    ) -> None:
        self.figure.clear()
        ax_loss = self.figure.add_subplot(211)
        ax_acc = self.figure.add_subplot(212)
        if overlay_histories:
            colors = ("#2563eb", "#16a34a", "#d97706", "#7c3aed", "#dc2626")
            for index, (label, series_history) in enumerate(overlay_histories):
                color = colors[index % len(colors)]
                val_loss = series_history.get("val_loss", [])
                val_acc = series_history.get("val_acc", [])
                if val_loss:
                    ax_loss.plot(
                        val_loss,
                        label=f"{label} Val",
                        color=color,
                        linestyle="-",
                        linewidth=2.0,
                    )
                if val_acc:
                    ax_acc.plot(
                        val_acc,
                        label=f"{label} Val",
                        color=color,
                        linestyle="-",
                        linewidth=2.0,
                    )
            for label, metrics in reference_metrics or []:
                if "val_loss" in metrics:
                    ax_loss.axhline(
                        float(metrics["val_loss"]),
                        label=f"{label} Val",
                        color="#dc2626",
                        linestyle=":",
                        linewidth=2.2,
                    )
                if "val_accuracy" in metrics:
                    ax_acc.axhline(
                        float(metrics["val_accuracy"]),
                        label=f"{label} Val",
                        color="#dc2626",
                        linestyle=":",
                        linewidth=2.2,
                    )
            ax_loss.legend(loc="best")
            ax_acc.legend(loc="best")
        elif history:
            ax_loss.plot(history.get("train_loss", []), label="Train-Loss")
            ax_loss.plot(history.get("val_loss", []), label="Val-Loss")
            ax_acc.plot(history.get("train_acc", []), label="Train-Accuracy")
            ax_acc.plot(history.get("val_acc", []), label="Val-Accuracy")
            ax_loss.legend(loc="best")
            ax_acc.legend(loc="best")
        ax_loss.set_title("Loss")
        ax_acc.set_title("Accuracy")
        self.canvas.draw_idle()

    def set_aggregate_histories(
        self,
        grouped_histories: list[tuple[str, list[dict[str, list[float]]]]],
    ) -> None:
        """Zeigt semantisch getrennte Trainingskurven als Median mit IQR."""

        from services.gui_multi_run_service import aggregate_histories

        self.figure.clear()
        ax_loss = self.figure.add_subplot(211)
        ax_acc = self.figure.add_subplot(212)
        colors = ("#2563eb", "#16a34a", "#d97706", "#7c3aed", "#dc2626")
        for index, (label, histories) in enumerate(grouped_histories):
            color = colors[index % len(colors)]
            loss = aggregate_histories(histories, "val_loss")
            accuracy = aggregate_histories(histories, "val_acc")
            if loss is not None:
                ax_loss.plot(loss.x, loss.median, color=color, linewidth=2.0, label=f"{label} Median")
                ax_loss.fill_between(loss.x, loss.q25, loss.q75, color=color, alpha=0.2, label=f"{label} IQR")
            if accuracy is not None:
                ax_acc.plot(
                    accuracy.x,
                    accuracy.median,
                    color=color,
                    linewidth=2.0,
                    label=f"{label} Median",
                )
                ax_acc.fill_between(
                    accuracy.x,
                    accuracy.q25,
                    accuracy.q75,
                    color=color,
                    alpha=0.2,
                    label=f"{label} IQR",
                )
        ax_loss.set_title("Validation-Loss: Median und IQR")
        ax_acc.set_title("Validation-Accuracy: Median und IQR")
        for axis in (ax_loss, ax_acc):
            if axis.lines:
                axis.legend(loc="best")
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
            smoothing_window = _adaptive_smoothing_window(len(batch_steps))
            raw_before_steps, raw_before_losses = _sample_raw_points(batch_steps, before_losses)
            raw_candidate_steps, raw_candidate_losses = _sample_raw_points(
                batch_steps,
                candidate_losses,
            )
            ax_batch.scatter(
                raw_before_steps,
                raw_before_losses,
                color="#2563eb",
                alpha=0.12,
                s=7,
                linewidths=0,
                label="_nolegend_",
            )
            ax_batch.scatter(
                raw_candidate_steps,
                raw_candidate_losses,
                color="#f97316",
                alpha=0.12,
                s=7,
                linewidths=0,
                label="_nolegend_",
            )
            ax_batch.plot(
                batch_steps,
                _rolling_median(before_losses, smoothing_window),
                color="#2563eb",
                linewidth=1.8,
                label=f"vorher, Median {smoothing_window}",
            )
            ax_batch.plot(
                batch_steps,
                _rolling_median(candidate_losses, smoothing_window),
                color="#f97316",
                linewidth=1.8,
                label=f"Kandidat, Median {smoothing_window}",
            )
            ax_val.plot(
                validation_steps,
                current_validation_losses,
                marker="o",
                markersize=3,
                label="aktueller Val-Loss (Diagnose)",
            )
            ax_val.plot(
                validation_steps,
                best_validation_losses,
                marker="o",
                markersize=3,
                label="diagnostisch bester Val-Loss",
            )

        if history:
            ax_temp.plot(steps, [step.temperature for step in history], label="Temperatur", color="#d97706")
            accepted_steps = [step.step_index for step in history if step.accepted]
            accepted_temps = [step.temperature for step in history if step.accepted]
            if accepted_steps:
                ax_temp.scatter(accepted_steps, accepted_temps, label="akzeptiert", color="#16a34a", s=8)
        elif start_temperature is not None:
            ax_temp.plot([0], [start_temperature], marker="o", label="Temperatur", color="#d97706")
        ax_batch.set_title("Online-Delta Batch-Loss (Rohpunkte + rollender Median)")
        ax_val.set_title("Validation-Loss-Diagnose waehrend der Suche")
        ax_temp.set_title("Temperatur")
        for axis in (ax_batch, ax_val, ax_temp):
            if axis.lines or axis.collections:
                axis.legend(loc="best")
        self.canvas.draw_idle()

    def set_aggregate_snapshots(self, snapshots: list[Any]) -> None:
        """Zeigt SA-Diagnosen ueber Runs als Median und IQR."""

        from services.gui_multi_run_service import (
            aggregate_sa_snapshots,
            aggregate_validation_by_online_epochs,
        )

        self.figure.clear()
        ax_batch = self.figure.add_subplot(411)
        ax_delta = self.figure.add_subplot(412)
        ax_val = self.figure.add_subplot(413)
        ax_temp = self.figure.add_subplot(414)
        for key, label, color in (
            ("batch_loss_before", "vorher", "#2563eb"),
            ("candidate_loss_after", "Kandidat", "#f97316"),
        ):
            series = aggregate_sa_snapshots(snapshots, key)
            if series is not None:
                ax_batch.plot(series.x, series.median, color=color, label=f"{label} Median")
                ax_batch.fill_between(series.x, series.q25, series.q75, color=color, alpha=0.2)
        delta = aggregate_sa_snapshots(snapshots, "delta")
        if delta is not None:
            ax_delta.plot(delta.x, delta.median, color="#7c3aed", label="Delta Median")
            ax_delta.fill_between(delta.x, delta.q25, delta.q75, color="#7c3aed", alpha=0.2)
            ax_delta.axhline(0.0, color="#64748b", linewidth=1.0, linestyle=":")
        validation = aggregate_validation_by_online_epochs(snapshots)
        if validation is not None:
            ax_val.plot(validation.x, validation.median, color="#16a34a", label="Val-Loss Median")
            ax_val.fill_between(
                validation.x,
                validation.q25,
                validation.q75,
                color="#16a34a",
                alpha=0.2,
                label="Val-Loss IQR",
            )
        temperature = aggregate_sa_snapshots(snapshots, "temperature")
        if temperature is not None:
            ax_temp.plot(temperature.x, temperature.median, color="#d97706", label="Temperatur Median")
            ax_temp.fill_between(
                temperature.x,
                temperature.q25,
                temperature.q75,
                color="#d97706",
                alpha=0.2,
                label="Temperatur IQR",
            )
        ax_batch.set_title("Online-Delta Batch-Loss nach Proposal-Schritt")
        ax_delta.set_title("Online-Delta nach Proposal-Schritt")
        ax_val.set_title("Validation-Loss nach effektiven Online-Epochen")
        ax_temp.set_title("Temperatur nach Proposal-Schritt")
        for axis in (ax_batch, ax_delta, ax_val, ax_temp):
            if axis.lines:
                axis.legend(loc="best")
        self.canvas.draw_idle()
