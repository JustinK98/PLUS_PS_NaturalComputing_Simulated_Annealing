"""Kompakte Aggregate-Ansichten fuer den Multi-Seed-GUI-Modus."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6 import QtCore, QtGui, QtWidgets

from activations import parse_layout_spec
from configs import SUPPORTED_ACTIVATIONS
from services.gui_multi_run_service import GuiRunRecord, paired_final_rows
from ui_qt.widgets.plot_widgets import AnnealingPlotWidget


class AggregatePlaceholderWidget(QtWidgets.QWidget):
    def __init__(self, title: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addStretch(1)
        label = QtWidgets.QLabel(
            f"<b>{title}</b><br><br>"
            "Diese qualitative Ansicht ist nur fuer einen einzelnen Run sinnvoll.<br>"
            "Waehle oben Run 1 bis Run N, um die Details zu untersuchen."
        )
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setWordWrap(True)
        layout.addWidget(label)
        layout.addStretch(1)


class AggregateLayoutHeatmapWidget(QtWidgets.QWidget):
    """Seed-sortierte Start-/Endlayout-Heatmap."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 5), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def set_runs(self, runs: list[GuiRunRecord], hidden_sizes: tuple[int, ...]) -> None:
        self.figure.clear()
        axes = self.figure.subplots(1, 2, squeeze=False)[0]
        activation_ids = {name: index for index, name in enumerate(SUPPORTED_ACTIVATIONS)}
        starts: list[list[int]] = []
        ends: list[list[int]] = []
        labels: list[str] = []
        for record in sorted(runs, key=lambda item: item.schedule.layout_seed):
            start = parse_layout_spec(record.start_layout_spec, hidden_sizes)
            snapshot = record.annealing_snapshot
            end = (
                snapshot.current_evaluation.layout
                if snapshot is not None and snapshot.current_evaluation is not None
                else start
            )
            starts.append([activation_ids[name] for layer in start.layers for name in layer])
            ends.append([activation_ids[name] for layer in end.layers for name in layer])
            labels.append(f"Run {record.run_index + 1}")
        for axis, matrix, title in zip(
            axes,
            (starts, ends),
            ("Random-Mixed-Startlayouts", "Finale SA-Layouts"),
            strict=True,
        ):
            if matrix:
                image = axis.imshow(np.asarray(matrix), aspect="auto", vmin=0, vmax=len(SUPPORTED_ACTIVATIONS) - 1)
                axis.set_yticks(range(len(labels)), labels)
                axis.set_xlabel("Hidden-Neuron-Position")
                self.figure.colorbar(image, ax=axis, ticks=range(len(SUPPORTED_ACTIVATIONS))).ax.set_yticklabels(
                    SUPPORTED_ACTIVATIONS
                )
            axis.set_title(title)
        self.canvas.draw_idle()


class AggregateAnnealingWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = AnnealingPlotWidget()
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.summary)

    def set_runs(self, runs: list[GuiRunRecord]) -> None:
        snapshots = [record.annealing_snapshot for record in runs if record.annealing_snapshot is not None]
        self.plot.set_aggregate_snapshots(snapshots)
        excluded = len(runs) - len(snapshots)
        self.summary.setText(
            f"Aggregiert: {len(snapshots)} Runs | ausgeschlossen/unvollstaendig: {excluded}"
        )


class AggregateFinalComparisonWidget(QtWidgets.QWidget):
    """Seed-sortierte gepaarte Endergebnisse als Heatmap-Tabelle."""

    HEADERS = (
        "Run",
        "Retrain-Seed",
        "Start Val Loss",
        "End Val Loss",
        "Verbesserung",
        "Start Val Acc",
        "End Val Acc",
        "Inherited Best Val",
        "Acceptance",
        "Online-Epochen",
    )

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        self.table = QtWidgets.QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        layout.addWidget(self.summary)
        layout.addWidget(self.table, 1)

    def set_runs(self, runs: list[GuiRunRecord]) -> None:
        rows = paired_final_rows(runs)
        self.table.setRowCount(len(rows))
        improvements = np.asarray([row["paired_improvement"] for row in rows], dtype=np.float64)
        for row_index, row in enumerate(rows):
            values = (
                row["run"],
                row["seed"],
                row["start_val_loss"],
                row["end_val_loss"],
                row["paired_improvement"],
                row["start_val_accuracy"],
                row["end_val_accuracy"],
                row["inherited_best_val_loss"],
                row["acceptance_rate"],
                row["effective_online_epochs"],
            )
            improvement = float(row["paired_improvement"])
            background = QtGui.QColor("#dcfce7" if improvement > 0 else "#fee2e2")
            for column, value in enumerate(values):
                text = "" if value is None else (f"{value:.5f}" if isinstance(value, float) else str(value))
                item = QtWidgets.QTableWidgetItem(text)
                item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                item.setBackground(background)
                self.table.setItem(row_index, column, item)
        if improvements.size:
            median = float(np.median(improvements))
            q25, q75 = np.percentile(improvements, [25, 75])
            wins = float(np.mean(improvements > 0))
            self.summary.setText(
                f"<b>Gepaarte Verbesserung:</b> Median {median:+.6f}, "
                f"IQR [{q25:+.6f}, {q75:+.6f}], Run-Win-Rate {wins:.1%}. "
                f"Fertig: {len(rows)}/{len(runs)}."
            )
        else:
            self.summary.setText("Noch keine vollstaendigen gepaarten Endvergleiche.")
