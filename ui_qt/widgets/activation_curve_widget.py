"""Aktivierungsfunktionsplot fuer den sample-spezifischen Neuron-Tracker."""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6 import QtWidgets

from activations import ACTIVATION_COLORS
from services.neuron_analysis_service import NeuronAnalysisPayload


class ActivationCurveWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(4, 3), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def set_payload(self, payload: NeuronAnalysisPayload | None) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        if payload is not None:
            inspection = payload.inspection
            axis.plot(
                payload.curve_x,
                payload.curve_y,
                color=ACTIVATION_COLORS[inspection.activation_name],
                linewidth=2.0,
            )
            axis.axvline(inspection.pre_activation, color="#111827", linestyle="--", linewidth=1.0)
            axis.scatter(
                [inspection.pre_activation],
                [inspection.output_value],
                color="#111827",
                zorder=3,
            )
            axis.set_title(f"{inspection.activation_name}(z)")
        else:
            axis.set_title("Activation")
        axis.set_xlabel("z")
        axis.set_ylabel("a")
        axis.grid(alpha=0.25)
        self.canvas.draw_idle()
