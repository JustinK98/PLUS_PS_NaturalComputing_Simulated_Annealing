"""Praesentationsansicht fuer Startlayout und auswaehlbare SA-Referenz."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from activations import diff_layouts
from model import ModularMLP
from services.network_projection_service import InputProjection
from services.online_annealing_training_service import (
    OnlineAnnealingSnapshot,
    OnlineLayoutEvaluation,
)
from .network_view import NetworkViewWidget


class NetworkComparisonPanel(QtWidgets.QWidget):
    """Zeigt zunaechst das Startnetz und nach dem ersten SA-Schritt eine Referenz."""

    hiddenNeuronSelected = QtCore.Signal(int, int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._snapshot: OnlineAnnealingSnapshot | None = None
        self._preview_model: ModularMLP | None = None
        self._projection: InputProjection | None = None
        self._selected_hidden: tuple[int, int] | None = None

        self.start_title = QtWidgets.QLabel("Random-Mixed-Startlayout")
        self.start_title.setProperty("role", "sectionTitle")
        self.reference_title = QtWidgets.QLabel("Vergleichslayout")
        self.reference_title.setProperty("role", "sectionTitle")
        self.reference_combo = QtWidgets.QComboBox()
        self.reference_combo.addItem("Bestes Layout", "best")
        self.reference_combo.addItem("Aktuelles Layout", "current")
        self.reference_combo.addItem("Letzter Kandidat", "candidate")
        self.reference_combo.currentIndexChanged.connect(self._refresh_reference)

        self.start_view = NetworkViewWidget()
        self.reference_view = NetworkViewWidget()
        self.start_view.hiddenNeuronSelected.connect(self.hiddenNeuronSelected)
        self.reference_view.hiddenNeuronSelected.connect(self.hiddenNeuronSelected)
        self.start_details = QtWidgets.QLabel()
        self.start_details.setWordWrap(True)
        self.start_details.setProperty("role", "muted")
        self.reference_details = QtWidgets.QLabel()
        self.reference_details.setWordWrap(True)
        self.reference_details.setProperty("role", "muted")

        start_box = QtWidgets.QWidget()
        start_layout = QtWidgets.QVBoxLayout(start_box)
        start_layout.setContentsMargins(0, 0, 0, 0)
        start_layout.addWidget(self.start_title)
        start_layout.addWidget(self.start_view, 1)
        start_layout.addWidget(self.start_details)

        self.reference_box = QtWidgets.QWidget()
        reference_layout = QtWidgets.QVBoxLayout(self.reference_box)
        reference_layout.setContentsMargins(0, 0, 0, 0)
        heading = QtWidgets.QHBoxLayout()
        heading.addWidget(self.reference_title)
        heading.addStretch(1)
        heading.addWidget(self.reference_combo)
        reference_layout.addLayout(heading)
        reference_layout.addWidget(self.reference_view, 1)
        reference_layout.addWidget(self.reference_details)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(start_box)
        splitter.addWidget(self.reference_box)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.addWidget(splitter, 1)
        self.reference_box.setVisible(False)

    def set_preview_model(self, model: ModularMLP | None) -> None:
        self._preview_model = model
        if self._snapshot is None or self._snapshot.start_evaluation is None:
            self.start_title.setText("Aktuelles Random-Mixed-Layout")
            self.start_view.set_model(model)
            self.start_details.setText(
                f"Layout: {model.layout.to_compact_spec()}" if model is not None else ""
            )
            self.reference_box.setVisible(False)

    def set_input_projection(self, projection: InputProjection | None) -> None:
        self._projection = projection
        self.start_view.set_input_projection(projection)
        self.reference_view.set_input_projection(projection)

    def set_selected_hidden(self, selection: tuple[int, int] | None) -> None:
        self._selected_hidden = selection
        self.start_view.set_selected_hidden(selection)
        self.reference_view.set_selected_hidden(selection)

    def set_snapshot(self, snapshot: OnlineAnnealingSnapshot | None) -> None:
        self._snapshot = snapshot
        if snapshot is None or snapshot.start_evaluation is None:
            self.set_preview_model(self._preview_model)
            return

        self.start_title.setText("SA-Startlayout")
        self.start_view.set_model(snapshot.start_evaluation.trained_model)
        self.start_view.set_highlighted_hidden((), None)
        self.start_details.setText(self._evaluation_details(snapshot.start_evaluation))
        self.reference_box.setVisible(bool(snapshot.history))
        self._refresh_reference()

    def _refresh_reference(self) -> None:
        snapshot = self._snapshot
        if snapshot is None or not snapshot.history:
            self.reference_box.setVisible(False)
            return
        evaluation = self._selected_evaluation(snapshot)
        self.reference_box.setVisible(evaluation is not None)
        if evaluation is None:
            return

        self.reference_view.set_model(evaluation.trained_model)
        positions: frozenset[tuple[int, int]] = frozenset()
        status: str | None = None
        last_step = snapshot.last_step
        if last_step is not None and self.reference_combo.currentData() in {"current", "candidate"}:
            positions = frozenset(
                (change.layer_index, change.neuron_index)
                for change in diff_layouts(last_step.previous_layout, last_step.candidate_layout)
            )
            status = "accepted" if last_step.accepted else "rejected"
        self.reference_view.set_highlighted_hidden(positions, status)
        self.reference_details.setText(self._evaluation_details(evaluation))

    def _selected_evaluation(
        self,
        snapshot: OnlineAnnealingSnapshot,
    ) -> OnlineLayoutEvaluation | None:
        selected = self.reference_combo.currentData()
        if selected == "current":
            return snapshot.current_evaluation
        if selected == "candidate":
            return snapshot.candidate_evaluation
        return snapshot.best_evaluation

    @staticmethod
    def _evaluation_details(evaluation: OnlineLayoutEvaluation) -> str:
        return (
            f"Layout: {evaluation.layout.to_compact_spec()}\n"
            f"Batch-Loss: {evaluation.objective_value:.6f} | "
            f"Validation-Loss: {evaluation.val_loss:.6f} | "
            f"Validation-Accuracy: {evaluation.val_accuracy:.4f}"
        )
