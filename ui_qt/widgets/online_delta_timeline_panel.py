"""Slider-basierte Netzvisualisierung eines Online-Delta-SA-Laufs."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6 import QtCore, QtWidgets

from activations import ActivationLayout, diff_layouts
from model import ModularMLP
from services.online_annealing_training_service import OnlineAnnealingSnapshot
from .network_view import NetworkViewWidget


@dataclass(frozen=True)
class _TimelineEntry:
    label: str
    layout: ActivationLayout
    accepted: bool | None
    details: tuple[str, ...]
    changed_positions: frozenset[tuple[int, int]]


class OnlineDeltaTimelinePanel(QtWidgets.QWidget):
    """Zeigt Startlayout und jeden vorgeschlagenen Kandidaten als Netz."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._entries: list[_TimelineEntry] = []
        self._template_model: ModularMLP | None = None

        self.title_label = QtWidgets.QLabel("Online-Delta-Verlauf")
        self.title_label.setProperty("role", "sectionTitle")
        self.position_label = QtWidgets.QLabel()
        self.position_label.setProperty("role", "muted")
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._render_current_entry)
        self.network_view = NetworkViewWidget()
        self.network_view.setMinimumHeight(330)
        self.details_browser = QtWidgets.QTextBrowser()
        self.details_browser.setMinimumHeight(150)
        self.details_browser.setMaximumHeight(220)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        root.addWidget(self.title_label)
        root.addWidget(self.position_label)
        root.addWidget(self.slider)
        root.addWidget(self.network_view, 1)
        root.addWidget(self.details_browser)
        self._render_placeholder()

    def set_language(self, _language: str) -> None:
        """Die öffentliche GUI ist deutschsprachig; Methode bleibt API-kompatibel."""

    def set_snapshot(self, snapshot: OnlineAnnealingSnapshot | None) -> None:
        self._entries = self._build_entries(snapshot)
        self._template_model = (
            snapshot.start_evaluation.trained_model.clone()
            if snapshot is not None and snapshot.start_evaluation is not None
            else None
        )
        if not self._entries:
            self.slider.setRange(0, 0)
            self.slider.setEnabled(False)
            self._render_placeholder()
            return

        self.slider.setEnabled(len(self._entries) > 1)
        self.slider.setRange(0, len(self._entries) - 1)
        self.slider.setValue(len(self._entries) - 1)
        self._render_current_entry()

    @staticmethod
    def _build_entries(snapshot: OnlineAnnealingSnapshot | None) -> list[_TimelineEntry]:
        if snapshot is None or snapshot.start_evaluation is None:
            return []

        start = snapshot.start_evaluation
        entries = [
            _TimelineEntry(
                label="Schritt 0: Startlayout",
                layout=start.layout,
                accepted=None,
                details=(
                    "Noch keine Layoutänderung vorgeschlagen.",
                    f"Start-Batch-Loss: {start.objective_value:.6f}",
                    f"Start-Validation-Loss: {start.val_loss:.6f}",
                    f"Start-Validation-Accuracy: {start.val_accuracy:.4f}",
                    f"Layout: {start.layout.to_compact_spec()}",
                ),
                changed_positions=frozenset(),
            )
        ]
        current_layout = start.layout

        for step in snapshot.history:
            changes = frozenset(
                (change.layer_index, change.neuron_index)
                for change in diff_layouts(step.previous_layout, step.candidate_layout)
            )
            if step.accepted:
                current_layout = step.candidate_layout
            decision = "AKZEPTIERT" if step.accepted else "VERWORFEN"
            entries.append(
                _TimelineEntry(
                    label=f"Schritt {step.step_index}: {decision}",
                    # Auch ein verworfener Kandidat bleibt sichtbar. Der rote
                    # Rahmen macht klar, dass er nicht zum aktuellen Layout wurde.
                    layout=step.candidate_layout,
                    accepted=step.accepted,
                    details=(
                        f"Entscheidung: {decision}",
                        f"Vorgeschlagene Änderung: {step.neighbor_label}",
                        f"Batch-Loss vorher: {step.batch_loss_before:.6f}",
                        f"Batch-Loss Kandidat: {step.candidate_loss_after:.6f}",
                        f"Online-Delta (Kandidat - vorher): {step.delta:+.6f}",
                        f"Temperatur: {step.temperature:.6f}",
                        f"Akzeptanzwahrscheinlichkeit: {step.acceptance_probability:.4f}",
                        f"Validation-Loss nach Entscheidung: {step.validation_loss_after_update:.6f}",
                        f"Aktuelles Layout nach Entscheidung: {current_layout.to_compact_spec()}",
                    ),
                    changed_positions=changes,
                )
            )
        return entries

    def _render_placeholder(self) -> None:
        self.position_label.setText("Noch keine Online-Delta-Session.")
        self.details_browser.setPlainText(
            "Start bewerten, um das Random-Mixed-Startlayout und danach jeden "
            "vorgeschlagenen Nachbarn im Slider zu untersuchen."
        )
        self.network_view.set_model(None)
        self.network_view.set_highlighted_hidden((), None)

    def _render_current_entry(self) -> None:
        if not self._entries or self._template_model is None:
            self._render_placeholder()
            return
        entry = self._entries[self.slider.value()]
        self.position_label.setText(
            f"{entry.label} ({self.slider.value() + 1}/{len(self._entries)})"
        )
        self.details_browser.setPlainText("\n".join(entry.details))
        model = self._template_model.clone()
        model.set_layout(entry.layout)
        self.network_view.set_model(model)
        status = None
        if entry.accepted is True:
            status = "accepted"
        elif entry.accepted is False:
            status = "rejected"
        self.network_view.set_highlighted_hidden(entry.changed_positions, status)
