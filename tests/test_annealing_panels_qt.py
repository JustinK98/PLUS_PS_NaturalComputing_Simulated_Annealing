from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", ".mplconfig")

from PySide6 import QtWidgets

from annealing import AnnealingConfig
from configs import DatasetConfig, TrainingConfig
from services.online_annealing_training_service import (
    OnlineAnnealingRequest,
    create_online_session,
    evaluate_online_start,
    online_step_once,
)
from ui_qt.widgets.annealing_history_panel import AnnealingHistoryPanel
from ui_qt.widgets.plot_widgets import _adaptive_smoothing_window, _rolling_median


def _app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class AnnealingPanelsQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _app()

    def _snapshot(self):
        request = OnlineAnnealingRequest(
            dataset_config=DatasetConfig(name="concentric_circles", random_state=6),
            hidden_sizes=(8,),
            layout_spec="relu",
            training_config=TrainingConfig(epochs=2, learning_rate=0.03, batch_size=16, random_state=6),
            annealing_config=AnnealingConfig(
                start_temperature=1.0,
                cooling_schedule="geometric",
                cooling_parameter=0.9,
                iterations_per_temperature=2,
                max_steps=4,
                min_temperature=0.01,
                neighborhood_operations=("set_neuron",),
            ),
            weight_scale=0.05,
            random_state=6,
        )
        session = create_online_session(request)
        evaluate_online_start(session, "en")
        return online_step_once(session, "en")

    def test_history_panel_renders_online_delta_fields(self) -> None:
        request = OnlineAnnealingRequest(
            dataset_config=DatasetConfig(name="concentric_circles", random_state=9),
            hidden_sizes=(8,),
            layout_spec="relu",
            training_config=TrainingConfig(epochs=2, learning_rate=0.03, batch_size=16, random_state=9),
            annealing_config=AnnealingConfig(
                start_temperature=1.0,
                cooling_schedule="geometric",
                cooling_parameter=0.9,
                iterations_per_temperature=1,
                max_steps=2,
                min_temperature=0.01,
                neighborhood_operations=("set_neuron",),
            ),
            weight_scale=0.05,
            random_state=9,
        )
        session = create_online_session(request)
        evaluate_online_start(session, "en")
        snapshot = online_step_once(session, "en")
        history = AnnealingHistoryPanel("en")

        history.set_snapshot(snapshot, "en")
        self.app.processEvents()

        self.assertIn("before=", history.text.toPlainText())
        self.assertEqual(len(history.plot.figure.axes), 3)
        self.assertEqual(
            history.plot.figure.axes[0].get_title(),
            "Online-Delta Batch-Loss (Rohpunkte + rollender Median)",
        )
        batch_labels = history.plot.figure.axes[0].get_legend_handles_labels()[1]
        self.assertIn("vorher, Median 1", batch_labels)
        self.assertIn("Kandidat, Median 1", batch_labels)
        validation_labels = [
            line.get_label()
            for line in history.plot.figure.axes[1].lines
        ]
        self.assertEqual(validation_labels.count("aktueller Val-Loss (Diagnose)"), 1)
        self.assertEqual(validation_labels.count("diagnostisch bester Val-Loss"), 1)
        history.close()

    def test_history_panel_uses_online_delta_axes_before_first_step(self) -> None:
        request = OnlineAnnealingRequest(
            dataset_config=DatasetConfig(name="concentric_circles", random_state=10),
            hidden_sizes=(8,),
            layout_spec="relu",
            training_config=TrainingConfig(epochs=2, learning_rate=0.03, batch_size=16, random_state=10),
            annealing_config=AnnealingConfig(
                start_temperature=1.0,
                cooling_schedule="geometric",
                cooling_parameter=0.9,
                iterations_per_temperature=1,
                max_steps=2,
                min_temperature=0.01,
                neighborhood_operations=("set_neuron",),
            ),
            weight_scale=0.05,
            random_state=10,
        )
        snapshot = evaluate_online_start(create_online_session(request), "en")
        history = AnnealingHistoryPanel("en")

        history.set_snapshot(snapshot, "en")
        self.app.processEvents()

        self.assertEqual(len(history.plot.figure.axes), 3)
        self.assertEqual(
            history.plot.figure.axes[0].get_title(),
            "Online-Delta Batch-Loss (Rohpunkte + rollender Median)",
        )
        self.assertEqual(history.plot.figure.axes[1].get_title(), "Validation-Loss-Diagnose waehrend der Suche")
        self.assertEqual(history.plot.figure.axes[2].get_title(), "Temperatur")
        history.close()

    def test_batch_loss_smoothing_uses_robust_adaptive_median(self) -> None:
        values = [1.0] * 30
        values[15] = 100.0
        window = _adaptive_smoothing_window(len(values))

        smoothed = _rolling_median(values, window)

        self.assertEqual(window, 11)
        self.assertEqual(len(smoothed), len(values))
        self.assertEqual(smoothed[15], 1.0)


if __name__ == "__main__":
    unittest.main()
