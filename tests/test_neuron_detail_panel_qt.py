from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", ".mplconfig")

from PySide6 import QtWidgets

from activations import parse_layout_spec
from benchmarks import load_benchmark
from configs import DatasetConfig
from model import ModularMLP
from services.analysis_sample_service import build_analysis_sample
from services.neuron_analysis_service import build_neuron_analysis_payload
from ui_qt.widgets.neuron_detail_panel import NeuronDetailPanel


def _app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class NeuronDetailPanelQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _app()

    def test_panel_updates_for_selected_neuron(self) -> None:
        dataset = load_benchmark(DatasetConfig(name="test_activation", random_state=13))
        model = ModularMLP(
            input_size=dataset.input_size,
            hidden_sizes=(4, 3),
            output_size=dataset.output_size,
            layout=parse_layout_spec("relu|tanh", (4, 3)),
            random_state=13,
        )
        sample = build_analysis_sample(dataset, "train", 0, language="en")
        payload = build_neuron_analysis_payload(model, dataset, sample, 0, 0, language="en")

        panel = NeuronDetailPanel("en")
        panel.set_payload(payload, "en")
        self.app.processEvents()

        self.assertIn("L1:n0", panel.header_label.text())
        self.assertIn("Compact computation", panel.detail_text.toPlainText())
        panel.close()


if __name__ == "__main__":
    unittest.main()
