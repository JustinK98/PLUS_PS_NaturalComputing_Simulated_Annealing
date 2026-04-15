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
from services.compare_service import build_compare_payload
from ui_qt.widgets.compare_panel import ComparePanel


def _app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class ComparePanelQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _app()

    def test_compare_panel_renders_layout_and_predictions(self) -> None:
        dataset = load_benchmark(DatasetConfig(name="wine", random_state=2))
        sample = build_analysis_sample(dataset, "val", 0, language="en")
        baseline_model = ModularMLP(
            input_size=dataset.input_size,
            hidden_sizes=(16, 8),
            output_size=dataset.output_size,
            layout=parse_layout_spec("relu|relu", (16, 8)),
            random_state=2,
        )
        current_model = ModularMLP(
            input_size=dataset.input_size,
            hidden_sizes=(16, 8),
            output_size=dataset.output_size,
            layout=parse_layout_spec("relu|tanh", (16, 8)),
            random_state=3,
        )
        payload = build_compare_payload(
            current_model,
            baseline_model,
            dataset,
            "en",
            analysis_sample=sample,
        )

        panel = ComparePanel("en")
        panel.set_payload(payload, "en")
        self.app.processEvents()
        text = panel.detail_text.toPlainText()
        self.assertIn("Current layout", text)
        self.assertIn("Baseline probabilities", text)
        panel.close()


if __name__ == "__main__":
    unittest.main()
