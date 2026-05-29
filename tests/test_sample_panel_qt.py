from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", ".mplconfig")

from PySide6 import QtWidgets

from benchmarks import load_benchmark
from configs import DatasetConfig
from services.analysis_sample_service import build_analysis_sample
from ui_qt.widgets.sample_panel import SampleDisplayPayload, SamplePanelWidget


def _app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class SamplePanelQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _app()

    def test_official_sample_panel_shows_feature_table(self) -> None:
        dataset = load_benchmark(DatasetConfig(name="two_moons", random_state=9))
        sample = build_analysis_sample(dataset, "val", 0, language="en")
        panel = SamplePanelWidget(language="en")
        panel.set_sample(
            SampleDisplayPayload(
                dataset=dataset,
                analysis_sample=sample,
                prediction_name=dataset.target_names[0],
                probabilities=[1.0 / dataset.output_size] * dataset.output_size,
            )
        )

        self.assertIs(panel.stack.currentWidget(), panel.generic_table)
        self.assertGreater(panel.generic_table.rowCount(), 0)
        self.assertIsNotNone(panel.generic_table.item(0, 1))
        panel.close()


if __name__ == "__main__":
    unittest.main()
