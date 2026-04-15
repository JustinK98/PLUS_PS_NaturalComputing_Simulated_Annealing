from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", ".mplconfig")

from PySide6 import QtWidgets

from services.stepper_service import StepperEntry
from ui_qt.widgets.stepper_panel import StepperPanel


def _app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class StepperPanelQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _app()

    def test_stepper_panel_navigates_entries(self) -> None:
        panel = StepperPanel("en")
        panel.set_entries(
            (
                StepperEntry("Input", "first"),
                StepperEntry("Forward L1", "second"),
                StepperEntry("Output", "third"),
            )
        )
        self.app.processEvents()

        self.assertIn("Input", panel.status_label.text())
        panel.step_next()
        self.assertIn("Forward L1", panel.status_label.text())
        panel.step_reset()
        self.assertIn("Input", panel.status_label.text())
        panel.close()


if __name__ == "__main__":
    unittest.main()
