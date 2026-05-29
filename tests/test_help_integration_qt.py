from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", ".mplconfig")

from PySide6 import QtWidgets

from configs import GuiExperimentConfig
from ui_qt.shell.main_window import MainWindow
from ui_qt.state import WorkspacePreferences
from ui_qt.workspaces.activation_workflow_workspace import ActivationWorkflowWorkspace
from ui_qt.workspaces.experiment_builder_workspace import ExperimentBuilderWorkspace


def _app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class HelpIntegrationQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _app()

    def test_main_window_opens_handbook_dialog(self) -> None:
        window = MainWindow(
            GuiExperimentConfig(
                benchmark="two_moons",
                hidden_sizes=(8,),
                app_mode="activation_workflow",
                layout_spec="relu",
                mode="expert",
                language="en",
            )
        )
        window._open_handbook()
        self.app.processEvents()
        self.assertIsNotNone(window.handbook_dialog)
        self.assertIn("Activation Playground Guide", window.handbook_dialog.browser.toPlainText())
        window.handbook_dialog.close()
        window.close()

    def test_activation_workflow_has_help_tab(self) -> None:
        workspace = ActivationWorkflowWorkspace(
            GuiExperimentConfig(
                benchmark="two_moons",
                hidden_sizes=(8,),
                app_mode="activation_workflow",
                layout_spec="relu",
                mode="expert",
                language="de",
            ),
            WorkspacePreferences(language="de", detail_mode="expert"),
        )
        self.assertIn("Activation Workflow", workspace.help_text.toPlainText())
        workspace.tabs.setCurrentIndex(workspace.tabs.count() - 1)
        self.app.processEvents()
        workspace.close()

    def test_builder_has_real_help_tab(self) -> None:
        builder = ExperimentBuilderWorkspace(
            GuiExperimentConfig(
                benchmark="two_moons",
                hidden_sizes=(8,),
                app_mode="experiment_builder",
                layout_spec="relu",
                mode="expert",
                language="en",
            ),
            WorkspacePreferences(language="en", detail_mode="expert"),
        )
        self.assertIn("How to read builder results", builder.help_text.toPlainText())
        builder.close()


if __name__ == "__main__":
    unittest.main()
