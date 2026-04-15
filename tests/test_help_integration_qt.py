from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", ".mplconfig")

from PySide6 import QtWidgets

from configs import GuiExperimentConfig
from ui_qt.shell.main_window import MainWindow
from ui_qt.state import WorkspacePreferences
from ui_qt.widgets.info_button import InfoButton
from ui_qt.workspaces.demo_workspace import DemoWorkspace
from ui_qt.workspaces.experiment_builder_workspace import ExperimentBuilderWorkspace
from ui_qt.workspaces.playground_workspace import PlaygroundWorkspace


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
                benchmark="wine",
                hidden_sizes=(16, 8),
                app_mode="demo",
                layout_spec="relu|tanh",
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

    def test_demo_workspace_has_help_tab_and_info_buttons(self) -> None:
        workspace = DemoWorkspace(
            GuiExperimentConfig(
                benchmark="digits",
                hidden_sizes=(16, 8),
                app_mode="demo",
                layout_spec="relu|tanh",
                mode="expert",
                language="de",
            ),
            WorkspacePreferences(language="de", detail_mode="expert"),
        )
        self.assertIn("Empfohlener Ablauf", workspace.help_text.toPlainText())
        self.assertGreaterEqual(len(workspace.findChildren(InfoButton)), 6)
        workspace.tabs.setCurrentIndex(1)
        self.app.processEvents()
        self.assertIn("Training", workspace.tab_help_button.toolTip())
        workspace.close()

    def test_playground_and_builder_have_real_help_tabs(self) -> None:
        playground = PlaygroundWorkspace(
            GuiExperimentConfig(
                benchmark="test_activation",
                hidden_sizes=(4, 3),
                app_mode="playground",
                layout_spec="relu|tanh",
                mode="expert",
                language="en",
            ),
            WorkspacePreferences(language="en", detail_mode="expert"),
        )
        builder = ExperimentBuilderWorkspace(
            GuiExperimentConfig(
                benchmark="wine",
                hidden_sizes=(16, 8),
                app_mode="experiment_builder",
                layout_spec="relu|relu",
                mode="expert",
                language="en",
            ),
            WorkspacePreferences(language="en", detail_mode="expert"),
        )
        self.assertIn("Recommended workflow", playground.help_text.toPlainText())
        self.assertIn("How to read builder results", builder.help_text.toPlainText())
        playground.close()
        builder.close()


if __name__ == "__main__":
    unittest.main()
