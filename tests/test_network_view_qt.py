from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", ".mplconfig")

from PySide6 import QtCore, QtWidgets

from activations import parse_layout_spec
from configs import GuiExperimentConfig
from model import ModularMLP
from ui_qt.state import WorkspacePreferences
from ui_qt.widgets.network_view import NetworkViewWidget
from ui_qt.workspaces.activation_workflow_workspace import ActivationWorkflowWorkspace


def _app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class NetworkViewQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _app()

    def test_network_view_renders_all_nodes_and_supports_fit(self) -> None:
        view = NetworkViewWidget()
        view.resize(680, 420)
        view.show()
        layout = parse_layout_spec("relu|tanh", (32, 16))
        model = ModularMLP(
            input_size=64,
            hidden_sizes=(32, 16),
            output_size=10,
            layout=layout,
            random_state=7,
        )
        view.set_model(model)
        self.app.processEvents()

        input_nodes = [key for key in view._node_positions if key[0] == "input"]
        hidden_nodes = [key for key in view._node_positions if key[0] == "hidden"]
        output_nodes = [key for key in view._node_positions if key[0] == "output"]

        self.assertEqual(len(input_nodes), 64)
        self.assertEqual(len(hidden_nodes), 48)
        self.assertEqual(len(output_nodes), 10)
        self.assertFalse(view.sceneRect().isEmpty())
        self.assertGreater(view.transform().m11(), 0.0)
        view.close()

    def test_network_view_manual_zoom_can_be_reset_to_auto_fit(self) -> None:
        view = NetworkViewWidget()
        view.resize(680, 420)
        view.show()
        layout = parse_layout_spec("relu|relu", (16, 8))
        model = ModularMLP(
            input_size=13,
            hidden_sizes=(16, 8),
            output_size=3,
            layout=layout,
            random_state=5,
        )
        view.set_model(model)
        self.app.processEvents()

        fitted_scale = view.transform().m11()
        view.zoom_in()
        self.app.processEvents()
        self.assertFalse(view.is_auto_fit_enabled())
        self.assertGreater(view.transform().m11(), fitted_scale)

        view.reset_view_to_fit()
        self.app.processEvents()
        self.assertTrue(view.is_auto_fit_enabled())
        self.assertGreater(view.transform().m11(), 0.0)
        view.close()

    def test_network_view_tracks_sa_highlight_status(self) -> None:
        view = NetworkViewWidget()
        layout = parse_layout_spec("relu", (8,))
        model = ModularMLP(
            input_size=2,
            hidden_sizes=(8,),
            output_size=1,
            layout=layout,
            random_state=5,
        )
        view.set_model(model)

        view.set_highlighted_hidden({(0, 3)}, "accepted")
        self.assertEqual(view._highlighted_hidden, frozenset({(0, 3)}))
        self.assertEqual(view._highlight_status, "accepted")

        view.set_highlighted_hidden({(0, 4)}, "rejected")
        self.assertEqual(view._highlighted_hidden, frozenset({(0, 4)}))
        self.assertEqual(view._highlight_status, "rejected")
        view.close()

    def test_activation_workflow_exposes_resizable_network_splitter(self) -> None:
        preferences = WorkspacePreferences(language="en", detail_mode="expert")
        workflow = ActivationWorkflowWorkspace(
            GuiExperimentConfig(
                benchmark="two_moons",
                mode="expert",
                language="en",
            ),
            preferences,
        )

        self.assertEqual(workflow.splitter.orientation(), QtCore.Qt.Horizontal)
        self.assertEqual(workflow.splitter.count(), 2)
        workflow.close()


if __name__ == "__main__":
    unittest.main()
