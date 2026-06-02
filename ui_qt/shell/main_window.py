"""Hauptfenster fuer die Qt-Version des Activation Playground."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from configs import GuiExperimentConfig
from ui_qt.state import WorkspacePreferences
from ui_qt.workspaces import ActivationWorkflowWorkspace


class MainWindow(QtWidgets.QMainWindow):
    """Top-Level-Fenster fuer die fokussierte Online-Delta-Demo."""

    def __init__(self, config: GuiExperimentConfig, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Activation Playground")
        self.resize(1560, 980)
        self.settings = QtCore.QSettings("ActivationPlayground", "ActivationPlaygroundQt")
        self.preferences = WorkspacePreferences()

        self.workspace = ActivationWorkflowWorkspace(config, self.preferences, self)
        self.workspace.statusMessage.connect(self.statusBar().showMessage)
        self.workspace.busyChanged.connect(self._on_busy_changed)
        self.workspace.restore_settings(self.settings)
        self.setCentralWidget(self.workspace)

        self.statusBar().showMessage("Bereit")

    def closeEvent(self, event) -> None:  # pragma: no cover - GUI lifecycle
        self.workspace.save_settings(self.settings)
        super().closeEvent(event)

    def _on_busy_changed(self, busy: bool) -> None:
        self.setCursor(QtCore.Qt.WaitCursor if busy else QtCore.Qt.ArrowCursor)
