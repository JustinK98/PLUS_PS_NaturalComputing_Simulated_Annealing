"""Gemeinsame Basisklasse fuer Qt-Workspaces."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ui_qt.state import WorkspacePreferences


class BaseWorkspace(QtWidgets.QWidget):
    """Basisklasse mit Signalen fuer Status- und Busy-Zustände."""

    statusMessage = QtCore.Signal(str)
    busyChanged = QtCore.Signal(bool)

    workspace_id = "base"

    def __init__(self, preferences: WorkspacePreferences, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.preferences = preferences
        self.splitter: QtWidgets.QSplitter | None = None

    def apply_preferences(self, preferences: WorkspacePreferences) -> None:
        self.preferences = preferences
        self.retranslate()
        self.apply_detail_mode()

    def retranslate(self) -> None:
        """Aktualisiert alle Texte im Workspace."""

    def apply_detail_mode(self) -> None:
        """Schaltet Anfänger/Expertensicht um."""

    def restore_settings(self, settings: QtCore.QSettings) -> None:
        if self.splitter is None:
            pass
        else:
            data = settings.value(f"{self.workspace_id}/splitter")
            if data is not None:
                self.splitter.restoreState(data)

    def save_settings(self, settings: QtCore.QSettings) -> None:
        if self.splitter is not None:
            settings.setValue(f"{self.workspace_id}/splitter", self.splitter.saveState())
