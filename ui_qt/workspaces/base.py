"""Gemeinsame Basisklasse fuer Qt-Workspaces."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ui_qt.state import WorkspacePreferences
from ui_qt.widgets.info_button import InfoButton


class BaseWorkspace(QtWidgets.QWidget):
    """Basisklasse mit Signalen fuer Status- und Busy-Zustände."""

    statusMessage = QtCore.Signal(str)
    busyChanged = QtCore.Signal(bool)

    workspace_id = "base"

    def __init__(self, preferences: WorkspacePreferences, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.preferences = preferences
        self.splitter: QtWidgets.QSplitter | None = None
        self._secondary_splitters: dict[str, QtWidgets.QSplitter] = {}
        self._info_buttons: list[InfoButton] = []

    def register_secondary_splitter(self, name: str, splitter: QtWidgets.QSplitter) -> None:
        self._secondary_splitters[name] = splitter

    def apply_preferences(self, preferences: WorkspacePreferences) -> None:
        self.preferences = preferences
        for button in self._info_buttons:
            button.set_language(self.preferences.language)
        self.retranslate()
        self.apply_detail_mode()

    def make_info_button(self, topic_key: str) -> InfoButton:
        button = InfoButton(topic_key, self.preferences.language, self)
        self._info_buttons.append(button)
        return button

    def make_help_label(self, text: str, topic_key: str) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        layout.addWidget(label)
        layout.addWidget(self.make_info_button(topic_key))
        layout.addStretch(1)
        return container

    def make_help_strip(self, topic_key: str) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch(1)
        layout.addWidget(self.make_info_button(topic_key))
        return container

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
        for name, splitter in self._secondary_splitters.items():
            data = settings.value(f"{self.workspace_id}/splitter/{name}")
            if data is not None:
                splitter.restoreState(data)

    def save_settings(self, settings: QtCore.QSettings) -> None:
        if self.splitter is not None:
            settings.setValue(f"{self.workspace_id}/splitter", self.splitter.saveState())
        for name, splitter in self._secondary_splitters.items():
            settings.setValue(f"{self.workspace_id}/splitter/{name}", splitter.saveState())
