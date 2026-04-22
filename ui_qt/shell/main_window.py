"""Hauptfenster fuer die Qt-Version des Activation Playground."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from configs import GuiExperimentConfig
from services.help_service import program_handbook_html
from ui_qt.state import AppState, WorkspacePreferences
from ui_qt.texts import text
from ui_qt.widgets.help_dialog import HelpDialog
from ui_qt.workspaces import (
    DemoWorkspace,
    ExperimentBuilderWorkspace,
    PlaygroundWorkspace,
    PresentationWorkspace,
)


class MainWindow(QtWidgets.QMainWindow):
    """Top-Level-Fenster mit Workspace-Router und globalen Präferenzen."""

    def __init__(self, config: GuiExperimentConfig, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Activation Playground")
        self.resize(1560, 980)
        self.settings = QtCore.QSettings("ActivationPlayground", "ActivationPlaygroundQt")
        self.state = AppState(
            workspace=config.app_mode,
            preferences=WorkspacePreferences(language=config.language, detail_mode=config.mode),
        )
        self.handbook_dialog: HelpDialog | None = None

        self.workspace_combo = QtWidgets.QComboBox()
        self.workspace_combo.addItems(("presentation", "demo", "playground", "experiment_builder"))
        self.workspace_combo.setCurrentText(config.app_mode)
        self.workspace_combo.currentTextChanged.connect(self._on_workspace_changed)

        self.language_combo = QtWidgets.QComboBox()
        self.language_combo.addItems(("de", "en"))
        self.language_combo.setCurrentText(config.language)
        self.language_combo.currentTextChanged.connect(self._on_language_changed)

        self.detail_combo = QtWidgets.QComboBox()
        self.detail_combo.addItems(("beginner", "expert"))
        self.detail_combo.setCurrentText(config.mode)
        self.detail_combo.currentTextChanged.connect(self._on_detail_changed)

        self.guide_button = QtWidgets.QPushButton()
        self.guide_button.clicked.connect(self._open_handbook)

        toolbar = QtWidgets.QToolBar()
        toolbar.setMovable(False)
        toolbar.addWidget(QtWidgets.QLabel("Workspace"))
        toolbar.addWidget(self.workspace_combo)
        toolbar.addSeparator()
        toolbar.addWidget(QtWidgets.QLabel("Language"))
        toolbar.addWidget(self.language_combo)
        toolbar.addSeparator()
        toolbar.addWidget(QtWidgets.QLabel("Detail"))
        toolbar.addWidget(self.detail_combo)
        toolbar.addSeparator()
        toolbar.addWidget(self.guide_button)
        self.addToolBar(toolbar)

        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)
        self.workspaces = {
            "presentation": PresentationWorkspace(config, self.state.preferences, self),
            "demo": DemoWorkspace(config, self.state.preferences, self),
            "playground": PlaygroundWorkspace(config, self.state.preferences, self),
            "experiment_builder": ExperimentBuilderWorkspace(config, self.state.preferences, self),
        }
        for workspace_id, workspace in self.workspaces.items():
            workspace.statusMessage.connect(self.statusBar().showMessage)
            workspace.busyChanged.connect(self._on_busy_changed)
            self.stack.addWidget(workspace)
            workspace.restore_settings(self.settings)

        self._retranslate_toolbar()
        self._set_current_workspace(config.app_mode)
        self.statusBar().showMessage(text(self.state.preferences.language, "shell.status.ready"))

    def closeEvent(self, event) -> None:  # pragma: no cover - GUI lifecycle
        for workspace in self.workspaces.values():
            workspace.save_settings(self.settings)
        self.settings.setValue("workspace", self.state.workspace)
        self.settings.setValue("language", self.state.preferences.language)
        self.settings.setValue("detail", self.state.preferences.detail_mode)
        super().closeEvent(event)

    def _retranslate_toolbar(self) -> None:
        self.workspace_combo.setToolTip(text(self.state.preferences.language, "shell.workspace"))
        self.language_combo.setToolTip(text(self.state.preferences.language, "shell.language"))
        self.detail_combo.setToolTip(text(self.state.preferences.language, "shell.detail"))
        self.guide_button.setText(text(self.state.preferences.language, "shell.guide"))
        self.guide_button.setToolTip(text(self.state.preferences.language, "shell.guide"))

    def _set_current_workspace(self, workspace_id: str) -> None:
        self.state.workspace = workspace_id
        self.stack.setCurrentWidget(self.workspaces[workspace_id])
        self.workspace_combo.blockSignals(True)
        self.workspace_combo.setCurrentText(workspace_id)
        self.workspace_combo.blockSignals(False)

    def _on_workspace_changed(self, workspace_id: str) -> None:
        self._set_current_workspace(workspace_id)

    def _on_language_changed(self, language: str) -> None:
        self.state.preferences.language = language
        self._retranslate_toolbar()
        for workspace in self.workspaces.values():
            workspace.apply_preferences(self.state.preferences)
        if self.handbook_dialog is not None:
            self.handbook_dialog.set_content(
                text(language, "shell.guide"),
                program_handbook_html(language),
            )
        self.statusBar().showMessage(text(language, "shell.status.ready"))

    def _on_detail_changed(self, detail_mode: str) -> None:
        self.state.preferences.detail_mode = detail_mode
        for workspace in self.workspaces.values():
            workspace.apply_preferences(self.state.preferences)

    def _on_busy_changed(self, busy: bool) -> None:
        self.setCursor(QtCore.Qt.WaitCursor if busy else QtCore.Qt.ArrowCursor)

    def _open_handbook(self) -> None:
        title = text(self.state.preferences.language, "shell.guide")
        html = program_handbook_html(self.state.preferences.language)
        if self.handbook_dialog is None:
            self.handbook_dialog = HelpDialog(title, html, self)
            self.handbook_dialog.finished.connect(self._on_handbook_closed)
        else:
            self.handbook_dialog.set_content(title, html)
        self.handbook_dialog.show()
        self.handbook_dialog.raise_()
        self.handbook_dialog.activateWindow()

    def _on_handbook_closed(self, _result: int) -> None:
        self.handbook_dialog = None
