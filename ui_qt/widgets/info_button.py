"""Kleiner Hilfebutton fuer lokale didaktische Erklaerungen."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from services.help_service import topic_html, topic_title
from ui_qt.widgets.help_dialog import HelpDialog


class InfoButton(QtWidgets.QToolButton):
    """Oeffnet eine kurze themenspezifische Hilfe in einem eigenen Fenster."""

    def __init__(
        self,
        topic_key: str,
        language: str,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.topic_key = topic_key
        self.language = language
        self._dialog: HelpDialog | None = None
        self.setText("?")
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setAutoRaise(True)
        self.setFixedSize(22, 22)
        self.clicked.connect(self._open_dialog)
        self._refresh_text()

    def set_language(self, language: str) -> None:
        self.language = language
        self._refresh_text()
        if self._dialog is not None:
            self._dialog.set_content(topic_title(self.topic_key, self.language), topic_html(self.topic_key, self.language))

    def set_topic_key(self, topic_key: str) -> None:
        self.topic_key = topic_key
        self._refresh_text()
        if self._dialog is not None:
            self._dialog.set_content(topic_title(self.topic_key, self.language), topic_html(self.topic_key, self.language))

    def _refresh_text(self) -> None:
        self.setToolTip(topic_title(self.topic_key, self.language))

    def _open_dialog(self) -> None:
        title = topic_title(self.topic_key, self.language)
        html = topic_html(self.topic_key, self.language)
        if self._dialog is None:
            self._dialog = HelpDialog(title, html, self.window())
            self._dialog.finished.connect(self._on_dialog_closed)
        else:
            self._dialog.set_content(title, html)
        self._dialog.show()
        self._dialog.raise_()
        self._dialog.activateWindow()

    def _on_dialog_closed(self, _result: int) -> None:
        self._dialog = None
