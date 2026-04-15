"""Nicht-modaler Dialog fuer Handbuch- und Hilfetexte."""

from __future__ import annotations

from PySide6 import QtWidgets


class HelpDialog(QtWidgets.QDialog):
    """Zeigt strukturierte HTML-Hilfe in einem separaten Fenster."""

    def __init__(
        self,
        title: str,
        html: str,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self.resize(920, 760)

        layout = QtWidgets.QVBoxLayout(self)
        self.browser = QtWidgets.QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        layout.addWidget(self.browser, 1)

        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)

        self.set_content(title, html)

    def set_content(self, title: str, html: str) -> None:
        self.setWindowTitle(title)
        self.browser.setHtml(html)
