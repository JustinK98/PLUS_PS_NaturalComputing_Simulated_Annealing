"""Kleine Zusammenfassungskarten im Glass-Akzent-Look."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class InfoCardWidget(QtWidgets.QFrame):
    """Verdichtet eine Kennzahl oder einen Status in einer Karte."""

    def __init__(self, title: str, value: str = "-", body: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("CardFrame")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        self.title_label = QtWidgets.QLabel(title)
        self.title_label.setProperty("role", "muted")
        self.value_label = QtWidgets.QLabel(value)
        font = self.value_label.font()
        font.setPointSize(font.pointSize() + 5)
        font.setBold(True)
        self.value_label.setFont(font)
        self.body_label = QtWidgets.QLabel(body)
        self.body_label.setWordWrap(True)
        self.body_label.setProperty("role", "muted")
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.body_label)
        layout.addStretch(1)

    def set_content(self, title: str | None = None, value: str | None = None, body: str | None = None) -> None:
        if title is not None:
            self.title_label.setText(title)
        if value is not None:
            self.value_label.setText(value)
        if body is not None:
            self.body_label.setText(body)
