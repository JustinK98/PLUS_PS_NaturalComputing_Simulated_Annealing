"""Interaktiver Forward/Backward-Stepper fuer Qt."""

from __future__ import annotations

from PySide6 import QtWidgets

from services.stepper_service import StepperEntry


class StepperPanel(QtWidgets.QWidget):
    def __init__(self, language: str = "de", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._language = language
        self._entries: tuple[StepperEntry, ...] = ()
        self._index = 0

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        toolbar = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel()
        toolbar.addWidget(self.status_label, 1)
        self.prev_button = QtWidgets.QPushButton("Prev")
        self.next_button = QtWidgets.QPushButton("Next")
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.prev_button.clicked.connect(self.step_prev)
        self.next_button.clicked.connect(self.step_next)
        self.reset_button.clicked.connect(self.step_reset)
        toolbar.addWidget(self.prev_button)
        toolbar.addWidget(self.next_button)
        toolbar.addWidget(self.reset_button)
        layout.addLayout(toolbar)

        self.text = QtWidgets.QTextBrowser()
        layout.addWidget(self.text, 1)
        self.set_language(language)
        self.set_placeholder(language)

    def set_language(self, language: str) -> None:
        self._language = language
        self.prev_button.setText("Prev" if language == "en" else "Zurueck")
        self.next_button.setText("Next" if language == "en" else "Weiter")
        self.reset_button.setText("Reset" if language == "en" else "Reset")
        self._refresh()

    def set_placeholder(self, language: str | None = None) -> None:
        if language is not None:
            self._language = language
        self._entries = ()
        self._index = 0
        message = "No step trace available yet." if self._language == "en" else "Noch keine Schrittspur verfuegbar."
        self.text.setPlainText(message)
        self._refresh()

    def set_entries(self, entries: tuple[StepperEntry, ...], index: int = 0) -> None:
        self._entries = entries
        self._index = min(max(index, 0), max(0, len(entries) - 1))
        self._refresh()

    def step_prev(self) -> None:
        if not self._entries:
            return
        self._index = max(0, self._index - 1)
        self._refresh()

    def step_next(self) -> None:
        if not self._entries:
            return
        self._index = min(len(self._entries) - 1, self._index + 1)
        self._refresh()

    def step_reset(self) -> None:
        if not self._entries:
            return
        self._index = 0
        self._refresh()

    def _refresh(self) -> None:
        has_entries = bool(self._entries)
        self.prev_button.setEnabled(has_entries and self._index > 0)
        self.next_button.setEnabled(has_entries and self._index < len(self._entries) - 1)
        self.reset_button.setEnabled(has_entries and self._index != 0)
        if not has_entries:
            self.status_label.setText("Stepper" if self._language == "en" else "Stepper")
            return
        entry = self._entries[self._index]
        prefix = "Step" if self._language == "en" else "Schritt"
        self.status_label.setText(f"{prefix} {self._index + 1}/{len(self._entries)}: {entry.title}")
        self.text.setPlainText(entry.body)
