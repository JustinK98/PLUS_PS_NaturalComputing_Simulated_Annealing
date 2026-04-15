"""Geführte Experimentrezepte für die Qt-GUI."""

from __future__ import annotations

from PySide6 import QtWidgets

from ui_qt.texts import RECIPES


class RecipePanelWidget(QtWidgets.QWidget):
    """Zeigt auswählbare Rezeptkarten mit Detailtext."""

    def __init__(self, language: str = "de", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.language = language
        layout = QtWidgets.QHBoxLayout(self)
        self.list_widget = QtWidgets.QListWidget()
        self.detail_browser = QtWidgets.QTextBrowser()
        layout.addWidget(self.list_widget, 1)
        layout.addWidget(self.detail_browser, 2)
        self.list_widget.currentRowChanged.connect(self._update_detail)
        self.apply_language(language)

    def apply_language(self, language: str) -> None:
        self.language = language
        self.list_widget.clear()
        for recipe_key, payload in RECIPES.items():
            item = QtWidgets.QListWidgetItem(payload["title"][language])
            item.setData(QtWidgets.QListWidgetItem.UserType, recipe_key)
            self.list_widget.addItem(item)
        if self.list_widget.count():
            self.list_widget.setCurrentRow(0)

    def _update_detail(self, row: int) -> None:
        item = self.list_widget.item(row)
        if item is None:
            self.detail_browser.clear()
            return
        recipe_key = item.data(QtWidgets.QListWidgetItem.UserType)
        payload = RECIPES[recipe_key]
        self.detail_browser.setHtml(
            f"<h3>{payload['title'][self.language]}</h3><p>{payload['body'][self.language]}</p>"
        )
