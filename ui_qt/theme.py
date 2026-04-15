"""Modernes, leichtes Theme fuer die Qt-Oberflaeche."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6 import QtGui, QtWidgets


@dataclass(frozen=True)
class ThemeTokens:
    background: str = "#eef3f9"
    surface: str = "#ffffff"
    surface_alt: str = "rgba(255, 255, 255, 0.78)"
    border: str = "#d9e4ef"
    text: str = "#16212e"
    text_muted: str = "#5b6978"
    accent: str = "#2563eb"
    accent_soft: str = "#dbeafe"
    success: str = "#059669"
    warning: str = "#d97706"
    danger: str = "#dc2626"


TOKENS = ThemeTokens()


def apply_theme(app: QtWidgets.QApplication) -> None:
    """Wendet das globale Light-Theme auf die Qt-Anwendung an."""

    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(TOKENS.background))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(TOKENS.text))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(TOKENS.surface))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#f8fbff"))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(TOKENS.surface))
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(TOKENS.text))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor(TOKENS.text))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(TOKENS.surface))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(TOKENS.text))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(TOKENS.accent))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))
    app.setPalette(palette)

    app.setStyleSheet(
        f"""
        QWidget {{
            color: {TOKENS.text};
            background: transparent;
            font-size: 13px;
        }}
        QMainWindow, QDialog {{
            background: {TOKENS.background};
        }}
        QFrame#CardFrame, QGroupBox {{
            background: {TOKENS.surface_alt};
            border: 1px solid {TOKENS.border};
            border-radius: 16px;
        }}
        QGroupBox {{
            margin-top: 10px;
            padding-top: 16px;
            font-weight: 600;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 4px;
        }}
        QPushButton {{
            background: {TOKENS.surface};
            border: 1px solid {TOKENS.border};
            border-radius: 12px;
            padding: 8px 12px;
        }}
        QPushButton:hover {{
            border-color: {TOKENS.accent};
            background: #f8fbff;
        }}
        QPushButton[role="primary"] {{
            background: {TOKENS.accent};
            color: white;
            border-color: {TOKENS.accent};
        }}
        QPushButton[role="primary"]:hover {{
            background: #1d4ed8;
        }}
        QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget, QTableView, QTreeView {{
            background: {TOKENS.surface};
            border: 1px solid {TOKENS.border};
            border-radius: 10px;
            padding: 6px 8px;
        }}
        QTabWidget::pane {{
            border: 1px solid {TOKENS.border};
            background: {TOKENS.surface_alt};
            border-radius: 14px;
            top: -1px;
        }}
        QTabBar::tab {{
            background: transparent;
            border: 1px solid transparent;
            padding: 8px 14px;
            color: {TOKENS.text_muted};
        }}
        QTabBar::tab:selected {{
            color: {TOKENS.accent};
            border-bottom: 2px solid {TOKENS.accent};
        }}
        QHeaderView::section {{
            background: #f7fafc;
            border: 0;
            border-bottom: 1px solid {TOKENS.border};
            padding: 8px;
            font-weight: 600;
        }}
        QSplitter::handle {{
            background: rgba(203, 213, 225, 0.35);
            border-radius: 4px;
        }}
        QSplitter::handle:hover {{
            background: rgba(148, 163, 184, 0.55);
        }}
        QSplitter::handle:horizontal {{
            width: 8px;
            margin: 0 2px;
        }}
        QSplitter::handle:vertical {{
            height: 8px;
            margin: 2px 0;
        }}
        QScrollBar:vertical {{
            width: 12px;
            background: transparent;
        }}
        QScrollBar::handle:vertical {{
            background: #cbd5e1;
            border-radius: 6px;
            min-height: 30px;
        }}
        QLabel[role="muted"] {{
            color: {TOKENS.text_muted};
        }}
        """
    )
