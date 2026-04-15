"""Qt-Anwendungsstart fuer den Activation Playground."""

from __future__ import annotations

import os
import sys

from PySide6 import QtCore, QtWidgets

from configs import GuiExperimentConfig
from ui_qt.shell.main_window import MainWindow
from ui_qt.theme import apply_theme


def launch_qt_gui(config: GuiExperimentConfig) -> None:
    """Startet die Qt-Oberfläche und blockiert bis zum Schließen."""

    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".mplconfig"))
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        QtCore.QCoreApplication.setOrganizationName("ActivationPlayground")
        QtCore.QCoreApplication.setApplicationName("ActivationPlaygroundQt")
        app = QtWidgets.QApplication(sys.argv)
    apply_theme(app)
    window = MainWindow(config)
    window.show()
    auto_quit_ms = os.environ.get("ACTIVATION_PLAYGROUND_AUTO_QUIT_MS")
    if auto_quit_ms:
        QtCore.QTimer.singleShot(int(auto_quit_ms), app.quit)
    if owns_app:
        app.exec()
