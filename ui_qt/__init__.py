"""Qt-basierte GUI-Schicht fuer den Activation Playground."""

from runtime_env import configure_runtime_environment

configure_runtime_environment()

from .app import launch_qt_gui

__all__ = ["launch_qt_gui"]
