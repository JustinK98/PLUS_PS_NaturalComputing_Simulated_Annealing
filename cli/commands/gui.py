"""GUI-Kommando fuer den modularisierten CLI-Dispatcher."""

from __future__ import annotations

from configs import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WEIGHT_SCALE,
    GuiExperimentConfig,
    default_epochs,
)
from ui_qt import launch_qt_gui


def run_gui_command(args) -> None:
    """Startet die konfigurierte Qt-GUI."""

    gui_config = GuiExperimentConfig(
        benchmark=args.benchmark,
        epochs=default_epochs(args.benchmark),
        learning_rate=DEFAULT_LEARNING_RATE,
        batch_size=DEFAULT_BATCH_SIZE,
        weight_scale=DEFAULT_WEIGHT_SCALE,
        random_state=args.seed,
        mode="beginner",
        language="de",
        gui_profile=args.gui_profile,
    )
    launch_qt_gui(gui_config)
