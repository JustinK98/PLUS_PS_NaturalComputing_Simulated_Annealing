"""GUI-Kommando fuer den modularisierten CLI-Dispatcher."""

from __future__ import annotations

from configs import GuiExperimentConfig, default_hidden_sizes
from ui_qt import launch_qt_gui


def run_gui_command(args) -> None:
    """Startet die konfigurierte Qt-GUI."""

    hidden_sizes = tuple(args.hidden_sizes) if args.hidden_sizes else default_hidden_sizes(args.benchmark)
    gui_config = GuiExperimentConfig(
        benchmark=args.benchmark,
        hidden_sizes=hidden_sizes,
        app_mode=args.gui_app_mode,
        layout_spec=args.layout,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        weight_scale=args.weight_scale,
        random_state=args.seed,
        mode=args.gui_mode,
        language=args.gui_language,
    )
    launch_qt_gui(gui_config)
