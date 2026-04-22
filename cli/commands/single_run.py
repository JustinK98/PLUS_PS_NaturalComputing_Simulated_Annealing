"""Single-Run-CLI ueber die neue Service-Schicht."""

from __future__ import annotations

from pathlib import Path

from activations import generate_single_step_neighbors
from configs import DatasetConfig, TrainingConfig, VisualizationConfig, default_hidden_sizes
from runtime_env import configure_runtime_environment
from services.training_service import TrainingRunRequest, run_single_training_experiment
from terminal_viz import (
    render_dataset_summary,
    render_layout,
    render_layout_diff,
    render_neighbor_preview,
    render_training_summary,
)


def run_single_command(args) -> None:
    """Fuehrt einen einzelnen Trainingslauf im Terminal aus."""

    hidden_sizes = tuple(args.hidden_sizes) if args.hidden_sizes else default_hidden_sizes(args.benchmark)
    artifacts = run_single_training_experiment(
        TrainingRunRequest(
            dataset_config=DatasetConfig(name=args.benchmark, random_state=args.seed),
            hidden_sizes=hidden_sizes,
            layout_spec=args.layout,
            training_config=TrainingConfig(
                epochs=args.epochs,
                learning_rate=args.lr,
                batch_size=args.batch_size,
                random_state=args.seed,
            ),
            weight_scale=args.weight_scale,
            random_state=args.seed,
            neighbor_operations=tuple(args.neighbor_op),
        )
    )

    visualization_config = VisualizationConfig(
        preview_neighbors=args.show_neighbors,
        show_plots=not args.no_plot,
        save_prefix=args.save_prefix,
    )

    print(render_dataset_summary(artifacts.dataset))
    print()
    print(render_layout(artifacts.base_layout, title="Basis-Layout"))

    if artifacts.neighbor_result is not None:
        print()
        print(render_layout(artifacts.neighbor_result.layout, title="Nachbar-Layout"))
        print()
        print(render_layout_diff(artifacts.base_layout, artifacts.neighbor_result.layout))

    if visualization_config.preview_neighbors > 0:
        neighbors = generate_single_step_neighbors(artifacts.training_layout)
        print()
        print(render_neighbor_preview(neighbors, visualization_config.preview_neighbors))

    print()
    print(f"Trainiertes Layout: {artifacts.training_layout.to_compact_spec()}")
    print()
    print(
        render_training_summary(
            artifacts.training_result.history,
            artifacts.training_result.test_metrics,
        )
    )

    if visualization_config.show_plots or visualization_config.save_prefix:
        _show_or_save_plots(
            benchmark_name=artifacts.dataset.name,
            base_layout=artifacts.base_layout,
            neighbor_layout=(
                artifacts.neighbor_result.layout if artifacts.neighbor_result is not None else None
            ),
            history=artifacts.training_result.history,
            visualization_config=visualization_config,
        )


def _show_or_save_plots(
    benchmark_name: str,
    base_layout,
    neighbor_layout,
    history: dict[str, list[float]],
    visualization_config: VisualizationConfig,
) -> None:
    """Erzeugt Matplotlib-Plots und speichert oder zeigt sie an."""

    configure_runtime_environment()

    import matplotlib.pyplot as plt

    from plotting import plot_layouts, plot_training_history, save_figure

    history_figure = plot_training_history(history, benchmark_name)
    layout_figure = plot_layouts(base_layout, neighbor_layout)

    if visualization_config.save_prefix:
        prefix = Path(visualization_config.save_prefix)
        history_path = save_figure(history_figure, prefix.parent / f"{prefix.name}_history.png")
        layout_path = save_figure(layout_figure, prefix.parent / f"{prefix.name}_layouts.png")
        print()
        print(f"Plots gespeichert: {history_path}")
        print(f"Plots gespeichert: {layout_path}")

    if visualization_config.show_plots:
        backend_name = plt.get_backend().lower()
        if "agg" in backend_name:
            print()
            print(
                "Hinweis: Das aktive Matplotlib-Backend ist 'Agg'. "
                "GUI-Fenster werden daher nicht angezeigt."
            )
            print("Nutze `--save-prefix`, um die Plots als PNG-Dateien zu erhalten.")
            plt.close(history_figure)
            plt.close(layout_figure)
        else:
            plt.show()
    else:
        plt.close(history_figure)
        plt.close(layout_figure)
