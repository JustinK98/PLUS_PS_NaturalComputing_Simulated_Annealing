"""Plot-Helfer fuer Builder-Ergebnisse."""

from __future__ import annotations

from collections import defaultdict

from matplotlib.figure import Figure
import numpy as np


def build_experiment_overview_figure(
    summary_payload: dict,
    run_payloads: list[dict],
    selected_config_id: str | None = None,
    selected_run_id: str | None = None,
) -> Figure:
    """Erzeugt eine kompakte 2x2-Uebersicht fuer Builder-Ergebnisse."""

    figure = Figure(figsize=(10, 6), dpi=100)
    axes = [
        figure.add_subplot(221),
        figure.add_subplot(222),
        figure.add_subplot(223),
        figure.add_subplot(224),
    ]

    _plot_config_ranking(axes[0], summary_payload)
    _plot_aggregated_metrics(axes[1], summary_payload)
    _plot_selected_run_history(axes[2], run_payloads, selected_run_id)
    _plot_seed_scatter(axes[3], run_payloads, selected_config_id)

    figure.tight_layout()
    return figure


def _plot_config_ranking(axis, summary_payload: dict) -> None:
    ranking = summary_payload.get("ranking", [])
    axis.clear()
    if not ranking:
        axis.text(0.5, 0.5, "No ranking", ha="center", va="center")
        axis.set_axis_off()
        return

    labels = [entry["config_id"] for entry in ranking]
    scores = [float(entry["ranking_score"]) for entry in ranking]
    axis.bar(labels, scores, color="#2563eb")
    axis.set_title("Ranking Score")
    axis.tick_params(axis="x", rotation=40)
    axis.grid(alpha=0.25)


def _plot_aggregated_metrics(axis, summary_payload: dict) -> None:
    aggregated = summary_payload.get("aggregated_metrics", [])
    axis.clear()
    if not aggregated:
        axis.text(0.5, 0.5, "No aggregated data", ha="center", va="center")
        axis.set_axis_off()
        return

    labels = [entry["config_id"] for entry in aggregated]
    values = [float(entry["mean_metrics"].get("val_accuracy", 0.0)) for entry in aggregated]
    errors = [float(entry["std_metrics"].get("val_accuracy", 0.0)) for entry in aggregated]
    axis.bar(labels, values, yerr=errors, color="#16a34a", alpha=0.85)
    axis.set_title("Mean Validation Accuracy")
    axis.tick_params(axis="x", rotation=40)
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.25)


def _plot_selected_run_history(axis, run_payloads: list[dict], selected_run_id: str | None) -> None:
    axis.clear()
    if not run_payloads:
        axis.text(0.5, 0.5, "No runs", ha="center", va="center")
        axis.set_axis_off()
        return

    selected_payload = None
    if selected_run_id is not None:
        selected_payload = next(
            (
                payload
                for payload in run_payloads
                if payload["run_definition"]["run_id"] == selected_run_id
            ),
            None,
        )
    if selected_payload is None:
        selected_payload = run_payloads[0]

    history = selected_payload.get("history", {})
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    if train_loss and val_loss:
        epochs = np.arange(1, len(train_loss) + 1)
        axis.plot(epochs, train_loss, label="train_loss", linewidth=2)
        axis.plot(epochs, val_loss, label="val_loss", linewidth=2)
        axis.set_title("Selected Run History")
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
        return

    annealing_history = selected_payload.get("extra", {}).get("annealing_history", [])
    if annealing_history:
        step_indices = [entry["step_index"] for entry in annealing_history]
        scores = [entry["best_score_after_step"] for entry in annealing_history]
        axis.plot(step_indices, scores, linewidth=2, color="#7c3aed")
        axis.set_title("Selected SA Score")
        axis.set_xlabel("Step")
        axis.grid(alpha=0.25)
        return

    axis.text(0.5, 0.5, "No detailed history", ha="center", va="center")
    axis.set_axis_off()


def _plot_seed_scatter(axis, run_payloads: list[dict], selected_config_id: str | None) -> None:
    axis.clear()
    if not run_payloads:
        axis.text(0.5, 0.5, "No seed data", ha="center", va="center")
        axis.set_axis_off()
        return

    grouped = defaultdict(list)
    for payload in run_payloads:
        grouped[payload["run_definition"]["config_id"]].append(payload)

    config_id = selected_config_id or sorted(grouped)[0]
    selected_runs = grouped[config_id]
    seeds = [int(payload["run_definition"]["seed"]) for payload in selected_runs]
    val_acc = [float(payload["metrics"].get("val_accuracy", 0.0)) for payload in selected_runs]
    test_acc = [float(payload["metrics"].get("test_accuracy", 0.0)) for payload in selected_runs]
    axis.scatter(seeds, val_acc, label="val_acc", color="#0f766e", s=50)
    axis.scatter(seeds, test_acc, label="test_acc", color="#dc2626", s=50)
    axis.set_title(f"Seeds for {config_id}")
    axis.set_xlabel("Seed")
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.25)
    axis.legend()
