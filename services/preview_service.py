"""Textuelle Zusammenfassungen fuer Builder- und Ergebnisansichten."""

from __future__ import annotations

from typing import Any


def format_experiment_summary(payload: dict[str, Any], max_ranking_entries: int = 5) -> str:
    """Erzeugt eine kompakte, CLI-taugliche Analyse eines gespeicherten Experiments."""

    manifest = payload["manifest"]
    summary = payload["summary"]
    lines = [
        f"experiment_id: {manifest['experiment_id']}",
        f"benchmark:     {manifest['benchmark']}",
        f"run_mode:      {manifest['run_mode']}",
        f"search_type:   {manifest['search_type']}",
        f"primary_metric:{manifest['primary_metric']}",
        f"runs:          {summary['number_of_runs']}",
        f"seeds:         {summary['number_of_seeds']}",
        f"configs:       {summary['configuration_count']}",
        "",
        "Top ranking entries",
        "-------------------",
    ]
    for entry in summary.get("ranking", [])[:max_ranking_entries]:
        lines.append(
            f"{entry['config_id']}: score={entry['ranking_score']:.6f}, "
            f"mean_val_acc={entry.get('mean_val_accuracy', 0.0):.4f}, "
            f"mean_val_loss={entry.get('mean_val_loss', 0.0):.4f}"
        )
    return "\n".join(lines)


def format_run_summary(run_payload: dict[str, Any]) -> str:
    """Erzeugt eine knappe Zusammenfassung eines einzelnen gespeicherten Runs."""

    run_definition = run_payload["run_definition"]
    metrics = run_payload["metrics"]
    return "\n".join(
        [
            f"run_id:      {run_definition['run_id']}",
            f"config_id:   {run_definition['config_id']}",
            f"seed:        {run_definition['seed']}",
            f"benchmark:   {run_definition['benchmark']}",
            f"run_mode:    {run_definition['run_mode']}",
            f"layout_spec: {run_payload['layout_spec']}",
            "",
            "metrics",
            "-------",
            *(f"{key}: {value:.6f}" for key, value in sorted(metrics.items())),
        ]
    )

