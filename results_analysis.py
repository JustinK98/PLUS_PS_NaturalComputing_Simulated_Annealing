"""Aggregation und Ranking fuer Builder-Experimente."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from experiment_builder import AggregatedMetrics, ExperimentSummary, RunResult


CORE_METRIC_KEYS = (
    "train_loss",
    "val_loss",
    "test_loss",
    "train_accuracy",
    "val_accuracy",
    "test_accuracy",
)

SA_EXTRA_METRIC_KEYS = ("best_objective", "acceptance_rate", "step_count")


def comparable_metric_value(metrics: dict[str, float], primary_metric: str) -> float:
    """Einheitlicher Vergleichswert fuer Ranking und Best-/Worst-Suche."""

    if primary_metric == "validation_loss":
        return float(metrics["val_loss"])
    if primary_metric == "validation_accuracy":
        return 1.0 - float(metrics["val_accuracy"])
    raise ValueError(f"Unbekannte primäre Metrik '{primary_metric}'.")


def aggregate_run_results(
    run_results: Iterable[RunResult],
    benchmark: str,
    run_mode: str,
    search_type: str,
    primary_metric: str,
    experiment_id: str,
) -> ExperimentSummary:
    """Aggregiert Builder-Run-Ergebnisse pro Konfiguration."""

    grouped_results: dict[str, list[RunResult]] = defaultdict(list)
    for run_result in run_results:
        grouped_results[run_result.run_definition.config_id].append(run_result)

    aggregated_entries: list[AggregatedMetrics] = []
    ranking_entries: list[dict[str, object]] = []

    for config_id, config_runs in sorted(grouped_results.items()):
        first_run = config_runs[0]
        metric_keys = list(CORE_METRIC_KEYS)
        for extra_key in SA_EXTRA_METRIC_KEYS:
            if extra_key in first_run.metrics:
                metric_keys.append(extra_key)

        values_by_metric = {
            metric_key: np.asarray(
                [float(run_result.metrics[metric_key]) for run_result in config_runs],
                dtype=np.float64,
            )
            for metric_key in metric_keys
        }

        comparable_values = np.asarray(
            [
                comparable_metric_value(run_result.metrics, primary_metric)
                for run_result in config_runs
            ],
            dtype=np.float64,
        )
        best_index = int(np.argmin(comparable_values))
        worst_index = int(np.argmax(comparable_values))
        ranking_score = float(np.mean(comparable_values))

        aggregated = AggregatedMetrics(
            config_id=config_id,
            config_index=first_run.run_definition.config_index,
            config_values=dict(first_run.run_definition.config_values),
            num_runs=len(config_runs),
            num_seeds=len({run_result.run_definition.seed for run_result in config_runs}),
            mean_metrics={
                metric_key: float(np.mean(metric_values))
                for metric_key, metric_values in values_by_metric.items()
            },
            std_metrics={
                metric_key: float(np.std(metric_values))
                for metric_key, metric_values in values_by_metric.items()
            },
            min_metrics={
                metric_key: float(np.min(metric_values))
                for metric_key, metric_values in values_by_metric.items()
            },
            max_metrics={
                metric_key: float(np.max(metric_values))
                for metric_key, metric_values in values_by_metric.items()
            },
            best_seed=int(config_runs[best_index].run_definition.seed),
            worst_seed=int(config_runs[worst_index].run_definition.seed),
            ranking_score=ranking_score,
        )
        aggregated_entries.append(aggregated)
        ranking_entries.append(
            {
                "config_id": config_id,
                "config_index": aggregated.config_index,
                "ranking_score": ranking_score,
                "config_values": dict(aggregated.config_values),
                "mean_val_loss": aggregated.mean_metrics.get("val_loss"),
                "mean_val_accuracy": aggregated.mean_metrics.get("val_accuracy"),
                "mean_test_accuracy": aggregated.mean_metrics.get("test_accuracy"),
                "best_seed": aggregated.best_seed,
                "worst_seed": aggregated.worst_seed,
            }
        )

    ranking_entries.sort(key=lambda entry: float(entry["ranking_score"]))
    return ExperimentSummary(
        experiment_id=experiment_id,
        benchmark=benchmark,
        run_mode=run_mode,
        search_type=search_type,
        primary_metric=primary_metric,
        number_of_runs=sum(len(runs) for runs in grouped_results.values()),
        number_of_seeds=len({run_result.run_definition.seed for runs in grouped_results.values() for run_result in runs}),
        configuration_count=len(aggregated_entries),
        aggregated_metrics=tuple(sorted(aggregated_entries, key=lambda entry: entry.ranking_score)),
        ranking=tuple(ranking_entries),
    )
