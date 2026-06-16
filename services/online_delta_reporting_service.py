"""Aggregated reporting for Online-Delta-SA artifacts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from configs import SUPPORTED_ACTIVATIONS


@dataclass(frozen=True)
class OnlineDeltaRun:
    """Normalized Online-Delta suite-run payload."""

    source_path: Path
    run_id: str
    experiment_id: str
    benchmark: str
    schema_version: int
    seed: int | None
    layout_seed: int | None
    layout_index: int | None
    replicate_index: int | None
    seeds: dict[str, object]
    learning_rate: float | None
    start_layout: str
    diagnostic_best_layout: str
    end_layout: str
    trained_batch_updates: int | None
    effective_online_epochs: float | None
    history: tuple[dict[str, object], ...]
    final_comparisons: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class OnlineDeltaReportResult:
    """Result object returned to the CLI."""

    input_path: Path
    output_dir: Path
    run_count: int
    artifacts: tuple[Path, ...]
    warnings: tuple[str, ...]


def build_online_delta_report(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    include_plots: bool = True,
) -> OnlineDeltaReportResult:
    """Build aggregate plots and CSVs from Online-Delta-SA run JSON files."""

    root = Path(input_path)
    if output_dir is not None:
        report_dir = Path(output_dir)
    elif root.is_file():
        report_dir = root.parent / f"{root.stem}_online_delta_report"
    else:
        report_dir = root / "online_delta_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    runs, warnings = _load_runs(root)
    artifacts: list[Path] = []
    if not runs:
        warning_path = report_dir / "warnings.txt"
        warning_path.write_text(
            "No Online-Delta histories found.\n",
            encoding="utf-8",
        )
        return OnlineDeltaReportResult(
            input_path=root,
            output_dir=report_dir,
            run_count=0,
            artifacts=(warning_path,),
            warnings=(*warnings, "No Online-Delta histories found."),
        )

    steps_path = report_dir / "online_delta_steps.csv"
    runs_path = report_dir / "online_delta_runs.csv"
    activation_counts_path = report_dir / "activation_counts_end.csv"
    similarity_path = report_dir / "layout_similarity_end.csv"
    paired_runs_path = report_dir / "paired_runs.csv"
    layout_summary_path = report_dir / "layout_level_summary.csv"
    benchmark_summary_path = report_dir / "benchmark_summary.csv"
    seed_manifest_path = report_dir / "seed_manifest.csv"
    activation_layout_features_path = report_dir / "activation_layout_features.csv"
    activation_effect_summary_path = report_dir / "activation_effect_summary.csv"

    _write_csv(steps_path, _step_rows(runs))
    _write_csv(runs_path, _run_rows(runs))
    _write_csv(activation_counts_path, _activation_count_rows(runs))
    _write_csv(similarity_path, _layout_similarity_rows(runs))
    paired_rows = _paired_run_rows(runs)
    _write_csv(paired_runs_path, paired_rows)
    _write_csv(layout_summary_path, _layout_level_summary_rows(paired_rows))
    _write_csv(benchmark_summary_path, _benchmark_summary_rows(paired_rows))
    _write_csv(seed_manifest_path, _seed_manifest_rows(runs))
    activation_layout_features = _activation_layout_feature_rows(runs, paired_rows)
    activation_effect_summary = _activation_effect_summary_rows(activation_layout_features)
    _write_csv(activation_layout_features_path, activation_layout_features)
    _write_csv(activation_effect_summary_path, activation_effect_summary)
    artifacts.extend(
        [
            steps_path,
            runs_path,
            activation_counts_path,
            similarity_path,
            paired_runs_path,
            layout_summary_path,
            benchmark_summary_path,
            seed_manifest_path,
            activation_layout_features_path,
            activation_effect_summary_path,
        ]
    )

    if include_plots:
        artifacts.append(_plot_fitness_progress(report_dir / "online_delta_fitness_mean.png", runs))
        artifacts.append(
            _plot_validation_progress(
                report_dir / "online_delta_validation_progress_mean.png",
                runs,
            )
        )
        artifacts.append(_plot_acceptance_rate(report_dir / "online_delta_acceptance_rate.png", runs))
        artifacts.append(_plot_delta_distribution(report_dir / "online_delta_delta_distribution.png", runs))
        artifacts.append(_plot_temperature(report_dir / "online_delta_temperature.png", runs))
        artifacts.append(_plot_activation_counts(report_dir / "activation_counts_end.png", runs))
        artifacts.append(_plot_paired_comparison(report_dir / "paired_random_vs_sa_end.png", paired_rows))
        artifacts.append(
            _plot_improvement_by_layout(
                report_dir / "paired_improvement_by_layout.png",
                paired_rows,
            )
        )
        artifacts.append(
            _plot_improvement_distribution(
                report_dir / "paired_improvement_distribution.png",
                paired_rows,
            )
        )
        artifacts.append(
            _plot_activation_effect_association(
                report_dir / "activation_effect_association.png",
                activation_effect_summary,
            )
        )

    if warnings:
        warning_path = report_dir / "warnings.txt"
        warning_path.write_text("\n".join(warnings) + "\n", encoding="utf-8")
        artifacts.append(warning_path)
    else:
        (report_dir / "warnings.txt").unlink(missing_ok=True)

    return OnlineDeltaReportResult(
        input_path=root,
        output_dir=report_dir,
        run_count=len(runs),
        artifacts=tuple(artifacts),
        warnings=tuple(warnings),
    )


def _load_runs(root: Path) -> tuple[list[OnlineDeltaRun], tuple[str, ...]]:
    warnings: list[str] = []
    paths = [root] if root.is_file() else sorted(root.rglob("*.json"))
    runs: list[OnlineDeltaRun] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"Skipped unreadable JSON {path}: {exc}")
            continue
        if not isinstance(payload, dict):
            continue
        normalized = _normalize_payload(path, payload, warnings)
        if normalized is not None:
            runs.append(normalized)
    return runs, tuple(warnings)


def _normalize_payload(
    path: Path,
    payload: dict[str, Any],
    warnings: list[str],
) -> OnlineDeltaRun | None:
    if payload.get("kind") == "experiment_suite_online_delta_run" or "online_history" in payload:
        schema_version = int(payload.get("schema_version", 1))
        if schema_version < 2:
            warnings.append(f"{path}: legacy Online-Delta schema v1.")
        history = _normalize_history(payload.get("online_history", []))
        if not history:
            warnings.append(f"{path}: Online-Delta suite run has no usable history.")
            return None
        return OnlineDeltaRun(
            source_path=path,
            run_id=str(payload.get("run_id", payload.get("run_index", path.stem))),
            experiment_id=str(payload.get("experiment_id", "")),
            benchmark=str(payload.get("benchmark", "")),
            schema_version=schema_version,
            seed=_optional_int(payload.get("training_seed", payload.get("seed"))),
            layout_seed=_optional_int(payload.get("layout_seed")),
            layout_index=_optional_int(payload.get("layout_index")),
            replicate_index=_optional_int(payload.get("replicate_index")),
            seeds=dict(payload.get("seeds", {})) if isinstance(payload.get("seeds"), dict) else {},
            learning_rate=_optional_float(payload.get("learning_rate")),
            start_layout=str(payload.get("start_layout", "")),
            diagnostic_best_layout=str(
                payload.get("diagnostic_best_layout", payload.get("best_layout", ""))
            ),
            end_layout=str(payload.get("end_layout", "")),
            trained_batch_updates=_optional_int(payload.get("trained_batch_updates")),
            effective_online_epochs=_optional_float(payload.get("effective_online_epochs")),
            history=tuple(history),
            final_comparisons=tuple(
                item for item in payload.get("final_comparisons", []) if isinstance(item, dict)
            ),
        )

    return None


def _normalize_history(history: object) -> list[dict[str, object]]:
    if not isinstance(history, list):
        return []
    rows: list[dict[str, object]] = []
    for index, item in enumerate(history):
        if not isinstance(item, dict):
            continue
        step_index = _optional_int(item.get("step_index"))
        if step_index is None:
            step_index = index
        val_loss = _optional_float(item.get("validation_loss_after_update"))
        if val_loss is None:
            continue
        rows.append(
            {
                "step_index": step_index,
                "validation_loss_after_update": val_loss,
                "accepted": bool(item.get("accepted", False)),
                "delta": _optional_float(item.get("delta", item.get("loss_delta"))),
                "temperature": _optional_float(item.get("temperature")),
                "post_training_batch_loss": _optional_float(
                    item.get("post_training_batch_loss", item.get("candidate_loss_after"))
                ),
                "trained_batch_updates_after_step": _optional_int(
                    item.get("trained_batch_updates_after_step")
                ),
                "effective_online_epochs_after_step": _optional_float(
                    item.get("effective_online_epochs_after_step")
                ),
                "diagnostics_refreshed": bool(item.get("diagnostics_refreshed", True)),
                "neighbor_label": str(item.get("neighbor_label", "")),
                "reason_code": str(item.get("reason_code", "")),
            }
        )
    return rows


def _step_rows(runs: list[OnlineDeltaRun]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in runs:
        for step in run.history:
            rows.append(
                {
                    "source_path": str(run.source_path),
                    "run_id": run.run_id,
                    "experiment_id": run.experiment_id,
                    "benchmark": run.benchmark,
                    "seed": "" if run.seed is None else run.seed,
                    "layout_seed": "" if run.layout_seed is None else run.layout_seed,
                    "layout_index": "" if run.layout_index is None else run.layout_index,
                    "replicate_index": "" if run.replicate_index is None else run.replicate_index,
                    "learning_rate": "" if run.learning_rate is None else run.learning_rate,
                    **step,
                }
            )
    return rows


def _run_rows(runs: list[OnlineDeltaRun]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in runs:
        accepted_values = [
            bool(step["accepted"])
            for step in run.history
            if int(step["step_index"]) != 0
        ]
        final_step = run.history[-1]
        start_step = run.history[0]
        start_val_loss = float(start_step["validation_loss_after_update"])
        final_val_loss = float(final_step["validation_loss_after_update"])
        rows.append(
            {
                "source_path": str(run.source_path),
                "run_id": run.run_id,
                "experiment_id": run.experiment_id,
                "benchmark": run.benchmark,
                "seed": "" if run.seed is None else run.seed,
                "layout_seed": "" if run.layout_seed is None else run.layout_seed,
                "layout_index": "" if run.layout_index is None else run.layout_index,
                "replicate_index": "" if run.replicate_index is None else run.replicate_index,
                "learning_rate": "" if run.learning_rate is None else run.learning_rate,
                "start_layout": run.start_layout,
                "schema_version": run.schema_version,
                "diagnostic_best_layout": run.diagnostic_best_layout,
                "end_layout": run.end_layout,
                "trained_batch_updates": (
                    "" if run.trained_batch_updates is None else run.trained_batch_updates
                ),
                "effective_online_epochs": (
                    "" if run.effective_online_epochs is None else run.effective_online_epochs
                ),
                "step_count": sum(1 for step in run.history if int(step["step_index"]) != 0),
                "acceptance_rate": (
                    float(np.mean(accepted_values)) if accepted_values else ""
                ),
                "start_validation_loss": start_val_loss,
                "final_validation_loss": final_val_loss,
                "online_validation_loss_delta": final_val_loss - start_val_loss,
            }
        )
    return rows


def _paired_run_rows(runs: list[OnlineDeltaRun]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in runs:
        comparisons = {
            str(item.get("label", "")): item
            for item in run.final_comparisons
        }
        start_loss = _comparison_metric(comparisons.get("same_random_start_retrained"), "val_loss")
        end_loss = _comparison_metric(comparisons.get("end_layout_from_sa_retrained"), "val_loss")
        if start_loss is None or end_loss is None:
            continue
        start_acc = _comparison_metric(comparisons.get("same_random_start_retrained"), "val_accuracy")
        end_acc = _comparison_metric(comparisons.get("end_layout_from_sa_retrained"), "val_accuracy")
        rows.append(
            {
                "run_id": run.run_id,
                "benchmark": run.benchmark,
                "layout_index": "" if run.layout_index is None else run.layout_index,
                "replicate_index": "" if run.replicate_index is None else run.replicate_index,
                "layout_seed": "" if run.layout_seed is None else run.layout_seed,
                "random_start_val_loss": start_loss,
                "end_sa_layout_val_loss": end_loss,
                "paired_val_loss_improvement": start_loss - end_loss,
                "paired_win": end_loss < start_loss,
                "random_start_val_accuracy": "" if start_acc is None else start_acc,
                "end_sa_layout_val_accuracy": "" if end_acc is None else end_acc,
            }
        )
    return rows


def _layout_level_summary_rows(paired_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[object, list[dict[str, object]]] = {}
    for row in paired_rows:
        grouped.setdefault(row["layout_index"], []).append(row)
    rows: list[dict[str, object]] = []
    for layout_index, items in sorted(grouped.items(), key=lambda item: str(item[0])):
        improvements = np.asarray(
            [float(item["paired_val_loss_improvement"]) for item in items],
            dtype=np.float64,
        )
        rows.append(
            {
                "layout_index": layout_index,
                "layout_seed": items[0]["layout_seed"],
                "replicates": len(items),
                "mean_random_start_val_loss": float(
                    np.mean([float(item["random_start_val_loss"]) for item in items])
                ),
                "mean_end_sa_layout_val_loss": float(
                    np.mean([float(item["end_sa_layout_val_loss"]) for item in items])
                ),
                "mean_paired_val_loss_improvement": float(np.mean(improvements)),
                "std_paired_val_loss_improvement": float(np.std(improvements, ddof=0)),
                "win_rate": float(np.mean([bool(item["paired_win"]) for item in items])),
            }
        )
    return rows


def _benchmark_summary_rows(paired_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not paired_rows:
        return []
    run_improvements = np.asarray(
        [float(row["paired_val_loss_improvement"]) for row in paired_rows],
        dtype=np.float64,
    )
    layout_rows = _layout_level_summary_rows(paired_rows)
    layout_improvements = np.asarray(
        [float(row["mean_paired_val_loss_improvement"]) for row in layout_rows],
        dtype=np.float64,
    )
    standard_error = (
        float(np.std(layout_improvements, ddof=1) / np.sqrt(len(layout_improvements)))
        if len(layout_improvements) > 1
        else 0.0
    )
    mean_improvement = float(np.mean(layout_improvements))
    return [
        {
            "benchmark": paired_rows[0]["benchmark"],
            "runs": len(paired_rows),
            "layouts": len(layout_rows),
            "ci_unit": "layout_mean",
            "mean_paired_val_loss_improvement": mean_improvement,
            "median_layout_mean_improvement": float(np.median(layout_improvements)),
            "std_layout_mean_improvement": float(np.std(layout_improvements, ddof=0)),
            "std_run_improvement": float(np.std(run_improvements, ddof=0)),
            "ci95_low": mean_improvement - 1.96 * standard_error,
            "ci95_high": mean_improvement + 1.96 * standard_error,
            "run_win_rate": float(np.mean([bool(row["paired_win"]) for row in paired_rows])),
            "layout_win_rate": float(
                np.mean(
                    [
                        float(row["mean_paired_val_loss_improvement"]) > 0.0
                        for row in layout_rows
                    ]
                )
            ),
        }
    ]


def _activation_layout_feature_rows(
    runs: list[OnlineDeltaRun],
    paired_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    paired_by_layout: dict[object, list[dict[str, object]]] = {}
    for row in paired_rows:
        paired_by_layout.setdefault(row["layout_index"], []).append(row)
    runs_by_layout: dict[object, list[OnlineDeltaRun]] = {}
    for run in runs:
        if run.layout_index is not None:
            runs_by_layout.setdefault(run.layout_index, []).append(run)

    rows: list[dict[str, object]] = []
    for layout_index, layout_runs in sorted(runs_by_layout.items(), key=lambda item: int(item[0])):
        paired = paired_by_layout.get(layout_index, [])
        if not paired:
            continue
        row: dict[str, object] = {
            "benchmark": layout_runs[0].benchmark,
            "layout_index": layout_index,
            "layout_seed": "" if layout_runs[0].layout_seed is None else layout_runs[0].layout_seed,
            "replicates": len(layout_runs),
            "mean_paired_val_loss_improvement": float(
                np.mean([float(item["paired_val_loss_improvement"]) for item in paired])
            ),
        }
        for activation in SUPPORTED_ACTIVATIONS:
            start_shares = [_activation_share(run.start_layout, activation) for run in layout_runs]
            end_shares = [_activation_share(run.end_layout, activation) for run in layout_runs]
            start_share = float(np.mean(start_shares))
            end_share = float(np.mean(end_shares))
            row[f"start_share_{activation}"] = start_share
            row[f"end_share_{activation}"] = end_share
            row[f"share_change_{activation}"] = end_share - start_share
        rows.append(row)
    return rows


def _activation_effect_summary_rows(
    layout_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not layout_rows:
        return []
    improvements = np.asarray(
        [float(row["mean_paired_val_loss_improvement"]) for row in layout_rows],
        dtype=np.float64,
    )
    rows: list[dict[str, object]] = []
    for activation_index, activation in enumerate(SUPPORTED_ACTIVATIONS):
        start_shares = np.asarray(
            [float(row[f"start_share_{activation}"]) for row in layout_rows],
            dtype=np.float64,
        )
        end_shares = np.asarray(
            [float(row[f"end_share_{activation}"]) for row in layout_rows],
            dtype=np.float64,
        )
        share_changes = np.asarray(
            [float(row[f"share_change_{activation}"]) for row in layout_rows],
            dtype=np.float64,
        )
        correlation = _pearson_correlation(share_changes, improvements)
        ci_low, ci_high = _bootstrap_correlation_interval(
            share_changes,
            improvements,
            seed=1729 + activation_index,
        )
        increased = share_changes > 0.0
        increased_values = improvements[increased]
        other_values = improvements[~increased]
        increased_mean = float(np.mean(increased_values)) if len(increased_values) else ""
        other_mean = float(np.mean(other_values)) if len(other_values) else ""
        rows.append(
            {
                "benchmark": layout_rows[0]["benchmark"],
                "activation": activation,
                "layout_blocks": len(layout_rows),
                "mean_start_share": float(np.mean(start_shares)),
                "mean_end_share": float(np.mean(end_shares)),
                "mean_share_change": float(np.mean(share_changes)),
                "correlation_start_share_with_improvement": _pearson_correlation(
                    start_shares,
                    improvements,
                ),
                "correlation_end_share_with_improvement": _pearson_correlation(
                    end_shares,
                    improvements,
                ),
                "correlation_share_change_with_improvement": correlation,
                "correlation_share_change_ci95_low": ci_low,
                "correlation_share_change_ci95_high": ci_high,
                "layouts_with_increased_share": int(np.sum(increased)),
                "mean_improvement_when_share_increased": increased_mean,
                "mean_improvement_when_not_increased": other_mean,
                "increased_share_improvement_contrast": (
                    float(increased_mean) - float(other_mean)
                    if increased_mean != "" and other_mean != ""
                    else ""
                ),
            }
        )
    return rows


def _activation_share(layout_spec: str, activation: str) -> float:
    flattened = _flatten_layout(layout_spec)
    if not flattened:
        return 0.0
    return flattened.count(activation) / len(flattened)


def _pearson_correlation(left: np.ndarray, right: np.ndarray) -> float | str:
    if len(left) < 3 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return ""
    return float(np.corrcoef(left, right)[0, 1])


def _bootstrap_correlation_interval(
    left: np.ndarray,
    right: np.ndarray,
    *,
    seed: int,
    samples: int = 2000,
) -> tuple[float | str, float | str]:
    if _pearson_correlation(left, right) == "":
        return "", ""
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, len(left), size=len(left))
        correlation = _pearson_correlation(left[indices], right[indices])
        if correlation != "":
            values.append(float(correlation))
    if not values:
        return "", ""
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _seed_manifest_rows(runs: list[OnlineDeltaRun]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in runs:
        rows.append(
            {
                "run_id": run.run_id,
                "benchmark": run.benchmark,
                "layout_index": "" if run.layout_index is None else run.layout_index,
                "replicate_index": "" if run.replicate_index is None else run.replicate_index,
                **run.seeds,
            }
        )
    return rows


def _activation_count_rows(runs: list[OnlineDeltaRun]) -> list[dict[str, object]]:
    counts = {activation_name: 0 for activation_name in SUPPORTED_ACTIVATIONS}
    for run in runs:
        for activation_name in _flatten_layout(run.end_layout):
            if activation_name in counts:
                counts[activation_name] += 1
    total = sum(counts.values())
    return [
        {
            "activation": activation_name,
            "count": count,
            "share": (count / total if total else 0.0),
        }
        for activation_name, count in counts.items()
    ]


def _layout_similarity_rows(runs: list[OnlineDeltaRun]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    flattened = [(run, _flatten_layout(run.end_layout)) for run in runs]
    for left_index, (left_run, left_layout) in enumerate(flattened):
        for right_run, right_layout in flattened[left_index + 1 :]:
            if len(left_layout) != len(right_layout) or not left_layout:
                rows.append(
                    {
                        "left_run_id": left_run.run_id,
                        "right_run_id": right_run.run_id,
                        "hamming_distance": "",
                        "normalized_distance": "",
                        "status": "incompatible_layout_lengths",
                    }
                )
                continue
            distance = sum(
                1
                for left_activation, right_activation in zip(left_layout, right_layout)
                if left_activation != right_activation
            )
            rows.append(
                {
                    "left_run_id": left_run.run_id,
                    "right_run_id": right_run.run_id,
                    "hamming_distance": distance,
                    "normalized_distance": distance / len(left_layout),
                    "status": "ok",
                }
            )
    return rows


def _plot_fitness_progress(path: Path, runs: list[OnlineDeltaRun]) -> Path:
    grouped: dict[int, list[float]] = {}
    for run in runs:
        fallback_update_index = 0
        for step in run.history:
            if not bool(step.get("accepted")):
                continue
            fallback_update_index += 1
            update_index = _optional_int(step.get("trained_batch_updates_after_step"))
            loss = _optional_float(step.get("post_training_batch_loss"))
            if loss is not None:
                grouped.setdefault(update_index or fallback_update_index, []).append(loss)
    fig, axis = plt.subplots(figsize=(7, 4))
    if grouped:
        steps = sorted(grouped)
        means = np.asarray([np.mean(grouped[step]) for step in steps], dtype=np.float64)
        stds = np.asarray([np.std(grouped[step], ddof=0) for step in steps], dtype=np.float64)
        axis.plot(steps, means, label="mean online fitness")
        axis.fill_between(steps, means - stds, means + stds, alpha=0.2)
        axis.set_xlabel("Trained mini-batch updates")
        axis.set_ylabel("Post-training batch loss")
        axis.legend()
    else:
        axis.text(0.5, 0.5, "No data", ha="center", va="center")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_validation_progress(path: Path, runs: list[OnlineDeltaRun]) -> Path:
    grouped: dict[float, list[float]] = {}
    for run in runs:
        for step in run.history:
            if not bool(step.get("diagnostics_refreshed")):
                continue
            online_epochs = _optional_float(step.get("effective_online_epochs_after_step"))
            val_loss = _optional_float(step.get("validation_loss_after_update"))
            if online_epochs is not None and val_loss is not None:
                grouped.setdefault(round(online_epochs, 8), []).append(val_loss)
    fig, axis = plt.subplots(figsize=(7, 4))
    if grouped:
        epochs = sorted(grouped)
        means = np.asarray([np.mean(grouped[epoch]) for epoch in epochs], dtype=np.float64)
        stds = np.asarray([np.std(grouped[epoch], ddof=0) for epoch in epochs], dtype=np.float64)
        axis.plot(epochs, means, label="mean validation loss")
        axis.fill_between(epochs, means - stds, means + stds, alpha=0.2)
        axis.set_xlabel("Effective online epochs")
        axis.set_ylabel("Validation loss")
        axis.legend()
    else:
        axis.text(0.5, 0.5, "Legacy run: no epoch checkpoints", ha="center", va="center")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_acceptance_rate(path: Path, runs: list[OnlineDeltaRun]) -> Path:
    grouped: dict[int, list[float]] = {}
    for run in runs:
        for step in run.history:
            step_index = int(step["step_index"])
            if step_index == 0:
                continue
            grouped.setdefault(step_index, []).append(1.0 if bool(step["accepted"]) else 0.0)
    fig, axis = plt.subplots(figsize=(7, 4))
    if grouped:
        steps = sorted(grouped)
        means = [float(np.mean(grouped[step])) for step in steps]
        axis.plot(steps, means, marker="o", markersize=3)
        axis.set_ylim(-0.05, 1.05)
        axis.set_xlabel("SA step")
        axis.set_ylabel("Acceptance rate")
    else:
        axis.text(0.5, 0.5, "No data", ha="center", va="center")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_delta_distribution(path: Path, runs: list[OnlineDeltaRun]) -> Path:
    values = [
        float(step["delta"])
        for run in runs
        for step in run.history
        if int(step["step_index"]) != 0
        and isinstance(step.get("delta"), float)
        and isfinite(float(step["delta"]))
    ]
    fig, axis = plt.subplots(figsize=(7, 4))
    if values:
        axis.hist(values, bins=min(30, max(5, len(values) // 2)), color="#4f7d7a")
        axis.axvline(0.0, color="#333333", linewidth=1)
        axis.set_xlabel("Loss delta")
        axis.set_ylabel("Count")
    else:
        axis.text(0.5, 0.5, "No data", ha="center", va="center")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_temperature(path: Path, runs: list[OnlineDeltaRun]) -> Path:
    grouped = _group_step_values(runs, "temperature")
    fig, axis = plt.subplots(figsize=(7, 4))
    if grouped:
        steps = sorted(grouped)
        means = [float(np.mean(grouped[step])) for step in steps]
        axis.plot(steps, means)
        axis.set_xlabel("SA step")
        axis.set_ylabel("Temperature")
    else:
        axis.text(0.5, 0.5, "No data", ha="center", va="center")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_activation_effect_association(
    path: Path,
    rows: list[dict[str, object]],
) -> Path:
    usable = [
        row
        for row in rows
        if row.get("correlation_share_change_with_improvement") not in ("", None)
        and row.get("correlation_share_change_ci95_low") not in ("", None)
        and row.get("correlation_share_change_ci95_high") not in ("", None)
    ]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    if usable:
        labels = [str(row["activation"]) for row in usable]
        values = [float(row["correlation_share_change_with_improvement"]) for row in usable]
        lows = [float(row["correlation_share_change_ci95_low"]) for row in usable]
        highs = [float(row["correlation_share_change_ci95_high"]) for row in usable]
        errors = [
            [value - low for value, low in zip(values, lows, strict=True)],
            [high - value for value, high in zip(values, highs, strict=True)],
        ]
        y = list(range(len(usable)))
        axis.errorbar(
            values,
            y,
            xerr=errors,
            fmt="o",
            color="#245a73",
            ecolor="#7f8c8d",
            capsize=4,
        )
        axis.axvline(0.0, color="#333333", linewidth=1, linestyle="--")
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.set_xlim(-1.05, 1.05)
        axis.set_xlabel("Correlation: activation-share change vs paired improvement")
    else:
        axis.text(0.5, 0.5, "Insufficient layout-level variation", ha="center", va="center")
    axis.set_title("Exploratory activation-function association (layout-level bootstrap CI)")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_activation_counts(path: Path, runs: list[OnlineDeltaRun]) -> Path:
    rows = _activation_count_rows(runs)
    fig, axis = plt.subplots(figsize=(7, 4))
    labels = [str(row["activation"]) for row in rows]
    values = [int(row["count"]) for row in rows]
    if sum(values):
        axis.bar(labels, values, color="#536b8e")
        axis.set_ylabel("Count in end layouts")
        axis.tick_params(axis="x", rotation=30)
    else:
        axis.text(0.5, 0.5, "No data", ha="center", va="center")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_paired_comparison(path: Path, rows: list[dict[str, object]]) -> Path:
    fig, axis = plt.subplots(figsize=(7, 5))
    if rows:
        start = np.asarray([float(row["random_start_val_loss"]) for row in rows])
        end = np.asarray([float(row["end_sa_layout_val_loss"]) for row in rows])
        lower = float(min(np.min(start), np.min(end)))
        upper = float(max(np.max(start), np.max(end)))
        axis.scatter(start, end, alpha=0.75)
        axis.plot([lower, upper], [lower, upper], linestyle="--", color="#555555")
        axis.set_xlabel("Random start retrained validation loss")
        axis.set_ylabel("End SA layout retrained validation loss")
    else:
        axis.text(0.5, 0.5, "No paired data", ha="center", va="center")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_improvement_by_layout(path: Path, rows: list[dict[str, object]]) -> Path:
    summary = _layout_level_summary_rows(rows)
    fig, axis = plt.subplots(figsize=(8, 4))
    if summary:
        indices = [int(row["layout_index"]) for row in summary]
        values = [float(row["mean_paired_val_loss_improvement"]) for row in summary]
        colors = ["#3a8f6b" if value >= 0.0 else "#c85c5c" for value in values]
        axis.bar(indices, values, color=colors)
        axis.axhline(0.0, color="#333333", linewidth=1)
        axis.set_xlabel("Layout index")
        axis.set_ylabel("Mean paired validation-loss improvement")
    else:
        axis.text(0.5, 0.5, "No paired data", ha="center", va="center")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_improvement_distribution(path: Path, rows: list[dict[str, object]]) -> Path:
    values = [float(row["paired_val_loss_improvement"]) for row in rows]
    fig, axis = plt.subplots(figsize=(7, 4))
    if values:
        axis.hist(values, bins=min(15, max(5, len(values) // 2)), color="#536b8e")
        axis.axvline(0.0, color="#333333", linewidth=1)
        axis.set_xlabel("Paired validation-loss improvement")
        axis.set_ylabel("Runs")
    else:
        axis.text(0.5, 0.5, "No paired data", ha="center", va="center")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _group_step_values(
    runs: list[OnlineDeltaRun],
    key: str,
) -> dict[int, list[float]]:
    grouped: dict[int, list[float]] = {}
    for run in runs:
        for step in run.history:
            value = step.get(key)
            if isinstance(value, (float, int)) and isfinite(float(value)):
                grouped.setdefault(int(step["step_index"]), []).append(float(value))
    return grouped


def _flatten_layout(layout_spec: str) -> tuple[str, ...]:
    values: list[str] = []
    for layer_spec in layout_spec.split("|"):
        for raw_token in layer_spec.split(","):
            token = raw_token.strip()
            if not token:
                continue
            if "*" in token:
                activation_name, repeat_text = token.split("*", 1)
                try:
                    repeat = int(repeat_text)
                except ValueError:
                    repeat = 1
                values.extend([activation_name.strip()] * max(0, repeat))
            else:
                values.append(token)
    return tuple(values)


def _comparison_metric(comparison: dict[str, object] | None, key: str) -> float | None:
    if comparison is None:
        return None
    metrics = comparison.get("metrics")
    if isinstance(metrics, dict):
        return _optional_float(metrics.get(key))
    return _optional_float(comparison.get(key))


def _optional_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["status"]
        rows = [{"status": "empty"}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
