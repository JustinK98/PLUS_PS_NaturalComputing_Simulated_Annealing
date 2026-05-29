"""Terminal benchmark suites for official Basics experiments."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from activations import parse_layout_spec
from annealing import AnnealingConfig
from benchmark_registry import benchmark_spec
from benchmarks import load_benchmark
from configs import (
    DEFAULT_ANNEALING_COOLING_PARAMETER,
    DEFAULT_ANNEALING_COOLING_SCHEDULE,
    DEFAULT_ANNEALING_ITERATIONS_PER_TEMPERATURE,
    DEFAULT_ANNEALING_MAX_STEPS,
    DEFAULT_ANNEALING_MIN_TEMPERATURE,
    DEFAULT_ANNEALING_NEIGHBORHOODS,
    DEFAULT_ANNEALING_START_TEMPERATURE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_WEIGHT_SCALE,
    OUTPUT_DIR,
    SUPPORTED_ACTIVATIONS,
    DatasetConfig,
    TrainingConfig,
    default_epochs,
    default_hidden_sizes,
)
from model import ModularMLP
from services.layout_evaluation_service import (
    LayoutCandidate,
    LayoutEvaluationRequest,
    layout_evaluation_to_dict,
    run_layout_evaluation,
)
from services.online_annealing_training_service import (
    OnlineAnnealingRequest,
    OnlineAnnealingSnapshot,
    create_online_session,
    online_run_to_completion,
)
from services.online_delta_reporting_service import build_online_delta_report
from trainer import train_model


CONFIG_PATH = Path("configs") / "experiment_suites.json"


@dataclass(frozen=True)
class ExperimentSuiteRequest:
    """User-facing suite command configuration."""

    exp: str
    benchmark: str
    learning_rate_preset: int
    seeds: tuple[int, ...] | None = None
    run_count: int | None = None
    epochs: int | None = None
    max_steps: int | None = None
    start_temperature: float | None = None
    cooling_parameter: float | None = None
    iterations_per_temperature: int | None = None
    min_temperature: float | None = None
    output_root: Path = OUTPUT_DIR / "experiment_suites"
    no_plots: bool = False


@dataclass(frozen=True)
class ExperimentSuiteResult:
    """Summary returned to the CLI command."""

    output_dir: Path
    learning_rates: tuple[float, ...]
    run_count: int
    summary_path: Path


def run_experiment_suite(request: ExperimentSuiteRequest) -> ExperimentSuiteResult:
    """Run one of the official terminal experiment suites."""

    suite_config = _load_suite_config()
    if request.exp not in {"online-delta", "random-baseline", "all-baseline", "swap-ablation"}:
        raise ValueError(
            "--exp muss online-delta, random-baseline, all-baseline oder swap-ablation sein."
        )

    spec = benchmark_spec(request.benchmark)
    hidden_sizes = default_hidden_sizes(request.benchmark)
    epochs = request.epochs if request.epochs is not None else default_epochs(request.benchmark)
    run_indices = _resolve_run_indices(request, suite_config)
    learning_rates = _resolve_learning_rates(request.learning_rate_preset, suite_config)
    annealing_settings = _resolve_annealing_settings(request, suite_config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(request.output_root)
        / f"{timestamp}_{request.benchmark}_{request.exp}_lr_preset_{request.learning_rate_preset}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summary_rows: list[dict[str, object]] = []
    run_count = 0
    for learning_rate in learning_rates:
        lr_dir = output_dir / _learning_rate_dir_name(learning_rate)
        lr_dir.mkdir(parents=True, exist_ok=True)
        suite_payload = {
            "exp": request.exp,
            "benchmark": request.benchmark,
            "topology": _format_topology(
                len(spec.feature_columns),
                hidden_sizes,
                spec.model_output_size,
            ),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": DEFAULT_BATCH_SIZE,
            "weight_scale": DEFAULT_WEIGHT_SCALE,
            "runs": list(run_indices),
            "learning_rate_preset": request.learning_rate_preset,
            "neighborhood_operations": _suite_neighborhood_operations(request.exp, suite_config),
            "annealing": annealing_settings,
        }
        _write_json(lr_dir / "config.json", suite_payload)

        if request.exp == "all-baseline":
            rows, count = _run_all_suite(
                lr_dir,
                benchmark=request.benchmark,
                hidden_sizes=hidden_sizes,
                run_indices=run_indices,
                epochs=epochs,
                learning_rate=learning_rate,
                no_plots=request.no_plots,
            )
        elif request.exp == "random-baseline":
            rows, count = _run_random_baseline_suite(
                lr_dir,
                experiment_id=_suite_experiment_id(request.exp, suite_config),
                benchmark=request.benchmark,
                hidden_sizes=hidden_sizes,
                run_indices=run_indices,
                epochs=epochs,
                learning_rate=learning_rate,
                no_plots=request.no_plots,
            )
        else:
            rows, count = _run_online_delta_suite(
                lr_dir,
                experiment_id=_suite_experiment_id(request.exp, suite_config),
                benchmark=request.benchmark,
                hidden_sizes=hidden_sizes,
                run_indices=run_indices,
                epochs=epochs,
                learning_rate=learning_rate,
                annealing_settings=annealing_settings,
                neighborhood_operations=_suite_neighborhood_operations(request.exp, suite_config),
                no_plots=request.no_plots,
            )

        run_count += count
        all_summary_rows.extend(rows)
        summary_path = lr_dir / "summary.csv"
        _write_csv(summary_path, rows)
        _write_json(
            lr_dir / "manifest.json",
            {
                "config_path": str(lr_dir / "config.json"),
                "summary_path": str(summary_path),
                "run_count": count,
                "runs_dir": str(lr_dir / "runs"),
                "plots_single_dir": str(lr_dir / "plots_single"),
            },
        )

    aggregate_dir = output_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    aggregate_summary = aggregate_dir / "summary.csv"
    _write_csv(aggregate_summary, all_summary_rows)
    if not request.no_plots:
        _write_aggregate_plots(aggregate_dir, all_summary_rows)
        if request.exp in {"online-delta", "swap-ablation"}:
            build_online_delta_report(output_dir, aggregate_dir)

    return ExperimentSuiteResult(
        output_dir=output_dir,
        learning_rates=tuple(learning_rates),
        run_count=run_count,
        summary_path=aggregate_summary,
    )


def _run_all_suite(
    output_dir: Path,
    *,
    benchmark: str,
    hidden_sizes: tuple[int, ...],
    run_indices: tuple[int, ...],
    epochs: int,
    learning_rate: float,
    no_plots: bool,
) -> tuple[list[dict[str, object]], int]:
    candidates = tuple(
        LayoutCandidate(f"all_{activation_name}", activation_name)
        for activation_name in SUPPORTED_ACTIVATIONS
    )
    result = run_layout_evaluation(
        LayoutEvaluationRequest(
            dataset_config=DatasetConfig(name=benchmark, random_state=DEFAULT_RANDOM_SEED),
            hidden_sizes=hidden_sizes,
            candidates=candidates,
            seeds=run_indices,
            training_config=TrainingConfig(
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=DEFAULT_BATCH_SIZE,
                random_state=DEFAULT_RANDOM_SEED,
            ),
            weight_scale=DEFAULT_WEIGHT_SCALE,
            primary_metric="validation_loss",
        )
    )
    return _persist_layout_evaluation(
        output_dir,
        experiment_id="E1_all_single_af",
        benchmark=benchmark,
        learning_rate=learning_rate,
        result_payload=layout_evaluation_to_dict(result),
        no_plots=no_plots,
    )


def _run_random_baseline_suite(
    output_dir: Path,
    *,
    experiment_id: str,
    benchmark: str,
    hidden_sizes: tuple[int, ...],
    run_indices: tuple[int, ...],
    epochs: int,
    learning_rate: float,
    no_plots: bool,
) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    runs_dir = output_dir / "runs"
    plots_dir = output_dir / "plots_single"
    runs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    for run_index in run_indices:
        layout_seed = int(run_index)
        training_seed = int(run_index)
        layout_spec = _random_layout_spec(hidden_sizes, layout_seed)
        run_payload = _train_layout_once(
            benchmark=benchmark,
            hidden_sizes=hidden_sizes,
            layout_spec=layout_spec,
            seed=training_seed,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        label = "random_start_baseline"
        rows.append(
            _summary_row_from_metrics(
                experiment_id=experiment_id,
                benchmark=benchmark,
                learning_rate=learning_rate,
                label=label,
                layout_spec=str(run_payload["layout_spec"]),
                seed=training_seed,
                evaluation_type="retrained",
                metrics=run_payload["metrics"],  # type: ignore[arg-type]
                layout_seed=layout_seed,
            )
        )
        payload = {
            "kind": "experiment_suite_random_baseline_run",
            "experiment_id": experiment_id,
            "benchmark": benchmark,
            "learning_rate": learning_rate,
            "label": label,
            "layout_seed": layout_seed,
            "training_seed": training_seed,
            **run_payload,
        }
        run_path = runs_dir / f"random_baseline_run_{run_index:04d}.json"
        _write_json(run_path, payload)
        if not no_plots:
            _plot_training_history(
                plots_dir / f"random_baseline_run_{run_index:04d}.png",
                label,
                run_payload["history"],
            )

    return _aggregate_summary_rows(rows), len(run_indices)


def _run_online_delta_suite(
    output_dir: Path,
    *,
    experiment_id: str,
    benchmark: str,
    hidden_sizes: tuple[int, ...],
    run_indices: tuple[int, ...],
    epochs: int,
    learning_rate: float,
    annealing_settings: dict[str, float | int],
    neighborhood_operations: tuple[str, ...],
    no_plots: bool,
) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    train_cache: dict[tuple[int, str], dict[str, object]] = {}

    runs_dir = output_dir / "runs"
    plots_dir = output_dir / "plots_single"
    runs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    for run_index in run_indices:
        layout_seed = int(run_index)
        training_seed = int(run_index)
        start_layout_spec = _random_layout_spec(hidden_sizes, layout_seed)
        snapshot = _run_online_sa(
            benchmark=benchmark,
            hidden_sizes=hidden_sizes,
            start_layout_spec=start_layout_spec,
            seed=training_seed,
            epochs=epochs,
            learning_rate=learning_rate,
            annealing_settings=annealing_settings,
            neighborhood_operations=neighborhood_operations,
        )
        if snapshot.best_evaluation is None or snapshot.current_evaluation is None:
            raise ValueError("Online-SA lieferte keine finalen Evaluationen.")

        final_comparisons: list[dict[str, object]] = []
        comparison_layouts = [
            ("same_random_start_retrained", start_layout_spec),
            (
                "best_layout_from_sa_retrained",
                snapshot.best_evaluation.layout.to_compact_spec(),
            ),
            (
                "end_layout_from_sa_retrained",
                snapshot.current_evaluation.layout.to_compact_spec(),
            ),
        ]

        for label, layout_spec in comparison_layouts:
            run_payload = _cached_training_run(
                train_cache,
                benchmark=benchmark,
                hidden_sizes=hidden_sizes,
                layout_spec=layout_spec,
                seed=training_seed,
                epochs=epochs,
                learning_rate=learning_rate,
            )
            final_comparisons.append({**run_payload, "label": label})
            rows.append(
                _summary_row_from_metrics(
                    experiment_id=experiment_id,
                    benchmark=benchmark,
                    learning_rate=learning_rate,
                    label=label,
                    layout_spec=str(run_payload["layout_spec"]),
                    seed=training_seed,
                    evaluation_type="retrained",
                    metrics=run_payload["metrics"],  # type: ignore[arg-type]
                    layout_seed=layout_seed,
                )
            )

        inherited_payloads = [
            (
                "best_inherited_model_from_sa",
                snapshot.best_evaluation.layout.to_compact_spec(),
                _evaluate_model_state(
                    benchmark,
                    snapshot.best_evaluation.trained_model.to_state_dict(),
                    training_seed,
                ),
            ),
            (
                "end_inherited_model_from_sa",
                snapshot.current_evaluation.layout.to_compact_spec(),
                _evaluate_model_state(
                    benchmark,
                    snapshot.current_evaluation.trained_model.to_state_dict(),
                    training_seed,
                ),
            ),
        ]
        for label, layout_spec, metrics in inherited_payloads:
            final_comparisons.append(
                {
                    "label": label,
                    "layout_spec": layout_spec,
                    "seed": training_seed,
                    "evaluation_type": "inherited",
                    "metrics": metrics,
                    "history": {},
                }
            )
            rows.append(
                _summary_row_from_metrics(
                    experiment_id=experiment_id,
                    benchmark=benchmark,
                    learning_rate=learning_rate,
                    label=label,
                    layout_spec=layout_spec,
                    seed=training_seed,
                    evaluation_type="inherited",
                    metrics=metrics,
                    layout_seed=layout_seed,
                )
            )

        run_payload = {
            "kind": "experiment_suite_online_delta_run",
            "experiment_id": experiment_id,
            "benchmark": benchmark,
            "run_index": run_index,
            "seed": training_seed,
            "training_seed": training_seed,
            "layout_seed": layout_seed,
            "learning_rate": learning_rate,
            "neighborhood_operations": list(neighborhood_operations),
            "start_layout": start_layout_spec,
            "best_layout": snapshot.best_evaluation.layout.to_compact_spec(),
            "end_layout": snapshot.current_evaluation.layout.to_compact_spec(),
            "accepted_steps": snapshot.accepted_steps,
            "acceptance_rate": snapshot.acceptance_rate,
            "neighbor_counts": _neighbor_counts(snapshot),
            "online_history": _online_history_to_dict(snapshot),
            "final_comparisons": final_comparisons,
        }
        run_path = runs_dir / f"{experiment_id}_run_{run_index:04d}.json"
        _write_json(run_path, run_payload)
        if not no_plots:
            _plot_online_history(
                plots_dir / f"{experiment_id}_run_{run_index:04d}.png",
                snapshot,
            )

    return _aggregate_summary_rows(rows), len(run_indices)


def _persist_layout_evaluation(
    output_dir: Path,
    *,
    experiment_id: str,
    benchmark: str,
    learning_rate: float,
    result_payload: dict[str, object],
    no_plots: bool,
) -> tuple[list[dict[str, object]], int]:
    runs_dir = output_dir / "runs"
    plots_dir = output_dir / "plots_single"
    runs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    runs = list(result_payload.get("runs", []))
    for run in runs:
        if not isinstance(run, dict):
            continue
        safe_label = _safe_name(str(run.get("label", "run")))
        seed = int(run.get("seed", 0))
        run_path = runs_dir / f"{safe_label}_seed_{seed:04d}.json"
        _write_json(
            run_path,
            {
                "kind": "experiment_suite_run",
                "experiment_id": experiment_id,
                "benchmark": benchmark,
                "learning_rate": learning_rate,
                **run,
            },
        )
        if not no_plots:
            _plot_training_history(
                plots_dir / f"{safe_label}_seed_{seed:04d}.png",
                str(run.get("label", "")),
                run.get("history", {}),
            )

    summary_rows = []
    for item in sorted(
        result_payload.get("aggregated", []),
        key=lambda value: float(value.get("ranking_score", 0.0)) if isinstance(value, dict) else 0.0,
    ):
        if not isinstance(item, dict):
            continue
        metrics = item.get("mean_metrics", {})
        std_metrics = item.get("std_metrics", {})
        if not isinstance(metrics, dict):
            continue
        summary_rows.append(
            {
                "experiment_id": experiment_id,
                "benchmark": benchmark,
                "learning_rate": learning_rate,
                "label": item.get("label", ""),
                "layout_spec": item.get("layout_spec", ""),
                "evaluation_type": item.get("evaluation_type", "retrained"),
                "num_runs": item.get("num_runs", ""),
                "mean_train_loss": metrics.get("train_loss", ""),
                "std_train_loss": _dict_get(std_metrics, "train_loss"),
                "mean_val_loss": metrics.get("val_loss", ""),
                "std_val_loss": _dict_get(std_metrics, "val_loss"),
                "mean_test_loss": metrics.get("test_loss", ""),
                "std_test_loss": _dict_get(std_metrics, "test_loss"),
                "mean_train_accuracy": metrics.get("train_accuracy", ""),
                "std_train_accuracy": _dict_get(std_metrics, "train_accuracy"),
                "mean_val_accuracy": metrics.get("val_accuracy", ""),
                "std_val_accuracy": _dict_get(std_metrics, "val_accuracy"),
                "mean_test_accuracy": metrics.get("test_accuracy", ""),
                "std_test_accuracy": _dict_get(std_metrics, "test_accuracy"),
                "ranking_score": item.get("ranking_score", ""),
            }
        )
    return summary_rows, len(runs)


def _cached_training_run(
    cache: dict[tuple[int, str], dict[str, object]],
    *,
    benchmark: str,
    hidden_sizes: tuple[int, ...],
    layout_spec: str,
    seed: int,
    epochs: int,
    learning_rate: float,
) -> dict[str, object]:
    layout = parse_layout_spec(layout_spec, hidden_sizes)
    normalized_spec = layout.to_compact_spec()
    cache_key = (seed, normalized_spec)
    if cache_key not in cache:
        cache[cache_key] = _train_layout_once(
            benchmark=benchmark,
            hidden_sizes=hidden_sizes,
            layout_spec=normalized_spec,
            seed=seed,
            epochs=epochs,
            learning_rate=learning_rate,
        )
    return dict(cache[cache_key])


def _train_layout_once(
    *,
    benchmark: str,
    hidden_sizes: tuple[int, ...],
    layout_spec: str,
    seed: int,
    epochs: int,
    learning_rate: float,
) -> dict[str, object]:
    dataset = load_benchmark(DatasetConfig(name=benchmark, random_state=seed))
    layout = parse_layout_spec(layout_spec, hidden_sizes)
    model = ModularMLP(
        input_size=dataset.input_size,
        hidden_sizes=hidden_sizes,
        output_size=dataset.model_output_size,
        layout=layout,
        num_classes=dataset.output_size,
        weight_scale=DEFAULT_WEIGHT_SCALE,
        random_state=seed,
    )
    training_result = train_model(
        model,
        dataset,
        TrainingConfig(
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=DEFAULT_BATCH_SIZE,
            random_state=seed,
        ),
    )
    return {
        "layout_spec": layout.to_compact_spec(),
        "seed": seed,
        "evaluation_type": "retrained",
        "metrics": _training_metrics(training_result),
        "history": training_result.history,
        "model_state": model.to_state_dict(),
    }


def _run_online_sa(
    *,
    benchmark: str,
    hidden_sizes: tuple[int, ...],
    start_layout_spec: str,
    seed: int,
    epochs: int,
    learning_rate: float,
    annealing_settings: dict[str, float | int],
    neighborhood_operations: tuple[str, ...],
) -> OnlineAnnealingSnapshot:
    request = OnlineAnnealingRequest(
        dataset_config=DatasetConfig(name=benchmark, random_state=seed),
        hidden_sizes=hidden_sizes,
        layout_spec=start_layout_spec,
        training_config=TrainingConfig(
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=DEFAULT_BATCH_SIZE,
            random_state=seed,
        ),
        annealing_config=AnnealingConfig(
            start_temperature=float(annealing_settings["start_temperature"]),
            cooling_schedule=DEFAULT_ANNEALING_COOLING_SCHEDULE,
            cooling_parameter=float(annealing_settings["cooling_parameter"]),
            iterations_per_temperature=int(annealing_settings["iterations_per_temperature"]),
            max_steps=int(annealing_settings["max_steps"]),
            min_temperature=float(annealing_settings["min_temperature"]),
            neighborhood_operations=neighborhood_operations,
        ),
        weight_scale=DEFAULT_WEIGHT_SCALE,
        random_state=seed,
    )
    session = create_online_session(request)
    return online_run_to_completion(session, language="en")


def _evaluate_model_state(
    benchmark: str,
    model_state: dict[str, object],
    seed: int,
) -> dict[str, float]:
    dataset = load_benchmark(DatasetConfig(name=benchmark, random_state=seed))
    model = ModularMLP.from_state_dict(model_state)
    train_loss, train_accuracy = model.evaluate(dataset.X_train, dataset.y_train)
    val_loss, val_accuracy = model.evaluate(dataset.X_val, dataset.y_val)
    test_loss, test_accuracy = model.evaluate(dataset.X_test, dataset.y_test)
    return {
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "test_loss": float(test_loss),
        "train_accuracy": float(train_accuracy),
        "val_accuracy": float(val_accuracy),
        "test_accuracy": float(test_accuracy),
    }


def _training_metrics(training_result) -> dict[str, float]:
    return {
        "train_loss": float(training_result.history["train_loss"][-1]),
        "val_loss": float(training_result.history["val_loss"][-1]),
        "test_loss": float(training_result.test_metrics["loss"]),
        "train_accuracy": float(training_result.history["train_acc"][-1]),
        "val_accuracy": float(training_result.history["val_acc"][-1]),
        "test_accuracy": float(training_result.test_metrics["accuracy"]),
    }


def _summary_row_from_metrics(
    *,
    experiment_id: str,
    benchmark: str,
    learning_rate: float,
    label: str,
    layout_spec: str,
    seed: int,
    evaluation_type: str,
    metrics: dict[str, float],
    layout_seed: int | None,
) -> dict[str, object]:
    return {
        "experiment_id": experiment_id,
        "benchmark": benchmark,
        "learning_rate": learning_rate,
        "label": label,
        "layout_spec": layout_spec,
        "seed": seed,
        "layout_seed": "" if layout_seed is None else layout_seed,
        "evaluation_type": evaluation_type,
        "train_loss": metrics["train_loss"],
        "val_loss": metrics["val_loss"],
        "test_loss": metrics["test_loss"],
        "train_accuracy": metrics["train_accuracy"],
        "val_accuracy": metrics["val_accuracy"],
        "test_accuracy": metrics["test_accuracy"],
    }


def _aggregate_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, object, object, object], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            row["experiment_id"],
            row["learning_rate"],
            row["label"],
            row["evaluation_type"],
        )
        grouped.setdefault(key, []).append(row)

    aggregate_rows: list[dict[str, object]] = []
    for (_, _, _, _), group in grouped.items():
        first = group[0]
        metric_arrays = {
            metric: np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            for metric in (
                "train_loss",
                "val_loss",
                "test_loss",
                "train_accuracy",
                "val_accuracy",
                "test_accuracy",
            )
        }
        aggregate_rows.append(
            {
                "experiment_id": first["experiment_id"],
                "benchmark": first["benchmark"],
                "learning_rate": first["learning_rate"],
                "label": first["label"],
                "layout_spec": first["layout_spec"],
                "evaluation_type": first["evaluation_type"],
                "num_runs": len(group),
                "mean_train_loss": float(metric_arrays["train_loss"].mean()),
                "std_train_loss": float(metric_arrays["train_loss"].std(ddof=0)),
                "mean_val_loss": float(metric_arrays["val_loss"].mean()),
                "std_val_loss": float(metric_arrays["val_loss"].std(ddof=0)),
                "mean_test_loss": float(metric_arrays["test_loss"].mean()),
                "std_test_loss": float(metric_arrays["test_loss"].std(ddof=0)),
                "mean_train_accuracy": float(metric_arrays["train_accuracy"].mean()),
                "std_train_accuracy": float(metric_arrays["train_accuracy"].std(ddof=0)),
                "mean_val_accuracy": float(metric_arrays["val_accuracy"].mean()),
                "std_val_accuracy": float(metric_arrays["val_accuracy"].std(ddof=0)),
                "mean_test_accuracy": float(metric_arrays["test_accuracy"].mean()),
                "std_test_accuracy": float(metric_arrays["test_accuracy"].std(ddof=0)),
                "ranking_score": float(metric_arrays["val_loss"].mean()),
            }
        )
    return sorted(aggregate_rows, key=lambda row: float(row["ranking_score"]))


def _write_aggregate_plots(output_dir: Path, rows: list[dict[str, object]]) -> None:
    _plot_metric_by_layout(output_dir / "val_loss_by_layout.png", rows, "mean_val_loss", "Validation loss")
    _plot_metric_by_layout(output_dir / "test_loss_by_layout.png", rows, "mean_test_loss", "Test loss")
    _plot_hyperparameter_comparison(output_dir / "hyperparameter_comparison.png", rows)
    _plot_mean_curves(output_dir / "mean_curves.png", output_dir.parent)


def _plot_metric_by_layout(
    path: Path,
    rows: list[dict[str, object]],
    metric: str,
    ylabel: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = sorted(rows, key=lambda row: float(row.get(metric) or 0.0))[:20]
    fig, axis = plt.subplots(figsize=(10, 5))
    if not selected:
        axis.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        labels = [f"{row['experiment_id']}:{row['label']}" for row in selected]
        values = [float(row.get(metric) or 0.0) for row in selected]
        axis.bar(range(len(labels)), values, color="#2f6f8f")
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_hyperparameter_comparison(path: Path, rows: list[dict[str, object]]) -> None:
    by_learning_rate: dict[float, list[float]] = {}
    for row in rows:
        by_learning_rate.setdefault(float(row["learning_rate"]), []).append(
            float(row["mean_val_loss"])
        )
    fig, axis = plt.subplots(figsize=(7, 4))
    if not by_learning_rate:
        axis.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        learning_rates = sorted(by_learning_rate)
        best_losses = [min(by_learning_rate[value]) for value in learning_rates]
        axis.plot([str(value) for value in learning_rates], best_losses, marker="o")
        axis.set_xlabel("Learning-rate preset value")
        axis.set_ylabel("Best mean validation loss")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_mean_curves(path: Path, suite_output_dir: Path) -> None:
    grouped: dict[str, list[list[float]]] = {}
    for run_path in sorted(suite_output_dir.glob("learning_rate_*/runs/*.json")):
        try:
            payload = json.loads(run_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload.get("history"), dict):
            history = payload["history"]
            values = history.get("val_loss", [])
            if values:
                label = str(payload.get("label", run_path.stem))
                grouped.setdefault(label, []).append([float(value) for value in values])
        for comparison in payload.get("final_comparisons", []):
            if not isinstance(comparison, dict):
                continue
            history = comparison.get("history", {})
            if not isinstance(history, dict):
                continue
            values = history.get("val_loss", [])
            if values:
                label = f"{payload.get('experiment_id', '')}:{comparison.get('label', '')}"
                grouped.setdefault(label, []).append([float(value) for value in values])

    fig, axis = plt.subplots(figsize=(7, 4))
    if not grouped:
        axis.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        ranked_groups = sorted(
            grouped.items(),
            key=lambda item: min(values[-1] for values in item[1]),
        )[:10]
        for label, histories in ranked_groups:
            min_length = min(len(values) for values in histories)
            values = np.asarray([history[:min_length] for history in histories], dtype=np.float64)
            axis.plot(values.mean(axis=0), label=label)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Mean validation loss")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_training_history(path: Path, label: str, history: object) -> None:
    if not isinstance(history, dict) or not history:
        return
    fig, axis = plt.subplots(figsize=(7, 4))
    for key in ("train_loss", "val_loss"):
        values = history.get(key, [])
        if values:
            axis.plot(values, label=key)
    axis.set_title(label)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_online_history(path: Path, snapshot: OnlineAnnealingSnapshot) -> None:
    fig, axis = plt.subplots(figsize=(7, 4))
    online_history = _online_history_to_dict(snapshot)
    steps = [int(row["step_index"]) for row in online_history]
    val_losses = [float(row["validation_loss_after_update"]) for row in online_history]
    if steps and val_losses:
        axis.plot(steps, val_losses, label="validation_loss_after_update")
    else:
        axis.text(0.5, 0.5, "No steps", ha="center", va="center")
    axis.set_xlabel("SA step")
    axis.set_ylabel("Validation loss")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _online_history_to_dict(snapshot: OnlineAnnealingSnapshot) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if snapshot.start_evaluation is not None:
        start_layout = snapshot.start_evaluation.layout.to_compact_spec()
        rows.append(
            {
                "step_index": 0,
                "temperature": (
                    float(snapshot.history[0].temperature)
                    if snapshot.history
                    else float(snapshot.current_temperature)
                ),
                "previous_layout": start_layout,
                "candidate_layout": start_layout,
                "delta": 0.0,
                "accepted": True,
                "reason_code": "start",
                "neighbor_label": "start",
                "batch_loss_before": float(snapshot.start_evaluation.objective_value),
                "candidate_loss_after": float(snapshot.start_evaluation.objective_value),
                "validation_loss_after_update": float(snapshot.start_evaluation.val_loss),
                "train_loss": float(snapshot.start_evaluation.train_loss),
                "val_accuracy": float(snapshot.start_evaluation.val_accuracy),
                "train_accuracy": float(snapshot.start_evaluation.train_accuracy),
            }
        )
    rows.extend(
        {
            "step_index": step.step_index,
            "temperature": step.temperature,
            "previous_layout": step.previous_layout.to_compact_spec(),
            "candidate_layout": step.candidate_layout.to_compact_spec(),
            "delta": step.delta,
            "accepted": step.accepted,
            "reason_code": step.reason_code,
            "neighbor_label": step.neighbor_label,
            "batch_loss_before": step.batch_loss_before,
            "candidate_loss_after": step.candidate_loss_after,
            "validation_loss_after_update": step.validation_loss_after_update,
            "train_loss": float(step.previous_evaluation.train_loss),
            "val_accuracy": float(step.previous_evaluation.val_accuracy),
            "train_accuracy": float(step.previous_evaluation.train_accuracy),
        }
        for step in snapshot.history
    )
    return rows


def _neighbor_counts(snapshot: OnlineAnnealingSnapshot) -> dict[str, int]:
    counts = {"set_neuron": 0, "swap_neurons": 0, "fill_layer": 0}
    for step in snapshot.history:
        if step.neighbor_label.startswith("set:"):
            counts["set_neuron"] += 1
        elif step.neighbor_label.startswith("swap:"):
            counts["swap_neurons"] += 1
        elif step.neighbor_label.startswith("fill:"):
            counts["fill_layer"] += 1
    return counts


def _random_layout_spec(hidden_sizes: tuple[int, ...], layout_seed: int) -> str:
    rng = np.random.default_rng(layout_seed)
    layers: list[str] = []
    for layer_size in hidden_sizes:
        layers.append(
            ",".join(
                str(value)
                for value in rng.choice(SUPPORTED_ACTIVATIONS, size=layer_size, replace=True)
            )
        )
    return "|".join(layers)


def _resolve_run_indices(
    request: ExperimentSuiteRequest,
    suite_config: dict[str, Any],
) -> tuple[int, ...]:
    if request.seeds:
        return tuple(int(seed) for seed in request.seeds)
    run_count = request.run_count or int(suite_config["defaults"]["run_count"])
    if run_count <= 0:
        raise ValueError("--runs muss positiv sein.")
    return tuple(range(run_count))


def _resolve_learning_rates(
    preset_index: int,
    suite_config: dict[str, Any],
) -> list[float]:
    preset_key = str(preset_index)
    presets = suite_config["learning_rate_presets"]
    if preset_key not in presets:
        allowed = ", ".join(sorted(presets))
        raise ValueError(f"Unbekanntes --learning-rate Preset {preset_index}. Erlaubt: {allowed}")
    return [float(value) for value in presets[preset_key]]


def _resolve_annealing_settings(
    request: ExperimentSuiteRequest,
    suite_config: dict[str, Any],
) -> dict[str, float | int]:
    defaults = suite_config.get("defaults", {})
    configured = defaults.get("annealing", {}) if isinstance(defaults, dict) else {}
    if not isinstance(configured, dict):
        configured = {}
    settings: dict[str, float | int] = {
        "start_temperature": request.start_temperature
        if request.start_temperature is not None
        else float(configured.get("start_temperature", DEFAULT_ANNEALING_START_TEMPERATURE)),
        "cooling_parameter": request.cooling_parameter
        if request.cooling_parameter is not None
        else float(configured.get("cooling_parameter", DEFAULT_ANNEALING_COOLING_PARAMETER)),
        "iterations_per_temperature": request.iterations_per_temperature
        if request.iterations_per_temperature is not None
        else int(
            configured.get(
                "iterations_per_temperature",
                DEFAULT_ANNEALING_ITERATIONS_PER_TEMPERATURE,
            )
        ),
        "max_steps": request.max_steps
        if request.max_steps is not None
        else int(configured.get("max_steps", DEFAULT_ANNEALING_MAX_STEPS)),
        "min_temperature": request.min_temperature
        if request.min_temperature is not None
        else float(configured.get("min_temperature", DEFAULT_ANNEALING_MIN_TEMPERATURE)),
    }
    AnnealingConfig(
        start_temperature=float(settings["start_temperature"]),
        cooling_schedule=DEFAULT_ANNEALING_COOLING_SCHEDULE,
        cooling_parameter=float(settings["cooling_parameter"]),
        iterations_per_temperature=int(settings["iterations_per_temperature"]),
        max_steps=int(settings["max_steps"]),
        min_temperature=float(settings["min_temperature"]),
        neighborhood_operations=DEFAULT_ANNEALING_NEIGHBORHOODS,
    )
    return settings


def _suite_experiment_id(exp: str, suite_config: dict[str, Any]) -> str:
    experiment = suite_config["experiments"].get(exp)
    if not isinstance(experiment, dict) or "id" not in experiment:
        raise ValueError(f"Suite-Konfiguration enthaelt kein Experiment '{exp}'.")
    return str(experiment["id"])


def _suite_neighborhood_operations(
    exp: str,
    suite_config: dict[str, Any],
) -> tuple[str, ...]:
    experiment = suite_config["experiments"].get(exp, {})
    if not isinstance(experiment, dict):
        return DEFAULT_ANNEALING_NEIGHBORHOODS
    values = experiment.get("neighborhood_operations")
    if values is None:
        return DEFAULT_ANNEALING_NEIGHBORHOODS
    return tuple(str(value) for value in values)


def _load_suite_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _learning_rate_dir_name(value: float) -> str:
    return f"learning_rate_{value:.3f}"


def _format_topology(input_size: int, hidden_sizes: tuple[int, ...], output_size: int) -> str:
    return "-".join(str(value) for value in (input_size, *hidden_sizes, output_size))


def _dict_get(value: object, key: str) -> object:
    return value.get(key, "") if isinstance(value, dict) else ""
