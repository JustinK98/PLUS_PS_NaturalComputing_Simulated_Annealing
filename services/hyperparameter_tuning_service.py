"""Staged validation-first hyperparameter tuning for Online-Delta-SA."""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import itertools
import json
import math
from pathlib import Path
import time
import traceback
from typing import Any, Callable

import numpy as np

from activations import parse_layout_spec, random_layout_spec, sample_set_neuron_neighbor
from annealing import AnnealingConfig
from benchmark_registry import OFFICIAL_BENCHMARKS
from benchmarks import load_benchmark
from configs import (
    DEFAULT_ANNEALING_COOLING_SCHEDULE,
    SUPPORTED_ACTIVATIONS,
    DatasetConfig,
    TrainingConfig,
    default_hidden_sizes,
)
from model import ModularMLP
from services.layout_visualization_service import write_online_delta_layout_artifacts
from services.online_annealing_training_service import (
    MiniBatchCursor,
    OnlineAnnealingRequest,
    OnlineAnnealingSnapshot,
    create_online_session,
    online_run_to_completion,
)
from services.online_delta_reporting_service import build_online_delta_report
from trainer import evaluate_batch, iterate_training_epochs, train_model, train_one_batch


CONFIG_PATH = Path("configs") / "hyperparameter_tuning.json"
PHASES = (
    "training-screen",
    "training-refine",
    "delta-probe",
    "sa-screen",
    "sa-refine",
    "confirm",
)


@dataclass(frozen=True)
class HyperparameterTuningRequest:
    """CLI-facing staged tuning configuration."""

    profile: str = "overnight"
    benchmark: str = "all"
    phase: str = "full"
    workers: int = 4
    output_root: Path = Path("outputs") / "hyperparameter_tuning"
    resume: Path | None = None
    smoke: bool = False
    export_layout_frames: bool = False


@dataclass(frozen=True)
class HyperparameterTuningResult:
    """Summary returned to the CLI."""

    output_dir: Path
    benchmarks: tuple[str, ...]
    phases: tuple[str, ...]
    summary_path: Path


def run_hyperparameter_tuning(
    request: HyperparameterTuningRequest,
) -> HyperparameterTuningResult:
    """Execute requested tuning phases with checkpointed CSV artifacts."""

    if request.workers <= 0:
        raise ValueError("--workers muss positiv sein.")
    if request.phase != "full" and request.phase not in PHASES:
        raise ValueError(f"Unbekannte Tuning-Phase '{request.phase}'.")

    config = _load_profile(request.profile, smoke=request.smoke)
    benchmarks = (
        tuple(OFFICIAL_BENCHMARKS)
        if request.benchmark == "all"
        else (request.benchmark,)
    )
    root = _resolve_output_dir(request)
    root.mkdir(parents=True, exist_ok=True)
    (root / "selected").mkdir(parents=True, exist_ok=True)
    phases = PHASES if request.phase == "full" else (request.phase,)
    _write_json(
        root / "manifest.json",
        {
            "profile": request.profile,
            "benchmarks": list(benchmarks),
            "requested_phase": request.phase,
            "phases": list(phases),
            "workers": request.workers,
            "smoke": request.smoke,
            "export_layout_frames": request.export_layout_frames,
            "validation_only_before_confirmation": True,
            "created_or_resumed_at": datetime.now().isoformat(timespec="seconds"),
            "config_path": str(CONFIG_PATH),
        },
    )

    for benchmark in benchmarks:
        for phase in phases:
            _run_phase(
                root,
                config,
                benchmark,
                phase,
                request.workers,
                export_layout_frames=request.export_layout_frames,
            )

    summary_path = root / "aggregate" / "tuning_summary.csv"
    _write_tuning_summary(root, benchmarks, summary_path)
    return HyperparameterTuningResult(
        output_dir=root,
        benchmarks=benchmarks,
        phases=phases,
        summary_path=summary_path,
    )


def temperatures_for_target_acceptance(
    median_positive_delta: float,
    target_acceptance_rates: tuple[float, ...],
) -> tuple[float, ...]:
    """Derive start temperatures from a representative positive loss delta."""

    if median_positive_delta <= 0.0:
        raise ValueError("median_positive_delta muss positiv sein.")
    temperatures: list[float] = []
    for target in target_acceptance_rates:
        if not 0.0 < target < 1.0:
            raise ValueError("Ziel-Akzeptanzraten muessen zwischen 0 und 1 liegen.")
        temperatures.append(-float(median_positive_delta) / math.log(float(target)))
    return tuple(temperatures)


def _run_phase(
    root: Path,
    config: dict[str, Any],
    benchmark: str,
    phase: str,
    workers: int,
    *,
    export_layout_frames: bool,
) -> None:
    if phase == "training-screen":
        _run_training_screen(root, config, benchmark, workers)
    elif phase == "training-refine":
        _run_training_refine(root, config, benchmark, workers)
    elif phase == "delta-probe":
        _run_delta_probe(root, config, benchmark, workers)
    elif phase == "sa-screen":
        _run_sa_screen(root, config, benchmark, workers)
    elif phase == "sa-refine":
        _run_sa_refine(root, config, benchmark, workers)
    else:
        _run_confirmation(
            root,
            config,
            benchmark,
            workers,
            export_layout_frames=export_layout_frames,
        )


def _run_training_screen(
    root: Path,
    profile: dict[str, Any],
    benchmark: str,
    workers: int,
) -> None:
    benchmark_config = profile["benchmarks"][benchmark]
    configs = [
        {
            "benchmark": benchmark,
            "hidden_sizes": list(default_hidden_sizes(benchmark)),
            "learning_rate": float(learning_rate),
            "epochs": int(epochs),
            "batch_size": 32,
            "weight_scale": 1.0,
        }
        for learning_rate, epochs in itertools.product(
            benchmark_config["learning_rates"],
            benchmark_config["epochs"],
        )
    ]
    phase_dir = root / "training" / benchmark / "screen"
    rows = _run_trials(
        phase_dir,
        configs,
        tuple(range(int(profile["training_screen_runs"]))),
        _training_trial_worker,
        workers,
        phase="training-screen",
    )
    ranking = _rank_training(rows)
    _write_csv(phase_dir / "ranking.csv", ranking)


def _run_training_refine(
    root: Path,
    profile: dict[str, Any],
    benchmark: str,
    workers: int,
) -> None:
    source = root / "training" / benchmark / "screen" / "ranking.csv"
    top_screen = _top_configs(source, int(profile["training_refine_top_screen"]))
    configs = [
        {
            **base,
            "batch_size": int(batch_size),
            "weight_scale": float(weight_scale),
        }
        for base, batch_size, weight_scale in itertools.product(
            top_screen,
            profile["batch_sizes"],
            profile["weight_scales"],
        )
    ]
    phase_dir = root / "training" / benchmark / "refine"
    initial_rows = _run_trials(
        phase_dir,
        configs,
        tuple(range(int(profile["training_refine_runs"]))),
        _training_trial_worker,
        workers,
        phase="training-refine",
    )
    initial_ranking = _rank_training(initial_rows)
    top_confirmation = [
        json.loads(row["config_json"])
        for row in initial_ranking[: int(profile["training_refine_top_confirmation"])]
    ]
    rows = _run_trials(
        phase_dir,
        top_confirmation,
        tuple(range(int(profile["training_refine_confirmation_runs"]))),
        _training_trial_worker,
        workers,
        phase="training-refine",
        existing_configs=configs,
    )
    ranking = _rank_training(rows)
    _write_csv(phase_dir / "ranking.csv", ranking)
    confirmed_ranking = [
        row
        for row in ranking
        if int(row["num_runs"]) >= int(profile["training_refine_confirmation_runs"])
    ]
    selected_row = (confirmed_ranking or ranking)[0]
    selected = json.loads(selected_row["config_json"])
    stop_rule = profile["benchmarks"][benchmark].get("stop_rule")
    if isinstance(stop_rule, dict):
        selected["stop_rule_passed"] = (
            float(selected_row["mean_val_loss"]) < float(stop_rule["max_mean_val_loss"])
            and float(selected_row["mean_val_accuracy"]) > float(stop_rule["min_mean_val_accuracy"])
        )
        selected["stop_rule"] = dict(stop_rule)
    _update_selected(root, benchmark, "training", selected)
    _run_all_baseline_references(
        phase_dir / "all_baseline",
        top_confirmation,
        tuple(range(int(profile["training_refine_confirmation_runs"]))),
        workers,
    )


def _run_all_baseline_references(
    phase_dir: Path,
    configs: list[dict[str, Any]],
    seeds: tuple[int, ...],
    workers: int,
) -> None:
    references = [
        {**config, "layout_spec": activation, "label": f"all_{activation}"}
        for config, activation in itertools.product(configs, SUPPORTED_ACTIVATIONS)
    ]
    rows = _run_trials(
        phase_dir,
        references,
        seeds,
        _training_trial_worker,
        workers,
        phase="training-refine-all-baseline",
    )
    _write_csv(phase_dir / "ranking.csv", _rank_training(rows))


def _run_delta_probe(
    root: Path,
    profile: dict[str, Any],
    benchmark: str,
    workers: int,
) -> None:
    selected = _require_selected(root, benchmark, "training")
    phase_dir = root / "delta_probe" / benchmark
    configs = [
        {
            **selected,
            "probe_steps": int(profile["delta_probe_steps"]),
        }
    ]
    rows = _run_trials(
        phase_dir,
        configs,
        tuple(range(int(profile["delta_probe_runs"]))),
        _delta_probe_worker,
        workers,
        phase="delta-probe",
    )
    positive_deltas = [
        float(value)
        for row in _completed_rows(rows)
        for value in json.loads(row["result_json"]).get("positive_deltas", [])
        if float(value) > 0.0
    ]
    median_positive_delta = (
        float(np.median(np.asarray(positive_deltas, dtype=np.float64)))
        if positive_deltas
        else 1e-9
    )
    target_rates = tuple(float(value) for value in profile["target_worse_acceptance_rates"])
    temperatures = temperatures_for_target_acceptance(median_positive_delta, target_rates)
    summary = {
        "benchmark": benchmark,
        "positive_delta_count": len(positive_deltas),
        "median_positive_delta": median_positive_delta,
        "positive_delta_quantiles": {
            str(quantile): float(np.quantile(positive_deltas, quantile))
            if positive_deltas
            else 0.0
            for quantile in (0.25, 0.5, 0.75, 0.9, 0.95)
        },
        "target_worse_acceptance_rates": list(target_rates),
        "start_temperatures": list(temperatures),
    }
    _write_json(phase_dir / "probe_summary.json", summary)
    _write_csv(
        phase_dir / "ranking.csv",
        [
            {
                "benchmark": benchmark,
                "target_worse_acceptance_rate": target,
                "start_temperature": temperature,
                "median_positive_delta": median_positive_delta,
            }
            for target, temperature in zip(target_rates, temperatures, strict=True)
        ],
    )
    _update_selected(root, benchmark, "delta_probe", summary)


def _run_sa_screen(
    root: Path,
    profile: dict[str, Any],
    benchmark: str,
    workers: int,
) -> None:
    training = _require_selected(root, benchmark, "training")
    probe = _require_selected(root, benchmark, "delta_probe")
    benchmark_config = profile["benchmarks"][benchmark]
    diagnostic_only = training.get("stop_rule_passed") is False
    product = [
        {
            "benchmark": benchmark,
            "hidden_sizes": list(default_hidden_sizes(benchmark)),
            "epochs": int(training["epochs"]),
            "retrain_learning_rate": float(training["learning_rate"]),
            "retrain_batch_size": int(training["batch_size"]),
            "weight_scale": float(training["weight_scale"]),
            "online_learning_rate": float(training["learning_rate"]) * float(lr_factor),
            "online_batch_size": int(online_batch_size),
            "start_temperature": float(start_temperature),
            "cooling_parameter": float(cooling_parameter),
            "iterations_per_temperature": int(iterations),
            "max_steps": int(max_steps),
            "min_temperature": 0.0,
            "diagnostic_only": diagnostic_only,
        }
        for start_temperature, cooling_parameter, iterations, max_steps, lr_factor, online_batch_size
        in itertools.product(
            probe["start_temperatures"],
            profile["cooling_parameters"],
            profile["iterations_per_temperature"],
            benchmark_config["max_steps"],
            profile["online_learning_rate_factors"],
            profile["online_batch_sizes"],
        )
    ]
    rng = np.random.default_rng(42)
    requested_candidates = 1 if diagnostic_only else int(benchmark_config["sa_screen_candidates"])
    sample_count = min(requested_candidates, len(product))
    indices = rng.choice(len(product), size=sample_count, replace=False)
    configs = [product[int(index)] for index in indices]
    phase_dir = root / "sa" / benchmark / "screen"
    if diagnostic_only:
        _write_json(
            phase_dir / "diagnostic_only.json",
            {
                "benchmark": benchmark,
                "reason": "training_stop_rule_not_reached",
                "stop_rule": training.get("stop_rule", {}),
                "screened_sa_configurations": sample_count,
            },
        )
    rows = _run_trials(
        phase_dir,
        configs,
        tuple(range(int(profile["sa_screen_runs"]))),
        _sa_search_worker,
        workers,
        phase="sa-screen",
    )
    _write_csv(phase_dir / "ranking.csv", _rank_sa_screen(rows))


def _run_sa_refine(
    root: Path,
    profile: dict[str, Any],
    benchmark: str,
    workers: int,
) -> None:
    source = root / "sa" / benchmark / "screen" / "ranking.csv"
    diagnostic_only = (root / "sa" / benchmark / "screen" / "diagnostic_only.json").exists()
    top_count = 1 if diagnostic_only else int(profile["sa_refine_top"])
    configs = _top_configs(source, top_count, require_healthy=True)
    phase_dir = root / "sa" / benchmark / "refine"
    rows = _run_trials(
        phase_dir,
        configs,
        tuple(range(int(profile["sa_refine_runs"]))),
        _sa_refine_worker,
        workers,
        phase="sa-refine",
    )
    ranking = _rank_sa_refine(rows)
    _write_csv(phase_dir / "ranking.csv", ranking)
    _update_selected(root, benchmark, "sa", json.loads(ranking[0]["config_json"]))


def _run_confirmation(
    root: Path,
    profile: dict[str, Any],
    benchmark: str,
    workers: int,
    *,
    export_layout_frames: bool,
) -> None:
    config = _require_selected(root, benchmark, "sa")
    phase_dir = root / "confirmation" / benchmark
    phase_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(range(int(profile["confirmation_runs"])))
    online_rows = _run_confirmation_online(
        phase_dir,
        config,
        seeds,
        workers,
        export_layout_frames=export_layout_frames,
    )
    baseline_configs = [
        {
            "benchmark": benchmark,
            "hidden_sizes": list(default_hidden_sizes(benchmark)),
            "epochs": int(config["epochs"]),
            "learning_rate": float(config["retrain_learning_rate"]),
            "batch_size": int(config["retrain_batch_size"]),
            "weight_scale": float(config["weight_scale"]),
        }
    ]
    random_rows = _run_trials(
        phase_dir / "random_baseline",
        baseline_configs,
        seeds,
        _full_training_trial_worker,
        workers,
        phase="confirm-random-baseline",
    )
    all_configs = [
        {**baseline_configs[0], "layout_spec": activation, "label": f"all_{activation}"}
        for activation in SUPPORTED_ACTIVATIONS
    ]
    all_rows = _run_trials(
        phase_dir / "all_baseline",
        all_configs,
        seeds,
        _full_training_trial_worker,
        workers,
        phase="confirm-all-baseline",
    )
    summary_rows = [
        *_rank_confirmation_online(online_rows),
        *_rank_training(random_rows),
        *_rank_training(all_rows),
    ]
    _write_csv(phase_dir / "summary.csv", summary_rows)
    build_online_delta_report(phase_dir / "online_delta", phase_dir / "aggregate")
    _update_selected(
        root,
        benchmark,
        "confirmation",
        {
            "summary_path": str(phase_dir / "summary.csv"),
            "online_delta_report": str(phase_dir / "aggregate"),
            "runs": len(seeds),
            "diagnostic_only": bool(config.get("diagnostic_only", False)),
        },
    )


def _run_confirmation_online(
    phase_dir: Path,
    config: dict[str, Any],
    seeds: tuple[int, ...],
    workers: int,
    *,
    export_layout_frames: bool,
) -> list[dict[str, str]]:
    online_dir = phase_dir / "online_delta"
    runs_dir = online_dir / "runs"
    plots_dir = online_dir / "plots_single"
    frames_dir = online_dir / "layout_frames"
    runs_dir.mkdir(parents=True, exist_ok=True)
    rows_path = online_dir / "trials.csv"
    existing = {row["trial_id"]: row for row in _read_csv(rows_path)}
    rows = list(existing.values())
    pending = [seed for seed in seeds if existing.get(f"seed_{seed:04d}", {}).get("status") != "completed"]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(_timed_confirmation_online_worker, config, seed): seed
            for seed in pending
        }
        for future in as_completed(future_map):
            seed = future_map[future]
            trial_id = f"seed_{seed:04d}"
            try:
                result, snapshot, payload, duration = future.result()
                _write_json(runs_dir / f"{trial_id}.json", payload)
                write_online_delta_layout_artifacts(
                    snapshot,
                    plots_dir=plots_dir,
                    frames_root=frames_dir,
                    run_id=trial_id,
                    export_frames=export_layout_frames,
                )
                row = _trial_row(trial_id, "confirmation", seed, config, result, "completed", "", duration)
            except Exception as exc:  # pragma: no cover - defensive checkpoint
                row = _trial_row(trial_id, "confirmation", seed, config, {}, "failed", f"{exc}\n{traceback.format_exc()}", 0.0)
            existing[trial_id] = row
            rows = list(existing.values())
            _write_csv(rows_path, rows)
    return rows


def _training_trial_worker(config: dict[str, Any], seed: int) -> dict[str, Any]:
    return _train_layout(
        config,
        seed,
        include_test=False,
        layout_spec=config.get("layout_spec"),
    )


def _full_training_trial_worker(config: dict[str, Any], seed: int) -> dict[str, Any]:
    return _train_layout(
        config,
        seed,
        include_test=True,
        layout_spec=config.get("layout_spec"),
    )


def _train_layout(
    config: dict[str, Any],
    seed: int,
    *,
    include_test: bool,
    layout_spec: str | None,
) -> dict[str, Any]:
    benchmark = str(config["benchmark"])
    hidden_sizes = tuple(int(value) for value in config["hidden_sizes"])
    effective_layout_spec = layout_spec or random_layout_spec(hidden_sizes, seed)
    dataset = load_benchmark(DatasetConfig(name=benchmark, random_state=seed))
    model = ModularMLP(
        input_size=dataset.input_size,
        hidden_sizes=hidden_sizes,
        output_size=dataset.model_output_size,
        layout=parse_layout_spec(effective_layout_spec, hidden_sizes),
        num_classes=dataset.output_size,
        weight_scale=float(config["weight_scale"]),
        random_state=seed,
    )
    training_config = TrainingConfig(
        epochs=int(config["epochs"]),
        learning_rate=float(config["learning_rate"]),
        batch_size=int(config["batch_size"]),
        random_state=seed,
    )
    if include_test:
        training_result = train_model(model, dataset, training_config)
        metrics = {
            "train_loss": float(training_result.history["train_loss"][-1]),
            "val_loss": float(training_result.history["val_loss"][-1]),
            "test_loss": float(training_result.test_metrics["loss"]),
            "train_accuracy": float(training_result.history["train_acc"][-1]),
            "val_accuracy": float(training_result.history["val_acc"][-1]),
            "test_accuracy": float(training_result.test_metrics["accuracy"]),
        }
    else:
        last = None
        for last in iterate_training_epochs(model, dataset, training_config):
            pass
        if last is None:
            raise ValueError("Training lieferte keine Epoche.")
        metrics = {
            "train_loss": float(last.train_loss),
            "val_loss": float(last.val_loss),
            "train_accuracy": float(last.train_acc),
            "val_accuracy": float(last.val_acc),
        }
    return {
        **metrics,
        "layout_spec": parse_layout_spec(effective_layout_spec, hidden_sizes).to_compact_spec(),
        "label": str(config.get("label", "random_start_baseline")),
    }


def _delta_probe_worker(config: dict[str, Any], seed: int) -> dict[str, Any]:
    benchmark = str(config["benchmark"])
    hidden_sizes = tuple(int(value) for value in config["hidden_sizes"])
    dataset = load_benchmark(DatasetConfig(name=benchmark, random_state=seed))
    start_layout = parse_layout_spec(random_layout_spec(hidden_sizes, seed), hidden_sizes)
    model = ModularMLP(
        input_size=dataset.input_size,
        hidden_sizes=hidden_sizes,
        output_size=dataset.model_output_size,
        layout=start_layout,
        num_classes=dataset.output_size,
        weight_scale=float(config["weight_scale"]),
        random_state=seed,
    )
    rng = np.random.default_rng(seed)
    cursor = MiniBatchCursor(
        train_size=dataset.train_size,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        rng=np.random.default_rng(seed),
    )
    deltas: list[float] = []
    for _ in range(int(config["probe_steps"])):
        indices = cursor.current_indices()
        X_batch, y_batch = dataset.X_train[indices], dataset.y_train[indices]
        before_loss, _ = evaluate_batch(model, X_batch, y_batch)
        previous_layout = model.layout
        candidate = sample_set_neuron_neighbor(previous_layout, rng)
        model.set_layout(candidate.layout)
        candidate_loss, _ = evaluate_batch(model, X_batch, y_batch)
        model.set_layout(previous_layout)
        deltas.append(float(candidate_loss - before_loss))
        train_one_batch(model, X_batch, y_batch, float(config["learning_rate"]))
        cursor.advance()
    positive = [value for value in deltas if value > 0.0]
    return {
        "delta_count": len(deltas),
        "positive_delta_count": len(positive),
        "positive_deltas": positive,
    }


def _sa_search_worker(config: dict[str, Any], seed: int) -> dict[str, Any]:
    result, _snapshot = _run_sa(config, seed, include_test=False)
    return result


def _sa_refine_worker(config: dict[str, Any], seed: int) -> dict[str, Any]:
    result, snapshot = _run_sa(config, seed, include_test=False)
    start = _train_layout(
        {
            "benchmark": config["benchmark"],
            "hidden_sizes": config["hidden_sizes"],
            "epochs": config["epochs"],
            "learning_rate": config["retrain_learning_rate"],
            "batch_size": config["retrain_batch_size"],
            "weight_scale": config["weight_scale"],
        },
        seed,
        include_test=False,
        layout_spec=snapshot.start_evaluation.layout.to_compact_spec(),  # type: ignore[union-attr]
    )
    best = _train_layout(
        {
            "benchmark": config["benchmark"],
            "hidden_sizes": config["hidden_sizes"],
            "epochs": config["epochs"],
            "learning_rate": config["retrain_learning_rate"],
            "batch_size": config["retrain_batch_size"],
            "weight_scale": config["weight_scale"],
        },
        seed,
        include_test=False,
        layout_spec=snapshot.best_evaluation.layout.to_compact_spec(),  # type: ignore[union-attr]
    )
    return {
        **result,
        "same_random_start_retrained_val_loss": float(start["val_loss"]),
        "best_layout_from_sa_retrained_val_loss": float(best["val_loss"]),
        "paired_val_loss_improvement": float(start["val_loss"]) - float(best["val_loss"]),
        "paired_win": float(best["val_loss"]) < float(start["val_loss"]),
    }


def _confirmation_online_worker(
    config: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], OnlineAnnealingSnapshot, dict[str, Any]]:
    result, snapshot = _run_sa(config, seed, include_test=True)
    base = {
        "benchmark": config["benchmark"],
        "hidden_sizes": config["hidden_sizes"],
        "epochs": config["epochs"],
        "learning_rate": config["retrain_learning_rate"],
        "batch_size": config["retrain_batch_size"],
        "weight_scale": config["weight_scale"],
    }
    comparisons = []
    for label, layout_spec in (
        ("same_random_start_retrained", snapshot.start_evaluation.layout.to_compact_spec()),  # type: ignore[union-attr]
        ("best_layout_from_sa_retrained", snapshot.best_evaluation.layout.to_compact_spec()),  # type: ignore[union-attr]
        ("end_layout_from_sa_retrained", snapshot.current_evaluation.layout.to_compact_spec()),  # type: ignore[union-attr]
    ):
        comparisons.append(
            {
                **_train_layout(base, seed, include_test=True, layout_spec=layout_spec),
                "label": label,
            }
        )
    for label, evaluation in (
        ("best_inherited_model_from_sa", snapshot.best_evaluation),
        ("end_inherited_model_from_sa", snapshot.current_evaluation),
    ):
        comparisons.append(
            {
                "label": label,
                "evaluation_type": "inherited",
                "layout_spec": evaluation.layout.to_compact_spec(),  # type: ignore[union-attr]
                "train_loss": float(evaluation.train_loss),  # type: ignore[union-attr]
                "val_loss": float(evaluation.val_loss),  # type: ignore[union-attr]
                "test_loss": float(evaluation.test_loss),  # type: ignore[union-attr]
                "train_accuracy": float(evaluation.train_accuracy),  # type: ignore[union-attr]
                "val_accuracy": float(evaluation.val_accuracy),  # type: ignore[union-attr]
                "test_accuracy": float(evaluation.test_accuracy),  # type: ignore[union-attr]
            }
        )
    best_comparison = next(item for item in comparisons if item["label"] == "best_layout_from_sa_retrained")
    start_comparison = next(item for item in comparisons if item["label"] == "same_random_start_retrained")
    result.update(
        {
            "same_random_start_retrained_val_loss": float(start_comparison["val_loss"]),
            "best_layout_from_sa_retrained_val_loss": float(best_comparison["val_loss"]),
            "best_layout_from_sa_retrained_test_loss": float(best_comparison["test_loss"]),
            "best_layout_from_sa_retrained_test_accuracy": float(best_comparison["test_accuracy"]),
            "paired_val_loss_improvement": float(start_comparison["val_loss"]) - float(best_comparison["val_loss"]),
            "paired_win": float(best_comparison["val_loss"]) < float(start_comparison["val_loss"]),
        }
    )
    payload = {
        "kind": "experiment_suite_online_delta_run",
        "experiment_id": "tuning_confirmation_online_delta",
        "benchmark": config["benchmark"],
        "run_index": seed,
        "seed": seed,
        "training_seed": seed,
        "layout_seed": seed,
        "learning_rate": config["retrain_learning_rate"],
        "online_learning_rate": config["online_learning_rate"],
        "online_batch_size": config["online_batch_size"],
        "start_layout": snapshot.start_evaluation.layout.to_compact_spec(),  # type: ignore[union-attr]
        "best_layout": snapshot.best_evaluation.layout.to_compact_spec(),  # type: ignore[union-attr]
        "end_layout": snapshot.current_evaluation.layout.to_compact_spec(),  # type: ignore[union-attr]
        "accepted_steps": snapshot.accepted_steps,
        "acceptance_rate": snapshot.acceptance_rate,
        "online_history": _snapshot_history(snapshot),
        "final_comparisons": comparisons,
    }
    return result, snapshot, payload


def _timed_confirmation_online_worker(
    config: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], OnlineAnnealingSnapshot, dict[str, Any], float]:
    started = time.perf_counter()
    result, snapshot, payload = _confirmation_online_worker(config, seed)
    return result, snapshot, payload, time.perf_counter() - started


def _run_sa(
    config: dict[str, Any],
    seed: int,
    *,
    include_test: bool,
) -> tuple[dict[str, Any], OnlineAnnealingSnapshot]:
    hidden_sizes = tuple(int(value) for value in config["hidden_sizes"])
    session = create_online_session(
        OnlineAnnealingRequest(
            dataset_config=DatasetConfig(name=str(config["benchmark"]), random_state=seed),
            hidden_sizes=hidden_sizes,
            layout_spec=random_layout_spec(hidden_sizes, seed),
            training_config=TrainingConfig(
                epochs=int(config["epochs"]),
                learning_rate=float(config["online_learning_rate"]),
                batch_size=int(config["online_batch_size"]),
                random_state=seed,
            ),
            annealing_config=AnnealingConfig(
                start_temperature=float(config["start_temperature"]),
                cooling_schedule=DEFAULT_ANNEALING_COOLING_SCHEDULE,
                cooling_parameter=float(config["cooling_parameter"]),
                iterations_per_temperature=int(config["iterations_per_temperature"]),
                max_steps=int(config["max_steps"]),
                min_temperature=float(config["min_temperature"]),
                neighborhood_operations=("set_neuron",),
            ),
            weight_scale=float(config["weight_scale"]),
            random_state=seed,
            include_test_metrics=include_test,
        )
    )
    snapshot = online_run_to_completion(session, "en")
    if snapshot.start_evaluation is None or snapshot.best_evaluation is None or snapshot.current_evaluation is None:
        raise ValueError("Online-Delta-SA lieferte keinen vollstaendigen Snapshot.")
    start_val = float(snapshot.start_evaluation.val_loss)
    best_val = float(snapshot.best_evaluation.val_loss)
    end_val = float(snapshot.current_evaluation.val_loss)
    finite = all(math.isfinite(value) for value in (start_val, best_val, end_val))
    return (
        {
            "acceptance_rate": float(snapshot.acceptance_rate),
            "start_val_loss": start_val,
            "best_observed_val_loss": best_val,
            "end_val_loss": end_val,
            "progress_improvement": start_val - best_val,
            "step_count": len(snapshot.history),
            "finite_metrics": finite,
        },
        snapshot,
    )


def _snapshot_history(snapshot: OnlineAnnealingSnapshot) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if snapshot.start_evaluation is not None:
        start = snapshot.start_evaluation
        rows.append(
            {
                "step_index": 0,
                "temperature": float(snapshot.history[0].temperature) if snapshot.history else float(snapshot.current_temperature),
                "previous_layout": start.layout.to_compact_spec(),
                "candidate_layout": start.layout.to_compact_spec(),
                "delta": 0.0,
                "accepted": True,
                "reason_code": "start",
                "neighbor_label": "start",
                "batch_loss_before": float(start.objective_value),
                "candidate_loss_after": float(start.objective_value),
                "validation_loss_after_update": float(start.val_loss),
            }
        )
    rows.extend(
        {
            "step_index": int(step.step_index),
            "temperature": float(step.temperature),
            "previous_layout": step.previous_layout.to_compact_spec(),
            "candidate_layout": step.candidate_layout.to_compact_spec(),
            "delta": float(step.delta),
            "accepted": bool(step.accepted),
            "reason_code": step.reason_code,
            "neighbor_label": step.neighbor_label,
            "batch_loss_before": float(step.batch_loss_before),
            "candidate_loss_after": float(step.candidate_loss_after),
            "validation_loss_after_update": float(step.validation_loss_after_update),
        }
        for step in snapshot.history
    )
    return rows


def _run_trials(
    phase_dir: Path,
    configs: list[dict[str, Any]],
    seeds: tuple[int, ...],
    worker: Callable[[dict[str, Any], int], dict[str, Any]],
    workers: int,
    *,
    phase: str,
    existing_configs: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    phase_dir.mkdir(parents=True, exist_ok=True)
    all_configs = _deduplicate_configs([*(existing_configs or []), *configs])
    config_ids = {
        json.dumps(config, sort_keys=True): f"cfg_{config_index:04d}"
        for config_index, config in enumerate(all_configs, start=1)
    }
    _write_json(phase_dir / "config.json", {"phase": phase, "configs": all_configs, "seeds": list(seeds)})
    rows_path = phase_dir / "trials.csv"
    rows_by_id = {row["trial_id"]: row for row in _read_csv(rows_path)}
    jobs: list[tuple[str, str, dict[str, Any], int]] = []
    active_configs = all_configs if existing_configs is None else _deduplicate_configs(configs)
    for config in active_configs:
        config_id = config_ids[json.dumps(config, sort_keys=True)]
        for seed in seeds:
            trial_id = f"{config_id}_seed_{seed:04d}"
            if rows_by_id.get(trial_id, {}).get("status") == "completed":
                continue
            jobs.append((trial_id, config_id, config, seed))
            rows_by_id[trial_id] = _trial_row(trial_id, config_id, seed, config, {}, "pending", "", 0.0)
    _write_csv(rows_path, list(rows_by_id.values()))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {}
        for trial_id, config_id, config, seed in jobs:
            rows_by_id[trial_id]["status"] = "running"
            future_map[executor.submit(_timed_worker, worker, config, seed)] = (trial_id, config_id, config, seed)
        _write_csv(rows_path, list(rows_by_id.values()))
        for future in as_completed(future_map):
            trial_id, config_id, config, seed = future_map[future]
            try:
                result, duration = future.result()
                row = _trial_row(trial_id, config_id, seed, config, result, "completed", "", duration)
            except Exception as exc:  # pragma: no cover - defensive checkpoint
                row = _trial_row(trial_id, config_id, seed, config, {}, "failed", f"{exc}\n{traceback.format_exc()}", 0.0)
            rows_by_id[trial_id] = row
            _write_csv(rows_path, list(rows_by_id.values()))
    return list(rows_by_id.values())


def _timed_worker(
    worker: Callable[[dict[str, Any], int], dict[str, Any]],
    config: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    return worker(config, seed), time.perf_counter() - started


def _trial_row(
    trial_id: str,
    config_id: str,
    seed: int,
    config: dict[str, Any],
    result: dict[str, Any],
    status: str,
    error: str,
    duration: float,
) -> dict[str, str]:
    row = {
        "trial_id": trial_id,
        "config_id": config_id,
        "seed": str(seed),
        "status": status,
        "duration_seconds": f"{duration:.6f}",
        "error": error,
        "config_json": json.dumps(config, sort_keys=True),
        "result_json": json.dumps(result, sort_keys=True),
    }
    for key, value in result.items():
        if isinstance(value, (bool, int, float, str)):
            row[key] = str(value)
    return row


def _rank_training(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return _rank_rows(
        rows,
        metric_names=("val_loss", "val_accuracy"),
        sort_key=lambda row: (
            float(row["mean_val_loss"]),
            float(row["std_val_loss"]),
            -float(row["mean_val_accuracy"]),
        ),
    )


def _rank_sa_screen(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    ranking = _rank_rows(
        rows,
        metric_names=("acceptance_rate", "best_observed_val_loss", "progress_improvement", "finite_metrics"),
        sort_key=lambda row: (
            float(row["mean_best_observed_val_loss"]),
            -float(row["mean_progress_improvement"]),
        ),
    )
    for row in ranking:
        acceptance = float(row["mean_acceptance_rate"])
        progress = float(row["mean_progress_improvement"])
        finite = float(row["mean_finite_metrics"]) == 1.0
        row["healthy"] = 0.15 <= acceptance <= 0.80 and progress > 0.0 and finite
    return sorted(
        ranking,
        key=lambda row: (
            not bool(row["healthy"]),
            float(row["mean_best_observed_val_loss"]),
            -float(row["mean_progress_improvement"]),
        ),
    )


def _rank_sa_refine(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    ranking = _rank_rows(
        rows,
        metric_names=(
            "acceptance_rate",
            "best_layout_from_sa_retrained_val_loss",
            "paired_val_loss_improvement",
            "paired_win",
        ),
        sort_key=lambda row: (
            float(row["mean_best_layout_from_sa_retrained_val_loss"]),
            -float(row["mean_paired_val_loss_improvement"]),
            -float(row["mean_paired_win"]),
        ),
    )
    for row in ranking:
        row["win_rate"] = row["mean_paired_win"]
    return ranking


def _rank_confirmation_online(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return _rank_rows(
        rows,
        metric_names=(
            "acceptance_rate",
            "best_layout_from_sa_retrained_val_loss",
            "best_layout_from_sa_retrained_test_loss",
            "best_layout_from_sa_retrained_test_accuracy",
            "paired_val_loss_improvement",
            "paired_win",
        ),
        sort_key=lambda row: float(row["mean_best_layout_from_sa_retrained_val_loss"]),
    )


def _rank_rows(
    rows: list[dict[str, str]],
    *,
    metric_names: tuple[str, ...],
    sort_key: Callable[[dict[str, Any]], Any],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in _completed_rows(rows):
        grouped.setdefault(row["config_id"], []).append(row)
    ranking: list[dict[str, Any]] = []
    for config_id, group in grouped.items():
        item: dict[str, Any] = {
            "config_id": config_id,
            "num_runs": len(group),
            "config_json": group[0]["config_json"],
        }
        for metric_name in metric_names:
            values = np.asarray([_metric_float(row[metric_name]) for row in group], dtype=np.float64)
            item[f"mean_{metric_name}"] = float(values.mean())
            item[f"std_{metric_name}"] = float(values.std(ddof=0))
        ranking.append(item)
    return sorted(ranking, key=sort_key)


def _completed_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("status") == "completed"]


def _metric_float(value: str) -> float:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return 1.0
    if normalized == "false":
        return 0.0
    return float(value)


def _top_configs(
    ranking_path: Path,
    count: int,
    *,
    require_healthy: bool = False,
) -> list[dict[str, Any]]:
    rows = _read_csv(ranking_path)
    if require_healthy:
        healthy_rows = [row for row in rows if str(row.get("healthy", "")).lower() in {"true", "1"}]
        rows = healthy_rows or rows[:1]
    if not rows:
        raise ValueError(f"Keine geeigneten Ranking-Eintraege in {ranking_path}.")
    return [json.loads(row["config_json"]) for row in rows[:count]]


def _require_selected(root: Path, benchmark: str, key: str) -> dict[str, Any]:
    path = root / "selected" / f"{benchmark}.json"
    if not path.exists():
        raise ValueError(f"Fehlende Auswahl {path}. Fuehre die vorherigen Phasen zuerst aus oder nutze --phase full.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if key not in payload:
        raise ValueError(f"Fehlende Auswahl '{key}' in {path}.")
    return dict(payload[key])


def _update_selected(root: Path, benchmark: str, key: str, value: dict[str, Any]) -> None:
    path = root / "selected" / f"{benchmark}.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"benchmark": benchmark}
    payload[key] = value
    _write_json(path, payload)


def _write_tuning_summary(
    root: Path,
    benchmarks: tuple[str, ...],
    path: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for benchmark in benchmarks:
        selected_path = root / "selected" / f"{benchmark}.json"
        if not selected_path.exists():
            continue
        payload = json.loads(selected_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "benchmark": benchmark,
                "training_json": json.dumps(payload.get("training", {}), sort_keys=True),
                "delta_probe_json": json.dumps(payload.get("delta_probe", {}), sort_keys=True),
                "sa_json": json.dumps(payload.get("sa", {}), sort_keys=True),
                "confirmation_json": json.dumps(payload.get("confirmation", {}), sort_keys=True),
            }
        )
    _write_csv(path, rows)


def _load_profile(profile_name: str, *, smoke: bool) -> dict[str, Any]:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    try:
        profile = dict(payload["profiles"][profile_name])
    except KeyError as exc:
        raise ValueError(f"Unbekanntes Tuning-Profil '{profile_name}'.") from exc
    if not smoke:
        return profile
    profile.update(
        {
            "training_screen_runs": 1,
            "training_refine_runs": 1,
            "training_refine_confirmation_runs": 1,
            "training_refine_top_screen": 1,
            "training_refine_top_confirmation": 1,
            "delta_probe_runs": 1,
            "delta_probe_steps": 2,
            "sa_screen_runs": 1,
            "sa_refine_runs": 1,
            "sa_refine_top": 1,
            "confirmation_runs": 1,
            "target_worse_acceptance_rates": [0.4],
            "batch_sizes": [32],
            "weight_scales": [1.0],
            "cooling_parameters": [0.95],
            "iterations_per_temperature": [1],
            "online_learning_rate_factors": [1.0],
            "online_batch_sizes": [32],
        }
    )
    profile["benchmarks"] = {
        benchmark: {
            **values,
            "learning_rates": [values["learning_rates"][0]],
            "epochs": [1],
            "max_steps": [2],
            "sa_screen_candidates": 1,
        }
        for benchmark, values in profile["benchmarks"].items()
    }
    return profile


def _resolve_output_dir(request: HyperparameterTuningRequest) -> Path:
    if request.resume is not None:
        if not request.resume.exists():
            raise ValueError(f"--resume Pfad existiert nicht: {request.resume}")
        return request.resume
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return request.output_root / f"{timestamp}_{request.profile}"


def _deduplicate_configs(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for config in configs:
        key = json.dumps(config, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(config)
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
