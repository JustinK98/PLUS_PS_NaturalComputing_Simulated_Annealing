"""Staged validation-first hyperparameter tuning for Online-Delta-SA."""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import hashlib
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
from annealing_schedules import SUPPORTED_COOLING_SCHEDULES
from benchmark_registry import OFFICIAL_BENCHMARKS
from benchmarks import load_benchmark
from configs import (
    DEFAULT_ANNEALING_COOLING_SCHEDULE,
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
from services.seed_schedule_service import SeedSchedule, build_seed_schedule
from trainer import evaluate_batch, iterate_training_epochs, train_model, train_one_batch


CONFIG_PATH = Path("configs") / "hyperparameter_tuning.json"
PHASES = (
    "training-screen",
    "training-refine",
    "delta-probe",
    "online-budget-probe",
    "sa-screen",
    "sa-refine",
    "confirm",
)
ONLINE_PHASES = (
    "delta-probe",
    "online-budget-probe",
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
    import_training_from: Path | None = None


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
    if request.phase not in {"full", "online-full"} and request.phase not in PHASES:
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
    phases = (
        PHASES
        if request.phase == "full"
        else ONLINE_PHASES
        if request.phase == "online-full"
        else (request.phase,)
    )
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
            "import_training_from": (
                str(request.import_training_from) if request.import_training_from is not None else None
            ),
        },
    )

    for benchmark in benchmarks:
        if request.import_training_from is not None:
            _import_training_selection(root, request.import_training_from, benchmark)
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


def _import_training_selection(root: Path, source_root: Path, benchmark: str) -> None:
    """Reuse the unchanged SGD selection from a reviewed predecessor run."""

    source_path = source_root / "selected" / f"{benchmark}.json"
    if not source_path.exists():
        raise ValueError(f"Fehlende importierbare Trainingsauswahl: {source_path}")
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    training = source_payload.get("training")
    if not isinstance(training, dict):
        raise ValueError(f"{source_path} enthaelt keine Trainingsauswahl.")
    expected_hidden_sizes = list(default_hidden_sizes(benchmark))
    if training.get("benchmark") != benchmark or training.get("hidden_sizes") != expected_hidden_sizes:
        raise ValueError(f"Trainingsauswahl in {source_path} passt nicht zu {benchmark}.")
    _update_selected(root, benchmark, "training", dict(training))
    provenance_dir = root / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        provenance_dir / f"{benchmark}_training_import.json",
        {
            "kind": "imported_training_selection",
            "benchmark": benchmark,
            "source": str(source_path),
            "config_sha256": _config_sha256(training),
        },
    )


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
    elif phase == "online-budget-probe":
        _run_online_budget_probe(root, config, benchmark, workers)
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


def _run_online_budget_probe(
    root: Path,
    profile: dict[str, Any],
    benchmark: str,
    workers: int,
) -> None:
    """Find the smallest stable online-training budget before SA screening."""

    training = _require_selected(root, benchmark, "training")
    probe = _require_selected(root, benchmark, "delta_probe")
    base_checkpoints = tuple(float(value) for value in profile["online_budget_probe_epochs"])
    extended_checkpoints = tuple(float(value) for value in profile["online_budget_probe_extended_epochs"])
    phase_dir = root / "online_budget_probe" / benchmark
    target_rates = tuple(float(value) for value in probe["target_worse_acceptance_rates"])
    temperatures = tuple(float(value) for value in probe["start_temperatures"])
    representative_temperature = temperatures[min(range(len(target_rates)), key=lambda index: abs(target_rates[index] - 0.4))]

    def configs_for(checkpoints: tuple[float, ...]) -> list[dict[str, Any]]:
        return [
            {
                "benchmark": benchmark,
                "hidden_sizes": list(default_hidden_sizes(benchmark)),
                "epochs": int(training["epochs"]),
                "retrain_learning_rate": float(training["learning_rate"]),
                "retrain_batch_size": int(training["batch_size"]),
                "weight_scale": float(training["weight_scale"]),
                "online_learning_rate": float(training["learning_rate"]),
                "online_batch_size": int(profile["online_budget_probe_batch_size"]),
                "start_temperature": representative_temperature,
                "cooling_schedule": DEFAULT_ANNEALING_COOLING_SCHEDULE,
                "cooling_parameter": float(profile["online_budget_probe_cooling_parameter"]),
                "iterations_per_temperature": int(profile["online_budget_probe_iterations_per_temperature"]),
                "max_steps": _proposal_safety_limit(
                    benchmark,
                    int(profile["online_budget_probe_batch_size"]),
                    float(max(checkpoints)),
                ),
                "min_temperature": 0.0,
                "target_online_epochs": float(target_online_epochs),
                "stop_at_target_online_epochs": True,
            }
            for target_online_epochs in checkpoints
        ]

    rows = _run_trials(
        phase_dir,
        configs_for(base_checkpoints),
        tuple(range(int(profile["online_budget_probe_runs"]))),
        _online_budget_probe_worker,
        workers,
        phase="online-budget-probe",
    )
    ranking = _rank_online_budget_probe(rows)
    selected_budget = _select_online_budget(ranking)
    if selected_budget is None and extended_checkpoints:
        rows = _run_trials(
            phase_dir,
            configs_for(extended_checkpoints),
            tuple(range(int(profile["online_budget_probe_runs"]))),
            _online_budget_probe_worker,
            workers,
            phase="online-budget-probe",
            existing_configs=configs_for(base_checkpoints),
        )
        ranking = _rank_online_budget_probe(rows)
        selected_budget = _select_online_budget(ranking)
    if selected_budget is None:
        selected_budget = float(max((*base_checkpoints, *extended_checkpoints)))
        plateau_status = "plateau_not_reached"
    else:
        plateau_status = "plateau_reached"
    _write_csv(phase_dir / "ranking.csv", ranking)
    _update_selected(
        root,
        benchmark,
        "online_budget",
        {
            "target_online_epochs": selected_budget,
            "plateau_status": plateau_status,
            "ranking_path": str(phase_dir / "ranking.csv"),
        },
    )


def _run_sa_screen(
    root: Path,
    profile: dict[str, Any],
    benchmark: str,
    workers: int,
) -> None:
    training = _require_selected(root, benchmark, "training")
    probe = _require_selected(root, benchmark, "delta_probe")
    online_budget = _require_selected(root, benchmark, "online_budget")
    benchmark_config = profile["benchmarks"][benchmark]
    diagnostic_only = training.get("stop_rule_passed") is False
    product: list[dict[str, Any]] = []
    for start_temperature, iterations, lr_factor, online_batch_size in itertools.product(
        probe["start_temperatures"],
        profile["iterations_per_temperature"],
        profile["online_learning_rate_factors"],
        profile["online_batch_sizes"],
    ):
        max_steps = _proposal_safety_limit(
            benchmark,
            int(online_batch_size),
            float(online_budget["target_online_epochs"]),
        )
        for cooling in _cooling_candidates(
            profile,
            start_temperature=float(start_temperature),
            iterations_per_temperature=int(iterations),
            max_steps=max_steps,
        ):
            product.append(
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
                    **cooling,
                    "iterations_per_temperature": int(iterations),
                    "max_steps": max_steps,
                    "min_temperature": float(start_temperature) * 0.01,
                    "target_online_epochs": float(online_budget["target_online_epochs"]),
                    "stop_at_target_online_epochs": False,
                    "diagnostic_only": diagnostic_only,
                }
            )
    rng = np.random.default_rng(42)
    requested_candidates = 1 if diagnostic_only else int(benchmark_config["sa_screen_candidates"])
    sample_count = min(requested_candidates, len(product))
    configs = (
        [product[0]]
        if diagnostic_only
        else _stratified_sample_configs(
            product,
            sample_count,
            group_key="cooling_schedule",
            rng=rng,
        )
    )
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
    config = {
        **_require_selected(root, benchmark, "sa"),
        "confirmation_replicate_count": int(profile["confirmation_replicate_count"]),
    }
    phase_dir = root / "confirmation" / benchmark
    phase_dir.mkdir(parents=True, exist_ok=True)
    layout_count = int(profile["confirmation_layout_count"])
    replicate_count = int(profile["confirmation_replicate_count"])
    expected_runs = int(profile.get("confirmation_runs", layout_count * replicate_count))
    if layout_count * replicate_count != expected_runs:
        raise ValueError(
            "confirmation_layout_count * confirmation_replicate_count muss "
            "confirmation_runs entsprechen."
        )
    seeds = tuple(range(layout_count * replicate_count))
    online_rows = _run_confirmation_online(
        phase_dir,
        config,
        seeds,
        workers,
        export_layout_frames=export_layout_frames,
    )
    summary_rows = _rank_confirmation_online(online_rows)
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
            "layout_count": layout_count,
            "replicate_count": replicate_count,
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
    schedule = _tuning_seed_schedule(config, seed)
    return _train_layout(
        config,
        seed,
        include_test=False,
        layout_spec=config.get("layout_spec")
        or random_layout_spec(
            tuple(int(value) for value in config["hidden_sizes"]),
            schedule.layout_seed,
        ),
        data_split_seed=schedule.data_split_seed,
        weight_seed=schedule.retraining_weight_seed,
        batch_seed=schedule.retraining_batch_seed,
    )


def _train_layout(
    config: dict[str, Any],
    seed: int,
    *,
    include_test: bool,
    layout_spec: str | None,
    data_split_seed: int | None = None,
    weight_seed: int | None = None,
    batch_seed: int | None = None,
) -> dict[str, Any]:
    benchmark = str(config["benchmark"])
    hidden_sizes = tuple(int(value) for value in config["hidden_sizes"])
    effective_layout_spec = layout_spec or random_layout_spec(hidden_sizes, seed)
    resolved_data_split_seed = seed if data_split_seed is None else data_split_seed
    resolved_weight_seed = seed if weight_seed is None else weight_seed
    resolved_batch_seed = seed if batch_seed is None else batch_seed
    dataset = load_benchmark(
        DatasetConfig(name=benchmark, random_state=resolved_data_split_seed)
    )
    model = ModularMLP(
        input_size=dataset.input_size,
        hidden_sizes=hidden_sizes,
        output_size=dataset.model_output_size,
        layout=parse_layout_spec(effective_layout_spec, hidden_sizes),
        num_classes=dataset.output_size,
        weight_scale=float(config["weight_scale"]),
        random_state=resolved_weight_seed,
    )
    training_config = TrainingConfig(
        epochs=int(config["epochs"]),
        learning_rate=float(config["learning_rate"]),
        batch_size=int(config["batch_size"]),
        random_state=resolved_batch_seed,
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
    schedule = _tuning_seed_schedule(config, seed)
    dataset = load_benchmark(
        DatasetConfig(name=benchmark, random_state=schedule.data_split_seed)
    )
    start_layout = parse_layout_spec(
        random_layout_spec(hidden_sizes, schedule.layout_seed),
        hidden_sizes,
    )
    model = ModularMLP(
        input_size=dataset.input_size,
        hidden_sizes=hidden_sizes,
        output_size=dataset.model_output_size,
        layout=start_layout,
        num_classes=dataset.output_size,
        weight_scale=float(config["weight_scale"]),
        random_state=schedule.online_weight_seed,
    )
    rng = np.random.default_rng(schedule.sa_proposal_seed)
    cursor = MiniBatchCursor(
        train_size=dataset.train_size,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        rng=np.random.default_rng(schedule.online_batch_seed),
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
    result, _snapshot = _run_sa(
        config,
        seed,
        include_test=False,
        schedule=_tuning_seed_schedule(config, seed),
    )
    return result


def _online_budget_probe_worker(config: dict[str, Any], seed: int) -> dict[str, Any]:
    result, _snapshot = _run_sa(
        config,
        seed,
        include_test=False,
        schedule=_tuning_seed_schedule(config, seed),
    )
    return result


def _sa_refine_worker(config: dict[str, Any], seed: int) -> dict[str, Any]:
    schedule = _tuning_seed_schedule(config, seed)
    result, snapshot = _run_sa(config, seed, include_test=False, schedule=schedule)
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
        data_split_seed=schedule.data_split_seed,
        weight_seed=schedule.retraining_weight_seed,
        batch_seed=schedule.retraining_batch_seed,
    )
    end = _train_layout(
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
        layout_spec=snapshot.current_evaluation.layout.to_compact_spec(),  # type: ignore[union-attr]
        data_split_seed=schedule.data_split_seed,
        weight_seed=schedule.retraining_weight_seed,
        batch_seed=schedule.retraining_batch_seed,
    )
    return {
        **result,
        "same_random_start_retrained_val_loss": float(start["val_loss"]),
        "end_layout_from_sa_retrained_val_loss": float(end["val_loss"]),
        "paired_val_loss_improvement": float(start["val_loss"]) - float(end["val_loss"]),
        "paired_win": float(end["val_loss"]) < float(start["val_loss"]),
    }


def _confirmation_online_worker(
    config: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], OnlineAnnealingSnapshot, dict[str, Any]]:
    replicate_count = int(config.get("confirmation_replicate_count", 3))
    layout_index, replicate_index = divmod(seed, replicate_count)
    schedule = build_seed_schedule(
        str(config["benchmark"]),
        layout_index,
        replicate_index,
    )
    result, snapshot = _run_sa(config, seed, include_test=True, schedule=schedule)
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
        ("end_layout_from_sa_retrained", snapshot.current_evaluation.layout.to_compact_spec()),  # type: ignore[union-attr]
    ):
        comparisons.append(
            {
                **_train_layout(
                    base,
                    seed,
                    include_test=True,
                    layout_spec=layout_spec,
                    data_split_seed=schedule.data_split_seed,
                    weight_seed=schedule.retraining_weight_seed,
                    batch_seed=schedule.retraining_batch_seed,
                ),
                "label": label,
            }
        )
    inherited_metrics = _evaluate_inherited_model(
        snapshot.best_evaluation,
        config,
        schedule.data_split_seed,
    )
    comparisons.append(
        {
            "label": "best_online_delta_value",
            "evaluation_type": "inherited",
            "layout_spec": snapshot.best_evaluation.layout.to_compact_spec(),
            **inherited_metrics,
        }
    )
    end_comparison = next(item for item in comparisons if item["label"] == "end_layout_from_sa_retrained")
    start_comparison = next(item for item in comparisons if item["label"] == "same_random_start_retrained")
    result.update(
        {
            "layout_index": layout_index,
            "replicate_index": replicate_index,
            "layout_seed": schedule.layout_seed,
            "data_split_seed": schedule.data_split_seed,
            "online_weight_seed": schedule.online_weight_seed,
            "online_batch_seed": schedule.online_batch_seed,
            "sa_proposal_seed": schedule.sa_proposal_seed,
            "sa_acceptance_seed": schedule.sa_acceptance_seed,
            "retraining_weight_seed": schedule.retraining_weight_seed,
            "retraining_batch_seed": schedule.retraining_batch_seed,
            "same_random_start_retrained_val_loss": float(start_comparison["val_loss"]),
            "end_layout_from_sa_retrained_val_loss": float(end_comparison["val_loss"]),
            "end_layout_from_sa_retrained_test_loss": float(end_comparison["test_loss"]),
            "end_layout_from_sa_retrained_test_accuracy": float(end_comparison["test_accuracy"]),
            "paired_val_loss_improvement": float(start_comparison["val_loss"]) - float(end_comparison["val_loss"]),
            "paired_win": float(end_comparison["val_loss"]) < float(start_comparison["val_loss"]),
        }
    )
    payload = {
        "kind": "experiment_suite_online_delta_run",
        "schema_version": 2,
        "experiment_id": "tuning_confirmation_online_delta",
        "benchmark": config["benchmark"],
        "run_index": seed,
        "run_id": schedule.run_id,
        "layout_index": layout_index,
        "replicate_index": replicate_index,
        "seeds": schedule.to_dict(),
        "seed": seed,
        "training_seed": schedule.retraining_weight_seed,
        "layout_seed": schedule.layout_seed,
        "learning_rate": config["retrain_learning_rate"],
        "online_learning_rate": config["online_learning_rate"],
        "online_batch_size": config["online_batch_size"],
        "start_layout": snapshot.start_evaluation.layout.to_compact_spec(),  # type: ignore[union-attr]
        "diagnostic_best_layout": snapshot.best_evaluation.layout.to_compact_spec(),  # type: ignore[union-attr]
        "end_layout": snapshot.current_evaluation.layout.to_compact_spec(),  # type: ignore[union-attr]
        "accepted_steps": snapshot.accepted_steps,
        "trained_batch_updates": snapshot.trained_batch_updates,
        "effective_online_epochs": snapshot.effective_online_epochs,
        "acceptance_rate": snapshot.acceptance_rate,
        "online_history": _snapshot_history(snapshot),
        "final_comparisons": comparisons,
    }
    return result, snapshot, payload


def _evaluate_inherited_model(
    evaluation,
    config: dict[str, Any],
    seed: int,
) -> dict[str, float]:
    """Evaluate one final inherited model once, including the held-out test split."""

    dataset = load_benchmark(DatasetConfig(name=str(config["benchmark"]), random_state=seed))
    model = evaluation.trained_model
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
    schedule: SeedSchedule | None = None,
) -> tuple[dict[str, Any], OnlineAnnealingSnapshot]:
    hidden_sizes = tuple(int(value) for value in config["hidden_sizes"])
    data_split_seed = seed if schedule is None else schedule.data_split_seed
    layout_seed = seed if schedule is None else schedule.layout_seed
    online_weight_seed = seed if schedule is None else schedule.online_weight_seed
    online_batch_seed = seed if schedule is None else schedule.online_batch_seed
    session = create_online_session(
        OnlineAnnealingRequest(
            dataset_config=DatasetConfig(
                name=str(config["benchmark"]),
                random_state=data_split_seed,
            ),
            hidden_sizes=hidden_sizes,
            layout_spec=random_layout_spec(hidden_sizes, layout_seed),
            training_config=TrainingConfig(
                epochs=int(config["epochs"]),
                learning_rate=float(config["online_learning_rate"]),
                batch_size=int(config["online_batch_size"]),
                random_state=online_batch_seed,
            ),
            annealing_config=AnnealingConfig(
                start_temperature=float(config["start_temperature"]),
                cooling_schedule=str(
                    config.get("cooling_schedule", DEFAULT_ANNEALING_COOLING_SCHEDULE)
                ),
                cooling_parameter=float(config["cooling_parameter"]),
                iterations_per_temperature=int(config["iterations_per_temperature"]),
                max_steps=int(config["max_steps"]),
                min_temperature=float(config["min_temperature"]),
                neighborhood_operations=("set_neuron",),
            ),
            weight_scale=float(config["weight_scale"]),
            random_state=online_weight_seed,
            weight_seed=online_weight_seed,
            batch_seed=online_batch_seed,
            proposal_seed=None if schedule is None else schedule.sa_proposal_seed,
            acceptance_seed=None if schedule is None else schedule.sa_acceptance_seed,
            target_online_epochs=(
                float(config["target_online_epochs"])
                if "target_online_epochs" in config
                else None
            ),
            stop_at_target_online_epochs=bool(
                config.get("stop_at_target_online_epochs", False)
            ),
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
            "diagnostic_best_observed_val_loss": best_val,
            "end_val_loss": end_val,
            "end_progress_improvement": start_val - end_val,
            "step_count": len(snapshot.history),
            "trained_batch_updates": snapshot.trained_batch_updates,
            "effective_online_epochs": snapshot.effective_online_epochs,
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
                "post_training_batch_loss": float(start.objective_value),
                "trained_batch_updates_after_step": 0,
                "effective_online_epochs_after_step": 0.0,
                "diagnostics_refreshed": True,
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
            "post_training_batch_loss": float(step.post_training_batch_loss),
            "trained_batch_updates_after_step": int(step.trained_batch_updates_after_step),
            "effective_online_epochs_after_step": float(step.effective_online_epochs_after_step),
            "diagnostics_refreshed": bool(step.diagnostics_refreshed),
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


def _rank_online_budget_probe(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    ranking = _rank_rows(
        rows,
        metric_names=(
            "end_val_loss",
            "acceptance_rate",
            "effective_online_epochs",
            "step_count",
        ),
        sort_key=lambda row: float(json.loads(row["config_json"])["target_online_epochs"]),
    )
    for row in ranking:
        row["target_online_epochs"] = float(json.loads(row["config_json"])["target_online_epochs"])
    return sorted(ranking, key=lambda row: float(row["target_online_epochs"]))


def _select_online_budget(ranking: list[dict[str, Any]]) -> float | None:
    """Select the first checkpoint followed by two sub-one-percent improvements."""

    if len(ranking) < 3:
        return None
    for index in range(len(ranking) - 2):
        current = ranking[index]
        next_row = ranking[index + 1]
        after_next = ranking[index + 2]
        target = float(current["target_online_epochs"])
        if target < 25.0:
            continue
        current_loss = float(current["mean_end_val_loss"])
        next_loss = float(next_row["mean_end_val_loss"])
        after_next_loss = float(after_next["mean_end_val_loss"])
        first_improvement = (current_loss - next_loss) / max(abs(current_loss), 1e-12)
        second_improvement = (next_loss - after_next_loss) / max(abs(next_loss), 1e-12)
        if first_improvement < 0.01 and second_improvement < 0.01:
            return target
    return None


def _rank_sa_screen(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    ranking = _rank_rows(
        rows,
        metric_names=(
            "acceptance_rate",
            "end_val_loss",
            "end_progress_improvement",
            "effective_online_epochs",
            "finite_metrics",
        ),
        sort_key=lambda row: (
            float(row["mean_end_val_loss"]),
            -float(row["mean_end_progress_improvement"]),
        ),
    )
    for row in ranking:
        acceptance = float(row["mean_acceptance_rate"])
        progress = float(row["mean_end_progress_improvement"])
        target_online_epochs = float(json.loads(row["config_json"]).get("target_online_epochs", 0.0))
        trained_enough = float(row["mean_effective_online_epochs"]) >= target_online_epochs
        finite = float(row["mean_finite_metrics"]) == 1.0
        row["healthy"] = 0.15 <= acceptance <= 0.80 and progress > 0.0 and trained_enough and finite
    return sorted(
        ranking,
        key=lambda row: (
            not bool(row["healthy"]),
            float(row["mean_end_val_loss"]),
            -float(row["mean_end_progress_improvement"]),
        ),
    )


def _rank_sa_refine(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    ranking = _rank_rows(
        rows,
        metric_names=(
            "acceptance_rate",
            "end_layout_from_sa_retrained_val_loss",
            "paired_val_loss_improvement",
            "paired_win",
        ),
        sort_key=lambda row: (
            float(row["mean_end_layout_from_sa_retrained_val_loss"]),
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
            "end_layout_from_sa_retrained_val_loss",
            "end_layout_from_sa_retrained_test_loss",
            "end_layout_from_sa_retrained_test_accuracy",
            "paired_val_loss_improvement",
            "paired_win",
        ),
        sort_key=lambda row: float(row["mean_end_layout_from_sa_retrained_val_loss"]),
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
                "online_budget_json": json.dumps(payload.get("online_budget", {}), sort_keys=True),
                "sa_json": json.dumps(payload.get("sa", {}), sort_keys=True),
                "confirmation_json": json.dumps(payload.get("confirmation", {}), sort_keys=True),
            }
        )
    _write_csv(path, rows)


def _load_profile(profile_name: str, *, smoke: bool) -> dict[str, Any]:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def resolve(name: str, trail: tuple[str, ...] = ()) -> dict[str, Any]:
        if name in trail:
            raise ValueError(f"Zyklische Tuning-Profil-Vererbung: {' -> '.join((*trail, name))}")
        try:
            resolved = dict(payload["profiles"][name])
        except KeyError as exc:
            raise ValueError(f"Unbekanntes Tuning-Profil '{name}'.") from exc
        inherited = resolved.pop("inherits", None)
        if inherited is None:
            return resolved
        return {**resolve(str(inherited), (*trail, name)), **resolved}

    profile = resolve(profile_name)
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
            "online_budget_probe_runs": 1,
            "online_budget_probe_epochs": [0.05, 0.1, 0.2],
            "online_budget_probe_extended_epochs": [],
            "online_budget_probe_batch_size": 32,
            "sa_screen_runs": 1,
            "sa_refine_runs": 1,
            "sa_refine_top": 1,
            "confirmation_runs": 1,
            "confirmation_layout_count": 1,
            "confirmation_replicate_count": 1,
            "target_worse_acceptance_rates": [0.4],
            "batch_sizes": [32],
            "weight_scales": [1.0],
            "cooling_parameters": [0.95],
            "iterations_per_temperature": [1],
            "online_learning_rate_factors": [1.0],
            "online_batch_sizes": [32],
        }
    )
    if "cooling_end_ratios" in profile:
        profile["cooling_end_ratios"] = [profile["cooling_end_ratios"][0]]
    smoke_sa_candidates = len(profile.get("cooling_schedules", ())) or 1
    profile["benchmarks"] = {
        benchmark: {
            **values,
            "learning_rates": [values["learning_rates"][0]],
            "epochs": [1],
            "max_steps": [2],
            "sa_screen_candidates": smoke_sa_candidates,
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


def _proposal_safety_limit(benchmark: str, batch_size: int, target_online_epochs: float) -> int:
    """Allow enough proposals to reach the training target at ten-percent acceptance."""

    dataset = load_benchmark(DatasetConfig(name=benchmark, random_state=0))
    required_updates = math.ceil(target_online_epochs * dataset.train_size / batch_size)
    return max(1, math.ceil(required_updates / 0.10))


def _tuning_seed_schedule(config: dict[str, Any], seed: int) -> SeedSchedule:
    """Map a tuning replicate to independent, paired random streams."""

    return build_seed_schedule(str(config["benchmark"]), seed, 0)


def _cooling_candidates(
    profile: dict[str, Any],
    *,
    start_temperature: float,
    iterations_per_temperature: int,
    max_steps: int,
) -> list[dict[str, float | str]]:
    """Build comparable cooling candidates with a shared target end ratio."""

    end_ratios = profile.get("cooling_end_ratios")
    if not isinstance(end_ratios, list):
        return [
            {
                "cooling_schedule": DEFAULT_ANNEALING_COOLING_SCHEDULE,
                "cooling_parameter": float(parameter),
            }
            for parameter in profile["cooling_parameters"]
        ]

    schedules = tuple(str(value) for value in profile.get("cooling_schedules", ()))
    if not schedules:
        raise ValueError("cooling_schedules darf fuer normalisiertes Cooling nicht leer sein.")
    unsupported = sorted(set(schedules) - set(SUPPORTED_COOLING_SCHEDULES))
    if unsupported:
        raise ValueError(f"Nicht unterstuetzte Cooling-Schedules: {', '.join(unsupported)}")

    cooling_step_index = max(1, (int(max_steps) - 1) // int(iterations_per_temperature))
    candidates: list[dict[str, float | str]] = []
    for schedule, raw_ratio in itertools.product(schedules, end_ratios):
        ratio = float(raw_ratio)
        if not 0.0 < ratio < 1.0:
            raise ValueError("cooling_end_ratios muessen zwischen 0 und 1 liegen.")
        if schedule == "geometric":
            parameter = ratio ** (1.0 / cooling_step_index)
        elif schedule == "linear":
            parameter = start_temperature * (1.0 - ratio) / cooling_step_index
        else:
            parameter = (1.0 / ratio - 1.0) / math.log1p(cooling_step_index)
        candidates.append(
            {
                "cooling_schedule": schedule,
                "cooling_parameter": float(parameter),
                "cooling_target_end_ratio": ratio,
            }
        )
    return candidates


def _stratified_sample_configs(
    configs: list[dict[str, Any]],
    sample_count: int,
    *,
    group_key: str,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Sample configurations while guaranteeing representation of each group."""

    if sample_count >= len(configs):
        return list(configs)
    groups: dict[str, list[dict[str, Any]]] = {}
    for config in configs:
        groups.setdefault(str(config.get(group_key, "")), []).append(config)
    if sample_count < len(groups):
        raise ValueError(
            f"sa_screen_candidates muss mindestens {len(groups)} sein, "
            f"um alle {group_key}-Werte abzudecken."
        )

    selected: list[dict[str, Any]] = []
    group_items = sorted(groups.items())
    base_count, remainder = divmod(sample_count, len(group_items))
    for group_index, (_group, group_configs) in enumerate(group_items):
        count = base_count + (1 if group_index < remainder else 0)
        indices = rng.choice(len(group_configs), size=count, replace=False)
        selected.extend(group_configs[int(index)] for index in indices)
    rng.shuffle(selected)
    return selected


def _config_sha256(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
