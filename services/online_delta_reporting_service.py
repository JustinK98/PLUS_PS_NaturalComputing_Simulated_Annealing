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
    """Normalized Online-Delta run payload from suite or Experiment Builder JSON."""

    source_path: Path
    run_id: str
    experiment_id: str
    benchmark: str
    seed: int | None
    layout_seed: int | None
    learning_rate: float | None
    start_layout: str
    best_layout: str
    end_layout: str
    history: tuple[dict[str, object], ...]


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
    activation_counts_path = report_dir / "activation_counts_best.csv"
    similarity_path = report_dir / "layout_similarity_best.csv"

    _write_csv(steps_path, _step_rows(runs))
    _write_csv(runs_path, _run_rows(runs))
    _write_csv(activation_counts_path, _activation_count_rows(runs))
    _write_csv(similarity_path, _layout_similarity_rows(runs))
    artifacts.extend([steps_path, runs_path, activation_counts_path, similarity_path])

    artifacts.append(_plot_progress(report_dir / "online_delta_progress_mean.png", runs))
    artifacts.append(_plot_acceptance_rate(report_dir / "online_delta_acceptance_rate.png", runs))
    artifacts.append(_plot_delta_distribution(report_dir / "online_delta_delta_distribution.png", runs))
    artifacts.append(_plot_temperature(report_dir / "online_delta_temperature.png", runs))
    artifacts.append(_plot_activation_counts(report_dir / "activation_counts_best.png", runs))

    if warnings:
        warning_path = report_dir / "warnings.txt"
        warning_path.write_text("\n".join(warnings) + "\n", encoding="utf-8")
        artifacts.append(warning_path)

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
        history = _normalize_history(payload.get("online_history", []))
        if not history:
            warnings.append(f"{path}: Online-Delta suite run has no usable history.")
            return None
        return OnlineDeltaRun(
            source_path=path,
            run_id=str(payload.get("run_index", path.stem)),
            experiment_id=str(payload.get("experiment_id", "")),
            benchmark=str(payload.get("benchmark", "")),
            seed=_optional_int(payload.get("training_seed", payload.get("seed"))),
            layout_seed=_optional_int(payload.get("layout_seed")),
            learning_rate=_optional_float(payload.get("learning_rate")),
            start_layout=str(payload.get("start_layout", "")),
            best_layout=str(payload.get("best_layout", "")),
            end_layout=str(payload.get("end_layout", "")),
            history=tuple(history),
        )

    extra = payload.get("extra", {})
    if isinstance(extra, dict) and extra.get("sa_evaluation_mode") == "online_delta":
        run_definition = payload.get("run_definition", {})
        history = _normalize_history(extra.get("annealing_history", []))
        if not history:
            warnings.append(f"{path}: Experiment Builder Online-Delta run has no usable history.")
            return None
        if int(history[0].get("step_index", -1)) != 0:
            warnings.append(f"{path}: history starts after step 0; start metrics are unavailable.")
        return OnlineDeltaRun(
            source_path=path,
            run_id=str(run_definition.get("run_id", path.stem)) if isinstance(run_definition, dict) else path.stem,
            experiment_id=(
                str(run_definition.get("experiment_id", ""))
                if isinstance(run_definition, dict)
                else ""
            ),
            benchmark=(
                str(run_definition.get("benchmark", ""))
                if isinstance(run_definition, dict)
                else ""
            ),
            seed=(
                _optional_int(run_definition.get("seed"))
                if isinstance(run_definition, dict)
                else None
            ),
            layout_seed=None,
            learning_rate=_learning_rate_from_builder_payload(payload),
            start_layout=str(extra.get("start_layout_spec", "")),
            best_layout=str(extra.get("best_layout_spec", payload.get("layout_spec", ""))),
            end_layout=str(extra.get("end_layout_spec", "")),
            history=tuple(history),
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
                "learning_rate": "" if run.learning_rate is None else run.learning_rate,
                "start_layout": run.start_layout,
                "best_layout": run.best_layout,
                "end_layout": run.end_layout,
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


def _activation_count_rows(runs: list[OnlineDeltaRun]) -> list[dict[str, object]]:
    counts = {activation_name: 0 for activation_name in SUPPORTED_ACTIVATIONS}
    for run in runs:
        for activation_name in _flatten_layout(run.best_layout):
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
    flattened = [(run, _flatten_layout(run.best_layout)) for run in runs]
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


def _plot_progress(path: Path, runs: list[OnlineDeltaRun]) -> Path:
    grouped = _group_step_values(runs, "validation_loss_after_update")
    fig, axis = plt.subplots(figsize=(7, 4))
    if grouped:
        steps = sorted(grouped)
        means = np.asarray([np.mean(grouped[step]) for step in steps], dtype=np.float64)
        stds = np.asarray([np.std(grouped[step], ddof=0) for step in steps], dtype=np.float64)
        axis.plot(steps, means, label="mean validation loss")
        axis.fill_between(steps, means - stds, means + stds, alpha=0.2)
        axis.set_xlabel("SA step")
        axis.set_ylabel("Validation loss")
        axis.legend()
    else:
        axis.text(0.5, 0.5, "No data", ha="center", va="center")
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


def _plot_activation_counts(path: Path, runs: list[OnlineDeltaRun]) -> Path:
    rows = _activation_count_rows(runs)
    fig, axis = plt.subplots(figsize=(7, 4))
    labels = [str(row["activation"]) for row in rows]
    values = [int(row["count"]) for row in rows]
    if sum(values):
        axis.bar(labels, values, color="#536b8e")
        axis.set_ylabel("Count in best layouts")
        axis.tick_params(axis="x", rotation=30)
    else:
        axis.text(0.5, 0.5, "No data", ha="center", va="center")
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


def _learning_rate_from_builder_payload(payload: dict[str, Any]) -> float | None:
    run_definition = payload.get("run_definition", {})
    if isinstance(run_definition, dict):
        config_values = run_definition.get("config_values", {})
        if isinstance(config_values, dict):
            value = _optional_float(config_values.get("learning_rate"))
            if value is not None:
                return value
    extra = payload.get("extra", {})
    if isinstance(extra, dict):
        effective = extra.get("effective_parameters", {})
        if isinstance(effective, dict):
            return _optional_float(effective.get("learning_rate"))
    return None


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
