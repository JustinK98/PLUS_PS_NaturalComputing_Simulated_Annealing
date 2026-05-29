"""Reporting assets for the current benchmark and annealing result state."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from benchmark_registry import OFFICIAL_BENCHMARKS, benchmark_spec
from configs import OUTPUT_DIR, SUPPORTED_ACTIVATIONS


@dataclass(frozen=True)
class ReportAssetBuildResult:
    """Paths and warnings produced by the report asset builder."""

    output_dir: Path
    files: tuple[Path, ...]
    warnings: tuple[str, ...]


def build_report_assets(
    *,
    output_dir: str | Path = OUTPUT_DIR / "report_assets",
    layout_grid_dir: str | Path = OUTPUT_DIR / "layout_grids",
    sa_dir: str | Path = OUTPUT_DIR / "sa_runs",
    neighborhood_summary_path: str | Path = OUTPUT_DIR / "neighborhood_grids" / "summary.csv",
) -> ReportAssetBuildResult:
    """Build consolidated CSV tables and plots from existing experiment artifacts."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    layout_rows = _collect_layout_grid_rows(Path(layout_grid_dir), warnings)
    sa_rows = _collect_sa_rows(Path(sa_dir), warnings)
    neighborhood_rows = _collect_neighborhood_rows(Path(neighborhood_summary_path), warnings)
    benchmark_rows = _build_benchmark_overview(layout_rows, sa_rows, neighborhood_rows)

    files: list[Path] = []
    files.append(_write_csv(output_path / "layout_grid_summary.csv", layout_rows))
    files.append(_write_csv(output_path / "sa_summary.csv", sa_rows))
    files.append(_write_csv(output_path / "neighborhood_summary.csv", neighborhood_rows))
    files.append(_write_csv(output_path / "benchmark_overview.csv", benchmark_rows))

    files.extend(_write_plots(output_path, Path(layout_grid_dir), layout_rows, sa_rows, neighborhood_rows))
    return ReportAssetBuildResult(
        output_dir=output_path,
        files=tuple(files),
        warnings=tuple(warnings),
    )


def _collect_layout_grid_rows(layout_grid_dir: Path, warnings: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for benchmark in OFFICIAL_BENCHMARKS:
        artifact = _select_layout_grid_artifact(layout_grid_dir, benchmark)
        if artifact is None:
            warnings.append(f"Missing layout-grid artifact for {benchmark}.")
            continue
        try:
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            ranking = payload.get("result", {}).get("ranking", [])
            if not ranking:
                warnings.append(f"Layout-grid artifact has no ranking: {artifact}")
                continue
            best = ranking[0]
            metrics = best.get("mean_metrics", {})
            best_run_metrics = payload.get("best_run", {}).get("metrics", {})
            training = payload.get("training", {})
            rows.append(
                {
                    "benchmark": benchmark,
                    "source_path": str(artifact),
                    "candidate_count": _int_or_empty(payload.get("candidate_count")),
                    "seeds": _join_values(payload.get("seeds", [])),
                    "epochs": _int_or_empty(training.get("epochs")),
                    "learning_rate": _float_or_empty(training.get("learning_rate")),
                    "batch_size": _int_or_empty(training.get("batch_size")),
                    "weight_scale": _float_or_empty(training.get("weight_scale")),
                    "best_label": best.get("label", ""),
                    "best_layout": best.get("layout_spec", ""),
                    "mean_val_loss": _float_or_empty(metrics.get("val_loss")),
                    "mean_val_accuracy": _float_or_empty(metrics.get("val_accuracy")),
                    "mean_test_loss": _float_or_empty(metrics.get("test_loss")),
                    "mean_test_accuracy": _float_or_empty(metrics.get("test_accuracy")),
                    "best_seed": _int_or_empty(payload.get("best_run", {}).get("seed")),
                    "best_run_val_loss": _float_or_empty(best_run_metrics.get("val_loss")),
                    "best_run_val_accuracy": _float_or_empty(best_run_metrics.get("val_accuracy")),
                    "best_run_test_accuracy": _float_or_empty(best_run_metrics.get("test_accuracy")),
                }
            )
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            warnings.append(f"Could not read layout-grid artifact {artifact}: {exc}")
    return rows


def _collect_sa_rows(sa_dir: Path, warnings: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for benchmark in OFFICIAL_BENCHMARKS:
        experiment_dir = sa_dir / f"sa_runs_{benchmark}"
        summary_path = experiment_dir / "summary.json"
        if not summary_path.exists():
            warnings.append(f"Missing SA summary for {benchmark}.")
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            ranking = summary.get("ranking", [])
            if not ranking:
                warnings.append(f"SA summary has no ranking: {summary_path}")
                continue
            top = ranking[0]
            run_metrics = _load_sa_run_metrics(experiment_dir / "runs", warnings)
            rows.append(
                {
                    "benchmark": benchmark,
                    "source_path": str(summary_path),
                    "number_of_seeds": _int_or_empty(summary.get("number_of_seeds")),
                    "annealing_score": _float_or_empty(top.get("ranking_score")),
                    "search_mean_val_loss": _float_or_empty(top.get("mean_val_loss")),
                    "search_mean_val_accuracy": _float_or_empty(top.get("mean_val_accuracy")),
                    "search_mean_test_accuracy": _float_or_empty(top.get("mean_test_accuracy")),
                    "final_mean_val_loss": _float_or_empty(run_metrics.get("final_val_loss")),
                    "final_mean_val_accuracy": _float_or_empty(run_metrics.get("final_val_accuracy")),
                    "final_mean_test_loss": _float_or_empty(run_metrics.get("final_test_loss")),
                    "final_mean_test_accuracy": _float_or_empty(run_metrics.get("final_test_accuracy")),
                    "mean_step_count": _float_or_empty(run_metrics.get("step_count")),
                    "mean_acceptance_rate": _float_or_empty(run_metrics.get("acceptance_rate")),
                    "best_seed": _int_or_empty(top.get("best_seed")),
                    "worst_seed": _int_or_empty(top.get("worst_seed")),
                }
            )
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            warnings.append(f"Could not read SA summary {summary_path}: {exc}")
    return rows


def _collect_neighborhood_rows(summary_path: Path, warnings: list[str]) -> list[dict[str, object]]:
    if not summary_path.exists():
        warnings.append(f"Missing neighborhood summary: {summary_path}")
        return []
    try:
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = []
            for row in reader:
                rows.append(
                    {
                        "benchmark": row.get("benchmark", ""),
                        "neighborhood": row.get("neighborhood", ""),
                        "experiment_id": row.get("experiment_id", ""),
                        "source_path": row.get("path", ""),
                        "ranking_score": _float_or_empty(row.get("ranking_score")),
                        "mean_val_loss": _float_or_empty(row.get("mean_val_loss")),
                        "mean_val_accuracy": _float_or_empty(row.get("mean_val_accuracy")),
                        "mean_test_accuracy": _float_or_empty(row.get("mean_test_accuracy")),
                        "best_seed": _int_or_empty(row.get("best_seed")),
                        "worst_seed": _int_or_empty(row.get("worst_seed")),
                    }
                )
            return rows
    except OSError as exc:
        warnings.append(f"Could not read neighborhood summary {summary_path}: {exc}")
        return []


def _build_benchmark_overview(
    layout_rows: list[dict[str, object]],
    sa_rows: list[dict[str, object]],
    neighborhood_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    layout_by_benchmark = {str(row["benchmark"]): row for row in layout_rows}
    sa_by_benchmark = {str(row["benchmark"]): row for row in sa_rows}
    top_neighborhood_by_benchmark = _top_neighborhood_rows(neighborhood_rows)

    rows: list[dict[str, object]] = []
    for benchmark in OFFICIAL_BENCHMARKS:
        spec = benchmark_spec(benchmark)
        grid = layout_by_benchmark.get(benchmark, {})
        sa = sa_by_benchmark.get(benchmark, {})
        neighborhood = top_neighborhood_by_benchmark.get(benchmark, {})
        rows.append(
            {
                "benchmark": benchmark,
                "topology": _format_topology(len(spec.feature_columns), spec.hidden_sizes, spec.model_output_size),
                "official_epochs": spec.epochs,
                "task_type": spec.task_type,
                "activation_set": ", ".join(SUPPORTED_ACTIVATIONS),
                "best_grid_label": grid.get("best_label", ""),
                "best_grid_layout": grid.get("best_layout", ""),
                "grid_mean_val_loss": grid.get("mean_val_loss", ""),
                "grid_mean_test_accuracy": grid.get("mean_test_accuracy", ""),
                "sa_annealing_score": sa.get("annealing_score", ""),
                "sa_final_mean_test_accuracy": sa.get("final_mean_test_accuracy", ""),
                "best_neighborhood": neighborhood.get("neighborhood", ""),
                "neighborhood_mean_test_accuracy": neighborhood.get("mean_test_accuracy", ""),
                "interpretation": _default_interpretation(benchmark),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _write_plots(
    output_path: Path,
    layout_grid_dir: Path,
    layout_rows: list[dict[str, object]],
    sa_rows: list[dict[str, object]],
    neighborhood_rows: list[dict[str, object]],
) -> list[Path]:
    files: list[Path] = []
    files.append(_plot_layout_grid_accuracy(output_path / "layout_grid_best_accuracy.png", layout_rows))
    files.append(_plot_sa_vs_grid_accuracy(output_path / "sa_vs_grid_accuracy.png", layout_rows, sa_rows))
    files.append(_plot_neighborhood_comparison(output_path / "neighborhood_comparison.png", neighborhood_rows))
    files.append(_plot_training_curves(output_path / "training_curves_best_layouts.png", layout_grid_dir))
    return files


def _plot_layout_grid_accuracy(path: Path, layout_rows: list[dict[str, object]]) -> Path:
    benchmarks = [str(row["benchmark"]) for row in layout_rows]
    val_values = [_as_float(row.get("mean_val_accuracy")) for row in layout_rows]
    test_values = [_as_float(row.get("mean_test_accuracy")) for row in layout_rows]
    _bar_plot(
        path,
        title="Best Layout-Grid Accuracy",
        benchmarks=benchmarks,
        series=(("Validation", val_values), ("Test", test_values)),
        ylabel="Accuracy",
    )
    return path


def _plot_sa_vs_grid_accuracy(
    path: Path,
    layout_rows: list[dict[str, object]],
    sa_rows: list[dict[str, object]],
) -> Path:
    grid_by_benchmark = {str(row["benchmark"]): row for row in layout_rows}
    sa_by_benchmark = {str(row["benchmark"]): row for row in sa_rows}
    benchmarks = [benchmark for benchmark in OFFICIAL_BENCHMARKS if benchmark in grid_by_benchmark or benchmark in sa_by_benchmark]
    grid_values = [_as_float(grid_by_benchmark.get(benchmark, {}).get("mean_test_accuracy")) for benchmark in benchmarks]
    sa_values = [_as_float(sa_by_benchmark.get(benchmark, {}).get("final_mean_test_accuracy")) for benchmark in benchmarks]
    _bar_plot(
        path,
        title="Final Test Accuracy: Grid Baseline vs SA",
        benchmarks=benchmarks,
        series=(("Layout Grid", grid_values), ("SA final", sa_values)),
        ylabel="Accuracy",
    )
    return path


def _plot_neighborhood_comparison(path: Path, neighborhood_rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, max(1, len(OFFICIAL_BENCHMARKS)), figsize=(14, 4), squeeze=False)
    for axis, benchmark in zip(axes[0], OFFICIAL_BENCHMARKS):
        rows = [row for row in neighborhood_rows if row.get("benchmark") == benchmark]
        labels = [str(row.get("neighborhood", "")) for row in rows]
        values = [_as_float(row.get("mean_test_accuracy")) for row in rows]
        if labels:
            axis.bar(range(len(labels)), values, color="#2f6f8f")
            axis.set_xticks(range(len(labels)))
            axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        axis.set_title(benchmark)
        axis.set_ylim(0.0, 1.05)
        axis.set_ylabel("Test Accuracy")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Neighborhood Comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_training_curves(path: Path, layout_grid_dir: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(OFFICIAL_BENCHMARKS), figsize=(14, 4), squeeze=False)
    for axis, benchmark in zip(axes[0], OFFICIAL_BENCHMARKS):
        artifact = _select_layout_grid_artifact(layout_grid_dir, benchmark)
        if artifact is None:
            axis.set_title(benchmark)
            axis.text(0.5, 0.5, "missing artifact", ha="center", va="center")
            continue
        try:
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            history = payload.get("best_run", {}).get("history", {})
            val_acc = history.get("val_acc", [])
            train_acc = history.get("train_acc", [])
            if train_acc:
                axis.plot(train_acc, label="train", color="#2f6f8f")
            if val_acc:
                axis.plot(val_acc, label="validation", color="#e28a2f")
            axis.set_ylim(0.0, 1.05)
            axis.set_title(benchmark)
            axis.grid(alpha=0.25)
        except (OSError, json.JSONDecodeError, TypeError):
            axis.set_title(benchmark)
            axis.text(0.5, 0.5, "unreadable artifact", ha="center", va="center")
    axes[0][0].set_ylabel("Accuracy")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle("Training Curves of Best Layout-Grid Runs")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _bar_plot(
    path: Path,
    *,
    title: str,
    benchmarks: list[str],
    series: tuple[tuple[str, list[float]], ...],
    ylabel: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9, 5))
    if not benchmarks:
        axis.text(0.5, 0.5, "No data available", ha="center", va="center")
    else:
        width = 0.8 / max(1, len(series))
        x_values = list(range(len(benchmarks)))
        for series_index, (label, values) in enumerate(series):
            offsets = [x - 0.4 + width / 2 + series_index * width for x in x_values]
            axis.bar(offsets, values, width=width, label=label)
        axis.set_xticks(x_values)
        axis.set_xticklabels(benchmarks, rotation=20, ha="right")
        axis.set_ylim(0.0, 1.05)
        axis.legend()
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _select_layout_grid_artifact(layout_grid_dir: Path, benchmark: str) -> Path | None:
    preferred = (
        layout_grid_dir / f"{benchmark}_layout_grid.json",
    )
    for path in preferred:
        if path.exists():
            return path
    matches = sorted(layout_grid_dir.glob(f"{benchmark}*.json"))
    return matches[0] if matches else None


def _load_sa_run_metrics(runs_dir: Path, warnings: list[str]) -> dict[str, float]:
    metric_values: dict[str, list[float]] = {}
    if not runs_dir.exists():
        warnings.append(f"Missing SA runs directory: {runs_dir}")
        return {}
    for path in sorted(runs_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            metrics = payload.get("metrics", {})
            for key, value in metrics.items():
                if isinstance(value, int | float):
                    metric_values.setdefault(key, []).append(float(value))
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            warnings.append(f"Could not read SA run {path}: {exc}")
    return {
        key: sum(values) / len(values)
        for key, values in metric_values.items()
        if values
    }


def _top_neighborhood_rows(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        benchmark = str(row.get("benchmark", ""))
        if not benchmark:
            continue
        current = result.get(benchmark)
        if current is None or _as_float(row.get("ranking_score")) < _as_float(current.get("ranking_score")):
            result[benchmark] = row
    return result


def _fieldnames(rows: list[dict[str, object]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames or ["status"]


def _format_topology(input_size: int, hidden_sizes: tuple[int, ...], output_size: int) -> str:
    return "-".join(str(value) for value in (input_size, *hidden_sizes, output_size))


def _default_interpretation(benchmark: str) -> str:
    if benchmark == "two_moons":
        return "Easy binary case; use it for quick smoke runs and first suite checks."
    if benchmark == "concentric_circles":
        return "Medium binary case; useful for validating deeper hidden layouts."
    if benchmark == "crossing_spirals":
        return "Hard binary case; report it with mean/std over seeds, not single runs."
    return ""


def _join_values(values: Iterable[object]) -> str:
    return ",".join(str(value) for value in values)


def _float_or_empty(value: object) -> float | str:
    if value is None or value == "":
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def _int_or_empty(value: object) -> int | str:
    if value is None or value == "":
        return ""
    try:
        return int(value)
    except (TypeError, ValueError):
        return ""


def _as_float(value: object) -> float:
    converted = _float_or_empty(value)
    return converted if isinstance(converted, float) else 0.0
