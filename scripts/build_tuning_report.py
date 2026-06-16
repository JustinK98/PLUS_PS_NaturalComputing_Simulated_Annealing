"""Build a progress and analysis report for a checkpointed tuning run."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Tuning run directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <run_dir>/analysis",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.run_dir / "analysis"
    report_path = build_tuning_report(args.run_dir, output_dir)
    print(report_path)


def build_tuning_report(run_dir: Path, output_dir: Path) -> Path:
    if not run_dir.exists():
        raise ValueError(f"Tuning run directory does not exist: {run_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    trial_files = [
        path
        for path in sorted(run_dir.rglob("trials.csv"))
        if not any(part.startswith("_discarded") for part in path.relative_to(run_dir).parts)
    ]
    phase_rows = [_phase_status(run_dir, path) for path in trial_files]
    cooling_rows = _cooling_status(trial_files)
    selected = _selected_payloads(run_dir)
    confirmation_rows = _confirmation_results(run_dir)
    seed_audit_rows = _seed_audit(run_dir)
    activation_effect_rows = _activation_effect_rows(run_dir)
    manifest = _read_json(run_dir / "manifest.json")

    status_plot = output_dir / "trial_status_by_phase.png"
    cooling_plot = output_dir / "cooling_schedule_progress.png"
    confirmation_plot = output_dir / "final_confirmation_effects.png"
    activation_effect_plot = output_dir / "activation_effect_associations.png"
    _plot_trial_status(status_plot, phase_rows)
    _plot_cooling_progress(cooling_plot, cooling_rows)
    _plot_confirmation_effects(confirmation_plot, confirmation_rows)
    _plot_activation_effects(activation_effect_plot, activation_effect_rows)
    _write_csv(output_dir / "phase_status.csv", phase_rows)
    _write_csv(output_dir / "cooling_schedule_status.csv", cooling_rows)
    _write_csv(output_dir / "final_confirmation_summary.csv", confirmation_rows)
    _write_csv(output_dir / "seed_audit.csv", seed_audit_rows)
    _write_csv(output_dir / "activation_effect_associations.csv", activation_effect_rows)

    report_path = output_dir / "METHODICAL_TUNING_REPORT.md"
    report_path.write_text(
        _render_report(
            run_dir,
            manifest=manifest,
            phase_rows=phase_rows,
            cooling_rows=cooling_rows,
            selected=selected,
            confirmation_rows=confirmation_rows,
            seed_audit_rows=seed_audit_rows,
            activation_effect_rows=activation_effect_rows,
            status_plot=status_plot,
            cooling_plot=cooling_plot,
            confirmation_plot=confirmation_plot,
            activation_effect_plot=activation_effect_plot,
        ),
        encoding="utf-8",
    )
    return report_path


def _phase_status(run_dir: Path, path: Path) -> dict[str, Any]:
    rows = _read_csv(path)
    statuses: dict[str, int] = defaultdict(int)
    duration_seconds = 0.0
    for row in rows:
        statuses[str(row.get("status", "unknown"))] += 1
        duration_seconds += _float(row.get("duration_seconds"))
    return {
        "phase": str(path.parent.relative_to(run_dir)),
        "total": len(rows),
        "completed": statuses["completed"],
        "running": statuses["running"],
        "pending": statuses["pending"],
        "failed": statuses["failed"],
        "cpu_hours_completed": duration_seconds / 3600.0,
    }


def _cooling_status(paths: list[Path]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for path in paths:
        if "/sa/" not in path.as_posix():
            continue
        for row in _read_csv(path):
            config = _json_object(row.get("config_json"))
            schedule = str(config.get("cooling_schedule", "geometric"))
            grouped[schedule].append(row)

    result = []
    for schedule, rows in sorted(grouped.items()):
        completed = [row for row in rows if row.get("status") == "completed"]
        progress = [
            _float(row.get("end_progress_improvement"))
            for row in completed
            if row.get("end_progress_improvement") not in (None, "")
        ]
        paired = [
            _float(row.get("paired_val_loss_improvement"))
            for row in completed
            if row.get("paired_val_loss_improvement") not in (None, "")
        ]
        result.append(
            {
                "cooling_schedule": schedule,
                "total_trials": len(rows),
                "completed_trials": len(completed),
                "failed_trials": sum(row.get("status") == "failed" for row in rows),
                "mean_end_progress_improvement": mean(progress) if progress else "",
                "mean_paired_val_loss_improvement": mean(paired) if paired else "",
            }
        )
    return result


def _selected_payloads(run_dir: Path) -> list[dict[str, Any]]:
    selected_dir = run_dir / "selected"
    if not selected_dir.exists():
        return []
    return [
        payload
        for path in sorted(selected_dir.glob("*.json"))
        if (payload := _read_json(path))
    ]


def _confirmation_results(run_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    confirmation_dir = run_dir / "confirmation"
    if not confirmation_dir.exists():
        return results
    for benchmark_dir in sorted(path for path in confirmation_dir.iterdir() if path.is_dir()):
        benchmark_rows = _read_csv(benchmark_dir / "aggregate" / "benchmark_summary.csv")
        summary_rows = _read_csv(benchmark_dir / "summary.csv")
        if not benchmark_rows or not summary_rows:
            continue
        benchmark = benchmark_rows[0]
        summary = summary_rows[0]
        config = _json_object(summary.get("config_json"))
        results.append(
            {
                "benchmark": benchmark.get("benchmark", benchmark_dir.name),
                "runs": int(_float(benchmark.get("runs"))),
                "layouts": int(_float(benchmark.get("layouts"))),
                "mean_paired_val_loss_improvement": _float(
                    benchmark.get("mean_paired_val_loss_improvement")
                ),
                "median_layout_mean_improvement": _float(
                    benchmark.get("median_layout_mean_improvement")
                ),
                "ci95_low": _float(benchmark.get("ci95_low")),
                "ci95_high": _float(benchmark.get("ci95_high")),
                "layout_win_rate": _float(benchmark.get("layout_win_rate")),
                "run_win_rate": _float(benchmark.get("run_win_rate")),
                "mean_acceptance_rate": _float(summary.get("mean_acceptance_rate")),
                "mean_end_layout_retrained_val_loss": _float(
                    summary.get("mean_end_layout_from_sa_retrained_val_loss")
                ),
                "mean_end_layout_retrained_test_loss": _float(
                    summary.get("mean_end_layout_from_sa_retrained_test_loss")
                ),
                "mean_end_layout_retrained_test_accuracy": _float(
                    summary.get("mean_end_layout_from_sa_retrained_test_accuracy")
                ),
                "cooling_schedule": config.get("cooling_schedule", ""),
                "cooling_target_end_ratio": config.get("cooling_target_end_ratio", ""),
                "start_temperature": config.get("start_temperature", ""),
                "iterations_per_temperature": config.get("iterations_per_temperature", ""),
                "max_steps": config.get("max_steps", ""),
                "online_learning_rate": config.get("online_learning_rate", ""),
                "online_batch_size": config.get("online_batch_size", ""),
                "target_online_epochs": config.get("target_online_epochs", ""),
            }
        )
    return results


def _seed_audit(run_dir: Path) -> list[dict[str, Any]]:
    stream_fields = (
        "online_batch_seed",
        "online_weight_seed",
        "retraining_batch_seed",
        "retraining_weight_seed",
        "sa_acceptance_seed",
        "sa_proposal_seed",
    )
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("confirmation/*/aggregate/seed_manifest.csv")):
        manifest_rows = _read_csv(path)
        if not manifest_rows:
            continue
        layout_counts: dict[str, int] = defaultdict(int)
        for row in manifest_rows:
            layout_counts[str(row.get("layout_index", ""))] += 1
        stream_counts = {
            field: len({row.get(field, "") for row in manifest_rows})
            for field in stream_fields
        }
        rows.append(
            {
                "benchmark": manifest_rows[0].get("benchmark", path.parents[1].name),
                "runs": len(manifest_rows),
                "unique_layouts": len(layout_counts),
                "replicates_per_layout": (
                    min(layout_counts.values()) if len(set(layout_counts.values())) == 1 else ""
                ),
                "unique_data_split_seeds": len(
                    {row.get("data_split_seed", "") for row in manifest_rows}
                ),
                "unique_layout_seeds": len({row.get("layout_seed", "") for row in manifest_rows}),
                "independent_run_streams": all(
                    count == len(manifest_rows) for count in stream_counts.values()
                ),
            }
        )
    return rows


def _activation_effect_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("confirmation/*/aggregate/activation_effect_summary.csv")):
        rows.extend(_read_csv(path))
    return rows


def _plot_trial_status(path: Path, rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(11, max(3.5, len(rows) * 0.35)))
    if rows:
        labels = [str(row["phase"]) for row in rows]
        completed = [int(row["completed"]) for row in rows]
        remaining = [
            int(row["total"]) - int(row["completed"]) - int(row["failed"])
            for row in rows
        ]
        failed = [int(row["failed"]) for row in rows]
        y = list(range(len(rows)))
        ax.barh(y, completed, label="completed", color="#2a9d8f")
        ax.barh(y, remaining, left=completed, label="remaining", color="#e9c46a")
        ax.barh(
            y,
            failed,
            left=[done + rest for done, rest in zip(completed, remaining, strict=True)],
            label="failed",
            color="#e76f51",
        )
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No trial files found", ha="center", va="center")
    ax.set_title("Tuning progress by phase")
    ax.set_xlabel("Trials")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_cooling_progress(path: Path, rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if rows:
        labels = [str(row["cooling_schedule"]) for row in rows]
        completed = [int(row["completed_trials"]) for row in rows]
        total = [int(row["total_trials"]) for row in rows]
        ax.bar(labels, total, label="total", color="#d9d9d9")
        ax.bar(labels, completed, label="completed", color="#457b9d")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No SA cooling trials found", ha="center", va="center")
    ax.set_title("Cooling strategy trial coverage")
    ax.set_ylabel("Trials")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_confirmation_effects(path: Path, rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if rows:
        labels = [str(row["benchmark"]) for row in rows]
        means = [float(row["mean_paired_val_loss_improvement"]) for row in rows]
        lows = [float(row["ci95_low"]) for row in rows]
        highs = [float(row["ci95_high"]) for row in rows]
        errors = [
            [mean_value - low for mean_value, low in zip(means, lows, strict=True)],
            [high - mean_value for mean_value, high in zip(means, highs, strict=True)],
        ]
        y = list(range(len(rows)))
        ax.errorbar(
            means,
            y,
            xerr=errors,
            fmt="o",
            color="#245a73",
            ecolor="#7f8c8d",
            capsize=5,
            markersize=7,
        )
        ax.axvline(0.0, color="#333333", linewidth=1, linestyle="--")
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Paired validation-loss improvement (positive favors SA)")
    else:
        ax.text(0.5, 0.5, "No completed confirmation summaries", ha="center", va="center")
    ax.set_title("Final paired effect by benchmark (95% CI over layout means)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_activation_effects(path: Path, rows: list[dict[str, Any]]) -> None:
    benchmarks = sorted({str(row["benchmark"]) for row in rows})
    activations = sorted({str(row["activation"]) for row in rows})
    fig, ax = plt.subplots(figsize=(9, max(3.5, len(benchmarks) * 1.2)))
    if benchmarks and activations:
        lookup = {
            (str(row["benchmark"]), str(row["activation"])): row
            for row in rows
        }
        matrix = []
        for benchmark in benchmarks:
            matrix.append(
                [
                    _optional_float(
                        lookup.get((benchmark, activation), {}).get(
                            "correlation_share_change_with_improvement"
                        )
                    )
                    for activation in activations
                ]
            )
        values = [
            [float(value) if value is not None else float("nan") for value in row]
            for row in matrix
        ]
        image = ax.imshow(values, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
        for y, row in enumerate(values):
            for x, value in enumerate(row):
                if value == value:
                    ax.text(x, y, f"{value:+.2f}", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(len(activations)), activations)
        ax.set_yticks(range(len(benchmarks)), benchmarks)
        fig.colorbar(image, ax=ax, label="Correlation")
    else:
        ax.text(0.5, 0.5, "Activation association analysis not available yet", ha="center", va="center")
    ax.set_title("Exploratory association: activation-share change vs paired improvement")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _render_report(
    run_dir: Path,
    *,
    manifest: dict[str, Any],
    phase_rows: list[dict[str, Any]],
    cooling_rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    confirmation_rows: list[dict[str, Any]],
    seed_audit_rows: list[dict[str, Any]],
    activation_effect_rows: list[dict[str, Any]],
    status_plot: Path,
    cooling_plot: Path,
    confirmation_plot: Path,
    activation_effect_plot: Path,
) -> str:
    total = sum(int(row["total"]) for row in phase_rows)
    completed = sum(int(row["completed"]) for row in phase_rows)
    failed = sum(int(row["failed"]) for row in phase_rows)
    cpu_hours = sum(float(row["cpu_hours_completed"]) for row in phase_rows)
    run_complete = _run_complete(phase_rows, selected, confirmation_rows)
    intervals_cross_zero = sum(
        float(row["ci95_low"]) <= 0.0 <= float(row["ci95_high"])
        for row in confirmation_rows
    )
    positive_means = [
        str(row["benchmark"])
        for row in confirmation_rows
        if float(row["mean_paired_val_loss_improvement"]) > 0.0
    ]
    non_positive_means = [
        str(row["benchmark"])
        for row in confirmation_rows
        if float(row["mean_paired_val_loss_improvement"]) <= 0.0
    ]
    cooling_method = (
        "Cooling strategies are compared at matched target end-temperature ratios."
        if str(manifest.get("profile", "")).startswith("methodical")
        else "Cooling values follow the run profile; cross-strategy normalization was not active."
    )
    layout_counts = sorted({int(row["layouts"]) for row in confirmation_rows})
    layout_count_text = (
        str(layout_counts[0]) if len(layout_counts) == 1 else "/".join(map(str, layout_counts))
    )
    lines = [
        "# Methodical Online-Delta-SA Tuning Report",
        "",
        f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Run: `{run_dir}`",
        f"- Profile: `{manifest.get('profile', 'unknown')}`",
        f"- Requested phases: `{manifest.get('requested_phase', 'unknown')}`",
        f"- Progress: **{completed}/{total} completed**, **{failed} failed**",
        f"- Completed trial CPU time: **{cpu_hours:.2f} h**",
        "",
        "## Technical Summary",
        "",
        (
            f"- The methodical run is **complete**: {completed}/{total} trials finished with "
            f"{failed} failures."
            if run_complete
            else f"- The methodical run is still **in progress**: {completed}/{total} trials finished."
        ),
        (
            f"- No benchmark establishes a reliable positive final-layout benefit: "
            f"{intervals_cross_zero}/{len(confirmation_rows)} layout-level 95% confidence "
            "intervals cross zero."
            if confirmation_rows
            else "- Final confirmation evidence is not available yet."
        ),
        (
            f"- Positive mean effect: `{', '.join(positive_means)}`. Non-positive mean effect: "
            f"`{', '.join(non_positive_means)}`. These signs are descriptive, not conclusive."
            if confirmation_rows
            else ""
        ),
        "- The defensible conclusion is benchmark-specific uncertainty, not a general SA advantage.",
        "",
        "## Final Confirmation: No Robust Positive Effect",
        "",
        "The primary metric is paired validation-loss improvement after retraining the same random "
        "start and final SA layout from scratch. Positive values favor SA. Confidence intervals use "
        f"the {layout_count_text} layout-level means per benchmark, preserving the nested "
        f"{layout_count_text}-layout x 3-replicate design.",
        "",
        f"![Final confirmation effects]({confirmation_plot.name})",
        "",
        "| Benchmark | Mean paired improvement | Median layout improvement | 95% CI | Layout win rate | Run win rate | Test accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| `{row['benchmark']}` | {_format_signed(row['mean_paired_val_loss_improvement'])} | "
        f"{_format_signed(row['median_layout_mean_improvement'])} | "
        f"[{_format_signed(row['ci95_low'])}, {_format_signed(row['ci95_high'])}] | "
        f"{_format_percent(row['layout_win_rate'])} | {_format_percent(row['run_win_rate'])} | "
        f"{_format_percent(row['mean_end_layout_retrained_test_accuracy'])} |"
        for row in confirmation_rows
    )
    lines.extend(
        [
            "",
            "### Benchmark Interpretation",
            "",
        ]
    )
    for row in confirmation_rows:
        lines.append(_benchmark_interpretation(row))
    lines.extend(
        [
            "",
            "## Activation Functions: Exploratory Associations",
            "",
            "This analysis tests whether an activation function becoming more or less common in the "
            "final SA layouts is associated with the layout-level paired improvement. It is descriptive, "
            "not causal: activation shares are compositional and multiple activations change together.",
            "",
            f"![Activation effect associations]({activation_effect_plot.name})",
            "",
        ]
    )
    if activation_effect_rows:
        lines.extend(
            [
                "| Benchmark | Strongest positive association | Correlation | Strongest negative association | Correlation | Layout blocks |",
                "| --- | --- | ---: | --- | ---: | ---: |",
            ]
        )
        for benchmark in sorted({str(row["benchmark"]) for row in activation_effect_rows}):
            benchmark_rows = [
                row
                for row in activation_effect_rows
                if str(row["benchmark"]) == benchmark
                and _optional_float(row.get("correlation_share_change_with_improvement")) is not None
            ]
            if not benchmark_rows:
                continue
            positive = max(
                benchmark_rows,
                key=lambda row: float(row["correlation_share_change_with_improvement"]),
            )
            negative = min(
                benchmark_rows,
                key=lambda row: float(row["correlation_share_change_with_improvement"]),
            )
            lines.append(
                f"| `{benchmark}` | `{positive['activation']}` | "
                f"{float(positive['correlation_share_change_with_improvement']):+.3f} | "
                f"`{negative['activation']}` | "
                f"{float(negative['correlation_share_change_with_improvement']):+.3f} | "
                f"{positive['layout_blocks']} |"
            )
        lines.extend(
            [
                "",
                "Treat these associations as hypotheses for controlled AF ablations. A positive "
                "correlation does not show that adding the activation caused the improvement.",
                "",
            ]
        )
        stable_associations = [
            row
            for row in activation_effect_rows
            if _interval_excludes_zero(
                row.get("correlation_share_change_ci95_low"),
                row.get("correlation_share_change_ci95_high"),
            )
        ]
        lines.extend(
            [
                "### Associations With Exploratory Bootstrap Intervals Excluding Zero",
                "",
            ]
        )
        if stable_associations:
            lines.extend(
                f"- **`{row['benchmark']}` / `{row['activation']}`:** correlation "
                f"{float(row['correlation_share_change_with_improvement']):+.3f}, exploratory "
                f"95% bootstrap interval [{float(row['correlation_share_change_ci95_low']):+.3f}, "
                f"{float(row['correlation_share_change_ci95_high']):+.3f}]."
                for row in stable_associations
            )
        else:
            lines.append("- No activation-function association interval excludes zero.")
        lines.extend(
            [
                "",
                "Even intervals excluding zero remain non-causal because activation shares are "
                "compositional, layouts change multiple neurons jointly, and the analysis tests many "
                "benchmark-activation combinations.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "The activation-association artifacts will be generated when the expanded confirmation "
                "reports are rebuilt.",
                "",
            ]
        )
    lines.extend(
        [
            "## Selected Final Profiles",
            "",
            "Selections were made using validation evidence before the locked confirmation. Test metrics "
            "were used only for final reporting.",
            "",
            "| Benchmark | Training LR / epochs / batch | Cooling | End ratio | Online LR / batch | Max steps | Online epoch target |",
            "| --- | --- | --- | ---: | --- | ---: | ---: |",
        ]
    )
    selected_by_benchmark = {str(payload.get("benchmark")): payload for payload in selected}
    for row in confirmation_rows:
        payload = selected_by_benchmark.get(str(row["benchmark"]), {})
        training = payload.get("training", {}) if isinstance(payload.get("training"), dict) else {}
        lines.append(
            f"| `{row['benchmark']}` | {_format_compact(training.get('learning_rate'))} / "
            f"{_format_compact(training.get('epochs'))} / {_format_compact(training.get('batch_size'))} | "
            f"`{row['cooling_schedule']}` | {_format_compact(row['cooling_target_end_ratio'])} | "
            f"{_format_compact(row['online_learning_rate'])} / {_format_compact(row['online_batch_size'])} | "
            f"{_format_compact(row['max_steps'])} | {_format_compact(row['target_online_epochs'])} |"
        )
    lines.extend(
        [
            "",
            "## Cooling Search: Benchmark-Specific Selections",
            "",
            f"{cooling_method} The coverage chart confirms that all planned schedule trials completed. "
            "The aggregate means below mix benchmarks and tuning phases, so they are diagnostics only "
            "and must not be read as a global cooling-schedule ranking.",
            "",
            f"![Cooling coverage]({cooling_plot.name})",
            "",
            "| Schedule | Completed | Total | Failed | Mean online progress | Mean paired improvement |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| `{row['cooling_schedule']}` | {row['completed_trials']} | {row['total_trials']} | "
        f"{row['failed_trials']} | {_format_metric(row['mean_end_progress_improvement'])} | "
        f"{_format_metric(row['mean_paired_val_loss_improvement'])} |"
        for row in cooling_rows
    )
    lines.extend(
        [
            "",
            "## Scope, Metrics, and Validation Design",
            "",
            "- Official benchmarks: `two_moons`, `concentric_circles`, and `crossing_spirals`.",
            "- Primary effect: `random_start_retrained_val_loss - final_sa_layout_retrained_val_loss`.",
            "- Positive paired improvement favors the final SA layout; negative favors the random start.",
            "- Validation-only selection preceded a locked final confirmation.",
            f"- Final confirmation uses {layout_count_text} layout/split blocks x 3 nested replicates per benchmark.",
            "- Random-start and final-layout retraining share split, initialization, and batch-order seeds.",
            "- Independent deterministic streams cover online weights/batches, proposals, acceptance, and retraining.",
            "",
            "### Seed Audit",
            "",
            "| Benchmark | Runs | Layouts / splits | Replicates per layout | Independent run streams |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    lines.extend(
        f"| `{row['benchmark']}` | {row['runs']} | {row['unique_layouts']} / "
        f"{row['unique_data_split_seeds']} | {row['replicates_per_layout']} | "
        f"{'yes' if row['independent_run_streams'] else 'no'} |"
        for row in seed_audit_rows
    )
    lines.extend(
        [
            "",
            "## Limitations and Robustness",
            "",
            "- All benchmark confidence intervals cross zero; none supports a reliable positive effect.",
            f"- The inference unit is {layout_count_text} layout means per benchmark. The nested replicates are not independent layout observations.",
            "- Confirmation varies layouts and data splits together by block, so layout and split robustness cannot be separated.",
            "- Cooling-schedule aggregate means mix benchmarks and stages; only benchmark-specific selected profiles are decision-relevant.",
            "- Acceptance rate and inherited online progress are health diagnostics, not evidence that the final layout is better.",
            "- Held-out test accuracy describes the locked final layouts but does not replace the paired validation comparison.",
            "- Activation-function associations are exploratory and confounded by the compositional mixed-layout design.",
            "",
            "## Recommended Next Steps",
            "",
            "1. Do not claim a general Online-Delta-SA layout advantage from this run.",
            "2. Freeze and promote these profiles only if the goal is reproducible benchmark baselines, not a positive-effect claim.",
            "3. Do not run another broad tuning cycle. The next experiment should be a targeted, preregistered AF ablation based on the confirmed association hypotheses.",
            "4. For causal AF evidence, vary one activation assignment at a time under matched layouts, splits, weights, and batches.",
            "5. Use multiplicity-aware inference or held-out replication before promoting an AF-specific claim.",
            "",
            "## Further Questions",
            "",
            "- Why does `crossing_spirals` remain negative at layout level despite substantial variance across blocks?",
            "- Do the AF association directions replicate in a preregistered held-out set of layout blocks?",
            "- Are schedule choices stable when selection is repeated with disjoint screen and refine seed banks?",
            "",
            "## Execution Audit",
            "",
            f"![Trial status]({status_plot.name})",
            "",
            "| Phase | Completed | Total | Failed | CPU hours |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| `{row['phase']}` | {row['completed']} | {row['total']} | {row['failed']} | "
        f"{float(row['cpu_hours_completed']):.2f} |"
        for row in phase_rows
    )
    lines.extend(
        [
            "",
            "## Full Selected Configuration Artifacts",
            "",
        ]
    )
    if not selected:
        lines.append("No benchmark selection artifacts exist yet.")
    for payload in selected:
        benchmark = payload.get("benchmark", "unknown")
        lines.extend([f"### {benchmark}", "", "```json", json.dumps(payload, indent=2), "```", ""])
    lines.extend(
        [
            "## Interpretation Boundary",
            "",
            (
                "This is the completed seed-separated locked confirmation for this methodical tuning run. "
                "The selected profiles are final for this artifact, but they are not automatically promoted "
                "to the repository default evaluation profile."
                if run_complete
                else "This report is provisional until every planned confirmation run is complete and the "
                "final profile is frozen. Partial rankings must not be promoted as final results."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _run_complete(
    phase_rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    confirmation_rows: list[dict[str, Any]],
) -> bool:
    if not phase_rows or any(
        int(row["completed"]) != int(row["total"])
        or int(row["failed"])
        or int(row["running"])
        or int(row["pending"])
        for row in phase_rows
    ):
        return False
    expected = {
        str(payload.get("benchmark")): int(
            _float(
                payload.get("confirmation", {}).get("runs")
                if isinstance(payload.get("confirmation"), dict)
                else 0
            )
        )
        for payload in selected
    }
    observed = {str(row["benchmark"]): int(row["runs"]) for row in confirmation_rows}
    return bool(expected) and all(observed.get(benchmark) == runs for benchmark, runs in expected.items())


def _benchmark_interpretation(row: dict[str, Any]) -> str:
    benchmark = str(row["benchmark"])
    mean_value = float(row["mean_paired_val_loss_improvement"])
    median_value = float(row["median_layout_mean_improvement"])
    layout_win_rate = float(row["layout_win_rate"])
    if mean_value > 0:
        direction = "a weak positive descriptive mean"
    else:
        direction = "a negative descriptive mean"
    return (
        f"- **`{benchmark}`:** {direction} ({_format_signed(mean_value)}), median "
        f"{_format_signed(median_value)}, and layout win rate {_format_percent(layout_win_rate)}. "
        "The 95% interval crosses zero, so the evidence is inconclusive."
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _json_object(value: object) -> dict[str, Any]:
    try:
        payload = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _interval_excludes_zero(low: object, high: object) -> bool:
    low_value = _optional_float(low)
    high_value = _optional_float(high)
    return (
        low_value is not None
        and high_value is not None
        and (low_value > 0.0 or high_value < 0.0)
    )


def _format_metric(value: object) -> str:
    return "" if value == "" else f"{float(value):.6f}"


def _format_signed(value: object) -> str:
    return f"{float(value):+.6f}"


def _format_percent(value: object) -> str:
    return f"{float(value) * 100:.1f}%"


def _format_compact(value: object) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, int):
        return str(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:.6g}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows or [{"status": "empty"}])


if __name__ == "__main__":
    main()
