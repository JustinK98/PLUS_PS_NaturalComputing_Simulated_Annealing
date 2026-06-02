"""Layout visualizations for Online-Delta experiment artifacts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
from matplotlib import patches
from matplotlib import pyplot as plt

from activations import (
    ACTIVATION_COLORS,
    ACTIVATION_SHORT_NAMES,
    ActivationLayout,
    LayoutChange,
    diff_layouts,
)
from configs import SUPPORTED_ACTIVATIONS
from services.online_annealing_training_service import (
    OnlineAnnealingSnapshot,
    OnlineAnnealingStep,
)


ACCEPTED_CHANGES_FIELDS = (
    "frame_index",
    "step_index",
    "layer_index",
    "neuron_index",
    "before_activation",
    "after_activation",
    "delta",
    "temperature",
    "validation_loss_after_update",
    "neighbor_label",
)


@dataclass(frozen=True)
class AcceptedChangeRecord:
    """Serializable accepted layout change for timelines and frame metadata."""

    frame_index: int
    step_index: int
    layer_index: int
    neuron_index: int
    before_activation: str
    after_activation: str
    delta: float
    temperature: float
    validation_loss_after_update: float
    neighbor_label: str


def write_online_delta_layout_artifacts(
    snapshot: OnlineAnnealingSnapshot,
    *,
    plots_dir: Path,
    frames_root: Path,
    run_id: str,
    export_frames: bool = False,
) -> tuple[Path, ...]:
    """Write per-run layout plots and optionally export regenerable frame files."""

    if snapshot.start_evaluation is None or snapshot.current_evaluation is None:
        return ()

    plots_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[Path] = []
    summary_path = plots_dir / f"{run_id}_layout_summary.png"
    render_layout_summary(
        summary_path,
        (
            ("Startlayout", snapshot.start_evaluation.layout, snapshot.start_evaluation.val_loss),
            (
                "Best layout",
                (
                    snapshot.best_evaluation.layout
                    if snapshot.best_evaluation is not None
                    else snapshot.current_evaluation.layout
                ),
                (
                    snapshot.best_evaluation.val_loss
                    if snapshot.best_evaluation is not None
                    else snapshot.current_evaluation.val_loss
                ),
            ),
            ("End layout", snapshot.current_evaluation.layout, snapshot.current_evaluation.val_loss),
        ),
        title=f"{run_id} layout summary",
    )
    artifacts.append(summary_path)

    records = _accepted_change_records(snapshot)
    timeline_path = plots_dir / f"{run_id}_accepted_timeline.png"
    render_accepted_timeline(
        timeline_path,
        snapshot.start_evaluation.layout.hidden_sizes,
        records,
        title=f"{run_id} accepted changes",
    )
    artifacts.append(timeline_path)

    if not export_frames:
        return tuple(artifacts)

    frame_dir = frames_root / run_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    csv_path = frame_dir / "accepted_changes.csv"
    _write_accepted_changes_csv(csv_path, records)
    artifacts.append(csv_path)
    artifacts.extend(
        export_layout_frames(
            frame_dir,
            snapshot,
            records,
        )
    )
    return tuple(artifacts)


def render_layout_summary(
    path: Path,
    layouts: Iterable[tuple[str, ActivationLayout, float]],
    *,
    title: str,
) -> Path:
    """Render several activation layouts as color-coded heatmaps."""

    entries = tuple(layouts)
    if not entries:
        raise ValueError("At least one layout is required.")
    fig, axes = plt.subplots(1, len(entries), figsize=(max(4.0, 3.6 * len(entries)), 3.8))
    if len(entries) == 1:
        axes = (axes,)
    fig.suptitle(title)
    for axis, (label, layout, val_loss) in zip(axes, entries, strict=True):
        _draw_layout(axis, layout, f"{label}\nval_loss={val_loss:.4f}")
    _add_activation_legend(fig)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def render_accepted_timeline(
    path: Path,
    hidden_sizes: tuple[int, ...],
    records: tuple[AcceptedChangeRecord, ...],
    *,
    title: str,
) -> Path:
    """Render accepted layout changes over SA steps."""

    fig, axis = plt.subplots(figsize=(9.5, 4.2))
    axis.set_title(title)
    axis.set_xlabel("SA step")
    axis.set_ylabel("Hidden neuron")
    if not records:
        axis.text(
            0.5,
            0.5,
            "No accepted layout changes",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    else:
        x_values = [record.step_index for record in records]
        y_values = [
            _flat_neuron_position(
                hidden_sizes,
                record.layer_index - 1,
                record.neuron_index,
            )
            for record in records
        ]
        colors = [
            ACTIVATION_COLORS.get(record.after_activation, "#94a3b8")
            for record in records
        ]
        axis.scatter(
            x_values,
            y_values,
            c=colors,
            edgecolors="#111827",
            linewidths=0.7,
            s=80,
            zorder=3,
        )
        if len(records) <= 24:
            for record, x_value, y_value in zip(records, x_values, y_values, strict=True):
                axis.annotate(
                    f"d={record.delta:.2g}\nv={record.validation_loss_after_update:.3g}",
                    (x_value, y_value),
                    textcoords="offset points",
                    xytext=(4, 5),
                    fontsize=7,
                )
        y_ticks = list(range(sum(hidden_sizes)))
        axis.set_yticks(y_ticks)
        axis.set_yticklabels(_flat_neuron_labels(hidden_sizes), fontsize=8)
        axis.grid(True, axis="x", alpha=0.25)
        axis.grid(True, axis="y", alpha=0.12)
        axis.set_xlim(min(x_values) - 1, max(x_values) + 1)
    _add_activation_legend(fig)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def export_layout_frames(
    frame_dir: Path,
    snapshot: OnlineAnnealingSnapshot,
    records: tuple[AcceptedChangeRecord, ...],
) -> tuple[Path, ...]:
    """Write a start frame and one frame per accepted Online-Delta step."""

    if snapshot.start_evaluation is None:
        return ()
    frame_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []
    start_path = frame_dir / "frame_000_start.png"
    _render_single_layout_frame(
        start_path,
        snapshot.start_evaluation.layout,
        title=f"Frame 000: start layout | val_loss={snapshot.start_evaluation.val_loss:.4f}",
        highlights=(),
    )
    artifacts.append(start_path)

    record_map: dict[int, list[AcceptedChangeRecord]] = {}
    for record in records:
        record_map.setdefault(record.step_index, []).append(record)

    frame_index = 0
    for step in snapshot.history:
        if not step.accepted:
            continue
        frame_index += 1
        changes = tuple(
            (record.layer_index - 1, record.neuron_index)
            for record in record_map.get(step.step_index, [])
        )
        frame_path = frame_dir / f"frame_{frame_index:03d}_step_{step.step_index:04d}.png"
        _render_single_layout_frame(
            frame_path,
            step.candidate_layout,
            title=(
                f"Frame {frame_index:03d}: step {step.step_index} | "
                f"{step.neighbor_label} | delta={step.delta:.4g} | "
                f"T={step.temperature:.4g} | val_loss={step.validation_loss_after_update:.4f}"
            ),
            highlights=changes,
        )
        artifacts.append(frame_path)
    return tuple(artifacts)


def _accepted_change_records(
    snapshot: OnlineAnnealingSnapshot,
) -> tuple[AcceptedChangeRecord, ...]:
    records: list[AcceptedChangeRecord] = []
    frame_index = 0
    for step in snapshot.history:
        if not step.accepted:
            continue
        frame_index += 1
        for change in _step_layout_changes(step):
            records.append(
                AcceptedChangeRecord(
                    frame_index=frame_index,
                    step_index=step.step_index,
                    layer_index=change.layer_index + 1,
                    neuron_index=change.neuron_index,
                    before_activation=change.before,
                    after_activation=change.after,
                    delta=float(step.delta),
                    temperature=float(step.temperature),
                    validation_loss_after_update=float(step.validation_loss_after_update),
                    neighbor_label=step.neighbor_label,
                )
            )
    return tuple(records)


def _step_layout_changes(step: OnlineAnnealingStep) -> tuple[LayoutChange, ...]:
    try:
        return diff_layouts(step.previous_layout, step.candidate_layout)
    except ValueError:
        return ()


def _write_accepted_changes_csv(
    path: Path,
    records: tuple[AcceptedChangeRecord, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ACCEPTED_CHANGES_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "frame_index": record.frame_index,
                    "step_index": record.step_index,
                    "layer_index": record.layer_index,
                    "neuron_index": record.neuron_index,
                    "before_activation": record.before_activation,
                    "after_activation": record.after_activation,
                    "delta": f"{record.delta:.12g}",
                    "temperature": f"{record.temperature:.12g}",
                    "validation_loss_after_update": f"{record.validation_loss_after_update:.12g}",
                    "neighbor_label": record.neighbor_label,
                }
            )


def _render_single_layout_frame(
    path: Path,
    layout: ActivationLayout,
    *,
    title: str,
    highlights: Iterable[tuple[int, int]],
) -> None:
    fig, axis = plt.subplots(figsize=(7.5, 3.8))
    _draw_layout(axis, layout, title, highlights=tuple(highlights))
    _add_activation_legend(fig)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _draw_layout(
    axis,
    layout: ActivationLayout,
    title: str,
    *,
    highlights: Iterable[tuple[int, int]] = (),
) -> None:
    highlight_set = set(highlights)
    max_neurons = max(len(layer) for layer in layout.layers)
    layer_count = len(layout.layers)
    for layer_index, layer in enumerate(layout.layers):
        for neuron_index, activation_name in enumerate(layer):
            color = ACTIVATION_COLORS.get(activation_name, "#94a3b8")
            axis.add_patch(
                patches.Rectangle(
                    (neuron_index, layer_index),
                    1.0,
                    1.0,
                    facecolor=color,
                    edgecolor="#f8fafc",
                    linewidth=1.0,
                )
            )
            if (layer_index, neuron_index) in highlight_set:
                axis.add_patch(
                    patches.Rectangle(
                        (neuron_index + 0.04, layer_index + 0.04),
                        0.92,
                        0.92,
                        fill=False,
                        edgecolor="#111827",
                        linewidth=2.2,
                    )
                )
            if max_neurons <= 18 and layer_count <= 4:
                axis.text(
                    neuron_index + 0.5,
                    layer_index + 0.5,
                    ACTIVATION_SHORT_NAMES.get(activation_name, activation_name[:3]).upper(),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#0f172a",
                    fontweight="bold",
                )
    axis.set_xlim(0, max_neurons)
    axis.set_ylim(layer_count, 0)
    axis.set_aspect("equal")
    axis.set_title(title, fontsize=10)
    axis.set_xticks([index + 0.5 for index in range(max_neurons)])
    axis.set_xticklabels([f"n{index}" for index in range(max_neurons)], fontsize=8)
    axis.set_yticks([index + 0.5 for index in range(layer_count)])
    axis.set_yticklabels([f"L{index + 1}" for index in range(layer_count)], fontsize=8)
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)


def _add_activation_legend(fig) -> None:
    handles = [
        patches.Patch(
            facecolor=ACTIVATION_COLORS.get(activation_name, "#94a3b8"),
            edgecolor="none",
            label=activation_name,
        )
        for activation_name in SUPPORTED_ACTIVATIONS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(handles), 6),
        frameon=False,
        fontsize=8,
    )


def _flat_neuron_position(
    hidden_sizes: tuple[int, ...],
    layer_index: int,
    neuron_index: int,
) -> int:
    return sum(hidden_sizes[:layer_index]) + neuron_index


def _flat_neuron_labels(hidden_sizes: tuple[int, ...]) -> list[str]:
    labels: list[str] = []
    for layer_index, layer_size in enumerate(hidden_sizes, start=1):
        labels.extend(f"L{layer_index}:n{neuron_index}" for neuron_index in range(layer_size))
    return labels
