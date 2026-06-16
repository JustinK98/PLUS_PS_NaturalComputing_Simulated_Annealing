"""GUI-neutrale Zustands-, Aggregations- und Persistenzlogik fuer Multi-Run-Demos."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from activations import parse_layout_spec, random_layout_spec
from benchmarks import load_benchmark
from configs import DatasetConfig, OUTPUT_DIR
from model import ModularMLP
from services.layout_evaluation_service import (
    AggregatedLayoutEvaluation,
    LayoutEvaluationResult,
    LayoutEvaluationRun,
)
from services.online_annealing_training_service import (
    OnlineAnnealingSession,
    OnlineAnnealingSnapshot,
)
from services.seed_schedule_service import SeedSchedule, build_seed_schedule
from services.training_service import TrainingRunArtifacts
from trainer import TrainingResult


GUI_MULTI_RUN_SCHEMA_VERSION = 1
RUN_STATUSES = ("pending", "running", "baseline_done", "sa_done", "complete", "failed")
GUI_COMPARISON_MODES = ("blocked_layout", "robustness")


@dataclass
class GuiRunRecord:
    """Alle GUI-Artefakte eines unabhaengigen Runs."""

    run_index: int
    schedule: SeedSchedule
    start_layout_spec: str
    status: str = "pending"
    current_stage: str = "layout"
    manually_edited: bool = False
    error: str | None = None
    dataset: object | None = None
    current_model: ModularMLP | None = None
    training_artifacts: TrainingRunArtifacts | None = None
    annealing_session: OnlineAnnealingSession | None = None
    annealing_snapshot: OnlineAnnealingSnapshot | None = None
    layout_evaluation: LayoutEvaluationResult | None = None

    def invalidate(self, *, keep_layout: bool = True) -> None:
        """Entfernt abhaengige Ergebnisse, ohne andere Runs anzufassen."""

        self.status = "pending"
        self.current_stage = "layout"
        self.error = None
        self.dataset = None
        self.current_model = None
        self.training_artifacts = None
        self.annealing_session = None
        self.annealing_snapshot = None
        self.layout_evaluation = None
        if not keep_layout:
            self.manually_edited = False


@dataclass
class GuiMultiRunSession:
    """Persistierbare Multi-Run-Session der Qt-GUI."""

    benchmark: str
    profile_name: str
    master_seed: int
    hidden_sizes: tuple[int, ...]
    config_snapshot: dict[str, object]
    runs: list[GuiRunRecord]
    output_dir: Path
    created_at: str
    updated_at: str
    comparison_mode: str = "robustness"
    schema_version: int = GUI_MULTI_RUN_SCHEMA_VERSION


@dataclass(frozen=True)
class AggregateSeries:
    """Median und IQR einer ausgerichteten Kurvensammlung."""

    x: tuple[float, ...]
    median: tuple[float, ...]
    q25: tuple[float, ...]
    q75: tuple[float, ...]
    count: tuple[int, ...]


def build_gui_seed_schedule(
    benchmark: str,
    run_index: int,
    master_seed: int,
    comparison_mode: str = "robustness",
) -> SeedSchedule:
    """Erzeugt Seeds fuer einen blockierten oder unabhaengigen GUI-Vergleich."""

    schedule = build_seed_schedule(
        benchmark,
        layout_index=run_index,
        replicate_index=0,
        master_seed=master_seed,
    )
    # Run 0 behaelt den seit jeher sichtbaren GUI-Layout-Seed. Alle anderen
    # Streams bleiben davon getrennt; Multi-Run 0 und Single sind damit identisch.
    if run_index == 0:
        schedule = replace(schedule, layout_seed=master_seed)
    if comparison_mode == "robustness":
        return schedule
    if comparison_mode != "blocked_layout":
        raise ValueError(f"Unbekannter GUI-Vergleichsmodus: {comparison_mode}")

    shared = build_seed_schedule(
        benchmark,
        layout_index=0,
        replicate_index=0,
        master_seed=master_seed,
    )
    return replace(
        schedule,
        data_split_seed=shared.data_split_seed,
        online_weight_seed=shared.online_weight_seed,
        online_batch_seed=shared.online_batch_seed,
        sa_proposal_seed=shared.sa_proposal_seed,
        sa_acceptance_seed=shared.sa_acceptance_seed,
        retraining_weight_seed=shared.retraining_weight_seed,
        retraining_batch_seed=shared.retraining_batch_seed,
    )


def generate_unique_gui_runs(
    benchmark: str,
    hidden_sizes: tuple[int, ...],
    run_count: int,
    master_seed: int,
    comparison_mode: str = "robustness",
) -> list[GuiRunRecord]:
    """Erzeugt deterministische, untereinander eindeutige Random-Mixed-Layouts."""

    if not 1 <= run_count <= 10:
        raise ValueError("Die GUI unterstuetzt zwischen 1 und 10 Runs.")

    records: list[GuiRunRecord] = []
    seen: set[str] = set()
    for run_index in range(run_count):
        schedule = build_gui_seed_schedule(
            benchmark,
            run_index,
            master_seed,
            comparison_mode,
        )
        rng = np.random.default_rng(schedule.layout_seed)
        layout_spec = ""
        for _attempt in range(10_000):
            candidate = parse_layout_spec(random_layout_spec(hidden_sizes, rng), hidden_sizes).to_compact_spec()
            if candidate not in seen:
                layout_spec = candidate
                break
        if not layout_spec:
            raise ValueError("Es konnten nicht genug eindeutige Random-Mixed-Layouts erzeugt werden.")
        seen.add(layout_spec)
        records.append(
            GuiRunRecord(
                run_index=run_index,
                schedule=schedule,
                start_layout_spec=layout_spec,
            )
        )
    return records


def create_gui_multi_run_session(
    *,
    benchmark: str,
    profile_name: str,
    master_seed: int,
    hidden_sizes: tuple[int, ...],
    config_snapshot: dict[str, object],
    run_count: int,
    comparison_mode: str = "robustness",
    output_root: Path = OUTPUT_DIR / "gui_multi_runs",
) -> GuiMultiRunSession:
    if comparison_mode not in GUI_COMPARISON_MODES:
        raise ValueError(f"Unbekannter GUI-Vergleichsmodus: {comparison_mode}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root) / f"{timestamp}_{benchmark}_{comparison_mode}_{profile_name}"
    now = datetime.now().isoformat(timespec="seconds")
    return GuiMultiRunSession(
        benchmark=benchmark,
        profile_name=profile_name,
        master_seed=master_seed,
        hidden_sizes=hidden_sizes,
        config_snapshot=dict(config_snapshot),
        runs=generate_unique_gui_runs(
            benchmark,
            hidden_sizes,
            run_count,
            master_seed,
            comparison_mode,
        ),
        output_dir=output_dir,
        created_at=now,
        updated_at=now,
        comparison_mode=comparison_mode,
    )


def aggregate_histories(
    histories: list[dict[str, list[float]]],
    key: str,
) -> AggregateSeries | None:
    """Aggregiert unterschiedlich lange Epochenkurven ohne Tail-Fortschreibung."""

    values = [np.asarray(history.get(key, []), dtype=np.float64) for history in histories]
    values = [array for array in values if array.size]
    if not values:
        return None
    width = max(len(array) for array in values)
    matrix = np.full((len(values), width), np.nan, dtype=np.float64)
    for row, array in enumerate(values):
        matrix[row, : len(array)] = array
    return _aggregate_matrix(np.arange(width, dtype=np.float64), matrix)


def aggregate_sa_snapshots(
    snapshots: list[OnlineAnnealingSnapshot],
    key: str,
) -> AggregateSeries | None:
    """Aggregiert SA-Diagnosen nach Proposal-Schritt."""

    getters = {
        "batch_loss_before": lambda step: step.batch_loss_before,
        "candidate_loss_after": lambda step: step.candidate_loss_after,
        "delta": lambda step: step.delta,
        "temperature": lambda step: step.temperature,
        "validation_loss": lambda step: step.validation_loss_after_update,
        "best_validation_loss": lambda step: step.best_score_after_step,
    }
    if key not in getters:
        raise ValueError(f"Unbekannte SA-Serie: {key}")
    histories = [
        np.asarray([getters[key](step) for step in snapshot.history], dtype=np.float64)
        for snapshot in snapshots
        if snapshot.history
    ]
    if not histories:
        return None
    width = max(len(array) for array in histories)
    matrix = np.full((len(histories), width), np.nan, dtype=np.float64)
    for row, array in enumerate(histories):
        matrix[row, : len(array)] = array
    return _aggregate_matrix(np.arange(1, width + 1, dtype=np.float64), matrix)


def aggregate_validation_by_online_epochs(
    snapshots: list[OnlineAnnealingSnapshot],
    *,
    points: int = 120,
) -> AggregateSeries | None:
    """Interpoliert die Validierungsdiagnose auf gemeinsame effektive Online-Epochen."""

    curves: list[tuple[np.ndarray, np.ndarray]] = []
    for snapshot in snapshots:
        if not snapshot.history:
            continue
        x = np.asarray(
            [step.effective_online_epochs_after_step for step in snapshot.history],
            dtype=np.float64,
        )
        y = np.asarray(
            [step.validation_loss_after_update for step in snapshot.history],
            dtype=np.float64,
        )
        unique_x, indices = np.unique(x, return_index=True)
        if unique_x.size:
            curves.append((unique_x, y[indices]))
    if not curves:
        return None
    max_x = max(float(x[-1]) for x, _ in curves)
    grid = np.linspace(0.0, max_x, max(2, points))
    matrix = np.full((len(curves), len(grid)), np.nan, dtype=np.float64)
    for row, (x, y) in enumerate(curves):
        valid = grid <= x[-1]
        matrix[row, valid] = np.interp(grid[valid], x, y)
    return _aggregate_matrix(grid, matrix)


def paired_final_rows(runs: list[GuiRunRecord]) -> list[dict[str, object]]:
    """Liefert eine seed-sortierte, gepaarte Ergebniszeile pro fertigem Run."""

    rows: list[dict[str, object]] = []
    for record in sorted(runs, key=lambda item: item.schedule.retraining_weight_seed):
        result = record.layout_evaluation
        if result is None:
            continue
        start = next((run for run in result.runs if run.label == "random_start_layout"), None)
        end = next((run for run in result.runs if run.label == "end_sa_layout"), None)
        if start is None:
            continue
        # Wenn SA das Startlayout behaelt, dedupliziert der bestehende
        # Vergleichsservice den identischen Endkandidaten. Methodisch ist die
        # gepaarte Verbesserung dann exakt null.
        end = end or start
        inherited = next(
            (run for run in result.inherited_runs if run.label == "best_online_delta_value"),
            None,
        )
        snapshot = record.annealing_snapshot
        rows.append(
            {
                "run": record.run_index + 1,
                "seed": record.schedule.retraining_weight_seed,
                "start_val_loss": start.metrics["val_loss"],
                "end_val_loss": end.metrics["val_loss"],
                "paired_improvement": start.metrics["val_loss"] - end.metrics["val_loss"],
                "start_val_accuracy": start.metrics["val_accuracy"],
                "end_val_accuracy": end.metrics["val_accuracy"],
                "inherited_best_val_loss": (
                    inherited.metrics["val_loss"] if inherited is not None else None
                ),
                "acceptance_rate": snapshot.acceptance_rate if snapshot is not None else None,
                "effective_online_epochs": (
                    snapshot.effective_online_epochs if snapshot is not None else None
                ),
            }
        )
    return rows


def save_gui_multi_run_session(session: GuiMultiRunSession) -> Path:
    """Schreibt eine Session atomar als JSON."""

    session.updated_at = datetime.now().isoformat(timespec="seconds")
    session.output_dir.mkdir(parents=True, exist_ok=True)
    path = session.output_dir / "session.json"
    temp_path = session.output_dir / "session.json.tmp"
    temp_path.write_text(json.dumps(_session_to_dict(session), indent=2), encoding="utf-8")
    temp_path.replace(path)
    return path


def load_gui_multi_run_session(path: Path) -> GuiMultiRunSession:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != GUI_MULTI_RUN_SCHEMA_VERSION:
        raise ValueError("Nicht unterstuetzte GUI-Multi-Run-Schema-Version.")
    benchmark = str(payload["benchmark"])
    hidden_sizes = tuple(int(value) for value in payload["hidden_sizes"])
    records = [
        _record_from_dict(item, benchmark=benchmark, hidden_sizes=hidden_sizes)
        for item in payload["runs"]
    ]
    return GuiMultiRunSession(
        benchmark=benchmark,
        profile_name=str(payload["profile_name"]),
        master_seed=int(payload["master_seed"]),
        hidden_sizes=hidden_sizes,
        config_snapshot=dict(payload.get("config_snapshot", {})),
        runs=records,
        output_dir=Path(path).parent,
        created_at=str(payload["created_at"]),
        updated_at=str(payload["updated_at"]),
        comparison_mode=str(payload.get("comparison_mode", "robustness")),
        schema_version=int(payload["schema_version"]),
    )


def latest_gui_multi_run_session(
    root: Path = OUTPUT_DIR / "gui_multi_runs",
) -> Path | None:
    candidates = sorted(Path(root).glob("*/session.json"), key=lambda item: item.stat().st_mtime)
    return candidates[-1] if candidates else None


def _aggregate_matrix(x: np.ndarray, matrix: np.ndarray) -> AggregateSeries:
    with np.errstate(all="ignore"):
        median = np.nanmedian(matrix, axis=0)
        q25 = np.nanpercentile(matrix, 25, axis=0)
        q75 = np.nanpercentile(matrix, 75, axis=0)
    count = np.sum(~np.isnan(matrix), axis=0)
    return AggregateSeries(
        x=tuple(float(value) for value in x),
        median=tuple(float(value) for value in median),
        q25=tuple(float(value) for value in q25),
        q75=tuple(float(value) for value in q75),
        count=tuple(int(value) for value in count),
    )


def _session_to_dict(session: GuiMultiRunSession) -> dict[str, object]:
    return {
        "schema_version": session.schema_version,
        "benchmark": session.benchmark,
        "profile_name": session.profile_name,
        "master_seed": session.master_seed,
        "comparison_mode": session.comparison_mode,
        "hidden_sizes": list(session.hidden_sizes),
        "config_snapshot": session.config_snapshot,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "runs": [_record_to_dict(record) for record in session.runs],
    }


def _record_to_dict(record: GuiRunRecord) -> dict[str, object]:
    payload: dict[str, object] = {
        "run_index": record.run_index,
        "schedule": record.schedule.to_dict(),
        "start_layout_spec": record.start_layout_spec,
        "status": record.status,
        "current_stage": record.current_stage,
        "manually_edited": record.manually_edited,
        "error": record.error,
    }
    if record.training_artifacts is not None:
        payload["training"] = {
            "model_state": record.training_artifacts.model.to_state_dict(),
            "history": record.training_artifacts.training_result.history,
            "test_metrics": record.training_artifacts.training_result.test_metrics,
        }
    if record.annealing_snapshot is not None:
        payload["annealing"] = _snapshot_to_dict(record.annealing_snapshot)
    if record.layout_evaluation is not None:
        payload["layout_evaluation"] = _layout_evaluation_to_dict(record.layout_evaluation)
    return payload


def _record_from_dict(
    payload: dict[str, Any],
    *,
    benchmark: str,
    hidden_sizes: tuple[int, ...],
) -> GuiRunRecord:
    schedule = SeedSchedule(**payload["schedule"])
    persisted_status = str(payload.get("status", "pending"))
    interrupted = persisted_status == "running"
    restored_status = (
        "baseline_done"
        if interrupted and payload.get("training")
        else ("pending" if interrupted else persisted_status)
    )
    restored_stage = (
        "baseline"
        if interrupted and payload.get("training")
        else ("layout" if interrupted else str(payload.get("current_stage", "layout")))
    )
    record = GuiRunRecord(
        run_index=int(payload["run_index"]),
        schedule=schedule,
        start_layout_spec=str(payload["start_layout_spec"]),
        status=restored_status,
        current_stage=restored_stage,
        manually_edited=bool(payload.get("manually_edited", False)),
        error=payload.get("error"),
    )
    training = payload.get("training")
    if training:
        dataset = load_benchmark(DatasetConfig(name=benchmark, random_state=schedule.data_split_seed))
        model = ModularMLP.from_state_dict(training["model_state"])
        layout = parse_layout_spec(record.start_layout_spec, hidden_sizes)
        record.dataset = dataset
        record.current_model = model
        record.training_artifacts = TrainingRunArtifacts(
            dataset=dataset,
            base_layout=layout,
            training_layout=layout,
            model=model,
            training_result=TrainingResult(
                history=training["history"],
                test_metrics=training["test_metrics"],
            ),
        )
    if payload.get("layout_evaluation") and not interrupted:
        record.layout_evaluation = _layout_evaluation_from_dict(payload["layout_evaluation"])
    # Eine laufende SA-Session wird absichtlich nicht rekonstruiert. Der Run startet
    # nach einem Abbruch an seiner Grenze neu; persistierte SA-Kurven bleiben erhalten.
    if payload.get("annealing") and not interrupted:
        record.annealing_snapshot = _snapshot_summary_from_dict(payload["annealing"], record)
        if record.annealing_snapshot.current_evaluation is not None:
            record.current_model = record.annealing_snapshot.current_evaluation.trained_model
            record.dataset = load_benchmark(
                DatasetConfig(name=benchmark, random_state=schedule.data_split_seed)
            )
    return record


def _snapshot_to_dict(snapshot: OnlineAnnealingSnapshot) -> dict[str, object]:
    return {
        "is_complete": snapshot.is_complete,
        "accepted_steps": snapshot.accepted_steps,
        "trained_batch_updates": snapshot.trained_batch_updates,
        "effective_online_epochs": snapshot.effective_online_epochs,
        "acceptance_rate": snapshot.acceptance_rate,
        "current_temperature": snapshot.current_temperature,
        "stop_reasons": list(snapshot.stop_reasons),
        "start": _evaluation_to_dict(snapshot.start_evaluation),
        "current": _evaluation_to_dict(snapshot.current_evaluation),
        "best": _evaluation_to_dict(snapshot.best_evaluation),
        "history": [
            {
                "step_index": step.step_index,
                "temperature": step.temperature,
                "previous_layout": step.previous_layout.to_compact_spec(),
                "candidate_layout": step.candidate_layout.to_compact_spec(),
                "delta": step.delta,
                "acceptance_probability": step.acceptance_probability,
                "random_draw": step.random_draw,
                "accepted": step.accepted,
                "reason_code": step.reason_code,
                "neighbor_label": step.neighbor_label,
                "best_score_after_step": step.best_score_after_step,
                "accepted_steps_after_step": step.accepted_steps_after_step,
                "rejected_steps_after_step": step.rejected_steps_after_step,
                "batch_loss_before": step.batch_loss_before,
                "candidate_loss_after": step.candidate_loss_after,
                "post_training_batch_loss": step.post_training_batch_loss,
                "trained_after_accept": step.trained_after_accept,
                "trained_batch_updates_after_step": step.trained_batch_updates_after_step,
                "effective_online_epochs_after_step": step.effective_online_epochs_after_step,
                "diagnostics_refreshed": step.diagnostics_refreshed,
                "epoch_index": step.epoch_index,
                "batch_index": step.batch_index,
                "batch_start": step.batch_start,
                "validation_loss_after_update": step.validation_loss_after_update,
            }
            for step in snapshot.history
        ],
    }


def _evaluation_to_dict(evaluation: object | None) -> dict[str, object] | None:
    if evaluation is None:
        return None
    return {
        "layout_spec": evaluation.layout.to_compact_spec(),
        "comparable_score": evaluation.comparable_score,
        "objective_value": evaluation.objective_value,
        "train_loss": evaluation.train_loss,
        "val_loss": evaluation.val_loss,
        "test_loss": evaluation.test_loss,
        "train_accuracy": evaluation.train_accuracy,
        "val_accuracy": evaluation.val_accuracy,
        "test_accuracy": evaluation.test_accuracy,
        "model_state": evaluation.trained_model.to_state_dict(),
        "metadata": evaluation.metadata,
    }


def _snapshot_summary_from_dict(payload: dict[str, Any], record: GuiRunRecord) -> OnlineAnnealingSnapshot:
    from services.online_annealing_training_service import OnlineAnnealingStep, OnlineLayoutEvaluation

    def evaluation(item: dict[str, Any] | None):
        if item is None:
            return None
        return OnlineLayoutEvaluation(
            layout=parse_layout_spec(item["layout_spec"], record.schedule_hidden_sizes),
            comparable_score=float(item["comparable_score"]),
            objective_value=float(item["objective_value"]),
            train_loss=float(item["train_loss"]),
            val_loss=float(item["val_loss"]),
            test_loss=float(item["test_loss"]),
            train_accuracy=float(item["train_accuracy"]),
            val_accuracy=float(item["val_accuracy"]),
            test_accuracy=float(item["test_accuracy"]),
            trained_model=ModularMLP.from_state_dict(item["model_state"]),
            metadata=dict(item.get("metadata", {})),
        )

    # Hidden-Sizes werden aus dem gespeicherten Modell gelesen, damit das JSON
    # keine duplizierte Topologie pro Step benoetigt.
    template = payload.get("start") or payload.get("current") or payload.get("best")
    if template is None:
        raise ValueError("Persistierter SA-Snapshot besitzt kein Modell.")
    record.schedule_hidden_sizes = tuple(template["model_state"]["hidden_sizes"])  # type: ignore[attr-defined]
    start = evaluation(payload.get("start"))
    current = evaluation(payload.get("current"))
    best = evaluation(payload.get("best"))
    reference = current or start or best
    assert reference is not None
    history = []
    for item in payload.get("history", []):
        history.append(
            OnlineAnnealingStep(
                step_index=int(item["step_index"]),
                temperature=float(item["temperature"]),
                previous_layout=parse_layout_spec(item["previous_layout"], record.schedule_hidden_sizes),  # type: ignore[attr-defined]
                candidate_layout=parse_layout_spec(item["candidate_layout"], record.schedule_hidden_sizes),  # type: ignore[attr-defined]
                previous_evaluation=reference,
                candidate_evaluation=reference,
                delta=float(item["delta"]),
                acceptance_probability=float(item["acceptance_probability"]),
                random_draw=float(item["random_draw"]),
                accepted=bool(item["accepted"]),
                reason_code=str(item["reason_code"]),
                neighbor_label=str(item["neighbor_label"]),
                best_score_after_step=float(item["best_score_after_step"]),
                accepted_steps_after_step=int(item["accepted_steps_after_step"]),
                rejected_steps_after_step=int(item["rejected_steps_after_step"]),
                batch_loss_before=float(item["batch_loss_before"]),
                candidate_loss_after=float(item["candidate_loss_after"]),
                post_training_batch_loss=float(item["post_training_batch_loss"]),
                trained_after_accept=bool(item["trained_after_accept"]),
                trained_batch_updates_after_step=int(item["trained_batch_updates_after_step"]),
                effective_online_epochs_after_step=float(item["effective_online_epochs_after_step"]),
                diagnostics_refreshed=bool(item["diagnostics_refreshed"]),
                epoch_index=int(item["epoch_index"]),
                batch_index=int(item["batch_index"]),
                batch_start=int(item["batch_start"]),
                validation_loss_after_update=float(item["validation_loss_after_update"]),
            )
        )
    delattr(record, "schedule_hidden_sizes")
    return OnlineAnnealingSnapshot(
        is_initialized=True,
        is_complete=bool(payload.get("is_complete", False)),
        state_label="Geladen",
        decision_label="geladener Endzustand",
        reason_label="aus Autosave geladen",
        start_evaluation=start,
        candidate_evaluation=None,
        current_evaluation=current,
        best_evaluation=best,
        end_evaluation=current if payload.get("is_complete") else None,
        last_step=history[-1] if history else None,
        history=tuple(history),
        accepted_steps=int(payload.get("accepted_steps", 0)),
        trained_batch_updates=int(payload.get("trained_batch_updates", 0)),
        effective_online_epochs=float(payload.get("effective_online_epochs", 0.0)),
        acceptance_rate=float(payload.get("acceptance_rate", 0.0)),
        current_temperature=float(payload.get("current_temperature", 0.0)),
        stop_reasons=tuple(payload.get("stop_reasons", [])),
        epoch_index=history[-1].epoch_index if history else 0,
        batch_index=history[-1].batch_index if history else 0,
        batch_start=history[-1].batch_start if history else 0,
    )


def _layout_evaluation_to_dict(result: LayoutEvaluationResult) -> dict[str, object]:
    return {
        "runs": [_layout_run_to_dict(run) for run in result.runs],
        "aggregated": [asdict(item) for item in result.aggregated],
        "ranking": [asdict(item) for item in result.ranking],
        "inherited_runs": [_layout_run_to_dict(run) for run in result.inherited_runs],
        "combined_ranking": [asdict(item) for item in result.combined_ranking],
    }


def _layout_run_to_dict(run: LayoutEvaluationRun) -> dict[str, object]:
    return asdict(run)


def _layout_evaluation_from_dict(payload: dict[str, Any]) -> LayoutEvaluationResult:
    return LayoutEvaluationResult(
        runs=tuple(LayoutEvaluationRun(**item) for item in payload.get("runs", [])),
        aggregated=tuple(AggregatedLayoutEvaluation(**item) for item in payload.get("aggregated", [])),
        ranking=tuple(AggregatedLayoutEvaluation(**item) for item in payload.get("ranking", [])),
        inherited_runs=tuple(
            LayoutEvaluationRun(**item) for item in payload.get("inherited_runs", [])
        ),
        combined_ranking=tuple(
            AggregatedLayoutEvaluation(**item) for item in payload.get("combined_ranking", [])
        ),
    )
