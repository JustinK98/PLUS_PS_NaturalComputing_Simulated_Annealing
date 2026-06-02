"""Finale Layout-Vergleiche unter identischen Trainingsbedingungen."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from activations import parse_layout_spec, random_layout_spec
from benchmarks import load_benchmark
from configs import DatasetConfig, TrainingConfig, SUPPORTED_ACTIVATIONS
from model import ModularMLP
from trainer import train_model


@dataclass(frozen=True)
class LayoutCandidate:
    """Ein Layout, das final trainiert und verglichen werden soll."""

    label: str
    layout_spec: str


@dataclass(frozen=True)
class LayoutEvaluationRun:
    """Ein einzelner Seed-Run fuer ein Layout."""

    label: str
    layout_spec: str
    seed: int
    metrics: dict[str, float]
    history: dict[str, list[float]]
    model_state: dict[str, object]
    evaluation_type: str = "retrained"


@dataclass(frozen=True)
class InheritedModelCandidate:
    """Ein bereits trainierter Modellzustand, der nur evaluiert wird."""

    label: str
    model_state: dict[str, object]
    seed: int


@dataclass(frozen=True)
class AggregatedLayoutEvaluation:
    """Aggregation ueber alle Seeds eines Layouts."""

    label: str
    layout_spec: str
    num_runs: int
    mean_metrics: dict[str, float]
    std_metrics: dict[str, float]
    min_metrics: dict[str, float]
    max_metrics: dict[str, float]
    ranking_score: float
    evaluation_type: str = "retrained"


@dataclass(frozen=True)
class LayoutEvaluationRequest:
    """Konfiguration fuer einen fairen Layout-Vergleich."""

    dataset_config: DatasetConfig
    hidden_sizes: tuple[int, ...]
    candidates: tuple[LayoutCandidate, ...]
    seeds: tuple[int, ...]
    training_config: TrainingConfig
    weight_scale: float
    primary_metric: str = "validation_loss"
    include_test_metrics: bool = True


@dataclass(frozen=True)
class LayoutEvaluationResult:
    """Vollstaendiges Ergebnis eines Layout-Vergleichs."""

    runs: tuple[LayoutEvaluationRun, ...]
    aggregated: tuple[AggregatedLayoutEvaluation, ...]
    ranking: tuple[AggregatedLayoutEvaluation, ...]
    inherited_runs: tuple[LayoutEvaluationRun, ...] = ()
    combined_ranking: tuple[AggregatedLayoutEvaluation, ...] = ()


def build_standard_layout_candidates(
    hidden_sizes: tuple[int, ...],
    *,
    start_layout_spec: str,
    best_layout_spec: str | None = None,
    end_layout_spec: str | None = None,
    random_state: int = 42,
) -> tuple[LayoutCandidate, ...]:
    """Erzeugt die Standard-Baselines fuer die finale Layout-Auswertung."""

    candidates: list[LayoutCandidate] = [LayoutCandidate("start_layout", start_layout_spec)]
    if best_layout_spec is not None:
        candidates.append(LayoutCandidate("best_layout_from_sa", best_layout_spec))
    if end_layout_spec is not None and end_layout_spec != best_layout_spec:
        candidates.append(LayoutCandidate("end_layout_from_sa", end_layout_spec))
    candidates.append(
        LayoutCandidate(
            "random_layout",
            random_layout_spec(hidden_sizes, random_state),
        )
    )
    for activation_name in SUPPORTED_ACTIVATIONS:
        candidates.append(LayoutCandidate(f"all_{activation_name}", activation_name))
    return _deduplicate_candidates(tuple(candidates))


def run_layout_evaluation(request: LayoutEvaluationRequest) -> LayoutEvaluationResult:
    """Trainiert mehrere Layouts fair gegeneinander und aggregiert die Ergebnisse."""

    if not request.candidates:
        raise ValueError("LayoutEvaluationRequest braucht mindestens ein Layout.")
    if not request.seeds:
        raise ValueError("LayoutEvaluationRequest braucht mindestens einen Seed.")
    if request.primary_metric not in {"validation_loss", "validation_accuracy"}:
        raise ValueError("primary_metric muss validation_loss oder validation_accuracy sein.")

    runs: list[LayoutEvaluationRun] = []
    for seed in request.seeds:
        dataset = load_benchmark(
            DatasetConfig(
                name=request.dataset_config.name,
                validation_size=request.dataset_config.validation_size,
                test_size=request.dataset_config.test_size,
                random_state=int(seed),
            )
        )
        for candidate in request.candidates:
            layout = parse_layout_spec(candidate.layout_spec, request.hidden_sizes)
            model = ModularMLP(
                input_size=dataset.input_size,
                hidden_sizes=request.hidden_sizes,
                output_size=dataset.model_output_size,
                layout=layout,
                num_classes=dataset.output_size,
                weight_scale=request.weight_scale,
                random_state=int(seed),
            )
            training_config = TrainingConfig(
                epochs=request.training_config.epochs,
                learning_rate=request.training_config.learning_rate,
                batch_size=request.training_config.batch_size,
                random_state=int(seed),
                shuffle=request.training_config.shuffle,
            )
            training_result = train_model(
                model,
                dataset,
                training_config,
                include_test_metrics=request.include_test_metrics,
            )
            metrics = {
                "train_loss": float(training_result.history["train_loss"][-1]),
                "val_loss": float(training_result.history["val_loss"][-1]),
                "train_accuracy": float(training_result.history["train_acc"][-1]),
                "val_accuracy": float(training_result.history["val_acc"][-1]),
            }
            if request.include_test_metrics:
                metrics.update(
                    {
                        "test_loss": float(training_result.test_metrics["loss"]),
                        "test_accuracy": float(training_result.test_metrics["accuracy"]),
                    }
                )
            runs.append(
                LayoutEvaluationRun(
                    label=candidate.label,
                    layout_spec=layout.to_compact_spec(),
                    seed=int(seed),
                    metrics=metrics,
                    history=training_result.history,
                    model_state=model.to_state_dict(),
                )
            )

    aggregated = _aggregate_runs(tuple(runs), request.primary_metric)
    ranking = tuple(sorted(aggregated, key=lambda item: item.ranking_score))
    return LayoutEvaluationResult(
        runs=tuple(runs),
        aggregated=aggregated,
        ranking=ranking,
        combined_ranking=ranking,
    )


def evaluate_inherited_models(
    dataset_config: DatasetConfig,
    candidates: tuple[InheritedModelCandidate, ...],
    *,
    primary_metric: str = "validation_loss",
    include_test_metrics: bool = True,
) -> tuple[LayoutEvaluationRun, ...]:
    """Evaluiert geerbte Modellzustaende ohne weiteres Training."""

    if primary_metric not in {"validation_loss", "validation_accuracy"}:
        raise ValueError("primary_metric muss validation_loss oder validation_accuracy sein.")

    runs: list[LayoutEvaluationRun] = []
    for candidate in candidates:
        dataset = load_benchmark(
            DatasetConfig(
                name=dataset_config.name,
                validation_size=dataset_config.validation_size,
                test_size=dataset_config.test_size,
                random_state=int(candidate.seed),
            )
        )
        model = ModularMLP.from_state_dict(candidate.model_state)
        runs.append(
            LayoutEvaluationRun(
                label=candidate.label,
                layout_spec=model.layout.to_compact_spec(),
                seed=int(candidate.seed),
                metrics=_evaluate_model_metrics(
                    model,
                    dataset,
                    include_test_metrics=include_test_metrics,
                ),
                history={},
                model_state=model.to_state_dict(),
                evaluation_type="inherited",
            )
        )
    return tuple(runs)


def with_inherited_model_evaluations(
    result: LayoutEvaluationResult,
    inherited_runs: tuple[LayoutEvaluationRun, ...],
    primary_metric: str,
) -> LayoutEvaluationResult:
    """Ergaenzt einen Retraining-Vergleich um geerbte SA-Modellzustaende."""

    if primary_metric not in {"validation_loss", "validation_accuracy"}:
        raise ValueError("primary_metric muss validation_loss oder validation_accuracy sein.")

    inherited_aggregated = _aggregate_runs(inherited_runs, primary_metric, evaluation_type="inherited")
    combined_ranking = tuple(
        sorted(
            result.aggregated + inherited_aggregated,
            key=lambda item: item.ranking_score,
        )
    )
    return LayoutEvaluationResult(
        runs=result.runs,
        aggregated=result.aggregated,
        ranking=result.ranking,
        inherited_runs=inherited_runs,
        combined_ranking=combined_ranking,
    )


def layout_evaluation_to_dict(result: LayoutEvaluationResult) -> dict[str, object]:
    """Serialisiert einen Layout-Vergleich inklusive trainierter Modellzustaende."""

    return {
        "runs": [
            {
                "label": run.label,
                "evaluation_type": run.evaluation_type,
                "layout_spec": run.layout_spec,
                "seed": run.seed,
                "metrics": run.metrics,
                "history": run.history,
                "model_state": run.model_state,
            }
            for run in result.runs
        ],
        "retrained_runs": [
            {
                "label": run.label,
                "evaluation_type": run.evaluation_type,
                "layout_spec": run.layout_spec,
                "seed": run.seed,
                "metrics": run.metrics,
                "history": run.history,
                "model_state": run.model_state,
            }
            for run in result.runs
        ],
        "inherited_runs": [
            {
                "label": run.label,
                "evaluation_type": run.evaluation_type,
                "layout_spec": run.layout_spec,
                "seed": run.seed,
                "metrics": run.metrics,
                "history": run.history,
                "model_state": run.model_state,
            }
            for run in result.inherited_runs
        ],
        "aggregated": [
            {
                "label": item.label,
                "evaluation_type": item.evaluation_type,
                "layout_spec": item.layout_spec,
                "num_runs": item.num_runs,
                "mean_metrics": item.mean_metrics,
                "std_metrics": item.std_metrics,
                "min_metrics": item.min_metrics,
                "max_metrics": item.max_metrics,
                "ranking_score": item.ranking_score,
            }
            for item in result.aggregated
        ],
        "ranking": [
            {
                "label": item.label,
                "evaluation_type": item.evaluation_type,
                "layout_spec": item.layout_spec,
                "ranking_score": item.ranking_score,
                "mean_metrics": item.mean_metrics,
            }
            for item in result.ranking
        ],
        "combined_ranking": [
            {
                "label": item.label,
                "evaluation_type": item.evaluation_type,
                "layout_spec": item.layout_spec,
                "ranking_score": item.ranking_score,
                "mean_metrics": item.mean_metrics,
            }
            for item in (result.combined_ranking or result.ranking)
        ],
    }


def best_layout_run(result: LayoutEvaluationResult, primary_metric: str) -> LayoutEvaluationRun:
    """Liefert den besten konkreten trainierten Run aus einem Grid."""

    if not result.runs:
        raise ValueError("LayoutEvaluationResult enthaelt keine Runs.")
    if primary_metric == "validation_loss":
        return min(result.runs, key=lambda run: run.metrics["val_loss"])
    if primary_metric == "validation_accuracy":
        return max(result.runs, key=lambda run: run.metrics["val_accuracy"])
    raise ValueError("primary_metric muss validation_loss oder validation_accuracy sein.")


def _aggregate_runs(
    runs: tuple[LayoutEvaluationRun, ...],
    primary_metric: str,
    evaluation_type: str = "retrained",
) -> tuple[AggregatedLayoutEvaluation, ...]:
    grouped: dict[str, list[LayoutEvaluationRun]] = {}
    for run in runs:
        grouped.setdefault(run.label, []).append(run)

    aggregated: list[AggregatedLayoutEvaluation] = []
    for label, label_runs in grouped.items():
        metric_names = tuple(label_runs[0].metrics)
        metric_values = {
            metric_name: np.asarray([run.metrics[metric_name] for run in label_runs], dtype=np.float64)
            for metric_name in metric_names
        }
        mean_metrics = {name: float(values.mean()) for name, values in metric_values.items()}
        std_metrics = {name: float(values.std(ddof=0)) for name, values in metric_values.items()}
        min_metrics = {name: float(values.min()) for name, values in metric_values.items()}
        max_metrics = {name: float(values.max()) for name, values in metric_values.items()}
        ranking_score = (
            mean_metrics["val_loss"]
            if primary_metric == "validation_loss"
            else 1.0 - mean_metrics["val_accuracy"]
        )
        aggregated.append(
            AggregatedLayoutEvaluation(
                label=label,
                layout_spec=label_runs[0].layout_spec,
                num_runs=len(label_runs),
                mean_metrics=mean_metrics,
                std_metrics=std_metrics,
                min_metrics=min_metrics,
                max_metrics=max_metrics,
                ranking_score=float(ranking_score),
                evaluation_type=evaluation_type,
            )
        )
    return tuple(aggregated)


def _evaluate_model_metrics(
    model: ModularMLP,
    dataset,
    *,
    include_test_metrics: bool,
) -> dict[str, float]:
    train_loss, train_accuracy = model.evaluate(dataset.X_train, dataset.y_train)
    val_loss, val_accuracy = model.evaluate(dataset.X_val, dataset.y_val)
    metrics = {
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "train_accuracy": float(train_accuracy),
        "val_accuracy": float(val_accuracy),
    }
    if include_test_metrics:
        test_loss, test_accuracy = model.evaluate(dataset.X_test, dataset.y_test)
        metrics.update(
            {
                "test_loss": float(test_loss),
                "test_accuracy": float(test_accuracy),
            }
        )
    return metrics


def _deduplicate_candidates(candidates: tuple[LayoutCandidate, ...]) -> tuple[LayoutCandidate, ...]:
    seen: set[str] = set()
    result: list[LayoutCandidate] = []
    for candidate in candidates:
        if candidate.label in seen:
            continue
        seen.add(candidate.label)
        result.append(candidate)
    return tuple(result)
