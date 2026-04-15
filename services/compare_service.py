"""UI-neutrale Vergleichslogik fuer Baseline vs. aktuelles Modell."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from activations import diff_layouts, generate_single_step_neighbors
from benchmarks import DatasetBundle
from model import ModularMLP
from services.analysis_sample_service import AnalysisSample
from trainer import TrainingResult


@dataclass(frozen=True)
class CompareMetrics:
    epochs: int
    test_accuracy: float
    test_loss: float


@dataclass(frozen=True)
class UncertainSampleEntry:
    sample_index: int
    true_label: str
    predicted_label: str
    confidence: float


@dataclass(frozen=True)
class ComparePayload:
    compatibility_issue: str | None
    summary_text: str
    current_benchmark: str
    baseline_benchmark: str
    current_hidden_sizes: tuple[int, ...]
    baseline_hidden_sizes: tuple[int, ...]
    current_layout_spec: str
    baseline_layout_spec: str
    sample_source: str | None
    baseline_prediction: str | None
    current_prediction: str | None
    baseline_probabilities: tuple[tuple[str, float], ...]
    current_probabilities: tuple[tuple[str, float], ...]
    baseline_metrics: CompareMetrics | None
    current_metrics: CompareMetrics | None
    layout_diff_lines: tuple[str, ...]
    natural_neighbor_count: int | None
    confusion_labels: tuple[str, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    uncertain_samples: tuple[UncertainSampleEntry, ...]


def check_baseline_compatibility(
    current_model: ModularMLP,
    baseline_model: ModularMLP,
    dataset: DatasetBundle,
    baseline_benchmark_name: str | None = None,
    language: str = "de",
) -> str | None:
    """Prueft, ob eine gespeicherte Baseline mit dem aktuellen Benchmark passt."""

    if baseline_benchmark_name is not None and baseline_benchmark_name != dataset.name:
        if language == "en":
            return (
                f"The stored baseline belongs to benchmark '{baseline_benchmark_name}', "
                f"while the current experiment uses '{dataset.name}'."
            )
        return (
            f"Die gespeicherte Baseline gehoert zum Benchmark '{baseline_benchmark_name}', "
            f"waehrend das aktuelle Experiment '{dataset.name}' verwendet."
        )

    if baseline_model.input_size != dataset.input_size:
        if language == "en":
            return (
                f"The baseline expects {baseline_model.input_size} input features, "
                f"but the current benchmark provides {dataset.input_size}."
            )
        return (
            f"Die Baseline erwartet {baseline_model.input_size} Eingabefeatures, "
            f"der aktuelle Benchmark liefert aber {dataset.input_size}."
        )

    if baseline_model.output_size != dataset.output_size:
        if language == "en":
            return (
                f"The baseline expects {baseline_model.output_size} output classes, "
                f"but the current benchmark provides {dataset.output_size}."
            )
        return (
            f"Die Baseline erwartet {baseline_model.output_size} Ausgabeklassen, "
            f"der aktuelle Benchmark liefert aber {dataset.output_size}."
        )
    return None


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> tuple[tuple[int, ...], ...]:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred, strict=True):
        matrix[int(true_label), int(pred_label)] += 1
    return tuple(tuple(int(value) for value in row) for row in matrix)


def _uncertain_samples(
    model: ModularMLP,
    dataset: DatasetBundle,
    top_k: int = 5,
) -> tuple[UncertainSampleEntry, ...]:
    probabilities = model.predict_proba(dataset.X_test)
    confidences = np.max(probabilities, axis=1)
    ranking = np.argsort(confidences)[:top_k]
    predictions = np.argmax(probabilities, axis=1)
    return tuple(
        UncertainSampleEntry(
            sample_index=int(index),
            true_label=dataset.target_names[int(dataset.y_test[index])],
            predicted_label=dataset.target_names[int(predictions[index])],
            confidence=float(confidences[index]),
        )
        for index in ranking
    )


def _metrics_from_result(
    model: ModularMLP,
    dataset: DatasetBundle,
    training_result: TrainingResult | None,
    completed_epochs: int,
) -> CompareMetrics:
    if training_result is None:
        test_loss, test_accuracy = model.evaluate(dataset.X_test, dataset.y_test)
        return CompareMetrics(epochs=completed_epochs, test_accuracy=test_accuracy, test_loss=test_loss)
    return CompareMetrics(
        epochs=completed_epochs,
        test_accuracy=float(training_result.test_metrics["accuracy"]),
        test_loss=float(training_result.test_metrics["loss"]),
    )


def build_compare_payload(
    current_model: ModularMLP,
    baseline_model: ModularMLP,
    dataset: DatasetBundle,
    language: str = "de",
    *,
    analysis_sample: AnalysisSample | None = None,
    baseline_training_result: TrainingResult | None = None,
    current_training_result: TrainingResult | None = None,
    baseline_completed_epochs: int = 0,
    current_completed_epochs: int = 0,
    baseline_benchmark_name: str | None = None,
) -> ComparePayload:
    """Baut einen vollstaendigen Vergleichs-Datensatz fuer das Qt-Panel."""

    compatibility_issue = check_baseline_compatibility(
        current_model,
        baseline_model,
        dataset,
        baseline_benchmark_name=baseline_benchmark_name,
        language=language,
    )
    current_layout = current_model.layout
    baseline_layout = baseline_model.layout

    sample_source = analysis_sample.source_label if analysis_sample is not None else None
    baseline_prediction: str | None = None
    current_prediction: str | None = None
    baseline_probabilities: tuple[tuple[str, float], ...] = ()
    current_probabilities: tuple[tuple[str, float], ...] = ()

    if compatibility_issue is None and analysis_sample is not None:
        sample = analysis_sample.scaled_sample.reshape(1, -1)
        baseline_probs_raw = baseline_model.predict_proba(sample)[0]
        current_probs_raw = current_model.predict_proba(sample)[0]
        baseline_prediction = dataset.target_names[int(np.argmax(baseline_probs_raw))]
        current_prediction = dataset.target_names[int(np.argmax(current_probs_raw))]
        baseline_probabilities = tuple(
            (dataset.target_names[index], float(value))
            for index, value in enumerate(baseline_probs_raw)
        )
        current_probabilities = tuple(
            (dataset.target_names[index], float(value))
            for index, value in enumerate(current_probs_raw)
        )

    layout_diff_lines: tuple[str, ...] = ()
    if compatibility_issue is None and baseline_layout.hidden_sizes == current_layout.hidden_sizes:
        layout_diff_lines = tuple(
            f"L{change.layer_index + 1} n{change.neuron_index}: {change.before} -> {change.after}"
            for change in diff_layouts(baseline_layout, current_layout)
        )

    natural_neighbor_count = (
        len(generate_single_step_neighbors(current_layout))
        if compatibility_issue is None
        else None
    )

    confusion_labels = tuple(dataset.target_names)
    confusion_matrix = ()
    uncertain_samples: tuple[UncertainSampleEntry, ...] = ()
    if compatibility_issue is None:
        predictions = current_model.predict(dataset.X_test)
        confusion_matrix = _confusion_matrix(dataset.y_test, predictions, dataset.output_size)
        uncertain_samples = _uncertain_samples(current_model, dataset)

    if compatibility_issue is not None:
        summary_text = (
            f"Baseline is not comparable to benchmark {dataset.name}."
            if language == "en"
            else f"Baseline nicht vergleichbar mit Benchmark {dataset.name}."
        )
    else:
        summary_text = (
            f"Baseline vs current model on {dataset.name}"
            if language == "en"
            else f"Baseline vs aktuelles Modell auf {dataset.name}"
        )

    baseline_metrics = None
    current_metrics = None
    if compatibility_issue is None:
        baseline_metrics = _metrics_from_result(
            baseline_model,
            dataset,
            baseline_training_result,
            baseline_completed_epochs,
        )
        current_metrics = _metrics_from_result(
            current_model,
            dataset,
            current_training_result,
            current_completed_epochs,
        )
    else:
        current_metrics = _metrics_from_result(
            current_model,
            dataset,
            current_training_result,
            current_completed_epochs,
        )

    return ComparePayload(
        compatibility_issue=compatibility_issue,
        summary_text=summary_text,
        current_benchmark=dataset.name,
        baseline_benchmark=baseline_benchmark_name or dataset.name,
        current_hidden_sizes=current_model.hidden_sizes,
        baseline_hidden_sizes=baseline_model.hidden_sizes,
        current_layout_spec=current_layout.to_compact_spec(),
        baseline_layout_spec=baseline_layout.to_compact_spec(),
        sample_source=sample_source,
        baseline_prediction=baseline_prediction,
        current_prediction=current_prediction,
        baseline_probabilities=baseline_probabilities,
        current_probabilities=current_probabilities,
        baseline_metrics=baseline_metrics,
        current_metrics=current_metrics,
        layout_diff_lines=layout_diff_lines,
        natural_neighbor_count=natural_neighbor_count,
        confusion_labels=confusion_labels,
        confusion_matrix=confusion_matrix,
        uncertain_samples=uncertain_samples,
    )
