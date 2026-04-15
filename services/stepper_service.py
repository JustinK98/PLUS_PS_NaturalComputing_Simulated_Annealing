"""UI-neutrale Stepper-Daten fuer die didaktische Rechenspur."""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks import DatasetBundle
from model import SampleTrace
from services.analysis_sample_service import AnalysisSample


@dataclass(frozen=True)
class StepperEntry:
    """Eine didaktische Karte im Forward/Backward-Stepper."""

    title: str
    body: str


def build_step_entries(
    trace: SampleTrace,
    dataset: DatasetBundle,
    analysis_sample: AnalysisSample,
    language: str = "de",
) -> tuple[StepperEntry, ...]:
    """Erzeugt textuelle Schrittkarten fuer den Qt-Stepper."""

    entries: list[StepperEntry] = []
    if language == "en":
        entries.append(
            StepperEntry(
                title="Input",
                body="\n".join(
                    [
                        f"Source: {analysis_sample.source_label}",
                        f"True target: {analysis_sample.actual_target_name}",
                        f"Analysis target: {analysis_sample.effective_target_name}",
                        "",
                        "Scaled inputs:",
                        ", ".join(
                            f"x{index}={value:+.4f}"
                            for index, value in enumerate(analysis_sample.scaled_sample)
                        ),
                    ]
                ),
            )
        )
    else:
        entries.append(
            StepperEntry(
                title="Input",
                body="\n".join(
                    [
                        f"Quelle: {analysis_sample.source_label}",
                        f"Echtes Ziel: {analysis_sample.actual_target_name}",
                        f"Analyse-Ziel: {analysis_sample.effective_target_name}",
                        "",
                        "Skalierte Eingaben:",
                        ", ".join(
                            f"x{index}={value:+.4f}"
                            for index, value in enumerate(analysis_sample.scaled_sample)
                        ),
                    ]
                ),
            )
        )

    for layer_trace in trace.forward_layers:
        if language == "en":
            body = "\n".join(
                [
                    f"Hidden layer L{layer_trace.layer_index + 1}",
                    f"Activations in this layer: {', '.join(layer_trace.activation_names)}",
                    "",
                    "Input to this layer:",
                    ", ".join(
                        f"{value:+.4f}"
                        for value in layer_trace.input_values[: min(12, len(layer_trace.input_values))]
                    ),
                    "",
                    "Pre-activations z:",
                    ", ".join(
                        f"n{index}={value:+.4f}"
                        for index, value in enumerate(layer_trace.pre_activations)
                    ),
                    "",
                    "Activations a:",
                    ", ".join(
                        f"n{index}={value:+.4f}"
                        for index, value in enumerate(layer_trace.activations)
                    ),
                ]
            )
        else:
            body = "\n".join(
                [
                    f"Hidden-Layer L{layer_trace.layer_index + 1}",
                    f"Aktivierungen im Layer: {', '.join(layer_trace.activation_names)}",
                    "",
                    "Input in diesen Layer:",
                    ", ".join(
                        f"{value:+.4f}"
                        for value in layer_trace.input_values[: min(12, len(layer_trace.input_values))]
                    ),
                    "",
                    "Praeaktivierungen z:",
                    ", ".join(
                        f"n{index}={value:+.4f}"
                        for index, value in enumerate(layer_trace.pre_activations)
                    ),
                    "",
                    "Aktivierungen a:",
                    ", ".join(
                        f"n{index}={value:+.4f}"
                        for index, value in enumerate(layer_trace.activations)
                    ),
                ]
            )
        entries.append(StepperEntry(title=f"Forward L{layer_trace.layer_index + 1}", body=body))

    entries.append(
        StepperEntry(
            title="Output",
            body="\n".join(
                [
                    "Output layer" if language == "en" else "Output-Layer",
                    "Logits:",
                    ", ".join(f"y{index}={value:+.4f}" for index, value in enumerate(trace.logits)),
                    "",
                    "Probabilities:" if language == "en" else "Wahrscheinlichkeiten:",
                    ", ".join(
                        f"{dataset.target_names[index]}={value:.4f}"
                        for index, value in enumerate(trace.probabilities)
                    ),
                    "",
                    (
                        f"Prediction: {dataset.target_names[trace.prediction_index]}"
                        if language == "en"
                        else f"Vorhersage: {dataset.target_names[trace.prediction_index]}"
                    ),
                ]
            ),
        )
    )

    if trace.loss is not None:
        entries.append(
            StepperEntry(
                title="Loss",
                body="\n".join(
                    [
                        (
                            f"Analysis target: {analysis_sample.effective_target_name}"
                            if language == "en"
                            else f"Analyse-Ziel: {analysis_sample.effective_target_name}"
                        ),
                        (
                            f"Cross-entropy loss for this single sample: {trace.loss:.6f}"
                            if language == "en"
                            else f"Cross-Entropy-Loss fuer dieses einzelne Sample: {trace.loss:.6f}"
                        ),
                    ]
                ),
            )
        )

    if trace.output_delta is not None:
        entries.append(
            StepperEntry(
                title="Backward Output",
                body="\n".join(
                    [
                        (
                            "Output error vector dL/dlogits"
                            if language == "en"
                            else "Output-Fehlervektor dL/dlogits"
                        ),
                        ", ".join(f"y{index}={value:+.4f}" for index, value in enumerate(trace.output_delta)),
                    ]
                ),
            )
        )

    for backward_trace in trace.backward_layers:
        entries.append(
            StepperEntry(
                title=f"Backward {backward_trace.layer_label}",
                body="\n".join(
                    [
                        f"Layer: {backward_trace.layer_label}",
                        "Delta values:" if language == "en" else "Delta-Werte:",
                        ", ".join(
                            f"{value:+.4f}"
                            for value in backward_trace.deltas[: min(16, len(backward_trace.deltas))]
                        ),
                        "",
                        (
                            f"Weight gradient norm: {backward_trace.weight_gradient_norm:.6f}"
                            if language == "en"
                            else f"Norm des Gewichtsgradienten: {backward_trace.weight_gradient_norm:.6f}"
                        ),
                        (
                            f"Bias gradient norm: {backward_trace.bias_gradient_norm:.6f}"
                            if language == "en"
                            else f"Norm des Biasgradienten: {backward_trace.bias_gradient_norm:.6f}"
                        ),
                    ]
                ),
            )
        )

    return tuple(entries)
