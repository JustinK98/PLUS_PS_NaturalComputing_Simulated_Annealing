"""UI-neutrale Didaktikdaten fuer den Hidden-Neuron-Tracker."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from activations import ACTIVATIONS
from benchmarks import DatasetBundle
from model import ModularMLP, NeuronInspection, WeightedTerm
from services.analysis_sample_service import AnalysisSample


@dataclass(frozen=True)
class NeuronAnalysisPayload:
    """Vollstaendige, sample-spezifische Sicht auf ein Hidden-Neuron."""

    inspection: NeuronInspection
    title: str
    formula_label: str
    source_label: str
    actual_target_label: str
    analysis_target_label: str
    equation_text: str
    interpretation_lines: tuple[str, ...]
    top_terms: tuple[WeightedTerm, ...]
    outgoing_weights: tuple[tuple[str, float], ...]
    curve_x: tuple[float, ...]
    curve_y: tuple[float, ...]


def _interpretation_lines(language: str) -> tuple[str, ...]:
    if language == "en":
        return (
            "Positive terms push z upward, negative terms push it downward.",
            "Only after that does the activation function decide how strongly this z",
            "is converted into an output a for this concrete sample.",
        )
    return (
        "Positive Summanden schieben z nach oben, negative nach unten.",
        "Erst danach entscheidet die Aktivierungsfunktion, wie stark dieses z",
        "fuer genau dieses Sample in eine Ausgabe a ueberfuehrt wird.",
    )


def _equation_text(inspection: NeuronInspection) -> str:
    visible_terms = (
        inspection.input_terms
        if len(inspection.input_terms) <= 6
        else inspection.top_terms
    )
    equation = " + ".join(
        f"({term.source_value:+.3f} * {term.weight:+.3f})" for term in visible_terms
    )
    if len(visible_terms) < len(inspection.input_terms):
        equation += " + ..."
    return f"{equation} + ({inspection.bias:+.3f})"


def _curve_points(inspection: NeuronInspection) -> tuple[tuple[float, ...], tuple[float, ...]]:
    activation_function = ACTIVATIONS[inspection.activation_name].forward
    z_values = np.linspace(-4.0, 4.0, 200)
    y_values = activation_function(z_values)
    return (
        tuple(float(value) for value in z_values),
        tuple(float(value) for value in y_values),
    )


def build_neuron_analysis_payload(
    model: ModularMLP,
    dataset: DatasetBundle,
    analysis_sample: AnalysisSample,
    layer_index: int,
    neuron_index: int,
    language: str = "de",
) -> NeuronAnalysisPayload:
    """Berechnet die didaktische Detailansicht fuer ein Hidden-Neuron."""

    inspection = model.inspect_hidden_neuron(
        X=analysis_sample.scaled_sample.reshape(1, -1),
        sample_index=0,
        layer_index=layer_index,
        neuron_index=neuron_index,
        split_name=analysis_sample.source_label,
        input_labels=dataset.feature_names if layer_index == 0 else None,
    )
    curve_x, curve_y = _curve_points(inspection)
    return NeuronAnalysisPayload(
        inspection=inspection,
        title=f"L{inspection.layer_index + 1}:n{inspection.neuron_index}",
        formula_label=inspection.activation_formula,
        source_label=analysis_sample.source_label,
        actual_target_label=analysis_sample.actual_target_name,
        analysis_target_label=analysis_sample.effective_target_name,
        equation_text=_equation_text(inspection),
        interpretation_lines=_interpretation_lines(language),
        top_terms=inspection.top_terms,
        outgoing_weights=tuple(
            sorted(inspection.outgoing_weights, key=lambda item: abs(item[1]), reverse=True)[:6]
        ),
        curve_x=curve_x,
        curve_y=curve_y,
    )
