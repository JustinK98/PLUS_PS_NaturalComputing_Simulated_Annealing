"""Didaktische Projektion fuer die Netzdarstellung in der GUI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmarks import DatasetBundle
from model import ModularMLP
from services.analysis_sample_service import AnalysisSample


@dataclass(frozen=True)
class InputProjection:
    """Beschreibt, welche Eingaben im sichtbaren Netzwerk gezeigt werden."""

    displayed_indices: tuple[int, ...]
    displayed_labels: tuple[str, ...]
    total_input_size: int


def build_input_projection(
    dataset: DatasetBundle,
    model: ModularMLP,
    analysis_sample: AnalysisSample,
    selected_hidden: tuple[int, int] | None,
    *,
    max_inputs: int = 12,
) -> InputProjection:
    """Waehlt die didaktisch sichtbarsten Input-Knoten fuer die Netzwerkansicht aus."""

    if dataset.input_size <= max_inputs:
        indices = tuple(range(dataset.input_size))
        return InputProjection(
            displayed_indices=indices,
            displayed_labels=tuple(f"x{index}" for index in indices),
            total_input_size=dataset.input_size,
        )

    chosen_indices: list[int] = []
    if selected_hidden is not None:
        layer_index, neuron_index = selected_hidden
        if layer_index == 0:
            inspection = model.inspect_hidden_neuron(
                X=analysis_sample.scaled_sample.reshape(1, -1),
                sample_index=0,
                layer_index=layer_index,
                neuron_index=neuron_index,
                split_name=analysis_sample.source_label,
                input_labels=dataset.feature_names,
            )
            feature_lookup = {
                str(feature_name): index for index, feature_name in enumerate(dataset.feature_names)
            }
            for term in inspection.top_terms:
                feature_index = feature_lookup.get(term.source_label)
                if feature_index is not None and feature_index not in chosen_indices:
                    chosen_indices.append(feature_index)
                    if len(chosen_indices) >= max_inputs:
                        break

    if len(chosen_indices) < max_inputs:
        fallback = np.linspace(0, dataset.input_size - 1, num=max_inputs, dtype=int).tolist()
        for feature_index in fallback:
            if feature_index not in chosen_indices:
                chosen_indices.append(int(feature_index))
            if len(chosen_indices) >= max_inputs:
                break

    indices = tuple(chosen_indices[:max_inputs])
    labels = tuple(f"x{index}" for index in indices)
    return InputProjection(
        displayed_indices=indices,
        displayed_labels=labels,
        total_input_size=dataset.input_size,
    )

