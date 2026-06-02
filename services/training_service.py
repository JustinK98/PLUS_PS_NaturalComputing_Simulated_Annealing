"""GUI-neutrale Services fuer einen einzelnen Trainingslauf."""

from __future__ import annotations

from dataclasses import dataclass

from activations import parse_layout_spec
from benchmarks import load_benchmark
from configs import DatasetConfig, TrainingConfig
from model import ModularMLP
from trainer import TrainingResult, train_model


@dataclass(frozen=True)
class TrainingRunRequest:
    """Beschreibt einen einzelnen Trainingslauf unabhaengig von der UI."""

    dataset_config: DatasetConfig
    hidden_sizes: tuple[int, ...]
    layout_spec: str
    training_config: TrainingConfig
    weight_scale: float
    random_state: int


@dataclass(frozen=True)
class TrainingRunArtifacts:
    """Enthaelt alle wichtigen Artefakte eines einzelnen Trainingslaufs."""

    dataset: object
    base_layout: object
    training_layout: object
    model: ModularMLP
    training_result: TrainingResult


def run_single_training_experiment(request: TrainingRunRequest) -> TrainingRunArtifacts:
    """Fuehrt einen kompletten Single-Run fuer ein explizites Layout aus."""

    dataset = load_benchmark(request.dataset_config)
    base_layout = parse_layout_spec(request.layout_spec, request.hidden_sizes)

    model = ModularMLP(
        input_size=dataset.input_size,
        hidden_sizes=request.hidden_sizes,
        output_size=dataset.model_output_size,
        layout=base_layout,
        num_classes=dataset.output_size,
        weight_scale=request.weight_scale,
        random_state=request.random_state,
    )
    training_result = train_model(model, dataset, request.training_config)
    return TrainingRunArtifacts(
        dataset=dataset,
        base_layout=base_layout,
        training_layout=base_layout,
        model=model,
        training_result=training_result,
    )
