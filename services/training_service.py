"""GUI-neutrale Services fuer einen einzelnen Trainingslauf."""

from __future__ import annotations

from dataclasses import dataclass

from activations import NeighborResult, apply_neighbor_operations, parse_layout_spec
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
    neighbor_operations: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainingRunArtifacts:
    """Enthaelt alle wichtigen Artefakte eines einzelnen Trainingslaufs."""

    dataset: object
    base_layout: object
    training_layout: object
    neighbor_result: NeighborResult | None
    model: ModularMLP
    training_result: TrainingResult


def run_single_training_experiment(request: TrainingRunRequest) -> TrainingRunArtifacts:
    """Fuehrt einen kompletten Single-Run inklusive optionaler Neighbor-Operationen aus."""

    dataset = load_benchmark(request.dataset_config)
    base_layout = parse_layout_spec(request.layout_spec, request.hidden_sizes)

    neighbor_result: NeighborResult | None = None
    training_layout = base_layout
    if request.neighbor_operations:
        neighbor_result = apply_neighbor_operations(base_layout, request.neighbor_operations)
        training_layout = neighbor_result.layout

    model = ModularMLP(
        input_size=dataset.input_size,
        hidden_sizes=request.hidden_sizes,
        output_size=dataset.output_size,
        layout=training_layout,
        weight_scale=request.weight_scale,
        random_state=request.random_state,
    )
    training_result = train_model(model, dataset, request.training_config)
    return TrainingRunArtifacts(
        dataset=dataset,
        base_layout=base_layout,
        training_layout=training_layout,
        neighbor_result=neighbor_result,
        model=model,
        training_result=training_result,
    )

