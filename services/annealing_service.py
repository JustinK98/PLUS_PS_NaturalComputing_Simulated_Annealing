"""GUI-neutrale Services fuer einzelne Simulated-Annealing-Laeufe."""

from __future__ import annotations

from dataclasses import dataclass

from activations import parse_layout_spec
from annealing import AnnealingConfig
from annealing_objectives import LayoutObjectiveEvaluator, ObjectiveConfig
from annealing_runner import AnnealingRunner
from benchmarks import load_benchmark
from configs import DatasetConfig


@dataclass(frozen=True)
class AnnealingRunRequest:
    """Beschreibt einen einzelnen SA-Lauf unabhaengig von GUI oder CLI."""

    dataset_config: DatasetConfig
    hidden_sizes: tuple[int, ...]
    layout_spec: str
    objective_config: ObjectiveConfig
    annealing_config: AnnealingConfig
    random_state: int


@dataclass(frozen=True)
class AnnealingRunArtifacts:
    """Enthaelt die wesentlichen Artefakte eines SA-Laufs."""

    dataset: object
    start_layout: object
    runner: AnnealingRunner
    state: object


def run_annealing_experiment(request: AnnealingRunRequest) -> AnnealingRunArtifacts:
    """Fuehrt einen kompletten SA-Lauf ueber Aktivierungs-Layouts aus."""

    dataset = load_benchmark(request.dataset_config)
    start_layout = parse_layout_spec(request.layout_spec, request.hidden_sizes)
    evaluator = LayoutObjectiveEvaluator(dataset, request.objective_config)
    runner = AnnealingRunner(
        evaluator=evaluator,
        config=request.annealing_config,
        random_state=request.random_state,
    )
    runner.initialize(start_layout)
    runner.run_until_complete()
    if runner.state is None:
        raise ValueError("Der AnnealingRunner lieferte keinen Endzustand.")
    return AnnealingRunArtifacts(
        dataset=dataset,
        start_layout=start_layout,
        runner=runner,
        state=runner.state,
    )

