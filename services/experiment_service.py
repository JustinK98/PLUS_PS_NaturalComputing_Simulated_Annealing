"""Services fuer experiment_builder-Definitionen und Builder-Ausfuehrungen."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from configs import (
    DEFAULT_ANNEALING_CANDIDATE_EPOCHS,
    DEFAULT_ANNEALING_COOLING_PARAMETER,
    DEFAULT_ANNEALING_COOLING_SCHEDULE,
    DEFAULT_ANNEALING_ITERATIONS_PER_TEMPERATURE,
    DEFAULT_ANNEALING_MAX_STEPS,
    DEFAULT_ANNEALING_MIN_TEMPERATURE,
    DEFAULT_ANNEALING_NEIGHBORHOODS,
    DEFAULT_ANNEALING_OBJECTIVE,
    DEFAULT_ANNEALING_START_TEMPERATURE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BENCHMARK,
    DEFAULT_EPOCHS,
    DEFAULT_LAYOUT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_WEIGHT_SCALE,
    OUTPUT_DIR,
    default_hidden_sizes,
)
from experiment_builder import ExperimentDefinition
from experiment_runner import ExperimentRunner
from results_store import load_experiment_results
from search_spaces import SearchSpaceDefinition, SearchValueDefinition


def load_experiment_definition(path: str | Path) -> ExperimentDefinition:
    """Laedt eine ExperimentDefinition aus einer JSON-Datei oder einem Manifest."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "definition" in payload:
        payload = payload["definition"]
    return ExperimentDefinition.from_dict(payload)


def save_experiment_definition(definition: ExperimentDefinition, path: str | Path) -> Path:
    """Schreibt eine ExperimentDefinition als JSON-Datei auf Platte."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(definition.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def save_experiment_template(path: str | Path) -> Path:
    """Schreibt eine didaktische Experiment-Template-Datei auf Platte."""

    definition = default_experiment_definition()
    return save_experiment_definition(definition, path)


def default_experiment_definition() -> ExperimentDefinition:
    """Liefert eine sinnvolle Default-Definition fuer CLI und Builder."""

    return ExperimentDefinition(
        experiment_id="example_wine_manual",
        benchmark=DEFAULT_BENCHMARK,
        hidden_sizes=default_hidden_sizes(DEFAULT_BENCHMARK),
        layout_spec=DEFAULT_LAYOUT,
        run_mode="manual_training",
        seeds=(DEFAULT_RANDOM_SEED,),
        primary_metric="validation_accuracy",
        language="en",
        save_json=True,
        output_dir=str(OUTPUT_DIR / "experiments"),
        shuffle=True,
        learning_rate=DEFAULT_LEARNING_RATE,
        batch_size=DEFAULT_BATCH_SIZE,
        weight_scale=DEFAULT_WEIGHT_SCALE,
        epochs=DEFAULT_EPOCHS,
        objective_name=DEFAULT_ANNEALING_OBJECTIVE,
        candidate_epochs=DEFAULT_ANNEALING_CANDIDATE_EPOCHS,
        neighborhood_operations=DEFAULT_ANNEALING_NEIGHBORHOODS,
        start_temperature=DEFAULT_ANNEALING_START_TEMPERATURE,
        cooling_schedule=DEFAULT_ANNEALING_COOLING_SCHEDULE,
        cooling_parameter=DEFAULT_ANNEALING_COOLING_PARAMETER,
        iterations_per_temperature=DEFAULT_ANNEALING_ITERATIONS_PER_TEMPERATURE,
        max_steps=DEFAULT_ANNEALING_MAX_STEPS,
        min_temperature=DEFAULT_ANNEALING_MIN_TEMPERATURE,
        search_space=SearchSpaceDefinition(
            search_type="none",
            value_definitions=(
                SearchValueDefinition(
                    parameter_name="learning_rate",
                    kind="fixed",
                    value_type="float",
                    fixed_value=DEFAULT_LEARNING_RATE,
                ),
            ),
        ),
    )


def run_experiment_definition(definition: ExperimentDefinition) -> dict[str, Any]:
    """Fuehrt eine Builder-Definition mit dem bestehenden ExperimentRunner aus."""

    return ExperimentRunner().run_experiment(definition)


def load_saved_experiment(path: str | Path) -> dict[str, Any]:
    """Laedt ein gespeichertes Experiment fuer Analyse und CLI-Ausgabe."""

    return load_experiment_results(path)

