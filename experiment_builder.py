"""Dataclasses fuer den Experiment Builder.

Diese Datei beschreibt die Datenmodelle fuer reproduzierbare Experimente:

- was ein Experiment ist
- wie ein einzelner Run aussieht
- wie Ergebnisse serialisiert werden
- wie Aggregationen beschrieben werden
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from search_spaces import SearchSpaceDefinition
from configs import (
    DEFAULT_ONLINE_TRAIN_POLICY,
    DEFAULT_SA_EVALUATION_MODE,
    SUPPORTED_ONLINE_TRAIN_POLICIES,
    SUPPORTED_SA_EVALUATION_MODES,
)


SUPPORTED_EXPERIMENT_RUN_MODES = ("manual_training", "simulated_annealing")

PRIMARY_METRIC_LABELS = ("validation_loss", "validation_accuracy")

SUGGESTED_TUNING_VALUES = {
    "learning_rate": ("0.001", "0.003", "0.01", "0.03", "0.1"),
    "batch_size": ("8", "16", "32", "64"),
    "weight_scale": ("0.01", "0.03", "0.05", "0.1"),
    "epochs": ("20", "50", "100"),
    "candidate_epochs": ("5", "10", "20", "30"),
    "start_temperature": ("0.5", "1.0", "1.5", "2.0"),
}


def utc_timestamp() -> str:
    """Zeitstempel fuer Ergebnisdateien."""

    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ExperimentDefinition:
    """Beschreibt ein gesamtes Builder-Experiment."""

    experiment_id: str
    benchmark: str
    hidden_sizes: tuple[int, ...]
    layout_spec: str
    run_mode: str
    seeds: tuple[int, ...]
    primary_metric: str
    language: str
    save_json: bool
    output_dir: str
    shuffle: bool
    learning_rate: float
    batch_size: int
    weight_scale: float
    epochs: int
    objective_name: str
    sa_evaluation_mode: str
    online_train_policy: str
    candidate_epochs: int
    neighborhood_operations: tuple[str, ...]
    start_temperature: float
    cooling_schedule: str
    cooling_parameter: float
    iterations_per_temperature: int
    max_steps: int
    min_temperature: float
    search_space: SearchSpaceDefinition = field(default_factory=SearchSpaceDefinition)
    created_at: str = field(default_factory=utc_timestamp)

    def __post_init__(self) -> None:
        if self.run_mode not in SUPPORTED_EXPERIMENT_RUN_MODES:
            raise ValueError(
                f"Unbekannter Run-Modus '{self.run_mode}'. "
                f"Erlaubt sind: {', '.join(SUPPORTED_EXPERIMENT_RUN_MODES)}"
            )
        if not self.seeds:
            raise ValueError("Ein Experiment braucht mindestens einen Seed.")
        if self.primary_metric not in PRIMARY_METRIC_LABELS:
            raise ValueError(
                f"Unbekannte primäre Metrik '{self.primary_metric}'. "
                f"Erlaubt sind: {', '.join(PRIMARY_METRIC_LABELS)}"
            )
        if self.sa_evaluation_mode not in SUPPORTED_SA_EVALUATION_MODES:
            raise ValueError(
                f"Unbekannter SA-Bewertungsmodus '{self.sa_evaluation_mode}'. "
                f"Erlaubt sind: {', '.join(SUPPORTED_SA_EVALUATION_MODES)}"
            )
        if self.online_train_policy not in SUPPORTED_ONLINE_TRAIN_POLICIES:
            raise ValueError(
                f"Unbekannte Online-Trainingspolicy '{self.online_train_policy}'. "
                f"Erlaubt sind: {', '.join(SUPPORTED_ONLINE_TRAIN_POLICIES)}"
            )

    @property
    def output_path(self) -> Path:
        """Experimentpfad unterhalb des Output-Ordners."""

        return Path(self.output_dir) / self.experiment_id

    def to_dict(self) -> dict[str, Any]:
        """Serialisiert das Experiment fuer JSON."""

        payload = asdict(self)
        payload["hidden_sizes"] = list(self.hidden_sizes)
        payload["seeds"] = list(self.seeds)
        payload["neighborhood_operations"] = list(self.neighborhood_operations)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentDefinition":
        """Rekonstruiert eine ExperimentDefinition aus JSON-nahen Daten."""

        return cls(
            experiment_id=str(payload["experiment_id"]),
            benchmark=str(payload["benchmark"]),
            hidden_sizes=tuple(int(size) for size in payload["hidden_sizes"]),
            layout_spec=str(payload["layout_spec"]),
            run_mode=str(payload["run_mode"]),
            seeds=tuple(int(seed) for seed in payload["seeds"]),
            primary_metric=str(payload["primary_metric"]),
            language=str(payload.get("language", "en")),
            save_json=bool(payload.get("save_json", True)),
            output_dir=str(payload["output_dir"]),
            shuffle=bool(payload["shuffle"]),
            learning_rate=float(payload["learning_rate"]),
            batch_size=int(payload["batch_size"]),
            weight_scale=float(payload["weight_scale"]),
            epochs=int(payload["epochs"]),
            objective_name=str(payload["objective_name"]),
            sa_evaluation_mode=str(payload.get("sa_evaluation_mode", "short_retrain")),
            online_train_policy=str(payload.get("online_train_policy", DEFAULT_ONLINE_TRAIN_POLICY)),
            candidate_epochs=int(payload["candidate_epochs"]),
            neighborhood_operations=tuple(str(value) for value in payload["neighborhood_operations"]),
            start_temperature=float(payload["start_temperature"]),
            cooling_schedule=str(payload["cooling_schedule"]),
            cooling_parameter=float(payload["cooling_parameter"]),
            iterations_per_temperature=int(payload["iterations_per_temperature"]),
            max_steps=int(payload["max_steps"]),
            min_temperature=float(payload["min_temperature"]),
            search_space=SearchSpaceDefinition.from_dict(payload.get("search_space", {})),
            created_at=str(payload.get("created_at", utc_timestamp())),
        )


@dataclass(frozen=True)
class RunDefinition:
    """Beschreibt einen einzelnen konkreten Run."""

    run_id: str
    experiment_id: str
    config_id: str
    config_index: int
    seed: int
    benchmark: str
    hidden_sizes: tuple[int, ...]
    layout_spec: str
    run_mode: str
    primary_metric: str
    config_values: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hidden_sizes"] = list(self.hidden_sizes)
        return payload


@dataclass(frozen=True)
class RunResult:
    """Ergebnis eines einzelnen Runs."""

    run_definition: RunDefinition
    metrics: dict[str, float]
    history: dict[str, list[float]]
    layout_spec: str
    created_at: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_definition": self.run_definition.to_dict(),
            "metrics": self.metrics,
            "history": self.history,
            "layout_spec": self.layout_spec,
            "created_at": self.created_at,
            "extra": self.extra,
        }


@dataclass(frozen=True)
class AggregatedMetrics:
    """Aggregation ueber mehrere Seeds fuer eine Konfiguration."""

    config_id: str
    config_index: int
    config_values: dict[str, Any]
    num_runs: int
    num_seeds: int
    mean_metrics: dict[str, float]
    std_metrics: dict[str, float]
    min_metrics: dict[str, float]
    max_metrics: dict[str, float]
    best_seed: int
    worst_seed: int
    ranking_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentSummary:
    """Kompakte Zusammenfassung eines Builder-Experiments."""

    experiment_id: str
    benchmark: str
    run_mode: str
    search_type: str
    primary_metric: str
    number_of_runs: int
    number_of_seeds: int
    configuration_count: int
    aggregated_metrics: tuple[AggregatedMetrics, ...]
    ranking: tuple[dict[str, Any], ...]
    created_at: str = field(default_factory=utc_timestamp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "benchmark": self.benchmark,
            "run_mode": self.run_mode,
            "search_type": self.search_type,
            "primary_metric": self.primary_metric,
            "number_of_runs": self.number_of_runs,
            "number_of_seeds": self.number_of_seeds,
            "configuration_count": self.configuration_count,
            "aggregated_metrics": [metrics.to_dict() for metrics in self.aggregated_metrics],
            "ranking": list(self.ranking),
            "created_at": self.created_at,
        }
