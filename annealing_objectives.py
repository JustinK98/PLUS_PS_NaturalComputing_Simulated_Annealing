"""Bewertungsfunktionen fuer Layout-Zustaende im Simulated Annealing."""

from __future__ import annotations

from dataclasses import dataclass, field

from activations import ActivationLayout
from benchmarks import DatasetBundle
from configs import TrainingConfig
from model import ModularMLP
from trainer import TrainingResult, train_model


SUPPORTED_OBJECTIVES = ("validation_loss", "validation_accuracy")


@dataclass(frozen=True)
class ObjectiveConfig:
    """Konfiguration der Layout-Bewertung fuer einen Benchmark."""

    objective_name: str
    candidate_epochs: int
    learning_rate: float
    batch_size: int
    weight_scale: float
    random_state: int
    shuffle: bool = True

    def __post_init__(self) -> None:
        if self.objective_name not in SUPPORTED_OBJECTIVES:
            raise ValueError(
                f"Unbekanntes Objective '{self.objective_name}'. "
                f"Erlaubt sind: {', '.join(SUPPORTED_OBJECTIVES)}"
            )
        if self.candidate_epochs <= 0:
            raise ValueError("candidate_epochs muss positiv sein.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate muss positiv sein.")
        if self.batch_size <= 0:
            raise ValueError("batch_size muss positiv sein.")
        if self.weight_scale <= 0.0:
            raise ValueError("weight_scale muss positiv sein.")
        if self.random_state < 0:
            raise ValueError("random_state darf nicht negativ sein.")


@dataclass
class ObjectiveEvaluation:
    """Bewertung eines konkreten Layout-Zustands."""

    layout: ActivationLayout
    comparable_score: float
    objective_value: float
    train_loss: float
    val_loss: float
    test_loss: float
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    training_result: TrainingResult
    trained_model: ModularMLP
    metadata: dict[str, str] = field(default_factory=dict)


class LayoutObjectiveEvaluator:
    """Trainiert und bewertet ein Layout reproduzierbar auf einem Benchmark."""

    def __init__(self, dataset: DatasetBundle, config: ObjectiveConfig) -> None:
        self.dataset = dataset
        self.config = config
        self._cache: dict[str, ObjectiveEvaluation] = {}

    def evaluate(self, layout: ActivationLayout) -> ObjectiveEvaluation:
        """Trainiert ein frisches Modell fuer ein Layout und berechnet dessen Zielwert."""

        cache_key = self._cache_key(layout)
        if cache_key in self._cache:
            return self._cache[cache_key]

        model = ModularMLP(
            input_size=self.dataset.input_size,
            hidden_sizes=layout.hidden_sizes,
            output_size=self.dataset.output_size,
            layout=layout,
            weight_scale=self.config.weight_scale,
            random_state=self.config.random_state,
        )
        training_config = TrainingConfig(
            epochs=self.config.candidate_epochs,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            random_state=self.config.random_state,
            shuffle=self.config.shuffle,
        )
        training_result = train_model(model, self.dataset, training_config)
        train_loss = float(training_result.history["train_loss"][-1])
        val_loss = float(training_result.history["val_loss"][-1])
        train_accuracy = float(training_result.history["train_acc"][-1])
        val_accuracy = float(training_result.history["val_acc"][-1])
        test_loss = float(training_result.test_metrics["loss"])
        test_accuracy = float(training_result.test_metrics["accuracy"])

        objective_value = (
            val_loss
            if self.config.objective_name == "validation_loss"
            else val_accuracy
        )
        comparable_score = (
            val_loss
            if self.config.objective_name == "validation_loss"
            else 1.0 - val_accuracy
        )
        evaluation = ObjectiveEvaluation(
            layout=layout,
            comparable_score=float(comparable_score),
            objective_value=float(objective_value),
            train_loss=train_loss,
            val_loss=val_loss,
            test_loss=test_loss,
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            test_accuracy=test_accuracy,
            training_result=training_result,
            trained_model=model,
            metadata={
                "objective_name": self.config.objective_name,
                "layout_spec": layout.to_compact_spec(),
                "epochs": str(self.config.candidate_epochs),
            },
        )
        self._cache[cache_key] = evaluation
        return evaluation

    def clear_cache(self) -> None:
        """Leert den Bewertungs-Cache."""

        self._cache.clear()

    def cache_size(self) -> int:
        """Anzahl der bisher gespeicherten Layout-Bewertungen."""

        return len(self._cache)

    def _cache_key(self, layout: ActivationLayout) -> str:
        """Erzeugt einen stabilen Cache-Key fuer Layout und Objective-Konfiguration."""

        return "|".join(
            [
                self.dataset.name,
                layout.to_compact_spec(),
                self.config.objective_name,
                str(self.config.candidate_epochs),
                str(self.config.learning_rate),
                str(self.config.batch_size),
                str(self.config.weight_scale),
                str(self.config.random_state),
                str(int(self.config.shuffle)),
            ]
        )
