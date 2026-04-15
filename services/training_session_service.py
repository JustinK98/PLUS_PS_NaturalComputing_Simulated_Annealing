"""UI-neutrale Session-Logik fuer sichtbares, schrittweises Training."""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks import DatasetBundle
from configs import TrainingConfig
from model import ModularMLP
from trainer import TrainingResult, copy_history, train_model


@dataclass(frozen=True)
class TrainingSessionSnapshot:
    """Serialisierbarer Zwischenstand eines Trainingslaufs."""

    model_state: dict[str, object]
    history: dict[str, list[float]]
    completed_epochs: int
    test_metrics: dict[str, float] | None = None


def create_training_session(model: ModularMLP) -> TrainingSessionSnapshot:
    """Erzeugt eine neue Trainingssession aus einem Modellzustand."""

    return TrainingSessionSnapshot(
        model_state=model.to_state_dict(),
        history=copy_history(None),
        completed_epochs=0,
        test_metrics=None,
    )


def run_training_session_epochs(
    session: TrainingSessionSnapshot,
    dataset: DatasetBundle,
    config: TrainingConfig,
) -> TrainingSessionSnapshot:
    """Fuehrt eine weitere Trainingsstrecke auf einem Session-Snapshot aus."""

    model = ModularMLP.from_state_dict(session.model_state)
    new_result = train_model(model, dataset, config)
    merged_history = {
        key: list(session.history.get(key, [])) + list(new_result.history.get(key, []))
        for key in new_result.history
    }
    return TrainingSessionSnapshot(
        model_state=model.to_state_dict(),
        history=merged_history,
        completed_epochs=session.completed_epochs + int(config.epochs),
        test_metrics=dict(new_result.test_metrics),
    )


def model_from_training_session(session: TrainingSessionSnapshot) -> ModularMLP:
    """Rekonstruiert das Modell fuer einen Session-Zustand."""

    return ModularMLP.from_state_dict(session.model_state)


def training_result_from_session(session: TrainingSessionSnapshot) -> TrainingResult:
    """Formatiert einen Session-Snapshot wie einen klassischen Trainingslauf."""

    return TrainingResult(
        history=copy_history(session.history),
        test_metrics=dict(session.test_metrics or {"loss": 0.0, "accuracy": 0.0}),
    )
