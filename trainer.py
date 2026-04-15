"""Trainingslogik fuer das kleine MLP.

Diese Datei trennt den Trainingsablauf sauber vom Modell selbst:

- `model.py` weiss, wie Vorwaerts- und Rueckwaertspass funktionieren
- `trainer.py` weiss, wie daraus ein Epochen-Loop wird

Didaktisch ist das sinnvoll, weil Studierende so Modellstruktur und Trainingsablauf
separat lesen koennen.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import numpy as np
except ImportError as exc:
    raise ImportError(
        "Dieses Projekt benoetigt NumPy. Installation z. B. mit 'pip install numpy'."
    ) from exc

from benchmarks import DatasetBundle
from configs import TrainingConfig
from model import ModularMLP


@dataclass
class TrainingResult:
    """Sammler fuer Verlaufskurven und finale Testmetriken."""

    history: dict[str, list[float]]
    test_metrics: dict[str, float]


@dataclass(frozen=True)
class EpochUpdate:
    """Zwischenstand nach genau einer Trainings-Epoche."""

    epoch_index: int
    history_snapshot: dict[str, list[float]]
    batch_loss: float
    train_loss: float
    val_loss: float
    train_acc: float
    val_acc: float


def _validate_training_config(config: TrainingConfig) -> None:
    if config.epochs <= 0:
        raise ValueError("epochs muss positiv sein.")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate muss positiv sein.")
    if config.batch_size <= 0:
        raise ValueError("batch_size muss positiv sein.")


def empty_history() -> dict[str, list[float]]:
    return {
        "batch_loss": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }


def copy_history(history: dict[str, list[float]] | None) -> dict[str, list[float]]:
    if history is None:
        return empty_history()
    return {
        "batch_loss": list(history.get("batch_loss", [])),
        "train_loss": list(history.get("train_loss", [])),
        "val_loss": list(history.get("val_loss", [])),
        "train_acc": list(history.get("train_acc", [])),
        "val_acc": list(history.get("val_acc", [])),
    }


def iterate_training_epochs(
    model: ModularMLP,
    dataset: DatasetBundle,
    config: TrainingConfig,
    history: dict[str, list[float]] | None = None,
):
    """Trainiert epochweise und liefert nach jeder Epoche einen Zwischenstand."""

    _validate_training_config(config)
    rng = np.random.default_rng(config.random_state)
    working_history = copy_history(history)

    for epoch_offset in range(config.epochs):
        batch_losses: list[float] = []

        for X_batch, y_batch in iterate_minibatches(
            dataset.X_train,
            dataset.y_train,
            batch_size=config.batch_size,
            rng=rng,
            shuffle=config.shuffle,
        ):
            batch_loss, gradients = model.loss_and_gradients(X_batch, y_batch)
            model.apply_gradients(gradients, config.learning_rate)
            batch_losses.append(batch_loss)

        train_loss, train_acc = model.evaluate(dataset.X_train, dataset.y_train)
        val_loss, val_acc = model.evaluate(dataset.X_val, dataset.y_val)
        batch_mean = float(np.mean(batch_losses))

        working_history["batch_loss"].append(batch_mean)
        working_history["train_loss"].append(train_loss)
        working_history["val_loss"].append(val_loss)
        working_history["train_acc"].append(train_acc)
        working_history["val_acc"].append(val_acc)

        yield EpochUpdate(
            epoch_index=epoch_offset,
            history_snapshot=copy_history(working_history),
            batch_loss=batch_mean,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
        )


def train_model_with_callback(
    model: ModularMLP,
    dataset: DatasetBundle,
    config: TrainingConfig,
    on_epoch,
    history: dict[str, list[float]] | None = None,
) -> TrainingResult:
    """Trainiert das Modell und ruft nach jeder Epoche einen Callback auf."""

    final_history = copy_history(history)
    for update in iterate_training_epochs(model, dataset, config, history=history):
        final_history = update.history_snapshot
        on_epoch(update)
    test_loss, test_acc = model.evaluate(dataset.X_test, dataset.y_test)
    return TrainingResult(
        history=final_history,
        test_metrics={"loss": test_loss, "accuracy": test_acc},
    )


def train_model(
    model: ModularMLP, dataset: DatasetBundle, config: TrainingConfig
) -> TrainingResult:
    """Trainiert das Modell mit Mini-Batch Gradient Descent.

    Pro Epoche passiert:
    1. Trainingsdaten in Batches zerlegen
    2. Fuer jeden Batch: Loss + Gradienten berechnen
    3. Gewichte aktualisieren
    4. Am Ende der Epoche Train- und Val-Metriken messen
    5. Nach allen Epochen final auf dem Test-Split auswerten
    """

    updates = list(iterate_training_epochs(model, dataset, config))
    history = updates[-1].history_snapshot if updates else empty_history()
    test_loss, test_acc = model.evaluate(dataset.X_test, dataset.y_test)
    return TrainingResult(history=history, test_metrics={"loss": test_loss, "accuracy": test_acc})


def iterate_minibatches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool = True,
):
    """Liefert nacheinander Mini-Batches.

    Die Funktion ist absichtlich nicht "zu clever" geschrieben, damit man
    beim Lesen sofort versteht, wo das Shuffling und das Batch-Slicing passiert.

    Wenn man spaeter andere Sampling-Strategien testen will, ist das hier
    eine gute Erweiterungsstelle.
    """

    indices = np.arange(len(y))
    if shuffle:
        indices = rng.permutation(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        yield X[batch_indices], y[batch_indices]
