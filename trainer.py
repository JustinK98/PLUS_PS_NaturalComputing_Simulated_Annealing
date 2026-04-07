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

    if config.epochs <= 0:
        raise ValueError("epochs muss positiv sein.")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate muss positiv sein.")
    if config.batch_size <= 0:
        raise ValueError("batch_size muss positiv sein.")

    rng = np.random.default_rng(config.random_state)

    # `history` speichert Verlaufskurven fuer spaetere ASCII- und Plot-Ausgaben.
    history = {
        "batch_loss": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for _epoch in range(config.epochs):
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

        # Nach jeder Epoche messen wir direkt, wie das Modell auf Train und
        # Validation steht. Das ist fuer Lernkurven und Fehlersuche hilfreich.
        train_loss, train_acc = model.evaluate(dataset.X_train, dataset.y_train)
        val_loss, val_acc = model.evaluate(dataset.X_val, dataset.y_val)

        history["batch_loss"].append(float(np.mean(batch_losses)))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

    test_loss, test_acc = model.evaluate(dataset.X_test, dataset.y_test)
    return TrainingResult(
        history=history,
        test_metrics={"loss": test_loss, "accuracy": test_acc},
    )


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
