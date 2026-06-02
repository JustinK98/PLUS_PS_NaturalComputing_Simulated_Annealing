"""ASCII-Ausgaben fuer den Terminalmodus.

Diese Datei ist bewusst getrennt von `main.py`, damit die Darstellung klar von
der eigentlichen Logik getrennt bleibt.

Vorteil:
- die Trainingslogik bleibt schlank
- die Darstellung ist an einer Stelle konzentriert
- spaetere Alternativen (z. B. Rich, curses, Web-UI) koennen leichter ergaenzt werden
"""

from __future__ import annotations

from typing import Iterable

from activations import ACTIVATION_SHORT_NAMES, ActivationLayout, NeighborResult
from benchmarks import DatasetBundle


def render_dataset_summary(bundle: DatasetBundle) -> str:
    """Erzeugt eine kompakte Text-Zusammenfassung des Datensatzes."""

    lines = [
        "Benchmark",
        "=========",
        f"Name:          {bundle.name}",
        f"Input-Features:{bundle.input_size:>5}",
        f"Klassen:       {bundle.output_size:>5}",
        f"Output-Neur.:  {bundle.model_output_size:>5}",
        f"Train/Val/Test:{bundle.train_size:>5}/{bundle.validation_size:>3}/{bundle.test_size:>3}",
    ]
    return "\n".join(lines)


def render_layout(layout: ActivationLayout, title: str = "Layout") -> str:
    """Zeigt ein Aktivierungs-Layout als lesbare ASCII-Tabelle.

    Jeder Eintrag wird als `Index:Kurzname` dargestellt, zum Beispiel `03:REL`.
    """

    lines = [title, "=" * len(title)]

    for layer_index, layer in enumerate(layout.layers):
        lines.append(f"L{layer_index + 1} ({len(layer)} Neuronen)")
        row_tokens = [
            f"{neuron_index:02d}:{ACTIVATION_SHORT_NAMES[activation_name]}"
            for neuron_index, activation_name in enumerate(layer)
        ]
        for chunk in _chunked(row_tokens, width=6):
            lines.append("  " + " | ".join(chunk))

    lines.append(f"Kompakt: {layout.to_compact_spec()}")
    return "\n".join(lines)


def render_neighbor_preview(neighbors: list[NeighborResult], limit: int = 6) -> str:
    """Zeigt eine Vorschau auf Single-Step-Nachbarn eines Layouts.

    Gerade fuer das spaetere Thema Simulated Annealing ist das hilfreich:
    Man sieht sofort, wie ein "naechster Schritt" im Suchraum aussieht.
    """

    lines = ["Single-Step-Neighbors", "====================="]

    if limit <= 0:
        lines.append("Vorschau deaktiviert.")
        return "\n".join(lines)

    if not neighbors:
        lines.append("Keine Nachbarn vorhanden.")
        return "\n".join(lines)

    for index, neighbor in enumerate(neighbors[:limit], start=1):
        change_summary = ", ".join(change.describe() for change in neighbor.changes)
        lines.append(f"{index:02d}. {neighbor.label}")
        lines.append(f"    {neighbor.layout.to_compact_spec()}")
        lines.append(f"    {change_summary}")

    remaining = len(neighbors) - min(limit, len(neighbors))
    if remaining > 0:
        lines.append(f"... {remaining} weitere Nachbarn ausgeblendet.")

    return "\n".join(lines)


def render_training_summary(
    history: dict[str, list[float]], test_metrics: dict[str, float]
) -> str:
    """Erzeugt die wichtigste Trainingszusammenfassung fuer das Terminal."""

    best_epoch = max(range(len(history["val_acc"])), key=lambda idx: history["val_acc"][idx]) + 1
    lines = [
        "Training",
        "========",
        f"Letzte Train-Loss: {history['train_loss'][-1]:.4f}",
        f"Letzte Val-Loss:   {history['val_loss'][-1]:.4f}",
        f"Letzte Train-Acc:  {history['train_acc'][-1]:.4f}",
        f"Letzte Val-Acc:    {history['val_acc'][-1]:.4f}",
        f"Beste Val-Acc:     {max(history['val_acc']):.4f} in Epoche {best_epoch}",
        f"Test-Loss:         {test_metrics['loss']:.4f}",
        f"Test-Acc:          {test_metrics['accuracy']:.4f}",
    ]
    return "\n".join(lines)


def _chunked(items: Iterable[str], width: int) -> list[list[str]]:
    """Zerlegt eine flache Liste in Zeilen fester Breite.

    Beispiel:
    `["a", "b", "c", "d"]` mit `width=2` wird zu
    `[["a", "b"], ["c", "d"]]`.
    """

    chunk: list[str] = []
    rows: list[list[str]] = []

    for item in items:
        chunk.append(item)
        if len(chunk) == width:
            rows.append(chunk)
            chunk = []

    if chunk:
        rows.append(chunk)
    return rows
