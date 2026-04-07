"""Matplotlib-Visualisierung fuer Lernkurven und Aktivierungs-Layouts.

Wichtige Hinweise fuer Studierende:

- Diese Datei enthaelt nur Darstellung, keine Trainingslogik.
- Farben und Plot-Stile koennen hier zentral angepasst werden.
- Falls man spaeter andere Diagramme moechte, ist dies die richtige Datei.

Zum Backend:
- Standardmaessig wird hier `Agg` gesetzt.
- Das ist robust fuer PNG-Erzeugung in headless oder eingeschraenkten Umgebungen.
- Wer zwingend interaktive GUI-Fenster moechte, kann diese Stelle spaeter bewusst
  anpassen und ein anderes Matplotlib-Backend verwenden.
"""

from __future__ import annotations

from pathlib import Path

try:
    import numpy as np
except ImportError as exc:
    raise ImportError(
        "Dieses Projekt benoetigt NumPy. Installation z. B. mit 'pip install numpy'."
    ) from exc

try:
    import matplotlib

    # `Agg` erzeugt Bilddateien ohne GUI-Fenster.
    # Das ist fuer diese Lernumgebung die stabilste Standardwahl.
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch, Rectangle
except ImportError as exc:
    raise ImportError(
        "Dieses Projekt benoetigt matplotlib. Installation z. B. mit "
        "'pip install matplotlib'."
    ) from exc

from activations import (
    ACTIVATION_COLORS,
    ACTIVATION_SHORT_NAMES,
    ActivationLayout,
    diff_layouts,
)
from configs import OUTPUT_DIR, SUPPORTED_ACTIVATIONS


def plot_training_history(history: dict[str, list[float]], title: str):
    """Zeichnet die Lernkurven fuer Loss und Accuracy.

    Das Diagramm ist bewusst simpel gehalten:
    - links: Train- und Validation-Loss
    - rechts: Train- und Validation-Accuracy

    Dadurch sehen Studierende schnell, ob das Modell lernt oder ueberfitten koennte.
    """

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="Train", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Validation", linewidth=2)
    axes[0].set_title(f"Loss auf {title}")
    axes[0].set_xlabel("Epoche")
    axes[0].set_ylabel("Cross-Entropy-Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], label="Validation", linewidth=2)
    axes[1].set_title(f"Accuracy auf {title}")
    axes[1].set_xlabel("Epoche")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    return figure


def plot_layouts(
    base_layout: ActivationLayout,
    neighbor_layout: ActivationLayout | None = None,
):
    """Visualisiert ein Layout oder ein Basis/Nachbar-Paar als Heatmap.

    Jede Zelle steht fuer genau ein Neuron in einem Hidden-Layer.
    Die Farbe codiert die Aktivierungsfunktion.
    """

    height = max(3.8, 2.0 + 0.8 * len(base_layout.layers))
    if neighbor_layout is None:
        figure, axes = plt.subplots(1, 1, figsize=(9, height))
        axes = [axes]
        titles = [("Trainiertes Layout", base_layout)]
    else:
        figure, axes = plt.subplots(1, 2, figsize=(13, max(4.2, height)), sharey=True)
        titles = [("Basis-Layout", base_layout), ("Nachbar-Layout", neighbor_layout)]

    # Ein fester Farbcode macht Vergleiche ueber Runs hinweg leichter.
    cmap = ListedColormap([ACTIVATION_COLORS[name] for name in SUPPORTED_ACTIVATIONS])
    norm = BoundaryNorm(np.arange(-0.5, len(SUPPORTED_ACTIVATIONS) + 0.5, 1), cmap.N)

    for axis, (panel_title, layout) in zip(axes, titles, strict=True):
        matrix = _layout_to_matrix(layout)

        # Kuerzere Layer werden mit NaN aufgefuellt. Diese Felder maskieren wir,
        # damit sie nicht als "falsche" Aktivierung erscheinen.
        masked_matrix = np.ma.masked_invalid(matrix)

        axis.imshow(masked_matrix, aspect="auto", cmap=cmap, norm=norm)
        axis.set_title(panel_title)
        axis.set_xlabel("Neuron-Index")
        axis.set_ylabel("Hidden-Layer")
        axis.set_yticks(
            np.arange(len(layout.layers)),
            labels=[f"L{layer_index + 1}" for layer_index in range(len(layout.layers))],
        )
        axis.set_xticks(np.arange(matrix.shape[1]))

        # Kurznamen wie REL, TAN, SIG direkt in die Zellen schreiben.
        for layer_index, layer in enumerate(layout.layers):
            for neuron_index, activation_name in enumerate(layer):
                axis.text(
                    neuron_index,
                    layer_index,
                    ACTIVATION_SHORT_NAMES[activation_name],
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                    fontweight="bold",
                )

    if neighbor_layout is not None:
        # Unterschiede werden im rechten Plot schwarz umrandet.
        for change in diff_layouts(base_layout, neighbor_layout):
            axes[1].add_patch(
                Rectangle(
                    (change.neuron_index - 0.5, change.layer_index - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor="black",
                    linewidth=2,
                )
            )

    legend_handles = [
        Patch(facecolor=ACTIVATION_COLORS[name], label=name) for name in SUPPORTED_ACTIVATIONS
    ]
    figure.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    return figure


def save_figure(figure, path_like: str | Path) -> Path:
    """Speichert eine Matplotlib-Figure und liefert den absoluten Pfad zurueck."""

    path = Path(path_like)
    if not path.is_absolute():
        path = OUTPUT_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    return path.resolve()


def _layout_to_matrix(layout: ActivationLayout) -> np.ndarray:
    """Wandelt ein Layout in eine 2D-Matrix fuer `imshow` um.

    Rueckgabeformat:
    - jede Zeile: ein Hidden-Layer
    - Spalten: Neuronenposition
    """

    max_neurons = max(layout.hidden_sizes)
    matrix = np.full((len(layout.layers), max_neurons), np.nan, dtype=np.float64)

    for layer_index, layer in enumerate(layout.layers):
        for neuron_index, activation_name in enumerate(layer):
            matrix[layer_index, neuron_index] = SUPPORTED_ACTIVATIONS.index(activation_name)

    return matrix
