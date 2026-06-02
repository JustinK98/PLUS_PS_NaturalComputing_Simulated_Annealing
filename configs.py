"""Zentrale Konfiguration des Playground-Projekts.

Diese Datei ist bewusst klein und sehr wichtig:

1. Hier liegen die Standardwerte, die beim Start ohne viele Angaben verwendet werden.
2. Hier stehen die "erlaubten" Namen fuer Benchmarks und Aktivierungen.
3. Hier sind kleine Dataclasses gesammelt, damit andere Module mit klaren,
   gut lesbaren Konfigurationsobjekten arbeiten koennen.

Wenn Studierende spaeter mit dem Projekt experimentieren wollen, ist dies oft
die erste Datei, die man sich ansehen sollte:

- andere Default-Epochen -> `DEFAULT_EPOCHS`
- andere Lernrate -> `DEFAULT_LEARNING_RATE`
- andere Standard-Hidden-Sizes -> `DEFAULT_HIDDEN_SIZES_BY_BENCHMARK`
- andere Ausgabeordner -> `OUTPUT_DIR`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from benchmark_registry import OFFICIAL_BENCHMARK_SPECS, OFFICIAL_BENCHMARKS


# Diese Tupel definieren die "offiziell" unterstuetzten Namen im Projekt.
# Der Parser und die Validierung greifen mehrfach darauf zu.
SUPPORTED_BENCHMARKS = OFFICIAL_BENCHMARKS
INTERNAL_BENCHMARKS = ("test_activation",)
ALL_BENCHMARKS = (*SUPPORTED_BENCHMARKS, *INTERNAL_BENCHMARKS)
SUPPORTED_ACTIVATIONS = ("relu", "gelu", "sigmoid", "tanh", "swish", "identity")


# Standardwerte fuer einen schnellen Einstieg.
# Diese Defaults werden fuer schnelle CLI-Laeufe und die GUI-Demo verwendet.
DEFAULT_BENCHMARK = "two_moons"
DEFAULT_LAYOUT = "relu"
DEFAULT_RANDOM_SEED = 42
DEFAULT_EPOCHS = 120
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_BATCH_SIZE = 32
DEFAULT_WEIGHT_SCALE = 1.0
DEFAULT_VALIDATION_SIZE = 0.2
DEFAULT_TEST_SIZE = 0.2
DEFAULT_NEIGHBOR_PREVIEW = 6
DEFAULT_ANNEALING_START_TEMPERATURE = 0.03
DEFAULT_ANNEALING_COOLING_SCHEDULE = "geometric"
DEFAULT_ANNEALING_COOLING_PARAMETER = 0.95
DEFAULT_ANNEALING_ITERATIONS_PER_TEMPERATURE = 5
DEFAULT_ANNEALING_MAX_STEPS = 120
DEFAULT_ANNEALING_MIN_TEMPERATURE = 0.001
DEFAULT_ANNEALING_NEIGHBORHOODS = ("set_neuron",)


# Kleine, bewusst gut lesbare Architekturen fuer die drei Datensaetze.
# Die Werte sind nicht "optimal", sondern didaktisch handhabbar.
DEFAULT_HIDDEN_SIZES_BY_BENCHMARK = {
    name: spec.hidden_sizes for name, spec in OFFICIAL_BENCHMARK_SPECS.items()
} | {
    "test_activation": (4, 3),
}

DEFAULT_EPOCHS_BY_BENCHMARK = {
    name: spec.epochs for name, spec in OFFICIAL_BENCHMARK_SPECS.items()
} | {"test_activation": DEFAULT_EPOCHS}


# Alle automatisch gespeicherten Plots landen unterhalb dieses Ordners.
OUTPUT_DIR = Path("outputs")


# Diese Texte werden fuer die CLI-Hilfe zentral gepflegt.
LAYOUT_SYNTAX_EXAMPLES = """Layout-Syntax:
  relu|tanh
      gesamte Layer: Hidden-Layer 1 nutzt relu, Hidden-Layer 2 nutzt tanh

  relu*16|tanh*8
      gleiche Idee, aber mit expliziter Neuron-Anzahl

  relu*8,tanh*8|sigmoid*4,swish*4
      Mischung innerhalb eines Layers

  relu,relu,tanh,sigmoid|swish*4
      voll explizite Belegung einzelner Neuronen

  relu|tanh|sigmoid
      Beispiel mit drei Hidden-Layern
"""

@dataclass(frozen=True)
class DatasetConfig:
    """Konfiguration fuer das Laden und Aufteilen eines Datensatzes.

    `validation_size` und `test_size` sind Anteile zwischen 0 und 1.
    `random_state` sorgt dafuer, dass Splits reproduzierbar bleiben.
    """

    name: str
    validation_size: float = DEFAULT_VALIDATION_SIZE
    test_size: float = DEFAULT_TEST_SIZE
    random_state: int = DEFAULT_RANDOM_SEED


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameter fuer das Training.

    Diese Werte sind die klassische Stelle fuer Experimente:
    - mehr Epochen -> oft besseres Lernen, aber laenger
    - andere Lernrate -> kann Training stabiler oder schneller machen
    - andere Batch-Groesse -> beeinflusst Rauschen und Laufzeit
    """

    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    batch_size: int = DEFAULT_BATCH_SIZE
    random_state: int = DEFAULT_RANDOM_SEED
    shuffle: bool = True


@dataclass(frozen=True)
class VisualizationConfig:
    """Steuert Terminal- und Plot-Ausgabe.

    - `preview_neighbors`: Wie viele Nachbarn im Terminal gezeigt werden
    - `show_plots`: Ob Matplotlib-Fenster gezeigt werden sollen
    - `save_prefix`: Falls gesetzt, werden PNG-Dateien gespeichert
    """

    preview_neighbors: int = DEFAULT_NEIGHBOR_PREVIEW
    show_plots: bool = True
    save_prefix: str | None = None


@dataclass(frozen=True)
class GuiExperimentConfig:
    """Konfigurationsobjekt fuer den Start der Qt-GUI."""

    benchmark: str
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    batch_size: int = DEFAULT_BATCH_SIZE
    weight_scale: float = DEFAULT_WEIGHT_SCALE
    random_state: int = DEFAULT_RANDOM_SEED
    mode: str = "beginner"
    language: str = "de"
    gui_profile: str = "demo"


def default_hidden_sizes(benchmark_name: str) -> tuple[int, ...]:
    """Liefert die Standard-Hidden-Sizes fuer einen Benchmark.

    Diese Hilfsfunktion kapselt den Zugriff auf die Default-Tabelle. Dadurch
    muss der Rest des Codes nicht direkt mit dem Dictionary arbeiten.
    """

    try:
        return DEFAULT_HIDDEN_SIZES_BY_BENCHMARK[benchmark_name]
    except KeyError as exc:
        supported = ", ".join(ALL_BENCHMARKS)
        raise ValueError(
            f"Unbekannter Benchmark '{benchmark_name}'. Erlaubt sind: {supported}"
        ) from exc

def default_epochs(benchmark_name: str) -> int:
    """Liefert die standardmaessige Trainingsdauer fuer einen Benchmark."""

    try:
        return DEFAULT_EPOCHS_BY_BENCHMARK[benchmark_name]
    except KeyError as exc:
        supported = ", ".join(ALL_BENCHMARKS)
        raise ValueError(
            f"Unbekannter Benchmark '{benchmark_name}'. Erlaubt sind: {supported}"
        ) from exc
