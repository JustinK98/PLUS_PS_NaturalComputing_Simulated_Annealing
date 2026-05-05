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
LEGACY_ACTIVATIONS = ("leaky_relu",)
SUPPORTED_GUI_MODES = ("beginner", "expert")
SUPPORTED_GUI_LANGUAGES = ("de", "en")
SUPPORTED_GUI_APP_MODES = (
    "activation_workflow",
    "presentation",
    "demo",
    "playground",
    "experiment_builder",
)


# Standardwerte fuer einen schnellen Einstieg.
# Wenn das Projekt einfach mit `python main.py` gestartet wird, werden genau
# diese Defaults benutzt, sofern der interaktive Assistent nichts anderes setzt.
DEFAULT_BENCHMARK = "concentric_circles"
DEFAULT_LAYOUT = "relu"
DEFAULT_RANDOM_SEED = 42
DEFAULT_GUI_APP_MODE = "activation_workflow"
DEFAULT_EPOCHS = 120
DEFAULT_LEARNING_RATE = 0.03
DEFAULT_BATCH_SIZE = 32
DEFAULT_WEIGHT_SCALE = 0.05
DEFAULT_VALIDATION_SIZE = 0.2
DEFAULT_TEST_SIZE = 0.2
DEFAULT_NEIGHBOR_PREVIEW = 6
DEFAULT_ANNEALING_OBJECTIVE = "validation_loss"
DEFAULT_ANNEALING_CANDIDATE_EPOCHS = 20
DEFAULT_ANNEALING_START_TEMPERATURE = 1.5
DEFAULT_ANNEALING_COOLING_SCHEDULE = "geometric"
DEFAULT_ANNEALING_COOLING_PARAMETER = 0.92
DEFAULT_ANNEALING_ITERATIONS_PER_TEMPERATURE = 5
DEFAULT_ANNEALING_MAX_STEPS = 40
DEFAULT_ANNEALING_MIN_TEMPERATURE = 0.02
DEFAULT_ANNEALING_NEIGHBORHOODS = ("set_neuron", "fill_layer", "swap_neurons")
DEFAULT_EXPERIMENT_OUTPUT_SUBDIR = "experiments"
DEFAULT_EXPERIMENT_SEARCH_TYPE = "none"
DEFAULT_EXPERIMENT_RANDOM_SEARCH_SAMPLES = 8
MIN_HIDDEN_LAYERS = 1
MAX_HIDDEN_LAYERS = 4


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


# Diese Texte werden sowohl in der CLI-Hilfe als auch im interaktiven Assistenten
# verwendet. So bleibt die Erklaerung an genau einer Stelle gepflegt.
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

NEIGHBOR_OPERATION_EXAMPLES = """Neighbor-Operationen:
  set:L1:0:tanh
      setze Neuron 0 in Hidden-Layer 1 auf tanh

  fill:L2:sigmoid
      setze den gesamten zweiten Hidden-Layer auf sigmoid

  cycle:L1:3
      wechsle Neuron 3 in Hidden-Layer 1 zur naechsten Aktivierung

  swap:L2:0:4
      tausche die Aktivierungen zweier Neuronen im zweiten Hidden-Layer

Hinweis:
  Layer-Indizes sind L1, L2, L3, ...
  Neuronen-Indizes sind absichtlich nullbasiert, damit sie direkt zu Python passen.
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
class ModelConfig:
    """Beschreibt die Struktur des MLPs.

    Wichtige Stellschrauben:
    - `hidden_sizes`: Anzahl der Neuronen in den Hidden-Layern
    - `layout_spec`: Welche Aktivierungen pro Layer/Neuron genutzt werden
    - `weight_scale`: Wie "gross" die Startgewichte zufaellig gezogen werden
    """

    input_size: int
    hidden_sizes: tuple[int, ...]
    output_size: int
    layout_spec: str
    weight_scale: float = DEFAULT_WEIGHT_SCALE
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
    hidden_sizes: tuple[int, ...]
    app_mode: str = DEFAULT_GUI_APP_MODE
    layout_spec: str = DEFAULT_LAYOUT
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    batch_size: int = DEFAULT_BATCH_SIZE
    weight_scale: float = DEFAULT_WEIGHT_SCALE
    random_state: int = DEFAULT_RANDOM_SEED
    mode: str = "beginner"
    language: str = "de"


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


def format_hidden_sizes(hidden_sizes: tuple[int, ...]) -> str:
    """Formatiert eine Folge von Hidden-Sizes kompakt fuer Anzeige und Logs."""

    return " / ".join(str(size) for size in hidden_sizes)
