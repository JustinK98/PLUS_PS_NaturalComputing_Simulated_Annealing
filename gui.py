"""Didaktische GUI fuer den Aktivierungs-Playground.

Diese GUI soll nicht nur Einstellungen setzen, sondern den gesamten Ablauf
eines kleinen neuronalen Netzes sichtbar und begreifbar machen:

- Welche Daten fliessen gerade ins Netz?
- Was ist Zielwert und was ist Vorhersage?
- Wie sehen Architektur und Aktivierungen aus?
- Was aendert Training ueber die Zeit?
- Was passiert lokal in einem einzelnen Neuron?

Die Oberflaeche ist bewusst als Lernwerkzeug gebaut:
- links: gefuehrter Kontrollbereich
- rechts oben: Netzwerk mit Kreisen und Verbindungen
- rechts unten: Tabs fuer Input-Ansicht, Trainingsplots, Neuron-Details und Lernhilfe
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import os
from typing import Any, Sequence

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
    from tkinter.scrolledtext import ScrolledText
except ImportError as exc:
    raise ImportError(
        "Dieses Projekt benoetigt tkinter fuer die GUI. "
        "Auf vielen Python-Installationen ist es bereits enthalten."
    ) from exc

try:
    import numpy as np
except ImportError as exc:
    raise ImportError(
        "Dieses Projekt benoetigt NumPy. Installation z. B. mit 'pip install numpy'."
    ) from exc

try:
    os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".mplconfig"))
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except ImportError as exc:
    raise ImportError(
        "Dieses Projekt benoetigt matplotlib fuer die eingebetteten GUI-Plots. "
        "Installation z. B. mit 'pip install matplotlib'."
    ) from exc

from activations import (
    ACTIVATION_COLORS,
    ACTIVATION_FORMULAS,
    ACTIVATIONS,
    ActivationLayout,
    diff_layouts,
    generate_single_step_neighbors,
    parse_layout_spec,
)
from benchmarks import DatasetBundle, describe_dataset, load_benchmark
from configs import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LAYOUT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_WEIGHT_SCALE,
    DatasetConfig,
    MAX_HIDDEN_LAYERS,
    MIN_HIDDEN_LAYERS,
    SUPPORTED_ACTIVATIONS,
    SUPPORTED_BENCHMARKS,
    SUPPORTED_GUI_MODES,
    TrainingConfig,
    default_hidden_sizes,
    format_hidden_sizes,
)
from model import ModularMLP, NeuronInspection, SampleTrace
from trainer import TrainingResult, train_model


BACKGROUND_COLOR = "#f4f7fb"
PANEL_COLOR = "#ffffff"
# Das Netzwerkfeld bleibt bewusst hell, damit Labels und Knoten klar lesbar sind.
NETWORK_BACKGROUND_COLOR = "#edf3f8"
NETWORK_TEXT_COLOR = "#000000"
INPUT_NODE_COLOR = "#cbd5e1"
OUTPUT_NODE_COLOR = "#1e293b"
# Die Basisverbindungen sind etwas dunkler als der Hintergrund, damit die
# Struktur des Netzes auch ohne aktive Hervorhebungen gut erkennbar bleibt.
CONNECTION_COLOR = "#9fb0c2"
POSITIVE_CONNECTION_COLOR = "#2563eb"
NEGATIVE_CONNECTION_COLOR = "#dc2626"
SELECTED_OUTLINE_COLOR = "#111827"
TARGET_COLOR = "#d97706"
PREDICTION_COLOR = "#16a34a"
ANALYSIS_TARGET_COLOR = "#0f766e"

TARGET_AUTO_LABEL = "(echtes Ziel)"
TARGET_NONE_LABEL = "(kein Ziel)"
MANUAL_DIGIT_VALUES = (0.0, 4.0, 8.0, 12.0, 16.0)

BENCHMARK_DESCRIPTIONS = {
    "breast_cancer": (
        "Der Benchmark breast_cancer ist eine binaere Klassifikation mit 30 numerischen "
        "Merkmalen aus Zellkernmessungen. Er ist gut fuer erste Experimente, weil das Netz "
        "nur zwischen zwei Klassen unterscheiden muss und Lernkurven oft schnell lesbar werden."
    ),
    "wine": (
        "Der Benchmark wine ist eine Mehrklassen-Klassifikation mit 13 chemischen Merkmalen. "
        "Er eignet sich besonders gut fuer Layout-Vergleiche, weil er klein, ueberschaubar "
        "und dennoch nicht trivial ist."
    ),
    "digits": (
        "Der Benchmark digits enthaelt 8x8 Grauwertbilder von Ziffern und hat 10 Klassen. "
        "Er ist didaktisch wertvoll, weil man Eingabe, Ziel und Vorhersage visuell als Bild "
        "diskutieren kann."
    ),
    "test_activation": (
        "test_activation ist kein echter Benchmark, sondern ein Lernlabor mit drei Inputs. "
        "Hier kann man rohe Vorwaertsrechnungen, Aktivierungsfunktionen und lokale Unterschiede "
        "im Netz besonders klar untersuchen."
    ),
}


@dataclass(frozen=True)
class GuiExperimentConfig:
    """Konfigurationsobjekt fuer den Start der GUI."""

    benchmark: str
    hidden_sizes: tuple[int, ...]
    layout_spec: str = DEFAULT_LAYOUT
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    batch_size: int = DEFAULT_BATCH_SIZE
    weight_scale: float = DEFAULT_WEIGHT_SCALE
    random_state: int = DEFAULT_RANDOM_SEED
    mode: str = "beginner"


@dataclass(frozen=True)
class AnalysisSample:
    """Beschreibt das aktuell in der GUI untersuchte Beispiel."""

    raw_sample: np.ndarray
    scaled_sample: np.ndarray
    actual_target_index: int | None
    effective_target_index: int | None
    actual_target_name: str
    effective_target_name: str
    split_name: str
    sample_index: int
    source_label: str
    is_custom: bool


def launch_playground_gui(config: GuiExperimentConfig) -> None:
    """Startet die GUI und blockiert bis zum Schliessen des Fensters."""

    root = tk.Tk()
    PlaygroundGUI(root, config)
    root.mainloop()


class PlaygroundGUI:
    """Tkinter-Oberflaeche fuer interaktive Aktivierungs-Experimente."""

    def __init__(self, root: tk.Tk, config: GuiExperimentConfig) -> None:
        self.root = root
        self.root.title("Activation Playground GUI")
        self.root.geometry("1620x1040")
        self.root.minsize(1360, 860)
        self.root.configure(bg=BACKGROUND_COLOR)

        self._configure_styles()

        self.dataset: DatasetBundle | None = None
        self.model: ModularMLP | None = None
        self.current_layout: ActivationLayout | None = None
        self.training_result: TrainingResult | None = None
        self.baseline_model: ModularMLP | None = None
        self.baseline_layout: ActivationLayout | None = None
        self.baseline_training_result: TrainingResult | None = None
        self.baseline_completed_epochs = 0
        self.completed_epochs = 0
        self.selected_hidden: tuple[int, int] = (0, 0)
        self.node_tags: dict[str, tuple[int, int]] = {}
        self.raw_custom_digit = np.zeros(64, dtype=np.float64)
        self.step_entries: list[tuple[str, str]] = []
        self.layer_size_controls_frame: ttk.Frame | None = None
        self.layer_fill_controls_frame: ttk.Frame | None = None
        self.hidden_size_spinboxes: list[ttk.Spinbox] = []
        self.hidden_layer_size_labels: list[ttk.Label] = []
        self.layer_fill_combos: list[ttk.Combobox] = []
        self.expert_only_widgets: list[Any] = []
        self.step_index_var = tk.IntVar(value=0)

        self.mode_var = tk.StringVar(value=config.mode)
        self.benchmark_var = tk.StringVar(value=config.benchmark)
        self.hidden_size_vars = [tk.IntVar(value=size) for size in config.hidden_sizes]
        self.split_var = tk.StringVar(value="train")
        self.sample_index_var = tk.IntVar(value=0)
        self.analysis_target_var = tk.StringVar(value=TARGET_AUTO_LABEL)
        self.use_custom_sample_var = tk.BooleanVar(value=config.benchmark == "test_activation")

        self.layout_string_var = tk.StringVar(value=config.layout_spec)
        self.neuron_layer_var = tk.StringVar(value="L1")
        self.neuron_index_var = tk.StringVar(value="0")
        self.neuron_activation_var = tk.StringVar(value="relu")
        self.layer_fill_vars = [tk.StringVar(value="relu") for _ in config.hidden_sizes]

        self.epochs_var = tk.IntVar(value=config.epochs)
        self.lr_var = tk.StringVar(value=str(config.learning_rate))
        self.batch_size_var = tk.IntVar(value=config.batch_size)
        self.weight_scale_var = tk.StringVar(value=str(config.weight_scale))
        self.seed_var = tk.IntVar(value=config.random_state)

        self.dataset_summary_var = tk.StringVar(value="")
        self.prediction_summary_var = tk.StringVar(value="Noch kein Sample berechnet.")
        self.metrics_summary_var = tk.StringVar(value="Noch kein Training durchgefuehrt.")
        self.workflow_summary_var = tk.StringVar(value="")
        self.context_hint_var = tk.StringVar(value="")
        self.sample_summary_var = tk.StringVar(value="")
        self.sample_hint_var = tk.StringVar(value="")
        self.network_summary_var = tk.StringVar(value="")
        self.compare_summary_var = tk.StringVar(value="Noch keine Baseline gespeichert.")
        self.layer_summary_var = tk.StringVar(value="")

        self.test_input_vars = [tk.DoubleVar(value=0.0) for _ in range(3)]

        self._build_layout()
        self._load_experiment(reinitialize_model=True)

    def _configure_styles(self) -> None:
        """Setzt ein ruhiges, gut lesbares GUI-Theme."""

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TLabelframe", background=PANEL_COLOR)
        style.configure("TLabelframe.Label", background=PANEL_COLOR, font=("Helvetica", 11, "bold"))
        style.configure("TFrame", background=BACKGROUND_COLOR)
        style.configure("White.TFrame", background=PANEL_COLOR)
        style.configure("Hint.TLabel", background=PANEL_COLOR, foreground="#334155", wraplength=320)
        style.configure("SectionValue.TLabel", background=PANEL_COLOR, foreground="#0f172a")
        style.configure("Headline.TLabel", background=PANEL_COLOR, font=("Helvetica", 12, "bold"))
        style.configure("Notebook.TFrame", background=PANEL_COLOR)

    def _build_layout(self) -> None:
        """Baut die Hauptstruktur des Fensters auf."""

        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=0)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        controls = ttk.Frame(outer, style="White.TFrame")
        controls.grid(row=0, column=0, sticky="ns", padx=(0, 12))

        content = ttk.Frame(outer, style="White.TFrame")
        content.grid(row=0, column=1, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=0)
        content.rowconfigure(1, weight=3)
        content.rowconfigure(2, weight=2)

        self._build_controls(controls)
        self._build_content(content)
        self._apply_mode_visibility()

    def _build_controls(self, parent: ttk.Frame) -> None:
        """Linke Seitenleiste mit Assistent, Layout, Training und Vergleich."""

        mode_frame = ttk.LabelFrame(parent, text="Modus", padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(
            mode_frame,
            text=(
                "Im Einsteiger-Modus bleiben nur die wichtigsten Schritte sichtbar. "
                "Im Experten-Modus erscheinen variable Layer, Layout-Eingriffe und "
                "Vergleichswerkzeuge."
            ),
            style="Hint.TLabel",
            justify=tk.LEFT,
        ).pack(anchor="w")
        mode_combo = ttk.Combobox(
            mode_frame,
            textvariable=self.mode_var,
            values=SUPPORTED_GUI_MODES,
            state="readonly",
            width=12,
        )
        mode_combo.pack(anchor="w", pady=(8, 0))
        mode_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_mode_changed())

        workflow_frame = ttk.LabelFrame(parent, text="Lernfluss", padding=10)
        workflow_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(
            workflow_frame,
            textvariable=self.workflow_summary_var,
            justify=tk.LEFT,
            style="SectionValue.TLabel",
            wraplength=330,
        ).pack(anchor="w")
        ttk.Label(
            workflow_frame,
            textvariable=self.context_hint_var,
            justify=tk.LEFT,
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(8, 0))

        preset_frame = ttk.LabelFrame(parent, text="Schnellstart-Presets", padding=10)
        preset_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(
            preset_frame,
            text="Presets setzen sinnvolle Startkonfigurationen, damit man direkt etwas sieht.",
            style="Hint.TLabel",
            justify=tk.LEFT,
        ).pack(anchor="w")
        preset_row_1 = ttk.Frame(preset_frame, style="White.TFrame")
        preset_row_1.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(preset_row_1, text="Wine Demo", command=lambda: self._apply_preset("wine_demo")).pack(side=tk.LEFT)
        ttk.Button(preset_row_1, text="Digits Demo", command=lambda: self._apply_preset("digits_demo")).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(preset_row_1, text="Nur ReLU", command=lambda: self._apply_preset("all_relu")).pack(side=tk.LEFT, padx=(8, 0))
        preset_row_2 = ttk.Frame(preset_frame, style="White.TFrame")
        preset_row_2.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(preset_row_2, text="Gemischt", command=lambda: self._apply_preset("mixed")).pack(side=tk.LEFT)
        ttk.Button(preset_row_2, text="Test Activation", command=lambda: self._apply_preset("test_activation")).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(preset_row_2, text="Reset", command=self._reset_to_defaults).pack(side=tk.LEFT, padx=(8, 0))

        experiment_frame = ttk.LabelFrame(parent, text="1. Daten und Beispiel", padding=10)
        experiment_frame.pack(fill=tk.X, pady=(0, 10))
        experiment_frame.columnconfigure(1, weight=1)
        ttk.Label(
            experiment_frame,
            text=(
                "Hier legst du fest, welcher Datensatz trainiert wird und welches "
                "einzelne Sample du im Netzwerk gerade beobachtest."
            ),
            style="Hint.TLabel",
            justify=tk.LEFT,
        ).grid(row=0, column=0, columnspan=2, sticky="ew")

        ttk.Label(experiment_frame, text="Benchmark").grid(row=1, column=0, sticky="w", pady=(10, 0))
        benchmark_combo = ttk.Combobox(
            experiment_frame,
            textvariable=self.benchmark_var,
            values=SUPPORTED_BENCHMARKS,
            state="readonly",
            width=18,
        )
        benchmark_combo.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(10, 0))
        benchmark_combo.bind("<<ComboboxSelected>>", self._on_benchmark_changed)

        self.hidden_layers_label = ttk.Label(experiment_frame, text="Hidden-Layer")
        self.hidden_layers_label.grid(row=2, column=0, sticky="nw", pady=(6, 0))
        self.layer_size_controls_frame = ttk.Frame(experiment_frame, style="White.TFrame")
        self.layer_size_controls_frame.grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))

        self.hidden_layer_buttons = ttk.Frame(experiment_frame, style="White.TFrame")
        self.hidden_layer_buttons.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.add_layer_button = ttk.Button(
            self.hidden_layer_buttons,
            text="Layer hinzufuegen",
            command=self._add_hidden_layer,
        )
        self.add_layer_button.pack(side=tk.LEFT)
        self.remove_layer_button = ttk.Button(
            self.hidden_layer_buttons,
            text="Letzten Layer loeschen",
            command=self._remove_hidden_layer,
        )
        self.remove_layer_button.pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(experiment_frame, text="Datensplit").grid(row=4, column=0, sticky="w", pady=(6, 0))
        split_combo = ttk.Combobox(
            experiment_frame,
            textvariable=self.split_var,
            values=("train", "val", "test"),
            state="readonly",
            width=10,
        )
        split_combo.grid(row=4, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        split_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_split_changed())

        ttk.Label(experiment_frame, text="Sample-Index").grid(row=5, column=0, sticky="w", pady=(6, 0))
        self.sample_spinbox = ttk.Spinbox(
            experiment_frame,
            from_=0,
            to=0,
            textvariable=self.sample_index_var,
            width=8,
            command=self._refresh_views,
        )
        self.sample_spinbox.grid(row=5, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        self.sample_spinbox.bind("<Return>", lambda _event: self._refresh_views())

        self.use_custom_sample_checkbutton = ttk.Checkbutton(
            experiment_frame,
            text="Eigenes Sample verwenden",
            variable=self.use_custom_sample_var,
            command=self._on_custom_sample_toggled,
        )
        self.use_custom_sample_checkbutton.grid(row=6, column=0, columnspan=2, sticky="w", pady=(10, 0))

        self.analysis_target_label = ttk.Label(experiment_frame, text="Analyse-Ziel")
        self.analysis_target_label.grid(row=7, column=0, sticky="w", pady=(6, 0))
        self.analysis_target_combo = ttk.Combobox(
            experiment_frame,
            textvariable=self.analysis_target_var,
            state="readonly",
            width=18,
        )
        self.analysis_target_combo.grid(row=7, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        self.analysis_target_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_views())

        experiment_button_row = ttk.Frame(experiment_frame, style="White.TFrame")
        experiment_button_row.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Button(experiment_button_row, text="Benchmark-Defaults", command=self._apply_benchmark_defaults).pack(side=tk.LEFT)
        ttk.Button(experiment_button_row, text="Experiment laden", command=self._reload_experiment).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(experiment_button_row, text="Zufaelliges Sample", command=self._choose_random_sample).pack(side=tk.LEFT, padx=(8, 0))

        layout_frame = ttk.LabelFrame(parent, text="2. Architektur und Aktivierungen", padding=10)
        layout_frame.pack(fill=tk.X, pady=(0, 10))
        layout_frame.columnconfigure(1, weight=1)
        ttk.Label(
            layout_frame,
            text=(
                "Hier veraenderst du die Aktivierungen. Das aendert nicht den Datensatz, "
                "sondern nur die Rechenweise der Hidden-Neuronen."
            ),
            style="Hint.TLabel",
            justify=tk.LEFT,
        ).grid(row=0, column=0, columnspan=3, sticky="ew")

        ttk.Label(layout_frame, text="Aktuelles Layout").grid(row=1, column=0, sticky="nw", pady=(10, 0))
        ttk.Label(
            layout_frame,
            textvariable=self.layout_string_var,
            wraplength=300,
            justify=tk.LEFT,
            style="SectionValue.TLabel",
        ).grid(row=1, column=1, columnspan=2, sticky="w", padx=(8, 0), pady=(10, 0))

        ttk.Label(layout_frame, text="Layer-Uebersicht").grid(row=2, column=0, sticky="nw", pady=(6, 0))
        ttk.Label(
            layout_frame,
            textvariable=self.layer_summary_var,
            wraplength=300,
            justify=tk.LEFT,
            style="Hint.TLabel",
        ).grid(row=2, column=1, columnspan=2, sticky="w", padx=(8, 0), pady=(6, 0))

        self.layer_fill_controls_frame = ttk.Frame(layout_frame, style="White.TFrame")
        self.layer_fill_controls_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        self.neuron_layer_label = ttk.Label(layout_frame, text="Layer")
        self.neuron_layer_label.grid(row=4, column=0, sticky="w", pady=(12, 0))
        self.neuron_layer_combo = ttk.Combobox(
            layout_frame,
            textvariable=self.neuron_layer_var,
            values=("L1",),
            state="readonly",
            width=8,
        )
        self.neuron_layer_combo.grid(row=4, column=1, sticky="w", padx=(8, 0), pady=(12, 0))
        self.neuron_layer_combo.bind("<<ComboboxSelected>>", self._on_neuron_layer_changed)

        self.neuron_index_label = ttk.Label(layout_frame, text="Neuron")
        self.neuron_index_label.grid(row=5, column=0, sticky="w", pady=(6, 0))
        self.neuron_index_combo = ttk.Combobox(
            layout_frame,
            textvariable=self.neuron_index_var,
            values=("0",),
            state="readonly",
            width=8,
        )
        self.neuron_index_combo.grid(row=5, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        self.neuron_index_combo.bind("<<ComboboxSelected>>", lambda _event: self._select_from_controls())

        self.neuron_activation_label = ttk.Label(layout_frame, text="Aktivierung")
        self.neuron_activation_label.grid(row=6, column=0, sticky="w", pady=(6, 0))
        self.neuron_activation_combo = ttk.Combobox(
            layout_frame,
            textvariable=self.neuron_activation_var,
            values=SUPPORTED_ACTIVATIONS,
            state="readonly",
            width=14,
        )
        self.neuron_activation_combo.grid(row=6, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        self.neuron_button_row = ttk.Frame(layout_frame, style="White.TFrame")
        self.neuron_button_row.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        ttk.Button(self.neuron_button_row, text="Neuron setzen", command=self._apply_neuron_setting).pack(side=tk.LEFT)
        ttk.Button(self.neuron_button_row, text="Cycle", command=self._cycle_selected_neuron).pack(side=tk.LEFT, padx=(8, 0))

        training_frame = ttk.LabelFrame(parent, text="3. Training und Beobachtung", padding=10)
        training_frame.pack(fill=tk.X, pady=(0, 10))
        training_frame.columnconfigure(1, weight=1)
        ttk.Label(
            training_frame,
            text=(
                "Training aendert Gewichte und Biases. Das aktuelle Sample dient nur zur Analyse; "
                "gelernt wird immer auf dem gesamten Trainingssplit."
            ),
            style="Hint.TLabel",
            justify=tk.LEFT,
        ).grid(row=0, column=0, columnspan=2, sticky="ew")

        self.epochs_label = ttk.Label(training_frame, text="Epochen")
        self.epochs_label.grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.epochs_spinbox = ttk.Spinbox(training_frame, from_=1, to=5000, textvariable=self.epochs_var, width=8)
        self.epochs_spinbox.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(10, 0))

        self.lr_label = ttk.Label(training_frame, text="Lernrate")
        self.lr_label.grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.lr_entry = ttk.Entry(training_frame, textvariable=self.lr_var, width=10)
        self.lr_entry.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        self.batch_label = ttk.Label(training_frame, text="Batch-Groesse")
        self.batch_label.grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.batch_spinbox = ttk.Spinbox(training_frame, from_=1, to=4096, textvariable=self.batch_size_var, width=8)
        self.batch_spinbox.grid(row=3, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        self.weight_scale_label = ttk.Label(training_frame, text="Weight-Scale")
        self.weight_scale_label.grid(row=4, column=0, sticky="w", pady=(6, 0))
        self.weight_scale_entry = ttk.Entry(training_frame, textvariable=self.weight_scale_var, width=10)
        self.weight_scale_entry.grid(row=4, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        self.seed_label = ttk.Label(training_frame, text="Seed")
        self.seed_label.grid(row=5, column=0, sticky="w", pady=(6, 0))
        self.seed_spinbox = ttk.Spinbox(training_frame, from_=0, to=999999, textvariable=self.seed_var, width=8)
        self.seed_spinbox.grid(row=5, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        training_button_row_1 = ttk.Frame(training_frame, style="White.TFrame")
        training_button_row_1.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Button(training_button_row_1, text="Neu initialisieren", command=lambda: self._load_experiment(reinitialize_model=True)).pack(side=tk.LEFT)
        ttk.Button(training_button_row_1, text="1 Epoche", command=lambda: self._train_for_epochs(1)).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(training_button_row_1, text="10 Epochen", command=lambda: self._train_for_epochs(10)).pack(side=tk.LEFT, padx=(8, 0))
        self.training_button_row_2 = ttk.Frame(training_frame, style="White.TFrame")
        self.training_button_row_2.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(self.training_button_row_2, text="N Epochen trainieren", command=self._train_current_model).pack(side=tk.LEFT)
        ttk.Button(self.training_button_row_2, text="Als Baseline speichern", command=self._store_baseline).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(self.training_button_row_2, text="Nur Ansicht aktualisieren", command=self._refresh_views).pack(side=tk.LEFT, padx=(8, 0))

        status_frame = ttk.LabelFrame(parent, text="4. Live-Status", padding=10)
        status_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            status_frame,
            textvariable=self.dataset_summary_var,
            justify=tk.LEFT,
            style="SectionValue.TLabel",
            wraplength=330,
        ).pack(anchor="w")
        ttk.Label(
            status_frame,
            textvariable=self.prediction_summary_var,
            justify=tk.LEFT,
            style="SectionValue.TLabel",
            wraplength=330,
        ).pack(anchor="w", pady=(10, 0))
        ttk.Label(
            status_frame,
            textvariable=self.metrics_summary_var,
            justify=tk.LEFT,
            style="SectionValue.TLabel",
            wraplength=330,
        ).pack(anchor="w", pady=(10, 0))
        ttk.Label(
            status_frame,
            textvariable=self.compare_summary_var,
            justify=tk.LEFT,
            style="Hint.TLabel",
            wraplength=330,
        ).pack(anchor="w", pady=(10, 0))

        self.expert_only_widgets.extend(
            [
                self.hidden_layers_label,
                self.layer_size_controls_frame,
                self.hidden_layer_buttons,
                self.analysis_target_label,
                self.analysis_target_combo,
                self.layer_fill_controls_frame,
                self.neuron_layer_label,
                self.neuron_layer_combo,
                self.neuron_index_label,
                self.neuron_index_combo,
                self.neuron_activation_label,
                self.neuron_activation_combo,
                self.neuron_button_row,
                self.epochs_label,
                self.epochs_spinbox,
                self.lr_label,
                self.lr_entry,
                self.batch_label,
                self.batch_spinbox,
                self.weight_scale_label,
                self.weight_scale_entry,
                self.seed_label,
                self.seed_spinbox,
                self.training_button_row_2,
            ]
        )

        self._rebuild_hidden_size_controls()
        self._rebuild_layer_fill_controls()
        self._apply_mode_visibility()

    def _build_content(self, parent: ttk.Frame) -> None:
        """Rechter Bereich mit Netzwerk, Input-Ansicht, Plotting und Lernhilfe."""

        summary_frame = ttk.LabelFrame(parent, text="Was passiert gerade?", padding=10)
        summary_frame.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            summary_frame,
            textvariable=self.network_summary_var,
            justify=tk.LEFT,
            style="SectionValue.TLabel",
            wraplength=1040,
        ).pack(anchor="w")

        canvas_frame = ttk.LabelFrame(parent, text="Netzwerk-Visualisierung", padding=8)
        canvas_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 10))
        canvas_frame.columnconfigure(0, weight=3)
        canvas_frame.columnconfigure(1, weight=2)
        canvas_frame.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, bg=NETWORK_BACKGROUND_COLOR, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self._redraw_network())

        live_detail_frame = ttk.Frame(canvas_frame, style="White.TFrame", padding=(12, 0, 0, 0))
        live_detail_frame.grid(row=0, column=1, sticky="nsew")
        live_detail_frame.columnconfigure(0, weight=1)
        live_detail_frame.rowconfigure(2, weight=1)
        live_detail_frame.rowconfigure(3, weight=0)

        ttk.Label(
            live_detail_frame,
            text="Live-Neuron-Inspektion",
            style="Headline.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            live_detail_frame,
            text=(
                "Klicke im Netz auf ein Hidden-Neuron. Rechts erscheint sofort dessen lokale "
                "Rechnung mit z, Aktivierung, Ableitung und den staerksten Beitragen."
            ),
            style="Hint.TLabel",
            justify=tk.LEFT,
            wraplength=360,
        ).grid(row=1, column=0, sticky="ew", pady=(6, 10))
        self.live_detail_text = ScrolledText(live_detail_frame, wrap=tk.WORD)
        self.live_detail_text.grid(row=2, column=0, sticky="nsew")
        self.live_detail_text.configure(state=tk.DISABLED, font=("Menlo", 10))
        self.activation_figure = Figure(figsize=(3.5, 2.4), dpi=100)
        self.activation_axis = self.activation_figure.add_subplot(111)
        self.activation_canvas = FigureCanvasTkAgg(self.activation_figure, master=live_detail_frame)
        self.activation_canvas.get_tk_widget().grid(row=3, column=0, sticky="ew", pady=(10, 0))

        notebook_frame = ttk.LabelFrame(parent, text="Analyse-Ansichten", padding=6)
        notebook_frame.grid(row=2, column=0, sticky="nsew")
        notebook_frame.columnconfigure(0, weight=1)
        notebook_frame.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(notebook_frame)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.input_tab = ttk.Frame(self.notebook, style="Notebook.TFrame")
        self.plot_tab = ttk.Frame(self.notebook, style="Notebook.TFrame")
        self.stepper_tab = ttk.Frame(self.notebook, style="Notebook.TFrame")
        self.detail_tab = ttk.Frame(self.notebook, style="Notebook.TFrame")
        self.compare_tab = ttk.Frame(self.notebook, style="Notebook.TFrame")
        self.help_tab = ttk.Frame(self.notebook, style="Notebook.TFrame")

        self.notebook.add(self.input_tab, text="Input & Ziel")
        self.notebook.add(self.plot_tab, text="Training & Plotting")
        self.notebook.add(self.stepper_tab, text="Forward/Backward")
        self.notebook.add(self.detail_tab, text="Neuron-Inspektion")
        self.notebook.add(self.compare_tab, text="Vergleich")
        self.notebook.add(self.help_tab, text="Lernhilfe")

        self._build_input_tab()
        self._build_plot_tab()
        self._build_stepper_tab()
        self._build_detail_tab()
        self._build_compare_tab()
        self._build_help_tab()

    def _build_input_tab(self) -> None:
        """Tab fuer Input-Darstellung, Zielwert und manuelle Eingaben."""

        self.input_tab.columnconfigure(0, weight=0)
        self.input_tab.columnconfigure(1, weight=1)
        self.input_tab.rowconfigure(0, weight=1)

        info_frame = ttk.Frame(self.input_tab, style="White.TFrame", padding=10)
        info_frame.grid(row=0, column=0, sticky="ns")

        ttk.Label(
            info_frame,
            text=(
                "Dieser Tab zeigt, was gerade in das Netz eingespeist wird. "
                "Bei digits siehst du die 8x8 Pixel, bei den anderen Benchmarks die "
                "wichtigsten Feature-Werte."
            ),
            style="Hint.TLabel",
            justify=tk.LEFT,
        ).pack(anchor="w")
        ttk.Label(
            info_frame,
            textvariable=self.sample_summary_var,
            justify=tk.LEFT,
            style="SectionValue.TLabel",
            wraplength=320,
        ).pack(anchor="w", pady=(10, 0))
        ttk.Label(
            info_frame,
            textvariable=self.sample_hint_var,
            justify=tk.LEFT,
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(10, 0))

        self.dynamic_input_container = ttk.Frame(self.input_tab, style="White.TFrame", padding=10)
        self.dynamic_input_container.grid(row=0, column=1, sticky="nsew")
        self.dynamic_input_container.columnconfigure(0, weight=1)
        self.dynamic_input_container.rowconfigure(0, weight=1)

        self.digits_frame = ttk.Frame(self.dynamic_input_container, style="White.TFrame")
        self.generic_features_frame = ttk.Frame(self.dynamic_input_container, style="White.TFrame")
        self.test_activation_frame = ttk.Frame(self.dynamic_input_container, style="White.TFrame")

        self._build_digits_frame()
        self._build_generic_features_frame()
        self._build_test_activation_frame()

    def _build_digits_frame(self) -> None:
        """8x8-Darstellung fuer den digits-Datensatz."""

        ttk.Label(
            self.digits_frame,
            text=(
                "Links-klick auf ein Feld aendert im Eigenmodus dessen Helligkeit. "
                "So kannst du eigene Digits ausprobieren."
            ),
            style="Hint.TLabel",
            justify=tk.LEFT,
        ).pack(anchor="w")
        button_row = ttk.Frame(self.digits_frame, style="White.TFrame")
        button_row.pack(anchor="w", pady=(8, 8))
        ttk.Button(button_row, text="Aktuelles Sample kopieren", command=self._copy_current_digit_to_custom).pack(side=tk.LEFT)
        ttk.Button(button_row, text="Eigenes Digit leeren", command=self._clear_custom_digit).pack(side=tk.LEFT, padx=(8, 0))

        self.digits_canvas = tk.Canvas(self.digits_frame, width=360, height=360, bg="#eef2f7", highlightthickness=0)
        self.digits_canvas.pack(fill=tk.BOTH, expand=False)
        self.digits_canvas.bind("<Button-1>", self._on_digits_canvas_clicked)

    def _build_generic_features_frame(self) -> None:
        """Feature-Tabelle fuer breast_cancer und wine."""

        ttk.Label(
            self.generic_features_frame,
            text=(
                "Die Tabelle zeigt die staerksten Eingabefeatures des aktuellen Samples. "
                "Rohwert = urspruengliche Eingabe, skaliert = Wert nach Standardisierung."
            ),
            style="Hint.TLabel",
            justify=tk.LEFT,
        ).pack(anchor="w")

        columns = ("rank", "feature", "raw", "scaled")
        self.feature_tree = ttk.Treeview(self.generic_features_frame, columns=columns, show="headings", height=18)
        for column, title, width in (
            ("rank", "#", 50),
            ("feature", "Feature", 220),
            ("raw", "Rohwert", 110),
            ("scaled", "Skaliert", 110),
        ):
            self.feature_tree.heading(column, text=title)
            self.feature_tree.column(column, width=width, anchor="w")
        self.feature_tree.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

    def _build_test_activation_frame(self) -> None:
        """Kleines Rechenlabor mit manuell setzbaren Inputs."""

        ttk.Label(
            self.test_activation_frame,
            text=(
                "Im test_activation-Modus kannst du rohe Eingabewerte direkt setzen. "
                "Das ist ideal, um Aktivierungsfunktionen ohne grossen Benchmark-Kontext zu verstehen."
            ),
            style="Hint.TLabel",
            justify=tk.LEFT,
        ).pack(anchor="w")

        slider_frame = ttk.Frame(self.test_activation_frame, style="White.TFrame")
        slider_frame.pack(anchor="w", pady=(12, 0))
        self.test_input_scales: list[tk.Scale] = []
        for index, label in enumerate(("input_a", "input_b", "input_c")):
            row = ttk.Frame(slider_frame, style="White.TFrame")
            row.pack(fill=tk.X, pady=(0, 8))
            ttk.Label(row, text=label, width=10).pack(side=tk.LEFT)
            scale = tk.Scale(
                row,
                from_=0.0,
                to=1.0,
                resolution=0.05,
                orient=tk.HORIZONTAL,
                length=220,
                variable=self.test_input_vars[index],
                command=lambda _value: self._refresh_views(),
                bg=PANEL_COLOR,
                highlightthickness=0,
            )
            scale.pack(side=tk.LEFT)
            self.test_input_scales.append(scale)

        ttk.Label(
            self.test_activation_frame,
            text=(
                "Regel des Lernmodus: Klasse 1, wenn mindestens zwei Inputs deutlich aktiv sind. "
                "Du kannst aber separat ein Analyse-Ziel setzen, um den Loss zu studieren."
            ),
            style="Hint.TLabel",
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(12, 0))

        columns = ("feature", "raw", "scaled")
        self.test_activation_tree = ttk.Treeview(
            self.test_activation_frame, columns=columns, show="headings", height=5
        )
        for column, title, width in (
            ("feature", "Feature", 180),
            ("raw", "Rohwert", 100),
            ("scaled", "Skaliert", 100),
        ):
            self.test_activation_tree.heading(column, text=title)
            self.test_activation_tree.column(column, width=width, anchor="w")
        self.test_activation_tree.pack(fill=tk.X, expand=False, pady=(10, 0))

    def _build_plot_tab(self) -> None:
        """Tab mit eingebetteten Trainingsplots."""

        self.plot_tab.columnconfigure(0, weight=1)
        self.plot_tab.rowconfigure(1, weight=1)

        ttk.Label(
            self.plot_tab,
            text=(
                "Die Plots zeigen sowohl die Trainingsgeschichte als auch den aktuellen Zustand "
                "des ausgewaehlten Samples. Training aendert Gewichte, nicht die gezeigte Sample-Auswahl."
            ),
            style="Hint.TLabel",
            justify=tk.LEFT,
        ).grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))

        figure_frame = ttk.Frame(self.plot_tab, style="White.TFrame")
        figure_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        figure_frame.columnconfigure(0, weight=1)
        figure_frame.rowconfigure(0, weight=1)

        self.training_figure = Figure(figsize=(10, 6), dpi=100)
        self.training_axes = [
            self.training_figure.add_subplot(221),
            self.training_figure.add_subplot(222),
            self.training_figure.add_subplot(223),
            self.training_figure.add_subplot(224),
        ]
        self.training_canvas_widget = FigureCanvasTkAgg(self.training_figure, master=figure_frame)
        self.training_canvas_widget.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _build_stepper_tab(self) -> None:
        """Tab fuer einen schrittweisen Forward/Backward-Durchlauf."""

        self.stepper_tab.columnconfigure(0, weight=1)
        self.stepper_tab.rowconfigure(2, weight=1)
        ttk.Label(
            self.stepper_tab,
            text=(
                "Dieser Tab zerlegt den aktuellen Forward- und Backward-Pass fuer genau das "
                "sichtbare Sample in einzelne didaktische Schritte."
            ),
            style="Hint.TLabel",
            justify=tk.LEFT,
        ).grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))

        control_row = ttk.Frame(self.stepper_tab, style="White.TFrame")
        control_row.grid(row=1, column=0, sticky="ew", padx=10, pady=(8, 0))
        ttk.Button(control_row, text="Zurueck", command=self._step_prev).pack(side=tk.LEFT)
        ttk.Button(control_row, text="Weiter", command=self._step_next).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(control_row, text="Auf Anfang", command=self._step_reset).pack(side=tk.LEFT, padx=(8, 0))
        self.step_status_label = ttk.Label(control_row, text="", style="SectionValue.TLabel")
        self.step_status_label.pack(side=tk.LEFT, padx=(12, 0))

        self.stepper_text = ScrolledText(self.stepper_tab, wrap=tk.WORD)
        self.stepper_text.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        self.stepper_text.configure(state=tk.DISABLED, font=("Menlo", 11))

    def _build_compare_tab(self) -> None:
        """Tab fuer den Vergleich von Baseline und aktuellem Experiment."""

        self.compare_tab.columnconfigure(0, weight=1)
        self.compare_tab.rowconfigure(1, weight=1)
        ttk.Label(
            self.compare_tab,
            text=(
                "Speichere einen Zustand als Baseline und vergleiche danach Layout, "
                "Metriken und Vorhersage des aktuellen Experiments mit genau diesem Referenzpunkt."
            ),
            style="Hint.TLabel",
            justify=tk.LEFT,
        ).grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        self.compare_text = ScrolledText(self.compare_tab, wrap=tk.WORD)
        self.compare_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.compare_text.configure(state=tk.DISABLED, font=("Menlo", 11))

    def _build_detail_tab(self) -> None:
        """Tab fuer die detaillierte neuronale Einzelbetrachtung."""

        self.detail_tab.columnconfigure(0, weight=1)
        self.detail_tab.rowconfigure(1, weight=1)
        ttk.Label(
            self.detail_tab,
            text=(
                "Dieselbe Neuron-Inspektion wie rechts neben der Netzwerkansicht, aber mit mehr "
                "Platz zum Lesen und Scrollen."
            ),
            style="Hint.TLabel",
            justify=tk.LEFT,
        ).grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        self.detail_text = ScrolledText(self.detail_tab, wrap=tk.WORD)
        self.detail_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.detail_text.configure(state=tk.DISABLED, font=("Menlo", 11))

    def _rebuild_hidden_size_controls(self) -> None:
        """Baut die Layergroessen-Steuerung passend zur aktuellen Layerzahl neu auf."""

        if self.layer_size_controls_frame is None:
            return
        for child in self.layer_size_controls_frame.winfo_children():
            child.destroy()
        self.hidden_size_spinboxes.clear()
        self.hidden_layer_size_labels.clear()

        for layer_index, variable in enumerate(self.hidden_size_vars):
            row = ttk.Frame(self.layer_size_controls_frame, style="White.TFrame")
            row.pack(fill=tk.X, pady=(0, 4))
            label = ttk.Label(row, text=f"L{layer_index + 1}", width=5)
            label.pack(side=tk.LEFT)
            spinbox = ttk.Spinbox(row, from_=1, to=256, textvariable=variable, width=8)
            spinbox.pack(side=tk.LEFT)
            self.hidden_layer_size_labels.append(label)
            self.hidden_size_spinboxes.append(spinbox)

    def _rebuild_layer_fill_controls(self) -> None:
        """Baut die layerweisen Aktivierungs-Controls passend zur aktuellen Layerzahl neu auf."""

        if self.layer_fill_controls_frame is None:
            return
        for child in self.layer_fill_controls_frame.winfo_children():
            child.destroy()
        self.layer_fill_combos.clear()

        for layer_index, variable in enumerate(self.layer_fill_vars):
            row = ttk.Frame(self.layer_fill_controls_frame, style="White.TFrame")
            row.pack(fill=tk.X, pady=(0, 6))
            ttk.Label(row, text=f"Layer {layer_index + 1} auf", width=12).pack(side=tk.LEFT)
            combo = ttk.Combobox(
                row,
                textvariable=variable,
                values=SUPPORTED_ACTIVATIONS,
                state="readonly",
                width=14,
            )
            combo.pack(side=tk.LEFT, padx=(8, 0))
            ttk.Button(
                row,
                text="setzen",
                command=lambda li=layer_index: self._fill_layer(li),
            ).pack(side=tk.LEFT, padx=(8, 0))
            self.layer_fill_combos.append(combo)

    def _apply_mode_visibility(self) -> None:
        """Blendet Expertenfunktionen je nach Modus ein oder aus."""

        is_expert = self.mode_var.get() == "expert"
        for widget in self.expert_only_widgets:
            if is_expert:
                widget.grid()
            else:
                widget.grid_remove()

        if hasattr(self, "notebook"):
            for tab_name, tab_widget in (
                ("Vergleich", self.compare_tab),
                ("Forward/Backward", self.stepper_tab),
            ):
                try:
                    if is_expert or tab_name == "Forward/Backward":
                        self.notebook.tab(tab_widget, state="normal")
                    else:
                        self.notebook.tab(tab_widget, state="hidden")
                except tk.TclError:
                    pass

    def _on_mode_changed(self) -> None:
        """Aktualisiert die GUI zwischen Einsteiger- und Expertenmodus."""

        self._apply_mode_visibility()
        self._refresh_views()

    def _add_hidden_layer(self) -> None:
        """Fuegt einen weiteren Hidden-Layer fuer den Expertenmodus hinzu."""

        if len(self.hidden_size_vars) >= MAX_HIDDEN_LAYERS:
            messagebox.showinfo(
                "Maximale Layerzahl erreicht",
                f"Die GUI unterstuetzt didaktisch bis zu {MAX_HIDDEN_LAYERS} Hidden-Layer.",
            )
            return
        last_size = self.hidden_size_vars[-1].get() if self.hidden_size_vars else 8
        last_activation = self.layer_fill_vars[-1].get() if self.layer_fill_vars else "relu"
        self.hidden_size_vars.append(tk.IntVar(value=last_size))
        self.layer_fill_vars.append(tk.StringVar(value=last_activation))
        if self.current_layout is not None:
            self.current_layout = self.current_layout.add_layer(last_size, activation_name=last_activation)
            self.layout_string_var.set(self.current_layout.to_compact_spec())
        self._rebuild_hidden_size_controls()
        self._rebuild_layer_fill_controls()
        self._reload_experiment()

    def _remove_hidden_layer(self) -> None:
        """Entfernt den letzten Hidden-Layer, sofern mindestens einer bleibt."""

        if len(self.hidden_size_vars) <= MIN_HIDDEN_LAYERS:
            messagebox.showinfo(
                "Mindestens ein Hidden-Layer",
                "Das Lernmodell benoetigt mindestens einen Hidden-Layer.",
            )
            return
        self.hidden_size_vars.pop()
        self.layer_fill_vars.pop()
        if self.current_layout is not None:
            self.current_layout = self.current_layout.remove_layer(len(self.current_layout.layers) - 1)
            self.layout_string_var.set(self.current_layout.to_compact_spec())
        self._rebuild_hidden_size_controls()
        self._rebuild_layer_fill_controls()
        self._reload_experiment()

    def _store_baseline(self) -> None:
        """Speichert den aktuellen Zustand als Vergleichs-Baseline."""

        if self.model is None or self.current_layout is None:
            return
        self.baseline_model = self.model.clone()
        self.baseline_layout = ActivationLayout(tuple(tuple(layer) for layer in self.current_layout.layers))
        self.baseline_training_result = copy.deepcopy(self.training_result)
        self.baseline_completed_epochs = self.completed_epochs
        self.compare_summary_var.set(
            f"Baseline gespeichert: Layout {self.baseline_layout.to_compact_spec()} | "
            f"Epochen {self.baseline_completed_epochs}"
        )
        self._update_compare_text()

    def _build_help_tab(self) -> None:
        """Tab mit Schritt-fuer-Schritt-Hinweisen."""

        self.help_tab.columnconfigure(0, weight=1)
        self.help_tab.rowconfigure(0, weight=1)
        self.help_text = ScrolledText(self.help_tab, wrap=tk.WORD)
        self.help_text.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.help_text.configure(state=tk.DISABLED, font=("Menlo", 11))

    def _apply_preset(self, preset_name: str) -> None:
        """Setzt eine kuratierte Startkonfiguration."""

        if preset_name == "wine_demo":
            self.benchmark_var.set("wine")
            self._set_hidden_sizes((16, 8))
            self.layout_string_var.set("relu|tanh")
            self.epochs_var.set(120)
            self.lr_var.set("0.03")
        elif preset_name == "digits_demo":
            self.benchmark_var.set("digits")
            self._set_hidden_sizes((32, 16))
            self.layout_string_var.set("relu*16,tanh*16|relu*8,sigmoid*8")
            self.epochs_var.set(80)
            self.lr_var.set("0.05")
        elif preset_name == "all_relu":
            self.layout_string_var.set("|".join(["relu"] * len(self.hidden_size_vars)))
        elif preset_name == "mixed":
            hidden_sizes = self._current_hidden_sizes()
            layer_specs: list[str] = []
            for layer_index, layer_size in enumerate(hidden_sizes):
                first_activation, second_activation = (
                    ("relu", "tanh") if layer_index % 2 == 0 else ("sigmoid", "leaky_relu")
                )
                first_count = max(layer_size // 2, 1)
                second_count = max(layer_size - first_count, 0)
                if second_count == 0:
                    layer_specs.append(f"{first_activation}*{first_count}")
                else:
                    layer_specs.append(
                        f"{first_activation}*{first_count},{second_activation}*{second_count}"
                    )
            self.layout_string_var.set("|".join(layer_specs))
        elif preset_name == "test_activation":
            self.benchmark_var.set("test_activation")
            self._set_hidden_sizes((4, 3))
            self.layout_string_var.set("relu,tanh,sigmoid,leaky_relu|relu,tanh,sigmoid")
            self.use_custom_sample_var.set(True)
            self.test_input_vars[0].set(1.0)
            self.test_input_vars[1].set(0.5)
            self.test_input_vars[2].set(0.0)

        self._reload_experiment()

    def _reset_to_defaults(self) -> None:
        """Setzt die GUI auf eine einfache Startkonfiguration zurueck."""

        self.benchmark_var.set("breast_cancer")
        self.mode_var.set("beginner")
        self._set_hidden_sizes(default_hidden_sizes("breast_cancer"))
        self.layout_string_var.set(DEFAULT_LAYOUT)
        self.epochs_var.set(DEFAULT_EPOCHS)
        self.lr_var.set(str(DEFAULT_LEARNING_RATE))
        self.batch_size_var.set(DEFAULT_BATCH_SIZE)
        self.weight_scale_var.set(str(DEFAULT_WEIGHT_SCALE))
        self.seed_var.set(DEFAULT_RANDOM_SEED)
        self.use_custom_sample_var.set(False)
        self.analysis_target_var.set(TARGET_AUTO_LABEL)
        self._apply_mode_visibility()
        self._reload_experiment()

    def _apply_benchmark_defaults(self) -> None:
        """Setzt die Hidden-Sizes passend zum Benchmark."""

        self._set_hidden_sizes(default_hidden_sizes(self.benchmark_var.get()))

    def _reload_experiment(self) -> None:
        """Laedt Datensatz, Layout und Modell neu."""

        self._load_experiment(reinitialize_model=True)

    def _current_hidden_sizes(self) -> tuple[int, ...]:
        """Liefert die aktuell in der GUI eingestellten Hidden-Sizes."""

        return tuple(int(variable.get()) for variable in self.hidden_size_vars)

    def _set_hidden_sizes(self, hidden_sizes: Sequence[int]) -> None:
        """Setzt Hidden-Sizes und synchronisiert die dazugehoerigen GUI-Variablen."""

        previous_fill_values = [variable.get() for variable in self.layer_fill_vars]
        self.hidden_size_vars = [tk.IntVar(value=int(size)) for size in hidden_sizes]
        self.layer_fill_vars = [
            tk.StringVar(
                value=(
                    previous_fill_values[layer_index]
                    if layer_index < len(previous_fill_values)
                    else "relu"
                )
            )
            for layer_index in range(len(hidden_sizes))
        ]
        self._rebuild_hidden_size_controls()
        self._rebuild_layer_fill_controls()

    def _load_experiment(self, reinitialize_model: bool) -> None:
        """Laedt Datensatz, Modell und alle GUI-Abhaengigkeiten neu."""

        try:
            benchmark = self.benchmark_var.get()
            hidden_sizes = self._current_hidden_sizes()
            weight_scale = float(self.weight_scale_var.get())
            seed = int(self.seed_var.get())

            self.dataset = load_benchmark(DatasetConfig(name=benchmark, random_state=seed))
            if benchmark != "digits" and benchmark != "test_activation":
                self.use_custom_sample_var.set(False)
            if benchmark == "test_activation":
                self.use_custom_sample_var.set(True)

            if self.current_layout is None or self.current_layout.hidden_sizes != hidden_sizes:
                self.current_layout = self._resolve_layout_for_hidden_sizes(hidden_sizes)
            else:
                self.current_layout = parse_layout_spec(self.layout_string_var.get(), hidden_sizes)
            self.layout_string_var.set(self.current_layout.to_compact_spec())
            self._set_hidden_sizes(hidden_sizes)
            for layer_index, layer in enumerate(self.current_layout.layers):
                self.layer_fill_vars[layer_index].set(layer[0])
            self.layer_summary_var.set(
                "\n".join(
                    f"L{layer_index + 1}: {len(layer)} Neuronen, Start-Aktivierung {layer[0]}"
                    for layer_index, layer in enumerate(self.current_layout.layers)
                )
            )

            if reinitialize_model or self.model is None:
                self.model = ModularMLP(
                    input_size=self.dataset.input_size,
                    hidden_sizes=hidden_sizes,
                    output_size=self.dataset.output_size,
                    layout=self.current_layout,
                    weight_scale=weight_scale,
                    random_state=seed,
                )
                self.training_result = None
                self.completed_epochs = 0
            else:
                self.model.layout = self.current_layout

            self.dataset_summary_var.set(self._build_dataset_summary_text())

            self._update_target_options()
            self._ensure_selected_hidden_is_valid()
            self._update_neuron_dropdowns()
            self._update_sample_spinbox_range()
            self._prepare_custom_samples_for_benchmark()
            self._refresh_views()
        except Exception as exc:
            messagebox.showerror("Experiment konnte nicht geladen werden", str(exc))

    def _resolve_layout_for_hidden_sizes(self, hidden_sizes: tuple[int, ...]) -> ActivationLayout:
        """Versucht, das aktuelle Layout auf neue Hidden-Sizes abzubilden."""

        current_layout_string = self.layout_string_var.get().strip() or DEFAULT_LAYOUT
        try:
            return parse_layout_spec(current_layout_string, hidden_sizes)
        except ValueError:
            return parse_layout_spec(DEFAULT_LAYOUT, hidden_sizes)

    def _prepare_custom_samples_for_benchmark(self) -> None:
        """Initialisiert benchmark-spezifische Custom-Samples."""

        if self.dataset is None:
            return
        if self.dataset.name == "digits":
            raw_sample = self.dataset.X_train_raw[0] if len(self.dataset.X_train_raw) > 0 else np.zeros(64)
            if not np.any(self.raw_custom_digit):
                self.raw_custom_digit = np.asarray(raw_sample, dtype=np.float64).copy()
        elif self.dataset.name == "test_activation":
            self.test_input_vars[0].set(1.0)
            self.test_input_vars[1].set(0.0)
            self.test_input_vars[2].set(1.0)

    def _update_target_options(self) -> None:
        """Aktualisiert die moeglichen Analyse-Ziele."""

        if self.dataset is None:
            return
        values = [TARGET_AUTO_LABEL, TARGET_NONE_LABEL, *self.dataset.target_names]
        self.analysis_target_combo.configure(values=values)
        if self.analysis_target_var.get() not in values:
            self.analysis_target_var.set(TARGET_AUTO_LABEL)

    def _ensure_selected_hidden_is_valid(self) -> None:
        """Sorgt dafuer, dass das selektierte Hidden-Neuron existiert."""

        if self.current_layout is None:
            self.selected_hidden = (0, 0)
            return
        layer_index, neuron_index = self.selected_hidden
        layer_index = min(max(layer_index, 0), len(self.current_layout.layers) - 1)
        neuron_index = min(max(neuron_index, 0), len(self.current_layout.layers[layer_index]) - 1)
        self.selected_hidden = (layer_index, neuron_index)

    def _update_neuron_dropdowns(self) -> None:
        """Aktualisiert die moeglichen Neuron-Indizes im Dropdown."""

        if self.current_layout is None:
            return
        layer_values = tuple(f"L{layer_index + 1}" for layer_index in range(len(self.current_layout.layers)))
        self.neuron_layer_combo.configure(values=layer_values)
        if self.neuron_layer_var.get() not in layer_values:
            self.neuron_layer_var.set(layer_values[0])
        layer_index = int(self.neuron_layer_var.get().replace("L", "")) - 1
        values = [str(index) for index in range(len(self.current_layout.layers[layer_index]))]
        self.neuron_index_combo.configure(values=values)

        selected_layer, selected_neuron = self.selected_hidden
        if selected_layer == layer_index:
            self.neuron_index_var.set(str(selected_neuron))
            self.neuron_activation_var.set(self.current_layout.layers[layer_index][selected_neuron])
        elif values:
            self.neuron_index_var.set(values[0])

    def _update_sample_spinbox_range(self) -> None:
        """Passt den Sample-Bereich an den aktuellen Split an."""

        X_split, _ = self._get_current_split_arrays()
        max_index = max(len(X_split) - 1, 0)
        self.sample_spinbox.configure(to=max_index)
        if self.sample_index_var.get() > max_index:
            self.sample_index_var.set(max_index)

    def _get_current_split_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Liefert skalierten Datensplit und Labels."""

        if self.dataset is None:
            raise ValueError("Kein Datensatz geladen.")
        split_name = self.split_var.get()
        if split_name == "train":
            return self.dataset.X_train, self.dataset.y_train
        if split_name == "val":
            return self.dataset.X_val, self.dataset.y_val
        return self.dataset.X_test, self.dataset.y_test

    def _get_current_raw_split_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Liefert Rohdaten-Split und Labels."""

        if self.dataset is None:
            raise ValueError("Kein Datensatz geladen.")
        split_name = self.split_var.get()
        if split_name == "train":
            return self.dataset.X_train_raw, self.dataset.y_train
        if split_name == "val":
            return self.dataset.X_val_raw, self.dataset.y_val
        return self.dataset.X_test_raw, self.dataset.y_test

    def _get_analysis_sample(self) -> AnalysisSample:
        """Erzeugt das aktuell in der GUI untersuchte Beispiel."""

        if self.dataset is None:
            raise ValueError("Kein Datensatz geladen.")

        benchmark = self.dataset.name
        split_name = self.split_var.get()
        actual_target_index: int | None
        source_label: str
        is_custom = bool(self.use_custom_sample_var.get())

        if benchmark == "digits" and is_custom:
            raw_sample = self.raw_custom_digit.copy()
            actual_target_index = None
            source_label = "Eigenes digits-Sample"
        elif benchmark == "test_activation" and is_custom:
            raw_sample = np.asarray([variable.get() for variable in self.test_input_vars], dtype=np.float64)
            actual_target_index = self._infer_test_activation_target(raw_sample)
            source_label = "Eigenes test_activation-Sample"
        else:
            X_scaled, y_split = self._get_current_split_arrays()
            X_raw, _ = self._get_current_raw_split_arrays()
            sample_index = min(max(self.sample_index_var.get(), 0), len(X_scaled) - 1)
            self.sample_index_var.set(sample_index)
            raw_sample = np.asarray(X_raw[sample_index], dtype=np.float64).copy()
            actual_target_index = int(y_split[sample_index])
            source_label = f"Datensatz-Sample aus {split_name}"

        if self.dataset.scaler is not None:
            scaled_sample = self.dataset.scaler.transform(raw_sample.reshape(1, -1))[0]
        else:
            scaled_sample = raw_sample.copy()

        actual_target_name = (
            self.dataset.target_names[actual_target_index]
            if actual_target_index is not None
            else "unbekannt"
        )
        effective_target_index = self._resolve_effective_target_index(actual_target_index)
        effective_target_name = (
            self.dataset.target_names[effective_target_index]
            if effective_target_index is not None
            else "kein Ziel gesetzt"
        )

        return AnalysisSample(
            raw_sample=raw_sample,
            scaled_sample=scaled_sample,
            actual_target_index=actual_target_index,
            effective_target_index=effective_target_index,
            actual_target_name=actual_target_name,
            effective_target_name=effective_target_name,
            split_name=split_name,
            sample_index=int(self.sample_index_var.get()),
            source_label=source_label,
            is_custom=is_custom,
        )

    def _resolve_effective_target_index(self, actual_target_index: int | None) -> int | None:
        """Bestimmt, welches Ziel fuer Loss-Analyse verwendet werden soll."""

        if self.dataset is None:
            return None

        selected = self.analysis_target_var.get()
        if selected == TARGET_NONE_LABEL:
            return None
        if selected == TARGET_AUTO_LABEL:
            return actual_target_index
        if selected in self.dataset.target_names:
            return self.dataset.target_names.index(selected)
        return actual_target_index

    def _infer_test_activation_target(self, raw_sample: np.ndarray) -> int:
        """Einfache Regel fuer den Lernmodus im test_activation-Datensatz."""

        return 1 if float(np.sum(raw_sample >= 0.5)) >= 2 else 0

    def _choose_random_sample(self) -> None:
        """Waehlt zufaellig ein Beispiel aus dem aktuellen Split."""

        X_split, _ = self._get_current_split_arrays()
        if len(X_split) == 0:
            return
        rng = np.random.default_rng()
        self.sample_index_var.set(int(rng.integers(0, len(X_split))))
        if self.dataset is not None and self.dataset.name == "digits" and self.use_custom_sample_var.get():
            self._copy_current_digit_to_custom()
        self._refresh_views()

    def _fill_layer(self, layer_index: int) -> None:
        """Setzt einen ganzen Hidden-Layer auf eine Aktivierung."""

        if self.current_layout is None or self.model is None:
            return
        activation_name = self.layer_fill_vars[layer_index].get()
        self.current_layout = self.current_layout.replace_layer(layer_index, activation_name)
        self.model.layout = self.current_layout
        self.layout_string_var.set(self.current_layout.to_compact_spec())
        self._refresh_views()

    def _apply_neuron_setting(self) -> None:
        """Setzt die Aktivierung eines einzelnen Hidden-Neurons."""

        if self.current_layout is None or self.model is None:
            return
        layer_index = int(self.neuron_layer_var.get().replace("L", "")) - 1
        neuron_index = int(self.neuron_index_var.get())
        activation_name = self.neuron_activation_var.get()
        self.current_layout = self.current_layout.replace_neuron(layer_index, neuron_index, activation_name)
        self.model.layout = self.current_layout
        self.selected_hidden = (layer_index, neuron_index)
        self.layout_string_var.set(self.current_layout.to_compact_spec())
        self._refresh_views()

    def _cycle_selected_neuron(self) -> None:
        """Schaltet das aktuell gewaehlte Neuron zur naechsten Aktivierung weiter."""

        if self.current_layout is None or self.model is None:
            return
        layer_index = int(self.neuron_layer_var.get().replace("L", "")) - 1
        neuron_index = int(self.neuron_index_var.get())
        self.current_layout = self.current_layout.cycle_neuron(layer_index, neuron_index)
        self.model.layout = self.current_layout
        self.selected_hidden = (layer_index, neuron_index)
        self.layout_string_var.set(self.current_layout.to_compact_spec())
        self.neuron_activation_var.set(self.current_layout.layers[layer_index][neuron_index])
        self._refresh_views()

    def _select_from_controls(self) -> None:
        """Uebernimmt die aktuelle Dropdown-Auswahl als selektiertes Neuron."""

        if self.current_layout is None:
            return
        layer_index = int(self.neuron_layer_var.get().replace("L", "")) - 1
        neuron_index = int(self.neuron_index_var.get())
        self.selected_hidden = (layer_index, neuron_index)
        self.neuron_activation_var.set(self.current_layout.layers[layer_index][neuron_index])
        self._refresh_views()

    def _on_benchmark_changed(self, _event=None) -> None:
        """Beim Benchmark-Wechsel zunaechst sinnvolle Default-Hidden-Sizes setzen."""

        self._apply_benchmark_defaults()
        if self.benchmark_var.get() == "test_activation":
            self.use_custom_sample_var.set(True)

    def _on_neuron_layer_changed(self, _event=None) -> None:
        """Passt die Neuron-Auswahl an den gewaehlten Layer an."""

        self._update_neuron_dropdowns()
        self._select_from_controls()

    def _on_split_changed(self) -> None:
        """Aktualisiert Sample-Bereich und Ansichten nach Split-Wechsel."""

        self._update_sample_spinbox_range()
        self._refresh_views()

    def _on_custom_sample_toggled(self) -> None:
        """Aktualisiert Ansichten, wenn zwischen Datensatz- und Eigen-Sample gewechselt wird."""

        if self.dataset is not None and self.dataset.name == "digits" and self.use_custom_sample_var.get():
            self._copy_current_digit_to_custom()
        self._refresh_views()

    def _train_current_model(self) -> None:
        """Trainiert fuer die in der GUI eingestellte Zahl von Epochen."""

        self._train_for_epochs(int(self.epochs_var.get()))

    def _train_for_epochs(self, epochs: int) -> None:
        """Trainiert das aktuelle Modell fuer eine gegebene Epochenanzahl."""

        if self.model is None or self.dataset is None:
            return
        try:
            training_config = TrainingConfig(
                epochs=int(epochs),
                learning_rate=float(self.lr_var.get()),
                batch_size=int(self.batch_size_var.get()),
                random_state=int(self.seed_var.get()),
            )
            new_result = train_model(self.model, self.dataset, training_config)
            self.training_result = self._merge_training_results(self.training_result, new_result)
            self.completed_epochs += int(epochs)
            self._refresh_views()
        except Exception as exc:
            messagebox.showerror("Training fehlgeschlagen", str(exc))

    def _merge_training_results(
        self,
        previous: TrainingResult | None,
        new_result: TrainingResult,
    ) -> TrainingResult:
        """Haengt neue Trainingskurven an die bisherigen an."""

        if previous is None:
            return new_result
        merged_history = {
            key: previous.history.get(key, []) + new_result.history.get(key, [])
            for key in new_result.history
        }
        return TrainingResult(history=merged_history, test_metrics=new_result.test_metrics)

    def _refresh_views(self) -> None:
        """Aktualisiert alle abhaengigen GUI-Bereiche."""

        self._refresh_guidance()
        self._refresh_sample_tab()
        self._update_prediction_summary()
        self._update_metrics_summary()
        self._update_neuron_detail()
        self._update_training_plot()
        self._update_stepper_view()
        self._update_compare_text()
        self._redraw_network()
        self._update_help_text()

    def _refresh_guidance(self) -> None:
        """Aktualisiert die kontextabhaengigen Erklaerungen."""

        benchmark = self.benchmark_var.get()
        selected_layer, selected_neuron = self.selected_hidden
        self.workflow_summary_var.set(
            "1. Waehle Datensatz und Beispiel.\n"
            "2. Beobachte Rohdaten, Zielwert und Vorhersage.\n"
            "3. Aendere Aktivierungen pro Layer oder Neuron.\n"
            "4. Trainiere schrittweise und beobachte, wie sich Gewichte und Kurven aendern.\n"
            "5. Nutze den Forward/Backward-Tab fuer den Rechenweg eines einzelnen Samples.\n"
            "6. Klicke auf ein Hidden-Neuron und analysiere seine lokale Rechnung."
        )

        if benchmark == "digits":
            hint = (
                "Digits ist besonders visuell: Im Tab 'Input & Ziel' siehst du 8x8 Pixel. "
                "Mit 'Eigenes Sample verwenden' kannst du ein eigenes Digit bauen."
            )
        elif benchmark == "test_activation":
            hint = (
                "test_activation ist das Rechenlabor: wenige Inputs, wenige Neuronen, "
                "schnell sichtbare Effekte von ReLU, tanh, sigmoid und leaky_relu. "
                "Nutze hier besonders die Live-Neuron-Inspektion rechts neben der Netzgrafik."
            )
        else:
            hint = (
                f"Aktuell ist L{selected_layer + 1} n{selected_neuron} selektiert. "
                "Dessen staerkste Ein- und Ausgaenge sind farblich hervorgehoben."
            )
        self.context_hint_var.set(hint)

    def _update_prediction_summary(self) -> None:
        """Zeigt Vorhersage, Ziel und Loss-Informationen fuer das aktuelle Sample."""

        if self.model is None or self.dataset is None:
            self.prediction_summary_var.set("Kein Modell geladen.")
            return

        analysis_sample = self._get_analysis_sample()
        probabilities = self.model.predict_proba(analysis_sample.scaled_sample.reshape(1, -1))[0]
        predicted_index = int(np.argmax(probabilities))
        predicted_name = self.dataset.target_names[predicted_index]

        actual_loss_text = "kein echtes Ziel"
        if analysis_sample.actual_target_index is not None:
            actual_probability = float(probabilities[analysis_sample.actual_target_index])
            actual_loss_text = f"{-np.log(max(actual_probability, 1e-12)):.4f}"

        effective_loss_text = "nicht berechnet"
        if analysis_sample.effective_target_index is not None:
            effective_probability = float(probabilities[analysis_sample.effective_target_index])
            effective_loss_text = f"{-np.log(max(effective_probability, 1e-12)):.4f}"

        probability_text = ", ".join(
            f"{self.dataset.target_names[index]}={probabilities[index]:.3f}"
            for index in range(len(probabilities))
        )
        self.prediction_summary_var.set(
            f"Beobachtetes Sample: {analysis_sample.source_label}\n"
            f"Vorhersage: {predicted_name}\n"
            f"Echtes Ziel: {analysis_sample.actual_target_name}\n"
            f"Analyse-Ziel: {analysis_sample.effective_target_name}\n"
            f"Loss zum echten Ziel: {actual_loss_text}\n"
            f"Loss zum Analyse-Ziel: {effective_loss_text}\n"
            f"Klassenwahrscheinlichkeiten: {probability_text}"
        )

    def _update_metrics_summary(self) -> None:
        """Schreibt Trainingsmetriken in die Statusspalte."""

        if self.training_result is None:
            self.metrics_summary_var.set(
                "Noch kein Training durchgefuehrt.\n"
                "Du kannst bereits Layouts aendern und lokale Berechnungen analysieren."
            )
            return

        history = self.training_result.history
        best_epoch = max(range(len(history["val_acc"])), key=lambda index: history["val_acc"][index]) + 1
        self.metrics_summary_var.set(
            f"Trainiert fuer insgesamt {self.completed_epochs} Epochen.\n"
            f"Letzte Train-Acc: {history['train_acc'][-1]:.4f}\n"
            f"Letzte Val-Acc:   {history['val_acc'][-1]:.4f}\n"
            f"Beste Val-Acc:    {max(history['val_acc']):.4f} in Epoche {best_epoch}\n"
            f"Test-Acc:         {self.training_result.test_metrics['accuracy']:.4f}\n"
            f"Test-Loss:        {self.training_result.test_metrics['loss']:.4f}"
        )

    def _refresh_sample_tab(self) -> None:
        """Aktualisiert Input-Ansicht, Zielinformationen und benchmark-spezifische Widgets."""

        if self.dataset is None or self.model is None:
            return

        analysis_sample = self._get_analysis_sample()
        probabilities = self.model.predict_proba(analysis_sample.scaled_sample.reshape(1, -1))[0]
        predicted_name = self.dataset.target_names[int(np.argmax(probabilities))]

        self.sample_summary_var.set(
            f"Quelle: {analysis_sample.source_label}\n"
            f"Split: {analysis_sample.split_name} | Sample-Index: {analysis_sample.sample_index}\n"
            f"Echtes Ziel: {analysis_sample.actual_target_name}\n"
            f"Analyse-Ziel: {analysis_sample.effective_target_name}\n"
            f"Vorhersage: {predicted_name}"
        )
        self.sample_hint_var.set(
            "Wichtig: Das aktuell gezeigte Sample ist ein Analysefenster. "
            "Das eigentliche Training passiert weiterhin auf dem gesamten Trainingssplit."
        )

        for frame in (self.digits_frame, self.generic_features_frame, self.test_activation_frame):
            frame.grid_forget()

        if self.dataset.name == "digits":
            self.digits_frame.grid(row=0, column=0, sticky="nsew")
            self._draw_digit_sample(analysis_sample.raw_sample)
        elif self.dataset.name == "test_activation":
            self.test_activation_frame.grid(row=0, column=0, sticky="nsew")
            self._update_test_activation_table(analysis_sample)
        else:
            self.generic_features_frame.grid(row=0, column=0, sticky="nsew")
            self._update_feature_tree(analysis_sample)

        self.use_custom_sample_checkbutton.state(["!disabled"])
        if self.dataset.name not in {"digits", "test_activation"}:
            self.use_custom_sample_checkbutton.state(["disabled"])

    def _update_feature_tree(self, analysis_sample: AnalysisSample) -> None:
        """Fuellt die Feature-Tabelle fuer nicht-bildhafte Benchmarks."""

        if self.dataset is None:
            return
        self.feature_tree.delete(*self.feature_tree.get_children())
        rows = []
        for feature_index, feature_name in enumerate(self.dataset.feature_names):
            raw_value = float(analysis_sample.raw_sample[feature_index])
            scaled_value = float(analysis_sample.scaled_sample[feature_index])
            rows.append((abs(scaled_value), feature_name, raw_value, scaled_value))
        rows.sort(key=lambda row: row[0], reverse=True)
        for rank, (_, feature_name, raw_value, scaled_value) in enumerate(rows[:16], start=1):
            self.feature_tree.insert(
                "",
                tk.END,
                values=(rank, feature_name, f"{raw_value:+.4f}", f"{scaled_value:+.4f}"),
            )

    def _update_test_activation_table(self, analysis_sample: AnalysisSample) -> None:
        """Zeigt Roh- und skalierte Werte des test_activation-Samples."""

        if self.dataset is None:
            return
        self.test_activation_tree.delete(*self.test_activation_tree.get_children())
        for feature_name, raw_value, scaled_value in zip(
            self.dataset.feature_names,
            analysis_sample.raw_sample,
            analysis_sample.scaled_sample,
            strict=True,
        ):
            self.test_activation_tree.insert(
                "",
                tk.END,
                values=(feature_name, f"{float(raw_value):+.3f}", f"{float(scaled_value):+.3f}"),
            )

    def _draw_digit_sample(self, raw_sample: np.ndarray) -> None:
        """Zeichnet ein 8x8-Digit als Pixelmatrix."""

        self.digits_canvas.delete("all")
        cell_size = 40
        padding = 20
        for row_index in range(8):
            for col_index in range(8):
                value = float(raw_sample[row_index * 8 + col_index])
                intensity = int(max(0, min(255, round(255 - (value / 16.0) * 255))))
                color = f"#{intensity:02x}{intensity:02x}{intensity:02x}"
                x0 = padding + col_index * cell_size
                y0 = padding + row_index * cell_size
                x1 = x0 + cell_size - 2
                y1 = y0 + cell_size - 2
                self.digits_canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="#94a3b8")
                self.digits_canvas.create_text(
                    (x0 + x1) / 2,
                    (y0 + y1) / 2,
                    text=str(int(value)),
                    fill="#0f172a" if value < 10 else "white",
                    font=("Helvetica", 9, "bold"),
                )

    def _copy_current_digit_to_custom(self) -> None:
        """Kopiert das aktuelle Datensatz-Digit in den editierbaren Custom-Modus."""

        if self.dataset is None or self.dataset.name != "digits":
            return
        X_raw, _ = self._get_current_raw_split_arrays()
        sample_index = min(max(self.sample_index_var.get(), 0), len(X_raw) - 1)
        self.raw_custom_digit = np.asarray(X_raw[sample_index], dtype=np.float64).copy()
        self.use_custom_sample_var.set(True)
        self._refresh_views()

    def _clear_custom_digit(self) -> None:
        """Setzt das eigene Digit auf leere Pixel zurueck."""

        self.raw_custom_digit = np.zeros(64, dtype=np.float64)
        self.use_custom_sample_var.set(True)
        self._refresh_views()

    def _on_digits_canvas_clicked(self, event: tk.Event) -> None:
        """Aendert im Custom-Digits-Modus den Wert eines Pixels per Klick."""

        if self.dataset is None or self.dataset.name != "digits" or not self.use_custom_sample_var.get():
            return
        cell_size = 40
        padding = 20
        col_index = int((event.x - padding) // cell_size)
        row_index = int((event.y - padding) // cell_size)
        if row_index < 0 or row_index >= 8 or col_index < 0 or col_index >= 8:
            return
        flat_index = row_index * 8 + col_index
        current_value = float(self.raw_custom_digit[flat_index])
        current_position = MANUAL_DIGIT_VALUES.index(current_value) if current_value in MANUAL_DIGIT_VALUES else 0
        next_value = MANUAL_DIGIT_VALUES[(current_position + 1) % len(MANUAL_DIGIT_VALUES)]
        self.raw_custom_digit[flat_index] = next_value
        self._refresh_views()

    def _update_neuron_detail(self) -> None:
        """Aktualisiert die Detailansicht fuer das selektierte Hidden-Neuron."""

        if self.model is None or self.dataset is None or self.current_layout is None:
            self._set_detail_text("Noch kein Modell oder Layout verfuegbar.")
            return

        analysis_sample = self._get_analysis_sample()
        layer_index, neuron_index = self.selected_hidden
        inspection = self.model.inspect_hidden_neuron(
            X=analysis_sample.scaled_sample.reshape(1, -1),
            sample_index=0,
            layer_index=layer_index,
            neuron_index=neuron_index,
            split_name=analysis_sample.source_label,
            input_labels=self.dataset.feature_names if layer_index == 0 else None,
        )
        detail_text = self._build_neuron_detail_text(inspection, analysis_sample)
        self._set_detail_text(detail_text)
        self._update_activation_curve_plot(inspection)

    def _build_neuron_detail_text(
        self, inspection: NeuronInspection, analysis_sample: AnalysisSample
    ) -> str:
        """Formatiert die Inspektion eines Neurons als Lerntext."""

        outgoing_sorted = sorted(
            inspection.outgoing_weights, key=lambda item: abs(item[1]), reverse=True
        )[:6]
        visible_terms = inspection.input_terms if len(inspection.input_terms) <= 6 else inspection.top_terms
        equation = " + ".join(
            f"({term.source_value:+.3f} * {term.weight:+.3f})" for term in visible_terms
        )
        if len(visible_terms) < len(inspection.input_terms):
            equation += " + ..."
        lines = [
            f"Neuron: L{inspection.layer_index + 1} n{inspection.neuron_index}",
            f"Aktivierung: {inspection.activation_name}",
            f"Formel: {inspection.activation_formula}",
            f"Quelle des Samples: {analysis_sample.source_label}",
            f"Echtes Ziel: {analysis_sample.actual_target_name}",
            f"Analyse-Ziel: {analysis_sample.effective_target_name}",
            "",
            "Schritt 1: gewichtete Summe",
            "---------------------------",
            "z = sum_i (eingang_i * gewicht_i) + bias",
            f"Kompakte Rechnung: {equation} + ({inspection.bias:+.3f})",
            f"bias = {inspection.bias:+.6f}",
            f"z    = {inspection.pre_activation:+.6f}",
            "",
            "Schritt 2: Aktivierung",
            "----------------------",
            f"a = {inspection.activation_name}(z) = {inspection.output_value:+.6f}",
            f"Ableitung an dieser Stelle = {inspection.derivative:+.6f}",
            "",
            "Staerkste Summanden von z",
            "-------------------------",
        ]
        for rank, term in enumerate(inspection.top_terms, start=1):
            lines.append(
                f"{rank:02d}. {term.source_label}: wert={term.source_value:+.6f}, "
                f"gewicht={term.weight:+.6f}, beitrag={term.contribution:+.6f}"
            )

        if len(inspection.input_terms) > len(inspection.top_terms):
            lines.append(
                f"... {len(inspection.input_terms) - len(inspection.top_terms)} weitere Beitraege ausgeblendet."
            )

        lines.extend(["", "Staerkste ausgehende Gewichte", "-----------------------------"])
        for label, weight in outgoing_sorted:
            lines.append(f"{label}: gewicht={weight:+.6f}")

        lines.extend(
            [
                "",
                "Interpretation",
                "--------------",
                "Positive Summanden schieben z nach oben, negative nach unten.",
                "Erst danach entscheidet die Aktivierungsfunktion, wie stark dieses z in eine",
                "Ausgabe a ueberfuehrt wird. Genau hier zeigen sich die Unterschiede zwischen",
                "ReLU, tanh, sigmoid und leaky_relu besonders deutlich.",
            ]
        )
        return "\n".join(lines)

    def _set_detail_text(self, text: str) -> None:
        """Schreibt Text in die schreibgeschuetzte Detailansicht."""

        for text_widget in (self.detail_text, self.live_detail_text):
            text_widget.configure(state=tk.NORMAL)
            text_widget.delete("1.0", tk.END)
            text_widget.insert("1.0", text)
            text_widget.configure(state=tk.DISABLED)

    def _update_activation_curve_plot(self, inspection: NeuronInspection) -> None:
        """Zeichnet die Aktivierungsfunktion des selektierten Neurons mit aktuellem z-Wert."""

        activation_function = ACTIVATIONS[inspection.activation_name].forward
        self.activation_axis.clear()
        z_values = np.linspace(-4.0, 4.0, 200)
        y_values = activation_function(z_values)
        self.activation_axis.plot(z_values, y_values, color=ACTIVATION_COLORS[inspection.activation_name], linewidth=2)
        self.activation_axis.axvline(inspection.pre_activation, color="#111827", linestyle="--", linewidth=1)
        self.activation_axis.scatter(
            [inspection.pre_activation],
            [inspection.output_value],
            color="#111827",
            zorder=3,
        )
        self.activation_axis.set_title(f"{inspection.activation_name}(z)")
        self.activation_axis.set_xlabel("z")
        self.activation_axis.set_ylabel("a")
        self.activation_axis.grid(alpha=0.25)
        self.activation_figure.tight_layout()
        self.activation_canvas.draw_idle()

    def _update_stepper_view(self) -> None:
        """Aktualisiert den Forward/Backward-Stepper fuer das aktuelle Sample."""

        if self.model is None or self.dataset is None:
            self._set_stepper_text("Noch kein Modell geladen.")
            return

        analysis_sample = self._get_analysis_sample()
        trace = self.model.trace_sample(
            analysis_sample.scaled_sample.reshape(1, -1),
            target_index=analysis_sample.effective_target_index,
        )
        self.step_entries = self._build_step_entries(trace, analysis_sample)
        if not self.step_entries:
            self._set_stepper_text("Noch keine Schrittspur verfuegbar.")
            return

        current_index = min(max(self.step_index_var.get(), 0), len(self.step_entries) - 1)
        self.step_index_var.set(current_index)
        title, text = self.step_entries[current_index]
        self.step_status_label.configure(text=f"Schritt {current_index + 1}/{len(self.step_entries)}: {title}")
        self._set_stepper_text(text)

    def _build_step_entries(
        self, trace: SampleTrace, analysis_sample: AnalysisSample
    ) -> list[tuple[str, str]]:
        """Erzeugt textuelle Schrittkarten fuer den Stepper."""

        entries: list[tuple[str, str]] = []
        entries.append(
            (
                "Input",
                "\n".join(
                    [
                        f"Quelle: {analysis_sample.source_label}",
                        f"Echtes Ziel: {analysis_sample.actual_target_name}",
                        f"Analyse-Ziel: {analysis_sample.effective_target_name}",
                        "",
                        "Skalierte Eingaben:",
                        ", ".join(f"x{index}={value:+.4f}" for index, value in enumerate(analysis_sample.scaled_sample)),
                    ]
                ),
            )
        )

        for layer_trace in trace.forward_layers:
            entries.append(
                (
                    f"Forward L{layer_trace.layer_index + 1}",
                    "\n".join(
                        [
                            f"Hidden-Layer L{layer_trace.layer_index + 1}",
                            f"Aktivierungen im Layer: {', '.join(layer_trace.activation_names)}",
                            "",
                            "Input in diesen Layer:",
                            ", ".join(
                                f"{value:+.4f}" for value in layer_trace.input_values[: min(12, len(layer_trace.input_values))]
                            ),
                            "",
                            "Praeaktivierungen z:",
                            ", ".join(
                                f"n{index}={value:+.4f}"
                                for index, value in enumerate(layer_trace.pre_activations)
                            ),
                            "",
                            "Aktivierungen a:",
                            ", ".join(
                                f"n{index}={value:+.4f}"
                                for index, value in enumerate(layer_trace.activations)
                            ),
                        ]
                    ),
                )
            )

        entries.append(
            (
                "Output",
                "\n".join(
                    [
                        "Output-Layer",
                        "Logits:",
                        ", ".join(f"y{index}={value:+.4f}" for index, value in enumerate(trace.logits)),
                        "",
                        "Wahrscheinlichkeiten:",
                        ", ".join(
                            f"{self.dataset.target_names[index]}={value:.4f}"
                            for index, value in enumerate(trace.probabilities)
                        ),
                        "",
                        f"Vorhersage: {self.dataset.target_names[trace.prediction_index]}",
                    ]
                ),
            )
        )

        if trace.loss is not None:
            entries.append(
                (
                    "Loss",
                    "\n".join(
                        [
                            f"Analyse-Ziel: {analysis_sample.effective_target_name}",
                            f"Cross-Entropy-Loss fuer dieses einzelne Sample: {trace.loss:.6f}",
                            "",
                            "Interpretation:",
                            "Kleine Werte bedeuten, dass die Zielklasse bereits hohe Wahrscheinlichkeit hat.",
                            "Grosse Werte bedeuten, dass das Modell fuer dieses Ziel noch stark daneben liegt.",
                        ]
                    ),
                )
            )

        if trace.output_delta is not None:
            entries.append(
                (
                    "Backward Output",
                    "\n".join(
                        [
                            "Output-Fehlervektor dL/dlogits",
                            ", ".join(
                                f"y{index}={value:+.4f}"
                                for index, value in enumerate(trace.output_delta)
                            ),
                        ]
                    ),
                )
            )

        for backward_trace in trace.backward_layers:
            entries.append(
                (
                    f"Backward {backward_trace.layer_label}",
                    "\n".join(
                        [
                            f"Layer: {backward_trace.layer_label}",
                            "Delta-Werte:",
                            ", ".join(
                                f"{value:+.4f}" for value in backward_trace.deltas[: min(16, len(backward_trace.deltas))]
                            ),
                            "",
                            f"Norm des Gewichtsgradienten: {backward_trace.weight_gradient_norm:.6f}",
                            f"Norm des Biasgradienten:    {backward_trace.bias_gradient_norm:.6f}",
                        ]
                    ),
                )
            )

        return entries

    def _set_stepper_text(self, text: str) -> None:
        """Schreibt Text in den Stepper-Tab."""

        self.stepper_text.configure(state=tk.NORMAL)
        self.stepper_text.delete("1.0", tk.END)
        self.stepper_text.insert("1.0", text)
        self.stepper_text.configure(state=tk.DISABLED)

    def _step_prev(self) -> None:
        """Geht einen Schritt im Stepper zurueck."""

        if self.step_entries:
            self.step_index_var.set(max(self.step_index_var.get() - 1, 0))
            self._update_stepper_view()

    def _step_next(self) -> None:
        """Geht einen Schritt im Stepper weiter."""

        if self.step_entries:
            self.step_index_var.set(min(self.step_index_var.get() + 1, len(self.step_entries) - 1))
            self._update_stepper_view()

    def _step_reset(self) -> None:
        """Setzt den Stepper auf den ersten Schritt zurueck."""

        self.step_index_var.set(0)
        self._update_stepper_view()

    def _update_compare_text(self) -> None:
        """Aktualisiert den Vergleich zwischen Baseline und aktuellem Experiment."""

        if self.baseline_model is None or self.baseline_layout is None or self.dataset is None or self.model is None:
            self._set_compare_text("Noch keine Baseline gespeichert. Nutze 'Als Baseline speichern'.")
            return

        analysis_sample = self._get_analysis_sample()
        current_probabilities = self.model.predict_proba(analysis_sample.scaled_sample.reshape(1, -1))[0]
        baseline_probabilities = self.baseline_model.predict_proba(analysis_sample.scaled_sample.reshape(1, -1))[0]
        current_prediction = self.dataset.target_names[int(np.argmax(current_probabilities))]
        baseline_prediction = self.dataset.target_names[int(np.argmax(baseline_probabilities))]

        lines = [
            "Baseline vs. aktuelles Experiment",
            "=================================",
            "",
            f"Aktueller Benchmark: {self.dataset.name}",
            f"Hidden-Sizes aktuell:  {format_hidden_sizes(self.model.hidden_sizes)}",
            f"Hidden-Sizes Baseline: {format_hidden_sizes(self.baseline_model.hidden_sizes)}",
            f"Layout aktuell:  {self.current_layout.to_compact_spec()}",
            f"Layout Baseline: {self.baseline_layout.to_compact_spec()}",
            "",
            f"Sample-Quelle: {analysis_sample.source_label}",
            f"Baseline-Vorhersage: {baseline_prediction}",
            f"Aktuelle Vorhersage: {current_prediction}",
            "",
            "Wahrscheinlichkeiten Baseline:",
            ", ".join(
                f"{self.dataset.target_names[index]}={baseline_probabilities[index]:.3f}"
                for index in range(len(baseline_probabilities))
            ),
            "",
            "Wahrscheinlichkeiten aktuell:",
            ", ".join(
                f"{self.dataset.target_names[index]}={current_probabilities[index]:.3f}"
                for index in range(len(current_probabilities))
            ),
            "",
        ]

        if self.baseline_training_result is not None:
            lines.extend(
                [
                    "Baseline-Metriken",
                    "-----------------",
                    f"Epochen:  {self.baseline_completed_epochs}",
                    f"Test-Acc: {self.baseline_training_result.test_metrics['accuracy']:.4f}",
                    f"Test-Loss:{self.baseline_training_result.test_metrics['loss']:.4f}",
                    "",
                ]
            )
        if self.training_result is not None:
            lines.extend(
                [
                    "Aktuelle Metriken",
                    "-----------------",
                    f"Epochen:  {self.completed_epochs}",
                    f"Test-Acc: {self.training_result.test_metrics['accuracy']:.4f}",
                    f"Test-Loss:{self.training_result.test_metrics['loss']:.4f}",
                    "",
                ]
            )

        if self.baseline_layout.hidden_sizes == self.current_layout.hidden_sizes:
            changes = diff_layouts(self.baseline_layout, self.current_layout)
            lines.append("Layout-Diff")
            lines.append("-----------")
            if changes:
                lines.extend(
                    f"L{change.layer_index + 1} n{change.neuron_index}: {change.before} -> {change.after}"
                    for change in changes
                )
            else:
                lines.append("Keine Unterschiede.")
            lines.append("")

        current_neighbors = len(generate_single_step_neighbors(self.current_layout))
        lines.extend(
            [
                "Natural-Computing-Sicht",
                "-----------------------",
                f"Aktueller Zustand: {self.current_layout.to_compact_spec()}",
                f"Anzahl direkter Single-Step-Nachbarn: {current_neighbors}",
                "Eine typische Bewertungsfunktion waere spaeter Validation-Accuracy oder Validation-Loss.",
            ]
        )

        if self.training_result is not None:
            predictions = self.model.predict(self.dataset.X_test)
            confusion = self._confusion_matrix(
                self.dataset.y_test,
                predictions,
                self.dataset.output_size,
            )
            lines.extend(
                [
                    "",
                    "Confusion-Matrix auf dem Testsplit",
                    "----------------------------------",
                    self._format_confusion_matrix(confusion, self.dataset.target_names),
                    "",
                    "Unsicherste Test-Samples",
                    "------------------------",
                    *self._uncertain_sample_lines(self.dataset.X_test, self.dataset.y_test),
                ]
            )

        self._set_compare_text("\n".join(lines))

    def _set_compare_text(self, text: str) -> None:
        """Schreibt Text in den Vergleichs-Tab."""

        self.compare_text.configure(state=tk.NORMAL)
        self.compare_text.delete("1.0", tk.END)
        self.compare_text.insert("1.0", text)
        self.compare_text.configure(state=tk.DISABLED)

    def _confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        num_classes: int,
    ) -> np.ndarray:
        """Berechnet eine kleine Confusion-Matrix ohne Zusatzbibliothek."""

        matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        for true_label, predicted_label in zip(y_true, y_pred, strict=True):
            matrix[int(true_label), int(predicted_label)] += 1
        return matrix

    def _format_confusion_matrix(
        self,
        matrix: np.ndarray,
        target_names: Sequence[str],
    ) -> str:
        """Formatiert eine Confusion-Matrix kompakt als Textblock."""

        header = "true\\pred " + " ".join(f"{name[:6]:>6}" for name in target_names)
        rows = [header]
        for row_index, row in enumerate(matrix):
            rows.append(
                f"{target_names[row_index][:6]:>9} "
                + " ".join(f"{int(value):>6}" for value in row)
            )
        return "\n".join(rows)

    def _uncertain_sample_lines(
        self,
        X_split: np.ndarray,
        y_split: np.ndarray,
    ) -> list[str]:
        """Liefert die unsichersten Samples eines Splits als Textzeilen."""

        if self.dataset is None or self.model is None or len(X_split) == 0:
            return ["Keine Samples verfuegbar."]
        probabilities = self.model.predict_proba(X_split)
        predictions = np.argmax(probabilities, axis=1)
        confidence = np.max(probabilities, axis=1)
        ordering = np.argsort(confidence)[: min(5, len(confidence))]
        return [
            (
                f"idx {int(index):03d}: true={self.dataset.target_names[int(y_split[index])]}, "
                f"pred={self.dataset.target_names[int(predictions[index])]}, "
                f"max_p={confidence[index]:.3f}"
            )
            for index in ordering
        ]

    def _update_training_plot(self) -> None:
        """Aktualisiert die eingebetteten Matplotlib-Plots."""

        if self.model is None or self.dataset is None:
            return

        analysis_sample = self._get_analysis_sample()
        cache = self.model.forward(analysis_sample.scaled_sample.reshape(1, -1))
        probabilities = cache.probabilities[0]
        predicted_index = int(np.argmax(probabilities))

        loss_axis, accuracy_axis, probability_axis, hidden_axis = self.training_axes
        for axis in self.training_axes:
            axis.clear()

        if self.training_result is not None:
            epochs = np.arange(1, len(self.training_result.history["train_loss"]) + 1)
            loss_axis.plot(epochs, self.training_result.history["train_loss"], label="Train", linewidth=2)
            loss_axis.plot(epochs, self.training_result.history["val_loss"], label="Val", linewidth=2)
            loss_axis.set_title("Loss")
            loss_axis.set_xlabel("Epoche")
            loss_axis.grid(alpha=0.3)
            loss_axis.legend()

            accuracy_axis.plot(epochs, self.training_result.history["train_acc"], label="Train", linewidth=2)
            accuracy_axis.plot(epochs, self.training_result.history["val_acc"], label="Val", linewidth=2)
            accuracy_axis.set_title("Accuracy")
            accuracy_axis.set_xlabel("Epoche")
            accuracy_axis.grid(alpha=0.3)
            accuracy_axis.legend()
        else:
            loss_axis.text(0.5, 0.5, "Noch kein Training", ha="center", va="center")
            accuracy_axis.text(0.5, 0.5, "Noch kein Training", ha="center", va="center")
            loss_axis.set_axis_off()
            accuracy_axis.set_axis_off()

        class_labels = [str(name) for name in self.dataset.target_names]
        bar_colors = [
            PREDICTION_COLOR if index == predicted_index else "#94a3b8"
            for index in range(len(class_labels))
        ]
        if analysis_sample.effective_target_index is not None:
            bar_colors[analysis_sample.effective_target_index] = ANALYSIS_TARGET_COLOR
        probability_axis.bar(class_labels, probabilities, color=bar_colors)
        probability_axis.set_ylim(0.0, 1.0)
        probability_axis.set_title("Klassenwahrscheinlichkeiten")
        probability_axis.set_ylabel("p")
        probability_axis.tick_params(axis="x", rotation=25)

        hidden_values = np.concatenate([layer_activation[0] for layer_activation in cache.activations])
        hidden_labels = [
            f"L{layer_index + 1}:{neuron_index}"
            for layer_index, layer_activation in enumerate(cache.activations)
            for neuron_index in range(len(layer_activation[0]))
        ]
        hidden_colors = [
            ACTIVATION_COLORS[activation_name]
            for layer in self.current_layout.layers
            for activation_name in layer
        ]
        hidden_axis.bar(hidden_labels, hidden_values, color=hidden_colors)
        hidden_axis.set_title("Aktivierungen des aktuellen Samples")
        hidden_axis.tick_params(axis="x", rotation=55)
        hidden_axis.axhline(0.0, color="#64748b", linewidth=1)

        self.training_figure.tight_layout()
        self.training_canvas_widget.draw_idle()

    def _update_help_text(self) -> None:
        """Aktualisiert den Lernhilfe-Tab passend zum aktuellen Kontext."""

        benchmark = self.benchmark_var.get()
        help_lines = [
            "Lernhilfe fuer den Playground",
            "=============================",
            "",
            "Empfohlener Ablauf",
            "------------------",
            "1. Waehle einen Benchmark oder den test_activation-Modus.",
            "2. Schaue im Tab 'Input & Ziel', welche Daten aktuell ins Netz fliessen.",
            "3. Veraendere Aktivierungen pro Layer oder pro Neuron.",
            "4. Trainiere in kleinen Schritten mit 1 oder 10 Epochen.",
            "5. Beobachte, wie sich Vorhersage, Loss und Aktivierungen veraendern.",
            "6. Klicke auf einzelne Hidden-Neuronen und lies ihre lokale Rechnung.",
            "",
            "Wichtige Konzepte",
            "-----------------",
            "z = gewichtete Summe + bias",
            "a = Aktivierungsfunktion(z)",
            "Trainieren aendert Gewichte und Biases, nicht den Datensatz.",
            "Das aktuell gezeigte Sample ist ein Analysefenster, nicht der ganze Trainingsprozess.",
            "",
            "Aktueller Benchmark",
            "-------------------",
            *self._benchmark_help_lines(benchmark),
            "",
            "Parameter-Lexikon",
            "-----------------",
            *self._parameter_help_lines(),
            "",
        ]

        if benchmark == "digits":
            help_lines.extend(
                [
                    "Digits-spezifische Hinweise",
                    "---------------------------",
                    "Jedes Sample ist ein 8x8 Pixelbild mit Werten von 0 bis 16.",
                    "Mit 'Eigenes Sample verwenden' kannst du Pixel selbst setzen.",
                    "So siehst du direkt, wie das Netz auf veraenderte Bilder reagiert.",
                    "Beobachte besonders, wie sich Vorhersagewahrscheinlichkeiten aendern, wenn nur wenige Pixel geaendert werden.",
                    "",
                ]
            )
        elif benchmark == "test_activation":
            help_lines.extend(
                [
                    "test_activation-spezifische Hinweise",
                    "-----------------------------------",
                    "Hier geht es weniger um Benchmarking als um rohe Berechnungen.",
                    "Setze wenige Inputs manuell und beobachte, wie sich z und a in jedem Neuron aendern.",
                    "Das ist ideal, um Unterschiede zwischen ReLU, tanh, sigmoid und leaky_relu zu sehen.",
                    "Rechts neben der Netzwerkvisualisierung siehst du immer die Rechnung des aktuell angeklickten Hidden-Neurons.",
                    "Achte darauf, wie sich kleine Input-Aenderungen in der gewichteten Summe z und danach in der Aktivierung a auswirken.",
                    "",
                ]
            )
        else:
            help_lines.extend(
                [
                    "Tabellen-Benchmarks",
                    "-------------------",
                    "Die Tabellenansicht zeigt dir die wichtigsten Eingabefeatures des aktuellen Samples.",
                    "Im Netzwerk werden dennoch alle Features beruecksichtigt, auch wenn nicht alle einzeln angezeigt werden.",
                    "Die sichtbaren Inputs im Netzwerk sind ein didaktischer Ausschnitt, damit man staerkste Einfluesse leichter lesen kann.",
                    "",
                ]
            )

        help_lines.extend(
            [
                "Beobachtungsfragen fuer Studierende",
                "-----------------------------------",
                "Welche Aktivierungen fuehren bei diesem Sample zu grossen oder kleinen Ausgaben?",
                "Welche Eingabefeatures beeinflussen das selektierte Neuron am staerksten?",
                "Wie veraendert Training die Klassenausgabe und die Hidden-Aktivierungen?",
                "Wann wirkt sigmoid eher saettigend, wann ist ReLU sparsamer?",
            ]
        )

        self.help_text.configure(state=tk.NORMAL)
        self.help_text.delete("1.0", tk.END)
        self.help_text.insert("1.0", "\n".join(help_lines))
        self.help_text.configure(state=tk.DISABLED)

    def _build_dataset_summary_text(self) -> str:
        """Erzeugt eine didaktische Kurzbeschreibung des aktuell geladenen Benchmarks."""

        if self.dataset is None:
            return "Kein Datensatz geladen."
        return (
            f"{describe_dataset(self.dataset)}\n"
            f"Trainingsziel: Das Modell lernt auf allen Beispielen aus dem Trainingssplit.\n"
            f"Klassen: {', '.join(self.dataset.target_names)}\n"
            f"Benchmark-Hinweis: {BENCHMARK_DESCRIPTIONS[self.dataset.name]}"
        )

    def _benchmark_help_lines(self, benchmark: str) -> list[str]:
        """Liefert erklaerende Stichpunkte zum aktuell gewaehlten Benchmark."""

        if benchmark == "breast_cancer":
            return [
                "Name: breast_cancer",
                "30 numerische Features, 2 Klassen.",
                "Didaktischer Nutzen: einfacher Einstieg in Klassifikation mit tabellarischen Daten.",
                "Beobachte hier besonders, wie schnell Val- und Test-Accuracy stabil werden.",
                "Gut geeignet, um den Effekt von ReLU vs. sigmoid in einem einfachen Setting zu sehen.",
            ]
        if benchmark == "wine":
            return [
                "Name: wine",
                "13 numerische Features, 3 Klassen.",
                "Didaktischer Nutzen: ueberschaubarer Mehrklassenfall mit gut lesbaren Lernkurven.",
                "Sehr geeignet fuer Vergleiche von Layouts, weil Unterschiede oft klar sichtbar werden.",
                "Beobachte, ob gemischte Aktivierungen im Hidden-Bereich zu anderen Wahrscheinlichkeitsprofilen fuehren.",
            ]
        if benchmark == "digits":
            return [
                "Name: digits",
                "64 Eingaben aus einem 8x8 Bild, 10 Klassen.",
                "Didaktischer Nutzen: Eingabestrom, Ziel und Vorhersage sind bildhaft nachvollziehbar.",
                "Ideal, um visuell zu sehen, welche Pixelkonfigurationen zu unsicheren Vorhersagen fuehren.",
                "Beobachte, wie sich kleine Pixelveraenderungen durch das Netz fortpflanzen.",
            ]
        return [
            "Name: test_activation",
            "3 kuenstliche Inputs, 2 Klassen.",
            "Didaktischer Nutzen: minimales Rechenlabor fuer rohe Aktivierungs- und Summenrechnungen.",
            "Hier ist weniger die absolute Accuracy wichtig, sondern das Verstehen von z, a und Ableitungen.",
            "Ideal, um durch Anklicken einzelner Neuronen die komplette lokale Rechnung zu diskutieren.",
        ]

    def _parameter_help_lines(self) -> list[str]:
        """Liefert ein kompaktes Lexikon der veraenderbaren GUI-Parameter."""

        return [
            "Benchmark: waehlt Datensatz und damit auch Art der Eingaben und Anzahl der Klassen.",
            "Hidden-Layer: Anzahl der Neuronen in den versteckten Schichten. Mehr Neuronen oder mehr Layer bedeuten mehr Kapazitaet, aber auch mehr Unuebersicht.",
            "Datensplit: bestimmt, aus welchem Split das aktuell sichtbare Analyse-Sample stammt. Das Training selbst verwendet weiterhin den Trainingssplit.",
            "Sample-Index: waehlt genau ein Beispiel aus, das du im Netz verfolgst.",
            "Eigenes Sample verwenden: erlaubt bei digits und test_activation manuelle Eingaben, um gezielt Reaktionen des Netzes zu studieren.",
            "Analyse-Ziel: setzt das Ziel, gegen das der Loss in der Ansicht berechnet wird. So kann man auch absichtlich ein 'falsches' Ziel untersuchen.",
            "Layer auf ...: setzt die Aktivierungsfunktion fuer einen kompletten Hidden-Layer auf einen Schlag.",
            "Layer / Neuron / Aktivierung / Neuron setzen: waehlt ein einzelnes Hidden-Neuron aus und aendert gezielt dessen Aktivierungsfunktion.",
            "Cycle: schaltet das ausgewaehlte Neuron in der festen Reihenfolge relu -> tanh -> sigmoid -> leaky_relu weiter.",
            "Modus: Einsteiger blendet Komplexitaet aus, Experte zeigt tiefe Steuerung und Vergleichswerkzeuge.",
            "Epochen: wie oft das Training den gesamten Trainingssplit durchlaeuft.",
            "Lernrate: Schrittweite des Lernens. Zu klein lernt langsam, zu gross kann instabil werden.",
            "Batch-Groesse: wie viele Trainingsbeispiele pro Gewichtsupdate gemeinsam verarbeitet werden.",
            "Weight-Scale: Groessenordnung der zufaelligen Startgewichte. Beeinflusst, wie stark Aktivierungen schon zu Beginn ausschlagen.",
            "Seed: sorgt fuer reproduzierbare Daten-Splits und reproduzierbare Initialisierung.",
            "Neu initialisieren: setzt das Modell mit aktuellen Einstellungen und neuen Startgewichten zurueck.",
            "1 Epoche / 10 Epochen / N Epochen trainieren: trainiert schrittweise, damit man Veraenderungen beobachten kann.",
            "Als Baseline speichern: friert einen Referenzzustand ein, mit dem das aktuelle Experiment spaeter verglichen wird.",
        ]

    def _redraw_network(self) -> None:
        """Zeichnet das aktuelle Netzwerk als interaktive Canvas-Grafik."""

        if self.model is None or self.dataset is None or self.current_layout is None:
            return

        analysis_sample = self._get_analysis_sample()
        cache = self.model.forward(analysis_sample.scaled_sample.reshape(1, -1))
        probabilities = cache.probabilities[0]
        predicted_index = int(np.argmax(probabilities))

        width = max(self.canvas.winfo_width(), 860)
        height = max(self.canvas.winfo_height(), 520)
        self.canvas.delete("all")
        self.node_tags.clear()

        padding_top = 70
        padding_bottom = 70
        column_positions = self._column_positions_for_width(width, self.model.num_hidden_layers)

        input_indices = self._displayed_input_indices(analysis_sample)
        input_positions = self._compute_positions(
            len(input_indices), column_positions[0], padding_top, height - padding_bottom
        )
        hidden_positions = [
            self._compute_positions(len(layer), column_positions[layer_index + 1], padding_top, height - padding_bottom)
            for layer_index, layer in enumerate(self.current_layout.layers)
        ]
        output_positions = self._compute_positions(
            self.dataset.output_size, column_positions[-1], padding_top, height - padding_bottom
        )

        self._draw_network_background(input_positions, hidden_positions, output_positions)
        self._draw_network_headers(input_indices, analysis_sample, predicted_index, column_positions)
        self._draw_input_nodes(input_indices, input_positions, analysis_sample)
        for layer_index, layer_positions in enumerate(hidden_positions):
            self._draw_hidden_nodes(layer_index, layer_positions)
        self._draw_output_nodes(output_positions, probabilities, predicted_index, analysis_sample)
        self._draw_selected_connection_overlay(
            input_indices,
            input_positions,
            hidden_positions,
            output_positions,
            analysis_sample,
        )
        self._draw_legend(width, height)

        self.network_summary_var.set(
            f"Aktuelles Sample: {analysis_sample.source_label} | "
            f"Echtes Ziel: {analysis_sample.actual_target_name} | "
            f"Analyse-Ziel: {analysis_sample.effective_target_name} | "
            f"Vorhersage: {self.dataset.target_names[predicted_index]}"
        )

    def _draw_network_background(
        self,
        input_positions: Sequence[tuple[float, float]],
        hidden_positions: Sequence[Sequence[tuple[float, float]]],
        output_positions: Sequence[tuple[float, float]],
    ) -> None:
        """Zeichnet alle Basisverbindungen in abgeschwaechter Form."""

        if not hidden_positions:
            return
        for input_position in input_positions:
            for hidden_position in hidden_positions[0]:
                self.canvas.create_line(*input_position, *hidden_position, fill=CONNECTION_COLOR, width=1)
        for layer_index in range(len(hidden_positions) - 1):
            for previous_position in hidden_positions[layer_index]:
                for next_position in hidden_positions[layer_index + 1]:
                    self.canvas.create_line(*previous_position, *next_position, fill=CONNECTION_COLOR, width=1)
        for last_hidden_position in hidden_positions[-1]:
            for output_position in output_positions:
                self.canvas.create_line(*last_hidden_position, *output_position, fill=CONNECTION_COLOR, width=1)

    def _draw_network_headers(
        self,
        input_indices: Sequence[int],
        analysis_sample: AnalysisSample,
        predicted_index: int,
        column_positions: Sequence[float],
    ) -> None:
        """Zeichnet Ueberschriften ueber alle Netzspalten."""

        self.canvas.create_text(
            column_positions[0],
            30,
            text=f"Eingaben ({len(input_indices)}/{self.dataset.input_size})",
            font=("Helvetica", 12, "bold"),
            fill=NETWORK_TEXT_COLOR,
        )
        for layer_index, layer in enumerate(self.current_layout.layers):
            self.canvas.create_text(
                column_positions[layer_index + 1],
                30,
                text=f"Hidden L{layer_index + 1} ({len(layer)})",
                font=("Helvetica", 12, "bold"),
                fill=NETWORK_TEXT_COLOR,
            )
        self.canvas.create_text(
            column_positions[-1],
            30,
            text=f"Output ({self.dataset.output_size})",
            font=("Helvetica", 12, "bold"),
            fill=NETWORK_TEXT_COLOR,
        )

        self.canvas.create_text(
            (column_positions[0] + column_positions[-1]) / 2.0,
            52,
            text=(
                f"Sample: {analysis_sample.source_label} | "
                f"Vorhersage: {self.dataset.target_names[predicted_index]} | "
                f"Analyse-Ziel: {analysis_sample.effective_target_name}"
            ),
            font=("Helvetica", 10),
            fill=NETWORK_TEXT_COLOR,
        )

    def _draw_input_nodes(
        self,
        input_indices: Sequence[int],
        input_positions: Sequence[tuple[float, float]],
        analysis_sample: AnalysisSample,
    ) -> None:
        """Zeichnet die aktuellen Eingaben, inklusive ihrer Werte."""

        radius = self._node_radius(len(input_positions))
        for display_index, feature_index in enumerate(input_indices):
            x_coord, y_coord = input_positions[display_index]
            scaled_value = float(analysis_sample.scaled_sample[feature_index])
            raw_value = float(analysis_sample.raw_sample[feature_index])
            fill_color = self._value_to_input_color(scaled_value)
            self.canvas.create_oval(
                x_coord - radius,
                y_coord - radius,
                x_coord + radius,
                y_coord + radius,
                fill=fill_color,
                outline="#64748b",
                width=1.5,
            )
            self.canvas.create_text(
                x_coord,
                y_coord - 2,
                text=f"x{feature_index}",
                font=("Helvetica", 8, "bold"),
                fill=NETWORK_TEXT_COLOR,
            )
            self.canvas.create_text(
                x_coord,
                y_coord + radius + 9,
                text=f"roh={raw_value:+.2f}",
                font=("Helvetica", 8),
                fill=NETWORK_TEXT_COLOR,
            )

    def _draw_hidden_nodes(self, layer_index: int, positions: Sequence[tuple[float, float]]) -> None:
        """Zeichnet Hidden-Neuronen inklusive Klick-Handler."""

        radius = self._node_radius(len(positions))
        layer = self.current_layout.layers[layer_index]
        for neuron_index, (x_coord, y_coord) in enumerate(positions):
            activation_name = layer[neuron_index]
            outline = SELECTED_OUTLINE_COLOR if self.selected_hidden == (layer_index, neuron_index) else "#1f2937"
            width = 3 if self.selected_hidden == (layer_index, neuron_index) else 1.5
            tag = f"hidden_{layer_index}_{neuron_index}"
            self.node_tags[tag] = (layer_index, neuron_index)
            self.canvas.create_oval(
                x_coord - radius,
                y_coord - radius,
                x_coord + radius,
                y_coord + radius,
                fill=ACTIVATION_COLORS[activation_name],
                outline=outline,
                width=width,
                tags=(tag,),
            )
            self.canvas.create_text(x_coord, y_coord - 2, text=str(neuron_index), font=("Helvetica", 9, "bold"), fill="white", tags=(tag,))
            self.canvas.create_text(
                x_coord,
                y_coord + radius + 10,
                text=activation_name,
                font=("Helvetica", 8),
                fill=NETWORK_TEXT_COLOR,
                tags=(tag,),
            )
            self.canvas.tag_bind(tag, "<Button-1>", lambda _event, li=layer_index, ni=neuron_index: self._on_hidden_node_clicked(li, ni))

    def _draw_output_nodes(
        self,
        positions: Sequence[tuple[float, float]],
        probabilities: np.ndarray,
        predicted_index: int,
        analysis_sample: AnalysisSample,
    ) -> None:
        """Zeichnet Ausgabeknoten mit Markierungen fuer Prediction und Ziele."""

        radius = self._node_radius(len(positions))
        for output_index, (x_coord, y_coord) in enumerate(positions):
            outline = "#0f172a"
            width = 1.5
            if output_index == predicted_index:
                outline = PREDICTION_COLOR
                width = 3
            if analysis_sample.actual_target_index is not None and output_index == analysis_sample.actual_target_index:
                self.canvas.create_text(x_coord, y_coord - radius - 14, text="echt", fill=TARGET_COLOR, font=("Helvetica", 8, "bold"))
            if analysis_sample.effective_target_index is not None and output_index == analysis_sample.effective_target_index:
                self.canvas.create_text(x_coord, y_coord - radius - 2, text="analyse", fill=ANALYSIS_TARGET_COLOR, font=("Helvetica", 8, "bold"))

            self.canvas.create_oval(
                x_coord - radius,
                y_coord - radius,
                x_coord + radius,
                y_coord + radius,
                fill=OUTPUT_NODE_COLOR,
                outline=outline,
                width=width,
            )
            self.canvas.create_text(x_coord, y_coord - 2, text=str(output_index), font=("Helvetica", 9, "bold"), fill="white")
            self.canvas.create_text(
                x_coord,
                y_coord + radius + 10,
                text=f"{self.dataset.target_names[output_index]} ({probabilities[output_index]:.2f})",
                font=("Helvetica", 8),
                fill=NETWORK_TEXT_COLOR,
            )

    def _draw_selected_connection_overlay(
        self,
        input_indices: Sequence[int],
        input_positions: Sequence[tuple[float, float]],
        hidden_positions: Sequence[Sequence[tuple[float, float]]],
        output_positions: Sequence[tuple[float, float]],
        analysis_sample: AnalysisSample,
    ) -> None:
        """Hebt fuer das selektierte Hidden-Neuron die wichtigsten Verbindungen hervor."""

        layer_index, neuron_index = self.selected_hidden
        inspection = self.model.inspect_hidden_neuron(
            X=analysis_sample.scaled_sample.reshape(1, -1),
            sample_index=0,
            layer_index=layer_index,
            neuron_index=neuron_index,
            split_name=analysis_sample.source_label,
            input_labels=self.dataset.feature_names if layer_index == 0 else None,
        )

        target_position = hidden_positions[layer_index][neuron_index]

        if layer_index == 0:
            displayed_feature_to_position = {
                feature_index: input_positions[position_index]
                for position_index, feature_index in enumerate(input_indices)
            }
            strongest_incoming = sorted(
                inspection.input_terms, key=lambda term: abs(term.contribution), reverse=True
            )[: min(8, len(displayed_feature_to_position))]
            feature_lookup = {
                feature_name: index for index, feature_name in enumerate(self.dataset.feature_names)
            }
            for term in strongest_incoming:
                if term.source_label not in feature_lookup:
                    continue
                feature_index = feature_lookup[term.source_label]
                if feature_index not in displayed_feature_to_position:
                    continue
                self._draw_highlight_line(displayed_feature_to_position[feature_index], target_position, term.contribution)

            strongest_outgoing = sorted(
                enumerate(self.model.weights[layer_index + 1][neuron_index]),
                key=lambda item: abs(item[1]),
                reverse=True,
            )[:8]
            for next_neuron_index, weight in strongest_outgoing:
                if layer_index + 1 < self.model.num_hidden_layers:
                    self._draw_highlight_line(
                        target_position,
                        hidden_positions[layer_index + 1][next_neuron_index],
                        float(weight),
                    )
                else:
                    self._draw_highlight_line(target_position, output_positions[next_neuron_index], float(weight))
        else:
            strongest_incoming = sorted(
                enumerate(self.model.weights[layer_index][:, neuron_index]),
                key=lambda item: abs(item[1]),
                reverse=True,
            )[:8]
            for prev_neuron_index, weight in strongest_incoming:
                self._draw_highlight_line(
                    hidden_positions[layer_index - 1][prev_neuron_index],
                    target_position,
                    float(weight),
                )

            if layer_index + 1 < self.model.num_hidden_layers:
                strongest_outgoing = sorted(
                    enumerate(self.model.weights[layer_index + 1][neuron_index]),
                    key=lambda item: abs(item[1]),
                    reverse=True,
                )[:8]
                for next_neuron_index, weight in strongest_outgoing:
                    self._draw_highlight_line(
                        target_position,
                        hidden_positions[layer_index + 1][next_neuron_index],
                        float(weight),
                    )
            else:
                strongest_outgoing = sorted(
                    enumerate(self.model.output_weight[neuron_index]),
                    key=lambda item: abs(item[1]),
                    reverse=True,
                )[: min(8, len(output_positions))]
                for output_index, weight in strongest_outgoing:
                    self._draw_highlight_line(
                        target_position,
                        output_positions[output_index],
                        float(weight),
                    )

    def _draw_highlight_line(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        signed_strength: float,
    ) -> None:
        """Zeichnet eine hervorgehobene Verbindung in Farbe und Staerke."""

        color = POSITIVE_CONNECTION_COLOR if signed_strength >= 0 else NEGATIVE_CONNECTION_COLOR
        width = 1.5 + min(4.5, abs(signed_strength) * 6.0)
        self.canvas.create_line(*start, *end, fill=color, width=width)

    def _draw_legend(self, width: int, height: int) -> None:
        """Zeichnet eine Legende fuer Aktivierungen und Ziel-Markierungen."""

        legend_y = height - 26
        x_coord = 36
        legend_step = max(110, min(150, int((width - 80) / 5)))
        for activation_name in SUPPORTED_ACTIVATIONS:
            self.canvas.create_rectangle(
                x_coord,
                legend_y - 8,
                x_coord + 14,
                legend_y + 6,
                fill=ACTIVATION_COLORS[activation_name],
                outline="#111827",
            )
            self.canvas.create_text(
                x_coord + 54,
                legend_y,
                text=f"{activation_name}",
                anchor="w",
                font=("Helvetica", 9),
                fill=NETWORK_TEXT_COLOR,
            )
            x_coord += legend_step
        self.canvas.create_text(
            width - 12,
            legend_y,
            text="Gruen = Prediction | orange Text = echtes Ziel | blaugruener Text = Analyse-Ziel",
            font=("Helvetica", 9),
            fill=NETWORK_TEXT_COLOR,
            anchor="e",
        )

    def _displayed_input_indices(self, analysis_sample: AnalysisSample) -> list[int]:
        """Waehlt aus, welche Eingaben im Netzwerk explizit angezeigt werden."""

        if self.dataset is None or self.model is None:
            return []
        max_inputs = 12
        if self.dataset.input_size <= max_inputs:
            return list(range(self.dataset.input_size))

        layer_index, neuron_index = self.selected_hidden
        if layer_index == 0:
            inspection = self.model.inspect_hidden_neuron(
                X=analysis_sample.scaled_sample.reshape(1, -1),
                sample_index=0,
                layer_index=layer_index,
                neuron_index=neuron_index,
                split_name=analysis_sample.source_label,
                input_labels=self.dataset.feature_names,
            )
            feature_lookup = {
                feature_name: index for index, feature_name in enumerate(self.dataset.feature_names)
            }
            indices = [
                feature_lookup[term.source_label]
                for term in inspection.top_terms
                if term.source_label in feature_lookup
            ]
            if len(indices) >= max_inputs:
                return indices[:max_inputs]
        return list(np.linspace(0, self.dataset.input_size - 1, num=max_inputs, dtype=int))

    def _compute_positions(
        self,
        count: int,
        x_coord: float,
        top: float,
        bottom: float,
    ) -> list[tuple[float, float]]:
        """Berechnet gleichmaessig verteilte Positionen fuer eine Spalte."""

        if count <= 0:
            return []
        span = bottom - top
        step = span / (count + 1)
        return [(x_coord, top + (index + 1) * step) for index in range(count)]

    def _column_positions_for_width(self, width: int, num_hidden_layers: int) -> list[float]:
        """Berechnet x-Positionen aller Netzspalten relativ zur Canvas-Breite."""

        left = 70.0
        right = max(float(width) - 70.0, left + 400.0)
        return list(np.linspace(left, right, num=num_hidden_layers + 2))

    def _node_radius(self, count: int) -> float:
        """Waehlt einen lesbaren Radius abhaengig von der Knotenzahl."""

        if count <= 8:
            return 20
        if count <= 16:
            return 16
        if count <= 32:
            return 12
        return 9

    def _value_to_input_color(self, value: float) -> str:
        """Kodiert skalierte Eingabewerte als Farbe.

        Positive Werte werden blau, negative roetlich, nahe null grau dargestellt.
        """

        clipped = max(-2.0, min(2.0, value))
        if clipped >= 0:
            ratio = clipped / 2.0
            red = int(210 - 110 * ratio)
            green = int(226 - 90 * ratio)
            blue = int(238)
        else:
            ratio = abs(clipped) / 2.0
            red = int(238)
            green = int(226 - 90 * ratio)
            blue = int(226 - 110 * ratio)
        return f"#{red:02x}{green:02x}{blue:02x}"

    def _on_hidden_node_clicked(self, layer_index: int, neuron_index: int) -> None:
        """Reagiert auf Klicks auf Hidden-Neuronen."""

        self.selected_hidden = (layer_index, neuron_index)
        self.neuron_layer_var.set(f"L{layer_index + 1}")
        self._update_neuron_dropdowns()
        self.neuron_index_var.set(str(neuron_index))
        if self.current_layout is not None:
            self.neuron_activation_var.set(self.current_layout.layers[layer_index][neuron_index])
        self._refresh_views()
