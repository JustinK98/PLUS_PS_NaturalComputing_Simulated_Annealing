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
from pathlib import Path
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
    generate_neighbors,
    generate_single_step_neighbors,
    parse_layout_spec,
)
from annealing import AnnealingConfig, AnnealingState, AnnealingStep, annealing_stop_reasons
from annealing_objectives import (
    ObjectiveConfig,
    ObjectiveEvaluation,
    SUPPORTED_OBJECTIVES,
    LayoutObjectiveEvaluator,
)
from annealing_runner import AnnealingRunner
from annealing_schedules import SUPPORTED_COOLING_SCHEDULES
from benchmarks import DatasetBundle, describe_dataset, load_benchmark
from configs import (
    DEFAULT_ANNEALING_CANDIDATE_EPOCHS,
    DEFAULT_ANNEALING_COOLING_PARAMETER,
    DEFAULT_ANNEALING_COOLING_SCHEDULE,
    DEFAULT_ANNEALING_ITERATIONS_PER_TEMPERATURE,
    DEFAULT_ANNEALING_MAX_STEPS,
    DEFAULT_ANNEALING_MIN_TEMPERATURE,
    DEFAULT_ANNEALING_NEIGHBORHOODS,
    DEFAULT_ANNEALING_OBJECTIVE,
    DEFAULT_ANNEALING_START_TEMPERATURE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_EXPERIMENT_OUTPUT_SUBDIR,
    DEFAULT_EXPERIMENT_RANDOM_SEARCH_SAMPLES,
    DEFAULT_EXPERIMENT_SEARCH_TYPE,
    DEFAULT_GUI_APP_MODE,
    DEFAULT_LAYOUT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_WEIGHT_SCALE,
    DatasetConfig,
    MAX_HIDDEN_LAYERS,
    MIN_HIDDEN_LAYERS,
    OUTPUT_DIR,
    SUPPORTED_ACTIVATIONS,
    SUPPORTED_BENCHMARKS,
    SUPPORTED_GUI_APP_MODES,
    SUPPORTED_GUI_LANGUAGES,
    SUPPORTED_GUI_MODES,
    TrainingConfig,
    default_hidden_sizes,
    format_hidden_sizes,
)
from experiment_builder import (
    PRIMARY_METRIC_LABELS,
    SUGGESTED_TUNING_VALUES,
    SUPPORTED_EXPERIMENT_RUN_MODES,
    ExperimentDefinition,
)
from experiment_plots import build_experiment_overview_figure
from experiment_runner import ExperimentRunner
from model import ModularMLP, NeuronInspection, SampleTrace
from results_store import load_experiment_results
from search_spaces import (
    SUPPORTED_SEARCH_TYPES,
    SearchSpaceDefinition,
    SearchValueDefinition,
    expand_search_space,
)
from terminal_viz import render_layout, render_layout_diff
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

MANUAL_DIGIT_VALUES = (0.0, 4.0, 8.0, 12.0, 16.0)

LANGUAGE_LABELS = {"de": "Deutsch", "en": "English"}
APP_MODE_LABELS = {
    "de": {
        "demo": "Demo Mode",
        "playground": "Playground Mode",
        "experiment_builder": "Experiment Builder",
    },
    "en": {
        "demo": "Demo Mode",
        "playground": "Playground Mode",
        "experiment_builder": "Experiment Builder",
    },
}
MODE_LABELS = {
    "de": {"beginner": "Einsteiger", "expert": "Experte"},
    "en": {"beginner": "Beginner", "expert": "Expert"},
}
RUN_MODE_LABELS = {
    "de": {
        "manual_training": "Manuelles Training",
        "simulated_annealing": "Simulated Annealing",
    },
    "en": {
        "manual_training": "Manual Training",
        "simulated_annealing": "Simulated Annealing",
    },
}
SEARCH_TYPE_LABELS = {
    "de": {
        "none": "Keine Suche",
        "grid_search": "Grid Search",
        "random_search": "Random Search",
    },
    "en": {
        "none": "No Search",
        "grid_search": "Grid Search",
        "random_search": "Random Search",
    },
}
SEED_STRATEGY_LABELS = {
    "de": {
        "count_range": "Anzahl + Start-Seed",
        "explicit_list": "Explizite Seed-Liste",
    },
    "en": {
        "count_range": "Count + Start Seed",
        "explicit_list": "Explicit Seed List",
    },
}
SEARCH_VALUE_KIND_LABELS = {
    "de": {
        "fixed": "Fest",
        "list": "Liste",
        "range": "Bereich",
    },
    "en": {
        "fixed": "Fixed",
        "list": "List",
        "range": "Range",
    },
}

EXPERIMENT_BUILDER_TEXTS = {
    "de": {
        "builder_mode_hint": (
            "Der Experiment Builder ist fuer reproduzierbare Multi-Seed-Experimente, "
            "Grid/Random Search, JSON-Ergebnisse und spaetere Wiederanalyse gedacht."
        ),
        "builder_setup_title": "1. Experiment Setup",
        "builder_setup_hint": (
            "Hier definierst du Benchmark, Hidden-Layer, Layout, Run-Modus und den Namen "
            "des Experiments. Diese Basis gilt fuer alle spaeteren Runs."
        ),
        "builder_experiment_name": "Experiment-Name",
        "builder_run_mode": "Run-Modus",
        "builder_layout_choice": "Aktuelle Layouts",
        "builder_layout_editor_title": "Layout-Editor",
        "builder_layout_editor_hint": (
            "Im Expertenmodus kannst du das Startlayout hier wie in Demo und Playground "
            "pro Layer oder pro Neuron bearbeiten. Diese Einstellungen definieren das "
            "Startlayout des naechsten Builder-Runs."
        ),
        "builder_primary_metric": "Primaere Metrik",
        "builder_primary_metric_hint": "Ranking und Aggregation erfolgen standardmaessig ueber Validation, nicht ueber Test.",
        "builder_seeds_title": "2. Seeds",
        "builder_seeds_hint": (
            "Mehrere Seeds zeigen, wie stabil eine Konfiguration wirklich ist. "
            "Gleiche Konfiguration und gleicher Seed bleiben reproduzierbar."
        ),
        "builder_seed_strategy": "Seed-Strategie",
        "builder_seed_count": "Anzahl Seeds",
        "builder_seed_start": "Start-Seed",
        "builder_seed_list": "Seed-Liste",
        "builder_seed_preview": "Effektive Seeds",
        "builder_training_title": "3. Training Setup",
        "builder_training_hint": (
            "Diese Hyperparameter gelten fuer normale Trainingslaeufe und fuer die "
            "Kandidatenbewertung in SA."
        ),
        "builder_sa_title": "4. Simulated Annealing",
        "builder_sa_hint": (
            "Dieser Bereich wird nur fuer SA-Experimente verwendet. Er bestimmt Suchraum, "
            "Bewertungsbudget und Annealing-Verlauf."
        ),
        "builder_tuning_title": "5. Hyperparameter-Tuning",
        "builder_tuning_hint": (
            "Aus einer Einzelkonfiguration wird hier ein diskreter Suchraum. Fest bedeutet "
            "ein einzelner Wert, Liste mehrere explizite Werte, Bereich eine diskrete Range."
        ),
        "builder_search_type": "Search-Typ",
        "builder_random_samples": "Random Samples",
        "builder_storage_title": "6. Storage",
        "builder_storage_hint": (
            "Experimente koennen als JSON gespeichert und spaeter wieder geladen und "
            "analysiert werden."
        ),
        "builder_output_dir": "Output-Ordner",
        "builder_save_json": "JSON speichern",
        "builder_load_path": "Pfad zu Manifest/Ordner",
        "builder_run_title": "7. Run Control",
        "builder_run_hint": (
            "Validiere zuerst die Konfiguration. Danach kannst du das Experiment ausfuehren "
            "oder gespeicherte Ergebnisse erneut laden."
        ),
        "builder_validate_button": "Experiment pruefen",
        "builder_run_button": "Experiment starten",
        "builder_load_results_button": "Ergebnisse laden",
        "builder_summary_title": "Experiment Summary",
        "builder_runs_tab": "Run List",
        "builder_analysis_tab": "Multi-Seed Analysis",
        "builder_detail_tab": "Per-Seed Detail",
        "builder_plots_tab": "Plots",
        "builder_overview_tab": "Summary",
        "builder_help_tab": "Builder Help",
        "builder_run_tree_hint": "Alle ausgefuehrten Runs. Auswahl aktualisiert Detailansicht und Plot-Fokus.",
        "builder_analysis_hint": "Aggregation ueber Seeds und Konfigurationen, mit Fokus auf Validation-Metriken.",
        "builder_detail_hint": "Detaillierte Sicht auf einen einzelnen Run inklusive History und SA-Zusatzdaten.",
        "builder_plot_hint": "Plots werden direkt aus den gespeicherten Resultaten rekonstruiert.",
        "builder_no_results": "Noch keine Builder-Ergebnisse verfuegbar.",
        "builder_status_idle": "Noch kein Experiment ausgefuehrt oder geladen.",
        "builder_validation_ok": "Konfiguration ist gueltig. Ausgefuehrte Runs: {run_count}.",
        "builder_validation_error_title": "Builder-Konfiguration ungueltig",
        "builder_run_error_title": "Experiment Builder Fehler",
        "builder_load_error_title": "Ergebnisse konnten nicht geladen werden",
    },
    "en": {
        "builder_mode_hint": (
            "The Experiment Builder is meant for reproducible multi-seed experiments, "
            "grid/random search, JSON result storage, and later re-analysis."
        ),
        "builder_setup_title": "1. Experiment Setup",
        "builder_setup_hint": (
            "Define benchmark, hidden layers, layout, run mode, and experiment name here. "
            "This base configuration applies to all later runs."
        ),
        "builder_experiment_name": "Experiment Name",
        "builder_run_mode": "Run Mode",
        "builder_layout_choice": "Current Layouts",
        "builder_layout_editor_title": "Layout Editor",
        "builder_layout_editor_hint": (
            "In expert mode you can edit the builder start layout here just like in Demo "
            "and Playground, per layer or per neuron. These settings define the start "
            "layout for the next builder run."
        ),
        "builder_primary_metric": "Primary Metric",
        "builder_primary_metric_hint": "Ranking and aggregation use validation by default, not test metrics.",
        "builder_seeds_title": "2. Seeds",
        "builder_seeds_hint": (
            "Multiple seeds show how stable a configuration really is. "
            "Same configuration and same seed remain reproducible."
        ),
        "builder_seed_strategy": "Seed Strategy",
        "builder_seed_count": "Seed Count",
        "builder_seed_start": "Start Seed",
        "builder_seed_list": "Seed List",
        "builder_seed_preview": "Effective Seeds",
        "builder_training_title": "3. Training Setup",
        "builder_training_hint": (
            "These hyperparameters apply to normal training runs and to candidate evaluation in SA."
        ),
        "builder_sa_title": "4. Simulated Annealing",
        "builder_sa_hint": (
            "This section is only used for SA experiments. It controls search space, "
            "evaluation budget, and annealing behavior."
        ),
        "builder_tuning_title": "5. Hyperparameter Tuning",
        "builder_tuning_hint": (
            "This turns one configuration into a discrete search space. Fixed means one value, "
            "list means explicit values, range means a discrete numeric range."
        ),
        "builder_search_type": "Search Type",
        "builder_random_samples": "Random Samples",
        "builder_storage_title": "6. Storage",
        "builder_storage_hint": (
            "Experiments can be stored as JSON and later loaded and analyzed again."
        ),
        "builder_output_dir": "Output Directory",
        "builder_save_json": "Save JSON",
        "builder_load_path": "Path to Manifest/Directory",
        "builder_run_title": "7. Run Control",
        "builder_run_hint": (
            "Validate the configuration first. After that you can execute the experiment "
            "or reload stored results."
        ),
        "builder_validate_button": "Validate Experiment",
        "builder_run_button": "Run Experiment",
        "builder_load_results_button": "Load Results",
        "builder_summary_title": "Experiment Summary",
        "builder_runs_tab": "Run List",
        "builder_analysis_tab": "Multi-Seed Analysis",
        "builder_detail_tab": "Per-Seed Detail",
        "builder_plots_tab": "Plots",
        "builder_overview_tab": "Summary",
        "builder_help_tab": "Builder Help",
        "builder_run_tree_hint": "All executed runs. Selection updates detail view and plot focus.",
        "builder_analysis_hint": "Aggregation across seeds and configurations, focused on validation metrics.",
        "builder_detail_hint": "Detailed view of one run including history and SA extras.",
        "builder_plot_hint": "Plots are reconstructed directly from stored results.",
        "builder_no_results": "No builder results available yet.",
        "builder_status_idle": "No experiment executed or loaded yet.",
        "builder_validation_ok": "Configuration is valid. Scheduled runs: {run_count}.",
        "builder_validation_error_title": "Invalid builder configuration",
        "builder_run_error_title": "Experiment builder error",
        "builder_load_error_title": "Could not load results",
    },
}

EXPERIMENT_BUILDER_INFO_TEXTS = {
    "de": {
        "builder_setup": "Ein Experiment kombiniert Benchmark, Architektur, Layout, Seeds und Run-Modus zu einer reproduzierbaren Definition.",
        "builder_seeds": "Mehrere Seeds helfen, Zufallseffekte sichtbar zu machen. Fuer faire Vergleiche sollten Konfigurationen mit denselben Seeds bewertet werden.",
        "builder_training": "Diese Trainingsparameter steuern normale Runs und auch die Kandidatenbewertung innerhalb von Simulated Annealing.",
        "builder_sa": "Im SA-Modus wird ein Aktivierungs-Layout optimiert. Jeder Kandidat wird kurz trainiert und ueber Validation bewertet.",
        "builder_search": "Grid Search testet alle diskreten Kombinationen. Random Search sampelt reproduzierbar eine Teilmenge dieses diskreten Suchraums.",
        "builder_storage": "Ein Experiment wird als Manifest, Summary und einzelne Run-Dateien gespeichert. So lassen sich Ergebnisse spaeter wieder laden.",
        "builder_run_control": "Validation prueft nur die Builder-Konfiguration. Der eigentliche Lauf fuehrt danach alle Seeds und Konfigurationen aus.",
        "builder_primary_metric": "Die primaere Metrik steuert Ranking und Best/Worst-Bewertung. Fuer die Auswahl wird Validation genutzt, nicht Test.",
        "builder_seed_strategy": "Du kannst entweder eine explizite Seed-Liste eingeben oder aus Startwert und Anzahl eine Liste erzeugen.",
        "builder_layout_choice": "Diese Liste sammelt das aktuell eingestellte Setup-Layout und rekonstruierbare Layouts aus dem selektierten Builder-Run. Die Auswahl aktualisiert rechts die Netzvorschau.",
        "builder_layout_editor": "Der Layout-Editor bearbeitet das Startlayout fuer den naechsten Builder-Run. Im Expertenmodus kannst du ganze Layer oder einzelne Neuronen direkt belegen. Die rechte Vorschau zeigt das konfigurierte Startlayout nur dann direkt, wenn in 'Aktuelle Layouts' das Setup-Layout aktiv ist.",
        "builder_search_value": "Fest = ein Wert. Liste = mehrere explizite Kandidaten. Bereich = diskrete Werte von start bis stop mit festem Schritt.",
        "builder_output_dir": "Unter diesem Ordner werden Manifest, Summary und einzelne Run-Dateien abgelegt.",
        "builder_load_path": "Du kannst direkt einen Experimentordner oder die manifest.json eines gespeicherten Experiments laden.",
    },
    "en": {
        "builder_setup": "An experiment combines benchmark, architecture, layout, seeds, and run mode into one reproducible definition.",
        "builder_seeds": "Multiple seeds make random effects visible. For fair comparisons, configurations should be evaluated with the same seeds.",
        "builder_training": "These training parameters control normal runs and also candidate evaluation inside simulated annealing.",
        "builder_sa": "In SA mode an activation layout is optimized. Every candidate is trained briefly and scored on validation.",
        "builder_search": "Grid search tests all discrete combinations. Random search reproducibly samples a subset of the same discrete search space.",
        "builder_storage": "An experiment is stored as a manifest, summary, and individual run files. This makes later re-loading possible.",
        "builder_run_control": "Validation only checks the builder configuration. The actual run then executes all seeds and configurations.",
        "builder_primary_metric": "The primary metric controls ranking and best/worst selection. Validation is used for selection, not test.",
        "builder_seed_strategy": "You can either enter an explicit seed list or generate one from a start seed and a count.",
        "builder_layout_choice": "This list collects the current setup layout and reconstructable layouts from the selected builder run. Choosing one updates the network preview on the right.",
        "builder_layout_editor": "The layout editor changes the start layout for the next builder run. In expert mode you can assign activations to full layers or to individual neurons. The network preview on the right only follows this configuration directly while 'Current Layouts' is set to the setup layout.",
        "builder_search_value": "Fixed = one value. List = explicit candidate values. Range = discrete values from start to stop with a fixed step.",
        "builder_output_dir": "Manifest, summary, and run JSON files are written below this directory.",
        "builder_load_path": "You can load either an experiment directory directly or the manifest.json of a stored experiment.",
    },
}

GUI_TEXTS = {
    "de": {
        "window_title": "Activation Playground GUI",
        "info_dialog_title": "Info",
        "guide_window_title": "Anleitung zum Activation Playground",
        "mode_frame_title": "Modus und Sprache",
        "app_mode_label": "Arbeitsmodus",
        "mode_hint": (
            "Waehle zuerst den Arbeitsmodus: Demo Mode fuer Verstehen und Training, Playground Mode "
            "fuer einzelne SA-Laeufe und Experiment Builder fuer reproduzierbare Multi-Seed- und Such-Experimente. "
            "Die Detailstufe bestimmt danach, wie viele Werkzeuge sichtbar sind."
        ),
        "mode_label": "Detailstufe",
        "language_label": "Sprache",
        "workflow_frame_title": "Lernfluss",
        "guide_button": "Anleitung oeffnen",
        "preset_frame_title": "Schnellstart-Presets",
        "preset_hint": "Presets setzen sinnvolle Startkonfigurationen, damit man direkt etwas sieht.",
        "preset_wine_demo": "Wine Demo",
        "preset_digits_demo": "Digits Demo",
        "preset_all_relu": "Nur ReLU",
        "preset_mixed": "Gemischt",
        "preset_test_activation": "Test Activation",
        "preset_reset": "Reset",
        "experiment_frame_title": "1. Daten und Beispiel",
        "experiment_hint": (
            "Hier legst du fest, welcher Datensatz trainiert wird und welches einzelne Sample "
            "du im Netzwerk gerade beobachtest."
        ),
        "benchmark_label": "Benchmark",
        "hidden_layers_label": "Hidden-Layer",
        "add_layer_button": "Layer hinzufuegen",
        "remove_layer_button": "Letzten Layer loeschen",
        "split_label": "Datensplit",
        "sample_index_label": "Sample-Index",
        "custom_sample_label": "Eigenes Sample verwenden",
        "analysis_target_label": "Analyse-Ziel",
        "benchmark_defaults_button": "Benchmark-Defaults",
        "load_experiment_button": "Experiment laden",
        "random_sample_button": "Zufaelliges Sample",
        "layout_frame_title": "2. Architektur und Aktivierungen",
        "layout_hint": (
            "Hier veraenderst du die Aktivierungen. Das aendert nicht den Datensatz, "
            "sondern nur die Rechenweise der Hidden-Neuronen."
        ),
        "current_layout_label": "Aktuelles Layout",
        "layer_overview_label": "Layer-Uebersicht",
        "layer_fill_label": "Layerweise setzen",
        "layer_label": "Layer",
        "neuron_label": "Neuron",
        "activation_label": "Aktivierung",
        "set_neuron_button": "Neuron setzen",
        "cycle_button": "Cycle",
        "training_frame_title": "3. Training und Beobachtung",
        "training_hint": (
            "Training aendert Gewichte und Biases. Das aktuelle Sample dient nur zur Analyse; "
            "gelernt wird immer auf dem gesamten Trainingssplit."
        ),
        "epochs_label": "Epochen",
        "learning_rate_label": "Lernrate",
        "batch_size_label": "Batch-Groesse",
        "weight_scale_label": "Weight-Scale",
        "seed_label": "Seed",
        "reinitialize_button": "Neu initialisieren",
        "train_1_button": "1 Epoche",
        "train_10_button": "10 Epochen",
        "train_n_button": "N Epochen trainieren",
        "save_baseline_button": "Als Baseline speichern",
        "refresh_view_button": "Nur Ansicht aktualisieren",
        "objective_frame_title": "3. Ziel und Bewertung",
        "objective_hint": (
            "Hier legst du fest, wie ein Layout fuer Simulated Annealing bewertet wird. "
            "Jeder Kandidat wird dazu kurz trainiert und auf der Validation gemessen."
        ),
        "objective_label": "Zielmetrik",
        "candidate_epochs_label": "Epochen pro Kandidat",
        "shuffle_label": "Trainingsdaten mischen",
        "neighborhood_frame_title": "4. Suchraum und Nachbarschaft",
        "neighborhood_hint": (
            "Nachbarn sind kleine Veraenderungen des Aktivierungs-Layouts. Simulated Annealing "
            "bewegt sich durch genau solche lokalen Aenderungen durch den Suchraum."
        ),
        "neighbor_set_label": "Einzelnes Neuron aendern",
        "neighbor_fill_label": "Ganzen Layer fuellen",
        "neighbor_swap_label": "Zwei Neuronen tauschen",
        "annealing_frame_title": "5. Annealing-Konfiguration",
        "annealing_hint": (
            "Die Temperatur steuert, wie risikofreudig die Suche ist. Hohe Temperaturen lassen "
            "auch schlechtere Kandidaten eher zu, niedrige Temperaturen machen die Suche strenger."
        ),
        "start_temperature_label": "Starttemperatur",
        "cooling_schedule_label": "Abkuehlung",
        "cooling_parameter_label": "Cooling-Parameter",
        "iterations_per_temperature_label": "Iterationen pro Temperatur",
        "max_steps_label": "Maximale Schritte",
        "min_temperature_label": "Mindesttemperatur",
        "run_frame_title": "6. Laufsteuerung",
        "run_hint": (
            "Du kannst den Suchlauf schrittweise oder komplett ausfuehren. Jeder Schritt zeigt, "
            "welcher Nachbar vorgeschlagen wurde und warum er akzeptiert oder verworfen wurde."
        ),
        "evaluate_start_button": "Startzustand bewerten",
        "anneal_step_button": "1 SA-Schritt",
        "anneal_10_button": "10 SA-Schritte",
        "anneal_run_button": "Bis Ende laufen",
        "anneal_reset_button": "Suche zuruecksetzen",
        "status_frame_title": "4. Live-Status",
        "anneal_status_frame_title": "7. Annealing-Status",
        "summary_frame_title": "Was passiert gerade?",
        "canvas_frame_title": "Netzwerk-Visualisierung",
        "network_toolbar_hint": (
            "Diese Hilfen erklaeren direkt an der Netzansicht, wie Knoten, Linien, Farben "
            "und die Aktivierungskurve zu lesen sind."
        ),
        "network_info_button": "Netz lesen",
        "activation_curve_info_button": "Kurve lesen",
        "live_inspection_title": "Live-Neuron-Inspektion",
        "live_inspection_hint": (
            "Klicke im Netz auf ein Hidden-Neuron. Rechts erscheint sofort dessen lokale "
            "Rechnung mit z, Aktivierung, Ableitung und den staerksten Beitragen."
        ),
        "notebook_frame_title": "Analyse-Ansichten",
        "tab_input": "Input & Ziel",
        "tab_plot": "Training & Plotting",
        "tab_stepper": "Forward/Backward",
        "tab_detail": "Neuron-Inspektion",
        "tab_compare": "Vergleich",
        "tab_annealing": "Annealing",
        "tab_help": "Lernhilfe",
        "input_tab_hint": (
            "Dieser Tab zeigt, was gerade in das Netz eingespeist wird. Bei digits siehst du die "
            "8x8 Pixel, bei den anderen Benchmarks die wichtigsten Feature-Werte."
        ),
        "digits_hint": (
            "Links-klick auf ein Feld aendert im Eigenmodus dessen Helligkeit. "
            "So kannst du eigene Digits ausprobieren."
        ),
        "copy_sample_button": "Aktuelles Sample kopieren",
        "clear_digit_button": "Eigenes Digit leeren",
        "generic_features_hint": (
            "Die Tabelle zeigt die staerksten Eingabefeatures des aktuellen Samples. "
            "Rohwert = urspruengliche Eingabe, skaliert = Wert nach Standardisierung."
        ),
        "rank_header": "#",
        "feature_header": "Feature",
        "raw_header": "Rohwert",
        "scaled_header": "Skaliert",
        "test_activation_hint": (
            "Im test_activation-Modus kannst du rohe Eingabewerte direkt setzen. "
            "Das ist ideal, um Aktivierungsfunktionen ohne grossen Benchmark-Kontext zu verstehen."
        ),
        "test_activation_rule_hint": (
            "Regel des Lernmodus: Klasse 1, wenn mindestens zwei Inputs deutlich aktiv sind. "
            "Du kannst aber separat ein Analyse-Ziel setzen, um den Loss zu studieren."
        ),
        "plot_tab_hint": (
            "Die Plots zeigen sowohl die Trainingsgeschichte als auch den aktuellen Zustand des "
            "ausgewaehlten Samples. Training aendert Gewichte, nicht die gezeigte Sample-Auswahl."
        ),
        "plot_info_hint": (
            "Zu jedem Plot gibt es eine eigene Kurz-Erklaerung. So laesst sich direkt ablesen, "
            "was Achsen, Farben und Werte bedeuten."
        ),
        "plot_overview_info_button": "Plot-Ueberblick",
        "loss_plot_info_button": "Loss",
        "accuracy_plot_info_button": "Accuracy",
        "probability_plot_info_button": "Wahrscheinlichkeiten",
        "sample_activation_plot_info_button": "Sample-Aktivierungen",
        "annealing_tab_hint": (
            "Dieser Tab zeigt den Suchverlauf des Simulated Annealing: aktueller Zustand, "
            "Kandidat, bester Zustand, Temperatur und Verlauf der Optimierung."
        ),
        "stepper_tab_hint": (
            "Dieser Tab zerlegt den aktuellen Forward- und Backward-Pass fuer genau das sichtbare "
            "Sample in einzelne didaktische Schritte."
        ),
        "step_back_button": "Zurueck",
        "step_forward_button": "Weiter",
        "step_reset_button": "Auf Anfang",
        "compare_tab_hint": (
            "Speichere einen Zustand als Baseline und vergleiche danach Layout, Metriken und "
            "Vorhersage des aktuellen Experiments mit genau diesem Referenzpunkt."
        ),
        "detail_tab_hint": (
            "Dieselbe Neuron-Inspektion wie rechts neben der Netzwerkansicht, aber mit mehr Platz "
            "zum Lesen und Scrollen."
        ),
        "guide_link_text": (
            "Die ausfuehrliche Anleitung erklaert Demo Mode, Playground Mode, "
            "Experiment Builder, Simulated Annealing, typische Arbeitsweisen und die wichtigsten Begriffe."
        ),
        "target_auto_label": "(echtes Ziel)",
        "target_none_label": "(kein Ziel)",
        "max_layers_title": "Maximale Layerzahl erreicht",
        "max_layers_message": "Die GUI unterstuetzt didaktisch bis zu {max_layers} Hidden-Layer.",
        "min_layers_title": "Mindestens ein Hidden-Layer",
        "min_layers_message": "Das Lernmodell benoetigt mindestens einen Hidden-Layer.",
        "load_error_title": "Experiment konnte nicht geladen werden",
        "train_error_title": "Training fehlgeschlagen",
        "annealing_error_title": "Simulated Annealing fehlgeschlagen",
        "guide_intro_title": "Anleitung",
    },
    "en": {
        "window_title": "Activation Playground GUI",
        "info_dialog_title": "Info",
        "guide_window_title": "Activation Playground Guide",
        "mode_frame_title": "Mode and Language",
        "app_mode_label": "Workspace",
        "mode_hint": (
            "Choose the workspace first: Demo Mode for understanding and training, Playground Mode "
            "for single simulated-annealing runs, and Experiment Builder for reproducible multi-seed and search experiments. "
            "The detail level then controls how many tools stay visible."
        ),
        "mode_label": "Detail Level",
        "language_label": "Language",
        "workflow_frame_title": "Learning Flow",
        "guide_button": "Open Guide",
        "preset_frame_title": "Quick Start Presets",
        "preset_hint": "Presets apply sensible starting configurations so you can see results immediately.",
        "preset_wine_demo": "Wine Demo",
        "preset_digits_demo": "Digits Demo",
        "preset_all_relu": "ReLU Only",
        "preset_mixed": "Mixed",
        "preset_test_activation": "Test Activation",
        "preset_reset": "Reset",
        "experiment_frame_title": "1. Data and Sample",
        "experiment_hint": (
            "Here you choose which dataset is trained and which single sample is currently "
            "observed inside the network."
        ),
        "benchmark_label": "Benchmark",
        "hidden_layers_label": "Hidden Layers",
        "add_layer_button": "Add Layer",
        "remove_layer_button": "Remove Last Layer",
        "split_label": "Data Split",
        "sample_index_label": "Sample Index",
        "custom_sample_label": "Use Custom Sample",
        "analysis_target_label": "Analysis Target",
        "benchmark_defaults_button": "Benchmark Defaults",
        "load_experiment_button": "Load Experiment",
        "random_sample_button": "Random Sample",
        "layout_frame_title": "2. Architecture and Activations",
        "layout_hint": (
            "Here you change the activations. This does not modify the dataset, only how the "
            "hidden neurons compute."
        ),
        "current_layout_label": "Current Layout",
        "layer_overview_label": "Layer Overview",
        "layer_fill_label": "Set Whole Layer",
        "layer_label": "Layer",
        "neuron_label": "Neuron",
        "activation_label": "Activation",
        "set_neuron_button": "Set Neuron",
        "cycle_button": "Cycle",
        "training_frame_title": "3. Training and Observation",
        "training_hint": (
            "Training changes weights and biases. The current sample is only for analysis; "
            "learning always happens on the full training split."
        ),
        "epochs_label": "Epochs",
        "learning_rate_label": "Learning Rate",
        "batch_size_label": "Batch Size",
        "weight_scale_label": "Weight Scale",
        "seed_label": "Seed",
        "reinitialize_button": "Reinitialize",
        "train_1_button": "1 Epoch",
        "train_10_button": "10 Epochs",
        "train_n_button": "Train N Epochs",
        "save_baseline_button": "Store as Baseline",
        "refresh_view_button": "Refresh View Only",
        "objective_frame_title": "3. Objective and Evaluation",
        "objective_hint": (
            "Choose how a layout is scored for simulated annealing. Each candidate is trained "
            "briefly and then measured on the validation split."
        ),
        "objective_label": "Objective",
        "candidate_epochs_label": "Epochs per Candidate",
        "shuffle_label": "Shuffle Training Data",
        "neighborhood_frame_title": "4. Search Space and Neighborhood",
        "neighborhood_hint": (
            "Neighbors are small changes to the activation layout. Simulated annealing explores "
            "the search space by moving through exactly these local modifications."
        ),
        "neighbor_set_label": "Change one neuron",
        "neighbor_fill_label": "Fill whole layer",
        "neighbor_swap_label": "Swap two neurons",
        "annealing_frame_title": "5. Annealing Configuration",
        "annealing_hint": (
            "Temperature controls how risk-seeking the search is. High temperatures make it easier "
            "to accept worse candidates, low temperatures make the search stricter."
        ),
        "start_temperature_label": "Start Temperature",
        "cooling_schedule_label": "Cooling Schedule",
        "cooling_parameter_label": "Cooling Parameter",
        "iterations_per_temperature_label": "Iterations per Temperature",
        "max_steps_label": "Maximum Steps",
        "min_temperature_label": "Minimum Temperature",
        "run_frame_title": "6. Run Control",
        "run_hint": (
            "You can execute the search step by step or run it to completion. Each step explains "
            "which neighbor was proposed and why it was accepted or rejected."
        ),
        "evaluate_start_button": "Evaluate Start State",
        "anneal_step_button": "1 SA Step",
        "anneal_10_button": "10 SA Steps",
        "anneal_run_button": "Run to Completion",
        "anneal_reset_button": "Reset Search",
        "status_frame_title": "4. Live Status",
        "anneal_status_frame_title": "7. Annealing Status",
        "summary_frame_title": "What is happening right now?",
        "canvas_frame_title": "Network Visualization",
        "network_toolbar_hint": (
            "These helpers explain right next to the network view how to read nodes, lines, "
            "colors, and the activation curve."
        ),
        "network_info_button": "Read Network",
        "activation_curve_info_button": "Read Curve",
        "live_inspection_title": "Live Neuron Inspection",
        "live_inspection_hint": (
            "Click a hidden neuron in the network. Its local computation immediately appears on "
            "the right, including z, activation, derivative, and strongest contributions."
        ),
        "notebook_frame_title": "Analysis Views",
        "tab_input": "Input & Target",
        "tab_plot": "Training & Plotting",
        "tab_stepper": "Forward/Backward",
        "tab_detail": "Neuron Inspection",
        "tab_compare": "Comparison",
        "tab_annealing": "Annealing",
        "tab_help": "Learning Help",
        "input_tab_hint": (
            "This tab shows what is currently fed into the network. For digits you see the 8x8 "
            "pixels, for the other benchmarks the strongest feature values."
        ),
        "digits_hint": (
            "Left-click a cell to change its brightness in custom mode. "
            "This lets you build your own digits."
        ),
        "copy_sample_button": "Copy Current Sample",
        "clear_digit_button": "Clear Custom Digit",
        "generic_features_hint": (
            "The table shows the strongest input features of the current sample. "
            "Raw = original input, scaled = value after standardization."
        ),
        "rank_header": "#",
        "feature_header": "Feature",
        "raw_header": "Raw",
        "scaled_header": "Scaled",
        "test_activation_hint": (
            "In test_activation mode you can directly set raw input values. "
            "This is ideal for understanding activation functions without a large benchmark context."
        ),
        "test_activation_rule_hint": (
            "Rule of the learning mode: class 1 if at least two inputs are clearly active. "
            "You can still set a separate analysis target to inspect the loss."
        ),
        "plot_tab_hint": (
            "The plots show both the training history and the current state of the selected sample. "
            "Training changes weights, not the displayed sample choice."
        ),
        "plot_info_hint": (
            "Each plot has its own short explanation so you can directly interpret axes, colors, "
            "and values."
        ),
        "plot_overview_info_button": "Plot Overview",
        "loss_plot_info_button": "Loss",
        "accuracy_plot_info_button": "Accuracy",
        "probability_plot_info_button": "Probabilities",
        "sample_activation_plot_info_button": "Sample Activations",
        "annealing_tab_hint": (
            "This tab shows the simulated-annealing search process: current state, candidate, "
            "best state, temperature, and the optimization history."
        ),
        "stepper_tab_hint": (
            "This tab breaks the current forward and backward pass for exactly the visible sample "
            "into individual learning steps."
        ),
        "step_back_button": "Back",
        "step_forward_button": "Next",
        "step_reset_button": "Reset",
        "compare_tab_hint": (
            "Store one state as a baseline and compare layout, metrics, and prediction of the "
            "current experiment against that reference."
        ),
        "detail_tab_hint": (
            "The same neuron inspection as on the right side of the network view, but with more "
            "space for reading and scrolling."
        ),
        "guide_link_text": (
            "The detailed guide explains Demo Mode, Playground Mode, Experiment Builder, simulated annealing, "
            "common workflows, and the main concepts of the program."
        ),
        "target_auto_label": "(true target)",
        "target_none_label": "(no target)",
        "max_layers_title": "Maximum number of layers reached",
        "max_layers_message": "For readability, the GUI supports up to {max_layers} hidden layers.",
        "min_layers_title": "At least one hidden layer",
        "min_layers_message": "The learning model requires at least one hidden layer.",
        "load_error_title": "Could not load experiment",
        "train_error_title": "Training failed",
        "annealing_error_title": "Simulated annealing failed",
        "guide_intro_title": "Guide",
    },
}

INFO_TEXTS = {
    "de": {
        "app_mode": "Der Arbeitsmodus trennt drei Ziele: Demo Mode zum Verstehen und Beobachten, Playground Mode fuer einen einzelnen SA-Lauf und Experiment Builder fuer reproduzierbare Multi-Seed- und Search-Experimente.",
        "mode": "Der Modus steuert, wie viel Komplexitaet sichtbar ist. Einsteiger konzentriert sich auf die wichtigsten Schritte, Experte zeigt tiefe Eingriffe und Vergleichswerkzeuge.",
        "language": "Hier schaltest du die komplette GUI zwischen Deutsch und Englisch um. Die Ansichten, Hinweise, Hilfe-Texte und Dialoge werden dabei gemeinsam aktualisiert.",
        "workflow": "Der Lernfluss zeigt die empfohlene Reihenfolge fuer eine Sitzung. Ueber die Anleitung bekommst du eine einfache Gesamterklaerung des Programms von Anfang bis Ende.",
        "presets": "Presets laden schnell eine funktionierende Ausgangskonfiguration. Sie sind praktisch, um ohne langes Einstellen sofort mit Beobachtung und Training zu beginnen.",
        "benchmark": "Der Benchmark bestimmt Datensatz, Anzahl der Klassen und Art der Eingaben. Davon haengt auch ab, welche Visualisierung im Input-Tab gezeigt wird.",
        "hidden_layers": "Hier bestimmst du Groesse und Anzahl der Hidden-Layer. Mehr Layer oder mehr Neuronen geben dem Modell mehr Kapazitaet, machen die Ansicht aber auch komplexer.",
        "split": "Der Datensplit bestimmt nur, aus welchem Bereich das aktuell sichtbare Beispiel stammt. Das Training selbst verwendet weiterhin immer den Trainingssplit.",
        "sample_index": "Mit dem Sample-Index waehlst du genau ein Beispiel aus, das im Netzwerk verfolgt wird.",
        "custom_sample": "Fuer digits und test_activation kannst du statt eines Datensatzbeispiels ein eigenes Sample verwenden und direkt beobachten, wie das Netz darauf reagiert.",
        "analysis_target": "Das Analyse-Ziel legt fest, gegen welche Klasse der aktuelle Loss in den Ansichten berechnet wird. Damit lassen sich auch absichtlich unpassende Ziele untersuchen.",
        "layout": "Das Layout beschreibt die Aktivierungsfunktionen aller Hidden-Layer. Es kann pro Layer oder pro Neuron veraendert werden.",
        "layer_overview": "Die Layer-Uebersicht fasst die aktuelle Hidden-Struktur kompakt zusammen: Layer, Anzahl der Neuronen und die erste Aktivierung im Layer.",
        "layer_fill": "Mit dieser Steuerung setzt du die Aktivierung eines kompletten Layers auf einen Schlag.",
        "neuron_layer": "Hier waehlst du aus, in welchem Hidden-Layer du ein einzelnes Neuron bearbeiten moechtest.",
        "neuron_index": "Hier waehlst du das konkrete Neuron im ausgewaehlten Layer aus.",
        "neuron_activation": "Hier legst du fest, welche Aktivierungsfunktion das ausgewaehlte Neuron verwenden soll.",
        "training": "Im Trainingsbereich stellst du die zentralen Hyperparameter ein und fuehrst schrittweise Trainingslaeufe aus.",
        "epochs": "Epochen geben an, wie oft das Training den gesamten Trainingssplit durchlaeuft.",
        "learning_rate": "Die Lernrate bestimmt die Schrittweite bei jeder Gewichtsaktualisierung.",
        "batch_size": "Die Batch-Groesse legt fest, wie viele Beispiele pro Gewichtsupdate gemeinsam verarbeitet werden.",
        "weight_scale": "Die Weight-Scale bestimmt die Groessenordnung der zufaelligen Startgewichte.",
        "seed": "Der Seed sorgt fuer reproduzierbare Initialisierung und reproduzierbare Datensplits.",
        "status": "Der Live-Status fasst Datensatz, aktuelle Vorhersage, Trainingsergebnisse und Baseline-Hinweise kompakt zusammen.",
        "objective": "Die Zielmetrik bestimmt, ob ein Layout als gut oder schlecht gilt. validation_loss soll minimiert werden, validation_accuracy soll maximiert werden und wird intern in einen minimierbaren Score umgerechnet.",
        "candidate_epochs": "So viele Epochen wird jeder SA-Kandidat trainiert, bevor er bewertet wird. Mehr Epochen geben eine fairere, aber langsamere Bewertung.",
        "shuffle": "Wenn aktiv, werden die Trainingsdaten pro Kandidat vor dem Mini-Batch-Training gemischt. Mit gleichem Seed bleibt das Verhalten trotzdem reproduzierbar.",
        "playground_neighborhood": "Diese Schalter bestimmen, welche Arten von Nachbarn das Annealing ueberhaupt vorschlagen darf.",
        "start_temperature": "Die Starttemperatur legt fest, wie leicht anfangs auch schlechtere Kandidaten akzeptiert werden.",
        "cooling_schedule": "Die Abkuehlung bestimmt, wie schnell die Temperatur sinkt. Geometrisch ist meist der robusteste Start, linear ist leicht zu verstehen, logarithmisch kuehlt besonders langsam.",
        "cooling_parameter": "Dieser Parameter steuert die Staerke der Abkuehlung. Seine genaue Bedeutung haengt von der gewaehlten Cooling-Strategie ab.",
        "iterations_per_temperature": "So viele SA-Schritte werden mit derselben Temperatur ausgefuehrt, bevor die naechste Abkuehlstufe beginnt.",
        "max_steps": "Hartes Limit fuer die gesamte Anzahl an SA-Schritten.",
        "min_temperature": "Wenn die Temperatur darunter faellt, endet die Suche.",
        "annealing_run_control": "Mit diesen Buttons kannst du die Suche initialisieren, schrittweise beobachten oder komplett laufen lassen.",
        "annealing_status": "Der Annealing-Status fasst aktuellen Zustand, besten Zustand, Temperatur, Akzeptanzrate und Cache-Status zusammen.",
        "network_visualization": (
            "Die Netzwerk-Visualisierung zeigt genau ein aktuell analysiertes Sample. "
            "Links stehen die sichtbaren Eingaben, in der Mitte die Hidden-Neuronen und rechts "
            "die Output-Neuronen. Die Farben der Hidden-Neuronen stehen fuer deren "
            "Aktivierungsfunktion. Graue Linien sind nur die Grundstruktur aller Verbindungen. "
            "Sobald ein Hidden-Neuron angeklickt wird, werden seine wichtigsten Verbindungen "
            "hervorgehoben: blau bedeutet positiver Einfluss, rot negativer Einfluss. "
            "Je dicker die Linie, desto staerker ist dieser Einfluss. Bei Verbindungen vom Input "
            "zum ersten Hidden-Layer bezieht sich die Staerke auf den aktuellen Beitrag dieses "
            "Samples, spaeter im Netz meist auf die Staerke des Gewichts. Am Output markiert der "
            "gruene Rand die Prediction, orange Text das echte Ziel und blaugruener Text das "
            "aktuelle Analyse-Ziel."
        ),
        "activation_curve": (
            "Die Kurve unten rechts gehoert immer zum aktuell angeklickten Hidden-Neuron. "
            "Auf der x-Achse liegt z, also die gewichtete Summe plus Bias. Auf der y-Achse liegt "
            "a, also die Ausgabe nach der Aktivierungsfunktion. Die farbige Kurve ist die "
            "vollstaendige Aktivierungsfunktion wie ReLU, tanh, sigmoid oder leaky_relu. "
            "Die gestrichelte senkrechte Linie markiert den aktuellen z-Wert dieses Neurons fuer "
            "das ausgewaehlte Sample. Der Punkt zeigt die konkrete Ausgabe a an dieser Stelle. "
            "So sieht man direkt, ob das Neuron gerade in einem linearen, toten oder saettigenden "
            "Bereich arbeitet."
        ),
        "plot_overview": (
            "Der Plot-Tab verbindet zwei Sichtweisen: oben die Trainingsgeschichte ueber viele "
            "Epochen und unten den aktuellen Zustand des gerade analysierten Samples. "
            "Links oben geht es um Loss, rechts oben um Accuracy. Links unten stehen die "
            "Klassenwahrscheinlichkeiten fuer genau dieses Sample, rechts unten die Aktivierungen "
            "aller Hidden-Neuronen fuer dasselbe Sample. So kann man Training und Einzelbeispiel "
            "gemeinsam lesen."
        ),
        "loss_plot": (
            "Der Loss-Plot zeigt, wie gross der Fehler des Modells waehrend des Trainings ist. "
            "Die x-Achse zeigt die Epoche, die y-Achse den mittleren Loss. Kleinere Werte sind "
            "besser. Train und Val sollten idealerweise beide sinken. Wenn Train stark sinkt, "
            "Val aber stehen bleibt oder steigt, spricht das fuer Overfitting. Wichtig: Loss ist "
            "kein Prozentwert. Er misst, wie schlecht die aktuellen Wahrscheinlichkeiten zum Ziel "
            "passen."
        ),
        "accuracy_plot": (
            "Der Accuracy-Plot zeigt den Anteil korrekt klassifizierter Beispiele. Die x-Achse "
            "zeigt die Epoche, die y-Achse einen Wert zwischen 0 und 1. Ein Wert von 0.80 "
            "bedeutet 80 Prozent korrekte Vorhersagen. Hoeher ist besser. Wenn Train-Accuracy "
            "stark ueber der Val-Accuracy liegt, lernt das Modell die Trainingsdaten besser als "
            "neue Beispiele."
        ),
        "class_probabilities": (
            "Dieser Plot zeigt fuer das aktuell sichtbare Sample die Wahrscheinlichkeiten aller "
            "Output-Klassen. Alle Balken zusammen ergeben 1. Der hoechste Balken entspricht der "
            "Prediction. Je hoeher ein Balken, desto staerker bevorzugt das Modell diese Klasse. "
            "Falls ein Analyse-Ziel gesetzt ist, wird dessen Balken markiert, auch wenn er nicht "
            "der groesste ist. So sieht man direkt, wie stark das Modell das gewaehlte Ziel "
            "unterstuetzt oder ablehnt."
        ),
        "sample_activations": (
            "Hier sieht man fuer das aktuelle Sample die Ausgaben aller Hidden-Neuronen. Auf der "
            "x-Achse stehen Layer und Neuron, auf der y-Achse die Aktivierung a. Die Farben "
            "entsprechen den verwendeten Aktivierungsfunktionen. Werte nahe 0 bedeuten oft wenig "
            "Einfluss fuer dieses Sample. Bei ReLU sind viele Nullen typisch fuer inaktive "
            "Neuronen. Bei tanh oder sigmoid zeigen sehr hohe oder sehr niedrige Werte oft, dass "
            "ein Neuron in einem gesaettigten Bereich arbeitet."
        ),
    },
    "en": {
        "app_mode": "The workspace mode separates three goals: Demo Mode for understanding and observing, Playground Mode for one SA run, and Experiment Builder for reproducible multi-seed and search experiments.",
        "mode": "The mode controls how much complexity is visible. Beginner focuses on the essential steps, while Expert exposes deeper interventions and comparison tools.",
        "language": "Switch the complete GUI between German and English here. Views, hints, help texts, and dialogs are updated together.",
        "workflow": "The learning flow shows the recommended order for using the tool. The guide opens a simple explanation of the whole program from start to finish.",
        "presets": "Presets load a working starting configuration quickly. They are useful if you want to start observing and training immediately.",
        "benchmark": "The benchmark defines the dataset, the number of classes, and the type of input. It also affects which visualization is shown in the input tab.",
        "hidden_layers": "Here you control the size and number of hidden layers. More layers or neurons increase capacity but also make the visualization more complex.",
        "split": "The data split only changes which sample is currently shown. Training itself still always uses the training split.",
        "sample_index": "The sample index chooses the exact example that is traced through the network.",
        "custom_sample": "For digits and test_activation you can use a custom sample instead of a dataset example and directly observe how the network reacts.",
        "analysis_target": "The analysis target defines which class is used for the displayed loss. This allows you to study intentionally mismatched targets as well.",
        "layout": "The layout describes the activation functions across all hidden layers. It can be changed per layer or per neuron.",
        "layer_overview": "The layer overview summarizes the current hidden structure: layer, neuron count, and first activation in that layer.",
        "layer_fill": "Use this control to set the activation of a complete layer at once.",
        "neuron_layer": "Choose the hidden layer in which you want to edit a single neuron.",
        "neuron_index": "Choose the exact neuron within the selected layer.",
        "neuron_activation": "Choose which activation function the selected neuron should use.",
        "training": "In the training section you set the core hyperparameters and run stepwise training sessions.",
        "epochs": "Epochs specify how often the training pass goes through the entire training split.",
        "learning_rate": "The learning rate defines the step size for each weight update.",
        "batch_size": "Batch size defines how many examples are processed together per weight update.",
        "weight_scale": "Weight scale defines the magnitude of the random initial weights.",
        "seed": "The seed makes initialization and data splits reproducible.",
        "status": "The live status summarizes dataset information, current prediction, training results, and baseline notes.",
        "objective": "The objective defines whether a layout is considered good or bad. validation_loss is minimized, while validation_accuracy is internally converted into a minimizable score.",
        "candidate_epochs": "This is how many epochs each SA candidate is trained before it is scored. More epochs make the score fairer but slower to compute.",
        "shuffle": "When enabled, training data is shuffled for each candidate before mini-batch training. With the same seed the behavior remains reproducible.",
        "playground_neighborhood": "These toggles decide which kinds of neighbors annealing is allowed to propose at all.",
        "start_temperature": "The start temperature determines how easily worse candidates can still be accepted at the beginning.",
        "cooling_schedule": "The cooling schedule determines how quickly the temperature drops. Geometric cooling is usually the most robust start, linear cooling is easy to understand, and logarithmic cooling decreases especially slowly.",
        "cooling_parameter": "This parameter controls how strong the cooling is. Its exact meaning depends on the selected cooling strategy.",
        "iterations_per_temperature": "This many SA steps are executed at the same temperature before moving to the next cooling level.",
        "max_steps": "Hard limit for the total number of SA steps.",
        "min_temperature": "The search stops once the temperature falls below this threshold.",
        "annealing_run_control": "These buttons let you initialize the search, observe it step by step, or run it to completion.",
        "annealing_status": "The annealing status summarizes the current state, best state, temperature, acceptance rate, and cache usage.",
        "network_visualization": (
            "The network view always shows one currently analyzed sample. Inputs are on the left, "
            "hidden neurons in the middle, and output neurons on the right. Hidden neuron colors "
            "represent their activation functions. Gray lines are only the base structure of all "
            "connections. Once you click a hidden neuron, its most important connections are "
            "highlighted: blue means positive influence, red means negative influence. Thicker "
            "lines indicate stronger influence. For input-to-first-hidden connections, this "
            "strength refers to the current sample contribution. Deeper in the network it usually "
            "reflects the weight magnitude. At the output, the green outline marks the prediction, "
            "orange text marks the true target, and teal text marks the current analysis target."
        ),
        "activation_curve": (
            "The curve at the bottom right always belongs to the currently selected hidden neuron. "
            "The x-axis shows z, the weighted sum plus bias. The y-axis shows a, the output after "
            "applying the activation function. The colored curve is the full activation function "
            "such as ReLU, tanh, sigmoid, or leaky_relu. The dashed vertical line marks the "
            "current z-value of this neuron for the selected sample. The point marks the concrete "
            "output a at that location. This lets you immediately see whether the neuron is in a "
            "linear, dead, or saturated regime."
        ),
        "plot_overview": (
            "The plot tab combines two viewpoints: the training history across many epochs on the "
            "top row and the current state of the selected sample on the bottom row. The top left "
            "shows loss, the top right shows accuracy. The bottom left shows class probabilities "
            "for the current sample, and the bottom right shows hidden neuron activations for that "
            "same sample. This makes it possible to read training dynamics and single-sample "
            "behavior together."
        ),
        "loss_plot": (
            "The loss plot shows how large the model error is during training. The x-axis shows "
            "the epoch, the y-axis shows the average loss. Lower values are better. Ideally both "
            "training and validation loss go down. If training loss drops strongly while validation "
            "loss stops improving or rises, that points to overfitting. Important: loss is not a "
            "percentage. It measures how poorly the current predicted probabilities match the "
            "target labels."
        ),
        "accuracy_plot": (
            "The accuracy plot shows the fraction of correctly classified examples. The x-axis "
            "shows the epoch, the y-axis a value between 0 and 1. A value of 0.80 means 80 "
            "percent correct predictions. Higher is better. If training accuracy becomes much "
            "higher than validation accuracy, the model fits the training data better than new "
            "unseen examples."
        ),
        "class_probabilities": (
            "This plot shows the output probabilities of all classes for the currently visible "
            "sample. All bars together sum to 1. The highest bar is the prediction. The higher a "
            "bar is, the more strongly the model prefers that class. If an analysis target is set, "
            "its bar is highlighted even when it is not the largest. This lets you directly see "
            "how strongly the model supports or rejects the chosen target."
        ),
        "sample_activations": (
            "This plot shows the outputs of all hidden neurons for the current sample. The x-axis "
            "lists layer and neuron, the y-axis shows the activation value a. Colors correspond to "
            "the activation functions in use. Values near 0 often indicate little influence for "
            "this sample. For ReLU, many zeros are typical for inactive neurons. For tanh or "
            "sigmoid, very high or very low values often indicate that a neuron is operating in a "
            "saturated region."
        ),
    },
}

BENCHMARK_DESCRIPTIONS = {
    "de": {
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
            "Er ist besonders anschaulich, weil Eingabe, Ziel und Vorhersage als Bild diskutiert "
            "werden koennen."
        ),
        "test_activation": (
            "test_activation ist kein echter Benchmark, sondern ein Lernlabor mit drei Inputs. "
            "Hier lassen sich rohe Vorwaertsrechnungen und Aktivierungsunterschiede besonders klar untersuchen."
        ),
    },
    "en": {
        "breast_cancer": (
            "The breast_cancer benchmark is a binary classification task with 30 numerical cell-nucleus features. "
            "It works well for first experiments because the network only separates two classes and learning curves "
            "often become readable quickly."
        ),
        "wine": (
            "The wine benchmark is a multiclass classification task with 13 chemical features. "
            "It is especially useful for comparing activation layouts because it is small, readable, and still non-trivial."
        ),
        "digits": (
            "The digits benchmark contains 8x8 grayscale images of handwritten digits and has 10 classes. "
            "It is especially useful because input, target, and prediction can be discussed visually."
        ),
        "test_activation": (
            "test_activation is not a real benchmark but a small learning lab with three inputs. "
            "It is ideal for inspecting raw forward computations and activation differences in a very compact setting."
        ),
    },
}


@dataclass(frozen=True)
class GuiExperimentConfig:
    """Konfigurationsobjekt fuer den Start der GUI."""

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
        self.root.geometry("1620x1040")
        self.root.minsize(980, 700)
        self.root.configure(bg=BACKGROUND_COLOR)

        self.dataset: DatasetBundle | None = None
        self.model: ModularMLP | None = None
        self.current_layout: ActivationLayout | None = None
        self.training_result: TrainingResult | None = None
        self.baseline_model: ModularMLP | None = None
        self.baseline_layout: ActivationLayout | None = None
        self.baseline_training_result: TrainingResult | None = None
        self.baseline_completed_epochs = 0
        self.baseline_benchmark_name: str | None = None
        self.completed_epochs = 0
        self.selected_hidden: tuple[int, int] = (0, 0)
        self.node_tags: dict[str, tuple[int, int]] = {}
        self.raw_custom_digit = np.zeros(64, dtype=np.float64)
        self.step_entries: list[tuple[str, str]] = []
        self.layer_size_controls_frame: ttk.Frame | None = None
        self.layer_fill_controls_frame: ttk.Frame | None = None
        self.builder_layer_fill_controls_frame: ttk.Frame | None = None
        self.hidden_size_spinboxes: list[ttk.Spinbox] = []
        self.hidden_layer_size_labels: list[ttk.Label] = []
        self.layer_fill_combos: list[ttk.Combobox] = []
        self.builder_layer_fill_combos: list[ttk.Combobox] = []
        self.expert_only_widgets: list[Any] = []
        self.builder_expert_only_widgets: list[Any] = []
        self.step_index_var = tk.IntVar(value=0)
        self.guide_window: tk.Toplevel | None = None
        self.controls_canvas: tk.Canvas | None = None
        self.controls_inner: ttk.Frame | None = None
        self.controls_window_id: int | None = None
        self.main_paned_window: ttk.Panedwindow | None = None
        self.controls_container: ttk.Frame | None = None
        self.canvas_frame: ttk.LabelFrame | None = None
        self.live_detail_frame: ttk.Frame | None = None
        self._network_draw_width = 0
        self._network_draw_height = 0

        self.app_mode_var = tk.StringVar(value=config.app_mode)
        self.app_mode_display_var = tk.StringVar(value="")
        self.mode_var = tk.StringVar(value=config.mode)
        self.mode_display_var = tk.StringVar(value="")
        self.language_var = tk.StringVar(value=config.language)
        self.language_display_var = tk.StringVar(value="")
        self.benchmark_var = tk.StringVar(value=config.benchmark)
        self.hidden_size_vars = [tk.IntVar(value=size) for size in config.hidden_sizes]
        self.split_var = tk.StringVar(value="train")
        self.sample_index_var = tk.IntVar(value=0)
        self.analysis_target_var = tk.StringVar(value=self._target_auto_label())
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
        self.prediction_summary_var = tk.StringVar(value="")
        self.metrics_summary_var = tk.StringVar(value="")
        self.workflow_summary_var = tk.StringVar(value="")
        self.context_hint_var = tk.StringVar(value="")
        self.sample_summary_var = tk.StringVar(value="")
        self.sample_hint_var = tk.StringVar(value="")
        self.network_summary_var = tk.StringVar(value="")
        self.compare_summary_var = tk.StringVar(value="")
        self.layer_summary_var = tk.StringVar(value="")
        self.playground_summary_var = tk.StringVar(value="")
        self.playground_decision_var = tk.StringVar(value="")
        self.builder_status_var = tk.StringVar(value=self.t("builder_status_idle"))
        self.builder_summary_var = tk.StringVar(value="")

        self.test_input_vars = [tk.DoubleVar(value=0.0) for _ in range(3)]
        self.objective_var = tk.StringVar(value=DEFAULT_ANNEALING_OBJECTIVE)
        self.candidate_epochs_var = tk.IntVar(value=DEFAULT_ANNEALING_CANDIDATE_EPOCHS)
        self.playground_shuffle_var = tk.BooleanVar(value=True)
        self.start_temperature_var = tk.StringVar(value=str(DEFAULT_ANNEALING_START_TEMPERATURE))
        self.cooling_schedule_var = tk.StringVar(value=DEFAULT_ANNEALING_COOLING_SCHEDULE)
        self.cooling_parameter_var = tk.StringVar(value=str(DEFAULT_ANNEALING_COOLING_PARAMETER))
        self.iterations_per_temperature_var = tk.IntVar(value=DEFAULT_ANNEALING_ITERATIONS_PER_TEMPERATURE)
        self.max_steps_var = tk.IntVar(value=DEFAULT_ANNEALING_MAX_STEPS)
        self.min_temperature_var = tk.StringVar(value=str(DEFAULT_ANNEALING_MIN_TEMPERATURE))
        self.playground_neighbor_vars = {
            operation: tk.BooleanVar(value=operation in DEFAULT_ANNEALING_NEIGHBORHOODS)
            for operation in ("set_neuron", "fill_layer", "swap_neurons")
        }

        default_builder_output = str(OUTPUT_DIR / DEFAULT_EXPERIMENT_OUTPUT_SUBDIR)
        self.builder_experiment_name_var = tk.StringVar(
            value=f"{config.benchmark}_{config.app_mode}"
        )
        self.builder_run_mode_var = tk.StringVar(value="manual_training")
        self.builder_run_mode_display_var = tk.StringVar(value="")
        self.builder_primary_metric_var = tk.StringVar(value="validation_accuracy")
        self.builder_seed_strategy_var = tk.StringVar(value="count_range")
        self.builder_seed_strategy_display_var = tk.StringVar(value="")
        self.builder_seed_count_var = tk.IntVar(value=5)
        self.builder_seed_start_var = tk.IntVar(value=config.random_state)
        self.builder_seed_list_var = tk.StringVar(value="42, 43, 44")
        self.builder_seed_preview_var = tk.StringVar(value="")
        self.builder_search_type_var = tk.StringVar(value=DEFAULT_EXPERIMENT_SEARCH_TYPE)
        self.builder_search_type_display_var = tk.StringVar(value="")
        self.builder_random_samples_var = tk.IntVar(value=DEFAULT_EXPERIMENT_RANDOM_SEARCH_SAMPLES)
        self.builder_random_seed_var = tk.IntVar(value=config.random_state)
        self.builder_layout_choice_var = tk.StringVar(value="")
        self.builder_output_dir_var = tk.StringVar(value=default_builder_output)
        self.builder_save_json_var = tk.BooleanVar(value=True)
        self.builder_load_path_var = tk.StringVar(value="")

        self.builder_tuning_mode_vars: dict[str, tk.StringVar] = {}
        self.builder_tuning_mode_display_vars: dict[str, tk.StringVar] = {}
        self.builder_tuning_value_vars: dict[str, tk.StringVar] = {}
        self.builder_tuning_rows: dict[str, ttk.Frame] = {}

        self.annealing_runner: AnnealingRunner | None = None
        self.annealing_state: AnnealingState | None = None
        self.annealing_last_step: AnnealingStep | None = None
        self.annealing_last_candidate: ObjectiveEvaluation | None = None
        self.annealing_text: ScrolledText | None = None
        self.annealing_status_label: ttk.Label | None = None
        self.annealing_figure: Figure | None = None
        self.annealing_axes: list[Any] = []
        self.annealing_canvas_widget: FigureCanvasTkAgg | None = None

        self.builder_runner = ExperimentRunner()
        self.builder_last_execution: dict[str, Any] | None = None
        self.builder_loaded_payload: dict[str, Any] | None = None
        self.builder_selected_run_id: str | None = None
        self.builder_selected_config_id: str | None = None
        self.builder_run_tree: ttk.Treeview | None = None
        self.builder_summary_text: ScrolledText | None = None
        self.builder_analysis_text: ScrolledText | None = None
        self.builder_detail_text: ScrolledText | None = None
        self.builder_help_text: ScrolledText | None = None
        self.builder_plot_container: ttk.Frame | None = None
        self.builder_plot_canvas_widget: FigureCanvasTkAgg | None = None
        self.builder_plot_figure: Figure | None = None
        self.builder_run_mode_sections: list[ttk.LabelFrame] = []
        self.builder_layout_options: dict[str, dict[str, Any]] = {}
        self.builder_layout_row: ttk.Frame | None = None
        self.builder_layout_combo: ttk.Combobox | None = None
        self.builder_layout_entry: ttk.Entry | None = None
        self.builder_layout_editor_frame: ttk.LabelFrame | None = None
        self.builder_neuron_layer_label: ttk.Frame | None = None
        self.builder_neuron_layer_combo: ttk.Combobox | None = None
        self.builder_neuron_index_label: ttk.Frame | None = None
        self.builder_neuron_index_combo: ttk.Combobox | None = None
        self.builder_neuron_activation_label: ttk.Frame | None = None
        self.builder_neuron_activation_combo: ttk.Combobox | None = None
        self.builder_neuron_button_row: ttk.Frame | None = None
        self.builder_config_layout: ActivationLayout | None = None
        self.builder_setup_frame: ttk.LabelFrame | None = None
        self.builder_seeds_frame: ttk.LabelFrame | None = None
        self.builder_training_frame: ttk.LabelFrame | None = None
        self.builder_sa_frame: ttk.LabelFrame | None = None
        self.builder_tuning_frame: ttk.LabelFrame | None = None
        self.builder_storage_frame: ttk.LabelFrame | None = None
        self.builder_run_frame: ttk.LabelFrame | None = None

        self._initialize_builder_tuning_vars(config)

        self._configure_styles()
        self._build_layout()
        self._load_experiment(reinitialize_model=True)

    def _language(self) -> str:
        """Liefert die aktuelle GUI-Sprache."""

        current_language = self.language_var.get()
        return current_language if current_language in SUPPORTED_GUI_LANGUAGES else "de"

    def _is_english(self) -> bool:
        """Kurzabfrage fuer englische GUI-Texte."""

        return self._language() == "en"

    def _target_auto_label(self) -> str:
        """Beschriftung fuer das automatische Analyse-Ziel in aktueller Sprache."""

        return self.t("target_auto_label")

    def _target_none_label(self) -> str:
        """Beschriftung fuer ein deaktiviertes Analyse-Ziel."""

        return self.t("target_none_label")

    def t(self, key: str, **kwargs: Any) -> str:
        """Liefert einen GUI-Text in der aktuellen Sprache."""

        language = self._language()
        template = GUI_TEXTS.get(language, GUI_TEXTS["de"]).get(key)
        if template is None:
            template = EXPERIMENT_BUILDER_TEXTS.get(language, EXPERIMENT_BUILDER_TEXTS["de"]).get(
                key,
                EXPERIMENT_BUILDER_TEXTS["de"].get(key, key),
            )
        return template.format(**kwargs)

    def info_text(self, key: str) -> str:
        """Liefert den Inhalt eines Info-Dialogs in der aktuellen Sprache."""

        language = self._language()
        template = INFO_TEXTS.get(language, INFO_TEXTS["de"]).get(key)
        if template is None:
            template = EXPERIMENT_BUILDER_INFO_TEXTS.get(
                language,
                EXPERIMENT_BUILDER_INFO_TEXTS["de"],
            ).get(key, EXPERIMENT_BUILDER_INFO_TEXTS["de"].get(key, key))
        return template

    def _app_mode(self) -> str:
        """Liefert den aktuellen Arbeitsmodus."""

        current_mode = self.app_mode_var.get()
        return current_mode if current_mode in SUPPORTED_GUI_APP_MODES else DEFAULT_GUI_APP_MODE

    def _app_mode_label(self, app_mode_value: str) -> str:
        """Uebersetzt interne App-Modi in sichtbare GUI-Labels."""

        return APP_MODE_LABELS.get(self._language(), APP_MODE_LABELS["de"]).get(
            app_mode_value,
            app_mode_value,
        )

    def _app_mode_from_label(self, app_mode_label: str) -> str:
        """Wandelt ein sichtbares App-Mode-Label zurueck in den internen Wert."""

        for app_mode_value in SUPPORTED_GUI_APP_MODES:
            if self._app_mode_label(app_mode_value) == app_mode_label:
                return app_mode_value
        return DEFAULT_GUI_APP_MODE

    def _sync_app_mode_display_var(self) -> None:
        """Synchronisiert das sichtbare App-Mode-Label mit dem internen Wert."""

        self.app_mode_display_var.set(self._app_mode_label(self._app_mode()))

    def _run_mode_label(self, mode_value: str) -> str:
        """Uebersetzt interne Builder-Run-Modi in sichtbare Labels."""

        return RUN_MODE_LABELS.get(self._language(), RUN_MODE_LABELS["de"]).get(mode_value, mode_value)

    def _run_mode_from_label(self, mode_label: str) -> str:
        """Wandelt ein sichtbares Builder-Run-Mode-Label in den internen Wert um."""

        for mode_value in SUPPORTED_EXPERIMENT_RUN_MODES:
            if self._run_mode_label(mode_value) == mode_label:
                return mode_value
        return "manual_training"

    def _sync_builder_run_mode_display_var(self) -> None:
        """Synchronisiert das sichtbare Builder-Run-Mode-Label."""

        self.builder_run_mode_display_var.set(self._run_mode_label(self.builder_run_mode_var.get()))

    def _search_type_label(self, search_type: str) -> str:
        """Uebersetzt Builder-Search-Typen."""

        return SEARCH_TYPE_LABELS.get(self._language(), SEARCH_TYPE_LABELS["de"]).get(search_type, search_type)

    def _search_type_from_label(self, label: str) -> str:
        """Wandelt ein sichtbares Search-Label in den internen Wert um."""

        for search_type in SUPPORTED_SEARCH_TYPES:
            if self._search_type_label(search_type) == label:
                return search_type
        return DEFAULT_EXPERIMENT_SEARCH_TYPE

    def _sync_builder_search_type_display_var(self) -> None:
        """Synchronisiert das sichtbare Search-Type-Label."""

        self.builder_search_type_display_var.set(
            self._search_type_label(self.builder_search_type_var.get())
        )

    def _seed_strategy_label(self, strategy: str) -> str:
        """Uebersetzt die Builder-Seed-Strategie."""

        return SEED_STRATEGY_LABELS.get(self._language(), SEED_STRATEGY_LABELS["de"]).get(strategy, strategy)

    def _seed_strategy_from_label(self, label: str) -> str:
        """Wandelt ein sichtbares Seed-Strategie-Label in den internen Wert um."""

        for strategy in SEED_STRATEGY_LABELS["de"]:
            if self._seed_strategy_label(strategy) == label:
                return strategy
        return "count_range"

    def _sync_builder_seed_strategy_display_var(self) -> None:
        """Synchronisiert das sichtbare Seed-Strategie-Label."""

        self.builder_seed_strategy_display_var.set(
            self._seed_strategy_label(self.builder_seed_strategy_var.get())
        )

    def _search_value_kind_label(self, kind: str) -> str:
        """Uebersetzt Fixed/List/Range fuer Builder-Tuningfelder."""

        return SEARCH_VALUE_KIND_LABELS.get(self._language(), SEARCH_VALUE_KIND_LABELS["de"]).get(kind, kind)

    def _search_value_kind_from_label(self, label: str) -> str:
        """Wandelt ein sichtbares Search-Value-Kind in den internen Wert um."""

        for kind in SEARCH_VALUE_KIND_LABELS["de"]:
            if self._search_value_kind_label(kind) == label:
                return kind
        return "fixed"

    def _initialize_builder_tuning_vars(self, config: GuiExperimentConfig) -> None:
        """Initialisiert die GUI-Variablen fuer diskrete Suchraeume."""

        base_values = {
            "learning_rate": str(config.learning_rate),
            "batch_size": str(config.batch_size),
            "weight_scale": str(config.weight_scale),
            "epochs": str(config.epochs),
            "candidate_epochs": str(DEFAULT_ANNEALING_CANDIDATE_EPOCHS),
            "start_temperature": str(DEFAULT_ANNEALING_START_TEMPERATURE),
            "cooling_schedule": DEFAULT_ANNEALING_COOLING_SCHEDULE,
            "cooling_parameter": str(DEFAULT_ANNEALING_COOLING_PARAMETER),
        }
        for parameter_name, value_text in base_values.items():
            self.builder_tuning_mode_vars[parameter_name] = tk.StringVar(value="fixed")
            self.builder_tuning_mode_display_vars[parameter_name] = tk.StringVar(value="")
            self.builder_tuning_value_vars[parameter_name] = tk.StringVar(value=value_text)
        self._sync_builder_run_mode_display_var()
        self._sync_builder_search_type_display_var()
        self._sync_builder_seed_strategy_display_var()
        for parameter_name in self.builder_tuning_mode_display_vars:
            self.builder_tuning_mode_display_vars[parameter_name].set(
                self._search_value_kind_label(self.builder_tuning_mode_vars[parameter_name].get())
            )

    def _mode_label(self, mode_value: str) -> str:
        """Uebersetzt interne Moduswerte in sichtbare GUI-Labels."""

        return MODE_LABELS.get(self._language(), MODE_LABELS["de"]).get(mode_value, mode_value)

    def _mode_from_label(self, mode_label: str) -> str:
        """Wandelt ein sichtbares Moduslabel zurueck in den internen Wert."""

        for mode_value in SUPPORTED_GUI_MODES:
            if self._mode_label(mode_value) == mode_label:
                return mode_value
        return "beginner"

    def _sync_mode_display_var(self) -> None:
        """Synchronisiert das sichtbare Moduslabel mit dem internen Wert."""

        self.mode_display_var.set(self._mode_label(self.mode_var.get()))

    def _sync_language_display_var(self) -> None:
        """Synchronisiert das sichtbare Sprachlabel mit dem internen Sprachwert."""

        self.language_display_var.set(LANGUAGE_LABELS.get(self._language(), "Deutsch"))

    def _configure_styles(self) -> None:
        """Setzt ein ruhiges, gut lesbares GUI-Theme."""

        self.root.title(self.t("window_title"))
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
        style.configure("Info.TButton", padding=(4, 0))

    def _build_layout(self) -> None:
        """Baut die Hauptstruktur des Fensters auf."""

        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)

        self.main_paned_window = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
        self.main_paned_window.grid(row=0, column=0, sticky="nsew")

        controls_container = ttk.Frame(self.main_paned_window, style="White.TFrame", padding=(0, 0, 8, 0))
        controls_container.rowconfigure(0, weight=1)
        controls_container.columnconfigure(0, weight=1)
        controls_container.columnconfigure(1, weight=0)
        self.controls_container = controls_container

        self.controls_canvas = tk.Canvas(
            controls_container,
            bg=PANEL_COLOR,
            highlightthickness=0,
            width=430,
        )
        self.controls_canvas.grid(row=0, column=0, sticky="nsew")
        controls_scrollbar = ttk.Scrollbar(
            controls_container,
            orient=tk.VERTICAL,
            command=self.controls_canvas.yview,
        )
        controls_scrollbar.grid(row=0, column=1, sticky="ns")
        self.controls_canvas.configure(yscrollcommand=controls_scrollbar.set)

        self.controls_inner = ttk.Frame(self.controls_canvas, style="White.TFrame")
        self.controls_window_id = self.controls_canvas.create_window(
            (0, 0),
            window=self.controls_inner,
            anchor="nw",
        )
        self.controls_inner.bind("<Configure>", self._on_controls_inner_configure)
        self.controls_canvas.bind("<Configure>", self._on_controls_canvas_configure)
        self.controls_canvas.bind("<Enter>", self._bind_controls_mousewheel)
        self.controls_canvas.bind("<Leave>", self._unbind_controls_mousewheel)
        self.controls_inner.bind("<Enter>", self._bind_controls_mousewheel)
        self.controls_inner.bind("<Leave>", self._unbind_controls_mousewheel)

        content = ttk.Frame(self.main_paned_window, style="White.TFrame")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=0)
        content.rowconfigure(1, weight=3)
        content.rowconfigure(2, weight=2)

        self.main_paned_window.add(controls_container, weight=0)
        self.main_paned_window.add(content, weight=1)

        self._build_controls(self.controls_inner)
        self._build_content(content)
        self._apply_mode_visibility()
        self.root.after(0, self._set_initial_pane_geometry)

    def _set_initial_pane_geometry(self) -> None:
        """Setzt eine sinnvolle Startbreite fuer die linke, ziehbare Seitenleiste."""

        if self.main_paned_window is None or self.controls_container is None:
            return
        try:
            total_width = max(self.main_paned_window.winfo_width(), 1)
            if total_width <= 1:
                return
            desired_width = min(max(int(total_width * 0.28), 360), 520)
            self.main_paned_window.sashpos(0, desired_width)
        except tk.TclError:
            return

    def _on_controls_inner_configure(self, _event: tk.Event) -> None:
        """Aktualisiert den Scrollbereich der linken Seitenleiste."""

        if self.controls_canvas is None:
            return
        self.controls_canvas.configure(scrollregion=self.controls_canvas.bbox("all"))

    def _on_controls_canvas_configure(self, event: tk.Event) -> None:
        """Haltet das innere Control-Frame auf Canvas-Breite."""

        if self.controls_canvas is None or self.controls_window_id is None:
            return
        self.controls_canvas.itemconfigure(self.controls_window_id, width=event.width)

    def _on_canvas_frame_configure(self, _event: tk.Event) -> None:
        """Reagiert auf Groessenwechsel des Netzwerkbereichs."""

        self._update_canvas_layout()
        self._redraw_network()

    def _update_canvas_layout(self) -> None:
        """Schaltet zwischen Seiten- und Stapel-Layout fuer die Netzinspektion."""

        if self.canvas_frame is None or self.live_detail_frame is None:
            return

        available_width = max(self.canvas_frame.winfo_width(), 1)
        use_stacked_layout = available_width < 1080

        if use_stacked_layout:
            self.canvas_frame.columnconfigure(0, weight=1)
            self.canvas_frame.columnconfigure(1, weight=0)
            self.canvas_frame.rowconfigure(0, weight=0)
            self.canvas_frame.rowconfigure(1, weight=4)
            self.canvas_frame.rowconfigure(2, weight=3)
            self.canvas.grid_configure(row=1, column=0, sticky="nsew")
            self.live_detail_frame.grid_configure(
                row=2,
                column=0,
                sticky="nsew",
                padx=(0, 0),
                pady=(10, 0),
            )
            self.live_detail_frame.configure(padding=(0, 10, 0, 0))
        else:
            self.canvas_frame.columnconfigure(0, weight=5)
            self.canvas_frame.columnconfigure(1, weight=3)
            self.canvas_frame.rowconfigure(0, weight=0)
            self.canvas_frame.rowconfigure(1, weight=1)
            self.canvas_frame.rowconfigure(2, weight=0)
            self.canvas.grid_configure(row=1, column=0, sticky="nsew")
            self.live_detail_frame.grid_configure(
                row=1,
                column=1,
                sticky="nsew",
                padx=(8, 0),
                pady=(0, 0),
            )
            self.live_detail_frame.configure(padding=(8, 0, 0, 0))

    def _bind_controls_mousewheel(self, _event: tk.Event) -> None:
        """Aktiviert Mausrad-Scrolling, solange die Maus ueber der Seitenleiste ist."""

        self.root.bind_all("<MouseWheel>", self._on_controls_mousewheel)
        self.root.bind_all("<Button-4>", self._on_controls_mousewheel)
        self.root.bind_all("<Button-5>", self._on_controls_mousewheel)

    def _unbind_controls_mousewheel(self, _event: tk.Event) -> None:
        """Entfernt die globale Mausrad-Bindung beim Verlassen der Seitenleiste."""

        self.root.unbind_all("<MouseWheel>")
        self.root.unbind_all("<Button-4>")
        self.root.unbind_all("<Button-5>")

    def _on_controls_mousewheel(self, event: tk.Event) -> None:
        """Scrollt die linke Seitenleiste per Mausrad."""

        if self.controls_canvas is None:
            return
        if getattr(event, "num", None) == 4:
            self.controls_canvas.yview_scroll(-1, "units")
            return
        if getattr(event, "num", None) == 5:
            self.controls_canvas.yview_scroll(1, "units")
            return
        delta = int(-1 * (event.delta / 120)) if getattr(event, "delta", 0) else 0
        if delta != 0:
            self.controls_canvas.yview_scroll(delta, "units")

    def _show_info(self, info_key: str) -> None:
        """Oeffnet einen kompakten Info-Dialog fuer einen GUI-Punkt."""

        messagebox.showinfo(self.t("info_dialog_title"), self.info_text(info_key))

    def _widget_exists(self, widget: Any | None) -> bool:
        """Prueft robust, ob ein Tk-Widget noch existiert."""

        if widget is None:
            return False
        try:
            return bool(widget.winfo_exists())
        except tk.TclError:
            return False

    def _create_info_button(self, parent: tk.Widget, info_key: str) -> ttk.Button:
        """Erzeugt einen kleinen 'i'-Button fuer Zusatzinfos."""

        return ttk.Button(
            parent,
            text="i",
            width=2,
            style="Info.TButton",
            command=lambda key=info_key: self._show_info(key),
        )

    def _create_info_action_button(
        self,
        parent: tk.Widget,
        label_text: str,
        info_key: str,
    ) -> ttk.Button:
        """Erzeugt einen beschrifteten Button, der eine laengere Info oeffnet."""

        return ttk.Button(
            parent,
            text=f"{label_text}  i",
            command=lambda key=info_key: self._show_info(key),
        )

    def _grid_label_with_info(
        self,
        parent: ttk.Frame,
        row: int,
        label_text: str,
        info_key: str,
        pady: tuple[int, int] = (6, 0),
    ) -> ttk.Frame:
        """Platziert eine Feldbeschriftung zusammen mit einem Info-Button."""

        holder = ttk.Frame(parent, style="White.TFrame")
        holder.grid(row=row, column=0, sticky="w", pady=pady)
        ttk.Label(holder, text=label_text).pack(side=tk.LEFT)
        self._create_info_button(holder, info_key).pack(side=tk.LEFT, padx=(6, 0))
        return holder

    def _analysis_target_token(self) -> str:
        """Normalisiert das ausgewaehlte Analyse-Ziel fuer Sprachwechsel."""

        selected = self.analysis_target_var.get()
        if selected == self._target_none_label():
            return "__none__"
        if selected == self._target_auto_label():
            return "__auto__"
        return selected

    def _set_analysis_target_from_token(self, token: str) -> None:
        """Setzt ein Analyse-Ziel aus seiner sprachunabhaengigen Darstellung."""

        if token == "__none__":
            self.analysis_target_var.set(self._target_none_label())
        elif token == "__auto__":
            self.analysis_target_var.set(self._target_auto_label())
        else:
            self.analysis_target_var.set(token)

    def _on_mode_selected(self, _event=None) -> None:
        """Uebernimmt die sichtbare Modusauswahl in den internen Moduswert."""

        self.mode_var.set(self._mode_from_label(self.mode_display_var.get()))
        self._on_mode_changed()

    def _on_app_mode_selected(self, _event=None) -> None:
        """Schaltet zwischen den grossen GUI-Arbeitsmodi um."""

        selected_app_mode = self._app_mode_from_label(self.app_mode_display_var.get())
        if selected_app_mode == self._app_mode():
            return
        analysis_target_token = self._analysis_target_token()
        selected_tab_index = 0
        if hasattr(self, "notebook"):
            try:
                selected_tab_index = int(self.notebook.index(self.notebook.select()))
            except tk.TclError:
                selected_tab_index = 0
        self.app_mode_var.set(selected_app_mode)
        self._set_analysis_target_from_token(analysis_target_token)
        self._rebuild_interface(selected_tab_index=selected_tab_index)

    def _on_language_selected(self, _event=None) -> None:
        """Schaltet die GUI-Sprache um und baut die Oberflaeche neu auf."""

        selected_label = self.language_display_var.get()
        selected_language = next(
            (code for code, label in LANGUAGE_LABELS.items() if label == selected_label),
            "de",
        )
        if selected_language == self._language():
            return
        analysis_target_token = self._analysis_target_token()
        selected_tab_index = 0
        if hasattr(self, "notebook"):
            try:
                selected_tab_index = int(self.notebook.index(self.notebook.select()))
            except tk.TclError:
                selected_tab_index = 0
        self.language_var.set(selected_language)
        self._set_analysis_target_from_token(analysis_target_token)
        self._rebuild_interface(selected_tab_index=selected_tab_index)

    def _rebuild_interface(self, selected_tab_index: int = 0) -> None:
        """Baut die sichtbare GUI neu auf, ohne Modellzustand zu verlieren."""

        guide_was_open = self.guide_window is not None and self.guide_window.winfo_exists()
        self._configure_styles()
        self.expert_only_widgets = []
        self.hidden_size_spinboxes = []
        self.hidden_layer_size_labels = []
        self.layer_fill_combos = []
        self.builder_layer_fill_combos = []
        self.layer_size_controls_frame = None
        self.layer_fill_controls_frame = None
        self.builder_layer_fill_controls_frame = None
        self.builder_expert_only_widgets = []
        self.node_tags = {}
        self.controls_canvas = None
        self.controls_inner = None
        self.controls_window_id = None
        self.main_paned_window = None
        self.controls_container = None
        self.guide_window = None
        self.annealing_text = None
        self.annealing_status_label = None
        self.annealing_figure = None
        self.annealing_axes = []
        self.annealing_canvas_widget = None
        self.builder_run_tree = None
        self.builder_summary_text = None
        self.builder_analysis_text = None
        self.builder_detail_text = None
        self.builder_help_text = None
        self.builder_layout_entry = None
        self.builder_layout_editor_frame = None
        self.builder_neuron_layer_label = None
        self.builder_neuron_layer_combo = None
        self.builder_neuron_index_label = None
        self.builder_neuron_index_combo = None
        self.builder_neuron_activation_label = None
        self.builder_neuron_activation_combo = None
        self.builder_neuron_button_row = None
        self.builder_plot_container = None
        self.builder_plot_canvas_widget = None
        self.builder_plot_figure = None
        self.builder_run_mode_sections = []
        self.builder_tuning_rows = {}
        self.builder_layout_row = None
        self.builder_layout_combo = None
        self.builder_layout_options = {}
        self.builder_setup_frame = None
        self.builder_seeds_frame = None
        self.builder_training_frame = None
        self.builder_sa_frame = None
        self.builder_tuning_frame = None
        self.builder_storage_frame = None
        self.builder_run_frame = None

        for child in self.root.winfo_children():
            child.destroy()

        self._build_layout()
        self._sync_app_mode_display_var()
        self._sync_mode_display_var()
        self._sync_language_display_var()
        self._sync_builder_run_mode_display_var()
        self._sync_builder_search_type_display_var()
        self._sync_builder_seed_strategy_display_var()
        for parameter_name in self.builder_tuning_mode_display_vars:
            self.builder_tuning_mode_display_vars[parameter_name].set(
                self._search_value_kind_label(self.builder_tuning_mode_vars[parameter_name].get())
            )
        self._update_target_options()
        self._set_analysis_target_from_token(self._analysis_target_token())
        self._apply_mode_visibility()
        self._refresh_views()
        if hasattr(self, "notebook"):
            try:
                self.notebook.select(selected_tab_index)
            except tk.TclError:
                pass
        if guide_was_open:
            self._open_program_guide()

    def _build_controls(self, parent: ttk.Frame) -> None:
        """Linke Seitenleiste mit Assistent, Layout, Training und Vergleich."""
        self._sync_app_mode_display_var()
        self._sync_mode_display_var()
        self._sync_language_display_var()
        self._sync_builder_run_mode_display_var()
        self._sync_builder_search_type_display_var()
        self._sync_builder_seed_strategy_display_var()

        mode_frame = ttk.LabelFrame(parent, text=self.t("mode_frame_title"), padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        mode_hint_row = ttk.Frame(mode_frame, style="White.TFrame")
        mode_hint_row.pack(fill=tk.X)
        ttk.Label(
            mode_hint_row,
            text=self.t("mode_hint"),
            style="Hint.TLabel",
            justify=tk.LEFT,
            wraplength=300,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        mode_controls = ttk.Frame(mode_frame, style="White.TFrame")
        mode_controls.pack(fill=tk.X, pady=(10, 0))

        app_mode_row = ttk.Frame(mode_controls, style="White.TFrame")
        app_mode_row.pack(fill=tk.X)
        app_mode_left = ttk.Frame(app_mode_row, style="White.TFrame")
        app_mode_left.pack(fill=tk.X)
        ttk.Label(app_mode_left, text=self.t("app_mode_label")).pack(side=tk.LEFT)
        self._create_info_button(app_mode_left, "app_mode").pack(side=tk.LEFT, padx=(6, 0))
        app_mode_combo = ttk.Combobox(
            app_mode_row,
            textvariable=self.app_mode_display_var,
            values=[self._app_mode_label(value) for value in SUPPORTED_GUI_APP_MODES],
            state="readonly",
            width=24,
        )
        app_mode_combo.pack(fill=tk.X, pady=(4, 0))
        app_mode_combo.bind("<<ComboboxSelected>>", self._on_app_mode_selected)

        mode_row = ttk.Frame(mode_controls, style="White.TFrame")
        mode_row.pack(fill=tk.X, pady=(10, 0))
        mode_left = ttk.Frame(mode_row, style="White.TFrame")
        mode_left.pack(fill=tk.X)
        ttk.Label(mode_left, text=self.t("mode_label")).pack(side=tk.LEFT)
        self._create_info_button(mode_left, "mode").pack(side=tk.LEFT, padx=(6, 0))
        mode_combo = ttk.Combobox(
            mode_row,
            textvariable=self.mode_display_var,
            values=[self._mode_label(value) for value in SUPPORTED_GUI_MODES],
            state="readonly",
            width=24,
        )
        mode_combo.pack(fill=tk.X, pady=(4, 0))
        mode_combo.bind("<<ComboboxSelected>>", self._on_mode_selected)

        language_row = ttk.Frame(mode_controls, style="White.TFrame")
        language_row.pack(fill=tk.X, pady=(10, 0))
        language_left = ttk.Frame(language_row, style="White.TFrame")
        language_left.pack(fill=tk.X)
        ttk.Label(language_left, text=self.t("language_label")).pack(side=tk.LEFT)
        self._create_info_button(language_left, "language").pack(side=tk.LEFT, padx=(6, 0))
        language_combo = ttk.Combobox(
            language_row,
            textvariable=self.language_display_var,
            values=[LANGUAGE_LABELS[code] for code in SUPPORTED_GUI_LANGUAGES],
            state="readonly",
            width=24,
        )
        language_combo.pack(fill=tk.X, pady=(4, 0))
        language_combo.bind("<<ComboboxSelected>>", self._on_language_selected)

        workflow_frame = ttk.LabelFrame(parent, text=self.t("workflow_frame_title"), padding=10)
        workflow_frame.pack(fill=tk.X, pady=(0, 10))
        workflow_button_row = ttk.Frame(workflow_frame, style="White.TFrame")
        workflow_button_row.pack(fill=tk.X, pady=(0, 8))
        self._create_info_button(workflow_button_row, "workflow").pack(side=tk.LEFT)
        ttk.Button(
            workflow_button_row,
            text=self.t("guide_button"),
            command=self._open_program_guide,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(
            workflow_frame,
            text=self.t("guide_link_text"),
            justify=tk.LEFT,
            style="Hint.TLabel",
            wraplength=330,
        ).pack(anchor="w", pady=(0, 8))
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
            wraplength=330,
        ).pack(anchor="w", pady=(8, 0))

        if self._app_mode() == "experiment_builder":
            self._build_experiment_builder_controls(parent)
            return

        preset_frame = ttk.LabelFrame(parent, text=self.t("preset_frame_title"), padding=10)
        preset_frame.pack(fill=tk.X, pady=(0, 10))
        preset_hint_row = ttk.Frame(preset_frame, style="White.TFrame")
        preset_hint_row.pack(fill=tk.X)
        ttk.Label(
            preset_hint_row,
            text=self.t("preset_hint"),
            style="Hint.TLabel",
            justify=tk.LEFT,
            wraplength=300,
        ).pack(side=tk.LEFT, anchor="w")
        self._create_info_button(preset_hint_row, "presets").pack(side=tk.RIGHT)
        preset_row_1 = ttk.Frame(preset_frame, style="White.TFrame")
        preset_row_1.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(preset_row_1, text=self.t("preset_wine_demo"), command=lambda: self._apply_preset("wine_demo")).pack(side=tk.LEFT)
        ttk.Button(preset_row_1, text=self.t("preset_digits_demo"), command=lambda: self._apply_preset("digits_demo")).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(preset_row_1, text=self.t("preset_all_relu"), command=lambda: self._apply_preset("all_relu")).pack(side=tk.LEFT, padx=(8, 0))
        preset_row_2 = ttk.Frame(preset_frame, style="White.TFrame")
        preset_row_2.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(preset_row_2, text=self.t("preset_mixed"), command=lambda: self._apply_preset("mixed")).pack(side=tk.LEFT)
        ttk.Button(preset_row_2, text=self.t("preset_test_activation"), command=lambda: self._apply_preset("test_activation")).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(preset_row_2, text=self.t("preset_reset"), command=self._reset_to_defaults).pack(side=tk.LEFT, padx=(8, 0))

        experiment_frame = ttk.LabelFrame(parent, text=self.t("experiment_frame_title"), padding=10)
        experiment_frame.pack(fill=tk.X, pady=(0, 10))
        experiment_frame.columnconfigure(1, weight=1)
        ttk.Label(
            experiment_frame,
            text=self.t("experiment_hint"),
            style="Hint.TLabel",
            justify=tk.LEFT,
        ).grid(row=0, column=0, columnspan=2, sticky="ew")

        self._grid_label_with_info(experiment_frame, 1, self.t("benchmark_label"), "benchmark", pady=(10, 0))
        benchmark_combo = ttk.Combobox(
            experiment_frame,
            textvariable=self.benchmark_var,
            values=SUPPORTED_BENCHMARKS,
            state="readonly",
            width=18,
        )
        benchmark_combo.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(10, 0))
        benchmark_combo.bind("<<ComboboxSelected>>", self._on_benchmark_changed)

        self.hidden_layers_label = self._grid_label_with_info(
            experiment_frame,
            2,
            self.t("hidden_layers_label"),
            "hidden_layers",
        )
        self.layer_size_controls_frame = ttk.Frame(experiment_frame, style="White.TFrame")
        self.layer_size_controls_frame.grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))

        self.hidden_layer_buttons = ttk.Frame(experiment_frame, style="White.TFrame")
        self.hidden_layer_buttons.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.add_layer_button = ttk.Button(
            self.hidden_layer_buttons,
            text=self.t("add_layer_button"),
            command=self._add_hidden_layer,
        )
        self.add_layer_button.pack(side=tk.LEFT)
        self.remove_layer_button = ttk.Button(
            self.hidden_layer_buttons,
            text=self.t("remove_layer_button"),
            command=self._remove_hidden_layer,
        )
        self.remove_layer_button.pack(side=tk.LEFT, padx=(8, 0))

        self._grid_label_with_info(experiment_frame, 4, self.t("split_label"), "split")
        split_combo = ttk.Combobox(
            experiment_frame,
            textvariable=self.split_var,
            values=("train", "val", "test"),
            state="readonly",
            width=10,
        )
        split_combo.grid(row=4, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        split_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_split_changed())

        self._grid_label_with_info(experiment_frame, 5, self.t("sample_index_label"), "sample_index")
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

        custom_row = ttk.Frame(experiment_frame, style="White.TFrame")
        custom_row.grid(row=6, column=0, columnspan=2, sticky="w", pady=(10, 0))
        self.use_custom_sample_checkbutton = ttk.Checkbutton(
            custom_row,
            text=self.t("custom_sample_label"),
            variable=self.use_custom_sample_var,
            command=self._on_custom_sample_toggled,
        )
        self.use_custom_sample_checkbutton.pack(side=tk.LEFT)
        self._create_info_button(custom_row, "custom_sample").pack(side=tk.LEFT, padx=(8, 0))

        self.analysis_target_label = self._grid_label_with_info(
            experiment_frame,
            7,
            self.t("analysis_target_label"),
            "analysis_target",
        )
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
        ttk.Button(experiment_button_row, text=self.t("benchmark_defaults_button"), command=self._apply_benchmark_defaults).pack(side=tk.LEFT)
        ttk.Button(experiment_button_row, text=self.t("load_experiment_button"), command=self._reload_experiment).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(experiment_button_row, text=self.t("random_sample_button"), command=self._choose_random_sample).pack(side=tk.LEFT, padx=(8, 0))

        layout_frame = ttk.LabelFrame(parent, text=self.t("layout_frame_title"), padding=10)
        layout_frame.pack(fill=tk.X, pady=(0, 10))
        layout_frame.columnconfigure(1, weight=1)
        ttk.Label(
            layout_frame,
            text=self.t("layout_hint"),
            style="Hint.TLabel",
            justify=tk.LEFT,
        ).grid(row=0, column=0, columnspan=3, sticky="ew")

        self._grid_label_with_info(layout_frame, 1, self.t("current_layout_label"), "layout", pady=(10, 0))
        ttk.Label(
            layout_frame,
            textvariable=self.layout_string_var,
            wraplength=300,
            justify=tk.LEFT,
            style="SectionValue.TLabel",
        ).grid(row=1, column=1, columnspan=2, sticky="w", padx=(8, 0), pady=(10, 0))

        self._grid_label_with_info(layout_frame, 2, self.t("layer_overview_label"), "layer_overview")
        ttk.Label(
            layout_frame,
            textvariable=self.layer_summary_var,
            wraplength=300,
            justify=tk.LEFT,
            style="Hint.TLabel",
        ).grid(row=2, column=1, columnspan=2, sticky="w", padx=(8, 0), pady=(6, 0))

        self._grid_label_with_info(layout_frame, 3, self.t("layer_fill_label"), "layer_fill", pady=(8, 0))
        self.layer_fill_controls_frame = ttk.Frame(layout_frame, style="White.TFrame")
        self.layer_fill_controls_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(6, 0))

        self.neuron_layer_label = self._grid_label_with_info(
            layout_frame,
            5,
            self.t("layer_label"),
            "neuron_layer",
            pady=(12, 0),
        )
        self.neuron_layer_combo = ttk.Combobox(
            layout_frame,
            textvariable=self.neuron_layer_var,
            values=("L1",),
            state="readonly",
            width=8,
        )
        self.neuron_layer_combo.grid(row=5, column=1, sticky="w", padx=(8, 0), pady=(12, 0))
        self.neuron_layer_combo.bind("<<ComboboxSelected>>", self._on_neuron_layer_changed)

        self.neuron_index_label = self._grid_label_with_info(
            layout_frame,
            6,
            self.t("neuron_label"),
            "neuron_index",
        )
        self.neuron_index_combo = ttk.Combobox(
            layout_frame,
            textvariable=self.neuron_index_var,
            values=("0",),
            state="readonly",
            width=8,
        )
        self.neuron_index_combo.grid(row=6, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        self.neuron_index_combo.bind("<<ComboboxSelected>>", lambda _event: self._select_from_controls())

        self.neuron_activation_label = self._grid_label_with_info(
            layout_frame,
            7,
            self.t("activation_label"),
            "neuron_activation",
        )
        self.neuron_activation_combo = ttk.Combobox(
            layout_frame,
            textvariable=self.neuron_activation_var,
            values=SUPPORTED_ACTIVATIONS,
            state="readonly",
            width=14,
        )
        self.neuron_activation_combo.grid(row=7, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        self.neuron_button_row = ttk.Frame(layout_frame, style="White.TFrame")
        self.neuron_button_row.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        ttk.Button(self.neuron_button_row, text=self.t("set_neuron_button"), command=self._apply_neuron_setting).pack(side=tk.LEFT)
        ttk.Button(self.neuron_button_row, text=self.t("cycle_button"), command=self._cycle_selected_neuron).pack(side=tk.LEFT, padx=(8, 0))

        if self._app_mode() == "playground":
            self._build_playground_controls(parent)
        else:
            self._build_demo_training_controls(parent)

        self._build_status_frame(parent)

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
            ]
        )

        self._rebuild_hidden_size_controls()
        self._rebuild_layer_fill_controls()
        self._apply_mode_visibility()

    def _build_demo_training_controls(self, parent: ttk.Frame) -> None:
        """Baut den bisherigen Trainingsblock fuer den Demo Mode."""

        training_frame = ttk.LabelFrame(parent, text=self.t("training_frame_title"), padding=10)
        training_frame.pack(fill=tk.X, pady=(0, 10))
        training_frame.columnconfigure(1, weight=1)
        training_hint_row = ttk.Frame(training_frame, style="White.TFrame")
        training_hint_row.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(
            training_hint_row,
            text=self.t("training_hint"),
            style="Hint.TLabel",
            justify=tk.LEFT,
            wraplength=300,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._create_info_button(training_hint_row, "training").pack(side=tk.RIGHT)

        self.epochs_label = self._grid_label_with_info(training_frame, 1, self.t("epochs_label"), "epochs", pady=(10, 0))
        self.epochs_spinbox = ttk.Spinbox(training_frame, from_=1, to=5000, textvariable=self.epochs_var, width=8)
        self.epochs_spinbox.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(10, 0))

        self.lr_label = self._grid_label_with_info(training_frame, 2, self.t("learning_rate_label"), "learning_rate")
        self.lr_entry = ttk.Entry(training_frame, textvariable=self.lr_var, width=10)
        self.lr_entry.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        self.batch_label = self._grid_label_with_info(training_frame, 3, self.t("batch_size_label"), "batch_size")
        self.batch_spinbox = ttk.Spinbox(training_frame, from_=1, to=4096, textvariable=self.batch_size_var, width=8)
        self.batch_spinbox.grid(row=3, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        self.weight_scale_label = self._grid_label_with_info(training_frame, 4, self.t("weight_scale_label"), "weight_scale")
        self.weight_scale_entry = ttk.Entry(training_frame, textvariable=self.weight_scale_var, width=10)
        self.weight_scale_entry.grid(row=4, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        self.seed_label = self._grid_label_with_info(training_frame, 5, self.t("seed_label"), "seed")
        self.seed_spinbox = ttk.Spinbox(training_frame, from_=0, to=999999, textvariable=self.seed_var, width=8)
        self.seed_spinbox.grid(row=5, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        training_button_row_1 = ttk.Frame(training_frame, style="White.TFrame")
        training_button_row_1.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Button(training_button_row_1, text=self.t("reinitialize_button"), command=lambda: self._load_experiment(reinitialize_model=True)).pack(side=tk.LEFT)
        ttk.Button(training_button_row_1, text=self.t("train_1_button"), command=lambda: self._train_for_epochs(1)).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(training_button_row_1, text=self.t("train_10_button"), command=lambda: self._train_for_epochs(10)).pack(side=tk.LEFT, padx=(8, 0))
        self.training_button_row_2 = ttk.Frame(training_frame, style="White.TFrame")
        self.training_button_row_2.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(self.training_button_row_2, text=self.t("train_n_button"), command=self._train_current_model).pack(side=tk.LEFT)
        ttk.Button(self.training_button_row_2, text=self.t("save_baseline_button"), command=self._store_baseline).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(self.training_button_row_2, text=self.t("refresh_view_button"), command=self._refresh_views).pack(side=tk.LEFT, padx=(8, 0))

        self.expert_only_widgets.extend(
            [
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

    def _build_playground_controls(self, parent: ttk.Frame) -> None:
        """Baut die SA-spezifischen Steuerungen fuer den Playground Mode."""

        objective_frame = ttk.LabelFrame(parent, text=self.t("objective_frame_title"), padding=10)
        objective_frame.pack(fill=tk.X, pady=(0, 10))
        objective_frame.columnconfigure(1, weight=1)
        objective_hint_row = ttk.Frame(objective_frame, style="White.TFrame")
        objective_hint_row.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(
            objective_hint_row,
            text=self.t("objective_hint"),
            style="Hint.TLabel",
            justify=tk.LEFT,
            wraplength=300,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._create_info_button(objective_hint_row, "objective").pack(side=tk.RIGHT)

        self.objective_label = self._grid_label_with_info(
            objective_frame,
            1,
            self.t("objective_label"),
            "objective",
            pady=(10, 0),
        )
        self.objective_combo = ttk.Combobox(
            objective_frame,
            textvariable=self.objective_var,
            values=SUPPORTED_OBJECTIVES,
            state="readonly",
            width=20,
        )
        self.objective_combo.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(10, 0))
        self.objective_combo.bind("<<ComboboxSelected>>", lambda _event: self._initialize_playground_state())

        self.candidate_epochs_label = self._grid_label_with_info(
            objective_frame,
            2,
            self.t("candidate_epochs_label"),
            "candidate_epochs",
        )
        self.candidate_epochs_spinbox = ttk.Spinbox(
            objective_frame,
            from_=1,
            to=5000,
            textvariable=self.candidate_epochs_var,
            width=8,
        )
        self.candidate_epochs_spinbox.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        self.lr_label = self._grid_label_with_info(objective_frame, 3, self.t("learning_rate_label"), "learning_rate")
        self.lr_entry = ttk.Entry(objective_frame, textvariable=self.lr_var, width=10)
        self.lr_entry.grid(row=3, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        self.batch_label = self._grid_label_with_info(objective_frame, 4, self.t("batch_size_label"), "batch_size")
        self.batch_spinbox = ttk.Spinbox(objective_frame, from_=1, to=4096, textvariable=self.batch_size_var, width=8)
        self.batch_spinbox.grid(row=4, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        self.weight_scale_label = self._grid_label_with_info(objective_frame, 5, self.t("weight_scale_label"), "weight_scale")
        self.weight_scale_entry = ttk.Entry(objective_frame, textvariable=self.weight_scale_var, width=10)
        self.weight_scale_entry.grid(row=5, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        self.seed_label = self._grid_label_with_info(objective_frame, 6, self.t("seed_label"), "seed")
        self.seed_spinbox = ttk.Spinbox(objective_frame, from_=0, to=999999, textvariable=self.seed_var, width=8)
        self.seed_spinbox.grid(row=6, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        shuffle_row = ttk.Frame(objective_frame, style="White.TFrame")
        shuffle_row.grid(row=7, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.playground_shuffle_checkbutton = ttk.Checkbutton(
            shuffle_row,
            text=self.t("shuffle_label"),
            variable=self.playground_shuffle_var,
        )
        self.playground_shuffle_checkbutton.pack(side=tk.LEFT)
        self._create_info_button(shuffle_row, "shuffle").pack(side=tk.LEFT, padx=(8, 0))

        neighborhood_frame = ttk.LabelFrame(parent, text=self.t("neighborhood_frame_title"), padding=10)
        neighborhood_frame.pack(fill=tk.X, pady=(0, 10))
        neighborhood_hint_row = ttk.Frame(neighborhood_frame, style="White.TFrame")
        neighborhood_hint_row.pack(fill=tk.X)
        ttk.Label(
            neighborhood_hint_row,
            text=self.t("neighborhood_hint"),
            style="Hint.TLabel",
            justify=tk.LEFT,
            wraplength=300,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._create_info_button(neighborhood_hint_row, "playground_neighborhood").pack(side=tk.RIGHT)

        self.neighbor_set_checkbutton = ttk.Checkbutton(
            neighborhood_frame,
            text=self.t("neighbor_set_label"),
            variable=self.playground_neighbor_vars["set_neuron"],
        )
        self.neighbor_set_checkbutton.pack(anchor="w", pady=(8, 0))
        self.neighbor_fill_checkbutton = ttk.Checkbutton(
            neighborhood_frame,
            text=self.t("neighbor_fill_label"),
            variable=self.playground_neighbor_vars["fill_layer"],
        )
        self.neighbor_fill_checkbutton.pack(anchor="w", pady=(6, 0))
        self.neighbor_swap_checkbutton = ttk.Checkbutton(
            neighborhood_frame,
            text=self.t("neighbor_swap_label"),
            variable=self.playground_neighbor_vars["swap_neurons"],
        )
        self.neighbor_swap_checkbutton.pack(anchor="w", pady=(6, 0))

        annealing_frame = ttk.LabelFrame(parent, text=self.t("annealing_frame_title"), padding=10)
        annealing_frame.pack(fill=tk.X, pady=(0, 10))
        annealing_frame.columnconfigure(1, weight=1)
        annealing_hint_row = ttk.Frame(annealing_frame, style="White.TFrame")
        annealing_hint_row.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(
            annealing_hint_row,
            text=self.t("annealing_hint"),
            style="Hint.TLabel",
            justify=tk.LEFT,
            wraplength=300,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._create_info_button(annealing_hint_row, "annealing_run_control").pack(side=tk.RIGHT)

        self.start_temperature_label = self._grid_label_with_info(
            annealing_frame,
            1,
            self.t("start_temperature_label"),
            "start_temperature",
            pady=(10, 0),
        )
        self.start_temperature_entry = ttk.Entry(annealing_frame, textvariable=self.start_temperature_var, width=10)
        self.start_temperature_entry.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(10, 0))

        self.cooling_schedule_label = self._grid_label_with_info(
            annealing_frame,
            2,
            self.t("cooling_schedule_label"),
            "cooling_schedule",
        )
        self.cooling_schedule_combo = ttk.Combobox(
            annealing_frame,
            textvariable=self.cooling_schedule_var,
            values=SUPPORTED_COOLING_SCHEDULES,
            state="readonly",
            width=16,
        )
        self.cooling_schedule_combo.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        self.cooling_parameter_label = self._grid_label_with_info(
            annealing_frame,
            3,
            self.t("cooling_parameter_label"),
            "cooling_parameter",
        )
        self.cooling_parameter_entry = ttk.Entry(annealing_frame, textvariable=self.cooling_parameter_var, width=10)
        self.cooling_parameter_entry.grid(row=3, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        self.iterations_per_temperature_label = self._grid_label_with_info(
            annealing_frame,
            4,
            self.t("iterations_per_temperature_label"),
            "iterations_per_temperature",
        )
        self.iterations_per_temperature_spinbox = ttk.Spinbox(
            annealing_frame,
            from_=1,
            to=500,
            textvariable=self.iterations_per_temperature_var,
            width=8,
        )
        self.iterations_per_temperature_spinbox.grid(row=4, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        self.max_steps_label = self._grid_label_with_info(
            annealing_frame,
            5,
            self.t("max_steps_label"),
            "max_steps",
        )
        self.max_steps_spinbox = ttk.Spinbox(
            annealing_frame,
            from_=1,
            to=5000,
            textvariable=self.max_steps_var,
            width=8,
        )
        self.max_steps_spinbox.grid(row=5, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        self.min_temperature_label = self._grid_label_with_info(
            annealing_frame,
            6,
            self.t("min_temperature_label"),
            "min_temperature",
        )
        self.min_temperature_entry = ttk.Entry(annealing_frame, textvariable=self.min_temperature_var, width=10)
        self.min_temperature_entry.grid(row=6, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        run_frame = ttk.LabelFrame(parent, text=self.t("run_frame_title"), padding=10)
        run_frame.pack(fill=tk.X, pady=(0, 10))
        run_hint_row = ttk.Frame(run_frame, style="White.TFrame")
        run_hint_row.pack(fill=tk.X)
        ttk.Label(
            run_hint_row,
            text=self.t("run_hint"),
            style="Hint.TLabel",
            justify=tk.LEFT,
            wraplength=300,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._create_info_button(run_hint_row, "annealing_run_control").pack(side=tk.RIGHT)

        run_button_row_1 = ttk.Frame(run_frame, style="White.TFrame")
        run_button_row_1.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(run_button_row_1, text=self.t("evaluate_start_button"), command=self._initialize_playground_state).pack(side=tk.LEFT)
        ttk.Button(run_button_row_1, text=self.t("anneal_step_button"), command=lambda: self._run_annealing_steps(1)).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(run_button_row_1, text=self.t("anneal_10_button"), command=lambda: self._run_annealing_steps(10)).pack(side=tk.LEFT, padx=(8, 0))
        run_button_row_2 = ttk.Frame(run_frame, style="White.TFrame")
        run_button_row_2.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(run_button_row_2, text=self.t("anneal_run_button"), command=self._run_annealing_to_completion).pack(side=tk.LEFT)
        ttk.Button(run_button_row_2, text=self.t("anneal_reset_button"), command=self._initialize_playground_state).pack(side=tk.LEFT, padx=(8, 0))

    def _build_status_frame(self, parent: ttk.Frame) -> None:
        """Baut den unteren Statusblock der linken Spalte."""

        frame_title = (
            self.t("anneal_status_frame_title")
            if self._app_mode() == "playground"
            else self.t("status_frame_title")
        )
        status_frame = ttk.LabelFrame(parent, text=frame_title, padding=10)
        status_frame.pack(fill=tk.BOTH, expand=True)
        status_top_row = ttk.Frame(status_frame, style="White.TFrame")
        status_top_row.pack(fill=tk.X)
        info_key = "annealing_status" if self._app_mode() == "playground" else "status"
        self._create_info_button(status_top_row, info_key).pack(side=tk.RIGHT)
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
        if self._app_mode() == "playground":
            ttk.Label(
                status_frame,
                textvariable=self.playground_summary_var,
                justify=tk.LEFT,
                style="SectionValue.TLabel",
                wraplength=330,
            ).pack(anchor="w", pady=(10, 0))
            ttk.Label(
                status_frame,
                textvariable=self.playground_decision_var,
                justify=tk.LEFT,
                style="Hint.TLabel",
                wraplength=330,
            ).pack(anchor="w", pady=(10, 0))
        else:
            ttk.Label(
                status_frame,
                textvariable=self.compare_summary_var,
                justify=tk.LEFT,
                style="Hint.TLabel",
                wraplength=330,
            ).pack(anchor="w", pady=(10, 0))

    def _build_network_preview_frame(
        self,
        parent: ttk.Frame,
        row: int,
        pady: tuple[int, int] = (10, 10),
    ) -> None:
        """Baut die gemeinsame Netzvorschau fuer Demo, Playground und Builder."""

        self.canvas_frame = ttk.LabelFrame(parent, text=self.t("canvas_frame_title"), padding=8)
        self.canvas_frame.grid(row=row, column=0, sticky="nsew", pady=pady)
        self.canvas_frame.columnconfigure(0, weight=5)
        self.canvas_frame.columnconfigure(1, weight=3)
        self.canvas_frame.rowconfigure(0, weight=0)
        self.canvas_frame.rowconfigure(1, weight=1)
        self.canvas_frame.rowconfigure(2, weight=0)
        self.canvas_frame.bind("<Configure>", self._on_canvas_frame_configure)

        network_toolbar = ttk.Frame(self.canvas_frame, style="White.TFrame")
        network_toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        network_toolbar.columnconfigure(0, weight=1)
        ttk.Label(
            network_toolbar,
            text=self.t("network_toolbar_hint"),
            style="Hint.TLabel",
            justify=tk.LEFT,
            wraplength=760,
        ).grid(row=0, column=0, sticky="w")
        network_button_row = ttk.Frame(network_toolbar, style="White.TFrame")
        network_button_row.grid(row=0, column=1, sticky="e", padx=(12, 0))
        self._create_info_action_button(
            network_button_row,
            self.t("network_info_button"),
            "network_visualization",
        ).pack(side=tk.LEFT)
        self._create_info_action_button(
            network_button_row,
            self.t("activation_curve_info_button"),
            "activation_curve",
        ).pack(side=tk.LEFT, padx=(8, 0))

        self.canvas = tk.Canvas(self.canvas_frame, bg=NETWORK_BACKGROUND_COLOR, highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self._redraw_network())

        self.live_detail_frame = ttk.Frame(self.canvas_frame, style="White.TFrame", padding=(8, 0, 0, 0))
        self.live_detail_frame.grid(row=1, column=1, sticky="nsew")
        self.live_detail_frame.columnconfigure(0, weight=1)
        self.live_detail_frame.rowconfigure(2, weight=1)
        self.live_detail_frame.rowconfigure(3, weight=0)

        ttk.Label(
            self.live_detail_frame,
            text=self.t("live_inspection_title"),
            style="Headline.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            self.live_detail_frame,
            text=self.t("live_inspection_hint"),
            style="Hint.TLabel",
            justify=tk.LEFT,
            wraplength=360,
        ).grid(row=1, column=0, sticky="ew", pady=(6, 10))
        self.live_detail_text = ScrolledText(self.live_detail_frame, wrap=tk.WORD, width=44, height=18)
        self.live_detail_text.grid(row=2, column=0, sticky="nsew")
        self.live_detail_text.configure(state=tk.DISABLED, font=("Menlo", 10))
        self.activation_figure = Figure(figsize=(3.5, 2.4), dpi=100)
        self.activation_axis = self.activation_figure.add_subplot(111)
        self.activation_canvas = FigureCanvasTkAgg(self.activation_figure, master=self.live_detail_frame)
        self.activation_canvas.get_tk_widget().grid(row=3, column=0, sticky="ew", pady=(10, 0))

        self._update_canvas_layout()

    def _build_experiment_builder_controls(self, parent: ttk.Frame) -> None:
        """Baut die linke Steuerleiste fuer den Experiment Builder."""

        self.builder_setup_frame = ttk.LabelFrame(parent, text=self.t("builder_setup_title"), padding=10)
        self.builder_setup_frame.pack(fill=tk.X, pady=(0, 10))
        self.builder_setup_frame.columnconfigure(1, weight=1)
        setup_hint_row = ttk.Frame(self.builder_setup_frame, style="White.TFrame")
        setup_hint_row.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(
            setup_hint_row,
            text=self.t("builder_setup_hint"),
            style="Hint.TLabel",
            justify=tk.LEFT,
            wraplength=300,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._create_info_button(setup_hint_row, "builder_setup").pack(side=tk.RIGHT)

        self._grid_label_with_info(
            self.builder_setup_frame,
            1,
            self.t("builder_experiment_name"),
            "builder_setup",
            pady=(10, 0),
        )
        ttk.Entry(self.builder_setup_frame, textvariable=self.builder_experiment_name_var).grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(10, 0),
        )

        self._grid_label_with_info(self.builder_setup_frame, 2, self.t("benchmark_label"), "benchmark")
        benchmark_combo = ttk.Combobox(
            self.builder_setup_frame,
            textvariable=self.benchmark_var,
            values=SUPPORTED_BENCHMARKS,
            state="readonly",
            width=18,
        )
        benchmark_combo.grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        benchmark_combo.bind("<<ComboboxSelected>>", self._on_benchmark_changed)

        self._grid_label_with_info(
            self.builder_setup_frame,
            3,
            self.t("hidden_layers_label"),
            "hidden_layers",
        )
        self.layer_size_controls_frame = ttk.Frame(self.builder_setup_frame, style="White.TFrame")
        self.layer_size_controls_frame.grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))

        hidden_button_row = ttk.Frame(self.builder_setup_frame, style="White.TFrame")
        hidden_button_row.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(
            hidden_button_row,
            text=self.t("add_layer_button"),
            command=self._add_hidden_layer,
        ).pack(side=tk.LEFT)
        ttk.Button(
            hidden_button_row,
            text=self.t("remove_layer_button"),
            command=self._remove_hidden_layer,
        ).pack(side=tk.LEFT, padx=(8, 0))

        self._grid_label_with_info(self.builder_setup_frame, 5, self.t("current_layout_label"), "layout")
        self.builder_layout_entry = ttk.Entry(self.builder_setup_frame, textvariable=self.layout_string_var)
        self.builder_layout_entry.grid(
            row=5,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(6, 0),
        )
        self.builder_layout_entry.bind("<Return>", self._on_layout_string_edited)
        self.builder_layout_entry.bind("<FocusOut>", self._on_layout_string_edited)

        self.builder_layout_editor_frame = ttk.LabelFrame(
            self.builder_setup_frame,
            text=self.t("builder_layout_editor_title"),
            padding=10,
        )
        self.builder_layout_editor_frame.grid(
            row=6,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(10, 0),
        )
        self.builder_layout_editor_frame.columnconfigure(0, weight=1)
        ttk.Label(
            self.builder_layout_editor_frame,
            text=self.t("builder_layout_editor_hint"),
            style="Hint.TLabel",
            justify=tk.LEFT,
            wraplength=300,
        ).grid(row=0, column=0, columnspan=2, sticky="ew")
        editor_info_row = ttk.Frame(self.builder_layout_editor_frame, style="White.TFrame")
        editor_info_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self._create_info_button(editor_info_row, "builder_layout_editor").pack(side=tk.RIGHT)
        self._grid_label_with_info(
            self.builder_layout_editor_frame,
            2,
            self.t("layer_fill_label"),
            "layer_fill",
            pady=(8, 0),
        )
        self.builder_layer_fill_controls_frame = ttk.Frame(
            self.builder_layout_editor_frame,
            style="White.TFrame",
        )
        self.builder_layer_fill_controls_frame.grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(6, 0),
        )
        self.builder_neuron_layer_label = self._grid_label_with_info(
            self.builder_layout_editor_frame,
            4,
            self.t("layer_label"),
            "neuron_layer",
            pady=(10, 0),
        )
        self.builder_neuron_layer_combo = ttk.Combobox(
            self.builder_layout_editor_frame,
            textvariable=self.neuron_layer_var,
            values=("L1",),
            state="readonly",
            width=8,
        )
        self.builder_neuron_layer_combo.grid(row=4, column=1, sticky="w", padx=(8, 0), pady=(10, 0))
        self.builder_neuron_layer_combo.bind("<<ComboboxSelected>>", self._on_neuron_layer_changed)

        self.builder_neuron_index_label = self._grid_label_with_info(
            self.builder_layout_editor_frame,
            5,
            self.t("neuron_label"),
            "neuron_index",
        )
        self.builder_neuron_index_combo = ttk.Combobox(
            self.builder_layout_editor_frame,
            textvariable=self.neuron_index_var,
            values=("0",),
            state="readonly",
            width=8,
        )
        self.builder_neuron_index_combo.grid(row=5, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        self.builder_neuron_index_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._select_from_controls(),
        )

        self.builder_neuron_activation_label = self._grid_label_with_info(
            self.builder_layout_editor_frame,
            6,
            self.t("activation_label"),
            "neuron_activation",
        )
        self.builder_neuron_activation_combo = ttk.Combobox(
            self.builder_layout_editor_frame,
            textvariable=self.neuron_activation_var,
            values=SUPPORTED_ACTIVATIONS,
            state="readonly",
            width=14,
        )
        self.builder_neuron_activation_combo.grid(row=6, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        self.builder_neuron_button_row = ttk.Frame(self.builder_layout_editor_frame, style="White.TFrame")
        self.builder_neuron_button_row.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Button(
            self.builder_neuron_button_row,
            text=self.t("set_neuron_button"),
            command=self._apply_neuron_setting,
        ).pack(side=tk.LEFT)
        ttk.Button(
            self.builder_neuron_button_row,
            text=self.t("cycle_button"),
            command=self._cycle_selected_neuron,
        ).pack(side=tk.LEFT, padx=(8, 0))

        self.builder_layout_row = self._grid_label_with_info(
            self.builder_setup_frame,
            7,
            self.t("builder_layout_choice"),
            "builder_layout_choice",
        )
        self.builder_layout_combo = ttk.Combobox(
            self.builder_setup_frame,
            textvariable=self.builder_layout_choice_var,
            state="readonly",
            width=26,
        )
        self.builder_layout_combo.grid(row=7, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        self.builder_layout_combo.bind("<<ComboboxSelected>>", self._on_builder_layout_option_selected)

        self._grid_label_with_info(
            self.builder_setup_frame,
            8,
            self.t("builder_run_mode"),
            "builder_setup",
        )
        builder_run_mode_combo = ttk.Combobox(
            self.builder_setup_frame,
            textvariable=self.builder_run_mode_display_var,
            values=[self._run_mode_label(value) for value in SUPPORTED_EXPERIMENT_RUN_MODES],
            state="readonly",
            width=22,
        )
        builder_run_mode_combo.grid(row=8, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        builder_run_mode_combo.bind("<<ComboboxSelected>>", self._on_builder_run_mode_selected)

        self._grid_label_with_info(
            self.builder_setup_frame,
            9,
            self.t("builder_primary_metric"),
            "builder_primary_metric",
        )
        ttk.Combobox(
            self.builder_setup_frame,
            textvariable=self.builder_primary_metric_var,
            values=PRIMARY_METRIC_LABELS,
            state="readonly",
            width=22,
        ).grid(row=9, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))

        ttk.Label(
            self.builder_setup_frame,
            text=self.t("builder_primary_metric_hint"),
            style="Hint.TLabel",
            justify=tk.LEFT,
            wraplength=300,
        ).grid(row=10, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        self.builder_seeds_frame = ttk.LabelFrame(parent, text=self.t("builder_seeds_title"), padding=10)
        self.builder_seeds_frame.pack(fill=tk.X, pady=(0, 10))
        self.builder_seeds_frame.columnconfigure(1, weight=1)
        seed_hint_row = ttk.Frame(self.builder_seeds_frame, style="White.TFrame")
        seed_hint_row.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(
            seed_hint_row,
            text=self.t("builder_seeds_hint"),
            style="Hint.TLabel",
            justify=tk.LEFT,
            wraplength=300,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._create_info_button(seed_hint_row, "builder_seeds").pack(side=tk.RIGHT)

        self._grid_label_with_info(
            self.builder_seeds_frame,
            1,
            self.t("builder_seed_strategy"),
            "builder_seed_strategy",
            pady=(10, 0),
        )
        seed_strategy_combo = ttk.Combobox(
            self.builder_seeds_frame,
            textvariable=self.builder_seed_strategy_display_var,
            values=[self._seed_strategy_label(value) for value in SEED_STRATEGY_LABELS["de"]],
            state="readonly",
            width=22,
        )
        seed_strategy_combo.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(10, 0))
        seed_strategy_combo.bind("<<ComboboxSelected>>", self._on_builder_seed_strategy_selected)

        self._grid_label_with_info(
            self.builder_seeds_frame,
            2,
            self.t("builder_seed_count"),
            "builder_seed_strategy",
        )
        self.builder_seed_count_spinbox = ttk.Spinbox(
            self.builder_seeds_frame,
            from_=1,
            to=256,
            textvariable=self.builder_seed_count_var,
            width=8,
            command=self._update_builder_seed_preview,
        )
        self.builder_seed_count_spinbox.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        self.builder_seed_count_spinbox.bind("<Return>", lambda _event: self._update_builder_seed_preview())

        self._grid_label_with_info(
            self.builder_seeds_frame,
            3,
            self.t("builder_seed_start"),
            "builder_seed_strategy",
        )
        self.builder_seed_start_spinbox = ttk.Spinbox(
            self.builder_seeds_frame,
            from_=0,
            to=999999,
            textvariable=self.builder_seed_start_var,
            width=8,
            command=self._update_builder_seed_preview,
        )
        self.builder_seed_start_spinbox.grid(row=3, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        self.builder_seed_start_spinbox.bind("<Return>", lambda _event: self._update_builder_seed_preview())

        self._grid_label_with_info(
            self.builder_seeds_frame,
            4,
            self.t("builder_seed_list"),
            "builder_seed_strategy",
        )
        self.builder_seed_list_entry = ttk.Entry(
            self.builder_seeds_frame,
            textvariable=self.builder_seed_list_var,
        )
        self.builder_seed_list_entry.grid(row=4, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        self.builder_seed_list_entry.bind("<KeyRelease>", lambda _event: self._update_builder_seed_preview())

        self._grid_label_with_info(
            self.builder_seeds_frame,
            5,
            self.t("builder_seed_preview"),
            "builder_seeds",
        )
        ttk.Label(
            self.builder_seeds_frame,
            textvariable=self.builder_seed_preview_var,
            justify=tk.LEFT,
            style="SectionValue.TLabel",
            wraplength=300,
        ).grid(row=5, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        self.builder_training_frame = ttk.LabelFrame(parent, text=self.t("builder_training_title"), padding=10)
        self.builder_training_frame.pack(fill=tk.X, pady=(0, 10))
        self.builder_training_frame.columnconfigure(1, weight=1)
        training_hint_row = ttk.Frame(self.builder_training_frame, style="White.TFrame")
        training_hint_row.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(
            training_hint_row,
            text=self.t("builder_training_hint"),
            style="Hint.TLabel",
            justify=tk.LEFT,
            wraplength=300,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._create_info_button(training_hint_row, "builder_training").pack(side=tk.RIGHT)

        self._grid_label_with_info(self.builder_training_frame, 1, self.t("epochs_label"), "epochs", pady=(10, 0))
        ttk.Spinbox(
            self.builder_training_frame,
            from_=1,
            to=5000,
            textvariable=self.epochs_var,
            width=8,
        ).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(10, 0))

        self._grid_label_with_info(self.builder_training_frame, 2, self.t("learning_rate_label"), "learning_rate")
        ttk.Entry(self.builder_training_frame, textvariable=self.lr_var, width=10).grid(
            row=2,
            column=1,
            sticky="w",
            padx=(8, 0),
            pady=(6, 0),
        )

        self._grid_label_with_info(self.builder_training_frame, 3, self.t("batch_size_label"), "batch_size")
        ttk.Spinbox(
            self.builder_training_frame,
            from_=1,
            to=4096,
            textvariable=self.batch_size_var,
            width=8,
        ).grid(row=3, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        self._grid_label_with_info(self.builder_training_frame, 4, self.t("weight_scale_label"), "weight_scale")
        ttk.Entry(self.builder_training_frame, textvariable=self.weight_scale_var, width=10).grid(
            row=4,
            column=1,
            sticky="w",
            padx=(8, 0),
            pady=(6, 0),
        )

        self._grid_label_with_info(self.builder_training_frame, 5, self.t("seed_label"), "seed")
        ttk.Spinbox(
            self.builder_training_frame,
            from_=0,
            to=999999,
            textvariable=self.builder_random_seed_var,
            width=8,
        ).grid(row=5, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        shuffle_row = ttk.Frame(self.builder_training_frame, style="White.TFrame")
        shuffle_row.grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Checkbutton(
            shuffle_row,
            text=self.t("shuffle_label"),
            variable=self.playground_shuffle_var,
        ).pack(side=tk.LEFT)
        self._create_info_button(shuffle_row, "shuffle").pack(side=tk.LEFT, padx=(8, 0))

        self.builder_sa_frame = ttk.LabelFrame(parent, text=self.t("builder_sa_title"), padding=10)
        self.builder_sa_frame.columnconfigure(1, weight=1)
        sa_hint_row = ttk.Frame(self.builder_sa_frame, style="White.TFrame")
        sa_hint_row.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(
            sa_hint_row,
            text=self.t("builder_sa_hint"),
            style="Hint.TLabel",
            justify=tk.LEFT,
            wraplength=300,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._create_info_button(sa_hint_row, "builder_sa").pack(side=tk.RIGHT)

        self._grid_label_with_info(self.builder_sa_frame, 1, self.t("objective_label"), "objective", pady=(10, 0))
        ttk.Combobox(
            self.builder_sa_frame,
            textvariable=self.objective_var,
            values=SUPPORTED_OBJECTIVES,
            state="readonly",
            width=20,
        ).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(10, 0))

        self._grid_label_with_info(
            self.builder_sa_frame,
            2,
            self.t("candidate_epochs_label"),
            "candidate_epochs",
        )
        ttk.Spinbox(
            self.builder_sa_frame,
            from_=1,
            to=5000,
            textvariable=self.candidate_epochs_var,
            width=8,
        ).grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        self._grid_label_with_info(
            self.builder_sa_frame,
            3,
            self.t("start_temperature_label"),
            "start_temperature",
        )
        ttk.Entry(self.builder_sa_frame, textvariable=self.start_temperature_var, width=10).grid(
            row=3,
            column=1,
            sticky="w",
            padx=(8, 0),
            pady=(6, 0),
        )

        self._grid_label_with_info(
            self.builder_sa_frame,
            4,
            self.t("cooling_schedule_label"),
            "cooling_schedule",
        )
        ttk.Combobox(
            self.builder_sa_frame,
            textvariable=self.cooling_schedule_var,
            values=SUPPORTED_COOLING_SCHEDULES,
            state="readonly",
            width=16,
        ).grid(row=4, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        self._grid_label_with_info(
            self.builder_sa_frame,
            5,
            self.t("cooling_parameter_label"),
            "cooling_parameter",
        )
        ttk.Entry(self.builder_sa_frame, textvariable=self.cooling_parameter_var, width=10).grid(
            row=5,
            column=1,
            sticky="w",
            padx=(8, 0),
            pady=(6, 0),
        )

        self._grid_label_with_info(
            self.builder_sa_frame,
            6,
            self.t("iterations_per_temperature_label"),
            "iterations_per_temperature",
        )
        ttk.Spinbox(
            self.builder_sa_frame,
            from_=1,
            to=500,
            textvariable=self.iterations_per_temperature_var,
            width=8,
        ).grid(row=6, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        self._grid_label_with_info(
            self.builder_sa_frame,
            7,
            self.t("max_steps_label"),
            "max_steps",
        )
        ttk.Spinbox(
            self.builder_sa_frame,
            from_=1,
            to=5000,
            textvariable=self.max_steps_var,
            width=8,
        ).grid(row=7, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        self._grid_label_with_info(
            self.builder_sa_frame,
            8,
            self.t("min_temperature_label"),
            "min_temperature",
        )
        ttk.Entry(self.builder_sa_frame, textvariable=self.min_temperature_var, width=10).grid(
            row=8,
            column=1,
            sticky="w",
            padx=(8, 0),
            pady=(6, 0),
        )

        neighborhood_row = ttk.Frame(self.builder_sa_frame, style="White.TFrame")
        neighborhood_row.grid(row=9, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Checkbutton(
            neighborhood_row,
            text=self.t("neighbor_set_label"),
            variable=self.playground_neighbor_vars["set_neuron"],
        ).pack(anchor="w")
        ttk.Checkbutton(
            neighborhood_row,
            text=self.t("neighbor_fill_label"),
            variable=self.playground_neighbor_vars["fill_layer"],
        ).pack(anchor="w", pady=(4, 0))
        ttk.Checkbutton(
            neighborhood_row,
            text=self.t("neighbor_swap_label"),
            variable=self.playground_neighbor_vars["swap_neurons"],
        ).pack(anchor="w", pady=(4, 0))

        self.builder_tuning_frame = ttk.LabelFrame(parent, text=self.t("builder_tuning_title"), padding=10)
        self.builder_tuning_frame.pack(fill=tk.X, pady=(0, 10))
        tuning_hint_row = ttk.Frame(self.builder_tuning_frame, style="White.TFrame")
        tuning_hint_row.pack(fill=tk.X)
        ttk.Label(
            tuning_hint_row,
            text=self.t("builder_tuning_hint"),
            style="Hint.TLabel",
            justify=tk.LEFT,
            wraplength=300,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._create_info_button(tuning_hint_row, "builder_search").pack(side=tk.RIGHT)

        search_type_row = ttk.Frame(self.builder_tuning_frame, style="White.TFrame")
        search_type_row.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(search_type_row, text=self.t("builder_search_type")).pack(side=tk.LEFT)
        self._create_info_button(search_type_row, "builder_search").pack(side=tk.LEFT, padx=(6, 0))
        search_type_combo = ttk.Combobox(
            search_type_row,
            textvariable=self.builder_search_type_display_var,
            values=[self._search_type_label(value) for value in SUPPORTED_SEARCH_TYPES],
            state="readonly",
            width=18,
        )
        search_type_combo.pack(side=tk.RIGHT)
        search_type_combo.bind("<<ComboboxSelected>>", self._on_builder_search_type_selected)

        random_samples_row = ttk.Frame(self.builder_tuning_frame, style="White.TFrame")
        random_samples_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(random_samples_row, text=self.t("builder_random_samples")).pack(side=tk.LEFT)
        ttk.Spinbox(
            random_samples_row,
            from_=1,
            to=1024,
            textvariable=self.builder_random_samples_var,
            width=8,
        ).pack(side=tk.RIGHT)

        tuning_parameters_frame = ttk.Frame(self.builder_tuning_frame, style="White.TFrame")
        tuning_parameters_frame.pack(fill=tk.X, pady=(10, 0))
        for parameter_name, label_text, is_sa_only in self._builder_search_parameter_specs():
            row = ttk.Frame(tuning_parameters_frame, style="White.TFrame")
            row.pack(fill=tk.X, pady=(0, 8))
            left = ttk.Frame(row, style="White.TFrame")
            left.pack(fill=tk.X)
            ttk.Label(left, text=label_text).pack(side=tk.LEFT)
            self._create_info_button(left, "builder_search_value").pack(side=tk.LEFT, padx=(6, 0))
            mode_combo = ttk.Combobox(
                row,
                textvariable=self.builder_tuning_mode_display_vars[parameter_name],
                values=[self._search_value_kind_label(value) for value in ("fixed", "list", "range")],
                state="readonly",
                width=10,
            )
            mode_combo.pack(side=tk.LEFT, padx=(0, 6))
            mode_combo.bind(
                "<<ComboboxSelected>>",
                lambda _event, param=parameter_name: self._on_builder_search_value_kind_selected(param),
            )
            ttk.Entry(row, textvariable=self.builder_tuning_value_vars[parameter_name], width=24).pack(
                side=tk.LEFT,
                fill=tk.X,
                expand=True,
            )
            suggested_values = ", ".join(SUGGESTED_TUNING_VALUES.get(parameter_name, ()))
            if suggested_values:
                ttk.Label(
                    tuning_parameters_frame,
                    text=(
                        f"Beispielwerte fuer {label_text}: {suggested_values}"
                        if not self._is_english()
                        else f"Suggested values for {label_text}: {suggested_values}"
                    ),
                    style="Hint.TLabel",
                    justify=tk.LEFT,
                    wraplength=300,
                ).pack(anchor="w", pady=(0, 4))
            self.builder_tuning_rows[parameter_name] = row
            if is_sa_only:
                row._builder_sa_only = True  # type: ignore[attr-defined]

        self.builder_storage_frame = ttk.LabelFrame(parent, text=self.t("builder_storage_title"), padding=10)
        self.builder_storage_frame.pack(fill=tk.X, pady=(0, 10))
        self.builder_storage_frame.columnconfigure(1, weight=1)
        storage_hint_row = ttk.Frame(self.builder_storage_frame, style="White.TFrame")
        storage_hint_row.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(
            storage_hint_row,
            text=self.t("builder_storage_hint"),
            style="Hint.TLabel",
            justify=tk.LEFT,
            wraplength=300,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._create_info_button(storage_hint_row, "builder_storage").pack(side=tk.RIGHT)

        self._grid_label_with_info(
            self.builder_storage_frame,
            1,
            self.t("builder_output_dir"),
            "builder_output_dir",
            pady=(10, 0),
        )
        ttk.Entry(self.builder_storage_frame, textvariable=self.builder_output_dir_var).grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(10, 0),
        )

        storage_check_row = ttk.Frame(self.builder_storage_frame, style="White.TFrame")
        storage_check_row.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Checkbutton(
            storage_check_row,
            text=self.t("builder_save_json"),
            variable=self.builder_save_json_var,
        ).pack(side=tk.LEFT)

        self._grid_label_with_info(
            self.builder_storage_frame,
            3,
            self.t("builder_load_path"),
            "builder_load_path",
        )
        ttk.Entry(self.builder_storage_frame, textvariable=self.builder_load_path_var).grid(
            row=3,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(6, 0),
        )

        self.builder_run_frame = ttk.LabelFrame(parent, text=self.t("builder_run_title"), padding=10)
        self.builder_run_frame.pack(fill=tk.X, pady=(0, 10))
        run_hint_row = ttk.Frame(self.builder_run_frame, style="White.TFrame")
        run_hint_row.pack(fill=tk.X)
        ttk.Label(
            run_hint_row,
            text=self.t("builder_run_hint"),
            style="Hint.TLabel",
            justify=tk.LEFT,
            wraplength=300,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._create_info_button(run_hint_row, "builder_run_control").pack(side=tk.RIGHT)

        button_row = ttk.Frame(self.builder_run_frame, style="White.TFrame")
        button_row.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(
            button_row,
            text=self.t("builder_validate_button"),
            command=self._validate_builder_experiment,
        ).pack(side=tk.LEFT)
        ttk.Button(
            button_row,
            text=self.t("builder_run_button"),
            command=self._run_builder_experiment,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(
            button_row,
            text=self.t("builder_load_results_button"),
            command=self._load_builder_results,
        ).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(
            self.builder_run_frame,
            textvariable=self.builder_status_var,
            justify=tk.LEFT,
            style="SectionValue.TLabel",
            wraplength=320,
        ).pack(anchor="w", pady=(10, 0))

        self.builder_expert_only_widgets = [
            self.builder_layout_editor_frame,
            self.builder_layout_row,
            self.builder_layout_combo,
        ]

        self._rebuild_hidden_size_controls()
        self._update_builder_seed_preview()
        self._update_builder_layout_options()
        self._apply_builder_detail_level_visibility()
        self._apply_builder_run_mode_visibility()
        self._apply_builder_search_type_visibility()

    def _build_experiment_builder_content(self, parent: ttk.Frame) -> None:
        """Baut den rechten Analysebereich fuer den Experiment Builder."""

        summary_frame = ttk.LabelFrame(parent, text=self.t("builder_summary_title"), padding=10)
        summary_frame.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            summary_frame,
            textvariable=self.builder_summary_var,
            justify=tk.LEFT,
            style="SectionValue.TLabel",
            wraplength=1040,
        ).pack(anchor="w")
        ttk.Label(
            summary_frame,
            textvariable=self.network_summary_var,
            justify=tk.LEFT,
            style="Hint.TLabel",
            wraplength=1040,
        ).pack(anchor="w", pady=(8, 0))

        self._build_network_preview_frame(parent, row=1, pady=(10, 10))

        notebook_frame = ttk.LabelFrame(parent, text=self.t("notebook_frame_title"), padding=6)
        notebook_frame.grid(row=2, column=0, sticky="nsew")
        notebook_frame.columnconfigure(0, weight=1)
        notebook_frame.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(notebook_frame)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.builder_overview_tab = ttk.Frame(self.notebook, style="Notebook.TFrame")
        self.builder_runs_tab = ttk.Frame(self.notebook, style="Notebook.TFrame")
        self.builder_analysis_tab = ttk.Frame(self.notebook, style="Notebook.TFrame")
        self.builder_detail_tab_frame = ttk.Frame(self.notebook, style="Notebook.TFrame")
        self.builder_plots_tab = ttk.Frame(self.notebook, style="Notebook.TFrame")
        self.help_tab = ttk.Frame(self.notebook, style="Notebook.TFrame")

        self.notebook.add(self.builder_overview_tab, text=self.t("builder_overview_tab"))
        self.notebook.add(self.builder_runs_tab, text=self.t("builder_runs_tab"))
        self.notebook.add(self.builder_analysis_tab, text=self.t("builder_analysis_tab"))
        self.notebook.add(self.builder_detail_tab_frame, text=self.t("builder_detail_tab"))
        self.notebook.add(self.builder_plots_tab, text=self.t("builder_plots_tab"))
        self.notebook.add(self.help_tab, text=self.t("builder_help_tab"))

        self.builder_overview_tab.columnconfigure(0, weight=1)
        self.builder_overview_tab.rowconfigure(1, weight=1)
        ttk.Label(
            self.builder_overview_tab,
            text=self.t("builder_mode_hint"),
            style="Hint.TLabel",
            justify=tk.LEFT,
        ).grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        self.builder_summary_text = ScrolledText(self.builder_overview_tab, wrap=tk.WORD)
        self.builder_summary_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.builder_summary_text.configure(state=tk.DISABLED, font=("Menlo", 11))

        self.builder_runs_tab.columnconfigure(0, weight=1)
        self.builder_runs_tab.rowconfigure(1, weight=1)
        ttk.Label(
            self.builder_runs_tab,
            text=self.t("builder_run_tree_hint"),
            style="Hint.TLabel",
            justify=tk.LEFT,
        ).grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        columns = ("run_id", "config_id", "seed", "val_acc", "val_loss", "test_acc", "mode")
        self.builder_run_tree = ttk.Treeview(
            self.builder_runs_tab,
            columns=columns,
            show="headings",
            height=18,
        )
        for column, title, width in (
            ("run_id", "run_id", 190),
            ("config_id", "config", 90),
            ("seed", "seed", 70),
            ("val_acc", "val_acc", 90),
            ("val_loss", "val_loss", 90),
            ("test_acc", "test_acc", 90),
            ("mode", "mode", 140),
        ):
            self.builder_run_tree.heading(column, text=title)
            self.builder_run_tree.column(column, width=width, anchor="w")
        self.builder_run_tree.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.builder_run_tree.bind("<<TreeviewSelect>>", self._on_builder_run_selected)

        self.builder_analysis_tab.columnconfigure(0, weight=1)
        self.builder_analysis_tab.rowconfigure(1, weight=1)
        ttk.Label(
            self.builder_analysis_tab,
            text=self.t("builder_analysis_hint"),
            style="Hint.TLabel",
            justify=tk.LEFT,
        ).grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        self.builder_analysis_text = ScrolledText(self.builder_analysis_tab, wrap=tk.WORD)
        self.builder_analysis_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.builder_analysis_text.configure(state=tk.DISABLED, font=("Menlo", 11))

        self.builder_detail_tab_frame.columnconfigure(0, weight=1)
        self.builder_detail_tab_frame.rowconfigure(1, weight=1)
        ttk.Label(
            self.builder_detail_tab_frame,
            text=self.t("builder_detail_hint"),
            style="Hint.TLabel",
            justify=tk.LEFT,
        ).grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        self.builder_detail_text = ScrolledText(self.builder_detail_tab_frame, wrap=tk.WORD)
        self.builder_detail_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.builder_detail_text.configure(state=tk.DISABLED, font=("Menlo", 11))

        self.builder_plots_tab.columnconfigure(0, weight=1)
        self.builder_plots_tab.rowconfigure(1, weight=1)
        ttk.Label(
            self.builder_plots_tab,
            text=self.t("builder_plot_hint"),
            style="Hint.TLabel",
            justify=tk.LEFT,
        ).grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        self.builder_plot_container = ttk.Frame(self.builder_plots_tab, style="White.TFrame")
        self.builder_plot_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.builder_plot_container.columnconfigure(0, weight=1)
        self.builder_plot_container.rowconfigure(0, weight=1)

        self._build_help_tab()

    def _builder_search_parameter_specs(self) -> list[tuple[str, str, bool]]:
        """Beschreibt die Builder-Tuningparameter mit Label und SA-Relevanz."""

        return [
            ("learning_rate", self.t("learning_rate_label"), False),
            ("batch_size", self.t("batch_size_label"), False),
            ("weight_scale", self.t("weight_scale_label"), False),
            ("epochs", self.t("epochs_label"), False),
            ("candidate_epochs", self.t("candidate_epochs_label"), True),
            ("start_temperature", self.t("start_temperature_label"), True),
            ("cooling_schedule", self.t("cooling_schedule_label"), True),
            ("cooling_parameter", self.t("cooling_parameter_label"), True),
        ]

    def _on_builder_run_mode_selected(self, _event=None) -> None:
        """Synchronisiert das sichtbare Builder-Run-Mode-Label."""

        self.builder_run_mode_var.set(
            self._run_mode_from_label(self.builder_run_mode_display_var.get())
        )
        self._apply_builder_run_mode_visibility()
        self._refresh_views()

    def _on_builder_search_type_selected(self, _event=None) -> None:
        """Synchronisiert Search-Type-Label und Sichtbarkeit."""

        self.builder_search_type_var.set(
            self._search_type_from_label(self.builder_search_type_display_var.get())
        )
        self._apply_builder_search_type_visibility()
        self._refresh_views()

    def _on_builder_seed_strategy_selected(self, _event=None) -> None:
        """Synchronisiert das sichtbare Seed-Strategie-Label."""

        self.builder_seed_strategy_var.set(
            self._seed_strategy_from_label(self.builder_seed_strategy_display_var.get())
        )
        self._update_builder_seed_preview()

    def _on_builder_search_value_kind_selected(self, parameter_name: str) -> None:
        """Uebernimmt den sichtbaren Search-Kind-Wert in den internen Builder-Zustand."""

        selected_kind = self._search_value_kind_from_label(
            self.builder_tuning_mode_display_vars[parameter_name].get()
        )
        self.builder_tuning_mode_vars[parameter_name].set(selected_kind)
        base_values = {
            "learning_rate": self.lr_var.get(),
            "batch_size": str(self.batch_size_var.get()),
            "weight_scale": self.weight_scale_var.get(),
            "epochs": str(self.epochs_var.get()),
            "candidate_epochs": str(self.candidate_epochs_var.get()),
            "start_temperature": self.start_temperature_var.get(),
            "cooling_schedule": self.cooling_schedule_var.get(),
            "cooling_parameter": self.cooling_parameter_var.get(),
        }
        if selected_kind == "fixed":
            self.builder_tuning_value_vars[parameter_name].set(base_values[parameter_name])
            return
        if selected_kind == "list":
            suggested = SUGGESTED_TUNING_VALUES.get(parameter_name)
            if suggested:
                self.builder_tuning_value_vars[parameter_name].set(", ".join(suggested))
            return
        if parameter_name == "cooling_schedule":
            self.builder_tuning_mode_vars[parameter_name].set("list")
            self.builder_tuning_mode_display_vars[parameter_name].set(
                self._search_value_kind_label("list")
            )
            self.builder_tuning_value_vars[parameter_name].set(
                ", ".join(SUPPORTED_COOLING_SCHEDULES)
            )
            return
        suggested = SUGGESTED_TUNING_VALUES.get(parameter_name)
        if suggested and len(suggested) >= 2:
            if parameter_name in {"batch_size", "epochs", "candidate_epochs"}:
                step = int(suggested[1]) - int(suggested[0])
                self.builder_tuning_value_vars[parameter_name].set(
                    f"{suggested[0]}:{suggested[-1]}:{step}"
                )
            else:
                step = float(suggested[1]) - float(suggested[0])
                self.builder_tuning_value_vars[parameter_name].set(
                    f"{suggested[0]}:{suggested[-1]}:{step:.3f}".rstrip("0").rstrip(".")
                )

    def _apply_builder_run_mode_visibility(self) -> None:
        """Blendet SA-spezifische Builder-Bereiche je nach Run-Modus ein oder aus."""

        if self.builder_sa_frame is None:
            return
        is_sa = self.builder_run_mode_var.get() == "simulated_annealing"
        if is_sa:
            if not self.builder_sa_frame.winfo_manager():
                self.builder_sa_frame.pack(fill=tk.X, pady=(0, 10), before=self.builder_tuning_frame)
        else:
            if self.builder_sa_frame.winfo_manager():
                self.builder_sa_frame.pack_forget()
        for parameter_name in ("candidate_epochs", "start_temperature", "cooling_schedule", "cooling_parameter"):
            row = self.builder_tuning_rows.get(parameter_name)
            if row is None:
                continue
            if is_sa:
                if not row.winfo_manager():
                    row.pack(fill=tk.X, pady=(0, 8))
            else:
                if row.winfo_manager():
                    row.pack_forget()

    def _apply_builder_search_type_visibility(self) -> None:
        """Passt Builder-Hinweise an den Search-Typ an."""

        search_type = self.builder_search_type_var.get()
        if search_type == "random_search":
            self.builder_status_var.set(
                (
                    "Random Search aktiv: Es werden reproduzierbar nur Teilmengen des diskreten Suchraums getestet."
                    if not self._is_english()
                    else "Random search active: only a reproducible subset of the discrete search space will be evaluated."
                )
            )
        elif self.builder_loaded_payload is None and self.builder_last_execution is None:
            self.builder_status_var.set(self.t("builder_status_idle"))

    def _update_builder_seed_preview(self) -> None:
        """Aktualisiert die sichtbare Vorschau der effektiven Seeds."""

        try:
            preview = ", ".join(str(seed) for seed in self._effective_builder_seeds())
            self.builder_seed_preview_var.set(preview)
        except Exception as exc:
            self.builder_seed_preview_var.set(str(exc))

    def _builder_layout_label(self, base_label: str, layout_spec: str) -> str:
        """Formatiert einen lesbaren Eintrag fuer die Builder-Layoutliste."""

        return f"{base_label}: {layout_spec}"

    def _update_builder_layout_options(self) -> None:
        """Sammelt aktuell verfuegbare Layouts fuer die Builder-Netzvorschau."""

        current_selection = self.builder_layout_choice_var.get()
        options: dict[str, dict[str, Any]] = {}
        setup_layout_spec = (
            self.builder_config_layout.to_compact_spec()
            if self.builder_config_layout is not None
            else self.layout_string_var.get().strip() or DEFAULT_LAYOUT
        )
        setup_label = (
            self._builder_layout_label("Setup-Layout", setup_layout_spec)
            if not self._is_english()
            else self._builder_layout_label("Setup Layout", setup_layout_spec)
        )
        options[setup_label] = {
            "source": "setup",
            "layout_spec": setup_layout_spec,
            "benchmark": self.benchmark_var.get(),
            "seed": int(self.builder_seed_start_var.get()),
            "model_state": None,
            "history": None,
            "metrics": None,
        }

        run_payload = self._selected_builder_run_payload()
        if run_payload is not None:
            run_definition = run_payload["run_definition"]
            extra = run_payload.get("extra", {})
            benchmark = run_definition["benchmark"]
            seed = int(run_definition["seed"])

            final_label = (
                self._builder_layout_label("Selektierter Run", run_payload["layout_spec"])
                if not self._is_english()
                else self._builder_layout_label("Selected Run", run_payload["layout_spec"])
            )
            options[final_label] = {
                "source": "selected_run",
                "layout_spec": run_payload["layout_spec"],
                "benchmark": benchmark,
                "seed": seed,
                "model_state": extra.get("model_state") or extra.get("current_model_state") or extra.get("best_model_state"),
                "history": run_payload.get("history"),
                "metrics": run_payload.get("metrics"),
            }

            if run_definition["run_mode"] == "simulated_annealing":
                for label_key, layout_key, model_key in (
                    ("Startlayout", "start_layout_spec", "start_model_state"),
                    ("Bestlayout", "best_layout_spec", "best_model_state"),
                    ("Endlayout", "end_layout_spec", "current_model_state"),
                ):
                    layout_spec = extra.get(layout_key)
                    if not layout_spec:
                        continue
                    label_prefix = label_key if not self._is_english() else {
                        "Startlayout": "Start Layout",
                        "Bestlayout": "Best Layout",
                        "Endlayout": "End Layout",
                    }[label_key]
                    label = self._builder_layout_label(label_prefix, str(layout_spec))
                    options[label] = {
                        "source": layout_key,
                        "layout_spec": str(layout_spec),
                        "benchmark": benchmark,
                        "seed": seed,
                        "model_state": extra.get(model_key),
                        "history": run_payload.get("history"),
                        "metrics": run_payload.get("metrics"),
                    }

        self.builder_layout_options = options
        if self._widget_exists(self.builder_layout_combo):
            values = list(options)
            self.builder_layout_combo.configure(values=values)
        if current_selection in options:
            self.builder_layout_choice_var.set(current_selection)
        elif options:
            self.builder_layout_choice_var.set(next(iter(options)))

    def _apply_builder_detail_level_visibility(self) -> None:
        """Blendet Builder-Expertendetails wie die Layoutliste je nach Detailstufe ein oder aus."""

        is_expert = self.mode_var.get() == "expert"
        for widget in self.builder_expert_only_widgets:
            if not self._widget_exists(widget):
                continue
            if is_expert:
                widget.grid()
            else:
                widget.grid_remove()

    def _on_builder_layout_option_selected(self, _event=None) -> None:
        """Aktualisiert die Builder-Netzvorschau anhand des gewaelten Layout-Eintrags."""

        self._sync_builder_preview_state()
        self._refresh_views()

    def _builder_training_result_from_run_payload(self, run_payload: dict[str, Any]) -> TrainingResult:
        """Rekonstruiert ein TrainingResult aus gespeicherten Builder-Run-Daten."""

        metrics = run_payload.get("metrics", {})
        return TrainingResult(
            history={
                key: [float(value) for value in values]
                for key, values in run_payload.get("history", {}).items()
            },
            test_metrics={
                "loss": float(metrics.get("test_loss", 0.0)),
                "accuracy": float(metrics.get("test_accuracy", 0.0)),
            },
        )

    def _sync_builder_preview_state(self) -> None:
        """Synchronisiert Dataset, Modell und Layout fuer die Builder-Netzvorschau."""

        selection = self.builder_layout_choice_var.get()
        if not selection or selection not in self.builder_layout_options:
            return
        option = self.builder_layout_options[selection]
        benchmark = str(option["benchmark"])
        seed = int(option["seed"])
        layout_spec = str(option["layout_spec"])
        source = str(option.get("source", "setup"))

        self.use_custom_sample_var.set(False)
        self.dataset = load_benchmark(DatasetConfig(name=benchmark, random_state=seed))
        model_state = option.get("model_state")
        if source == "setup":
            hidden_sizes = self._current_hidden_sizes()
            editable_layout = self.builder_config_layout or parse_layout_spec(layout_spec, hidden_sizes)
            self.current_layout = editable_layout
            self.model = ModularMLP(
                input_size=self.dataset.input_size,
                hidden_sizes=hidden_sizes,
                output_size=self.dataset.output_size,
                layout=editable_layout,
                weight_scale=float(self.weight_scale_var.get()),
                random_state=seed,
            )
        elif model_state:
            self.model = ModularMLP.from_state_dict(model_state)
            if self.model.hidden_sizes != self._current_hidden_sizes():
                self._set_hidden_sizes(self.model.hidden_sizes)
            self.current_layout = self.model.layout
        else:
            hidden_sizes = self._current_hidden_sizes()
            self.current_layout = parse_layout_spec(layout_spec, hidden_sizes)
            self.model = ModularMLP(
                input_size=self.dataset.input_size,
                hidden_sizes=hidden_sizes,
                output_size=self.dataset.output_size,
                layout=self.current_layout,
                weight_scale=float(self.weight_scale_var.get()),
                random_state=seed,
            )
        history = option.get("history")
        metrics = option.get("metrics")
        if history and metrics:
            fake_run_payload = {"history": history, "metrics": metrics}
            self.training_result = self._builder_training_result_from_run_payload(fake_run_payload)
            self.completed_epochs = len(self.training_result.history.get("train_loss", []))
        else:
            self.training_result = None
            self.completed_epochs = 0
        self._update_target_options()
        self._ensure_selected_hidden_is_valid()
        self._update_neuron_dropdowns()
        self._update_sample_spinbox_range()
        self._prepare_custom_samples_for_benchmark()

    def _build_content(self, parent: ttk.Frame) -> None:
        """Rechter Bereich mit Netzwerk, Input-Ansicht, Plotting und Lernhilfe."""

        if self._app_mode() == "experiment_builder":
            self._build_experiment_builder_content(parent)
            return

        summary_frame = ttk.LabelFrame(parent, text=self.t("summary_frame_title"), padding=10)
        summary_frame.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            summary_frame,
            textvariable=self.network_summary_var,
            justify=tk.LEFT,
            style="SectionValue.TLabel",
            wraplength=1040,
        ).pack(anchor="w")
        self._build_network_preview_frame(parent, row=1, pady=(10, 10))

        notebook_frame = ttk.LabelFrame(parent, text=self.t("notebook_frame_title"), padding=6)
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
        self.annealing_tab = ttk.Frame(self.notebook, style="Notebook.TFrame")
        self.help_tab = ttk.Frame(self.notebook, style="Notebook.TFrame")

        self.notebook.add(self.input_tab, text=self.t("tab_input"))
        self.notebook.add(self.plot_tab, text=self.t("tab_plot"))
        self.notebook.add(self.stepper_tab, text=self.t("tab_stepper"))
        self.notebook.add(self.detail_tab, text=self.t("tab_detail"))
        self.notebook.add(self.compare_tab, text=self.t("tab_compare"))
        if self._app_mode() == "playground":
            self.notebook.add(self.annealing_tab, text=self.t("tab_annealing"))
        self.notebook.add(self.help_tab, text=self.t("tab_help"))

        self._build_input_tab()
        self._build_plot_tab()
        self._build_stepper_tab()
        self._build_detail_tab()
        self._build_compare_tab()
        if self._app_mode() == "playground":
            self._build_annealing_tab()
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
            text=self.t("input_tab_hint"),
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
            text=self.t("digits_hint"),
            style="Hint.TLabel",
            justify=tk.LEFT,
        ).pack(anchor="w")
        button_row = ttk.Frame(self.digits_frame, style="White.TFrame")
        button_row.pack(anchor="w", pady=(8, 8))
        ttk.Button(button_row, text=self.t("copy_sample_button"), command=self._copy_current_digit_to_custom).pack(side=tk.LEFT)
        ttk.Button(button_row, text=self.t("clear_digit_button"), command=self._clear_custom_digit).pack(side=tk.LEFT, padx=(8, 0))

        self.digits_canvas = tk.Canvas(self.digits_frame, width=360, height=360, bg="#eef2f7", highlightthickness=0)
        self.digits_canvas.pack(fill=tk.BOTH, expand=False)
        self.digits_canvas.bind("<Button-1>", self._on_digits_canvas_clicked)

    def _build_generic_features_frame(self) -> None:
        """Feature-Tabelle fuer breast_cancer und wine."""

        ttk.Label(
            self.generic_features_frame,
            text=self.t("generic_features_hint"),
            style="Hint.TLabel",
            justify=tk.LEFT,
        ).pack(anchor="w")

        columns = ("rank", "feature", "raw", "scaled")
        self.feature_tree = ttk.Treeview(self.generic_features_frame, columns=columns, show="headings", height=18)
        for column, title, width in (
            ("rank", self.t("rank_header"), 50),
            ("feature", self.t("feature_header"), 220),
            ("raw", self.t("raw_header"), 110),
            ("scaled", self.t("scaled_header"), 110),
        ):
            self.feature_tree.heading(column, text=title)
            self.feature_tree.column(column, width=width, anchor="w")
        self.feature_tree.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

    def _build_test_activation_frame(self) -> None:
        """Kleines Rechenlabor mit manuell setzbaren Inputs."""

        ttk.Label(
            self.test_activation_frame,
            text=self.t("test_activation_hint"),
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
            text=self.t("test_activation_rule_hint"),
            style="Hint.TLabel",
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(12, 0))

        columns = ("feature", "raw", "scaled")
        self.test_activation_tree = ttk.Treeview(
            self.test_activation_frame, columns=columns, show="headings", height=5
        )
        for column, title, width in (
            ("feature", self.t("feature_header"), 180),
            ("raw", self.t("raw_header"), 100),
            ("scaled", self.t("scaled_header"), 100),
        ):
            self.test_activation_tree.heading(column, text=title)
            self.test_activation_tree.column(column, width=width, anchor="w")
        self.test_activation_tree.pack(fill=tk.X, expand=False, pady=(10, 0))

    def _build_plot_tab(self) -> None:
        """Tab mit eingebetteten Trainingsplots."""

        self.plot_tab.columnconfigure(0, weight=1)
        self.plot_tab.rowconfigure(0, weight=0)
        self.plot_tab.rowconfigure(1, weight=0)
        self.plot_tab.rowconfigure(2, weight=1)

        ttk.Label(
            self.plot_tab,
            text=self.t("plot_tab_hint"),
            style="Hint.TLabel",
            justify=tk.LEFT,
        ).grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))

        plot_info_row = ttk.Frame(self.plot_tab, style="White.TFrame")
        plot_info_row.grid(row=1, column=0, sticky="ew", padx=10, pady=(8, 0))
        plot_info_row.columnconfigure(0, weight=1)
        ttk.Label(
            plot_info_row,
            text=self.t("plot_info_hint"),
            style="Hint.TLabel",
            justify=tk.LEFT,
            wraplength=760,
        ).grid(row=0, column=0, sticky="w")
        plot_info_button_row = ttk.Frame(plot_info_row, style="White.TFrame")
        plot_info_button_row.grid(row=0, column=1, sticky="e", padx=(12, 0))
        self._create_info_action_button(
            plot_info_button_row,
            self.t("plot_overview_info_button"),
            "plot_overview",
        ).pack(side=tk.LEFT)
        self._create_info_action_button(
            plot_info_button_row,
            self.t("loss_plot_info_button"),
            "loss_plot",
        ).pack(side=tk.LEFT, padx=(8, 0))
        self._create_info_action_button(
            plot_info_button_row,
            self.t("accuracy_plot_info_button"),
            "accuracy_plot",
        ).pack(side=tk.LEFT, padx=(8, 0))
        self._create_info_action_button(
            plot_info_button_row,
            self.t("probability_plot_info_button"),
            "class_probabilities",
        ).pack(side=tk.LEFT, padx=(8, 0))
        self._create_info_action_button(
            plot_info_button_row,
            self.t("sample_activation_plot_info_button"),
            "sample_activations",
        ).pack(side=tk.LEFT, padx=(8, 0))

        figure_frame = ttk.Frame(self.plot_tab, style="White.TFrame")
        figure_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
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
            text=self.t("stepper_tab_hint"),
            style="Hint.TLabel",
            justify=tk.LEFT,
        ).grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))

        control_row = ttk.Frame(self.stepper_tab, style="White.TFrame")
        control_row.grid(row=1, column=0, sticky="ew", padx=10, pady=(8, 0))
        ttk.Button(control_row, text=self.t("step_back_button"), command=self._step_prev).pack(side=tk.LEFT)
        ttk.Button(control_row, text=self.t("step_forward_button"), command=self._step_next).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(control_row, text=self.t("step_reset_button"), command=self._step_reset).pack(side=tk.LEFT, padx=(8, 0))
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
            text=self.t("compare_tab_hint"),
            style="Hint.TLabel",
            justify=tk.LEFT,
        ).grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        self.compare_text = ScrolledText(self.compare_tab, wrap=tk.WORD)
        self.compare_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.compare_text.configure(state=tk.DISABLED, font=("Menlo", 11))

    def _build_annealing_tab(self) -> None:
        """Tab fuer Live-Ansicht und Verlauf von Simulated Annealing."""

        self.annealing_tab.columnconfigure(0, weight=1)
        self.annealing_tab.rowconfigure(2, weight=1)
        self.annealing_tab.rowconfigure(3, weight=1)
        ttk.Label(
            self.annealing_tab,
            text=self.t("annealing_tab_hint"),
            style="Hint.TLabel",
            justify=tk.LEFT,
        ).grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))

        summary_frame = ttk.Frame(self.annealing_tab, style="White.TFrame")
        summary_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(8, 0))
        summary_frame.columnconfigure(0, weight=1)
        self.annealing_status_label = ttk.Label(
            summary_frame,
            textvariable=self.playground_summary_var,
            justify=tk.LEFT,
            style="SectionValue.TLabel",
            wraplength=1100,
        )
        self.annealing_status_label.grid(row=0, column=0, sticky="w")

        self.annealing_text = ScrolledText(self.annealing_tab, wrap=tk.WORD)
        self.annealing_text.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        self.annealing_text.configure(state=tk.DISABLED, font=("Menlo", 11))

        figure_frame = ttk.Frame(self.annealing_tab, style="White.TFrame")
        figure_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=(0, 10))
        figure_frame.columnconfigure(0, weight=1)
        figure_frame.rowconfigure(0, weight=1)

        self.annealing_figure = Figure(figsize=(10, 6), dpi=100)
        self.annealing_axes = [
            self.annealing_figure.add_subplot(221),
            self.annealing_figure.add_subplot(222),
            self.annealing_figure.add_subplot(223),
            self.annealing_figure.add_subplot(224),
        ]
        self.annealing_canvas_widget = FigureCanvasTkAgg(self.annealing_figure, master=figure_frame)
        self.annealing_canvas_widget.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _build_detail_tab(self) -> None:
        """Tab fuer die detaillierte neuronale Einzelbetrachtung."""

        self.detail_tab.columnconfigure(0, weight=1)
        self.detail_tab.rowconfigure(1, weight=1)
        ttk.Label(
            self.detail_tab,
            text=self.t("detail_tab_hint"),
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
            spinbox = ttk.Spinbox(
                row,
                from_=1,
                to=256,
                textvariable=variable,
                width=8,
                command=self._on_hidden_size_controls_changed,
            )
            spinbox.pack(side=tk.LEFT)
            spinbox.bind("<Return>", self._on_hidden_size_controls_changed)
            spinbox.bind("<FocusOut>", self._on_hidden_size_controls_changed)
            self.hidden_layer_size_labels.append(label)
            self.hidden_size_spinboxes.append(spinbox)

    def _editable_layout(self) -> ActivationLayout | None:
        """Liefert das Layout, das durch die linken Layout-Controls bearbeitet wird."""

        if self._app_mode() == "experiment_builder":
            return self.builder_config_layout
        return self.current_layout

    def _layer_summary_text(self, layout: ActivationLayout) -> str:
        """Erzeugt die kompakte Layerzusammenfassung fuer ein Layout."""

        return "\n".join(
            (
                f"L{layer_index + 1}: {len(layer)} Neuronen, Start-Aktivierung {layer[0]}"
                if not self._is_english()
                else f"L{layer_index + 1}: {len(layer)} neurons, starting activation {layer[0]}"
            )
            for layer_index, layer in enumerate(layout.layers)
        )

    def _sync_layout_editor_state(
        self,
        layout: ActivationLayout,
        *,
        update_preview: bool,
    ) -> None:
        """Synchronisiert String, Editor-Variablen und optional die Builder-Vorschau."""

        if self._app_mode() == "experiment_builder":
            self.builder_config_layout = layout
        else:
            self.current_layout = layout
            if self.model is not None:
                self.model.layout = layout

        self.layout_string_var.set(layout.to_compact_spec())
        self.layer_summary_var.set(self._layer_summary_text(layout))
        for layer_index, layer in enumerate(layout.layers):
            if layer_index < len(self.layer_fill_vars):
                self.layer_fill_vars[layer_index].set(layer[0])
        self._ensure_selected_hidden_is_valid()
        self._update_neuron_dropdowns()
        self._rebuild_layer_fill_controls()

        if self._app_mode() == "experiment_builder":
            self._update_builder_layout_options()
            if update_preview and self._builder_preview_tracks_setup_layout():
                self._sync_builder_preview_state()
                self._refresh_views()
            elif update_preview:
                self._refresh_views()
        elif self._app_mode() == "playground":
            self._initialize_playground_state()
        else:
            self._refresh_views()

    def _builder_preview_tracks_setup_layout(self) -> bool:
        """Prueft, ob die Builder-Vorschau aktuell auf dem Setup-Layout steht."""

        selection = self.builder_layout_choice_var.get()
        if not selection or selection not in self.builder_layout_options:
            return True
        return self.builder_layout_options[selection].get("source") == "setup"

    def _on_hidden_size_controls_changed(self, _event=None) -> None:
        """Aktualisiert Layout und Vorschau nach Aenderungen an Hidden-Groessen."""

        try:
            hidden_sizes = self._current_hidden_sizes()
            self.current_layout = None
            self.builder_config_layout = None
            resolved_layout = self._resolve_layout_for_hidden_sizes(hidden_sizes)
            self.layout_string_var.set(resolved_layout.to_compact_spec())
            self._reload_experiment()
        except (tk.TclError, ValueError):
            return

    def _rebuild_layer_fill_controls_for(
        self,
        frame: ttk.Frame,
        combo_store: list[ttk.Combobox],
    ) -> None:
        """Baut die layerweisen Aktivierungs-Controls fuer einen Ziel-Frame neu auf."""

        for child in frame.winfo_children():
            child.destroy()
        combo_store.clear()

        for layer_index, variable in enumerate(self.layer_fill_vars):
            row = ttk.Frame(frame, style="White.TFrame")
            row.pack(fill=tk.X, pady=(0, 6))
            ttk.Label(
                row,
                text=(
                    f"Layer {layer_index + 1} auf"
                    if not self._is_english()
                    else f"Set layer {layer_index + 1} to"
                ),
                width=16,
            ).pack(side=tk.LEFT)
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
                text="setzen" if not self._is_english() else "set",
                command=lambda li=layer_index: self._fill_layer(li),
            ).pack(side=tk.LEFT, padx=(8, 0))
            combo_store.append(combo)

    def _rebuild_layer_fill_controls(self) -> None:
        """Baut die layerweisen Aktivierungs-Controls passend zur aktuellen Layerzahl neu auf."""

        if self.layer_fill_controls_frame is not None:
            self._rebuild_layer_fill_controls_for(
                self.layer_fill_controls_frame,
                self.layer_fill_combos,
            )
        if self.builder_layer_fill_controls_frame is not None:
            self._rebuild_layer_fill_controls_for(
                self.builder_layer_fill_controls_frame,
                self.builder_layer_fill_combos,
            )

    def _apply_mode_visibility(self) -> None:
        """Blendet Expertenfunktionen je nach Modus ein oder aus."""

        if self._app_mode() == "experiment_builder":
            return

        is_expert = self.mode_var.get() == "expert"
        is_playground = self._app_mode() == "playground"
        for widget in self.expert_only_widgets:
            if is_expert:
                widget.grid()
            else:
                widget.grid_remove()

        if hasattr(self, "notebook"):
            try:
                self.notebook.tab(self.stepper_tab, state="normal")
                self.notebook.tab(
                    self.compare_tab,
                    state="normal" if (is_expert and not is_playground) else "hidden",
                )
                if hasattr(self, "annealing_tab"):
                    self.notebook.tab(
                        self.annealing_tab,
                        state="normal" if is_playground else "hidden",
                    )
            except tk.TclError:
                pass

    def _on_mode_changed(self) -> None:
        """Aktualisiert die GUI zwischen Einsteiger- und Expertenmodus."""

        self._sync_mode_display_var()
        if self._app_mode() == "experiment_builder":
            self._apply_builder_detail_level_visibility()
        self._apply_mode_visibility()
        self._refresh_views()

    def _add_hidden_layer(self) -> None:
        """Fuegt einen weiteren Hidden-Layer fuer den Expertenmodus hinzu."""

        if len(self.hidden_size_vars) >= MAX_HIDDEN_LAYERS:
            messagebox.showinfo(
                self.t("max_layers_title"),
                self.t("max_layers_message", max_layers=MAX_HIDDEN_LAYERS),
            )
            return
        last_size = self.hidden_size_vars[-1].get() if self.hidden_size_vars else 8
        last_activation = self.layer_fill_vars[-1].get() if self.layer_fill_vars else "relu"
        self.hidden_size_vars.append(tk.IntVar(value=last_size))
        self.layer_fill_vars.append(tk.StringVar(value=last_activation))
        editable_layout = self._editable_layout()
        if editable_layout is not None:
            updated_layout = editable_layout.add_layer(last_size, activation_name=last_activation)
            self.layout_string_var.set(updated_layout.to_compact_spec())
        self._rebuild_hidden_size_controls()
        self._rebuild_layer_fill_controls()
        self._reload_experiment()

    def _remove_hidden_layer(self) -> None:
        """Entfernt den letzten Hidden-Layer, sofern mindestens einer bleibt."""

        if len(self.hidden_size_vars) <= MIN_HIDDEN_LAYERS:
            messagebox.showinfo(
                self.t("min_layers_title"),
                self.t("min_layers_message"),
            )
            return
        self.hidden_size_vars.pop()
        self.layer_fill_vars.pop()
        editable_layout = self._editable_layout()
        if editable_layout is not None:
            updated_layout = editable_layout.remove_layer(len(editable_layout.layers) - 1)
            self.layout_string_var.set(updated_layout.to_compact_spec())
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
        self.baseline_benchmark_name = self.dataset.name if self.dataset is not None else None
        self.compare_summary_var.set(
            (
                f"Baseline gespeichert: Layout {self.baseline_layout.to_compact_spec()} | "
                f"Epochen {self.baseline_completed_epochs}"
            )
            if not self._is_english()
            else (
                f"Baseline stored: layout {self.baseline_layout.to_compact_spec()} | "
                f"epochs {self.baseline_completed_epochs}"
            )
        )
        self._update_compare_text()

    def _build_help_tab(self) -> None:
        """Tab mit Schritt-fuer-Schritt-Hinweisen."""

        self.help_tab.columnconfigure(0, weight=1)
        self.help_tab.rowconfigure(0, weight=1)
        self.help_text = ScrolledText(self.help_tab, wrap=tk.WORD)
        self.help_text.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.help_text.configure(state=tk.DISABLED, font=("Menlo", 11))

    def _open_program_guide(self) -> None:
        """Oeffnet eine einfache A-Z-Anleitung als separates Fenster."""

        if self.guide_window is not None and self.guide_window.winfo_exists():
            self._update_guide_window()
            self.guide_window.lift()
            self.guide_window.focus_force()
            return

        self.guide_window = tk.Toplevel(self.root)
        self.guide_window.title(self.t("guide_window_title"))
        self.guide_window.geometry("900x760")
        self.guide_window.configure(bg=BACKGROUND_COLOR)
        self.guide_window.protocol("WM_DELETE_WINDOW", self._close_guide_window)

        frame = ttk.Frame(self.guide_window, padding=12, style="White.TFrame")
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(
            frame,
            text=self.t("guide_intro_title"),
            style="Headline.TLabel",
        ).grid(row=0, column=0, sticky="w")

        self.guide_text_widget = ScrolledText(frame, wrap=tk.WORD)
        self.guide_text_widget.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.guide_text_widget.configure(state=tk.DISABLED, font=("Menlo", 11))
        self._update_guide_window()

    def _close_guide_window(self) -> None:
        """Schliesst das Guide-Fenster kontrolliert."""

        if self.guide_window is not None and self.guide_window.winfo_exists():
            self.guide_window.destroy()
        self.guide_window = None

    def _update_guide_window(self) -> None:
        """Aktualisiert Titel und Inhalt des Guide-Fensters."""

        if self.guide_window is None or not self.guide_window.winfo_exists():
            return
        self.guide_window.title(self.t("guide_window_title"))
        self.guide_text_widget.configure(state=tk.NORMAL)
        self.guide_text_widget.delete("1.0", tk.END)
        self.guide_text_widget.insert("1.0", self._program_guide_text())
        self.guide_text_widget.configure(state=tk.DISABLED)

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
        self.analysis_target_var.set(self._target_auto_label())
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
            if self._app_mode() == "experiment_builder":
                self.builder_config_layout = self.current_layout
            self.layout_string_var.set(self.current_layout.to_compact_spec())
            self._set_hidden_sizes(hidden_sizes)
            for layer_index, layer in enumerate(self.current_layout.layers):
                self.layer_fill_vars[layer_index].set(layer[0])
            self.layer_summary_var.set(self._layer_summary_text(self.current_layout))

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
            if self._app_mode() == "playground":
                self._initialize_playground_state(refresh_views=False)
            self._refresh_views()
        except Exception as exc:
            messagebox.showerror(self.t("load_error_title"), str(exc))

    def _resolve_layout_for_hidden_sizes(self, hidden_sizes: tuple[int, ...]) -> ActivationLayout:
        """Versucht, das aktuelle Layout auf neue Hidden-Sizes abzubilden."""

        current_layout_string = self.layout_string_var.get().strip() or DEFAULT_LAYOUT
        try:
            return parse_layout_spec(current_layout_string, hidden_sizes)
        except ValueError:
            return parse_layout_spec(DEFAULT_LAYOUT, hidden_sizes)

    def _on_layout_string_edited(self, _event=None) -> None:
        """Parst einen manuell geaenderten Layout-String und synchronisiert den Editor."""

        try:
            hidden_sizes = self._current_hidden_sizes()
            layout = parse_layout_spec(self.layout_string_var.get().strip() or DEFAULT_LAYOUT, hidden_sizes)
        except ValueError as exc:
            messagebox.showerror(self.t("load_error_title"), str(exc))
            editable_layout = self._editable_layout()
            if editable_layout is not None:
                self.layout_string_var.set(editable_layout.to_compact_spec())
            return

        self._sync_layout_editor_state(layout, update_preview=True)

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
        if not self._widget_exists(getattr(self, "analysis_target_combo", None)):
            return
        current_token = self._analysis_target_token()
        values = [self._target_auto_label(), self._target_none_label(), *self.dataset.target_names]
        self.analysis_target_combo.configure(values=values)
        self._set_analysis_target_from_token(current_token)
        if self.analysis_target_var.get() not in values:
            self.analysis_target_var.set(self._target_auto_label())

    def _ensure_selected_hidden_is_valid(self) -> None:
        """Sorgt dafuer, dass das selektierte Hidden-Neuron existiert."""

        layout = self._editable_layout()
        if layout is None:
            self.selected_hidden = (0, 0)
            return
        layer_index, neuron_index = self.selected_hidden
        layer_index = min(max(layer_index, 0), len(layout.layers) - 1)
        neuron_index = min(max(neuron_index, 0), len(layout.layers[layer_index]) - 1)
        self.selected_hidden = (layer_index, neuron_index)

    def _update_neuron_dropdowns(self) -> None:
        """Aktualisiert die moeglichen Neuron-Indizes im Dropdown."""

        layout = self._editable_layout()
        if layout is None:
            return
        layer_values = tuple(f"L{layer_index + 1}" for layer_index in range(len(layout.layers)))
        combo_widgets = [
            widget
            for widget in (
                getattr(self, "neuron_layer_combo", None),
                getattr(self, "builder_neuron_layer_combo", None),
            )
            if self._widget_exists(widget)
        ]
        if not combo_widgets:
            return
        for combo in combo_widgets:
            combo.configure(values=layer_values)
        if self.neuron_layer_var.get() not in layer_values:
            self.neuron_layer_var.set(layer_values[0])
        layer_index = int(self.neuron_layer_var.get().replace("L", "")) - 1
        values = [str(index) for index in range(len(layout.layers[layer_index]))]
        for combo in (
            getattr(self, "neuron_index_combo", None),
            getattr(self, "builder_neuron_index_combo", None),
        ):
            if self._widget_exists(combo):
                combo.configure(values=values)

        selected_layer, selected_neuron = self.selected_hidden
        if selected_layer == layer_index:
            self.neuron_index_var.set(str(selected_neuron))
            self.neuron_activation_var.set(layout.layers[layer_index][selected_neuron])
        elif values:
            self.neuron_index_var.set(values[0])

    def _update_sample_spinbox_range(self) -> None:
        """Passt den Sample-Bereich an den aktuellen Split an."""

        if not self._widget_exists(getattr(self, "sample_spinbox", None)):
            return
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
            source_label = "Eigenes digits-Sample" if not self._is_english() else "Custom digits sample"
        elif benchmark == "test_activation" and is_custom:
            raw_sample = np.asarray([variable.get() for variable in self.test_input_vars], dtype=np.float64)
            actual_target_index = self._infer_test_activation_target(raw_sample)
            source_label = (
                "Eigenes test_activation-Sample"
                if not self._is_english()
                else "Custom test_activation sample"
            )
        else:
            X_scaled, y_split = self._get_current_split_arrays()
            X_raw, _ = self._get_current_raw_split_arrays()
            sample_index = min(max(self.sample_index_var.get(), 0), len(X_scaled) - 1)
            self.sample_index_var.set(sample_index)
            raw_sample = np.asarray(X_raw[sample_index], dtype=np.float64).copy()
            actual_target_index = int(y_split[sample_index])
            source_label = (
                f"Datensatz-Sample aus {split_name}"
                if not self._is_english()
                else f"Dataset sample from {split_name}"
            )

        if self.dataset.scaler is not None:
            scaled_sample = self.dataset.scaler.transform(raw_sample.reshape(1, -1))[0]
        else:
            scaled_sample = raw_sample.copy()

        actual_target_name = (
            self.dataset.target_names[actual_target_index]
            if actual_target_index is not None
            else ("unbekannt" if not self._is_english() else "unknown")
        )
        effective_target_index = self._resolve_effective_target_index(actual_target_index)
        effective_target_name = (
            self.dataset.target_names[effective_target_index]
            if effective_target_index is not None
            else ("kein Ziel gesetzt" if not self._is_english() else "no target set")
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
        if selected == self._target_none_label():
            return None
        if selected == self._target_auto_label():
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

        layout = self._editable_layout()
        if layout is None:
            return
        activation_name = self.layer_fill_vars[layer_index].get()
        updated_layout = layout.replace_layer(layer_index, activation_name)
        self._sync_layout_editor_state(updated_layout, update_preview=True)

    def _apply_neuron_setting(self) -> None:
        """Setzt die Aktivierung eines einzelnen Hidden-Neurons."""

        layout = self._editable_layout()
        if layout is None:
            return
        layer_index = int(self.neuron_layer_var.get().replace("L", "")) - 1
        neuron_index = int(self.neuron_index_var.get())
        activation_name = self.neuron_activation_var.get()
        self.selected_hidden = (layer_index, neuron_index)
        updated_layout = layout.replace_neuron(layer_index, neuron_index, activation_name)
        self._sync_layout_editor_state(updated_layout, update_preview=True)

    def _cycle_selected_neuron(self) -> None:
        """Schaltet das aktuell gewaehlte Neuron zur naechsten Aktivierung weiter."""

        layout = self._editable_layout()
        if layout is None:
            return
        layer_index = int(self.neuron_layer_var.get().replace("L", "")) - 1
        neuron_index = int(self.neuron_index_var.get())
        self.selected_hidden = (layer_index, neuron_index)
        updated_layout = layout.cycle_neuron(layer_index, neuron_index)
        self._sync_layout_editor_state(updated_layout, update_preview=True)
        self.neuron_activation_var.set(updated_layout.layers[layer_index][neuron_index])

    def _select_from_controls(self) -> None:
        """Uebernimmt die aktuelle Dropdown-Auswahl als selektiertes Neuron."""

        layout = self._editable_layout()
        if layout is None:
            return
        layer_index = int(self.neuron_layer_var.get().replace("L", "")) - 1
        neuron_index = int(self.neuron_index_var.get())
        self.selected_hidden = (layer_index, neuron_index)
        self.neuron_activation_var.set(layout.layers[layer_index][neuron_index])
        self._refresh_views()

    def _on_benchmark_changed(self, _event=None) -> None:
        """Beim Benchmark-Wechsel zunaechst sinnvolle Default-Hidden-Sizes setzen."""

        self._apply_benchmark_defaults()
        if self.benchmark_var.get() == "test_activation":
            self.use_custom_sample_var.set(True)
        if self._app_mode() == "experiment_builder":
            self._load_experiment(reinitialize_model=True)
            self._update_builder_layout_options()

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

    def _build_objective_config(self) -> ObjectiveConfig:
        """Erzeugt die aktuelle Objective-Konfiguration fuer den Playground Mode."""

        return ObjectiveConfig(
            objective_name=self.objective_var.get(),
            candidate_epochs=int(self.candidate_epochs_var.get()),
            learning_rate=float(self.lr_var.get()),
            batch_size=int(self.batch_size_var.get()),
            weight_scale=float(self.weight_scale_var.get()),
            random_state=int(self.seed_var.get()),
            shuffle=bool(self.playground_shuffle_var.get()),
        )

    def _build_annealing_config(self) -> AnnealingConfig:
        """Erzeugt die aktuelle Annealing-Konfiguration aus der GUI."""

        neighborhood_operations = tuple(
            operation
            for operation, variable in self.playground_neighbor_vars.items()
            if variable.get()
        )
        return AnnealingConfig(
            start_temperature=float(self.start_temperature_var.get()),
            cooling_schedule=self.cooling_schedule_var.get(),
            cooling_parameter=float(self.cooling_parameter_var.get()),
            iterations_per_temperature=int(self.iterations_per_temperature_var.get()),
            max_steps=int(self.max_steps_var.get()),
            min_temperature=float(self.min_temperature_var.get()),
            neighborhood_operations=neighborhood_operations,
        )

    def _initialize_playground_state(self, refresh_views: bool = True) -> None:
        """Initialisiert oder resettet den Simulated-Annealing-Lauf."""

        if self.dataset is None or self.current_layout is None:
            return
        try:
            evaluator = LayoutObjectiveEvaluator(self.dataset, self._build_objective_config())
            runner = AnnealingRunner(
                evaluator=evaluator,
                config=self._build_annealing_config(),
                random_state=int(self.seed_var.get()),
            )
            state = runner.initialize(self.current_layout)
            self.annealing_runner = runner
            self.annealing_state = state
            self.annealing_last_step = None
            self.annealing_last_candidate = None
            self.model = state.current_evaluation.trained_model
            self.current_layout = state.current_evaluation.layout
            self.training_result = state.current_evaluation.training_result
            self.completed_epochs = int(self.candidate_epochs_var.get())
            self.playground_decision_var.set(
                (
                    "Startzustand bewertet. Jetzt kannst du einzelne SA-Schritte ausfuehren oder den kompletten Lauf starten."
                    if not self._is_english()
                    else "Start state evaluated. You can now execute individual SA steps or start the full run."
                )
            )
            if refresh_views:
                self._refresh_views()
        except Exception as exc:
            messagebox.showerror(self.t("annealing_error_title"), str(exc))

    def _run_annealing_steps(self, step_count: int) -> None:
        """Fuehrt mehrere SA-Schritte aus und aktualisiert die GUI danach."""

        if self._app_mode() != "playground":
            return
        try:
            if self.annealing_runner is None or self.annealing_state is None:
                self._initialize_playground_state(refresh_views=False)
            if self.annealing_runner is None or self.annealing_state is None:
                return
            executed_steps = self.annealing_runner.run_steps(step_count)
            self.annealing_state = self.annealing_runner.state
            if executed_steps:
                self.annealing_last_step = executed_steps[-1]
                self.annealing_last_candidate = executed_steps[-1].candidate_evaluation
            if self.annealing_state is not None:
                self.model = self.annealing_state.current_evaluation.trained_model
                self.current_layout = self.annealing_state.current_evaluation.layout
                self.training_result = self.annealing_state.current_evaluation.training_result
                self.completed_epochs = int(self.candidate_epochs_var.get())
            self._refresh_views()
        except Exception as exc:
            messagebox.showerror(self.t("annealing_error_title"), str(exc))

    def _run_annealing_to_completion(self) -> None:
        """Fuehrt den SA-Lauf bis zum Stopkriterium aus."""

        if self._app_mode() != "playground":
            return
        try:
            if self.annealing_runner is None or self.annealing_state is None:
                self._initialize_playground_state(refresh_views=False)
            if self.annealing_runner is None or self.annealing_state is None:
                return
            executed_steps = self.annealing_runner.run_until_complete()
            self.annealing_state = self.annealing_runner.state
            if executed_steps:
                self.annealing_last_step = executed_steps[-1]
                self.annealing_last_candidate = executed_steps[-1].candidate_evaluation
            if self.annealing_state is not None:
                self.model = self.annealing_state.current_evaluation.trained_model
                self.current_layout = self.annealing_state.current_evaluation.layout
                self.training_result = self.annealing_state.current_evaluation.training_result
                self.completed_epochs = int(self.candidate_epochs_var.get())
            self._refresh_views()
        except Exception as exc:
            messagebox.showerror(self.t("annealing_error_title"), str(exc))

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
            messagebox.showerror(self.t("train_error_title"), str(exc))

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

    def _effective_builder_seeds(self) -> tuple[int, ...]:
        """Liefert die aktuelle Seed-Liste aus dem Builder."""

        if self.builder_seed_strategy_var.get() == "explicit_list":
            tokens = [
                token.strip()
                for token in self.builder_seed_list_var.get().replace(";", ",").split(",")
                if token.strip()
            ]
            if not tokens:
                raise ValueError(
                    "Bitte mindestens einen Seed angeben."
                    if not self._is_english()
                    else "Please provide at least one seed."
                )
            return tuple(int(token) for token in tokens)

        count = int(self.builder_seed_count_var.get())
        start = int(self.builder_seed_start_var.get())
        if count <= 0:
            raise ValueError(
                "Die Anzahl der Seeds muss positiv sein."
                if not self._is_english()
                else "Seed count must be positive."
            )
        return tuple(start + offset for offset in range(count))

    def _builder_search_value_type(self, parameter_name: str) -> str:
        """Liefert den Datentyp eines Builder-Suchparameters."""

        if parameter_name in {"batch_size", "epochs", "candidate_epochs"}:
            return "int"
        if parameter_name == "cooling_schedule":
            return "str"
        return "float"

    def _builder_current_parameter_value(self, parameter_name: str) -> str:
        """Liefert den aktuell sichtbaren Basiswert fuer einen Builder-Parameter."""

        value_map = {
            "learning_rate": self.lr_var.get(),
            "batch_size": str(self.batch_size_var.get()),
            "weight_scale": self.weight_scale_var.get(),
            "epochs": str(self.epochs_var.get()),
            "candidate_epochs": str(self.candidate_epochs_var.get()),
            "start_temperature": self.start_temperature_var.get(),
            "cooling_schedule": self.cooling_schedule_var.get(),
            "cooling_parameter": self.cooling_parameter_var.get(),
        }
        return value_map[parameter_name]

    def _normalized_experiment_id(self, raw_name: str, benchmark: str, run_mode: str) -> str:
        """Leitet einen dateisicheren Experiment-Identifier aus einem Builder-Namen ab."""

        normalized = "".join(
            character.lower() if character.isalnum() else "_"
            for character in raw_name
        ).strip("_")
        return normalized or f"{benchmark}_{run_mode}"

    def _parse_builder_search_value_definition(self, parameter_name: str) -> SearchValueDefinition:
        """Parst eine Builder-Zeile zu einer diskreten SearchValueDefinition."""

        kind = self.builder_tuning_mode_vars[parameter_name].get()
        value_type = self._builder_search_value_type(parameter_name)
        raw_text = self.builder_tuning_value_vars[parameter_name].get().strip()
        default_value = self._builder_current_parameter_value(parameter_name)

        if kind == "fixed":
            value_text = raw_text or default_value
            return SearchValueDefinition(
                parameter_name=parameter_name,
                kind="fixed",
                value_type=value_type,
                fixed_value=value_text,
            )

        if kind == "list":
            parts = [
                part.strip()
                for part in raw_text.replace(";", ",").split(",")
                if part.strip()
            ]
            if not parts:
                parts = [default_value]
            return SearchValueDefinition(
                parameter_name=parameter_name,
                kind="list",
                value_type=value_type,
                values=tuple(parts),
            )

        if value_type == "str":
            raise ValueError(
                (
                    f"{parameter_name}: Bereich ist fuer Textwerte nicht erlaubt."
                    if not self._is_english()
                    else f"{parameter_name}: range is not allowed for string values."
                )
            )
        range_parts = [part.strip() for part in raw_text.split(":") if part.strip()]
        if len(range_parts) != 3:
            raise ValueError(
                (
                    f"{parameter_name}: Bereich erwartet start:stop:step."
                    if not self._is_english()
                    else f"{parameter_name}: range expects start:stop:step."
                )
            )
        return SearchValueDefinition(
            parameter_name=parameter_name,
            kind="range",
            value_type=value_type,
            range_start=float(range_parts[0]),
            range_stop=float(range_parts[1]),
            range_step=float(range_parts[2]),
        )

    def _build_builder_search_space(self, run_mode: str) -> SearchSpaceDefinition:
        """Erzeugt den diskreten Suchraum aus dem aktuellen Builder-Zustand."""

        search_type = self.builder_search_type_var.get()
        if search_type == "none":
            return SearchSpaceDefinition(search_type="none")

        if run_mode == "manual_training":
            relevant_parameters = ("learning_rate", "batch_size", "weight_scale", "epochs")
        else:
            relevant_parameters = (
                "learning_rate",
                "batch_size",
                "weight_scale",
                "candidate_epochs",
                "start_temperature",
                "cooling_schedule",
                "cooling_parameter",
            )

        value_definitions = tuple(
            self._parse_builder_search_value_definition(parameter_name)
            for parameter_name in relevant_parameters
        )
        return SearchSpaceDefinition(
            search_type=search_type,
            value_definitions=value_definitions,
            random_samples=int(self.builder_random_samples_var.get()),
            random_state=int(self.builder_random_seed_var.get()),
        )

    def _build_experiment_definition(self) -> ExperimentDefinition:
        """Erzeugt eine Builder-Definition aus den GUI-Werten."""

        benchmark = self.benchmark_var.get()
        hidden_sizes = self._current_hidden_sizes()
        if self._app_mode() == "experiment_builder" and self.builder_config_layout is not None:
            layout = self.builder_config_layout
        else:
            layout = parse_layout_spec(self.layout_string_var.get(), hidden_sizes)
        self.layout_string_var.set(layout.to_compact_spec())
        run_mode = self.builder_run_mode_var.get()

        experiment_name = self.builder_experiment_name_var.get().strip()
        if not experiment_name:
            experiment_name = f"{benchmark}_{run_mode}"
            self.builder_experiment_name_var.set(experiment_name)
        experiment_id = self._normalized_experiment_id(experiment_name, benchmark, run_mode)

        return ExperimentDefinition(
            experiment_id=experiment_id,
            benchmark=benchmark,
            hidden_sizes=hidden_sizes,
            layout_spec=layout.to_compact_spec(),
            run_mode=run_mode,
            seeds=self._effective_builder_seeds(),
            primary_metric=self.builder_primary_metric_var.get(),
            language=self._language(),
            save_json=bool(self.builder_save_json_var.get()),
            output_dir=self.builder_output_dir_var.get().strip() or str(OUTPUT_DIR / DEFAULT_EXPERIMENT_OUTPUT_SUBDIR),
            shuffle=bool(self.playground_shuffle_var.get()),
            learning_rate=float(self.lr_var.get()),
            batch_size=int(self.batch_size_var.get()),
            weight_scale=float(self.weight_scale_var.get()),
            epochs=int(self.epochs_var.get()),
            objective_name=self.objective_var.get(),
            candidate_epochs=int(self.candidate_epochs_var.get()),
            neighborhood_operations=tuple(
                operation
                for operation, variable in self.playground_neighbor_vars.items()
                if variable.get()
            ),
            start_temperature=float(self.start_temperature_var.get()),
            cooling_schedule=self.cooling_schedule_var.get(),
            cooling_parameter=float(self.cooling_parameter_var.get()),
            iterations_per_temperature=int(self.iterations_per_temperature_var.get()),
            max_steps=int(self.max_steps_var.get()),
            min_temperature=float(self.min_temperature_var.get()),
            search_space=self._build_builder_search_space(run_mode),
        )

    def _validate_builder_experiment(self) -> None:
        """Validiert den aktuellen Builder-Zustand und zeigt die geplante Run-Anzahl."""

        try:
            definition = self._build_experiment_definition()
            expanded_configs = expand_search_space(definition.search_space)
            run_count = len(expanded_configs) * len(definition.seeds)
            self.builder_status_var.set(self.t("builder_validation_ok", run_count=run_count))
            messagebox.showinfo(
                self.t("info_dialog_title"),
                self.t("builder_validation_ok", run_count=run_count),
            )
        except Exception as exc:
            self.builder_status_var.set(str(exc))
            messagebox.showerror(self.t("builder_validation_error_title"), str(exc))

    def _run_builder_experiment(self) -> None:
        """Fuehrt das aktuelle Builder-Experiment aus und uebernimmt die Resultate."""

        try:
            definition = self._build_experiment_definition()
            running_text = (
                "Experiment wird ausgefuehrt ..."
                if not self._is_english()
                else "Executing experiment ..."
            )
            self.builder_status_var.set(running_text)
            self.root.update_idletasks()
            result = self.builder_runner.run_experiment(definition)
            self.builder_last_execution = result

            if result["output_path"] is not None:
                self.builder_load_path_var.set(str(result["output_path"]))
                self.builder_loaded_payload = load_experiment_results(result["output_path"])
            else:
                run_results = result["run_results"]
                self.builder_loaded_payload = {
                    "manifest": {
                        "experiment_id": definition.experiment_id,
                        "benchmark": definition.benchmark,
                        "run_mode": definition.run_mode,
                        "search_type": definition.search_space.search_type,
                        "primary_metric": definition.primary_metric,
                        "created_at": definition.created_at,
                        "definition": definition.to_dict(),
                        "summary_file": "summary.json",
                        "run_files": [
                            f"runs/{run_result.run_definition.run_id}.json"
                            for run_result in run_results
                        ],
                    },
                    "summary": result["summary"].to_dict(),
                    "runs": [run_result.to_dict() for run_result in run_results],
                    "experiment_dir": str(definition.output_path),
                }

            active_runs = self.builder_loaded_payload.get("runs", []) if self.builder_loaded_payload else []
            if active_runs:
                self.builder_selected_run_id = active_runs[0]["run_definition"]["run_id"]
                self.builder_selected_config_id = active_runs[0]["run_definition"]["config_id"]
            finished_text = (
                f"Experiment abgeschlossen: {len(active_runs)} Runs."
                if not self._is_english()
                else f"Experiment finished: {len(active_runs)} runs."
            )
            if result["output_path"] is not None:
                finished_text += (
                    f" Gespeichert unter {result['output_path']}."
                    if not self._is_english()
                    else f" Stored in {result['output_path']}."
                )
            self.builder_status_var.set(finished_text)
            self._refresh_views()
        except Exception as exc:
            self.builder_status_var.set(str(exc))
            messagebox.showerror(self.t("builder_run_error_title"), str(exc))

    def _load_builder_results(self) -> None:
        """Laedt gespeicherte Experiment-Builder-Ergebnisse von Platte."""

        try:
            load_path = self.builder_load_path_var.get().strip()
            if not load_path:
                experiment_name = self.builder_experiment_name_var.get().strip()
                if not experiment_name:
                    raise ValueError(
                        "Bitte einen Pfad oder einen Experiment-Namen angeben."
                        if not self._is_english()
                        else "Please provide a path or an experiment name."
                    )
                load_path = str(
                    Path(self.builder_output_dir_var.get().strip() or str(OUTPUT_DIR / DEFAULT_EXPERIMENT_OUTPUT_SUBDIR))
                    / self._normalized_experiment_id(
                        experiment_name,
                        self.benchmark_var.get(),
                        self.builder_run_mode_var.get(),
                    )
                )
                self.builder_load_path_var.set(load_path)

            self.builder_loaded_payload = load_experiment_results(load_path)
            active_runs = self.builder_loaded_payload.get("runs", [])
            if active_runs:
                self.builder_selected_run_id = active_runs[0]["run_definition"]["run_id"]
                self.builder_selected_config_id = active_runs[0]["run_definition"]["config_id"]
            self.builder_status_var.set(
                (
                    f"Ergebnisse geladen aus {self.builder_loaded_payload['experiment_dir']}."
                    if not self._is_english()
                    else f"Loaded results from {self.builder_loaded_payload['experiment_dir']}."
                )
            )
            self._refresh_views()
        except Exception as exc:
            self.builder_status_var.set(str(exc))
            messagebox.showerror(self.t("builder_load_error_title"), str(exc))

    def _active_builder_payload(self) -> dict[str, Any] | None:
        """Liefert das aktive Builder-Ergebnis fuer Analyse und Plotting."""

        return self.builder_loaded_payload

    def _selected_builder_run_payload(self) -> dict[str, Any] | None:
        """Liefert den aktuell selektierten Run im Builder."""

        payload = self._active_builder_payload()
        if payload is None:
            return None
        runs = payload.get("runs", [])
        if not runs:
            return None
        if self.builder_selected_run_id is None:
            return runs[0]
        return next(
            (
                run_payload
                for run_payload in runs
                if run_payload["run_definition"]["run_id"] == self.builder_selected_run_id
            ),
            runs[0],
        )

    def _refresh_experiment_builder_views(self) -> None:
        """Aktualisiert alle Builder-Ansichten auf Basis des aktiven Payloads."""

        self._update_builder_seed_preview()
        self._update_builder_layout_options()
        self._sync_builder_preview_state()
        self._update_builder_summary()
        self._update_builder_run_tree()
        self._update_builder_analysis_text()
        self._update_builder_detail_text()
        self._update_builder_plot()
        if self.model is not None and self.dataset is not None and self.current_layout is not None:
            self._update_prediction_summary()
            self._update_metrics_summary()
            self._update_neuron_detail()
            self._redraw_network()

    def _update_builder_summary(self) -> None:
        """Aktualisiert Kurz- und Langzusammenfassung des Builder-Modus."""

        payload = self._active_builder_payload()
        if payload is None:
            self.builder_summary_var.set(self.t("builder_no_results"))
            if self.builder_summary_text is not None:
                self._set_readonly_text_widget(self.builder_summary_text, self.t("builder_no_results"))
            return

        manifest = payload["manifest"]
        summary = payload["summary"]
        experiment_dir = payload.get("experiment_dir", "")
        self.builder_summary_var.set(
            (
                f"Experiment: {manifest['experiment_id']} | Benchmark: {manifest['benchmark']} | "
                f"Modus: {manifest['run_mode']} | Search: {manifest['search_type']} | "
                f"Konfigurationen: {summary['configuration_count']} | Runs: {summary['number_of_runs']} | "
                f"Seeds: {summary['number_of_seeds']}"
            )
            if not self._is_english()
            else (
                f"Experiment: {manifest['experiment_id']} | Benchmark: {manifest['benchmark']} | "
                f"Mode: {manifest['run_mode']} | Search: {manifest['search_type']} | "
                f"Configurations: {summary['configuration_count']} | Runs: {summary['number_of_runs']} | "
                f"Seeds: {summary['number_of_seeds']}"
            )
        )

        ranking_lines = []
        for rank, entry in enumerate(summary.get("ranking", [])[:8], start=1):
            ranking_lines.append(
                (
                    f"{rank:02d}. {entry['config_id']} | Score={entry['ranking_score']:.6f} | "
                    f"val_acc={entry.get('mean_val_accuracy', 0.0):.4f} | "
                    f"val_loss={entry.get('mean_val_loss', 0.0):.4f}"
                )
            )
        overview_lines = [
            "Experiment Builder Summary" if self._is_english() else "Experiment-Builder-Zusammenfassung",
            "=" * (25 if self._is_english() else 34),
            "",
            f"experiment_id: {manifest['experiment_id']}",
            f"benchmark:     {manifest['benchmark']}",
            f"run_mode:      {manifest['run_mode']}",
            f"search_type:   {manifest['search_type']}",
            f"primary_metric:{summary['primary_metric']}",
            f"runs:          {summary['number_of_runs']}",
            f"seeds:         {summary['number_of_seeds']}",
            f"configs:       {summary['configuration_count']}",
            f"directory:     {experiment_dir}",
            "",
            "Top ranking" if self._is_english() else "Top-Ranking",
            "-----------" if self._is_english() else "-----------",
            *ranking_lines,
        ]
        if self.builder_summary_text is not None:
            self._set_readonly_text_widget(self.builder_summary_text, "\n".join(overview_lines))

    def _update_builder_run_tree(self) -> None:
        """Fuellt die Run-Liste des Builders."""

        if self.builder_run_tree is None:
            return
        self.builder_run_tree.delete(*self.builder_run_tree.get_children())
        payload = self._active_builder_payload()
        if payload is None:
            return
        for run_payload in payload.get("runs", []):
            run_definition = run_payload["run_definition"]
            metrics = run_payload["metrics"]
            run_id = run_definition["run_id"]
            config_id = run_definition["config_id"]
            values = (
                run_id,
                config_id,
                run_definition["seed"],
                f"{metrics.get('val_accuracy', 0.0):.4f}",
                f"{metrics.get('val_loss', 0.0):.4f}",
                f"{metrics.get('test_accuracy', 0.0):.4f}",
                run_definition["run_mode"],
            )
            self.builder_run_tree.insert("", tk.END, iid=run_id, values=values)
        if self.builder_selected_run_id and self.builder_selected_run_id in self.builder_run_tree.get_children():
            self.builder_run_tree.selection_set(self.builder_selected_run_id)
        elif self.builder_run_tree.get_children():
            first_item = self.builder_run_tree.get_children()[0]
            self.builder_run_tree.selection_set(first_item)

    def _update_builder_analysis_text(self) -> None:
        """Aktualisiert die Aggregationsansicht des Builders."""

        if self.builder_analysis_text is None:
            return
        payload = self._active_builder_payload()
        if payload is None:
            self._set_readonly_text_widget(self.builder_analysis_text, self.t("builder_no_results"))
            return

        summary = payload["summary"]
        lines = [
            "Multi-Seed Analysis" if self._is_english() else "Multi-Seed-Analyse",
            "===================" if self._is_english() else "==================",
            "",
            (
                "Ranking uses validation metrics only."
                if self._is_english()
                else "Das Ranking nutzt nur Validation-Metriken."
            ),
            "",
        ]
        for aggregated in summary.get("aggregated_metrics", []):
            lines.extend(
                [
                    f"{aggregated['config_id']} | ranking_score={aggregated['ranking_score']:.6f}",
                    (
                        f"  mean val_acc={aggregated['mean_metrics'].get('val_accuracy', 0.0):.4f} "
                        f"(std {aggregated['std_metrics'].get('val_accuracy', 0.0):.4f})"
                    ),
                    (
                        f"  mean val_loss={aggregated['mean_metrics'].get('val_loss', 0.0):.4f} "
                        f"(std {aggregated['std_metrics'].get('val_loss', 0.0):.4f})"
                    ),
                    (
                        f"  mean test_acc={aggregated['mean_metrics'].get('test_accuracy', 0.0):.4f} "
                        f"(std {aggregated['std_metrics'].get('test_accuracy', 0.0):.4f})"
                    ),
                    (
                        f"  best seed={aggregated['best_seed']} | worst seed={aggregated['worst_seed']}"
                    ),
                ]
            )
            if "best_objective" in aggregated["mean_metrics"]:
                lines.append(
                    f"  mean best_objective={aggregated['mean_metrics'].get('best_objective', 0.0):.6f}"
                )
                lines.append(
                    f"  mean acceptance_rate={aggregated['mean_metrics'].get('acceptance_rate', 0.0):.4f}"
                )
            if aggregated.get("config_values"):
                lines.append(f"  config_values={aggregated['config_values']}")
            lines.append("")
        self._set_readonly_text_widget(self.builder_analysis_text, "\n".join(lines))

    def _update_builder_detail_text(self) -> None:
        """Aktualisiert die Detailansicht fuer einen selektierten Builder-Run."""

        if self.builder_detail_text is None:
            return
        run_payload = self._selected_builder_run_payload()
        if run_payload is None:
            self._set_readonly_text_widget(self.builder_detail_text, self.t("builder_no_results"))
            return

        run_definition = run_payload["run_definition"]
        metrics = run_payload["metrics"]
        extra = run_payload.get("extra", {})
        effective_parameters = extra.get("effective_parameters", {})
        lines = [
            "Per-Seed Detail" if self._is_english() else "Per-Seed-Detail",
            "===============" if self._is_english() else "===============",
            "",
            f"run_id:      {run_definition['run_id']}",
            f"config_id:   {run_definition['config_id']}",
            f"seed:        {run_definition['seed']}",
            f"benchmark:   {run_definition['benchmark']}",
            f"run_mode:    {run_definition['run_mode']}",
            f"layout_spec: {run_payload['layout_spec']}",
            (
                f"search_overrides: {run_definition['config_values']}"
                if self._is_english()
                else f"search_overrides: {run_definition['config_values']}"
            ),
            "",
            "Metrics" if self._is_english() else "Metriken",
            "-------" if self._is_english() else "--------",
        ]
        for metric_name, metric_value in metrics.items():
            lines.append(f"{metric_name}: {metric_value:.6f}")

        if effective_parameters:
            lines.extend(
                [
                    "",
                    "Effective parameters" if self._is_english() else "Effektive Parameter",
                    "-------------------" if self._is_english() else "-------------------",
                ]
            )
            for parameter_name in sorted(effective_parameters):
                lines.append(f"{parameter_name}: {effective_parameters[parameter_name]}")

        history = run_payload.get("history", {})
        if history:
            lines.extend(
                [
                    "",
                    "History lengths" if self._is_english() else "History-Laengen",
                    "---------------" if self._is_english() else "---------------",
                ]
            )
            for key, values in history.items():
                lines.append(f"{key}: {len(values)}")

        if run_definition["run_mode"] == "simulated_annealing":
            start_layout_spec = extra.get("start_layout_spec")
            best_layout_spec = extra.get("best_layout_spec")
            end_layout_spec = extra.get("end_layout_spec")
            if start_layout_spec and best_layout_spec:
                try:
                    hidden_sizes = tuple(int(size) for size in run_definition["hidden_sizes"])
                    start_layout = parse_layout_spec(start_layout_spec, hidden_sizes)
                    best_layout = parse_layout_spec(best_layout_spec, hidden_sizes)
                    diff_text = render_layout_diff(start_layout, best_layout)
                except Exception:
                    diff_text = (
                        "Layout-Diff konnte nicht rekonstruiert werden."
                        if not self._is_english()
                        else "Layout diff could not be reconstructed."
                    )
                lines.extend(
                    [
                        "",
                        "Simulated Annealing" if self._is_english() else "Simulated Annealing",
                        "-------------------",
                        f"start_layout: {start_layout_spec}",
                        f"best_layout:  {best_layout_spec}",
                        f"end_layout:   {end_layout_spec}",
                        f"acceptance_rate: {metrics.get('acceptance_rate', 0.0):.4f}",
                        f"stop_reasons: {extra.get('stop_reasons', [])}",
                        "",
                        "Layout diff" if self._is_english() else "Layout-Diff",
                        "-----------" if self._is_english() else "-----------",
                        diff_text,
                    ]
                )
                annealing_history = extra.get("annealing_history", [])
                if annealing_history:
                    lines.extend(
                        [
                            "",
                            "Last SA steps" if self._is_english() else "Letzte SA-Schritte",
                            "-------------" if self._is_english() else "------------------",
                        ]
                    )
                    for entry in annealing_history[-5:]:
                        lines.append(
                            f"step {entry['step_index']}: T={entry['temperature']:.4f}, "
                            f"delta={entry['delta']:+.6f}, accepted={entry['accepted']}, "
                            f"best_score={entry['best_score_after_step']:.6f}"
                        )

        self._set_readonly_text_widget(self.builder_detail_text, "\n".join(lines))

    def _update_builder_plot(self) -> None:
        """Aktualisiert die Builder-Plotflaeche."""

        if self.builder_plot_container is None:
            return
        for child in self.builder_plot_container.winfo_children():
            child.destroy()
        payload = self._active_builder_payload()
        if payload is None:
            ttk.Label(
                self.builder_plot_container,
                text=self.t("builder_no_results"),
                style="Hint.TLabel",
                justify=tk.LEFT,
            ).grid(row=0, column=0, sticky="nsew")
            return
        self.builder_plot_figure = build_experiment_overview_figure(
            summary_payload=payload["summary"],
            run_payloads=payload["runs"],
            selected_config_id=self.builder_selected_config_id,
            selected_run_id=self.builder_selected_run_id,
        )
        self.builder_plot_canvas_widget = FigureCanvasTkAgg(
            self.builder_plot_figure,
            master=self.builder_plot_container,
        )
        self.builder_plot_canvas_widget.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.builder_plot_canvas_widget.draw_idle()

    def _on_builder_run_selected(self, _event=None) -> None:
        """Synchronisiert Auswahl aus der Builder-Run-Liste."""

        if self.builder_run_tree is None:
            return
        selection = self.builder_run_tree.selection()
        if not selection:
            return
        selected_item = selection[0]
        self.builder_selected_run_id = selected_item
        values = self.builder_run_tree.item(selected_item, "values")
        if len(values) >= 2:
            self.builder_selected_config_id = str(values[1])
        self._update_builder_layout_options()
        self._sync_builder_preview_state()
        self._update_builder_detail_text()
        self._update_builder_plot()
        if self.model is not None and self.dataset is not None and self.current_layout is not None:
            self._update_prediction_summary()
            self._update_metrics_summary()
            self._update_neuron_detail()
            self._redraw_network()

    def _set_readonly_text_widget(self, widget: ScrolledText, text: str) -> None:
        """Schreibt Text in ein schreibgeschuetztes Text-Widget."""

        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.configure(state=tk.DISABLED)

    def _refresh_views(self) -> None:
        """Aktualisiert alle abhaengigen GUI-Bereiche."""

        if self._app_mode() == "experiment_builder":
            self._refresh_guidance()
            self._refresh_experiment_builder_views()
            self._update_help_text()
            return

        self._refresh_guidance()
        self._refresh_sample_tab()
        self._update_prediction_summary()
        self._update_metrics_summary()
        self._update_neuron_detail()
        self._update_training_plot()
        self._update_stepper_view()
        self._update_compare_text()
        self._update_annealing_tab()
        self._redraw_network()
        self._update_help_text()

    def _refresh_guidance(self) -> None:
        """Aktualisiert die kontextabhaengigen Erklaerungen."""

        benchmark = self.benchmark_var.get()
        selected_layer, selected_neuron = self.selected_hidden
        if self._app_mode() == "experiment_builder":
            if self._is_english():
                self.workflow_summary_var.set(
                    "1. Define benchmark, layout, and run mode.\n"
                    "2. Choose a seed strategy and inspect the effective seed list.\n"
                    "3. Set training or annealing parameters.\n"
                    "4. Optional: turn the setup into a grid or random search.\n"
                    "5. Validate the experiment and run it.\n"
                    "6. Inspect ranking, multi-seed aggregation, and per-seed details.\n"
                    "7. In expert mode, edit the builder start layout directly per layer or neuron."
                )
                self.context_hint_var.set(
                    "Experiment Builder is focused on reproducible runs, not on one visible sample. "
                    "Validation metrics drive ranking, while test metrics remain a final evaluation. "
                    "The layout editor changes the next start layout, while 'Current Layouts' is only a preview list."
                )
            else:
                self.workflow_summary_var.set(
                    "1. Definiere Benchmark, Layout und Run-Modus.\n"
                    "2. Waehle eine Seed-Strategie und pruefe die effektive Seed-Liste.\n"
                    "3. Setze Trainings- oder Annealing-Parameter.\n"
                    "4. Optional: mache daraus eine Grid- oder Random-Search.\n"
                    "5. Validiere das Experiment und fuehre es aus.\n"
                    "6. Analysiere Ranking, Multi-Seed-Aggregation und Per-Seed-Details.\n"
                    "7. Bearbeite im Expertenmodus das Builder-Startlayout direkt pro Layer oder Neuron."
                )
                self.context_hint_var.set(
                    "Der Experiment Builder fokussiert reproduzierbare Runs statt eines einzelnen sichtbaren Samples. "
                    "Das Ranking basiert auf Validation, waehrend Testmetriken fuer die finale Einordnung bleiben. "
                    "Der Layout-Editor veraendert das naechste Startlayout, waehrend 'Aktuelle Layouts' nur eine Vorschau ist."
                )
            return
        if self._app_mode() == "playground":
            if self._is_english():
                self.workflow_summary_var.set(
                    "1. Choose benchmark, hidden layers, and a start layout.\n"
                    "2. Choose the objective and candidate training budget.\n"
                    "3. Select which neighborhood moves are allowed.\n"
                    "4. Configure temperature and cooling.\n"
                    "5. Evaluate the start state.\n"
                    "6. Run SA step by step and inspect why moves are accepted or rejected."
                )
                self.context_hint_var.set(
                    "Playground Mode turns the layout into a search state. The current network view still "
                    "shows one concrete sample, but the optimization itself is driven by validation metrics."
                )
            else:
                self.workflow_summary_var.set(
                    "1. Waehle Benchmark, Hidden-Layer und ein Startlayout.\n"
                    "2. Lege Zielmetrik und Trainingsbudget pro Kandidat fest.\n"
                    "3. Waehle erlaubte Nachbarschaftsoperationen.\n"
                    "4. Konfiguriere Temperatur und Abkuehlung.\n"
                    "5. Bewerte den Startzustand.\n"
                    "6. Fuehre SA schrittweise aus und beobachte, warum Zustaende akzeptiert oder verworfen werden."
                )
                self.context_hint_var.set(
                    "Im Playground Mode wird das Layout zu einem Suchzustand. Die Netzansicht zeigt weiterhin "
                    "ein konkretes Sample, die Optimierung selbst wird aber von Validation-Metriken gesteuert."
                )
            return

        if self._is_english():
            self.workflow_summary_var.set(
                "1. Choose a dataset and a sample.\n"
                "2. Observe raw data, target, and prediction.\n"
                "3. Change activations per layer or per neuron.\n"
                "4. Train step by step and watch weights and curves change.\n"
                "5. Use the Forward/Backward tab for the computation path of one sample.\n"
                "6. Click a hidden neuron and inspect its local computation."
            )
        else:
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
                if not self._is_english()
                else (
                    "Digits is especially visual: in the 'Input & Target' tab you see the 8x8 pixels. "
                    "With 'Use Custom Sample' you can build your own digit."
                )
            )
        elif benchmark == "test_activation":
            hint = (
                "test_activation ist das Rechenlabor: wenige Inputs, wenige Neuronen, "
                "schnell sichtbare Effekte von ReLU, tanh, sigmoid und leaky_relu. "
                "Nutze hier besonders die Live-Neuron-Inspektion rechts neben der Netzgrafik."
                if not self._is_english()
                else (
                    "test_activation is the computation lab: few inputs, few neurons, and quickly visible "
                    "effects of ReLU, tanh, sigmoid, and leaky_relu. Use the live neuron inspection on the right."
                )
            )
        else:
            hint = (
                f"Aktuell ist L{selected_layer + 1} n{selected_neuron} selektiert. "
                "Dessen staerkste Ein- und Ausgaenge sind farblich hervorgehoben."
                if not self._is_english()
                else (
                    f"Currently L{selected_layer + 1} n{selected_neuron} is selected. "
                    "Its strongest incoming and outgoing connections are highlighted."
                )
            )
        self.context_hint_var.set(hint)

    def _update_prediction_summary(self) -> None:
        """Zeigt Vorhersage, Ziel und Loss-Informationen fuer das aktuelle Sample."""

        if self.model is None or self.dataset is None:
            self.prediction_summary_var.set(
                "Kein Modell geladen." if not self._is_english() else "No model loaded."
            )
            return

        analysis_sample = self._get_analysis_sample()
        probabilities = self.model.predict_proba(analysis_sample.scaled_sample.reshape(1, -1))[0]
        predicted_index = int(np.argmax(probabilities))
        predicted_name = self.dataset.target_names[predicted_index]

        actual_loss_text = "kein echtes Ziel" if not self._is_english() else "no true target"
        if analysis_sample.actual_target_index is not None:
            actual_probability = float(probabilities[analysis_sample.actual_target_index])
            actual_loss_text = f"{-np.log(max(actual_probability, 1e-12)):.4f}"

        effective_loss_text = "nicht berechnet" if not self._is_english() else "not computed"
        if analysis_sample.effective_target_index is not None:
            effective_probability = float(probabilities[analysis_sample.effective_target_index])
            effective_loss_text = f"{-np.log(max(effective_probability, 1e-12)):.4f}"

        probability_text = ", ".join(
            f"{self.dataset.target_names[index]}={probabilities[index]:.3f}"
            for index in range(len(probabilities))
        )
        if self._is_english():
            self.prediction_summary_var.set(
                f"Observed sample: {analysis_sample.source_label}\n"
                f"Prediction: {predicted_name}\n"
                f"True target: {analysis_sample.actual_target_name}\n"
                f"Analysis target: {analysis_sample.effective_target_name}\n"
                f"Loss to true target: {actual_loss_text}\n"
                f"Loss to analysis target: {effective_loss_text}\n"
                f"Class probabilities: {probability_text}"
            )
        else:
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
                (
                    "Noch kein Training durchgefuehrt.\n"
                    "Du kannst bereits Layouts aendern und lokale Berechnungen analysieren."
                )
                if not self._is_english()
                else (
                    "No training run yet.\n"
                    "You can already change layouts and inspect local computations."
                )
            )
            return

        if self._app_mode() == "playground" and self.annealing_state is not None:
            current_evaluation = self.annealing_state.current_evaluation
            best_evaluation = self.annealing_state.best_evaluation
            if self._is_english():
                self.metrics_summary_var.set(
                    f"Candidate epochs: {self.candidate_epochs_var.get()}\n"
                    f"Current val acc:  {current_evaluation.val_accuracy:.4f}\n"
                    f"Current val loss: {current_evaluation.val_loss:.4f}\n"
                    f"Best val acc:     {best_evaluation.val_accuracy:.4f}\n"
                    f"Best val loss:    {best_evaluation.val_loss:.4f}\n"
                    f"Test acc (current): {current_evaluation.test_accuracy:.4f}"
                )
            else:
                self.metrics_summary_var.set(
                    f"Epochen pro Kandidat: {self.candidate_epochs_var.get()}\n"
                    f"Aktuelle Val-Acc:   {current_evaluation.val_accuracy:.4f}\n"
                    f"Aktueller Val-Loss: {current_evaluation.val_loss:.4f}\n"
                    f"Beste Val-Acc:      {best_evaluation.val_accuracy:.4f}\n"
                    f"Bester Val-Loss:    {best_evaluation.val_loss:.4f}\n"
                    f"Test-Acc (aktuell): {current_evaluation.test_accuracy:.4f}"
                )
            return

        history = self.training_result.history
        best_epoch = max(range(len(history["val_acc"])), key=lambda index: history["val_acc"][index]) + 1
        if self._is_english():
            self.metrics_summary_var.set(
                f"Trained for a total of {self.completed_epochs} epochs.\n"
                f"Latest train acc: {history['train_acc'][-1]:.4f}\n"
                f"Latest val acc:   {history['val_acc'][-1]:.4f}\n"
                f"Best val acc:     {max(history['val_acc']):.4f} in epoch {best_epoch}\n"
                f"Test acc:         {self.training_result.test_metrics['accuracy']:.4f}\n"
                f"Test loss:        {self.training_result.test_metrics['loss']:.4f}"
            )
        else:
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

        if self._is_english():
            self.sample_summary_var.set(
                f"Source: {analysis_sample.source_label}\n"
                f"Split: {analysis_sample.split_name} | Sample index: {analysis_sample.sample_index}\n"
                f"True target: {analysis_sample.actual_target_name}\n"
                f"Analysis target: {analysis_sample.effective_target_name}\n"
                f"Prediction: {predicted_name}"
            )
            self.sample_hint_var.set(
                "Important: the currently shown sample is an analysis window. "
                "Actual training still happens on the full training split."
            )
        else:
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
            self._set_detail_text(
                "Noch kein Modell oder Layout verfuegbar."
                if not self._is_english()
                else "No model or layout available yet."
            )
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
        if self._is_english():
            lines = [
                f"Neuron: L{inspection.layer_index + 1} n{inspection.neuron_index}",
                f"Activation: {inspection.activation_name}",
                f"Formula: {inspection.activation_formula}",
                f"Sample source: {analysis_sample.source_label}",
                f"True target: {analysis_sample.actual_target_name}",
                f"Analysis target: {analysis_sample.effective_target_name}",
                "",
                "Step 1: weighted sum",
                "--------------------",
                "z = sum_i (input_i * weight_i) + bias",
                f"Compact computation: {equation} + ({inspection.bias:+.3f})",
                f"bias = {inspection.bias:+.6f}",
                f"z    = {inspection.pre_activation:+.6f}",
                "",
                "Step 2: activation",
                "------------------",
                f"a = {inspection.activation_name}(z) = {inspection.output_value:+.6f}",
                f"Derivative at this point = {inspection.derivative:+.6f}",
                "",
                "Strongest terms in z",
                "--------------------",
            ]
        else:
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
            if self._is_english():
                lines.append(
                    f"{rank:02d}. {term.source_label}: value={term.source_value:+.6f}, "
                    f"weight={term.weight:+.6f}, contribution={term.contribution:+.6f}"
                )
            else:
                lines.append(
                    f"{rank:02d}. {term.source_label}: wert={term.source_value:+.6f}, "
                    f"gewicht={term.weight:+.6f}, beitrag={term.contribution:+.6f}"
                )

        if len(inspection.input_terms) > len(inspection.top_terms):
            lines.append(
                (
                    f"... {len(inspection.input_terms) - len(inspection.top_terms)} weitere Beitraege ausgeblendet."
                )
                if not self._is_english()
                else (
                    f"... {len(inspection.input_terms) - len(inspection.top_terms)} more contributions hidden."
                )
            )

        lines.extend(
            ["", "Staerkste ausgehende Gewichte", "-----------------------------"]
            if not self._is_english()
            else ["", "Strongest outgoing weights", "------------------------"]
        )
        for label, weight in outgoing_sorted:
            lines.append(
                f"{label}: gewicht={weight:+.6f}"
                if not self._is_english()
                else f"{label}: weight={weight:+.6f}"
            )

        if self._is_english():
            lines.extend(
                [
                    "",
                    "Interpretation",
                    "--------------",
                    "Positive terms push z upward, negative terms push it downward.",
                    "Only after that does the activation function decide how strongly this z",
                    "is turned into an output a. This is exactly where the differences between",
                    "ReLU, tanh, sigmoid, and leaky_relu become visible.",
                ]
            )
        else:
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

        for widget_name in ("detail_text", "live_detail_text"):
            text_widget = getattr(self, widget_name, None)
            if not self._widget_exists(text_widget):
                continue
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
            self._set_stepper_text(
                "Noch kein Modell geladen." if not self._is_english() else "No model loaded yet."
            )
            return

        analysis_sample = self._get_analysis_sample()
        trace = self.model.trace_sample(
            analysis_sample.scaled_sample.reshape(1, -1),
            target_index=analysis_sample.effective_target_index,
        )
        self.step_entries = self._build_step_entries(trace, analysis_sample)
        if not self.step_entries:
            self._set_stepper_text(
                "Noch keine Schrittspur verfuegbar."
                if not self._is_english()
                else "No step trace available yet."
            )
            return

        current_index = min(max(self.step_index_var.get(), 0), len(self.step_entries) - 1)
        self.step_index_var.set(current_index)
        title, text = self.step_entries[current_index]
        self.step_status_label.configure(
            text=(
                f"Schritt {current_index + 1}/{len(self.step_entries)}: {title}"
                if not self._is_english()
                else f"Step {current_index + 1}/{len(self.step_entries)}: {title}"
            )
        )
        self._set_stepper_text(text)

    def _build_step_entries(
        self, trace: SampleTrace, analysis_sample: AnalysisSample
    ) -> list[tuple[str, str]]:
        """Erzeugt textuelle Schrittkarten fuer den Stepper."""

        entries: list[tuple[str, str]] = []
        if self._is_english():
            entries.append(
                (
                    "Input",
                    "\n".join(
                        [
                            f"Source: {analysis_sample.source_label}",
                            f"True target: {analysis_sample.actual_target_name}",
                            f"Analysis target: {analysis_sample.effective_target_name}",
                            "",
                            "Scaled inputs:",
                            ", ".join(
                                f"x{index}={value:+.4f}" for index, value in enumerate(analysis_sample.scaled_sample)
                            ),
                        ]
                    ),
                )
            )
        else:
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
                            ", ".join(
                                f"x{index}={value:+.4f}" for index, value in enumerate(analysis_sample.scaled_sample)
                            ),
                        ]
                    ),
                )
            )

        for layer_trace in trace.forward_layers:
            if self._is_english():
                entries.append(
                    (
                        f"Forward L{layer_trace.layer_index + 1}",
                        "\n".join(
                            [
                                f"Hidden layer L{layer_trace.layer_index + 1}",
                                f"Activations in this layer: {', '.join(layer_trace.activation_names)}",
                                "",
                                "Input to this layer:",
                                ", ".join(
                                    f"{value:+.4f}"
                                    for value in layer_trace.input_values[: min(12, len(layer_trace.input_values))]
                                ),
                                "",
                                "Pre-activations z:",
                                ", ".join(
                                    f"n{index}={value:+.4f}"
                                    for index, value in enumerate(layer_trace.pre_activations)
                                ),
                                "",
                                "Activations a:",
                                ", ".join(
                                    f"n{index}={value:+.4f}"
                                    for index, value in enumerate(layer_trace.activations)
                                ),
                            ]
                        ),
                    )
                )
            else:
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
                                    f"{value:+.4f}"
                                    for value in layer_trace.input_values[: min(12, len(layer_trace.input_values))]
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
                        "Output layer" if self._is_english() else "Output-Layer",
                        "Logits:",
                        ", ".join(f"y{index}={value:+.4f}" for index, value in enumerate(trace.logits)),
                        "",
                        "Probabilities:" if self._is_english() else "Wahrscheinlichkeiten:",
                        ", ".join(
                            f"{self.dataset.target_names[index]}={value:.4f}"
                            for index, value in enumerate(trace.probabilities)
                        ),
                        "",
                        (
                            f"Prediction: {self.dataset.target_names[trace.prediction_index]}"
                            if self._is_english()
                            else f"Vorhersage: {self.dataset.target_names[trace.prediction_index]}"
                        ),
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
                            (
                                f"Analysis target: {analysis_sample.effective_target_name}"
                                if self._is_english()
                                else f"Analyse-Ziel: {analysis_sample.effective_target_name}"
                            ),
                            (
                                f"Cross-entropy loss for this single sample: {trace.loss:.6f}"
                                if self._is_english()
                                else f"Cross-Entropy-Loss fuer dieses einzelne Sample: {trace.loss:.6f}"
                            ),
                            "",
                            "Interpretation:",
                            (
                                "Small values mean the target class already has high probability."
                                if self._is_english()
                                else "Kleine Werte bedeuten, dass die Zielklasse bereits hohe Wahrscheinlichkeit hat."
                            ),
                            (
                                "Large values mean the model is still far away from this target."
                                if self._is_english()
                                else "Grosse Werte bedeuten, dass das Modell fuer dieses Ziel noch stark daneben liegt."
                            ),
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
                            (
                                "Output error vector dL/dlogits"
                                if self._is_english()
                                else "Output-Fehlervektor dL/dlogits"
                            ),
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
                            "Delta values:" if self._is_english() else "Delta-Werte:",
                            ", ".join(
                                f"{value:+.4f}" for value in backward_trace.deltas[: min(16, len(backward_trace.deltas))]
                            ),
                            "",
                            (
                                f"Weight gradient norm: {backward_trace.weight_gradient_norm:.6f}"
                                if self._is_english()
                                else f"Norm des Gewichtsgradienten: {backward_trace.weight_gradient_norm:.6f}"
                            ),
                            (
                                f"Bias gradient norm:   {backward_trace.bias_gradient_norm:.6f}"
                                if self._is_english()
                                else f"Norm des Biasgradienten:    {backward_trace.bias_gradient_norm:.6f}"
                            ),
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

    def _baseline_compatibility_issue(self) -> str | None:
        """Prueft, ob die gespeicherte Baseline mit dem aktuellen Benchmark vergleichbar ist."""

        if self.baseline_model is None or self.dataset is None:
            return None

        if self.baseline_benchmark_name is not None and self.baseline_benchmark_name != self.dataset.name:
            if self._is_english():
                return (
                    f"The stored baseline belongs to benchmark '{self.baseline_benchmark_name}', "
                    f"while the current experiment uses '{self.dataset.name}'."
                )
            return (
                f"Die gespeicherte Baseline gehoert zum Benchmark '{self.baseline_benchmark_name}', "
                f"waehrend das aktuelle Experiment '{self.dataset.name}' verwendet."
            )

        if self.baseline_model.input_size != self.dataset.input_size:
            if self._is_english():
                return (
                    f"The baseline expects {self.baseline_model.input_size} input features, "
                    f"but the current benchmark provides {self.dataset.input_size}."
                )
            return (
                f"Die Baseline erwartet {self.baseline_model.input_size} Eingabefeatures, "
                f"der aktuelle Benchmark liefert aber {self.dataset.input_size}."
            )

        if self.baseline_model.output_size != self.dataset.output_size:
            if self._is_english():
                return (
                    f"The baseline expects {self.baseline_model.output_size} output classes, "
                    f"but the current benchmark provides {self.dataset.output_size}."
                )
            return (
                f"Die Baseline erwartet {self.baseline_model.output_size} Ausgabeklassen, "
                f"der aktuelle Benchmark liefert aber {self.dataset.output_size}."
            )

        return None

    def _update_compare_text(self) -> None:
        """Aktualisiert den Vergleich zwischen Baseline und aktuellem Experiment."""

        if self.baseline_model is None or self.baseline_layout is None or self.dataset is None or self.model is None:
            self.compare_summary_var.set(
                "Noch keine Baseline gespeichert."
                if not self._is_english()
                else "No baseline stored yet."
            )
            self._set_compare_text(
                "Noch keine Baseline gespeichert. Nutze 'Als Baseline speichern'."
                if not self._is_english()
                else "No baseline stored yet. Use 'Store as Baseline'."
            )
            return

        compatibility_issue = self._baseline_compatibility_issue()
        if compatibility_issue is not None:
            self.compare_summary_var.set(
                (
                    f"Baseline nicht vergleichbar mit Benchmark {self.dataset.name}."
                    if not self._is_english()
                    else f"Baseline is not comparable to benchmark {self.dataset.name}."
                )
            )
            if self._is_english():
                lines = [
                    "Baseline vs. Current Experiment",
                    "===============================",
                    "",
                    "Comparison currently unavailable",
                    "--------------------------------",
                    compatibility_issue,
                    "Store a new baseline after switching benchmarks or dataset dimensions.",
                    "",
                    f"Current benchmark: {self.dataset.name}",
                    f"Current hidden sizes:  {format_hidden_sizes(self.model.hidden_sizes)}",
                    f"Baseline benchmark:    {self.baseline_benchmark_name or 'unknown'}",
                    f"Baseline hidden sizes: {format_hidden_sizes(self.baseline_model.hidden_sizes)}",
                    f"Current layout:  {self.current_layout.to_compact_spec()}",
                    f"Baseline layout: {self.baseline_layout.to_compact_spec()}",
                ]
            else:
                lines = [
                    "Baseline vs. aktuelles Experiment",
                    "=================================",
                    "",
                    "Vergleich aktuell nicht moeglich",
                    "--------------------------------",
                    compatibility_issue,
                    "Speichere nach dem Wechsel des Benchmarks oder der Datensatzdimensionen eine neue Baseline.",
                    "",
                    f"Aktueller Benchmark: {self.dataset.name}",
                    f"Hidden-Sizes aktuell:  {format_hidden_sizes(self.model.hidden_sizes)}",
                    f"Benchmark der Baseline: {self.baseline_benchmark_name or 'unbekannt'}",
                    f"Hidden-Sizes Baseline: {format_hidden_sizes(self.baseline_model.hidden_sizes)}",
                    f"Layout aktuell:  {self.current_layout.to_compact_spec()}",
                    f"Layout Baseline: {self.baseline_layout.to_compact_spec()}",
                ]
            self._set_compare_text("\n".join(lines))
            return

        analysis_sample = self._get_analysis_sample()
        current_probabilities = self.model.predict_proba(analysis_sample.scaled_sample.reshape(1, -1))[0]
        baseline_probabilities = self.baseline_model.predict_proba(analysis_sample.scaled_sample.reshape(1, -1))[0]
        current_prediction = self.dataset.target_names[int(np.argmax(current_probabilities))]
        baseline_prediction = self.dataset.target_names[int(np.argmax(baseline_probabilities))]

        if self._is_english():
            lines = [
                "Baseline vs. Current Experiment",
                "===============================",
                "",
                f"Current benchmark: {self.dataset.name}",
                f"Current hidden sizes:  {format_hidden_sizes(self.model.hidden_sizes)}",
                f"Baseline hidden sizes: {format_hidden_sizes(self.baseline_model.hidden_sizes)}",
                f"Current layout:  {self.current_layout.to_compact_spec()}",
                f"Baseline layout: {self.baseline_layout.to_compact_spec()}",
                "",
                f"Sample source: {analysis_sample.source_label}",
                f"Baseline prediction: {baseline_prediction}",
                f"Current prediction: {current_prediction}",
                "",
                "Baseline probabilities:",
                ", ".join(
                    f"{self.dataset.target_names[index]}={baseline_probabilities[index]:.3f}"
                    for index in range(len(baseline_probabilities))
                ),
                "",
                "Current probabilities:",
                ", ".join(
                    f"{self.dataset.target_names[index]}={current_probabilities[index]:.3f}"
                    for index in range(len(current_probabilities))
                ),
                "",
            ]
        else:
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
                    "Baseline Metrics" if self._is_english() else "Baseline-Metriken",
                    "----------------",
                    f"{'Epochs' if self._is_english() else 'Epochen'}:  {self.baseline_completed_epochs}",
                    f"Test-Acc: {self.baseline_training_result.test_metrics['accuracy']:.4f}",
                    f"Test-Loss:{self.baseline_training_result.test_metrics['loss']:.4f}",
                    "",
                ]
            )
        if self.training_result is not None:
            lines.extend(
                [
                    "Current Metrics" if self._is_english() else "Aktuelle Metriken",
                    "---------------",
                    f"{'Epochs' if self._is_english() else 'Epochen'}:  {self.completed_epochs}",
                    f"Test-Acc: {self.training_result.test_metrics['accuracy']:.4f}",
                    f"Test-Loss:{self.training_result.test_metrics['loss']:.4f}",
                    "",
                ]
            )

        if self.baseline_layout.hidden_sizes == self.current_layout.hidden_sizes:
            changes = diff_layouts(self.baseline_layout, self.current_layout)
            lines.append("Layout Diff" if self._is_english() else "Layout-Diff")
            lines.append("-----------")
            if changes:
                lines.extend(
                    f"L{change.layer_index + 1} n{change.neuron_index}: {change.before} -> {change.after}"
                    for change in changes
                )
            else:
                lines.append("No differences." if self._is_english() else "Keine Unterschiede.")
            lines.append("")

        current_neighbors = len(generate_single_step_neighbors(self.current_layout))
        lines.extend(
            [
                "Natural Computing View" if self._is_english() else "Natural-Computing-Sicht",
                "----------------------",
                (
                    f"Current state: {self.current_layout.to_compact_spec()}"
                    if self._is_english()
                    else f"Aktueller Zustand: {self.current_layout.to_compact_spec()}"
                ),
                (
                    f"Number of direct single-step neighbors: {current_neighbors}"
                    if self._is_english()
                    else f"Anzahl direkter Single-Step-Nachbarn: {current_neighbors}"
                ),
                (
                    "A typical objective function later on would be validation accuracy or validation loss."
                    if self._is_english()
                    else "Eine typische Bewertungsfunktion waere spaeter Validation-Accuracy oder Validation-Loss."
                ),
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
                    "Confusion Matrix on Test Split" if self._is_english() else "Confusion-Matrix auf dem Testsplit",
                    "--------------------------------",
                    self._format_confusion_matrix(confusion, self.dataset.target_names),
                    "",
                    "Most Uncertain Test Samples" if self._is_english() else "Unsicherste Test-Samples",
                    "-------------------------",
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

    def _update_annealing_tab(self) -> None:
        """Aktualisiert Text und Verlaufsgrafiken des SA-Tabs."""

        if self._app_mode() != "playground" or self.annealing_text is None:
            return

        if self.annealing_state is None or self.annealing_runner is None:
            self.playground_summary_var.set(
                (
                    "Noch kein SA-Lauf initialisiert."
                    if not self._is_english()
                    else "No SA run initialized yet."
                )
            )
            self.playground_decision_var.set(
                (
                    "Bewerte zuerst den Startzustand."
                    if not self._is_english()
                    else "Evaluate the start state first."
                )
            )
            self._set_annealing_text(
                (
                    "Noch keine Annealing-Historie verfuegbar."
                    if not self._is_english()
                    else "No annealing history available yet."
                )
            )
            self._update_annealing_plot()
            return

        state = self.annealing_state
        objective_name = self.objective_var.get()
        if self._is_english():
            self.playground_summary_var.set(
                f"Step {state.step_index} | Temperature {state.current_temperature:.4f} | "
                f"Current objective {state.current_evaluation.objective_value:.4f} | "
                f"Best objective {state.best_evaluation.objective_value:.4f} | "
                f"Acceptance rate {state.acceptance_rate:.3f} | Cache {self.annealing_runner.evaluator.cache_size()}"
            )
        else:
            self.playground_summary_var.set(
                f"Schritt {state.step_index} | Temperatur {state.current_temperature:.4f} | "
                f"Aktuelles Ziel {state.current_evaluation.objective_value:.4f} | "
                f"Bestes Ziel {state.best_evaluation.objective_value:.4f} | "
                f"Akzeptanzrate {state.acceptance_rate:.3f} | Cache {self.annealing_runner.evaluator.cache_size()}"
            )

        if self.annealing_last_step is None:
            self.playground_decision_var.set(
                (
                    f"Startlayout mit Objective '{objective_name}' bewertet."
                    if not self._is_english()
                    else f"Start layout evaluated with objective '{objective_name}'."
                )
            )
        else:
            self.playground_decision_var.set(self._annealing_reason_text(self.annealing_last_step))

        self._set_annealing_text(self._build_annealing_text())
        self._update_annealing_plot()

    def _set_annealing_text(self, text: str) -> None:
        """Schreibt Text in den Annealing-Tab."""

        if self.annealing_text is None:
            return
        self.annealing_text.configure(state=tk.NORMAL)
        self.annealing_text.delete("1.0", tk.END)
        self.annealing_text.insert("1.0", text)
        self.annealing_text.configure(state=tk.DISABLED)

    def _build_annealing_text(self) -> str:
        """Formatiert die aktuelle SA-Situation als Textbericht."""

        if self.annealing_state is None:
            return (
                "Noch kein SA-Lauf initialisiert."
                if not self._is_english()
                else "No SA run initialized yet."
            )

        state = self.annealing_state
        current_layout_text = render_layout(
            state.current_evaluation.layout,
            title="Aktueller Zustand" if not self._is_english() else "Current State",
        )
        best_layout_text = render_layout(
            state.best_evaluation.layout,
            title="Bester Zustand" if not self._is_english() else "Best State",
        )
        lines = [
            "Simulated Annealing" if self._is_english() else "Simulated Annealing",
            "===================",
            "",
            (
                f"Objective: {self.objective_var.get()}"
                if self._is_english()
                else f"Zielmetrik: {self.objective_var.get()}"
            ),
            (
                f"Current layout: {state.current_evaluation.layout.to_compact_spec()}"
                if self._is_english()
                else f"Aktuelles Layout: {state.current_evaluation.layout.to_compact_spec()}"
            ),
            (
                f"Best layout:    {state.best_evaluation.layout.to_compact_spec()}"
                if self._is_english()
                else f"Bestes Layout:  {state.best_evaluation.layout.to_compact_spec()}"
            ),
            (
                f"Current objective value: {state.current_evaluation.objective_value:.6f}"
                if self._is_english()
                else f"Aktueller Zielwert: {state.current_evaluation.objective_value:.6f}"
            ),
            (
                f"Best objective value:    {state.best_evaluation.objective_value:.6f}"
                if self._is_english()
                else f"Bester Zielwert:    {state.best_evaluation.objective_value:.6f}"
            ),
            "",
        ]

        if self.annealing_last_step is not None:
            step = self.annealing_last_step
            lines.extend(
                [
                    "Latest decision" if self._is_english() else "Letzte Entscheidung",
                    "----------------" if self._is_english() else "-------------------",
                    self._annealing_reason_text(step),
                    (
                        f"Neighbor: {step.neighbor_label}"
                        if self._is_english()
                        else f"Nachbar: {step.neighbor_label}"
                    ),
                    (
                        f"Old comparable score: {step.previous_evaluation.comparable_score:.6f}"
                        if self._is_english()
                        else f"Alter Vergleichsscore: {step.previous_evaluation.comparable_score:.6f}"
                    ),
                    (
                        f"Candidate comparable score: {step.candidate_evaluation.comparable_score:.6f}"
                        if self._is_english()
                        else f"Kandidaten-Score: {step.candidate_evaluation.comparable_score:.6f}"
                    ),
                    f"Delta = {step.delta:+.6f}",
                    f"T = {step.temperature:.6f}",
                    (
                        f"Acceptance probability = {step.acceptance_probability:.6f}"
                        if self._is_english()
                        else f"Akzeptanzwahrscheinlichkeit = {step.acceptance_probability:.6f}"
                    ),
                    (
                        f"Random draw = {step.random_draw:.6f}"
                        if self._is_english()
                        else f"Zufallszahl = {step.random_draw:.6f}"
                    ),
                    "",
                ]
            )

            candidate_title = "Candidate Layout" if self._is_english() else "Kandidaten-Layout"
            lines.append(render_layout(step.candidate_layout, title=candidate_title))
            lines.append("")
            if step.previous_layout.hidden_sizes == step.candidate_layout.hidden_sizes:
                diff_title = "Layout-Diff" if not self._is_english() else "Layout Diff"
                lines.append(diff_title)
                lines.append("-" * len(diff_title))
                lines.append(render_layout_diff(step.previous_layout, step.candidate_layout))
                lines.append("")

        if state.start_evaluation.layout.hidden_sizes == state.best_evaluation.layout.hidden_sizes:
            title = "Start vs. Best" if self._is_english() else "Start vs. Best"
            lines.append(title)
            lines.append("-" * len(title))
            lines.append(render_layout_diff(state.start_evaluation.layout, state.best_evaluation.layout))
            lines.append("")

        if self.annealing_runner is not None:
            stop_reasons = annealing_stop_reasons(
                state=state,
                config=self.annealing_runner.config,
                neighbor_count=len(
                    generate_neighbors(
                        state.current_evaluation.layout,
                        self.annealing_runner.config.neighborhood_operations,
                    )
                ),
            )
            if stop_reasons:
                lines.extend(
                    [
                        "Stop Reasons" if self._is_english() else "Stopgruende",
                        "------------" if self._is_english() else "-----------",
                    ]
                )
                for reason in stop_reasons:
                    if reason == "max_steps_reached":
                        lines.append(
                            "Maximum number of steps reached."
                            if self._is_english()
                            else "Maximale Schrittzahl erreicht."
                        )
                    elif reason == "temperature_below_threshold":
                        lines.append(
                            "Temperature is at or below the minimum threshold."
                            if self._is_english()
                            else "Temperatur liegt auf oder unter der Mindestschwelle."
                        )
                    elif reason == "no_neighbors":
                        lines.append(
                            "No further neighbors can be generated for the current search space."
                            if self._is_english()
                            else "Im aktuellen Suchraum koennen keine weiteren Nachbarn erzeugt werden."
                        )
                lines.append("")

        lines.extend([current_layout_text, "", best_layout_text])
        return "\n".join(lines)

    def _annealing_reason_text(self, step: AnnealingStep) -> str:
        """Erzeugt einen didaktischen Klartext fuer eine SA-Entscheidung."""

        if step.reason_code == "improved_or_equal":
            return (
                "Der Kandidat war mindestens so gut wie der aktuelle Zustand und wurde deshalb direkt akzeptiert."
                if not self._is_english()
                else "The candidate was at least as good as the current state and was therefore accepted immediately."
            )
        if step.reason_code == "accepted_worse":
            return (
                f"Der Kandidat war schlechter, wurde aber wegen der Temperatur noch akzeptiert "
                f"({step.random_draw:.4f} <= {step.acceptance_probability:.4f})."
                if not self._is_english()
                else (
                    f"The candidate was worse but was still accepted because of the current temperature "
                    f"({step.random_draw:.4f} <= {step.acceptance_probability:.4f})."
                )
            )
        return (
            f"Der Kandidat war schlechter und wurde verworfen "
            f"({step.random_draw:.4f} > {step.acceptance_probability:.4f})."
            if not self._is_english()
            else (
                f"The candidate was worse and was rejected "
                f"({step.random_draw:.4f} > {step.acceptance_probability:.4f})."
            )
        )

    def _update_annealing_plot(self) -> None:
        """Aktualisiert die Verlaufsgrafiken im SA-Tab."""

        if self.annealing_figure is None or self.annealing_canvas_widget is None:
            return

        for axis in self.annealing_axes:
            axis.clear()

        if self.annealing_state is None:
            for axis in self.annealing_axes:
                axis.text(
                    0.5,
                    0.5,
                    "Noch kein SA-Lauf" if not self._is_english() else "No SA run yet",
                    ha="center",
                    va="center",
                )
                axis.set_axis_off()
            self.annealing_figure.tight_layout()
            self.annealing_canvas_widget.draw_idle()
            return

        state = self.annealing_state
        score_axis, temperature_axis, probability_axis, rate_axis = self.annealing_axes

        current_scores = [state.start_evaluation.comparable_score]
        best_scores = [state.start_evaluation.comparable_score]
        temperatures = []
        probabilities = []
        accepted_flags = []
        acceptance_rates = [0.0]

        current_score = state.start_evaluation.comparable_score
        best_score = state.start_evaluation.comparable_score
        accepted_count = 0
        for step in state.history:
            temperatures.append(step.temperature)
            probabilities.append(step.acceptance_probability)
            accepted_flags.append(1.0 if step.accepted else 0.0)
            if step.accepted:
                current_score = step.candidate_evaluation.comparable_score
                accepted_count += 1
            best_score = min(best_score, step.best_score_after_step)
            current_scores.append(current_score)
            best_scores.append(best_score)
            acceptance_rates.append(accepted_count / step.step_index)

        score_x = np.arange(len(current_scores))
        score_axis.plot(score_x, current_scores, label="Current", linewidth=2)
        score_axis.plot(score_x, best_scores, label="Best", linewidth=2)
        score_axis.set_title("Comparable Score")
        score_axis.set_xlabel("Step" if self._is_english() else "Schritt")
        score_axis.grid(alpha=0.3)
        score_axis.legend()

        if temperatures:
            temp_x = np.arange(1, len(temperatures) + 1)
            temperature_axis.plot(temp_x, temperatures, linewidth=2, color="#2563eb")
            temperature_axis.set_title("Temperature")
            temperature_axis.set_xlabel("Step" if self._is_english() else "Schritt")
            temperature_axis.grid(alpha=0.3)
            probability_axis.plot(temp_x, probabilities, linewidth=2, color="#f59e0b")
            probability_axis.scatter(temp_x, accepted_flags, color="#16a34a", s=20)
            probability_axis.set_title(
                "Acceptance Probability"
                if self._is_english()
                else "Akzeptanzwahrscheinlichkeit"
            )
            probability_axis.set_xlabel("Step" if self._is_english() else "Schritt")
            probability_axis.set_ylim(-0.05, 1.05)
            probability_axis.grid(alpha=0.3)
            rate_axis.plot(score_x, acceptance_rates, linewidth=2, color="#7c3aed")
            rate_axis.set_title(
                "Acceptance Rate" if self._is_english() else "Akzeptanzrate"
            )
            rate_axis.set_xlabel("Step" if self._is_english() else "Schritt")
            rate_axis.set_ylim(0.0, 1.0)
            rate_axis.grid(alpha=0.3)
        else:
            empty_text = "Nur Startzustand" if not self._is_english() else "Only start state"
            for axis in (temperature_axis, probability_axis, rate_axis):
                axis.text(0.5, 0.5, empty_text, ha="center", va="center")
                axis.set_axis_off()

        self.annealing_figure.tight_layout()
        self.annealing_canvas_widget.draw_idle()

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
            return ["Keine Samples verfuegbar."] if not self._is_english() else ["No samples available."]
        probabilities = self.model.predict_proba(X_split)
        predictions = np.argmax(probabilities, axis=1)
        confidence = np.max(probabilities, axis=1)
        ordering = np.argsort(confidence)[: min(5, len(confidence))]
        return [
            (
                f"idx {int(index):03d}: {'true' if self._is_english() else 'true'}={self.dataset.target_names[int(y_split[index])]}, "
                f"{'pred' if self._is_english() else 'pred'}={self.dataset.target_names[int(predictions[index])]}, "
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
            train_label = "Train"
            val_label = "Val"
            loss_axis.plot(epochs, self.training_result.history["train_loss"], label=train_label, linewidth=2)
            loss_axis.plot(epochs, self.training_result.history["val_loss"], label=val_label, linewidth=2)
            loss_axis.set_title("Loss")
            loss_axis.set_xlabel("Epoch" if self._is_english() else "Epoche")
            loss_axis.grid(alpha=0.3)
            loss_axis.legend()

            accuracy_axis.plot(epochs, self.training_result.history["train_acc"], label=train_label, linewidth=2)
            accuracy_axis.plot(epochs, self.training_result.history["val_acc"], label=val_label, linewidth=2)
            accuracy_axis.set_title("Accuracy")
            accuracy_axis.set_xlabel("Epoch" if self._is_english() else "Epoche")
            accuracy_axis.grid(alpha=0.3)
            accuracy_axis.legend()
        else:
            empty_text = "Noch kein Training" if not self._is_english() else "No training yet"
            loss_axis.text(0.5, 0.5, empty_text, ha="center", va="center")
            accuracy_axis.text(0.5, 0.5, empty_text, ha="center", va="center")
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
        probability_axis.set_title(
            "Klassenwahrscheinlichkeiten" if not self._is_english() else "Class Probabilities"
        )
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
        hidden_axis.set_title(
            "Aktivierungen des aktuellen Samples"
            if not self._is_english()
            else "Activations of the Current Sample"
        )
        hidden_axis.tick_params(axis="x", rotation=55)
        hidden_axis.axhline(0.0, color="#64748b", linewidth=1)

        self.training_figure.tight_layout()
        self.training_canvas_widget.draw_idle()

    def _update_help_text(self) -> None:
        """Aktualisiert den Lernhilfe-Tab passend zum aktuellen Kontext."""

        benchmark = self.benchmark_var.get()
        is_playground = self._app_mode() == "playground"
        if self._app_mode() == "experiment_builder":
            if self._is_english():
                help_lines = [
                    "Experiment Builder Help",
                    "=======================",
                    "",
                    "What is this mode for?",
                    "----------------------",
                    "This mode is not focused on one visible sample.",
                    "It is focused on reproducible experiments, multiple seeds, search spaces, and later re-analysis.",
                    "",
                    "Core concepts",
                    "-------------",
                    "Experiment = one full definition consisting of benchmark, layout, seeds, mode, and optional search space.",
                    "Run = one concrete execution for exactly one configuration and one seed.",
                    "Configuration = one concrete parameter setting produced from the search space.",
                    "Seed = controls randomness for initialization, data split, and reproducible comparison.",
                    "Start layout = the layout configured on the left for the next run.",
                    "Current layouts = a preview list for setup, start, best, or end layouts from a selected run.",
                    "",
                    "Recommended workflow",
                    "--------------------",
                    "1. Choose benchmark, hidden layers, layout, and run mode.",
                    "2. Choose seed strategy and check the effective seed list.",
                    "3. Set training or annealing parameters.",
                    "4. Optional: enable grid search or random search.",
                    "5. Validate the experiment before running it.",
                    "6. Use the run list, multi-seed analysis, and per-seed detail tabs after execution.",
                    "",
                    "Why validation instead of test?",
                    "-------------------------------",
                    "Validation is used to compare configurations because test should remain a final evaluation set.",
                    "If you rank by test, you leak information from the final evaluation into model selection.",
                    "",
                    "Search types",
                    "------------",
                    "No Search: run exactly one configuration across multiple seeds.",
                    "Grid Search: test all discrete combinations from the builder search space.",
                    "Random Search: reproducibly sample only part of the same discrete search space.",
                    "",
                    "Simulated annealing inside the builder",
                    "--------------------------------------",
                    "In SA mode, each run optimizes the activation layout instead of only training one fixed layout.",
                    "The stored results include the start layout, best layout, end layout, acceptance rate, and step history.",
                    "The layout editor on the left defines the start layout of the next SA run. It does not change old stored runs.",
                    "",
                    "Practical parameter hints",
                    "-------------------------",
                    *self._parameter_help_lines(),
                    "",
                    "SA start recommendations",
                    "------------------------",
                    *self._sa_start_recommendation_lines(),
                    "",
                    "Suggested experiment recipes",
                    "----------------------------",
                    *self._experiment_recipe_lines(),
                    "",
                    "Result files",
                    "------------",
                    "Stored experiments write one manifest, one summary, and one JSON file per run.",
                    "This makes later loading and analysis possible without rerunning everything.",
                ]
            else:
                help_lines = [
                    "Lernhilfe fuer den Experiment Builder",
                    "=====================================",
                    "",
                    "Wofuer ist dieser Modus gedacht?",
                    "--------------------------------",
                    "Dieser Modus fokussiert nicht ein einzelnes sichtbares Sample.",
                    "Er fokussiert reproduzierbare Experimente, mehrere Seeds, Suchraeume und spaetere Wiederanalyse.",
                    "",
                    "Wichtige Begriffe",
                    "-----------------",
                    "Experiment = eine vollstaendige Definition aus Benchmark, Layout, Seeds, Modus und optionalem Suchraum.",
                    "Run = eine konkrete Ausfuehrung fuer genau eine Konfiguration und genau einen Seed.",
                    "Konfiguration = eine konkrete Parameterbelegung aus dem Suchraum.",
                    "Seed = steuert Zufall fuer Initialisierung, Datensplit und reproduzierbare Vergleiche.",
                    "Startlayout = das links konfigurierte Layout fuer den naechsten Run.",
                    "Aktuelle Layouts = reine Vorschau-Liste fuer Setup-, Start-, Best- oder Endlayout eines selektierten Runs.",
                    "",
                    "Empfohlener Ablauf",
                    "------------------",
                    "1. Waehle Benchmark, Hidden-Layer, Layout und Run-Modus.",
                    "2. Waehle die Seed-Strategie und pruefe die effektive Seed-Liste.",
                    "3. Setze Trainings- oder Annealing-Parameter.",
                    "4. Optional: aktiviere Grid Search oder Random Search.",
                    "5. Validiere das Experiment vor dem Start.",
                    "6. Nutze danach Run-Liste, Multi-Seed-Analyse und Per-Seed-Details.",
                    "",
                    "Warum Validation statt Test?",
                    "----------------------------",
                    "Validation wird fuer den Vergleich von Konfigurationen genutzt, weil Test eine finale Bewertungsmenge bleiben soll.",
                    "Wenn nach Test gerankt wird, fliesst die Endbewertung in die Modellwahl ein.",
                    "",
                    "Search-Typen",
                    "------------",
                    "Keine Suche: genau eine Konfiguration ueber mehrere Seeds ausfuehren.",
                    "Grid Search: alle diskreten Kombinationen aus dem Builder-Suchraum testen.",
                    "Random Search: reproduzierbar nur einen Teil desselben diskreten Suchraums sampeln.",
                    "",
                    "Simulated Annealing im Builder",
                    "------------------------------",
                    "Im SA-Modus optimiert jeder Run das Aktivierungs-Layout statt nur ein fixes Layout zu trainieren.",
                    "Die gespeicherten Ergebnisse enthalten Startlayout, Bestlayout, Endlayout, Akzeptanzrate und Schrittverlauf.",
                    "Der Layout-Editor links definiert das Startlayout des naechsten SA-Runs. Alte gespeicherte Runs werden dadurch nicht veraendert.",
                    "",
                    "Praktische Parameterhinweise",
                    "----------------------------",
                    *self._parameter_help_lines(),
                    "",
                    "SA-Startempfehlungen",
                    "--------------------",
                    *self._sa_start_recommendation_lines(),
                    "",
                    "Empfohlene Experimentrezepte",
                    "----------------------------",
                    *self._experiment_recipe_lines(),
                    "",
                    "Ergebnisdateien",
                    "---------------",
                    "Gespeicherte Experimente schreiben ein Manifest, eine Summary und genau eine JSON-Datei pro Run.",
                    "So lassen sich Ergebnisse spaeter wieder laden und analysieren, ohne alles neu laufen zu lassen.",
                ]
            self.help_text.configure(state=tk.NORMAL)
            self.help_text.delete("1.0", tk.END)
            self.help_text.insert("1.0", "\n".join(help_lines))
            self.help_text.configure(state=tk.DISABLED)
            return
        if self._is_english():
            workflow_lines = (
                [
                    "1. Choose a benchmark or the test_activation mode.",
                    "2. Use the 'Input & Target' tab to inspect the current data flowing into the network.",
                    "3. Change activations per layer or per neuron.",
                    "4. Train in small steps with 1 or 10 epochs.",
                    "5. Observe how prediction, loss, and activations change.",
                    "6. Click hidden neurons and inspect their local computation.",
                ]
                if not is_playground
                else [
                    "1. Choose a benchmark, hidden layers, and a start layout.",
                    "2. Decide how each candidate is scored: validation loss or validation accuracy.",
                    "3. Choose which neighborhood moves are allowed in the search.",
                    "4. Set temperature, cooling, and the training budget per candidate.",
                    "5. Evaluate the start state and inspect the initial objective value.",
                    "6. Run simulated annealing step by step and compare current, candidate, and best state.",
                ]
            )
            core_lines = (
                [
                    "z = weighted sum + bias",
                    "a = activation function(z)",
                    "Training changes weights and biases, not the dataset.",
                    "The currently shown sample is an analysis window, not the full training process.",
                ]
                if not is_playground
                else [
                    "State = one concrete activation layout.",
                    "Neighbor = a small modification of that layout.",
                    "Objective = validation loss or validation accuracy after short candidate training.",
                    "Temperature controls how often worse candidates may still be accepted.",
                ]
            )
            help_lines = [
                "Learning Help",
                "=============",
                "",
                "Recommended Workflow",
                "--------------------",
                *workflow_lines,
                "",
                "Core Concepts",
                "-------------",
                *core_lines,
                "",
                "Current Benchmark",
                "-----------------",
                *self._benchmark_help_lines(benchmark),
                "",
                "Parameter Glossary",
                "------------------",
                *self._parameter_help_lines(),
                "",
            ]
            if is_playground:
                help_lines.extend(
                    [
                        "Playground Mode Notes",
                        "---------------------",
                        "The visible network still shows one concrete sample, but simulated annealing optimizes layouts using validation metrics over the whole benchmark.",
                        "The current state is the layout currently used for inspection and for the current score.",
                        "The candidate state is the newly proposed neighbor before acceptance or rejection.",
                        "The best state is the best layout found so far during the search.",
                        "The annealing plots track score, temperature, acceptance probability, and cumulative acceptance rate.",
                        "",
                    ]
                )
            elif benchmark == "digits":
                help_lines.extend(
                    [
                        "Digits-Specific Notes",
                        "---------------------",
                        "Each sample is an 8x8 pixel image with values between 0 and 16.",
                        "With 'Use Custom Sample' you can set pixels manually.",
                        "This makes it easy to see how the network reacts to small image changes.",
                        "Pay attention to how probabilities shift when only a few pixels are modified.",
                        "",
                    ]
                )
            elif benchmark == "test_activation":
                help_lines.extend(
                    [
                        "test_activation Notes",
                        "---------------------",
                        "This mode is less about benchmarking and more about raw computation.",
                        "Set a few inputs manually and watch how z and a change in each neuron.",
                        "It is ideal for seeing the differences between ReLU, tanh, sigmoid, and leaky_relu.",
                        "The panel next to the network visualization always shows the computation of the currently selected hidden neuron.",
                        "Pay attention to how small input changes affect z first and activation a afterwards.",
                        "",
                    ]
                )
            else:
                help_lines.extend(
                    [
                        "Tabular Benchmark Notes",
                        "-----------------------",
                        "The table view shows the strongest input features of the current sample.",
                        "The network still uses all features, even when not every one is displayed.",
                        "The visible input nodes are a didactic subset to keep the strongest influences readable.",
                        "",
                    ]
                )
            help_lines.extend(
                [
                    "Useful Questions",
                    "----------------",
                    *(
                        [
                            "Which activations produce large or small outputs for this sample?",
                            "Which input features influence the selected neuron most strongly?",
                            "How does training change class output and hidden activations?",
                            "When does sigmoid saturate, and when is ReLU more sparse?",
                        ]
                        if not is_playground
                        else [
                            "Which neighborhood operations produce meaningful alternative layouts?",
                            "Is the objective changing because the layout improved or because candidate training is noisy?",
                            "How quickly does the temperature become restrictive for worse states?",
                            "Does the best state really outperform the start layout on validation and test metrics?",
                        ]
                    ),
                ]
            )
        else:
            workflow_lines = (
                [
                    "1. Waehle einen Benchmark oder den test_activation-Modus.",
                    "2. Schaue im Tab 'Input & Ziel', welche Daten aktuell ins Netz fliessen.",
                    "3. Veraendere Aktivierungen pro Layer oder pro Neuron.",
                    "4. Trainiere in kleinen Schritten mit 1 oder 10 Epochen.",
                    "5. Beobachte, wie sich Vorhersage, Loss und Aktivierungen veraendern.",
                    "6. Klicke auf einzelne Hidden-Neuronen und lies ihre lokale Rechnung.",
                ]
                if not is_playground
                else [
                    "1. Waehle Benchmark, Hidden-Layer und ein Startlayout.",
                    "2. Lege fest, wie Kandidaten bewertet werden: Validation-Loss oder Validation-Accuracy.",
                    "3. Waehle erlaubte Nachbarschaftsoperationen fuer die Suche.",
                    "4. Setze Temperatur, Abkuehlung und Trainingsbudget pro Kandidat.",
                    "5. Bewerte den Startzustand und betrachte den ersten Zielwert.",
                    "6. Fuehre Simulated Annealing schrittweise aus und vergleiche aktuellen Zustand, Kandidat und bestes Layout.",
                ]
            )
            core_lines = (
                [
                    "z = gewichtete Summe + bias",
                    "a = Aktivierungsfunktion(z)",
                    "Trainieren aendert Gewichte und Biases, nicht den Datensatz.",
                    "Das aktuell gezeigte Sample ist ein Analysefenster, nicht der ganze Trainingsprozess.",
                ]
                if not is_playground
                else [
                    "Zustand = ein konkretes Aktivierungs-Layout.",
                    "Nachbar = eine kleine Aenderung dieses Layouts.",
                    "Zielmetrik = Validation-Loss oder Validation-Accuracy nach kurzem Kandidatentraining.",
                    "Die Temperatur steuert, wie oft auch schlechtere Kandidaten noch akzeptiert werden duerfen.",
                ]
            )
            help_lines = [
                "Lernhilfe fuer den Playground",
                "=============================",
                "",
                "Empfohlener Ablauf",
                "------------------",
                *workflow_lines,
                "",
                "Wichtige Konzepte",
                "-----------------",
                *core_lines,
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
            if is_playground:
                help_lines.extend(
                    [
                        "Hinweise zum Playground Mode",
                        "----------------------------",
                        "Die sichtbare Netzwerkansicht zeigt weiterhin ein einzelnes Sample, aber Simulated Annealing optimiert Layouts ueber Validation-Metriken des gesamten Benchmarks.",
                        "Der aktuelle Zustand ist das Layout, das gerade fuer Anzeige und Score gilt.",
                        "Der Kandidat ist der neu vorgeschlagene Nachbar vor Annahme oder Verwerfung.",
                        "Der beste Zustand ist das beste bisher gefundene Layout.",
                        "Die Annealing-Plots zeigen Score, Temperatur, Akzeptanzwahrscheinlichkeit und die kumulative Akzeptanzrate.",
                        "",
                    ]
                )
            elif benchmark == "digits":
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
                    "Beobachtungsfragen",
                    "------------------",
                    *(
                        [
                            "Welche Aktivierungen fuehren bei diesem Sample zu grossen oder kleinen Ausgaben?",
                            "Welche Eingabefeatures beeinflussen das selektierte Neuron am staerksten?",
                            "Wie veraendert Training die Klassenausgabe und die Hidden-Aktivierungen?",
                            "Wann wirkt sigmoid eher saettigend, wann ist ReLU sparsamer?",
                        ]
                        if not is_playground
                        else [
                            "Welche Nachbarschaftsoperationen liefern sinnvolle alternative Layouts?",
                            "Aendert sich die Zielmetrik wirklich wegen eines besseren Layouts oder nur wegen leichtem Trainingsrauschen?",
                            "Wie schnell wird die Temperatur fuer schlechtere Kandidaten restriktiv?",
                            "Schlaegt der beste Zustand das Startlayout wirklich auf Validation- und Testmetriken?",
                        ]
                    ),
                ]
            )

        self.help_text.configure(state=tk.NORMAL)
        self.help_text.delete("1.0", tk.END)
        self.help_text.insert("1.0", "\n".join(help_lines))
        self.help_text.configure(state=tk.DISABLED)

    def _build_dataset_summary_text(self) -> str:
        """Erzeugt eine didaktische Kurzbeschreibung des aktuell geladenen Benchmarks."""

        if self.dataset is None:
            return "Kein Datensatz geladen." if not self._is_english() else "No dataset loaded."
        if self._is_english():
            return (
                f"{self.dataset.name}: {self.dataset.input_size} inputs, {self.dataset.output_size} classes, "
                f"train/val/test = {self.dataset.train_size}/{self.dataset.validation_size}/{self.dataset.test_size}\n"
                "Training objective: the model learns from all samples in the training split.\n"
                f"Classes: {', '.join(self.dataset.target_names)}\n"
                f"Benchmark note: {BENCHMARK_DESCRIPTIONS['en'][self.dataset.name]}"
            )
        return (
            f"{describe_dataset(self.dataset)}\n"
            f"Trainingsziel: Das Modell lernt auf allen Beispielen aus dem Trainingssplit.\n"
            f"Klassen: {', '.join(self.dataset.target_names)}\n"
            f"Benchmark-Hinweis: {BENCHMARK_DESCRIPTIONS['de'][self.dataset.name]}"
        )

    def _benchmark_help_lines(self, benchmark: str) -> list[str]:
        """Liefert erklaerende Stichpunkte zum aktuell gewaehlten Benchmark."""

        if benchmark == "breast_cancer":
            return (
                [
                    "Name: breast_cancer",
                    "30 numerical features, 2 classes.",
                    "Use: simple entry point for classification with tabular data.",
                    "Watch how quickly validation and test accuracy stabilize.",
                    "Useful for comparing ReLU and sigmoid in a simple setting.",
                ]
                if self._is_english()
                else [
                    "Name: breast_cancer",
                    "30 numerische Features, 2 Klassen.",
                    "Didaktischer Nutzen: einfacher Einstieg in Klassifikation mit tabellarischen Daten.",
                    "Beobachte hier besonders, wie schnell Val- und Test-Accuracy stabil werden.",
                    "Gut geeignet, um den Effekt von ReLU vs. sigmoid in einem einfachen Setting zu sehen.",
                ]
            )
        if benchmark == "wine":
            return (
                [
                    "Name: wine",
                    "13 numerical features, 3 classes.",
                    "Use: readable multiclass setup with compact learning curves.",
                    "Very useful for layout comparisons because differences often become visible quickly.",
                    "Watch whether mixed activations lead to different probability profiles.",
                ]
                if self._is_english()
                else [
                    "Name: wine",
                    "13 numerische Features, 3 Klassen.",
                    "Didaktischer Nutzen: ueberschaubarer Mehrklassenfall mit gut lesbaren Lernkurven.",
                    "Sehr geeignet fuer Vergleiche von Layouts, weil Unterschiede oft klar sichtbar werden.",
                    "Beobachte, ob gemischte Aktivierungen im Hidden-Bereich zu anderen Wahrscheinlichkeitsprofilen fuehren.",
                ]
            )
        if benchmark == "digits":
            return (
                [
                    "Name: digits",
                    "64 inputs from an 8x8 image, 10 classes.",
                    "Use: input stream, target, and prediction can be understood visually.",
                    "Ideal for seeing which pixel constellations lead to uncertain predictions.",
                    "Watch how small pixel changes propagate through the network.",
                ]
                if self._is_english()
                else [
                    "Name: digits",
                    "64 Eingaben aus einem 8x8 Bild, 10 Klassen.",
                    "Didaktischer Nutzen: Eingabestrom, Ziel und Vorhersage sind bildhaft nachvollziehbar.",
                    "Ideal, um visuell zu sehen, welche Pixelkonfigurationen zu unsicheren Vorhersagen fuehren.",
                    "Beobachte, wie sich kleine Pixelveraenderungen durch das Netz fortpflanzen.",
                ]
            )
        return (
            [
                "Name: test_activation",
                "3 artificial inputs, 2 classes.",
                "Use: minimal computation lab for raw activation and summation behavior.",
                "Here the important part is not absolute accuracy but understanding z, a, and derivatives.",
                "Ideal for discussing the full local computation by clicking single neurons.",
            ]
            if self._is_english()
            else [
                "Name: test_activation",
                "3 kuenstliche Inputs, 2 Klassen.",
                "Didaktischer Nutzen: minimales Rechenlabor fuer rohe Aktivierungs- und Summenrechnungen.",
                "Hier ist weniger die absolute Accuracy wichtig, sondern das Verstehen von z, a und Ableitungen.",
                "Ideal, um durch Anklicken einzelner Neuronen die komplette lokale Rechnung zu diskutieren.",
            ]
        )

    def _sa_start_recommendation_lines(self) -> list[str]:
        """Liefert einfache Startempfehlungen fuer erste SA-Laeufe."""

        if self._is_english():
            return [
                "wine: start with 2 hidden layers such as 16 / 8, mixed layouts, objective = validation_loss, candidate_epochs = 10 to 20, geometric cooling, and medium temperatures.",
                "digits: start with 32 / 16, do not make candidate training too short, and prefer more SA steps because the benchmark is noisier.",
                "test_activation: keep the network tiny, use few steps, and focus on understanding start layout, neighbor proposals, and why states are accepted.",
                "If SA looks random, first increase candidate_epochs a little before changing the cooling aggressively.",
                "If almost no worse state is accepted, the temperature may be too low or the cooling too aggressive.",
            ]
        return [
            "wine: starte z. B. mit 2 Hidden-Layern wie 16 / 8, gemischten Layouts, objective = validation_loss, candidate_epochs = 10 bis 20, geometrischer Abkuehlung und mittlerer Temperatur.",
            "digits: starte mit 32 / 16, mache das Kandidatentraining nicht zu kurz und nutze eher mehr SA-Schritte, weil der Benchmark rauschanfaelliger ist.",
            "test_activation: halte das Netz klein, nutze wenige Schritte und konzentriere dich darauf, Startlayout, Nachbarn und Annahmeentscheidungen zu verstehen.",
            "Wenn SA zufaellig wirkt, erhoehe zuerst candidate_epochs leicht, bevor du die Abkuehlung stark veraenderst.",
            "Wenn fast nie schlechtere Zustaende akzeptiert werden, ist die Temperatur oft zu niedrig oder die Abkuehlung zu hart.",
        ]

    def _experiment_recipe_lines(self) -> list[str]:
        """Liefert konkrete erste Experimentideen fuer neue Nutzer."""

        if self._is_english():
            return [
                "Recipe 1: ReLU vs Tanh on wine. Keep hidden sizes fixed, run several seeds, and compare mean validation accuracy plus probability profiles.",
                "Recipe 2: Layout comparison on digits. Start with one homogeneous layout and one mixed layout, then inspect which classes stay uncertain.",
                "Recipe 3: SA from a homogeneous ReLU start. Use Playground or Builder SA mode and compare start layout, best layout, and end layout.",
                "Recipe 4: Multi-seed stability check. Run the same configuration with 5 or more seeds and compare mean and standard deviation before changing anything else.",
                "Recipe 5: test_activation as computation lab. Manually set inputs, choose simple hidden sizes, and inspect z, a, and local neuron details.",
            ]
        return [
            "Rezept 1: ReLU vs Tanh auf wine. Hidden-Sizes fixieren, mehrere Seeds laufen lassen und mittlere Validation-Accuracy plus Wahrscheinlichkeitsprofile vergleichen.",
            "Rezept 2: Layout-Vergleich auf digits. Starte mit einem homogenen und einem gemischten Layout und beobachte, welche Klassen unsicher bleiben.",
            "Rezept 3: SA von einem homogenen ReLU-Startlayout. Nutze Playground oder Builder-SA und vergleiche Startlayout, Bestlayout und Endlayout.",
            "Rezept 4: Multi-Seed-Stabilitaet. Fuehre dieselbe Konfiguration mit mindestens 5 Seeds aus und betrachte Mittelwert und Standardabweichung, bevor du weiter optimierst.",
            "Rezept 5: test_activation als Rechenlabor. Setze Inputs manuell, waehle kleine Hidden-Sizes und analysiere z, a und die lokale Neuron-Rechnung.",
        ]

    def _parameter_help_lines(self) -> list[str]:
        """Liefert ein kompaktes Lexikon der veraenderbaren GUI-Parameter."""

        if self._is_english():
            lines = [
                "Benchmark: selects dataset, number of classes, and the kind of input representation.",
                "Hidden layers: number of neurons in the hidden part of the network. More neurons or layers mean more capacity but also more visual complexity.",
                "Data split: determines which split the visible analysis sample comes from. Training itself still uses the training split.",
                "Sample index: selects the exact example you follow through the network.",
                "Use custom sample: allows manual inputs for digits and test_activation to study targeted reactions of the network.",
                "Analysis target: defines which target the visible loss is computed against.",
                "Set whole layer: applies one activation function to a complete hidden layer.",
                "Layer / Neuron / Activation / Set neuron: selects and changes one specific hidden neuron.",
                "Cycle: advances the selected neuron through relu -> tanh -> sigmoid -> leaky_relu.",
                "Workspace: demo focuses on one model and its behavior, playground focuses on simulated annealing over activation layouts.",
                "Mode: beginner hides complexity, expert shows deeper controls and comparison tools.",
            ]
            if self._app_mode() == "experiment_builder":
                lines.extend(
                    [
                        "Experiment name: identifier for one stored experiment folder and its manifest.",
                        "Run mode: manual_training trains one fixed layout, simulated_annealing optimizes the layout first.",
                        "Current layout: this is the builder start layout for the next run. In expert mode the layout editor changes it per layer or per neuron.",
                        "Current layouts: this list is only for preview and analysis of setup, start, best, or end layouts from a selected run.",
                        "Seed strategy: either build a consecutive seed list from count and start seed, or enter an explicit list.",
                        "Primary metric: drives ranking. Validation metrics are used for selection because test should remain final evaluation only.",
                        "Search type: none means exactly one configuration, grid search tests all discrete combinations, random search samples a reproducible subset.",
                        "Fixed/list/range: fixed means one value, list means explicit candidates, range means a discrete numeric sweep.",
                        "Builder output directory: where manifest, summary, and one JSON file per run are stored.",
                    ]
                )
            elif self._app_mode() == "playground":
                lines.extend(
                    [
                        "Objective: decides what is optimized, for example validation loss or validation accuracy.",
                        "Epochs per candidate: how long each layout candidate is trained before it is scored.",
                        "Learning rate / Batch size / Weight scale / Seed: define how every candidate is trained and keep candidate comparisons fair.",
                        "Neighborhoods: decide what kinds of local layout changes simulated annealing may propose.",
                        "Start temperature: initial openness to worse candidates.",
                        "Cooling schedule: rule for how temperature decreases over time.",
                        "Cooling parameter: strength of that decrease; its exact meaning depends on the chosen schedule.",
                        "Iterations per temperature: how many SA steps are executed before temperature is reduced once.",
                        "Maximum steps: hard upper bound for the run length.",
                        "Minimum temperature: stop threshold once the search has cooled down enough.",
                    ]
                )
            else:
                lines.extend(
                    [
                        "Epochs: number of full passes through the training split.",
                        "Learning rate: size of each learning step.",
                        "Batch size: number of training examples processed together per update.",
                        "Weight scale: magnitude of the random initial weights.",
                        "Seed: makes splits and initialization reproducible.",
                        "Reinitialize: resets the model with current settings and fresh initial weights.",
                        "1 epoch / 10 epochs / Train N epochs: stepwise training controls for observation.",
                        "Store as baseline: freezes a reference state for later comparison.",
                    ]
                )
            return lines

        lines = [
            "Benchmark: waehlt Datensatz und damit auch Art der Eingaben und Anzahl der Klassen.",
            "Hidden-Layer: Anzahl der Neuronen in den versteckten Schichten. Mehr Neuronen oder mehr Layer bedeuten mehr Kapazitaet, aber auch mehr Unuebersicht.",
            "Datensplit: bestimmt, aus welchem Split das aktuell sichtbare Analyse-Sample stammt. Das Training selbst verwendet weiterhin den Trainingssplit.",
            "Sample-Index: waehlt genau ein Beispiel aus, das du im Netz verfolgst.",
            "Eigenes Sample verwenden: erlaubt bei digits und test_activation manuelle Eingaben, um gezielt Reaktionen des Netzes zu studieren.",
            "Analyse-Ziel: setzt das Ziel, gegen das der Loss in der Ansicht berechnet wird. So kann man auch absichtlich ein 'falsches' Ziel untersuchen.",
            "Layer auf ...: setzt die Aktivierungsfunktion fuer einen kompletten Hidden-Layer auf einen Schlag.",
            "Layer / Neuron / Aktivierung / Neuron setzen: waehlt ein einzelnes Hidden-Neuron aus und aendert gezielt dessen Aktivierungsfunktion.",
            "Cycle: schaltet das ausgewaehlte Neuron in der festen Reihenfolge relu -> tanh -> sigmoid -> leaky_relu weiter.",
            "Arbeitsmodus: Demo beobachtet ein einzelnes Modell und sein Verhalten, Playground optimiert Aktivierungs-Layouts mit Simulated Annealing.",
            "Modus: Einsteiger blendet Komplexitaet aus, Experte zeigt tiefe Steuerung und Vergleichswerkzeuge.",
        ]
        if self._app_mode() == "experiment_builder":
            lines.extend(
                [
                    "Experiment-Name: Kennung fuer einen gespeicherten Experimentordner mit Manifest und JSON-Ergebnissen.",
                    "Run-Modus: manual_training trainiert ein fixes Layout, simulated_annealing optimiert zuerst das Layout.",
                    "Aktuelles Layout: Das ist das Startlayout des naechsten Builder-Runs. Im Expertenmodus kannst du es im Layout-Editor pro Layer oder pro Neuron bearbeiten.",
                    "Aktuelle Layouts: Diese Liste ist nur fuer Vorschau und Analyse von Setup-, Start-, Best- oder Endlayout eines selektierten Runs da.",
                    "Seed-Strategie: Entweder erzeugst du eine fortlaufende Seed-Liste aus Anzahl und Startwert oder gibst eine explizite Liste ein.",
                    "Primaere Metrik: steuert das Ranking. Fuer Auswahl wird Validation genutzt, Test bleibt die finale Bewertung.",
                    "Search-Typ: keine Suche = genau eine Konfiguration, Grid Search = alle diskreten Kombinationen, Random Search = reproduzierbare Teilmenge.",
                    "Fest / Liste / Bereich: fest bedeutet ein Wert, Liste explizite Kandidaten, Bereich einen diskreten Zahlen-Sweep.",
                    "Output-Ordner: Dort landen Manifest, Summary und pro Run genau eine JSON-Datei.",
                ]
            )
        elif self._app_mode() == "playground":
            lines.extend(
                [
                    "Zielmetrik: legt fest, was optimiert wird, zum Beispiel Validation-Loss oder Validation-Accuracy.",
                    "Epochen pro Kandidat: wie lange jedes Layout trainiert wird, bevor es bewertet wird.",
                    "Lernrate / Batch-Groesse / Weight-Scale / Seed: definieren, wie jeder Kandidat trainiert wird und halten Kandidatenvergleiche fair.",
                    "Nachbarschaften: bestimmen, welche lokalen Layout-Aenderungen Simulated Annealing vorschlagen darf.",
                    "Starttemperatur: anfaengliche Offenheit gegenueber schlechteren Kandidaten.",
                    "Abkuehlung: Regel, nach der die Temperatur ueber die Zeit sinkt.",
                    "Cooling-Parameter: Staerke dieses Abkuehlens; die genaue Bedeutung haengt vom gewaehlten Schedule ab.",
                    "Iterationen pro Temperatur: wie viele SA-Schritte pro Temperaturniveau ausgefuehrt werden.",
                    "Maximale Schritte: harte Obergrenze fuer die Laenge des Suchlaufs.",
                    "Mindesttemperatur: Stoppschwelle, sobald die Suche weit genug abgekuehlt ist.",
                ]
            )
        else:
            lines.extend(
                [
                    "Epochen: wie oft das Training den gesamten Trainingssplit durchlaeuft.",
                    "Lernrate: Schrittweite des Lernens. Zu klein lernt langsam, zu gross kann instabil werden.",
                    "Batch-Groesse: wie viele Trainingsbeispiele pro Gewichtsupdate gemeinsam verarbeitet werden.",
                    "Weight-Scale: Groessenordnung der zufaelligen Startgewichte. Beeinflusst, wie stark Aktivierungen schon zu Beginn ausschlagen.",
                    "Seed: sorgt fuer reproduzierbare Daten-Splits und reproduzierbare Initialisierung.",
                    "Neu initialisieren: setzt das Modell mit aktuellen Einstellungen und neuen Startgewichten zurueck.",
                    "1 Epoche / 10 Epochen / N Epochen trainieren: trainiert schrittweise, damit man Veraenderungen beobachten kann.",
                    "Als Baseline speichern: friert einen Referenzzustand ein, mit dem das aktuelle Experiment spaeter verglichen wird.",
                ]
            )
        return lines

    def _program_guide_text(self) -> str:
        """Liefert eine einfache A-Z-Anleitung fuer das gesamte Programm."""

        if self._is_english():
            return "\n".join(
                [
                    "Activation Playground Guide",
                    "==========================",
                    "",
                    "What this program is",
                    "--------------------",
                    "This program is a compact playground for small neural networks with editable activation layouts.",
                    "It allows you to inspect how architecture, activations, inputs, training, and optimization interact.",
                    "",
                    "Three workspaces",
                    "----------------",
                    "Demo Mode is for understanding a single network in detail: input, prediction, loss, hidden activations, and training behavior.",
                    "Playground Mode uses the same benchmark and model setup but adds simulated annealing over activation layouts.",
                    "Experiment Builder is for reproducible multi-seed experiments, stored JSON results, and later re-analysis.",
                    "The workspace setting therefore changes the main goal of the interface, not only its level of detail.",
                    "",
                    "Main idea",
                    "---------",
                    "A network consists of inputs, one to four hidden layers, and an output layer.",
                    "Each hidden neuron applies one activation function: relu, tanh, sigmoid, or leaky_relu.",
                    "You can assign activations per full layer or per individual neuron.",
                    "",
                    "How to work with Demo Mode",
                    "--------------------------",
                    "1. Choose a benchmark.",
                    "2. Choose how many hidden layers and neurons you want.",
                    "3. Inspect the current sample in the input view.",
                    "4. Change activations and watch the network view update.",
                    "5. Train the model step by step.",
                    "6. Compare prediction, loss, and hidden activations before and after training.",
                    "7. Click a hidden neuron to inspect its local formula.",
                    "",
                    "How to work with Playground Mode",
                    "--------------------------------",
                    "1. Choose a benchmark, hidden layers, and a start layout.",
                    "2. Choose an objective: validation loss or validation accuracy.",
                    "3. Decide which neighborhood moves are allowed.",
                    "4. Set temperature, cooling rule, candidate epochs, and stop limits.",
                    "5. Evaluate the start state.",
                    "6. Run simulated annealing step by step or to completion.",
                    "7. Compare current state, candidate state, and best state.",
                    "",
                    "How to work with Experiment Builder",
                    "-----------------------------------",
                    "1. Choose benchmark, hidden layers, layout, and run mode.",
                    "2. Decide whether you want one configuration, grid search, or random search.",
                    "3. Define a seed strategy and inspect the effective seed list.",
                    "4. In expert mode, edit the builder start layout per layer or per neuron.",
                    "5. Run the experiment and inspect ranking, multi-seed aggregation, and per-seed detail.",
                    "",
                    "Important builder concepts",
                    "--------------------------",
                    "Sample = one visible example used for inspection on the right side.",
                    "Seed = one reproducible random setup for initialization and comparison.",
                    "Run = one concrete execution for exactly one configuration and one seed.",
                    "Configuration = one fixed parameter setting produced by the search space.",
                    "Experiment = the full builder definition including benchmark, layout, seeds, run mode, and optional search.",
                    "",
                    "What the left side does",
                    "-----------------------",
                    "The left side is the control strip. It defines dataset, sample, layout, and either training or annealing settings.",
                    "Beginner mode keeps the essential controls visible.",
                    "Expert mode adds layer changes, neuron-level edits, and comparison tools.",
                    "",
                    "What the right side does",
                    "------------------------",
                    "The right side shows the live network, analysis tabs, plots, and the local computation of the selected neuron.",
                    "The colored hidden nodes show which activation function is used.",
                    "",
                    "What training changes",
                    "---------------------",
                    "Training changes weights and biases.",
                    "It does not change the dataset and it does not directly change the activation layout.",
                    "The displayed sample is only an inspection window.",
                    "",
                    "What simulated annealing changes",
                    "--------------------------------",
                    "Simulated annealing does not directly optimize weights. It optimizes the activation layout.",
                    "Each candidate layout is trained briefly from scratch and then scored on validation data.",
                    "A better candidate is accepted directly. A worse candidate may still be accepted while the temperature is high.",
                    "Start layout = the configured layout before search begins.",
                    "Best layout = the best state found so far.",
                    "End layout = the layout where the run finally stops.",
                    "",
                    "Why validation instead of test?",
                    "-------------------------------",
                    "Validation is used for ranking and model selection because test should remain the final unseen evaluation split.",
                    "Selecting by test metrics would leak final evaluation information into the search.",
                    "",
                    "What the test_activation mode is for",
                    "------------------------------------",
                    "test_activation is the smallest learning lab in the project.",
                    "It is useful when you want to understand raw forward computations without a complex benchmark.",
                    "",
                    "What the compare tab is for",
                    "---------------------------",
                    "Store one state as a baseline.",
                    "Then modify the model or train it further and compare layout, prediction, and metrics against that baseline.",
                    "",
                    "What the Forward/Backward tab is for",
                    "------------------------------------",
                    "This tab breaks one visible sample into the main steps of the forward pass and the backward pass.",
                    "It helps you see what happens mathematically inside the network.",
                    "",
                    "What the Annealing tab is for",
                    "-----------------------------",
                    "This tab is available in Playground Mode.",
                    "It shows the SA history, current state, candidate, best state, score development, temperature, and acceptance behavior.",
                    "",
                    "What comes later",
                    "----------------",
                    "The current codebase already contains layout operations and neighbor generation.",
                    "This prepares the project for later search and optimization methods such as simulated annealing.",
                    "",
                    "Useful first experiments",
                    "------------------------",
                    *self._experiment_recipe_lines(),
                ]
            )
        return "\n".join(
            [
                "Anleitung zum Activation Playground",
                "===================================",
                "",
                "Was dieses Programm ist",
                "-----------------------",
                "Dieses Programm ist ein kompakter Playground fuer kleine neuronale Netze mit veraenderbaren Aktivierungs-Layouts.",
                "Es zeigt, wie Architektur, Aktivierungen, Eingaben, Training und Optimierung zusammenwirken.",
                "",
                "Drei Arbeitsmodi",
                "----------------",
                "Demo Mode dient dazu, ein einzelnes Netz im Detail zu verstehen: Eingabe, Vorhersage, Loss, Hidden-Aktivierungen und Trainingsverhalten.",
                "Playground Mode nutzt dieselbe Benchmark- und Modellbasis, erweitert sie aber um Simulated Annealing ueber Aktivierungs-Layouts.",
                "Experiment Builder dient reproduzierbaren Multi-Seed-Experimenten, gespeicherten JSON-Ergebnissen und spaeterer Wiederanalyse.",
                "Der Arbeitsmodus aendert also das Hauptziel der Oberflaeche und nicht nur den Detailgrad.",
                "",
                "Grundidee",
                "---------",
                "Ein Netz besteht aus Eingaben, ein bis vier Hidden-Layern und einem Output-Layer.",
                "Jedes Hidden-Neuron benutzt eine Aktivierungsfunktion: relu, tanh, sigmoid oder leaky_relu.",
                "Du kannst Aktivierungen fuer ganze Layer oder fuer einzelne Neuronen setzen.",
                "",
                "So arbeitest du im Demo Mode",
                "----------------------------",
                "1. Waehl einen Benchmark.",
                "2. Waehl Anzahl und Groesse der Hidden-Layer.",
                "3. Schau dir im Input-Tab das aktuelle Sample an.",
                "4. Veraendere Aktivierungen und beobachte die Netzansicht.",
                "5. Trainiere das Modell schrittweise.",
                "6. Vergleiche Vorhersage, Loss und Hidden-Aktivierungen vor und nach dem Training.",
                "7. Klick auf ein Hidden-Neuron und lies seine lokale Rechnung.",
                "",
                "So arbeitest du im Playground Mode",
                "----------------------------------",
                "1. Waehle Benchmark, Hidden-Layer und ein Startlayout.",
                "2. Lege eine Zielmetrik fest: Validation-Loss oder Validation-Accuracy.",
                "3. Bestimme erlaubte Nachbarschaftsoperationen.",
                "4. Setze Temperatur, Abkuehlregel, Kandidaten-Epochen und Stopplimits.",
                "5. Bewerte den Startzustand.",
                "6. Fuehre Simulated Annealing schrittweise oder komplett aus.",
                "7. Vergleiche aktuellen Zustand, Kandidat und bestes Layout.",
                "",
                "So arbeitest du im Experiment Builder",
                "-------------------------------------",
                "1. Waehle Benchmark, Hidden-Layer, Layout und Run-Modus.",
                "2. Entscheide, ob du eine Einzelkonfiguration, Grid Search oder Random Search willst.",
                "3. Lege eine Seed-Strategie fest und pruefe die effektive Seed-Liste.",
                "4. Bearbeite im Expertenmodus das Builder-Startlayout pro Layer oder pro Neuron.",
                "5. Fuehre das Experiment aus und analysiere Ranking, Multi-Seed-Aggregation und Per-Seed-Details.",
                "",
                "Wichtige Builder-Begriffe",
                "-------------------------",
                "Sample = ein sichtbares Beispiel, das rechts zur Analyse gezeigt wird.",
                "Seed = eine reproduzierbare Zufallskonfiguration fuer Initialisierung und Vergleich.",
                "Run = eine konkrete Ausfuehrung fuer genau eine Konfiguration und genau einen Seed.",
                "Konfiguration = eine feste Parameterbelegung aus dem Suchraum.",
                "Experiment = die komplette Builder-Definition aus Benchmark, Layout, Seeds, Run-Modus und optionaler Suche.",
                "",
                "Was die linke Seite macht",
                "-------------------------",
                "Die linke Seite ist die Steuerleiste. Hier stellst du Datensatz, Sample, Layout und je nach Arbeitsmodus Training oder Annealing ein.",
                "Im Einsteiger-Modus siehst du nur die wichtigsten Steuerungen.",
                "Im Experten-Modus kommen Layer-Aenderungen, Neuron-Eingriffe und Vergleichswerkzeuge dazu.",
                "",
                "Was die rechte Seite macht",
                "--------------------------",
                "Rechts siehst du das Netz live, die Analyse-Tabs, die Plots und die lokale Rechnung des ausgewaehlten Neurons.",
                "Die farbigen Hidden-Knoten zeigen, welche Aktivierungsfunktion verwendet wird.",
                "",
                "Was Training aendert",
                "--------------------",
                "Training aendert Gewichte und Biases.",
                "Es aendert nicht den Datensatz und nicht direkt das Aktivierungs-Layout.",
                "Das sichtbare Sample ist nur ein Analysefenster.",
                "",
                "Was Simulated Annealing aendert",
                "--------------------------------",
                "Simulated Annealing optimiert nicht direkt die Gewichte, sondern das Aktivierungs-Layout.",
                "Jedes Kandidaten-Layout wird kurz frisch trainiert und danach ueber Validation-Daten bewertet.",
                "Ein besserer Kandidat wird direkt akzeptiert. Ein schlechterer kann bei hoher Temperatur trotzdem noch angenommen werden.",
                "Startlayout = das konfigurierte Layout vor Beginn der Suche.",
                "Bestlayout = der bisher beste gefundene Zustand.",
                "Endlayout = das Layout, bei dem der Lauf schliesslich stoppt.",
                "",
                "Warum Validation statt Test?",
                "----------------------------",
                "Validation wird fuer Ranking und Modellauswahl benutzt, weil Test die finale ungesehene Bewertungsmenge bleiben soll.",
                "Wer nach Testmetriken auswaehlt, mischt Endbewertung und Suche unguenstig miteinander.",
                "",
                "Wofuer test_activation da ist",
                "-----------------------------",
                "test_activation ist das kleinste Rechenlabor im Projekt.",
                "Es ist ideal, wenn du rohe Vorwaertsrechnungen ohne komplexen Benchmark verstehen willst.",
                "",
                "Wofuer der Vergleichs-Tab da ist",
                "--------------------------------",
                "Speichere einen Zustand als Baseline.",
                "Danach kannst du das Modell veraendern oder weiter trainieren und Layout, Vorhersage und Metriken dagegen vergleichen.",
                "",
                "Wofuer der Forward/Backward-Tab da ist",
                "--------------------------------------",
                "Dieser Tab zerlegt ein einzelnes sichtbares Sample in die Hauptschritte des Forward- und Backward-Passes.",
                "Damit wird sichtbar, was mathematisch im Netz passiert.",
                "",
                "Wofuer der Annealing-Tab da ist",
                "-------------------------------",
                "Dieser Tab ist im Playground Mode sichtbar.",
                "Er zeigt den SA-Verlauf, aktuellen Zustand, Kandidaten, besten Zustand, Score-Entwicklung, Temperatur und Akzeptanzverhalten.",
                "",
                "Was spaeter darauf aufbauen kann",
                "--------------------------------",
                "Die aktuelle Codebasis enthaelt bereits Layout-Operationen und Neighbor-Generierung.",
                "Damit ist das Projekt vorbereitet fuer spaetere Such- und Optimierungsverfahren wie Simulated Annealing.",
                "",
                "Sinnvolle erste Experimente",
                "---------------------------",
                *self._experiment_recipe_lines(),
            ]
        )

    def _redraw_network(self) -> None:
        """Zeichnet das aktuelle Netzwerk als interaktive Canvas-Grafik."""

        if self.model is None or self.dataset is None or self.current_layout is None:
            return

        analysis_sample = self._get_analysis_sample()
        cache = self.model.forward(analysis_sample.scaled_sample.reshape(1, -1))
        probabilities = cache.probabilities[0]
        predicted_index = int(np.argmax(probabilities))

        width = max(int(self.canvas.winfo_width()), 280)
        height = max(int(self.canvas.winfo_height()), 240)
        self._network_draw_width = width
        self._network_draw_height = height
        self.canvas.delete("all")
        self.node_tags.clear()

        compact_view = width < 900
        padding_top = 58 if compact_view else 70
        padding_bottom = 54 if compact_view else 70
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

        if self._is_english():
            self.network_summary_var.set(
                f"Current sample: {analysis_sample.source_label} | "
                f"True target: {analysis_sample.actual_target_name} | "
                f"Analysis target: {analysis_sample.effective_target_name} | "
                f"Prediction: {self.dataset.target_names[predicted_index]}"
            )
        else:
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

        width = max(self._network_draw_width, 280)
        compact_view = width < 900
        title_font = ("Helvetica", 10 if compact_view else 12, "bold")
        summary_font = ("Helvetica", 8 if compact_view else 10)
        summary_y = 46 if compact_view else 52

        self.canvas.create_text(
            column_positions[0],
            28,
            text=(
                f"Eingaben ({len(input_indices)}/{self.dataset.input_size})"
                if not self._is_english()
                else f"Inputs ({len(input_indices)}/{self.dataset.input_size})"
            ),
            font=title_font,
            fill=NETWORK_TEXT_COLOR,
        )
        for layer_index, layer in enumerate(self.current_layout.layers):
            self.canvas.create_text(
                column_positions[layer_index + 1],
                28,
                text=f"Hidden L{layer_index + 1} ({len(layer)})",
                font=title_font,
                fill=NETWORK_TEXT_COLOR,
            )
        self.canvas.create_text(
            column_positions[-1],
            28,
            text=f"Output ({self.dataset.output_size})",
            font=title_font,
            fill=NETWORK_TEXT_COLOR,
        )

        self.canvas.create_text(
            (column_positions[0] + column_positions[-1]) / 2.0,
            summary_y,
            text=(
                (
                    f"Sample: {analysis_sample.source_label} | "
                    f"Vorhersage: {self.dataset.target_names[predicted_index]} | "
                    f"Analyse-Ziel: {analysis_sample.effective_target_name}"
                )
                if not self._is_english()
                else (
                    f"Sample: {analysis_sample.source_label} | "
                    f"Prediction: {self.dataset.target_names[predicted_index]} | "
                    f"Analysis target: {analysis_sample.effective_target_name}"
                )
            ),
            font=summary_font,
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
        node_font = ("Helvetica", 8 if radius < 15 else 9, "bold")
        value_font = ("Helvetica", 7 if radius < 15 else 8)
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
                font=node_font,
                fill=NETWORK_TEXT_COLOR,
            )
            self.canvas.create_text(
                x_coord,
                y_coord + radius + 9,
                text=f"roh={raw_value:+.2f}" if not self._is_english() else f"raw={raw_value:+.2f}",
                font=value_font,
                fill=NETWORK_TEXT_COLOR,
            )

    def _draw_hidden_nodes(self, layer_index: int, positions: Sequence[tuple[float, float]]) -> None:
        """Zeichnet Hidden-Neuronen inklusive Klick-Handler."""

        radius = self._node_radius(len(positions))
        node_font = ("Helvetica", 8 if radius < 15 else 9, "bold")
        label_font = ("Helvetica", 7 if radius < 15 else 8)
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
            self.canvas.create_text(x_coord, y_coord - 2, text=str(neuron_index), font=node_font, fill="white", tags=(tag,))
            self.canvas.create_text(
                x_coord,
                y_coord + radius + 10,
                text=activation_name,
                font=label_font,
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
        node_font = ("Helvetica", 8 if radius < 15 else 9, "bold")
        label_font = ("Helvetica", 7 if radius < 15 else 8)
        for output_index, (x_coord, y_coord) in enumerate(positions):
            outline = "#0f172a"
            width = 1.5
            if output_index == predicted_index:
                outline = PREDICTION_COLOR
                width = 3
            if analysis_sample.actual_target_index is not None and output_index == analysis_sample.actual_target_index:
                self.canvas.create_text(
                    x_coord,
                    y_coord - radius - 14,
                    text="echt" if not self._is_english() else "true",
                    fill=TARGET_COLOR,
                    font=("Helvetica", 7 if radius < 15 else 8, "bold"),
                )
            if analysis_sample.effective_target_index is not None and output_index == analysis_sample.effective_target_index:
                self.canvas.create_text(
                    x_coord,
                    y_coord - radius - 2,
                    text="analyse" if not self._is_english() else "analysis",
                    fill=ANALYSIS_TARGET_COLOR,
                    font=("Helvetica", 7 if radius < 15 else 8, "bold"),
                )

            self.canvas.create_oval(
                x_coord - radius,
                y_coord - radius,
                x_coord + radius,
                y_coord + radius,
                fill=OUTPUT_NODE_COLOR,
                outline=outline,
                width=width,
            )
            self.canvas.create_text(x_coord, y_coord - 2, text=str(output_index), font=node_font, fill="white")
            self.canvas.create_text(
                x_coord,
                y_coord + radius + 10,
                text=f"{self.dataset.target_names[output_index]} ({probabilities[output_index]:.2f})",
                font=label_font,
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

        compact_view = width < 950
        legend_y = height - (18 if compact_view else 26)
        x_coord = 20 if compact_view else 36
        legend_step = max(78, min(140, int((width - 40) / 6)))
        legend_font = ("Helvetica", 8 if compact_view else 9)
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
                font=legend_font,
                fill=NETWORK_TEXT_COLOR,
            )
            x_coord += legend_step
        legend_explanation = (
            "Gruen = Prediction | orange Text = echtes Ziel | blaugruener Text = Analyse-Ziel"
            if not self._is_english()
            else "Green = prediction | orange text = true target | teal text = analysis target"
        )
        if compact_view:
            self.canvas.create_text(
                width / 2.0,
                max(legend_y - 18, 20),
                text=legend_explanation,
                font=("Helvetica", 7),
                fill=NETWORK_TEXT_COLOR,
                anchor="center",
            )
        else:
            self.canvas.create_text(
                width - 12,
                legend_y,
                text=legend_explanation,
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

        left = 36.0 if width < 900 else 70.0
        right = max(float(width) - (36.0 if width < 900 else 70.0), left + 120.0)
        return list(np.linspace(left, right, num=num_hidden_layers + 2))

    def _node_radius(self, count: int) -> float:
        """Waehlt einen lesbaren Radius abhaengig von der Knotenzahl."""

        width = max(self._network_draw_width, 280)
        height = max(self._network_draw_height, 240)
        compact_view = width < 900 or height < 520
        if count <= 8:
            base_radius = 20
        elif count <= 16:
            base_radius = 16
        elif count <= 32:
            base_radius = 12
        else:
            base_radius = 9
        if compact_view:
            return max(7.0, base_radius - 3.0)
        if count <= 16:
            return base_radius
        return base_radius

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
        editable_layout = self._editable_layout()
        if editable_layout is not None:
            self.neuron_activation_var.set(editable_layout.layers[layer_index][neuron_index])
        self._refresh_views()
