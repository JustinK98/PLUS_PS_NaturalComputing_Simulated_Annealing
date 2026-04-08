"""Haupteinstieg des Aktivierungsfunktions-Playgrounds.

Diese Datei verbindet alle anderen Module:

- Parser und interaktive Eingaben
- Datensatz laden
- Layout parsen und optional veraendern
- Modell bauen
- Training starten
- Ergebnisse im Terminal und optional als Plot ausgeben

Wichtige Benutzungsmodi:

1. Interaktiv:
   `python main.py`
   Dann startet ein gefuehrter Assistent mit Erklaerungen.

2. Klassische CLI:
   `python main.py --benchmark wine --layout "relu|tanh" --epochs 100`
   Praktisch fuer reproduzierbare Experimente oder spaetere Skripte.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from configs import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BENCHMARK,
    DEFAULT_EPOCHS,
    DEFAULT_GUI_APP_MODE,
    DEFAULT_LAYOUT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NEIGHBOR_PREVIEW,
    DEFAULT_RANDOM_SEED,
    DEFAULT_WEIGHT_SCALE,
    LAYOUT_SYNTAX_EXAMPLES,
    NEIGHBOR_OPERATION_EXAMPLES,
    DatasetConfig,
    MAX_HIDDEN_LAYERS,
    MIN_HIDDEN_LAYERS,
    ModelConfig,
    SUPPORTED_ACTIVATIONS,
    SUPPORTED_BENCHMARKS,
    SUPPORTED_GUI_APP_MODES,
    SUPPORTED_GUI_LANGUAGES,
    SUPPORTED_GUI_MODES,
    TrainingConfig,
    VisualizationConfig,
    default_hidden_sizes,
    format_hidden_sizes,
)


def build_parser() -> argparse.ArgumentParser:
    """Erzeugt den Kommandozeilen-Parser.

    Der Parser bleibt auch im interaktiven Modus relevant:
    Die interaktiven Antworten schreiben am Ende einfach in dieselben Felder,
    die sonst von CLI-Argumenten befuellt werden.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Aktivierungsfunktions-Playground fuer kleine MLPs mit variabler Hidden-Layer-Anzahl."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=f"{LAYOUT_SYNTAX_EXAMPLES}\n{NEIGHBOR_OPERATION_EXAMPLES}",
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Starte den gefuehrten, interaktiven Experiment-Assistenten.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Starte die didaktische GUI statt des reinen Terminal-Modus.",
    )
    parser.add_argument(
        "--gui-app-mode",
        choices=SUPPORTED_GUI_APP_MODES,
        default=DEFAULT_GUI_APP_MODE,
        help=(
            "Arbeitsmodus der GUI: demo fuer Verstehen/Visualisieren, "
            "playground fuer Simulated Annealing, "
            "experiment_builder fuer Multi-Seed-Experimente und Suchlaeufe."
        ),
    )
    parser.add_argument(
        "--gui-mode",
        choices=SUPPORTED_GUI_MODES,
        default="beginner",
        help="GUI-Modus: beginner reduziert Komplexitaet, expert zeigt alle Werkzeuge.",
    )
    parser.add_argument(
        "--gui-language",
        choices=SUPPORTED_GUI_LANGUAGES,
        default="de",
        help="Startsprache der GUI: de fuer Deutsch, en fuer Englisch.",
    )
    parser.add_argument(
        "--benchmark",
        choices=SUPPORTED_BENCHMARKS,
        default=DEFAULT_BENCHMARK,
        help="Datensatz: breast_cancer, wine oder digits.",
    )
    parser.add_argument(
        "--hidden-sizes",
        nargs="+",
        type=int,
        metavar="H",
        help="Neuronenzahlen fuer die Hidden-Layer, z. B. --hidden-sizes 16 8 oder 24 12 6.",
    )
    parser.add_argument(
        "--layout",
        default=DEFAULT_LAYOUT,
        help="Layout-Syntax, z. B. 'relu|tanh', 'relu|tanh|sigmoid' oder 'relu*16|sigmoid*8'.",
    )
    parser.add_argument(
        "--neighbor-op",
        action="append",
        default=[],
        help="Neighbor-Operation, mehrfach angebbar. Beispiel: set:L1:0:tanh",
    )
    parser.add_argument(
        "--show-neighbors",
        type=int,
        default=DEFAULT_NEIGHBOR_PREVIEW,
        help="Wie viele Single-Step-Neighbors im Terminal gezeigt werden sollen.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Anzahl der Trainingsepochen.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Lernrate fuer den einfachen Gradient Descent.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch-Groesse fuer das Training.",
    )
    parser.add_argument(
        "--weight-scale",
        type=float,
        default=DEFAULT_WEIGHT_SCALE,
        help="Skalierung der zufaelligen Startgewichte.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Zufalls-Seed fuer reproduzierbare Experimente.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Keine Matplotlib-Ausgabe erzeugen.",
    )
    parser.add_argument(
        "--save-prefix",
        type=str,
        default=None,
        help="Speichert Plots als '<prefix>_history.png' und '<prefix>_layouts.png'.",
    )
    parser.add_argument(
        "--show-layout-help",
        action="store_true",
        help="Zeigt Erklaerungen zur Layout- und Neighbor-Syntax und beendet das Programm.",
    )

    return parser


def main() -> None:
    """Programmstart.

    Wenn keine CLI-Argumente angegeben wurden, startet automatisch der
    interaktive Assistent. Das ist fuer ein didaktisches Lernwerkzeug die
    benutzerfreundlichste Default-Variante.
    """

    parser = build_parser()
    args = parser.parse_args()

    if args.show_layout_help:
        print(LAYOUT_SYNTAX_EXAMPLES)
        print(NEIGHBOR_OPERATION_EXAMPLES)
        return

    if _should_run_interactive(args):
        args = _run_interactive_setup(parser, args)

    if args.gui:
        _launch_gui(args)
        return

    try:
        from activations import (
            apply_neighbor_operations,
            generate_single_step_neighbors,
            parse_layout_spec,
        )
        from benchmarks import load_benchmark
        from model import ModularMLP
        from terminal_viz import (
            render_dataset_summary,
            render_layout,
            render_layout_diff,
            render_neighbor_preview,
            render_training_summary,
        )
        from trainer import train_model

        hidden_sizes = (
            tuple(args.hidden_sizes) if args.hidden_sizes else default_hidden_sizes(args.benchmark)
        )

        dataset_config = DatasetConfig(name=args.benchmark, random_state=args.seed)
        model_config = ModelConfig(
            input_size=0,
            hidden_sizes=hidden_sizes,
            output_size=0,
            layout_spec=args.layout,
            weight_scale=args.weight_scale,
            random_state=args.seed,
        )
        visualization_config = VisualizationConfig(
            preview_neighbors=args.show_neighbors,
            show_plots=not args.no_plot,
            save_prefix=args.save_prefix,
        )

        dataset = load_benchmark(dataset_config)
        base_layout = parse_layout_spec(model_config.layout_spec, model_config.hidden_sizes)

        neighbor_result = None
        training_layout = base_layout
        if args.neighbor_op:
            neighbor_result = apply_neighbor_operations(base_layout, args.neighbor_op)
            training_layout = neighbor_result.layout

        print(render_dataset_summary(dataset))
        print()
        print(render_layout(base_layout, title="Basis-Layout"))

        if neighbor_result is not None:
            print()
            print(render_layout(neighbor_result.layout, title="Nachbar-Layout"))
            print()
            print(render_layout_diff(base_layout, neighbor_result.layout))

        if visualization_config.preview_neighbors > 0:
            neighbors = generate_single_step_neighbors(training_layout)
            print()
            print(render_neighbor_preview(neighbors, visualization_config.preview_neighbors))

        print()
        print(f"Trainiertes Layout: {training_layout.to_compact_spec()}")

        model = ModularMLP(
            input_size=dataset.input_size,
            hidden_sizes=model_config.hidden_sizes,
            output_size=dataset.output_size,
            layout=training_layout,
            weight_scale=model_config.weight_scale,
            random_state=model_config.random_state,
        )
        training_config = TrainingConfig(
            epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            random_state=args.seed,
        )
        result = train_model(model, dataset, training_config)

        print()
        print(render_training_summary(result.history, result.test_metrics))

        if visualization_config.show_plots or visualization_config.save_prefix:
            _show_or_save_plots(
                benchmark_name=dataset.name,
                base_layout=base_layout,
                neighbor_layout=neighbor_result.layout if neighbor_result else None,
                history=result.history,
                visualization_config=visualization_config,
            )

    except ImportError as exc:
        parser.exit(1, f"{exc}\n")
    except ValueError as exc:
        parser.error(str(exc))


def _should_run_interactive(args: argparse.Namespace) -> bool:
    """Entscheidet, ob der interaktive Assistent gestartet werden soll.

    Regeln:
    - `--interactive` erzwingt den Assistenten
    - ganz ohne Zusatzargumente startet der Assistent automatisch
    """

    return args.interactive or len(sys.argv) == 1


def _run_interactive_setup(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> argparse.Namespace:
    """Fuehrt Benutzer Schritt fuer Schritt durch ein Experiment.

    Die Eingaben sind bewusst nicht zu technisch formuliert. Ziel ist, dass auch
    unerfahrene Studierende verstehen, welche Entscheidung welchen Effekt hat.
    """

    print("Interaktiver Experiment-Assistent")
    print("=================================")
    print("Dieses Programm hilft dir, ein kleines MLP-Experiment aufzubauen.")
    print("Leere Eingaben uebernehmen jeweils den vorgeschlagenen Standardwert.")
    print()

    args.benchmark = _prompt_choice(
        label="Benchmark",
        explanation=(
            "Waehle den Datensatz, auf dem trainiert werden soll.\n"
            "- breast_cancer: kleiner binaerer Klassifikationsdatensatz\n"
            "- wine: kleiner Mehrklassen-Datensatz, oft gut fuer erste Tests\n"
            "- digits: groesserer Bild-/Merkmalsdatensatz mit 10 Klassen\n"
            "- test_activation: kleines Rechenlabor mit wenigen Inputs fuer rohe Experimente"
        ),
        options=SUPPORTED_BENCHMARKS,
        default=args.benchmark,
    )

    use_default_hidden_sizes = _prompt_yes_no(
        label="Standard-Hidden-Sizes verwenden",
        explanation=(
            "Die Hidden-Sizes bestimmen, wie viele Neuronen in den versteckten "
            "Schichten liegen. Mehr Neuronen bedeuten mehr Kapazitaet, aber "
            "auch mehr Parameter."
        ),
        default=args.hidden_sizes is None,
    )
    if use_default_hidden_sizes:
        args.hidden_sizes = None
    else:
        default_hidden_sizes_for_benchmark = default_hidden_sizes(args.benchmark)
        layer_count = _prompt_int(
            label="Anzahl Hidden-Layer",
            explanation=(
                f"Fuer die GUI und das didaktische Modell sind {MIN_HIDDEN_LAYERS} bis "
                f"{MAX_HIDDEN_LAYERS} Hidden-Layer vorgesehen."
            ),
            default=len(default_hidden_sizes_for_benchmark),
            min_value=MIN_HIDDEN_LAYERS,
            max_value=MAX_HIDDEN_LAYERS,
        )
        args.hidden_sizes = []
        for layer_index in range(layer_count):
            default_size = (
                default_hidden_sizes_for_benchmark[layer_index]
                if layer_index < len(default_hidden_sizes_for_benchmark)
                else default_hidden_sizes_for_benchmark[-1]
            )
            args.hidden_sizes.append(
                _prompt_int(
                    label=f"Neuronen in Hidden-Layer {layer_index + 1}",
                    explanation=f"Empfohlener Startwert fuer {args.benchmark}: {default_size}",
                    default=default_size,
                    min_value=1,
                )
            )

    hidden_sizes = (
        tuple(args.hidden_sizes) if args.hidden_sizes else default_hidden_sizes(args.benchmark)
    )

    layout_mode = _prompt_choice(
        label="Layout-Modus",
        explanation=(
            "Du kannst das Aktivierungs-Layout auf zwei Arten festlegen:\n"
            "- layerweise: pro Hidden-Layer genau eine Aktivierung waehlen\n"
            "- direkt: einen kompletten Layout-String selbst eingeben"
        ),
        options=("layerweise", "direkt"),
        default="layerweise",
    )
    if layout_mode == "layerweise":
        args.layout = _build_layerwise_layout_from_prompt(len(hidden_sizes))
    else:
        args.layout = _prompt_layout_string(hidden_sizes, default=args.layout)

    use_neighbor_ops = _prompt_yes_no(
        label="Neighbor-Operationen anwenden",
        explanation=(
            "Neighbor-Operationen veraendern das Basis-Layout lokal.\n"
            "Das ist didaktisch interessant und spaeter direkt fuer "
            "Simulated Annealing relevant."
        ),
        default=bool(args.neighbor_op),
    )
    if use_neighbor_ops:
        args.neighbor_op = _prompt_neighbor_operations()
    else:
        args.neighbor_op = []

    show_neighbors = _prompt_yes_no(
        label="Single-Step-Neighbors im Terminal zeigen",
        explanation=(
            "Dabei wird gezeigt, welche direkten Nachbarn das aktuelle Layout hat.\n"
            "So sieht man den Suchraum rund um das aktuelle Experiment."
        ),
        default=args.show_neighbors > 0,
    )
    if show_neighbors:
        args.show_neighbors = _prompt_int(
            label="Wie viele Nachbarn anzeigen",
            explanation="Groessere Werte zeigen mehr Varianten, machen die Ausgabe aber laenger.",
            default=args.show_neighbors,
            min_value=0,
        )
    else:
        args.show_neighbors = 0

    use_default_training = _prompt_yes_no(
        label="Standard-Trainingswerte verwenden",
        explanation=(
            "Die Standardwerte sind ein guter Start. Wenn du lernen willst, wie "
            "Hyperparameter das Verhalten beeinflussen, kannst du sie hier aendern."
        ),
        default=(
            args.epochs == DEFAULT_EPOCHS
            and args.lr == DEFAULT_LEARNING_RATE
            and args.batch_size == DEFAULT_BATCH_SIZE
        ),
    )
    if not use_default_training:
        args.epochs = _prompt_int(
            label="Epochen",
            explanation="Mehr Epochen bedeuten laengeres Training, oft aber bessere Anpassung.",
            default=args.epochs,
            min_value=1,
        )
        args.lr = _prompt_float(
            label="Lernrate",
            explanation=(
                "Die Lernrate bestimmt, wie gross jeder Trainingsschritt ist. "
                "Zu klein = langsam, zu gross = instabil."
            ),
            default=args.lr,
            min_value=1e-8,
        )
        args.batch_size = _prompt_int(
            label="Batch-Groesse",
            explanation=(
                "Die Batch-Groesse bestimmt, wie viele Beispiele pro Update genutzt werden."
            ),
            default=args.batch_size,
            min_value=1,
        )

    use_advanced_settings = _prompt_yes_no(
        label="Erweiterte Einstellungen anpassen",
        explanation=(
            "Hier koennen Seed und Gewichtsskala angepasst werden.\n"
            "Das ist eher fuer bewusstere Experimente interessant."
        ),
        default=False,
    )
    if use_advanced_settings:
        args.weight_scale = _prompt_float(
            label="Gewichtsskala",
            explanation=(
                "Steuert, wie gross die Startgewichte zufaellig initialisiert werden."
            ),
            default=args.weight_scale,
            min_value=1e-8,
        )
        args.seed = _prompt_int(
            label="Random Seed",
            explanation=(
                "Mit demselben Seed bleiben Daten-Splits und Initialisierungen reproduzierbar."
            ),
            default=args.seed,
            min_value=0,
        )

    args.gui = _prompt_yes_no(
        label="GUI starten",
        explanation=(
            "Die GUI zeigt das Netzwerk mit Kreisen und Verbindungen, erlaubt Klicks "
            "auf Neuronen und zeigt lokale Berechnungen eines ausgewaehlten Neurons."
        ),
        default=args.gui,
    )

    if args.gui:
        args.gui_app_mode = _prompt_choice(
            label="GUI-Arbeitsmodus",
            explanation=(
                "demo ist fuer Verstehen, Visualisieren und schrittweises Training gedacht.\n"
                "playground erweitert dieselbe Codebasis um Simulated Annealing und eine "
                "sichtbare Optimierung ueber Aktivierungs-Layouts.\n"
                "experiment_builder ist fuer reproduzierbare Multi-Seed-Runs, JSON-Ergebnisse "
                "und systematische Suchlaeufe gedacht."
            ),
            options=SUPPORTED_GUI_APP_MODES,
            default=args.gui_app_mode,
        )
        args.gui_mode = _prompt_choice(
            label="GUI-Modus",
            explanation=(
                "beginner zeigt nur die wichtigsten Lernschritte.\n"
                "expert zeigt variable Layer, Layout-Eingriffe, Stepper und Vergleichswerkzeuge."
            ),
            options=SUPPORTED_GUI_MODES,
            default=args.gui_mode,
        )
        args.gui_language = _prompt_choice(
            label="GUI-Sprache",
            explanation=(
                "Die GUI kann vollstaendig auf Deutsch oder Englisch laufen. "
                "Das betrifft Labels, Hilfetexte, Info-Boxen und die Lernanleitung."
            ),
            options=SUPPORTED_GUI_LANGUAGES,
            default=args.gui_language,
        )
        args.save_prefix = None
        args.no_plot = True
    else:
        save_plots = _prompt_yes_no(
            label="Plots als PNG speichern",
            explanation=(
                "Gespeichert werden Lernkurven und das Layout/Neighbor-Bild.\n"
                "Das ist meist sinnvoller als ein GUI-Fenster, besonders in Lernumgebungen."
            ),
            default=args.save_prefix is not None,
        )
        if save_plots:
            default_prefix = args.save_prefix or f"interactive/{args.benchmark}_experiment"
            args.save_prefix = _prompt_text(
                label="Speicher-Praefix",
                explanation=(
                    "Beispiel: 'demo/mein_run'. Dann entstehen Dateien wie\n"
                    "outputs/demo/mein_run_history.png"
                ),
                default=default_prefix,
            )
            # Im interaktiven Assistenten speichern wir bevorzugt Plots als Dateien.
            args.no_plot = True
        else:
            args.save_prefix = None
            args.no_plot = True

    print()
    print("Zusammenfassung des geplanten Experiments")
    print("-----------------------------------------")
    for line in _build_summary_lines(args, hidden_sizes):
        print(line)
    print()

    if not _prompt_yes_no(
        label="Experiment jetzt starten",
        explanation="Mit 'nein' kannst du den Lauf hier abbrechen.",
        default=True,
    ):
        parser.exit(0, "Experiment abgebrochen.\n")

    print()
    return args


def _build_layerwise_layout_from_prompt(num_layers: int) -> str:
    """Erfragt eine einfache Layout-Konfiguration pro Hidden-Layer."""

    print()
    print("Layerweises Aktivierungs-Layout")
    print("-------------------------------")
    print(
        "Hier waehlst du fuer jeden Hidden-Layer genau eine Aktivierungsfunktion.\n"
        "Wenn du spaeter einzelne Neuronen unterschiedlich belegen willst, nutze "
        "stattdessen den direkten Layout-String."
    )

    layer_activations: list[str] = []
    for layer_index in range(num_layers):
        layer_activations.append(
            _prompt_choice(
                label=f"Aktivierung fuer Hidden-Layer {layer_index + 1}",
                explanation=(
                    "Moeglich sind: relu, tanh, sigmoid, leaky_relu.\n"
                    f"Diese Wahl gilt dann fuer alle Neuronen in Layer {layer_index + 1}."
                ),
                options=SUPPORTED_ACTIVATIONS,
                default="relu",
            )
        )
    return "|".join(layer_activations)


def _prompt_layout_string(hidden_sizes: tuple[int, ...], default: str) -> str:
    """Erfragt einen kompletten Layout-String und validiert ihn nach Moeglichkeit."""

    print()
    print("Direkte Layout-Eingabe")
    print("----------------------")
    print(
        "Du kannst hier ein einfaches Layer-Layout oder ein gemischtes Per-Neuron-Layout eingeben."
    )
    print(LAYOUT_SYNTAX_EXAMPLES)

    while True:
        layout_spec = _prompt_text(
            label="Layout-String",
            explanation=(
                "Beispiele: 'relu|tanh', 'relu|tanh|sigmoid', "
                "'relu*16|tanh*8', 'relu*8,tanh*8|sigmoid*4,leaky_relu*4'"
            ),
            default=default,
        )
        try:
            from activations import parse_layout_spec

            parse_layout_spec(layout_spec, hidden_sizes)
            return layout_spec
        except ImportError:
            return layout_spec
        except ValueError as exc:
            print(f"Ungueltiges Layout: {exc}")
            print("Bitte erneut eingeben.\n")


def _prompt_neighbor_operations() -> list[str]:
    """Erfragt eine Liste von Neighbor-Operationen.

    Eine leere Eingabe beendet die Eingabephase.
    """

    print()
    print("Neighbor-Operationen")
    print("--------------------")
    print(
        "Neighbor-Operationen veraendern ein Layout lokal. Das ist spaeter ideal "
        "fuer Suchverfahren wie Simulated Annealing."
    )
    print(NEIGHBOR_OPERATION_EXAMPLES)

    operations: list[str] = []
    while True:
        operation = input(
            "Neighbor-Operation eingeben (leer lassen, um fertig zu sein): "
        ).strip()
        if not operation:
            break
        operations.append(operation)
        print("Operation gespeichert.\n")

    return operations


def _build_summary_lines(
    args: argparse.Namespace, hidden_sizes: tuple[int, ...]
) -> list[str]:
    """Baut eine kurze Zusammenfassung des kommenden Laufs."""

    return [
        f"Benchmark:           {args.benchmark}",
        f"Hidden-Sizes:        {format_hidden_sizes(hidden_sizes)}",
        f"Layout:              {args.layout}",
        f"Neighbor-Operationen:{' keine' if not args.neighbor_op else ' ' + ', '.join(args.neighbor_op)}",
        f"Neighbor-Vorschau:   {args.show_neighbors}",
        f"Epochen:             {args.epochs}",
        f"Lernrate:            {args.lr}",
        f"Batch-Groesse:       {args.batch_size}",
        f"Weight-Scale:        {args.weight_scale}",
        f"Seed:                {args.seed}",
        f"GUI-Modus:           {'ja' if args.gui else 'nein'}",
        f"GUI-Arbeitsmodus:    {args.gui_app_mode if args.gui else '-'}",
        f"GUI-Ansicht:         {args.gui_mode if args.gui else '-'}",
        f"GUI-Sprache:         {args.gui_language if args.gui else '-'}",
        f"Plots speichern:     {args.save_prefix if args.save_prefix else 'nein'}",
    ]


def _prompt_choice(
    label: str,
    explanation: str,
    options: tuple[str, ...],
    default: str,
) -> str:
    """Fragt eine Auswahl aus einer festen Menge moeglicher Werte ab."""

    print()
    print(f"{label}")
    print("-" * len(label))
    print(explanation)
    print(f"Moegliche Werte: {', '.join(options)}")

    normalized_options = {option.lower(): option for option in options}
    default_value = normalized_options[default.lower()]

    while True:
        raw_value = input(f"{label} [{default_value}]: ").strip().lower()
        if not raw_value:
            return default_value
        if raw_value in normalized_options:
            return normalized_options[raw_value]
        print("Ungueltige Eingabe. Bitte einen der angezeigten Werte verwenden.")


def _prompt_yes_no(label: str, explanation: str, default: bool) -> bool:
    """Fragt eine Ja/Nein-Entscheidung ab."""

    print()
    print(f"{label}")
    print("-" * len(label))
    print(explanation)

    prompt_suffix = "[J/n]" if default else "[j/N]"
    while True:
        raw_value = input(f"{label} {prompt_suffix}: ").strip().lower()
        if not raw_value:
            return default
        if raw_value in {"j", "ja", "y", "yes"}:
            return True
        if raw_value in {"n", "nein", "no"}:
            return False
        print("Bitte mit ja oder nein antworten.")


def _prompt_int(
    label: str,
    explanation: str,
    default: int,
    min_value: int,
    max_value: int | None = None,
) -> int:
    """Fragt einen ganzzahligen Wert ab."""

    print()
    print(f"{label}")
    print("-" * len(label))
    print(explanation)

    while True:
        raw_value = input(f"{label} [{default}]: ").strip()
        if not raw_value:
            return default
        try:
            value = int(raw_value)
        except ValueError:
            print("Bitte eine ganze Zahl eingeben.")
            continue
        if value < min_value:
            print(f"Bitte eine Zahl groesser oder gleich {min_value} eingeben.")
            continue
        if max_value is not None and value > max_value:
            print(f"Bitte eine Zahl kleiner oder gleich {max_value} eingeben.")
            continue
        return value


def _prompt_float(
    label: str, explanation: str, default: float, min_value: float
) -> float:
    """Fragt einen Float-Wert ab."""

    print()
    print(f"{label}")
    print("-" * len(label))
    print(explanation)

    while True:
        raw_value = input(f"{label} [{default}]: ").strip()
        if not raw_value:
            return default
        try:
            value = float(raw_value)
        except ValueError:
            print("Bitte eine Kommazahl eingeben.")
            continue
        if value < min_value:
            print(f"Bitte eine Zahl groesser oder gleich {min_value} eingeben.")
            continue
        return value


def _prompt_text(label: str, explanation: str, default: str) -> str:
    """Fragt freien Text ab, mit Defaultwert bei leerer Eingabe."""

    print()
    print(f"{label}")
    print("-" * len(label))
    print(explanation)

    raw_value = input(f"{label} [{default}]: ").strip()
    return raw_value if raw_value else default


def _show_or_save_plots(
    benchmark_name: str,
    base_layout,
    neighbor_layout,
    history: dict[str, list[float]],
    visualization_config: VisualizationConfig,
) -> None:
    """Erzeugt Matplotlib-Plots und speichert oder zeigt sie an.

    `MPLCONFIGDIR` wird standardmaessig auf einen lokalen Ordner gesetzt, damit
    Matplotlib keine Probleme mit nicht schreibbaren Home-Verzeichnissen bekommt.
    """

    os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig")))

    from plotting import plot_layouts, plot_training_history, save_figure
    import matplotlib.pyplot as plt

    history_figure = plot_training_history(history, benchmark_name)
    layout_figure = plot_layouts(base_layout, neighbor_layout)

    if visualization_config.save_prefix:
        prefix = Path(visualization_config.save_prefix)
        history_path = save_figure(history_figure, prefix.parent / f"{prefix.name}_history.png")
        layout_path = save_figure(layout_figure, prefix.parent / f"{prefix.name}_layouts.png")
        print()
        print(f"Plots gespeichert: {history_path}")
        print(f"Plots gespeichert: {layout_path}")

    if visualization_config.show_plots:
        backend_name = plt.get_backend().lower()
        if "agg" in backend_name:
            print()
            print(
                "Hinweis: Das aktive Matplotlib-Backend ist 'Agg'. "
                "GUI-Fenster werden daher nicht angezeigt."
            )
            print("Nutze `--save-prefix`, um die Plots als PNG-Dateien zu erhalten.")
            plt.close(history_figure)
            plt.close(layout_figure)
        else:
            plt.show()
    else:
        plt.close(history_figure)
        plt.close(layout_figure)


def _launch_gui(args: argparse.Namespace) -> None:
    """Startet die GUI mit den aktuellen Parser-/Assistentwerten."""

    try:
        from gui import GuiExperimentConfig, launch_playground_gui

        hidden_sizes = (
            tuple(args.hidden_sizes) if args.hidden_sizes else default_hidden_sizes(args.benchmark)
        )
        gui_config = GuiExperimentConfig(
            benchmark=args.benchmark,
            hidden_sizes=hidden_sizes,
            app_mode=args.gui_app_mode,
            layout_spec=args.layout,
            epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            weight_scale=args.weight_scale,
            random_state=args.seed,
            mode=args.gui_mode,
            language=args.gui_language,
        )
        launch_playground_gui(gui_config)
    except ImportError as exc:
        raise SystemExit(f"{exc}\n") from exc


if __name__ == "__main__":
    main()
