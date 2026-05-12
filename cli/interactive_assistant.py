"""Interaktiver CLI-Assistent fuer die modularisierte Anwendung."""

from __future__ import annotations

import argparse

from configs import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WEIGHT_SCALE,
    LAYOUT_SYNTAX_EXAMPLES,
    MAX_HIDDEN_LAYERS,
    MIN_HIDDEN_LAYERS,
    NEIGHBOR_OPERATION_EXAMPLES,
    SUPPORTED_ACTIVATIONS,
    SUPPORTED_BENCHMARKS,
    SUPPORTED_GUI_APP_MODES,
    SUPPORTED_GUI_LANGUAGES,
    SUPPORTED_GUI_MODES,
    default_epochs,
    default_hidden_sizes,
    format_hidden_sizes,
)


def should_run_interactive(args: argparse.Namespace, argv: list[str]) -> bool:
    """Entscheidet, ob der gefuehrte Assistent laufen soll."""

    return bool(getattr(args, "interactive", False)) or len(argv) == 0


def run_interactive_setup(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> argparse.Namespace:
    """Fuehrt den Benutzer durch einen didaktischen CLI-Setup-Flow."""

    print("Interaktiver Experiment-Assistent")
    print("=================================")
    print("Dieses Programm hilft dir, ein kleines MLP-Experiment aufzubauen.")
    print("Leere Eingaben uebernehmen jeweils den vorgeschlagenen Standardwert.")
    print()

    args.benchmark = _prompt_choice(
        label="Benchmark",
        explanation=(
            "Waehle den Datensatz, auf dem trainiert werden soll.\n"
            "- concentric_circles: offizieller Easy-Benchmark mit 2 Eingaben\n"
            "- iris: offizieller Medium-Benchmark mit 4 Eingaben und 3 Klassen\n"
            "- crossing_spirals: offizieller Hard-Benchmark mit 6 Eingaben"
        ),
        options=SUPPORTED_BENCHMARKS,
        default=args.benchmark,
    )
    if args.epochs is None:
        args.epochs = default_epochs(args.benchmark)

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
    args.show_neighbors = (
        _prompt_int(
            label="Wie viele Nachbarn anzeigen",
            explanation="Groessere Werte zeigen mehr Varianten, machen die Ausgabe aber laenger.",
            default=args.show_neighbors,
            min_value=0,
        )
        if show_neighbors
        else 0
    )

    use_default_training = _prompt_yes_no(
        label="Standard-Trainingswerte verwenden",
        explanation=(
            "Die Standardwerte sind ein guter Start. Wenn du lernen willst, wie "
            "Hyperparameter das Verhalten beeinflussen, kannst du sie hier aendern."
        ),
        default=(
            args.epochs == default_epochs(args.benchmark)
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
            explanation="Die Batch-Groesse bestimmt, wie viele Beispiele pro Update genutzt werden.",
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
            explanation="Steuert, wie gross die Startgewichte zufaellig initialisiert werden.",
            default=args.weight_scale,
            min_value=1e-8,
        )
        args.seed = _prompt_int(
            label="Random Seed",
            explanation="Mit demselben Seed bleiben Daten-Splits und Initialisierungen reproduzierbar.",
            default=args.seed,
            min_value=0,
        )

    start_gui = _prompt_yes_no(
        label="GUI starten",
        explanation=(
            "Die GUI zeigt das Netzwerk mit Kreisen und Verbindungen, erlaubt Klicks "
            "auf Neuronen und zeigt lokale Berechnungen eines ausgewaehlten Neurons."
        ),
        default=args.command == "gui",
    )

    if start_gui:
        args.gui_app_mode = _prompt_choice(
            label="GUI-Arbeitsmodus",
            explanation=(
                "activation_workflow verbindet Training, SA-Suche und finalen Layout-Vergleich.\n"
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
        args.command = "gui"
        args.save_prefix = None
        args.no_plot = True
    else:
        args.command = "run"
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
                    "Beispiel: 'workflow/mein_run'. Dann entstehen Dateien wie\n"
                    "outputs/workflow/mein_run_history.png"
                ),
                default=default_prefix,
            )
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
                    "Moeglich sind: relu, gelu, sigmoid, tanh, swish, identity.\n"
                    f"Diese Wahl gilt dann fuer alle Neuronen in Layer {layer_index + 1}."
                ),
                options=SUPPORTED_ACTIVATIONS,
                default="relu",
            )
        )
    return "|".join(layer_activations)


def _prompt_layout_string(hidden_sizes: tuple[int, ...], default: str) -> str:
    print()
    print("Direkte Layout-Eingabe")
    print("----------------------")
    print("Du kannst hier ein einfaches Layer-Layout oder ein gemischtes Per-Neuron-Layout eingeben.")
    print(LAYOUT_SYNTAX_EXAMPLES)

    while True:
        layout_spec = _prompt_text(
            label="Layout-String",
            explanation=(
                "Beispiele: 'relu|tanh', 'relu|tanh|sigmoid', "
                "'relu*16|tanh*8', 'relu*8,tanh*8|sigmoid*4,swish*4'"
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
        operation = input("Neighbor-Operation eingeben (leer lassen, um fertig zu sein): ").strip()
        if not operation:
            break
        operations.append(operation)
        print("Operation gespeichert.\n")
    return operations


def _build_summary_lines(
    args: argparse.Namespace, hidden_sizes: tuple[int, ...]
) -> list[str]:
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
        f"GUI-Modus:           {'ja' if args.command == 'gui' else 'nein'}",
        f"GUI-Arbeitsmodus:    {args.gui_app_mode if args.command == 'gui' else '-'}",
        f"GUI-Ansicht:         {args.gui_mode if args.command == 'gui' else '-'}",
        f"GUI-Sprache:         {args.gui_language if args.command == 'gui' else '-'}",
        f"Plots speichern:     {args.save_prefix if args.save_prefix else 'nein'}",
    ]


def _prompt_choice(label: str, explanation: str, options: tuple[str, ...], default: str) -> str:
    print()
    print(label)
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
    print()
    print(label)
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
    print()
    print(label)
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


def _prompt_float(label: str, explanation: str, default: float, min_value: float) -> float:
    print()
    print(label)
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
    print()
    print(label)
    print("-" * len(label))
    print(explanation)

    raw_value = input(f"{label} [{default}]: ").strip()
    return raw_value if raw_value else default
