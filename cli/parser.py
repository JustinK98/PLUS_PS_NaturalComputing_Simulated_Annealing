"""Parser und Dispatch-Entscheidungen fuer die refaktorierte Anwendung."""

from __future__ import annotations

import argparse

from configs import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BENCHMARK,
    DEFAULT_GUI_APP_MODE,
    DEFAULT_LAYOUT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NEIGHBOR_PREVIEW,
    DEFAULT_RANDOM_SEED,
    DEFAULT_WEIGHT_SCALE,
    LAYOUT_SYNTAX_EXAMPLES,
    NEIGHBOR_OPERATION_EXAMPLES,
    SUPPORTED_BENCHMARKS,
    SUPPORTED_GUI_APP_MODES,
    SUPPORTED_GUI_LANGUAGES,
    SUPPORTED_GUI_MODES,
)


def build_parser() -> argparse.ArgumentParser:
    """Erzeugt den Parser fuer die Qt-only Anwendung."""

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
        help="Kompatibilitaetsalias: startet direkt die Qt-GUI.",
    )
    parser.add_argument(
        "--mode",
        "--gui-app-mode",
        dest="gui_app_mode",
        choices=SUPPORTED_GUI_APP_MODES,
        default=DEFAULT_GUI_APP_MODE,
        help="Kompatibilitaetsalias fuer den GUI-Arbeitsmodus.",
    )
    parser.add_argument(
        "--detail-level",
        "--gui-mode",
        dest="gui_mode",
        choices=SUPPORTED_GUI_MODES,
        default="beginner",
        help="Kompatibilitaetsalias fuer die GUI-Detailstufe.",
    )
    parser.add_argument(
        "--language",
        "--gui-language",
        dest="gui_language",
        choices=SUPPORTED_GUI_LANGUAGES,
        default="de",
        help="Kompatibilitaetsalias fuer die GUI-Sprache.",
    )
    parser.add_argument(
        "--show-layout-help",
        action="store_true",
        help="Zeigt Erklaerungen zur Layout- und Neighbor-Syntax und beendet das Programm.",
    )
    _add_common_run_arguments(parser)

    subparsers = parser.add_subparsers(dest="command")

    gui_parser = subparsers.add_parser("gui", help="Starte die GUI.")
    _add_common_run_arguments(gui_parser)
    gui_parser.add_argument(
        "--mode",
        "--gui-app-mode",
        dest="gui_app_mode",
        choices=SUPPORTED_GUI_APP_MODES,
        default=DEFAULT_GUI_APP_MODE,
        help=(
            "GUI-Arbeitsmodus: activation_workflow, presentation, demo, "
            "playground oder experiment_builder."
        ),
    )
    gui_parser.add_argument(
        "--detail-level",
        "--gui-mode",
        dest="gui_mode",
        choices=SUPPORTED_GUI_MODES,
        default="beginner",
        help="GUI-Detailstufe: beginner oder expert.",
    )
    gui_parser.add_argument(
        "--language",
        "--gui-language",
        dest="gui_language",
        choices=SUPPORTED_GUI_LANGUAGES,
        default="de",
        help="Startsprache der GUI.",
    )

    run_parser = subparsers.add_parser("run", help="Fuehre einen einzelnen CLI-Trainingslauf aus.")
    _add_common_run_arguments(run_parser)

    experiment_parser = subparsers.add_parser(
        "experiment",
        help="Erzeuge, starte oder analysiere Builder-Experimente headless.",
    )
    experiment_subparsers = experiment_parser.add_subparsers(dest="experiment_command", required=True)

    experiment_run_parser = experiment_subparsers.add_parser(
        "run",
        help="Fuehre eine gespeicherte ExperimentDefinition aus.",
    )
    experiment_run_parser.add_argument("--config", required=True, help="Pfad zur Experiment-JSON.")
    experiment_run_parser.add_argument("--output-dir", default=None, help="Optionales Output-Override.")
    experiment_run_parser.add_argument("--experiment-id", default=None, help="Optionales ID-Override.")
    experiment_run_parser.add_argument(
        "--no-save-json",
        dest="save_json",
        action="store_false",
        default=None,
        help="Fuehrt den Lauf aus, ohne Ergebnisse nach JSON zu schreiben.",
    )
    experiment_run_parser.add_argument(
        "--max-ranking",
        type=int,
        default=5,
        help="Wie viele Ranking-Eintraege nach Abschluss gezeigt werden.",
    )

    experiment_analyze_parser = experiment_subparsers.add_parser(
        "analyze",
        help="Analysiere ein bereits gespeichertes Experiment.",
    )
    experiment_analyze_parser.add_argument("--path", required=True, help="Pfad zu Manifest oder Experimentordner.")
    experiment_analyze_parser.add_argument("--max-ranking", type=int, default=5)
    experiment_analyze_parser.add_argument(
        "--show-runs",
        type=int,
        default=0,
        help="Zeige die ersten N Run-Details zusaetzlich an.",
    )

    experiment_template_parser = experiment_subparsers.add_parser(
        "template",
        help="Schreibt eine Beispiel-ExperimentDefinition.",
    )
    experiment_template_parser.add_argument("--output", required=True, help="Zielpfad fuer die JSON-Datei.")

    layout_grid_parser = experiment_subparsers.add_parser(
        "layout-grid",
        help="Trainiert ein Layout-Grid und speichert ein vortrainiertes Demo-Artefakt.",
    )
    layout_grid_parser.add_argument(
        "--benchmark",
        choices=SUPPORTED_BENCHMARKS,
        default=DEFAULT_BENCHMARK,
        help="Offizieller CSV-Benchmark.",
    )
    layout_grid_parser.add_argument(
        "--hidden-sizes",
        nargs="+",
        type=int,
        metavar="H",
        help="Hidden-Sizes fuer das Grid. Ohne Angabe gilt der Benchmark-Default.",
    )
    layout_grid_parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[DEFAULT_RANDOM_SEED],
        help="Seeds fuer faire Multi-Seed-Auswertung.",
    )
    layout_grid_parser.add_argument("--epochs", type=int, default=None)
    layout_grid_parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    layout_grid_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    layout_grid_parser.add_argument("--weight-scale", type=float, default=DEFAULT_WEIGHT_SCALE)
    layout_grid_parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    layout_grid_parser.add_argument(
        "--primary-metric",
        choices=("validation_loss", "validation_accuracy"),
        default="validation_loss",
    )
    layout_grid_parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optionales Limit fuer schnelle Demo-Suchen.",
    )
    layout_grid_parser.add_argument(
        "--no-mixed",
        action="store_true",
        help="Nur homogene/per-layer Layouts testen, keine gemischten Layer.",
    )
    layout_grid_parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Wie viele Top-Layouts im Terminal ausgegeben werden.",
    )
    layout_grid_parser.add_argument(
        "--output",
        default=None,
        help="Zielpfad. Default: outputs/layout_grids/<benchmark>_layout_grid.json",
    )

    report_parser = experiment_subparsers.add_parser(
        "report",
        help="Erzeuge CSV-/Plot-Artefakte fuer Bericht und Demo aus vorhandenen Ergebnissen.",
    )
    report_parser.add_argument(
        "--output-dir",
        default="outputs/report_assets",
        help="Zielordner fuer Report-Artefakte.",
    )
    report_parser.add_argument(
        "--layout-grid-dir",
        default="outputs/layout_grids",
        help="Ordner mit Layout-Grid-JSON-Dateien.",
    )
    report_parser.add_argument(
        "--demo-sa-dir",
        default="outputs/demo_sa",
        help="Ordner mit Demo-SA-Ergebnissen.",
    )
    report_parser.add_argument(
        "--neighborhood-summary",
        default="outputs/neighborhood_grids/summary.csv",
        help="CSV-Zusammenfassung der Neighborhood-Runs.",
    )

    return parser


def resolve_command(args: argparse.Namespace) -> str:
    """Mappt Parserpfade auf einen Dispatcher-Key."""

    if args.command == "gui":
        return "gui"
    if args.command == "run":
        return "run"
    if args.command == "experiment":
        return f"experiment_{args.experiment_command.replace('-', '_')}"
    if getattr(args, "gui", False):
        return "gui"
    return "run"


def _add_common_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--benchmark",
        choices=SUPPORTED_BENCHMARKS,
        default=DEFAULT_BENCHMARK,
        help=(
            "Offizieller CSV-Benchmark: concentric_circles, iris oder crossing_spirals."
        ),
    )
    parser.add_argument(
        "--hidden-sizes",
        nargs="+",
        type=int,
        metavar="H",
        help="Neuronenzahlen fuer die Hidden-Layer, z. B. --hidden-sizes 16 8.",
    )
    parser.add_argument(
        "--layout",
        default=DEFAULT_LAYOUT,
        help="Layout-Syntax, z. B. 'relu|tanh' oder 'relu*16|sigmoid*8'.",
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
        default=None,
        help="Anzahl der Trainingsepochen. Ohne Angabe gilt der Benchmark-Default.",
    )
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, help="Lernrate.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch-Groesse.")
    parser.add_argument("--weight-scale", type=float, default=DEFAULT_WEIGHT_SCALE, help="Gewichtsskala.")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Reproduzierbarer Seed.")
    parser.add_argument("--no-plot", action="store_true", help="Keine Matplotlib-Ausgabe erzeugen.")
    parser.add_argument(
        "--save-prefix",
        type=str,
        default=None,
        help="Speichert Plots als '<prefix>_history.png' und '<prefix>_layouts.png'.",
    )
