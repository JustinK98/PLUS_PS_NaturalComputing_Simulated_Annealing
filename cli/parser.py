"""Parser und Dispatch-Entscheidungen fuer die refaktorierte Anwendung."""

from __future__ import annotations

import argparse

from configs import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BENCHMARK,
    DEFAULT_LAYOUT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NEIGHBOR_PREVIEW,
    DEFAULT_RANDOM_SEED,
    DEFAULT_WEIGHT_SCALE,
    LAYOUT_SYNTAX_EXAMPLES,
    SUPPORTED_BENCHMARKS,
)


def build_parser() -> argparse.ArgumentParser:
    """Erzeugt den Parser fuer die Qt-only Anwendung."""

    parser = argparse.ArgumentParser(
        description=(
            "Aktivierungsfunktions-Playground fuer kleine MLPs mit variabler Hidden-Layer-Anzahl."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=LAYOUT_SYNTAX_EXAMPLES,
    )
    parser.add_argument(
        "--show-layout-help",
        action="store_true",
        help="Zeigt Erklaerungen zur Layout-Syntax und beendet das Programm.",
    )

    subparsers = parser.add_subparsers(dest="command")

    gui_parser = subparsers.add_parser("gui", help="Starte die GUI.")
    _add_gui_arguments(gui_parser)
    gui_parser.add_argument(
        "--profile",
        dest="gui_profile",
        choices=("demo", "tuned_20260530"),
        default="demo",
        help="GUI-Parameterprofil: schnelle Demo oder fixierte Evaluation.",
    )

    run_parser = subparsers.add_parser("run", help="Fuehre einen einzelnen CLI-Trainingslauf aus.")
    _add_common_run_arguments(run_parser)

    experiment_parser = subparsers.add_parser(
        "experiment",
        help="Fuehre reproduzierbare Online-Delta-Experimente aus.",
    )
    experiment_subparsers = experiment_parser.add_subparsers(dest="experiment_command", required=True)

    experiment_suite_parser = experiment_subparsers.add_parser(
        "suite",
        help="Fuehre die offiziellen Terminal-Benchmark-Suites aus.",
    )
    experiment_suite_parser.add_argument(
        "--exp",
        choices=("online-delta", "random-baseline", "all-baseline", "swap-ablation"),
        default="online-delta",
        help="Suite-Typ: online-delta, random-baseline, all-baseline oder swap-ablation.",
    )
    experiment_suite_parser.add_argument(
        "--benchmark",
        choices=SUPPORTED_BENCHMARKS,
        default=DEFAULT_BENCHMARK,
        help="Offizieller Basics-Benchmark. Default: two_moons.",
    )
    experiment_suite_parser.add_argument(
        "--learning-rate",
        type=int,
        default=1,
        help="Index der Lernraten-Preset-Gruppe aus configs/experiment_suites.json.",
    )
    experiment_suite_parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Anzahl unabhaengiger Runs. Default kommt aus configs/experiment_suites.json.",
    )
    experiment_suite_parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optionales Epochen-Override fuer schnelle Smokes.",
    )
    experiment_suite_parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optionales Override fuer Online-Delta-SA-Schritte.",
    )
    experiment_suite_parser.add_argument(
        "--start-temperature",
        type=float,
        default=None,
        help="Optionales Override fuer die Online-Delta-SA-Starttemperatur.",
    )
    experiment_suite_parser.add_argument(
        "--cooling-parameter",
        type=float,
        default=None,
        help="Optionales Override fuer den geometrischen Cooling-Faktor.",
    )
    experiment_suite_parser.add_argument(
        "--iterations-per-temperature",
        type=int,
        default=None,
        help="Optionales Override fuer Schritte pro Temperaturstufe.",
    )
    experiment_suite_parser.add_argument(
        "--min-temperature",
        type=float,
        default=None,
        help="Optionales Override fuer die minimale SA-Temperatur.",
    )
    experiment_suite_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optionales Batch-Groessen-Override fuer Training und standardmaessig auch Online-Delta.",
    )
    experiment_suite_parser.add_argument(
        "--weight-scale",
        type=float,
        default=None,
        help="Optionales Override fuer den Xavier/Glorot-Skalierungsfaktor.",
    )
    experiment_suite_parser.add_argument(
        "--online-learning-rate",
        type=float,
        default=None,
        help="Optional getrennte Lernrate waehrend Online-Delta-SA.",
    )
    experiment_suite_parser.add_argument(
        "--online-batch-size",
        type=int,
        default=None,
        help="Optional getrennte Batch-Groesse waehrend Online-Delta-SA.",
    )
    experiment_suite_parser.add_argument(
        "--evaluation-profile",
        default=None,
        help=(
            "Versioniertes Evaluationsprofil aus configs/evaluation_profiles.json. "
            "Schaltet die finale Testauswertung frei."
        ),
    )
    experiment_suite_parser.add_argument(
        "--output-root",
        default="outputs/experiment_suites",
        help="Basisordner fuer Suite-Artefakte.",
    )
    experiment_suite_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Erzeuge nur JSON/CSV-Artefakte, keine PNG-Plots.",
    )
    experiment_suite_parser.add_argument(
        "--export-layout-frames",
        action="store_true",
        help="Exportiere zusaetzlich regenerierbare PNG-Frames akzeptierter Online-Delta-Schritte.",
    )

    experiment_tune_parser = experiment_subparsers.add_parser(
        "tune",
        help="Fuehre gestuftes Hyperparameter-Tuning fuer Training und Online-Delta-SA aus.",
    )
    experiment_tune_parser.add_argument(
        "--profile",
        choices=("overnight",),
        default="overnight",
        help="Tuning-Profil aus configs/hyperparameter_tuning.json.",
    )
    experiment_tune_parser.add_argument(
        "--benchmark",
        choices=(*SUPPORTED_BENCHMARKS, "all"),
        default="all",
        help="Benchmark oder alle offiziellen Benchmarks.",
    )
    experiment_tune_parser.add_argument(
        "--phase",
        choices=(
            "full",
            "training-screen",
            "training-refine",
            "delta-probe",
            "sa-screen",
            "sa-refine",
            "confirm",
        ),
        default="full",
        help="Pipeline-Phase. Downstream-Einzelphasen benoetigen --resume.",
    )
    experiment_tune_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Anzahl paralleler lokaler Worker.",
    )
    experiment_tune_parser.add_argument(
        "--resume",
        default=None,
        help="Vorhandenen Tuning-Ordner fortsetzen; abgeschlossene Trials werden uebersprungen.",
    )
    experiment_tune_parser.add_argument(
        "--output-root",
        default="outputs/hyperparameter_tuning",
        help="Basisordner fuer neue Tuning-Artefakte.",
    )
    experiment_tune_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Kleine End-to-End-Konfiguration fuer schnelle Verifikation.",
    )
    experiment_tune_parser.add_argument(
        "--export-layout-frames",
        action="store_true",
        help="Exportiere in der Confirmation zusaetzlich regenerierbare PNG-Frames akzeptierter Online-Delta-Schritte.",
    )

    online_delta_report_parser = experiment_subparsers.add_parser(
        "report-online-delta",
        help="Aggregiere Online-Delta-SA-Historien aus vorhandenen Suite-Ergebnissen.",
    )
    online_delta_report_parser.add_argument(
        "--path",
        required=True,
        help="Pfad zu einem Online-Delta-Run oder Suite-Ordner.",
    )
    online_delta_report_parser.add_argument(
        "--output-dir",
        default=None,
        help="Optionaler Zielordner. Default: <path>/online_delta_report.",
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
    raise ValueError("Kein ausfuehrbares Subcommand ausgewaehlt.")


def _add_common_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--benchmark",
        choices=SUPPORTED_BENCHMARKS,
        default=DEFAULT_BENCHMARK,
        help=(
            "Offizieller CSV-Benchmark: two_moons, concentric_circles oder crossing_spirals."
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


def _add_gui_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--benchmark",
        choices=SUPPORTED_BENCHMARKS,
        default=DEFAULT_BENCHMARK,
        help="Offizieller CSV-Benchmark fuer die Online-Delta-Demo.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Reproduzierbarer Demo-Seed.")
