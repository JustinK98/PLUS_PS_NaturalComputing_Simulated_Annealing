"""Duenner Programmeinstieg fuer den Aktivierungsfunktions-Playground."""

from __future__ import annotations

import argparse
import sys

from runtime_env import configure_runtime_environment

configure_runtime_environment()

from cli.commands.experiment_analyze import analyze_experiment_command
from cli.commands.experiment_layout_grid import run_layout_grid_command
from cli.commands.experiment_report import build_report_command
from cli.commands.experiment_report_online_delta import build_online_delta_report_command
from cli.commands.experiment_run import run_experiment_command
from cli.commands.experiment_suite import run_experiment_suite_command
from cli.commands.experiment_template import create_experiment_template_command
from cli.commands.gui import run_gui_command
from cli.commands.single_run import run_single_command
from cli.interactive_assistant import run_interactive_setup, should_run_interactive
from cli.parser import build_parser, resolve_command
from configs import LAYOUT_SYNTAX_EXAMPLES, NEIGHBOR_OPERATION_EXAMPLES


def main(argv: list[str] | None = None) -> None:
    """Programmstart mit Subcommands und Legacy-Kompatibilitaet."""

    effective_argv = _normalize_compat_argv(list(sys.argv[1:] if argv is None else argv))
    parser = build_parser()
    args = parser.parse_args(effective_argv)

    if args.show_layout_help:
        print(LAYOUT_SYNTAX_EXAMPLES)
        print(NEIGHBOR_OPERATION_EXAMPLES)
        return

    if args.command is None and should_run_interactive(args, effective_argv):
        args = run_interactive_setup(parser, args)

    _dispatch_command(args)


def _dispatch_command(args: argparse.Namespace) -> None:
    """Leitet die aufgeloeste Parser-Entscheidung an das passende Modul weiter."""

    command = resolve_command(args)
    if command == "gui":
        run_gui_command(args)
        return
    if command == "run":
        run_single_command(args)
        return
    if command == "experiment_run":
        run_experiment_command(args)
        return
    if command == "experiment_suite":
        run_experiment_suite_command(args)
        return
    if command == "experiment_analyze":
        analyze_experiment_command(args)
        return
    if command == "experiment_template":
        create_experiment_template_command(args)
        return
    if command == "experiment_layout_grid":
        run_layout_grid_command(args)
        return
    if command == "experiment_report":
        build_report_command(args)
        return
    if command == "experiment_report_online_delta":
        build_online_delta_report_command(args)
        return
    raise SystemExit(f"Unbekanntes Kommando: {command}")


def _normalize_compat_argv(argv: list[str]) -> list[str]:
    """Mappt alte GUI-Aufrufe auf den Qt-Subcommand-Pfad."""

    if not argv:
        return argv
    if argv[0] in {"gui", "run", "experiment"}:
        return argv

    gui_flags = {
        "--gui",
        "--mode",
        "--gui-app-mode",
        "--detail-level",
        "--gui-mode",
        "--language",
        "--gui-language",
    }
    if not any(flag in gui_flags for flag in argv):
        return argv

    normalized = [value for value in argv if value != "--gui"]
    return ["gui", *normalized]


if __name__ == "__main__":
    main()
