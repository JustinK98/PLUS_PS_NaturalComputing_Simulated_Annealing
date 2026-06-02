"""Duenner Programmeinstieg fuer den Aktivierungsfunktions-Playground."""

from __future__ import annotations

import sys
import argparse

from runtime_env import configure_runtime_environment

configure_runtime_environment()

from cli.commands.experiment_report_online_delta import build_online_delta_report_command
from cli.commands.experiment_suite import run_experiment_suite_command
from cli.commands.experiment_tune import run_hyperparameter_tuning_command
from cli.commands.gui import run_gui_command
from cli.commands.single_run import run_single_command
from cli.parser import build_parser, resolve_command
from configs import LAYOUT_SYNTAX_EXAMPLES


def main(argv: list[str] | None = None) -> None:
    """Programmstart mit expliziten CLI- und GUI-Subcommands."""

    effective_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not effective_argv:
        parser.print_help()
        return
    args = parser.parse_args(effective_argv)

    if args.show_layout_help:
        print(LAYOUT_SYNTAX_EXAMPLES)
        return

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
    if command == "experiment_suite":
        run_experiment_suite_command(args)
        return
    if command == "experiment_report_online_delta":
        build_online_delta_report_command(args)
        return
    if command == "experiment_tune":
        run_hyperparameter_tuning_command(args)
        return
    raise SystemExit(f"Unbekanntes Kommando: {command}")


if __name__ == "__main__":
    main()
