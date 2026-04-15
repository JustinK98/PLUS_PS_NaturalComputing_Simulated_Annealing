"""CLI-Kommando zum Erzeugen einer Experiment-Definition."""

from __future__ import annotations

from services.experiment_service import save_experiment_template


def create_experiment_template_command(args) -> None:
    """Schreibt eine Beispieldefinition fuer headless Experimentlaeufe."""

    output_path = save_experiment_template(args.output)
    print(f"template: {output_path}")
