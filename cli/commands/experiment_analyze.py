"""CLI-Kommando fuer Analyse gespeicherter Builder-Ergebnisse."""

from __future__ import annotations

from services.experiment_service import load_saved_experiment
from services.preview_service import format_experiment_summary, format_run_summary


def analyze_experiment_command(args) -> None:
    """Laedt ein gespeichertes Experiment und rendert eine CLI-Zusammenfassung."""

    payload = load_saved_experiment(args.path)
    print(format_experiment_summary(payload, max_ranking_entries=args.max_ranking))
    if args.show_runs:
        for run_payload in payload["runs"][: args.show_runs]:
            print()
            print(format_run_summary(run_payload))
