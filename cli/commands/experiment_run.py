"""CLI-Kommando fuer headless Experiment-Builder-Laeufe."""

from __future__ import annotations

from dataclasses import replace

from services.experiment_service import load_experiment_definition, run_experiment_definition
from services.preview_service import format_experiment_summary


def run_experiment_command(args) -> None:
    """Fuehrt ein gespeichertes Experiment headless aus."""

    definition = load_experiment_definition(args.config)
    if getattr(args, "experiment_id", None):
        definition = replace(definition, experiment_id=args.experiment_id)
    if getattr(args, "output_dir", None):
        definition = replace(definition, output_dir=args.output_dir)
    if getattr(args, "save_json", None) is False:
        definition = replace(definition, save_json=False)

    payload = run_experiment_definition(definition)
    if payload["output_path"]:
        print(f"results_dir:   {payload['output_path']}")
    analysis_payload = {
        "manifest": {
            "experiment_id": definition.experiment_id,
            "benchmark": definition.benchmark,
            "run_mode": definition.run_mode,
            "search_type": definition.search_space.search_type,
            "primary_metric": definition.primary_metric,
        },
        "summary": payload["summary"].to_dict(),
    }
    print(format_experiment_summary(analysis_payload, max_ranking_entries=args.max_ranking))
