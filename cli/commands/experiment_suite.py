"""CLI command for official terminal experiment suites."""

from __future__ import annotations

from pathlib import Path

from services.experiment_suite_service import ExperimentSuiteRequest, run_experiment_suite


def run_experiment_suite_command(args) -> None:
    """Runs the selected official experiment suite."""

    result = run_experiment_suite(
        ExperimentSuiteRequest(
            exp=args.exp,
            benchmark=args.benchmark,
            learning_rate_preset=args.learning_rate,
            seeds=tuple(args.seeds) if args.seeds else None,
            run_count=args.runs if args.runs is not None else args.seed_count,
            epochs=args.epochs,
            max_steps=args.max_steps,
            start_temperature=args.start_temperature,
            cooling_parameter=args.cooling_parameter,
            iterations_per_temperature=args.iterations_per_temperature,
            min_temperature=args.min_temperature,
            output_root=Path(args.output_root),
            no_plots=args.no_plots,
        )
    )

    print(f"suite:          {args.exp}")
    print(f"benchmark:      {args.benchmark}")
    print(f"learning_rates: {', '.join(f'{value:.3f}' for value in result.learning_rates)}")
    print(f"runs:           {result.run_count}")
    print(f"output_dir:     {result.output_dir}")
    print(f"summary:        {result.summary_path}")
