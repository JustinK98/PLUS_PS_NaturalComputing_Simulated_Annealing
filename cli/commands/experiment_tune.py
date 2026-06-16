"""CLI command for staged hyperparameter tuning."""

from __future__ import annotations

from pathlib import Path

from services.hyperparameter_tuning_service import (
    HyperparameterTuningRequest,
    run_hyperparameter_tuning,
)


def run_hyperparameter_tuning_command(args) -> None:
    """Run or resume the validation-first tuning pipeline."""

    result = run_hyperparameter_tuning(
        HyperparameterTuningRequest(
            profile=args.profile,
            benchmark=args.benchmark,
            phase=args.phase,
            workers=args.workers,
            output_root=Path(args.output_root),
            resume=Path(args.resume) if args.resume else None,
            smoke=args.smoke,
            export_layout_frames=args.export_layout_frames,
            import_training_from=Path(args.import_training_from) if args.import_training_from else None,
        )
    )
    print(f"tuning:     {args.profile}")
    print(f"benchmarks: {', '.join(result.benchmarks)}")
    print(f"phases:     {', '.join(result.phases)}")
    print(f"output_dir: {result.output_dir}")
    print(f"summary:    {result.summary_path}")
