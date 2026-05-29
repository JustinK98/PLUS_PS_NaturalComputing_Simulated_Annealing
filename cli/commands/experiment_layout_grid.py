"""CLI-Kommando fuer reproduzierbare Layout-Grid-Suchen."""

from __future__ import annotations

from pathlib import Path

from configs import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_WEIGHT_SCALE,
    OUTPUT_DIR,
    DatasetConfig,
    TrainingConfig,
    default_epochs,
    default_hidden_sizes,
)
from services.layout_evaluation_service import (
    LayoutEvaluationRequest,
    build_layout_grid_candidates,
    run_layout_evaluation,
    save_layout_grid_result,
)


def run_layout_grid_command(args) -> None:
    """Trainiert ein Layout-Grid und speichert ein reproduzierbares Artefakt."""

    hidden_sizes = tuple(args.hidden_sizes) if args.hidden_sizes else default_hidden_sizes(args.benchmark)
    epochs = args.epochs if args.epochs is not None else default_epochs(args.benchmark)
    seeds = tuple(int(seed) for seed in args.seeds)
    candidates = build_layout_grid_candidates(
        hidden_sizes,
        include_mixed=not args.no_mixed,
        max_candidates=args.max_candidates,
    )
    request = LayoutEvaluationRequest(
        dataset_config=DatasetConfig(name=args.benchmark, random_state=args.seed),
        hidden_sizes=hidden_sizes,
        candidates=candidates,
        seeds=seeds,
        training_config=TrainingConfig(
            epochs=epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            random_state=args.seed,
            shuffle=True,
        ),
        weight_scale=args.weight_scale,
        primary_metric=args.primary_metric,
    )
    result = run_layout_evaluation(request)
    output_path = Path(args.output) if args.output else _default_output_path(args.benchmark)
    saved_path = save_layout_grid_result(output_path, request=request, result=result)

    print(f"layout_grid:   {saved_path}")
    print(f"benchmark:     {args.benchmark}")
    print(f"candidates:    {len(candidates)}")
    print(f"seeds:         {', '.join(str(seed) for seed in seeds)}")
    print("top layouts:")
    for index, item in enumerate(result.ranking[: args.top_k], start=1):
        print(
            f"  {index:02d}. {item.label:24s} "
            f"val_loss={item.mean_metrics['val_loss']:.4f} "
            f"val_acc={item.mean_metrics['val_accuracy']:.4f} "
            f"test_acc={item.mean_metrics['test_accuracy']:.4f} "
            f"layout={item.layout_spec}"
        )


def _default_output_path(benchmark: str) -> Path:
    return OUTPUT_DIR / "layout_grids" / f"{benchmark}_layout_grid.json"
