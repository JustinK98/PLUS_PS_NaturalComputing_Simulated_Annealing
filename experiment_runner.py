"""Runner fuer Builder-Experimente, Multi-Seed-Runs und Search-Experimente."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from activations import generate_neighbors, parse_layout_spec
from annealing import AnnealingConfig, annealing_stop_reasons
from annealing_objectives import LayoutObjectiveEvaluator, ObjectiveConfig
from annealing_runner import AnnealingRunner
from benchmarks import load_benchmark
from configs import DatasetConfig, TrainingConfig
from experiment_builder import (
    ExperimentDefinition,
    RunDefinition,
    RunResult,
    utc_timestamp,
)
from model import ModularMLP
from results_analysis import aggregate_run_results
from results_store import save_experiment_results
from search_spaces import expand_search_space
from services.layout_evaluation_service import (
    InheritedModelCandidate,
    LayoutEvaluationRequest,
    build_standard_layout_candidates,
    evaluate_inherited_models,
    run_layout_evaluation,
    with_inherited_model_evaluations,
)
from services.online_annealing_training_service import (
    OnlineAnnealingRequest,
    create_online_session,
    online_run_to_completion,
)
from trainer import train_model


class ExperimentRunner:
    """Fuehrt Builder-Experimente reproduzierbar aus."""

    def run_experiment(self, definition: ExperimentDefinition) -> dict[str, Any]:
        """Fuehrt ein komplettes Experiment inkl. Aggregation aus."""

        if definition.run_mode == "simulated_annealing" and definition.sa_evaluation_mode == "online_delta":
            definition = replace(definition, primary_metric="validation_loss")

        expanded_configs = expand_search_space(definition.search_space)
        run_results: list[RunResult] = []

        for config_index, config_values in enumerate(expanded_configs, start=1):
            config_id = f"cfg_{config_index:04d}"
            for seed in definition.seeds:
                run_definition = RunDefinition(
                    run_id=f"{config_id}_seed_{int(seed):04d}",
                    experiment_id=definition.experiment_id,
                    config_id=config_id,
                    config_index=config_index,
                    seed=int(seed),
                    benchmark=definition.benchmark,
                    hidden_sizes=definition.hidden_sizes,
                    layout_spec=definition.layout_spec,
                    run_mode=definition.run_mode,
                    primary_metric=definition.primary_metric,
                    config_values=dict(config_values),
                )
                run_results.append(
                    self._run_single_definition(definition, run_definition)
                )

        summary = aggregate_run_results(
            run_results=run_results,
            benchmark=definition.benchmark,
            run_mode=definition.run_mode,
            search_type=definition.search_space.search_type,
            primary_metric=definition.primary_metric,
            experiment_id=definition.experiment_id,
        )
        output_path = None
        if definition.save_json:
            output_path = save_experiment_results(definition, summary, run_results)
        return {
            "definition": definition,
            "run_results": run_results,
            "summary": summary,
            "output_path": str(output_path) if output_path is not None else None,
        }

    def _run_single_definition(
        self,
        definition: ExperimentDefinition,
        run_definition: RunDefinition,
    ) -> RunResult:
        """Fuehrt genau einen konkreten Seed-Run aus."""

        effective_definition = self._apply_config_values(definition, run_definition.config_values)
        effective_run_definition = replace(
            run_definition,
            benchmark=effective_definition.benchmark,
            hidden_sizes=effective_definition.hidden_sizes,
            layout_spec=effective_definition.layout_spec,
            run_mode=effective_definition.run_mode,
            primary_metric=effective_definition.primary_metric,
        )
        dataset = load_benchmark(
            DatasetConfig(
                name=effective_definition.benchmark,
                random_state=effective_run_definition.seed,
            )
        )
        layout = parse_layout_spec(
            effective_definition.layout_spec,
            effective_definition.hidden_sizes,
        )

        if effective_definition.run_mode == "manual_training":
            return self._run_manual_training(
                effective_definition,
                effective_run_definition,
                dataset=dataset,
                layout=layout,
            )
        return self._run_simulated_annealing(
            effective_definition,
            effective_run_definition,
            dataset=dataset,
            layout=layout,
        )

    def _apply_config_values(
        self,
        definition: ExperimentDefinition,
        config_values: dict[str, Any],
    ) -> ExperimentDefinition:
        """Wendet eine Search-Konfiguration auf die Builder-Definition an."""

        updated = definition
        for parameter_name, parameter_value in config_values.items():
            if not hasattr(updated, parameter_name):
                continue
            updated = replace(updated, **{parameter_name: parameter_value})
        return updated

    def _run_manual_training(
        self,
        definition: ExperimentDefinition,
        run_definition: RunDefinition,
        dataset,
        layout,
    ) -> RunResult:
        """Fuehrt einen normalen Trainings-Run aus."""

        model = ModularMLP(
            input_size=dataset.input_size,
            hidden_sizes=definition.hidden_sizes,
            output_size=dataset.model_output_size,
            layout=layout,
            num_classes=dataset.output_size,
            weight_scale=definition.weight_scale,
            random_state=run_definition.seed,
        )
        training_result = train_model(
            model,
            dataset,
            TrainingConfig(
                epochs=definition.epochs,
                learning_rate=definition.learning_rate,
                batch_size=definition.batch_size,
                random_state=run_definition.seed,
                shuffle=definition.shuffle,
            ),
        )
        metrics = {
            "train_loss": float(training_result.history["train_loss"][-1]),
            "val_loss": float(training_result.history["val_loss"][-1]),
            "test_loss": float(training_result.test_metrics["loss"]),
            "train_accuracy": float(training_result.history["train_acc"][-1]),
            "val_accuracy": float(training_result.history["val_acc"][-1]),
            "test_accuracy": float(training_result.test_metrics["accuracy"]),
        }
        return RunResult(
            run_definition=run_definition,
            metrics=metrics,
            history=training_result.history,
            layout_spec=layout.to_compact_spec(),
            created_at=utc_timestamp(),
            extra={
                "dataset_sizes": {
                    "train": int(dataset.train_size),
                    "val": int(dataset.validation_size),
                    "test": int(dataset.test_size),
                },
                "effective_parameters": self._effective_parameters_payload(definition),
                "model_state": model.to_state_dict(),
            },
        )

    def _run_simulated_annealing(
        self,
        definition: ExperimentDefinition,
        run_definition: RunDefinition,
        dataset,
        layout,
    ) -> RunResult:
        """Fuehrt einen SA-Run aus und sammelt das beste Ergebnis."""

        if definition.sa_evaluation_mode == "online_delta":
            return self._run_online_delta_annealing(
                definition,
                run_definition,
            )

        evaluator = LayoutObjectiveEvaluator(
            dataset,
            ObjectiveConfig(
                objective_name=definition.objective_name,
                candidate_epochs=definition.candidate_epochs,
                learning_rate=definition.learning_rate,
                batch_size=definition.batch_size,
                weight_scale=definition.weight_scale,
                random_state=run_definition.seed,
                shuffle=definition.shuffle,
            ),
        )
        runner = AnnealingRunner(
            evaluator=evaluator,
            config=AnnealingConfig(
                start_temperature=definition.start_temperature,
                cooling_schedule=definition.cooling_schedule,
                cooling_parameter=definition.cooling_parameter,
                iterations_per_temperature=definition.iterations_per_temperature,
                max_steps=definition.max_steps,
                min_temperature=definition.min_temperature,
                neighborhood_operations=definition.neighborhood_operations,
            ),
            random_state=run_definition.seed,
        )
        state = runner.initialize(layout)
        runner.run_until_complete()
        state = runner.state
        if state is None:
            raise ValueError("Simulated Annealing lieferte keinen Zustand.")

        best_evaluation = state.best_evaluation
        current_evaluation = state.current_evaluation
        stop_reasons = annealing_stop_reasons(
            state=state,
            config=runner.config,
            neighbor_count=len(
                generate_neighbors(
                    current_evaluation.layout,
                    runner.config.neighborhood_operations,
                )
            ),
        )
        metrics = {
            "train_loss": float(best_evaluation.train_loss),
            "val_loss": float(best_evaluation.val_loss),
            "test_loss": float(best_evaluation.test_loss),
            "train_accuracy": float(best_evaluation.train_accuracy),
            "val_accuracy": float(best_evaluation.val_accuracy),
            "test_accuracy": float(best_evaluation.test_accuracy),
            "best_objective": float(best_evaluation.objective_value),
            "acceptance_rate": float(state.acceptance_rate),
            "step_count": float(state.step_index),
        }
        final_layout_evaluation = run_layout_evaluation(
            LayoutEvaluationRequest(
                dataset_config=DatasetConfig(
                    name=definition.benchmark,
                    random_state=run_definition.seed,
                ),
                hidden_sizes=definition.hidden_sizes,
                candidates=build_standard_layout_candidates(
                    definition.hidden_sizes,
                    start_layout_spec=state.start_evaluation.layout.to_compact_spec(),
                    best_layout_spec=best_evaluation.layout.to_compact_spec(),
                    end_layout_spec=current_evaluation.layout.to_compact_spec(),
                    random_state=run_definition.seed,
                ),
                seeds=(run_definition.seed,),
                training_config=TrainingConfig(
                    epochs=definition.epochs,
                    learning_rate=definition.learning_rate,
                    batch_size=definition.batch_size,
                    random_state=run_definition.seed,
                    shuffle=definition.shuffle,
                ),
                weight_scale=definition.weight_scale,
                primary_metric=definition.primary_metric,
            )
        )
        final_best = next(
            (
                item
                for item in final_layout_evaluation.aggregated
                if item.label == "best_layout_from_sa"
            ),
            None,
        )
        if final_best is not None:
            metrics["final_val_loss"] = float(final_best.mean_metrics["val_loss"])
            metrics["final_val_accuracy"] = float(final_best.mean_metrics["val_accuracy"])
            metrics["final_test_loss"] = float(final_best.mean_metrics["test_loss"])
            metrics["final_test_accuracy"] = float(final_best.mean_metrics["test_accuracy"])
        sa_history = [
            {
                "step_index": int(step.step_index),
                "temperature": float(step.temperature),
                "previous_layout_spec": step.previous_layout.to_compact_spec(),
                "candidate_layout_spec": step.candidate_layout.to_compact_spec(),
                "previous_score": float(step.previous_evaluation.comparable_score),
                "candidate_score": float(step.candidate_evaluation.comparable_score),
                "delta": float(step.delta),
                "acceptance_probability": float(step.acceptance_probability),
                "random_draw": float(step.random_draw),
                "accepted": bool(step.accepted),
                "reason_code": step.reason_code,
                "neighbor_label": step.neighbor_label,
                "best_score_after_step": float(step.best_score_after_step),
            }
            for step in state.history
        ]
        return RunResult(
            run_definition=run_definition,
            metrics=metrics,
            history=best_evaluation.training_result.history,
            layout_spec=best_evaluation.layout.to_compact_spec(),
            created_at=utc_timestamp(),
            extra={
                "objective_name": definition.objective_name,
                "effective_parameters": self._effective_parameters_payload(definition),
                "start_layout_spec": state.start_evaluation.layout.to_compact_spec(),
                "best_layout_spec": best_evaluation.layout.to_compact_spec(),
                "end_layout_spec": current_evaluation.layout.to_compact_spec(),
                "current_layout_spec": current_evaluation.layout.to_compact_spec(),
                "start_model_state": state.start_evaluation.trained_model.to_state_dict(),
                "best_model_state": best_evaluation.trained_model.to_state_dict(),
                "current_model_state": current_evaluation.trained_model.to_state_dict(),
                "comparable_score": float(best_evaluation.comparable_score),
                "current_objective": float(current_evaluation.objective_value),
                "best_validation_loss": float(best_evaluation.val_loss),
                "best_validation_accuracy": float(best_evaluation.val_accuracy),
                "stop_reasons": stop_reasons,
                "annealing_config": {
                    "start_temperature": definition.start_temperature,
                    "cooling_schedule": definition.cooling_schedule,
                    "cooling_parameter": definition.cooling_parameter,
                    "iterations_per_temperature": definition.iterations_per_temperature,
                    "max_steps": definition.max_steps,
                    "min_temperature": definition.min_temperature,
                    "neighborhood_operations": list(definition.neighborhood_operations),
                },
                "annealing_history": sa_history,
                "final_layout_evaluation": _layout_evaluation_to_payload(final_layout_evaluation),
            },
        )

    def _run_online_delta_annealing(
        self,
        definition: ExperimentDefinition,
        run_definition: RunDefinition,
    ) -> RunResult:
        """Fuehrt den Online-Delta-SA-Modus mit interleaved Mini-Batch-Training aus."""

        session = create_online_session(
            OnlineAnnealingRequest(
                dataset_config=DatasetConfig(
                    name=definition.benchmark,
                    random_state=run_definition.seed,
                ),
                hidden_sizes=definition.hidden_sizes,
                layout_spec=definition.layout_spec,
                training_config=TrainingConfig(
                    epochs=definition.epochs,
                    learning_rate=definition.learning_rate,
                    batch_size=definition.batch_size,
                    random_state=run_definition.seed,
                    shuffle=definition.shuffle,
                ),
                annealing_config=AnnealingConfig(
                    start_temperature=definition.start_temperature,
                    cooling_schedule=definition.cooling_schedule,
                    cooling_parameter=definition.cooling_parameter,
                    iterations_per_temperature=definition.iterations_per_temperature,
                    max_steps=definition.max_steps,
                    min_temperature=definition.min_temperature,
                    neighborhood_operations=definition.neighborhood_operations,
                ),
                weight_scale=definition.weight_scale,
                random_state=run_definition.seed,
                train_policy=definition.online_train_policy,
            )
        )
        snapshot = online_run_to_completion(session, definition.language)
        if snapshot.best_evaluation is None or snapshot.current_evaluation is None or snapshot.start_evaluation is None:
            raise ValueError("Online-Delta-SA lieferte keinen vollstaendigen Zustand.")

        best_evaluation = snapshot.best_evaluation
        current_evaluation = snapshot.current_evaluation
        final_retrained_layout_evaluation = run_layout_evaluation(
            LayoutEvaluationRequest(
                dataset_config=DatasetConfig(
                    name=definition.benchmark,
                    random_state=run_definition.seed,
                ),
                hidden_sizes=definition.hidden_sizes,
                candidates=build_standard_layout_candidates(
                    definition.hidden_sizes,
                    start_layout_spec=snapshot.start_evaluation.layout.to_compact_spec(),
                    best_layout_spec=best_evaluation.layout.to_compact_spec(),
                    end_layout_spec=current_evaluation.layout.to_compact_spec(),
                    random_state=run_definition.seed,
                ),
                seeds=(run_definition.seed,),
                training_config=TrainingConfig(
                    epochs=definition.epochs,
                    learning_rate=definition.learning_rate,
                    batch_size=definition.batch_size,
                    random_state=run_definition.seed,
                    shuffle=definition.shuffle,
                ),
                weight_scale=definition.weight_scale,
                primary_metric="validation_loss",
            )
        )
        inherited_runs = evaluate_inherited_models(
            DatasetConfig(name=definition.benchmark, random_state=run_definition.seed),
            (
                InheritedModelCandidate(
                    "best_inherited_model_from_sa",
                    best_evaluation.trained_model.to_state_dict(),
                    run_definition.seed,
                ),
                InheritedModelCandidate(
                    "end_inherited_model_from_sa",
                    current_evaluation.trained_model.to_state_dict(),
                    run_definition.seed,
                ),
            ),
            primary_metric="validation_loss",
        )
        final_layout_evaluation = with_inherited_model_evaluations(
            final_retrained_layout_evaluation,
            inherited_runs,
            "validation_loss",
        )
        final_best = next(
            (
                item
                for item in final_retrained_layout_evaluation.aggregated
                if item.label == "best_layout_from_sa"
            ),
            None,
        )
        metrics = {
            "train_loss": float(best_evaluation.train_loss),
            "val_loss": float(best_evaluation.val_loss),
            "test_loss": float(best_evaluation.test_loss),
            "train_accuracy": float(best_evaluation.train_accuracy),
            "val_accuracy": float(best_evaluation.val_accuracy),
            "test_accuracy": float(best_evaluation.test_accuracy),
            "best_objective": float(best_evaluation.val_loss),
            "acceptance_rate": float(snapshot.acceptance_rate),
            "step_count": float(len(snapshot.history)),
        }
        if final_best is not None:
            metrics["final_val_loss"] = float(final_best.mean_metrics["val_loss"])
            metrics["final_val_accuracy"] = float(final_best.mean_metrics["val_accuracy"])
            metrics["final_test_loss"] = float(final_best.mean_metrics["test_loss"])
            metrics["final_test_accuracy"] = float(final_best.mean_metrics["test_accuracy"])

        history = {
            "batch_loss": [float(step.candidate_loss_after) for step in snapshot.history],
            "val_loss": [float(step.validation_loss_after_update) for step in snapshot.history],
            "train_loss": [float(step.previous_evaluation.train_loss) for step in snapshot.history],
            "val_acc": [float(step.previous_evaluation.val_accuracy) for step in snapshot.history],
            "train_acc": [float(step.previous_evaluation.train_accuracy) for step in snapshot.history],
        }
        online_history = [
            {
                "step_index": int(step.step_index),
                "temperature": float(step.temperature),
                "previous_layout_spec": step.previous_layout.to_compact_spec(),
                "candidate_layout_spec": step.candidate_layout.to_compact_spec(),
                "batch_loss_before": float(step.batch_loss_before),
                "candidate_loss_after": float(step.candidate_loss_after),
                "loss_delta": float(step.delta),
                "acceptance_probability": float(step.acceptance_probability),
                "random_draw": float(step.random_draw),
                "accepted": bool(step.accepted),
                "trained_after_accept": bool(step.trained_after_accept),
                "epoch_index": int(step.epoch_index),
                "batch_index": int(step.batch_index),
                "batch_start": int(step.batch_start),
                "validation_loss_after_update": float(step.validation_loss_after_update),
                "reason_code": step.reason_code,
                "neighbor_label": step.neighbor_label,
                "best_score_after_step": float(step.best_score_after_step),
            }
            for step in snapshot.history
        ]
        return RunResult(
            run_definition=run_definition,
            metrics=metrics,
            history=history,
            layout_spec=best_evaluation.layout.to_compact_spec(),
            created_at=utc_timestamp(),
            extra={
                "sa_evaluation_mode": "online_delta",
                "online_train_policy": definition.online_train_policy,
                "objective_name": "batch_loss_delta",
                "effective_parameters": self._effective_parameters_payload(definition),
                "start_layout_spec": snapshot.start_evaluation.layout.to_compact_spec(),
                "best_layout_spec": best_evaluation.layout.to_compact_spec(),
                "end_layout_spec": current_evaluation.layout.to_compact_spec(),
                "current_layout_spec": current_evaluation.layout.to_compact_spec(),
                "start_model_state": snapshot.start_evaluation.trained_model.to_state_dict(),
                "best_model_state": best_evaluation.trained_model.to_state_dict(),
                "current_model_state": current_evaluation.trained_model.to_state_dict(),
                "comparable_score": float(best_evaluation.val_loss),
                "current_objective": float(current_evaluation.objective_value),
                "best_validation_loss": float(best_evaluation.val_loss),
                "best_validation_accuracy": float(best_evaluation.val_accuracy),
                "best_layout_selection_metric": "validation_loss_after_update",
                "stop_reasons": list(snapshot.stop_reasons),
                "annealing_config": {
                    "start_temperature": definition.start_temperature,
                    "cooling_schedule": definition.cooling_schedule,
                    "cooling_parameter": definition.cooling_parameter,
                    "iterations_per_temperature": definition.iterations_per_temperature,
                    "max_steps": definition.max_steps,
                    "min_temperature": definition.min_temperature,
                    "neighborhood_operations": list(definition.neighborhood_operations),
                },
                "annealing_history": online_history,
                "final_layout_evaluation": _layout_evaluation_to_payload(final_layout_evaluation),
                "final_retrained_layout_evaluation": _layout_evaluation_to_payload(final_retrained_layout_evaluation),
                "best_inherited_model_evaluation": _layout_run_to_payload(inherited_runs[0]),
                "end_inherited_model_evaluation": _layout_run_to_payload(inherited_runs[1]),
                "combined_final_comparison": _combined_ranking_to_payload(final_layout_evaluation),
            },
        )

    def _effective_parameters_payload(
        self,
        definition: ExperimentDefinition,
    ) -> dict[str, Any]:
        """Serialisiert die tatsaechlich verwendeten Run-Parameter.

        `config_values` in `RunDefinition` enthalten nur die Search-spezifischen
        Abweichungen. Fuer Debugging und Builder-Analyse ist es hilfreicher,
        jede aufgeloeste Run-Konfiguration zusaetzlich explizit zu speichern.
        """

        return {
            "benchmark": definition.benchmark,
            "hidden_sizes": [int(size) for size in definition.hidden_sizes],
            "layout_spec": definition.layout_spec,
            "run_mode": definition.run_mode,
            "primary_metric": definition.primary_metric,
            "shuffle": bool(definition.shuffle),
            "learning_rate": float(definition.learning_rate),
            "batch_size": int(definition.batch_size),
            "weight_scale": float(definition.weight_scale),
            "epochs": int(definition.epochs),
            "objective_name": definition.objective_name,
            "sa_evaluation_mode": definition.sa_evaluation_mode,
            "online_train_policy": definition.online_train_policy,
            "candidate_epochs": int(definition.candidate_epochs),
            "neighborhood_operations": [str(operation) for operation in definition.neighborhood_operations],
            "start_temperature": float(definition.start_temperature),
            "cooling_schedule": definition.cooling_schedule,
            "cooling_parameter": float(definition.cooling_parameter),
            "iterations_per_temperature": int(definition.iterations_per_temperature),
            "max_steps": int(definition.max_steps),
            "min_temperature": float(definition.min_temperature),
        }


def _layout_evaluation_to_payload(result) -> dict[str, Any]:
    """Serialisiert den finalen Layout-Vergleich fuer JSON-Ergebnisse."""

    return {
        "runs": [
            {
                "label": run.label,
                "evaluation_type": getattr(run, "evaluation_type", "retrained"),
                "layout_spec": run.layout_spec,
                "seed": int(run.seed),
                "metrics": dict(run.metrics),
                "history": dict(run.history),
                "model_state": run.model_state,
            }
            for run in result.runs
        ],
        "retrained_runs": [
            {
                "label": run.label,
                "evaluation_type": getattr(run, "evaluation_type", "retrained"),
                "layout_spec": run.layout_spec,
                "seed": int(run.seed),
                "metrics": dict(run.metrics),
                "history": dict(run.history),
                "model_state": run.model_state,
            }
            for run in result.runs
        ],
        "inherited_runs": [
            {
                "label": run.label,
                "evaluation_type": getattr(run, "evaluation_type", "inherited"),
                "layout_spec": run.layout_spec,
                "seed": int(run.seed),
                "metrics": dict(run.metrics),
                "history": dict(run.history),
                "model_state": run.model_state,
            }
            for run in getattr(result, "inherited_runs", ())
        ],
        "aggregated": [
            {
                "label": item.label,
                "evaluation_type": getattr(item, "evaluation_type", "retrained"),
                "layout_spec": item.layout_spec,
                "num_runs": int(item.num_runs),
                "mean_metrics": dict(item.mean_metrics),
                "std_metrics": dict(item.std_metrics),
                "min_metrics": dict(item.min_metrics),
                "max_metrics": dict(item.max_metrics),
                "ranking_score": float(item.ranking_score),
            }
            for item in result.aggregated
        ],
        "ranking": [
            {
                "label": item.label,
                "evaluation_type": getattr(item, "evaluation_type", "retrained"),
                "layout_spec": item.layout_spec,
                "ranking_score": float(item.ranking_score),
                "mean_metrics": dict(item.mean_metrics),
            }
            for item in result.ranking
        ],
        "combined_ranking": _combined_ranking_to_payload(result),
    }


def _layout_run_to_payload(run) -> dict[str, Any]:
    return {
        "label": run.label,
        "evaluation_type": getattr(run, "evaluation_type", "inherited"),
        "layout_spec": run.layout_spec,
        "seed": int(run.seed),
        "metrics": dict(run.metrics),
        "history": dict(run.history),
        "model_state": run.model_state,
    }


def _combined_ranking_to_payload(result) -> list[dict[str, Any]]:
    return [
        {
            "label": item.label,
            "evaluation_type": getattr(item, "evaluation_type", "retrained"),
            "layout_spec": item.layout_spec,
            "ranking_score": float(item.ranking_score),
            "mean_metrics": dict(item.mean_metrics),
        }
        for item in (getattr(result, "combined_ranking", ()) or result.ranking)
    ]
