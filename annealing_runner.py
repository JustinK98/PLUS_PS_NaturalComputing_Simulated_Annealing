"""Laufsteuerung fuer Simulated Annealing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from activations import generate_neighbors, sample_neighbor
from annealing import AnnealingConfig, AnnealingState, AnnealingStep, acceptance_probability
from annealing_objectives import LayoutObjectiveEvaluator
from annealing_schedules import temperature_for_step


@dataclass(frozen=True)
class RunnerSnapshot:
    """Kompakte Sicht auf einen SA-Lauf fuer GUI und Tests."""

    current_layout_spec: str
    best_layout_spec: str
    current_objective: float
    best_objective: float
    temperature: float
    step_index: int
    acceptance_rate: float
    cache_size: int


class AnnealingRunner:
    """Fuehrt einen Simulated-Annealing-Lauf ueber Aktivierungs-Layouts aus."""

    def __init__(
        self,
        evaluator: LayoutObjectiveEvaluator,
        config: AnnealingConfig,
        random_state: int,
    ) -> None:
        self.evaluator = evaluator
        self.config = config
        self.rng = np.random.default_rng(random_state)
        self.state: AnnealingState | None = None

    def initialize(self, initial_layout) -> AnnealingState:
        """Bewertet den Startzustand und initialisiert den Lauf."""

        start_evaluation = self.evaluator.evaluate(initial_layout)
        self.state = AnnealingState(
            start_evaluation=start_evaluation,
            current_evaluation=start_evaluation,
            best_evaluation=start_evaluation,
            current_temperature=self.config.start_temperature,
        )
        return self.state

    def step(self) -> AnnealingStep:
        """Fuehrt genau einen Annealing-Schritt aus."""

        if self.state is None:
            raise ValueError("AnnealingRunner muss zuerst mit initialize(...) gestartet werden.")

        previous_evaluation = self.state.current_evaluation
        current_layout = previous_evaluation.layout
        neighbors = generate_neighbors(current_layout, self.config.neighborhood_operations)
        if not neighbors:
            raise ValueError("Fuer das aktuelle Layout wurden keine Nachbarn erzeugt.")

        if self.config.operation_probabilities or self.config.activation_probabilities:
            chosen_neighbor = sample_neighbor(
                current_layout,
                self.config.neighborhood_operations,
                self.rng,
                operation_probabilities=self.config.operation_probabilities,
                activation_probabilities=self.config.activation_probabilities,
            )
        else:
            chosen_neighbor = neighbors[int(self.rng.integers(0, len(neighbors)))]
        candidate_evaluation = self.evaluator.evaluate(chosen_neighbor.layout)

        current_score = previous_evaluation.comparable_score
        candidate_score = candidate_evaluation.comparable_score
        delta = candidate_score - current_score

        current_temperature = self._temperature_for_iteration(self.state.step_index)
        probability = acceptance_probability(delta, current_temperature)
        random_draw = float(self.rng.random())
        accepted = delta <= 0.0 or random_draw <= probability

        reason_code = "improved_or_equal"
        if delta > 0.0:
            reason_code = "accepted_worse" if accepted else "rejected_worse"

        if accepted:
            self.state.current_evaluation = candidate_evaluation
            self.state.accepted_steps += 1
            if delta > 0.0:
                self.state.accepted_worse_steps += 1
        else:
            self.state.rejected_steps += 1

        if candidate_evaluation.comparable_score < self.state.best_evaluation.comparable_score:
            self.state.best_evaluation = candidate_evaluation

        self.state.step_index += 1
        self.state.current_temperature = self._temperature_for_iteration(self.state.step_index)
        self.state.latest_candidate = candidate_evaluation
        self.state.latest_neighbor_label = chosen_neighbor.label

        step_record = AnnealingStep(
            step_index=self.state.step_index,
            temperature=current_temperature,
            previous_layout=current_layout,
            candidate_layout=chosen_neighbor.layout,
            previous_evaluation=previous_evaluation,
            candidate_evaluation=candidate_evaluation,
            delta=float(delta),
            acceptance_probability=float(probability),
            random_draw=random_draw,
            accepted=accepted,
            reason_code=reason_code,
            neighbor_label=chosen_neighbor.label,
            best_score_after_step=float(self.state.best_evaluation.comparable_score),
            accepted_steps_after_step=self.state.accepted_steps,
            rejected_steps_after_step=self.state.rejected_steps,
        )

        self.state.history.append(step_record)
        return step_record

    def run_steps(self, step_count: int) -> list[AnnealingStep]:
        """Fuehrt mehrere Schritte nacheinander aus."""

        if step_count <= 0:
            return []
        steps: list[AnnealingStep] = []
        for _ in range(step_count):
            if self.should_stop():
                break
            steps.append(self.step())
        return steps

    def run_until_complete(self) -> list[AnnealingStep]:
        """Laeuft bis zum Stopkriterium weiter."""

        steps: list[AnnealingStep] = []
        while not self.should_stop():
            steps.append(self.step())
        return steps

    def should_stop(self) -> bool:
        """Prueft die grundlegenden Stopkriterien."""

        if self.state is None:
            return True
        if self.state.step_index >= self.config.max_steps:
            return True
        if self.state.current_temperature <= self.config.min_temperature:
            return True
        neighbors = generate_neighbors(
            self.state.current_evaluation.layout,
            self.config.neighborhood_operations,
        )
        return len(neighbors) == 0

    def snapshot(self) -> RunnerSnapshot:
        """Erzeugt eine kompakte Zusammenfassung fuer GUI und Tests."""

        if self.state is None:
            raise ValueError("Es ist noch kein SA-Lauf initialisiert.")
        return RunnerSnapshot(
            current_layout_spec=self.state.current_evaluation.layout.to_compact_spec(),
            best_layout_spec=self.state.best_evaluation.layout.to_compact_spec(),
            current_objective=self.state.current_evaluation.objective_value,
            best_objective=self.state.best_evaluation.objective_value,
            temperature=self.state.current_temperature,
            step_index=self.state.step_index,
            acceptance_rate=self.state.acceptance_rate,
            cache_size=self.evaluator.cache_size(),
        )

    def _temperature_for_iteration(self, iteration_index: int) -> float:
        """Berechnet die fuer eine Iteration gueltige Temperatur."""

        cooling_step_index = iteration_index // self.config.iterations_per_temperature
        return temperature_for_step(
            schedule_name=self.config.cooling_schedule,
            start_temperature=self.config.start_temperature,
            cooling_parameter=self.config.cooling_parameter,
            cooling_step_index=cooling_step_index,
        )
