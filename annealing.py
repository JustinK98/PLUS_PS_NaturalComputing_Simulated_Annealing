"""Kern-Dataclasses und Akzeptanzlogik fuer Simulated Annealing."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from activations import ActivationLayout
from annealing_objectives import ObjectiveEvaluation


SUPPORTED_NEIGHBORHOOD_OPERATIONS = ("set_neuron", "fill_layer", "swap_neurons")


@dataclass(frozen=True)
class AnnealingConfig:
    """Konfiguration des Simulated-Annealing-Laufs."""

    start_temperature: float
    cooling_schedule: str
    cooling_parameter: float
    iterations_per_temperature: int
    max_steps: int
    min_temperature: float
    neighborhood_operations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.start_temperature <= 0.0:
            raise ValueError("start_temperature muss positiv sein.")
        if self.cooling_parameter <= 0.0:
            raise ValueError("cooling_parameter muss positiv sein.")
        if self.iterations_per_temperature <= 0:
            raise ValueError("iterations_per_temperature muss positiv sein.")
        if self.max_steps <= 0:
            raise ValueError("max_steps muss positiv sein.")
        if self.min_temperature < 0.0:
            raise ValueError("min_temperature darf nicht negativ sein.")
        invalid_operations = [
            operation
            for operation in self.neighborhood_operations
            if operation not in SUPPORTED_NEIGHBORHOOD_OPERATIONS
        ]
        if invalid_operations:
            raise ValueError(
                "Unbekannte Nachbarschaftstypen: " + ", ".join(invalid_operations)
            )
        if not self.neighborhood_operations:
            raise ValueError("Mindestens ein Nachbarschaftstyp muss aktiv sein.")


@dataclass
class AnnealingStep:
    """Ein einzelner Simulated-Annealing-Schritt."""

    step_index: int
    temperature: float
    previous_layout: ActivationLayout
    candidate_layout: ActivationLayout
    previous_evaluation: ObjectiveEvaluation
    candidate_evaluation: ObjectiveEvaluation
    delta: float
    acceptance_probability: float
    random_draw: float
    accepted: bool
    reason_code: str
    neighbor_label: str
    best_score_after_step: float
    accepted_steps_after_step: int
    rejected_steps_after_step: int


@dataclass
class AnnealingState:
    """Laufender Zustand eines Simulated-Annealing-Prozesses."""

    start_evaluation: ObjectiveEvaluation
    current_evaluation: ObjectiveEvaluation
    best_evaluation: ObjectiveEvaluation
    current_temperature: float
    step_index: int = 0
    accepted_steps: int = 0
    rejected_steps: int = 0
    accepted_worse_steps: int = 0
    history: list[AnnealingStep] = field(default_factory=list)
    latest_candidate: ObjectiveEvaluation | None = None
    latest_neighbor_label: str = ""

    @property
    def acceptance_rate(self) -> float:
        """Anteil akzeptierter Schritte."""

        if self.step_index == 0:
            return 0.0
        return self.accepted_steps / self.step_index


def acceptance_probability(delta: float, temperature: float) -> float:
    """Akzeptanzwahrscheinlichkeit fuer einen Kandidaten.

    `delta` ist in minimierter Form definiert:
    - `delta <= 0`: Kandidat ist mindestens so gut wie der aktuelle Zustand
    - `delta > 0`: Kandidat ist schlechter
    """

    if delta <= 0.0:
        return 1.0
    if temperature <= 0.0:
        return 0.0
    return math.exp(-delta / temperature)


def annealing_stop_reasons(
    state: AnnealingState,
    config: AnnealingConfig,
    neighbor_count: int,
) -> list[str]:
    """Liefert alle aktuell aktiven Stopgruende."""

    reasons: list[str] = []
    if state.step_index >= config.max_steps:
        reasons.append("max_steps_reached")
    if state.current_temperature <= config.min_temperature:
        reasons.append("temperature_below_threshold")
    if neighbor_count <= 0:
        reasons.append("no_neighbors")
    return reasons
