"""Kern-Dataclasses und Akzeptanzlogik fuer Simulated Annealing."""

from __future__ import annotations

from dataclasses import dataclass
import math

SUPPORTED_NEIGHBORHOOD_OPERATIONS = ("set_neuron", "swap_neurons")


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
