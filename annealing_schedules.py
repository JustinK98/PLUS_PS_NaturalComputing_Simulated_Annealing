"""Cooling-Schedules fuer Simulated Annealing.

Diese Datei enthaelt bewusst nur die Temperaturentwicklung.
So bleibt die Kuehllogik getrennt von:

- der Zustandslogik
- der Bewertungsfunktion
- der GUI-Darstellung
"""

from __future__ import annotations

import math


SUPPORTED_COOLING_SCHEDULES = ("geometric", "linear", "logarithmic")


def temperature_for_step(
    schedule_name: str,
    start_temperature: float,
    cooling_parameter: float,
    cooling_step_index: int,
) -> float:
    """Berechnet die Temperatur fuer einen Temperatur-Schritt."""

    _validate_schedule(schedule_name, start_temperature, cooling_parameter, cooling_step_index)

    if schedule_name == "geometric":
        return geometric_cooling(start_temperature, cooling_parameter, cooling_step_index)
    if schedule_name == "linear":
        return linear_cooling(start_temperature, cooling_parameter, cooling_step_index)
    if schedule_name == "logarithmic":
        return logarithmic_cooling(start_temperature, cooling_parameter, cooling_step_index)
    raise ValueError(f"Unbekannter Cooling-Schedule '{schedule_name}'.")


def geometric_cooling(
    start_temperature: float,
    factor: float,
    cooling_step_index: int,
) -> float:
    """Geometrische Abkuehlung: T_k = T0 * factor^k."""

    if not 0.0 < factor < 1.0:
        raise ValueError("Geometric cooling benoetigt einen Faktor zwischen 0 und 1.")
    return start_temperature * (factor**cooling_step_index)


def linear_cooling(
    start_temperature: float,
    decrement: float,
    cooling_step_index: int,
) -> float:
    """Lineare Abkuehlung: T_k = max(0, T0 - decrement * k)."""

    if decrement <= 0.0:
        raise ValueError("Linear cooling benoetigt einen positiven Abzugswert.")
    return max(0.0, start_temperature - decrement * cooling_step_index)


def logarithmic_cooling(
    start_temperature: float,
    scale: float,
    cooling_step_index: int,
) -> float:
    """Logarithmische Abkuehlung mit stabiler Starttemperatur."""

    if scale <= 0.0:
        raise ValueError("Logarithmic cooling benoetigt einen positiven Skalierungswert.")
    return start_temperature / (1.0 + scale * math.log1p(cooling_step_index))


def _validate_schedule(
    schedule_name: str,
    start_temperature: float,
    cooling_parameter: float,
    cooling_step_index: int,
) -> None:
    """Validiert allgemeine Eingaben fuer die Temperaturberechnung."""

    if schedule_name not in SUPPORTED_COOLING_SCHEDULES:
        raise ValueError(
            f"Unbekannter Cooling-Schedule '{schedule_name}'. "
            f"Erlaubt sind: {', '.join(SUPPORTED_COOLING_SCHEDULES)}"
        )
    if start_temperature <= 0.0:
        raise ValueError("Die Starttemperatur muss positiv sein.")
    if cooling_parameter <= 0.0:
        raise ValueError("Der Cooling-Parameter muss positiv sein.")
    if cooling_step_index < 0:
        raise ValueError("cooling_step_index darf nicht negativ sein.")
