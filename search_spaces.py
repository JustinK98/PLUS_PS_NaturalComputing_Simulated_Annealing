"""Diskrete Suchraeume fuer den Experiment Builder.

Diese Datei kapselt alles, was aus einer einzelnen Builder-Konfiguration einen
echten Suchraum macht:

- feste Werte
- explizite Listen
- diskrete Bereiche
- Grid Search
- Random Search

Die Suchraeume bleiben bewusst endlich und diskret. Das passt gut zum
didaktischen Charakter des Projekts und macht Reproduzierbarkeit deutlich
einfacher als kontinuierliche Sampling-Verfahren.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Any

import numpy as np


SUPPORTED_SEARCH_TYPES = ("none", "grid_search", "random_search")
SUPPORTED_SEARCH_VALUE_KINDS = ("fixed", "list", "range")
SUPPORTED_VALUE_TYPES = ("int", "float", "str")


@dataclass(frozen=True)
class SearchValueDefinition:
    """Beschreibt den Suchraum eines einzelnen Parameters."""

    parameter_name: str
    kind: str
    value_type: str
    fixed_value: Any | None = None
    values: tuple[Any, ...] = ()
    range_start: float | int | None = None
    range_stop: float | int | None = None
    range_step: float | int | None = None

    def __post_init__(self) -> None:
        if self.kind not in SUPPORTED_SEARCH_VALUE_KINDS:
            raise ValueError(
                f"Unbekannte Suchraum-Art '{self.kind}'. "
                f"Erlaubt sind: {', '.join(SUPPORTED_SEARCH_VALUE_KINDS)}"
            )
        if self.value_type not in SUPPORTED_VALUE_TYPES:
            raise ValueError(
                f"Unbekannter Werttyp '{self.value_type}'. "
                f"Erlaubt sind: {', '.join(SUPPORTED_VALUE_TYPES)}"
            )
        if self.kind == "fixed" and self.fixed_value is None:
            raise ValueError(f"{self.parameter_name}: fixed benoetigt einen Wert.")
        if self.kind == "list" and not self.values:
            raise ValueError(f"{self.parameter_name}: list benoetigt mindestens einen Wert.")
        if self.kind == "range":
            if self.value_type == "str":
                raise ValueError(f"{self.parameter_name}: range ist fuer Strings nicht erlaubt.")
            if self.range_start is None or self.range_stop is None or self.range_step is None:
                raise ValueError(
                    f"{self.parameter_name}: range benoetigt start, stop und step."
                )
            if self.range_step == 0:
                raise ValueError(f"{self.parameter_name}: range_step darf nicht 0 sein.")

    def candidates(self) -> tuple[Any, ...]:
        """Liefert alle diskreten Kandidatenwerte fuer diesen Parameter."""

        if self.kind == "fixed":
            return (self._cast_value(self.fixed_value),)
        if self.kind == "list":
            return tuple(self._cast_value(value) for value in self.values)

        assert self.range_start is not None
        assert self.range_stop is not None
        assert self.range_step is not None

        candidates: list[Any] = []
        current = float(self.range_start)
        stop = float(self.range_stop)
        step = float(self.range_step)
        epsilon = abs(step) / 1000.0

        if step > 0:
            while current <= stop + epsilon:
                candidates.append(self._cast_value(current))
                current += step
        else:
            while current >= stop - epsilon:
                candidates.append(self._cast_value(current))
                current += step

        if not candidates:
            raise ValueError(f"{self.parameter_name}: range erzeugt keine Kandidaten.")
        return tuple(candidates)

    def _cast_value(self, value: Any) -> Any:
        """Castet einen Rohwert passend zum Werttyp."""

        if self.value_type == "int":
            return int(round(float(value)))
        if self.value_type == "float":
            return float(value)
        return str(value)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SearchValueDefinition":
        """Rekonstruiert eine SearchValueDefinition aus JSON-nahen Daten."""

        values = tuple(payload.get("values", ()))
        return cls(
            parameter_name=str(payload["parameter_name"]),
            kind=str(payload["kind"]),
            value_type=str(payload["value_type"]),
            fixed_value=payload.get("fixed_value"),
            values=values,
            range_start=payload.get("range_start"),
            range_stop=payload.get("range_stop"),
            range_step=payload.get("range_step"),
        )


@dataclass(frozen=True)
class SearchSpaceDefinition:
    """Gesamter Suchraum eines Experiments."""

    search_type: str = "none"
    value_definitions: tuple[SearchValueDefinition, ...] = ()
    random_samples: int = 8
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.search_type not in SUPPORTED_SEARCH_TYPES:
            raise ValueError(
                f"Unbekannter Search-Typ '{self.search_type}'. "
                f"Erlaubt sind: {', '.join(SUPPORTED_SEARCH_TYPES)}"
            )
        if self.random_samples <= 0:
            raise ValueError("random_samples muss positiv sein.")
        if self.random_state < 0:
            raise ValueError("random_state darf nicht negativ sein.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SearchSpaceDefinition":
        """Rekonstruiert einen SearchSpaceDefinition aus JSON-nahen Daten."""

        value_definitions = tuple(
            SearchValueDefinition.from_dict(item)
            for item in payload.get("value_definitions", ())
        )
        return cls(
            search_type=str(payload.get("search_type", "none")),
            value_definitions=value_definitions,
            random_samples=int(payload.get("random_samples", 8)),
            random_state=int(payload.get("random_state", 42)),
        )


def expand_search_space(search_space: SearchSpaceDefinition) -> list[dict[str, Any]]:
    """Expandiert einen Suchraum zu konkreten Konfigurations-Dictionaries."""

    if not search_space.value_definitions:
        return [{}]

    candidate_lists = [
        (definition.parameter_name, definition.candidates())
        for definition in search_space.value_definitions
    ]

    if search_space.search_type == "none":
        return [
            {
                parameter_name: candidates[0]
                for parameter_name, candidates in candidate_lists
            }
        ]

    all_combinations = [
        dict(zip((name for name, _values in candidate_lists), combo, strict=True))
        for combo in itertools.product(*(values for _name, values in candidate_lists))
    ]

    if search_space.search_type == "grid_search":
        return all_combinations

    rng = np.random.default_rng(search_space.random_state)
    sample_count = min(search_space.random_samples, len(all_combinations))
    chosen_indices = rng.choice(len(all_combinations), size=sample_count, replace=False)
    return [all_combinations[int(index)] for index in chosen_indices]
