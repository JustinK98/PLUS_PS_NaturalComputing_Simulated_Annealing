"""Aktivierungsfunktionen, Layout-Repraesentation und Neighbor-Operationen.

Diese Datei definiert die "Experiment-Sprache" des Projekts:

- welche Aktivierungsfunktionen verfuegbar sind
- wie Layouts fuer beliebig viele Hidden-Layer beschrieben werden
- wie Layouts veraendert, verglichen und als Nachbarn erzeugt werden

Gerade fuer das Natural-Computing-Thema ist diese Ebene wichtig, weil ein
Aktivierungs-Layout spaeter als Zustand in einem Suchraum interpretiert werden
kann.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

try:
    import numpy as np
except ImportError as exc:
    raise ImportError(
        "Dieses Projekt benoetigt NumPy. Installation z. B. mit 'pip install numpy'."
    ) from exc

from configs import SUPPORTED_ACTIVATIONS


Array = np.ndarray
LEAKY_RELU_SLOPE = 0.01


@dataclass(frozen=True)
class ActivationFunction:
    """Kapselt Aktivierungsfunktion und Ableitung."""

    name: str
    forward: Callable[[Array], Array]
    derivative: Callable[[Array], Array]


def relu(values: Array) -> Array:
    """Klassische ReLU: negative Werte werden zu 0."""

    return np.maximum(0.0, values)


def relu_derivative(values: Array) -> Array:
    """Ableitung der ReLU."""

    return (values > 0.0).astype(np.float64)


def tanh(values: Array) -> Array:
    """Hyperbolische Tangensfunktion mit Ausgaben in (-1, 1)."""

    return np.tanh(values)


def tanh_derivative(values: Array) -> Array:
    """Ableitung von tanh."""

    tanh_values = np.tanh(values)
    return 1.0 - tanh_values**2


def sigmoid(values: Array) -> Array:
    """Sigmoid-Funktion mit Clipping fuer numerische Stabilitaet."""

    clipped = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def sigmoid_derivative(values: Array) -> Array:
    """Ableitung der Sigmoid-Funktion."""

    sigmoid_values = sigmoid(values)
    return sigmoid_values * (1.0 - sigmoid_values)


def leaky_relu(values: Array) -> Array:
    """Leaky ReLU: negative Werte behalten eine kleine Steigung."""

    return np.where(values > 0.0, values, LEAKY_RELU_SLOPE * values)


def leaky_relu_derivative(values: Array) -> Array:
    """Ableitung der Leaky ReLU."""

    return np.where(values > 0.0, 1.0, LEAKY_RELU_SLOPE)


ACTIVATIONS = {
    "relu": ActivationFunction("relu", relu, relu_derivative),
    "tanh": ActivationFunction("tanh", tanh, tanh_derivative),
    "sigmoid": ActivationFunction("sigmoid", sigmoid, sigmoid_derivative),
    "leaky_relu": ActivationFunction("leaky_relu", leaky_relu, leaky_relu_derivative),
}

ACTIVATION_SHORT_NAMES = {
    "relu": "REL",
    "tanh": "TAN",
    "sigmoid": "SIG",
    "leaky_relu": "LRE",
}

ACTIVATION_COLORS = {
    "relu": "#3b82f6",
    "tanh": "#ef4444",
    "sigmoid": "#10b981",
    "leaky_relu": "#f59e0b",
}

ACTIVATION_FORMULAS = {
    "relu": "a = max(0, z)",
    "tanh": "a = tanh(z)",
    "sigmoid": "a = 1 / (1 + exp(-z))",
    "leaky_relu": f"a = z falls z > 0, sonst {LEAKY_RELU_SLOPE} * z",
}


@dataclass(frozen=True)
class ActivationLayout:
    """Aktivierungs-Layout fuer eine beliebige Anzahl von Hidden-Layern.

    Beispiel fuer drei Hidden-Layer:
    `(("relu", "relu"), ("tanh", "sigmoid"), ("relu",))`
    """

    layers: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        """Validiert Struktur und Aktivierungsnamen direkt beim Erzeugen."""

        if not self.layers:
            raise ValueError("Ein Layout braucht mindestens einen Hidden-Layer.")
        for layer_index, layer in enumerate(self.layers, start=1):
            if not layer:
                raise ValueError(
                    f"Hidden-Layer L{layer_index} muss mindestens ein Neuron besitzen."
                )
            for activation_name in layer:
                validate_activation_name(activation_name)

    @property
    def hidden_sizes(self) -> tuple[int, ...]:
        """Liefert die Neuronenzahlen aller Hidden-Layer."""

        return tuple(len(layer) for layer in self.layers)

    @property
    def num_hidden_layers(self) -> int:
        """Anzahl der Hidden-Layer."""

        return len(self.layers)

    def replace_neuron(
        self, layer_index: int, neuron_index: int, activation_name: str
    ) -> "ActivationLayout":
        """Erzeugt ein neues Layout mit geaenderter Aktivierung eines Neurons."""

        validate_activation_name(activation_name)
        _validate_position(self, layer_index, neuron_index)
        new_layers = [list(layer) for layer in self.layers]
        new_layers[layer_index][neuron_index] = activation_name
        return ActivationLayout(tuple(tuple(layer) for layer in new_layers))

    def replace_layer(self, layer_index: int, activation_name: str) -> "ActivationLayout":
        """Ersetzt einen kompletten Hidden-Layer durch eine Aktivierung."""

        validate_activation_name(activation_name)
        _validate_layer_index(self, layer_index)
        new_layers = [list(layer) for layer in self.layers]
        new_layers[layer_index] = [activation_name] * len(new_layers[layer_index])
        return ActivationLayout(tuple(tuple(layer) for layer in new_layers))

    def swap_neurons(
        self, layer_index: int, first_index: int, second_index: int
    ) -> "ActivationLayout":
        """Tauscht zwei Aktivierungen im selben Layer."""

        _validate_position(self, layer_index, first_index)
        _validate_position(self, layer_index, second_index)
        new_layers = [list(layer) for layer in self.layers]
        new_layers[layer_index][first_index], new_layers[layer_index][second_index] = (
            new_layers[layer_index][second_index],
            new_layers[layer_index][first_index],
        )
        return ActivationLayout(tuple(tuple(layer) for layer in new_layers))

    def cycle_neuron(self, layer_index: int, neuron_index: int) -> "ActivationLayout":
        """Schaltet ein Neuron zur naechsten Aktivierung weiter."""

        _validate_position(self, layer_index, neuron_index)
        current_name = self.layers[layer_index][neuron_index]
        current_position = SUPPORTED_ACTIVATIONS.index(current_name)
        next_name = SUPPORTED_ACTIVATIONS[(current_position + 1) % len(SUPPORTED_ACTIVATIONS)]
        return self.replace_neuron(layer_index, neuron_index, next_name)

    def with_hidden_sizes(self, hidden_sizes: Sequence[int]) -> "ActivationLayout":
        """Passt ein Layout moeglichst sanft an neue Hidden-Sizes an.

        Regeln:
        - bestehende Layer bleiben so weit wie moeglich erhalten
        - fehlt ein Layer, wird der letzte bekannte Aktivierungsstil wiederverwendet
        - wachsen Layer, wird die letzte bekannte Aktivierung fortgeschrieben
        - schrumpfen Layer, werden sie abgeschnitten
        """

        if not hidden_sizes:
            raise ValueError("Es muss mindestens ein Hidden-Layer vorhanden sein.")

        new_layers: list[tuple[str, ...]] = []
        last_known_activation = "relu"
        for layer_index, layer_size in enumerate(hidden_sizes):
            if layer_size <= 0:
                raise ValueError("Alle Hidden-Layer muessen mindestens ein Neuron haben.")
            if layer_index < len(self.layers):
                source_layer = list(self.layers[layer_index])
                last_known_activation = source_layer[-1]
            else:
                source_layer = [last_known_activation]

            if len(source_layer) < layer_size:
                source_layer.extend([source_layer[-1]] * (layer_size - len(source_layer)))
            else:
                source_layer = source_layer[:layer_size]

            new_layers.append(tuple(source_layer))
            last_known_activation = source_layer[-1]

        return ActivationLayout(tuple(new_layers))

    def add_layer(
        self,
        layer_size: int,
        activation_name: str = "relu",
        position: int | None = None,
    ) -> "ActivationLayout":
        """Fuegt einen Hidden-Layer an einer Position hinzu."""

        if layer_size <= 0:
            raise ValueError("Ein neuer Hidden-Layer braucht mindestens ein Neuron.")
        validate_activation_name(activation_name)
        new_layer = tuple([activation_name] * layer_size)
        new_layers = list(self.layers)
        insert_position = len(new_layers) if position is None else position
        if insert_position < 0 or insert_position > len(new_layers):
            raise ValueError("Die Einfuegeposition fuer den neuen Layer ist ungueltig.")
        new_layers.insert(insert_position, new_layer)
        return ActivationLayout(tuple(new_layers))

    def remove_layer(self, layer_index: int) -> "ActivationLayout":
        """Entfernt einen Hidden-Layer, sofern mindestens einer uebrig bleibt."""

        _validate_layer_index(self, layer_index)
        if self.num_hidden_layers == 1:
            raise ValueError("Das Layout muss mindestens einen Hidden-Layer behalten.")
        new_layers = list(self.layers)
        new_layers.pop(layer_index)
        return ActivationLayout(tuple(new_layers))

    def flatten(self) -> tuple[str, ...]:
        """Legt alle Layer hintereinander in ein flaches Tupel."""

        return tuple(name for layer in self.layers for name in layer)

    def to_compact_spec(self) -> str:
        """Wandelt ein Layout wieder in die kompakte String-Syntax um."""

        return "|".join(_compress_layer(layer) for layer in self.layers)


@dataclass(frozen=True)
class LayoutChange:
    """Beschreibt eine einzelne Aenderung zwischen zwei Layouts."""

    layer_index: int
    neuron_index: int
    before: str
    after: str
    operation: str

    def describe(self) -> str:
        """Kurze menschenlesbare Beschreibung."""

        return f"L{self.layer_index + 1}:n{self.neuron_index} {self.before} -> {self.after}"


@dataclass(frozen=True)
class NeighborResult:
    """Ergebnis einer Nachbar-Erzeugung oder Neighbor-Operation."""

    label: str
    layout: ActivationLayout
    changes: tuple[LayoutChange, ...]


def validate_activation_name(activation_name: str) -> None:
    """Prueft, ob ein Aktivierungsname im Projekt erlaubt ist."""

    if activation_name not in ACTIVATIONS:
        supported = ", ".join(SUPPORTED_ACTIVATIONS)
        raise ValueError(
            f"Nicht unterstuetzte Aktivierung '{activation_name}'. Erlaubt sind: {supported}"
        )


def parse_layout_spec(layout_spec: str, hidden_sizes: Sequence[int]) -> ActivationLayout:
    """Parst eine Layoutbeschreibung fuer beliebig viele Hidden-Layer.

    Regeln:
    - `relu` spiegelt sich auf alle Hidden-Layer
    - `relu|tanh|sigmoid` beschreibt drei Hidden-Layer explizit
    - `relu*8,tanh*8` beschreibt einen Layer explizit pro Neuron/Block
    """

    if not hidden_sizes:
        raise ValueError("Es muss mindestens eine Hidden-Layer-Groesse angegeben werden.")

    cleaned = layout_spec.strip().lower().replace(" ", "")
    if not cleaned:
        raise ValueError("Der Layout-String darf nicht leer sein.")

    raw_layers = cleaned.split("|")
    if len(raw_layers) == 1:
        raw_layers = raw_layers * len(hidden_sizes)
    if len(raw_layers) != len(hidden_sizes):
        raise ValueError(
            f"Das Layout beschreibt {len(raw_layers)} Hidden-Layer, "
            f"das Modell erwartet aber {len(hidden_sizes)}."
        )

    parsed_layers = []
    for raw_layer, layer_size in zip(raw_layers, hidden_sizes, strict=True):
        parsed_layers.append(tuple(_parse_layer_spec(raw_layer, layer_size)))
    return ActivationLayout(tuple(parsed_layers))


def apply_activation_layout(values: Array, layer_layout: Sequence[str]) -> Array:
    """Wendet moeglicherweise unterschiedliche Aktivierungen pro Neuron an."""

    if values.ndim != 2:
        raise ValueError("Die Aktivierungsanwendung erwartet eine 2D-Matrix.")
    if values.shape[1] != len(layer_layout):
        raise ValueError("Anzahl der Spalten und Layout-Laenge muessen uebereinstimmen.")

    activated = np.empty_like(values, dtype=np.float64)
    for neuron_index, activation_name in enumerate(layer_layout):
        activated[:, neuron_index] = ACTIVATIONS[activation_name].forward(values[:, neuron_index])
    return activated


def apply_activation_derivatives(values: Array, layer_layout: Sequence[str]) -> Array:
    """Berechnet Aktivierungs-Ableitungen pro Neuronsspalte."""

    if values.ndim != 2:
        raise ValueError("Die Ableitungsanwendung erwartet eine 2D-Matrix.")
    if values.shape[1] != len(layer_layout):
        raise ValueError("Anzahl der Spalten und Layout-Laenge muessen uebereinstimmen.")

    derivatives = np.empty_like(values, dtype=np.float64)
    for neuron_index, activation_name in enumerate(layer_layout):
        derivatives[:, neuron_index] = ACTIVATIONS[activation_name].derivative(
            values[:, neuron_index]
        )
    return derivatives


def diff_layouts(base_layout: ActivationLayout, other_layout: ActivationLayout) -> tuple[LayoutChange, ...]:
    """Vergleicht zwei Layouts neuronweise und liefert alle Unterschiede."""

    if base_layout.hidden_sizes != other_layout.hidden_sizes:
        raise ValueError("Nur Layouts mit gleichen Hidden-Sizes koennen verglichen werden.")

    changes: list[LayoutChange] = []
    for layer_index, (base_layer, other_layer) in enumerate(
        zip(base_layout.layers, other_layout.layers, strict=True)
    ):
        for neuron_index, (before, after) in enumerate(zip(base_layer, other_layer, strict=True)):
            if before != after:
                changes.append(
                    LayoutChange(
                        layer_index=layer_index,
                        neuron_index=neuron_index,
                        before=before,
                        after=after,
                        operation="diff",
                    )
                )
    return tuple(changes)


def apply_neighbor_operations(layout: ActivationLayout, operations: Sequence[str]) -> NeighborResult:
    """Wendet eine oder mehrere Neighbor-Operationen nacheinander an."""

    current_layout = layout
    all_changes: list[LayoutChange] = []
    for operation in operations:
        current_layout, changes = _apply_single_neighbor_operation(current_layout, operation)
        all_changes.extend(changes)

    label = " | ".join(operations) if operations else "identity"
    return NeighborResult(label=label, layout=current_layout, changes=tuple(all_changes))


def generate_single_step_neighbors(layout: ActivationLayout) -> list[NeighborResult]:
    """Erzeugt alle Nachbarn im Hamming-Abstand 1."""

    neighbors: list[NeighborResult] = []
    for layer_index, layer in enumerate(layout.layers):
        for neuron_index, current_activation in enumerate(layer):
            for candidate in SUPPORTED_ACTIVATIONS:
                if candidate == current_activation:
                    continue
                change = LayoutChange(
                    layer_index=layer_index,
                    neuron_index=neuron_index,
                    before=current_activation,
                    after=candidate,
                    operation="single_step",
                )
                neighbors.append(
                    NeighborResult(
                        label=f"set:L{layer_index + 1}:{neuron_index}:{candidate}",
                        layout=layout.replace_neuron(layer_index, neuron_index, candidate),
                        changes=(change,),
                    )
                )
    return neighbors


def _parse_layer_spec(layer_spec: str, layer_size: int) -> list[str]:
    """Parst die Spezifikation eines einzelnen Hidden-Layers."""

    if not layer_spec:
        raise ValueError("Jeder Hidden-Layer braucht eine nicht-leere Aktivierungsangabe.")

    if "," not in layer_spec and "*" not in layer_spec:
        validate_activation_name(layer_spec)
        return [layer_spec] * layer_size

    expanded: list[str] = []
    for token in layer_spec.split(","):
        if not token:
            raise ValueError("Ungueltiges Layout: leerer Token zwischen Kommata gefunden.")
        activation_name, repeat_count = _parse_activation_token(token)
        expanded.extend([activation_name] * repeat_count)

    if len(expanded) != layer_size:
        raise ValueError(
            f"Layer-Spezifikation '{layer_spec}' expandiert zu {len(expanded)} Neuronen, "
            f"der Layer erwartet aber {layer_size}."
        )
    return expanded


def _parse_activation_token(token: str) -> tuple[str, int]:
    """Parst ein einzelnes Token wie `relu` oder `tanh*8`."""

    if "*" not in token:
        validate_activation_name(token)
        return token, 1

    activation_name, count_text = token.split("*", maxsplit=1)
    validate_activation_name(activation_name)
    try:
        repeat_count = int(count_text)
    except ValueError as exc:
        raise ValueError(f"Ungueltige Wiederholungszahl im Token '{token}'.") from exc
    if repeat_count <= 0:
        raise ValueError(f"Die Wiederholungszahl in '{token}' muss positiv sein.")
    return activation_name, repeat_count


def _parse_layer_reference(layer_token: str, layout: ActivationLayout) -> int:
    """Wandelt `L1`, `L2`, ... in den internen Layer-Index um."""

    cleaned = layer_token.strip().upper()
    if cleaned.startswith("L"):
        cleaned = cleaned[1:]
    try:
        layer_index = int(cleaned) - 1
    except ValueError as exc:
        raise ValueError(
            f"Ungueltige Layer-Referenz '{layer_token}'. Verwende L1, L2, L3, ..."
        ) from exc
    _validate_layer_index(layout, layer_index)
    return layer_index


def _validate_layer_index(layout: ActivationLayout, layer_index: int) -> None:
    """Prueft, ob ein Layer-Index im Layout existiert."""

    if layer_index < 0 or layer_index >= layout.num_hidden_layers:
        valid_layers = ", ".join(f"L{index + 1}" for index in range(layout.num_hidden_layers))
        raise ValueError(f"Gueltig sind nur die Layer {valid_layers}.")


def _validate_position(layout: ActivationLayout, layer_index: int, neuron_index: int) -> None:
    """Prueft, ob Layer- und Neuron-Index in ein Layout passen."""

    _validate_layer_index(layout, layer_index)
    if neuron_index < 0 or neuron_index >= len(layout.layers[layer_index]):
        raise ValueError(
            f"Neuron-Index {neuron_index} liegt ausserhalb des Bereichs von L{layer_index + 1}."
        )


def _apply_single_neighbor_operation(
    layout: ActivationLayout, operation: str
) -> tuple[ActivationLayout, tuple[LayoutChange, ...]]:
    """Wendet genau eine Neighbor-Operation an."""

    parts = operation.strip().split(":")
    operator = parts[0].lower()

    if operator == "set" and len(parts) == 4:
        layer_index = _parse_layer_reference(parts[1], layout)
        neuron_index = int(parts[2])
        activation_name = parts[3].lower()
        validate_activation_name(activation_name)
        _validate_position(layout, layer_index, neuron_index)
        before = layout.layers[layer_index][neuron_index]
        updated_layout = layout.replace_neuron(layer_index, neuron_index, activation_name)
        changes: tuple[LayoutChange, ...] = ()
        if before != activation_name:
            changes = (
                LayoutChange(layer_index, neuron_index, before, activation_name, operation="set"),
            )
        return updated_layout, changes

    if operator == "fill" and len(parts) == 3:
        layer_index = _parse_layer_reference(parts[1], layout)
        activation_name = parts[2].lower()
        validate_activation_name(activation_name)
        updated_layout = layout.replace_layer(layer_index, activation_name)
        changes = tuple(
            LayoutChange(layer_index, neuron_index, before, activation_name, "fill")
            for neuron_index, before in enumerate(layout.layers[layer_index])
            if before != activation_name
        )
        return updated_layout, changes

    if operator == "cycle" and len(parts) == 3:
        layer_index = _parse_layer_reference(parts[1], layout)
        neuron_index = int(parts[2])
        _validate_position(layout, layer_index, neuron_index)
        before = layout.layers[layer_index][neuron_index]
        updated_layout = layout.cycle_neuron(layer_index, neuron_index)
        after = updated_layout.layers[layer_index][neuron_index]
        return updated_layout, (
            LayoutChange(layer_index, neuron_index, before, after, "cycle"),
        )

    if operator == "swap" and len(parts) == 4:
        layer_index = _parse_layer_reference(parts[1], layout)
        first_index = int(parts[2])
        second_index = int(parts[3])
        _validate_position(layout, layer_index, first_index)
        _validate_position(layout, layer_index, second_index)
        before_first = layout.layers[layer_index][first_index]
        before_second = layout.layers[layer_index][second_index]
        updated_layout = layout.swap_neurons(layer_index, first_index, second_index)
        changes: list[LayoutChange] = []
        after_first = updated_layout.layers[layer_index][first_index]
        after_second = updated_layout.layers[layer_index][second_index]
        if before_first != after_first:
            changes.append(
                LayoutChange(layer_index, first_index, before_first, after_first, "swap")
            )
        if before_second != after_second:
            changes.append(
                LayoutChange(layer_index, second_index, before_second, after_second, "swap")
            )
        return updated_layout, tuple(changes)

    raise ValueError(
        "Nicht unterstuetzte Neighbor-Operation. Beispiele: "
        "set:L1:0:tanh, fill:L2:sigmoid, cycle:L1:3, swap:L2:0:4"
    )


def _compress_layer(layer: Sequence[str]) -> str:
    """Komprimiert einen Layer wieder in die kompakte String-Syntax."""

    parts: list[str] = []
    current_name = layer[0]
    count = 1

    for activation_name in layer[1:]:
        if activation_name == current_name:
            count += 1
            continue
        parts.append(_compress_run(current_name, count))
        current_name = activation_name
        count = 1

    parts.append(_compress_run(current_name, count))
    return ",".join(parts)


def _compress_run(activation_name: str, count: int) -> str:
    """Komprimiert einen zusammenhaengenden Lauf gleicher Aktivierungen."""

    if count == 1:
        return activation_name
    return f"{activation_name}*{count}"
