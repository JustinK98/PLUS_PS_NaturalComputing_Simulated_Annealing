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


def gelu(values: Array) -> Array:
    """GELU-Approximation nach Hendrycks/Gimpel mit tanh-Form."""

    coefficient = np.sqrt(2.0 / np.pi)
    inner = coefficient * (values + 0.044715 * values**3)
    return 0.5 * values * (1.0 + np.tanh(inner))


def gelu_derivative(values: Array) -> Array:
    """Ableitung der tanh-basierten GELU-Approximation."""

    coefficient = np.sqrt(2.0 / np.pi)
    inner = coefficient * (values + 0.044715 * values**3)
    tanh_inner = np.tanh(inner)
    sech_squared = 1.0 - tanh_inner**2
    inner_derivative = coefficient * (1.0 + 3.0 * 0.044715 * values**2)
    return 0.5 * (1.0 + tanh_inner) + 0.5 * values * sech_squared * inner_derivative


def sigmoid(values: Array) -> Array:
    """Sigmoid-Funktion mit Clipping fuer numerische Stabilitaet."""

    clipped = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def sigmoid_derivative(values: Array) -> Array:
    """Ableitung der Sigmoid-Funktion."""

    sigmoid_values = sigmoid(values)
    return sigmoid_values * (1.0 - sigmoid_values)


def swish(values: Array) -> Array:
    """Swish: glatte, nichtlineare Aktivierung mit Sigmoid-Gating."""

    return values * sigmoid(values)


def swish_derivative(values: Array) -> Array:
    """Ableitung von Swish."""

    sigmoid_values = sigmoid(values)
    return sigmoid_values + values * sigmoid_values * (1.0 - sigmoid_values)


def identity(values: Array) -> Array:
    """Identity-Aktivierung als lineare Durchleitung."""

    return values


def identity_derivative(values: Array) -> Array:
    """Ableitung der Identity-Aktivierung."""

    return np.ones_like(values, dtype=np.float64)


ACTIVATIONS = {
    "relu": ActivationFunction("relu", relu, relu_derivative),
    "gelu": ActivationFunction("gelu", gelu, gelu_derivative),
    "sigmoid": ActivationFunction("sigmoid", sigmoid, sigmoid_derivative),
    "tanh": ActivationFunction("tanh", tanh, tanh_derivative),
    "swish": ActivationFunction("swish", swish, swish_derivative),
    "identity": ActivationFunction("identity", identity, identity_derivative),
}

ACTIVATION_SHORT_NAMES = {
    "relu": "REL",
    "gelu": "GEL",
    "sigmoid": "SIG",
    "tanh": "TAN",
    "swish": "SWH",
    "identity": "IDN",
}

ACTIVATION_COLORS = {
    "relu": "#3b82f6",
    "gelu": "#a855f7",
    "sigmoid": "#10b981",
    "tanh": "#ef4444",
    "swish": "#8b5cf6",
    "identity": "#64748b",
}

ACTIVATION_FORMULAS = {
    "relu": "a = max(0, z)",
    "gelu": "a = 0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715*z^3)))",
    "sigmoid": "a = 1 / (1 + exp(-z))",
    "tanh": "a = tanh(z)",
    "swish": "a = z / (1 + exp(-z))",
    "identity": "a = z",
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


def random_layout_spec(
    hidden_sizes: Sequence[int],
    random_state: int | np.random.Generator,
) -> str:
    """Erzeugt ein reproduzierbares random-mixed Layout fuer Hidden-Layer."""

    if not hidden_sizes:
        raise ValueError("Es muss mindestens eine Hidden-Layer-Groesse angegeben werden.")
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(int(random_state))
    )
    layers: list[str] = []
    for layer_size in hidden_sizes:
        if layer_size <= 0:
            raise ValueError("Alle Hidden-Layer muessen mindestens ein Neuron haben.")
        layer = rng.choice(SUPPORTED_ACTIVATIONS, size=int(layer_size), replace=True)
        layers.append(",".join(str(name) for name in layer))
    return "|".join(layers)


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


def generate_swap_neighbors(layout: ActivationLayout) -> list[NeighborResult]:
    """Erzeugt Nachbarn, die zwei Aktivierungen innerhalb eines Layers tauschen."""

    neighbors: list[NeighborResult] = []
    for layer_index, layer in enumerate(layout.layers):
        for first_index in range(len(layer)):
            for second_index in range(first_index + 1, len(layer)):
                if layer[first_index] == layer[second_index]:
                    continue
                neighbors.append(
                    NeighborResult(
                        label=f"swap:L{layer_index + 1}:{first_index}:{second_index}",
                        layout=layout.swap_neurons(layer_index, first_index, second_index),
                        changes=(
                            LayoutChange(
                                layer_index=layer_index,
                                neuron_index=first_index,
                                before=layer[first_index],
                                after=layer[second_index],
                                operation="swap",
                            ),
                            LayoutChange(
                                layer_index=layer_index,
                                neuron_index=second_index,
                                before=layer[second_index],
                                after=layer[first_index],
                                operation="swap",
                            ),
                        ),
                    )
                )
    return neighbors


def generate_neighbors(
    layout: ActivationLayout,
    operations: Sequence[str],
) -> list[NeighborResult]:
    """Erzeugt Nachbarn fuer ausgewaehlte Nachbarschaftstypen.

    Unterstuetzte Operationen:
    - `set_neuron`: ein einzelnes Neuron auf eine andere Aktivierung setzen
    - `swap_neurons`: zwei Aktivierungen im selben Layer tauschen
    """

    supported_operations = {"set_neuron", "swap_neurons"}
    normalized_operations = tuple(dict.fromkeys(operation for operation in operations if operation))
    invalid_operations = [operation for operation in normalized_operations if operation not in supported_operations]
    if invalid_operations:
        raise ValueError(
            "Unbekannte Nachbarschaftstypen: " + ", ".join(invalid_operations)
        )

    neighbors: list[NeighborResult] = []
    if "set_neuron" in normalized_operations:
        neighbors.extend(generate_single_step_neighbors(layout))
    if "swap_neurons" in normalized_operations:
        neighbors.extend(generate_swap_neighbors(layout))
    return neighbors


def sample_neighbor(
    layout: ActivationLayout,
    operations: Sequence[str],
    rng: np.random.Generator,
) -> NeighborResult:
    """Zieht einen konkreten Nachbarn gleichverteilt."""

    neighbors = generate_neighbors(layout, operations)
    if not neighbors:
        raise ValueError("Fuer das aktuelle Layout wurden keine Nachbarn erzeugt.")
    return neighbors[int(rng.integers(0, len(neighbors)))]


def sample_set_neuron_neighbor(
    layout: ActivationLayout,
    rng: np.random.Generator,
) -> NeighborResult:
    """Zieht genau eine einzelne Hidden-Neuron-Aenderung.

    Alle Hidden-Neuronen haben dieselbe Chance, danach wird eine andere
    Aktivierung aus dem offiziellen Aktivierungsset gleichverteilt gezogen.
    """

    total_neurons = sum(len(layer) for layer in layout.layers)
    if total_neurons <= 0:
        raise ValueError("Das Layout enthaelt keine Hidden-Neuronen.")

    flat_index = int(rng.integers(0, total_neurons))
    remaining = flat_index
    layer_index = 0
    neuron_index = 0
    for current_layer_index, layer in enumerate(layout.layers):
        if remaining < len(layer):
            layer_index = current_layer_index
            neuron_index = remaining
            break
        remaining -= len(layer)

    current_activation = layout.layers[layer_index][neuron_index]
    alternatives = [
        activation_name
        for activation_name in SUPPORTED_ACTIVATIONS
        if activation_name != current_activation
    ]
    if not alternatives:
        raise ValueError("Keine alternative Aktivierung verfuegbar.")
    next_activation = alternatives[int(rng.integers(0, len(alternatives)))]
    updated_layout = layout.replace_neuron(layer_index, neuron_index, next_activation)
    change = LayoutChange(
        layer_index=layer_index,
        neuron_index=neuron_index,
        before=current_activation,
        after=next_activation,
        operation="set_neuron",
    )
    return NeighborResult(
        label=f"set:L{layer_index + 1}:{neuron_index}:{next_activation}",
        layout=updated_layout,
        changes=(change,),
    )


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
