"""Das eigentliche MLP-Modell des Projekts.

Didaktische Leitideen:

- kein Deep-Learning-Framework, damit Forward- und Backward-Pass lesbar bleiben
- beliebig viele, aber weiterhin kleine Hidden-Layer
- Aktivierungen pro Neuron steuerbar
- Rechenspuren fuer GUI, Stepper und Neuron-Inspektion direkt im Modell
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

try:
    import numpy as np
except ImportError as exc:
    raise ImportError(
        "Dieses Projekt benoetigt NumPy. Installation z. B. mit 'pip install numpy'."
    ) from exc

from activations import (
    ACTIVATION_FORMULAS,
    ACTIVATIONS,
    ActivationLayout,
    apply_activation_derivatives,
    apply_activation_layout,
)


Array = np.ndarray


@dataclass(frozen=True)
class ForwardCache:
    """Zwischenergebnisse eines kompletten Forward-Passes."""

    X: Array
    pre_activations: tuple[Array, ...]
    activations: tuple[Array, ...]
    logits: Array
    probabilities: Array

    @property
    def z1(self) -> Array:
        """Kompatibilitaetszugriff fuer den ersten Hidden-Layer."""

        return self.pre_activations[0]

    @property
    def a1(self) -> Array:
        """Kompatibilitaetszugriff fuer den ersten Hidden-Layer."""

        return self.activations[0]

    @property
    def z2(self) -> Array:
        """Kompatibilitaetszugriff fuer den zweiten Hidden-Layer."""

        return self.pre_activations[1]

    @property
    def a2(self) -> Array:
        """Kompatibilitaetszugriff fuer den zweiten Hidden-Layer."""

        return self.activations[1]


@dataclass(frozen=True)
class WeightedTerm:
    """Ein einzelner Summand einer gewichteten Neuron-Summe."""

    source_label: str
    source_value: float
    weight: float
    contribution: float


@dataclass(frozen=True)
class NeuronInspection:
    """Didaktische Detailansicht eines ausgewaehlten Hidden-Neurons."""

    layer_index: int
    neuron_index: int
    activation_name: str
    activation_formula: str
    bias: float
    pre_activation: float
    output_value: float
    derivative: float
    input_terms: tuple[WeightedTerm, ...]
    top_terms: tuple[WeightedTerm, ...]
    outgoing_weights: tuple[tuple[str, float], ...]
    split_name: str
    sample_index: int


@dataclass(frozen=True)
class ForwardLayerTrace:
    """Schritt eines Forward-Passes fuer einen Hidden-Layer."""

    layer_index: int
    input_values: tuple[float, ...]
    pre_activations: tuple[float, ...]
    activations: tuple[float, ...]
    activation_names: tuple[str, ...]


@dataclass(frozen=True)
class BackwardLayerTrace:
    """Schritt eines Backward-Passes fuer einen Layer."""

    layer_label: str
    deltas: tuple[float, ...]
    weight_gradient_norm: float
    bias_gradient_norm: float


@dataclass(frozen=True)
class SampleTrace:
    """Didaktische Rechenspur eines einzelnen Samples."""

    forward_layers: tuple[ForwardLayerTrace, ...]
    logits: tuple[float, ...]
    probabilities: tuple[float, ...]
    prediction_index: int
    target_index: int | None
    loss: float | None
    output_delta: tuple[float, ...] | None
    backward_layers: tuple[BackwardLayerTrace, ...]


class ModularMLP:
    """Kleines, modulares Mehrschicht-Perzeptron.

    Architektur:
    `Eingabe -> Hidden 1 -> Hidden 2 -> ... -> Output`
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int, ...],
        output_size: int,
        layout: ActivationLayout,
        weight_scale: float = 0.05,
        random_state: int = 42,
    ) -> None:
        if not hidden_sizes:
            raise ValueError("Das Modell erwartet mindestens einen Hidden-Layer.")
        if tuple(hidden_sizes) != layout.hidden_sizes:
            raise ValueError(
                "Hidden-Sizes und Aktivierungs-Layout passen nicht zusammen. "
                f"Erwartet: {hidden_sizes}, erhalten: {layout.hidden_sizes}."
            )

        self.input_size = input_size
        self.hidden_sizes = tuple(hidden_sizes)
        self.output_size = output_size
        self.layout = layout

        rng = np.random.default_rng(random_state)
        layer_sizes = (input_size, *self.hidden_sizes, output_size)
        self.weights: list[Array] = [
            rng.normal(0.0, weight_scale, size=(layer_sizes[index], layer_sizes[index + 1]))
            for index in range(len(layer_sizes) - 1)
        ]
        self.biases: list[Array] = [
            np.zeros(layer_sizes[index + 1], dtype=np.float64)
            for index in range(len(layer_sizes) - 1)
        ]

    @property
    def num_hidden_layers(self) -> int:
        """Anzahl der Hidden-Layer."""

        return len(self.hidden_sizes)

    @property
    def hidden_weights(self) -> tuple[Array, ...]:
        """Gewichtsmatrizen zwischen Eingabe/Hidden-Layern."""

        return tuple(self.weights[:-1])

    @property
    def hidden_biases(self) -> tuple[Array, ...]:
        """Bias-Vektoren der Hidden-Layer."""

        return tuple(self.biases[:-1])

    @property
    def output_weight(self) -> Array:
        """Gewichtsmatrix des Output-Layers."""

        return self.weights[-1]

    @property
    def output_bias(self) -> Array:
        """Bias-Vektor des Output-Layers."""

        return self.biases[-1]

    def clone(self) -> "ModularMLP":
        """Erzeugt eine tiefe Kopie des Modells fuer Vergleichsmodi."""

        cloned_model = ModularMLP(
            input_size=self.input_size,
            hidden_sizes=self.hidden_sizes,
            output_size=self.output_size,
            layout=self.layout,
            weight_scale=1.0,
            random_state=0,
        )
        cloned_model.weights = [weight_matrix.copy() for weight_matrix in self.weights]
        cloned_model.biases = [bias_vector.copy() for bias_vector in self.biases]
        return cloned_model

    @property
    def W1(self) -> Array:
        """Kompatibilitaetszugriff fuer den ersten Hidden-Layer."""

        return self.weights[0]

    @property
    def b1(self) -> Array:
        """Kompatibilitaetszugriff fuer den ersten Hidden-Layer."""

        return self.biases[0]

    @property
    def W2(self) -> Array:
        """Kompatibilitaetszugriff fuer den zweiten Hidden- bzw. letzten Hidden-Ausgang."""

        return self.weights[1]

    @property
    def b2(self) -> Array:
        """Kompatibilitaetszugriff fuer den zweiten Hidden-Layer."""

        return self.biases[1]

    @property
    def W3(self) -> Array:
        """Kompatibilitaetszugriff fuer den Output-Layer in klassischen 2-Layer-Setups."""

        return self.weights[-1]

    @property
    def b3(self) -> Array:
        """Kompatibilitaetszugriff fuer den Output-Layer in klassischen 2-Layer-Setups."""

        return self.biases[-1]

    def forward(self, X: Array) -> ForwardCache:
        """Fuehrt einen kompletten Forward-Pass durch."""

        current_values = X
        pre_activations: list[Array] = []
        activations: list[Array] = []

        for layer_index, layer_layout in enumerate(self.layout.layers):
            z_values = current_values @ self.weights[layer_index] + self.biases[layer_index]
            current_values = apply_activation_layout(z_values, layer_layout)
            pre_activations.append(z_values)
            activations.append(current_values)

        logits = current_values @ self.weights[-1] + self.biases[-1]
        probabilities = softmax(logits)

        return ForwardCache(
            X=X,
            pre_activations=tuple(pre_activations),
            activations=tuple(activations),
            logits=logits,
            probabilities=probabilities,
        )

    def predict_proba(self, X: Array) -> Array:
        """Gibt Klassenwahrscheinlichkeiten fuer alle Beispiele zurueck."""

        return self.forward(X).probabilities

    def predict(self, X: Array) -> Array:
        """Gibt die wahrscheinlichste Klasse pro Beispiel zurueck."""

        return np.argmax(self.predict_proba(X), axis=1)

    def loss_and_gradients(self, X: Array, y: Array) -> tuple[float, dict[str, tuple[Array, ...]]]:
        """Berechnet Loss und alle benoetigten Gradienten."""

        cache = self.forward(X)
        batch_size = X.shape[0]
        loss = cross_entropy_loss(cache.probabilities, y)

        dlogits = cache.probabilities.copy()
        dlogits[np.arange(batch_size), y] -= 1.0
        dlogits /= batch_size

        weight_gradients: list[Array] = [np.zeros_like(weight_matrix) for weight_matrix in self.weights]
        bias_gradients: list[Array] = [np.zeros_like(bias_vector) for bias_vector in self.biases]

        previous_activation = cache.activations[-1]
        weight_gradients[-1] = previous_activation.T @ dlogits
        bias_gradients[-1] = dlogits.sum(axis=0)

        upstream = dlogits @ self.weights[-1].T
        for hidden_index in reversed(range(self.num_hidden_layers)):
            dz = upstream * apply_activation_derivatives(
                cache.pre_activations[hidden_index], self.layout.layers[hidden_index]
            )
            previous_activation = X if hidden_index == 0 else cache.activations[hidden_index - 1]
            weight_gradients[hidden_index] = previous_activation.T @ dz
            bias_gradients[hidden_index] = dz.sum(axis=0)
            if hidden_index > 0:
                upstream = dz @ self.weights[hidden_index].T

        return loss, {
            "weights": tuple(weight_gradients),
            "biases": tuple(bias_gradients),
        }

    def apply_gradients(self, gradients: dict[str, tuple[Array, ...]], learning_rate: float) -> None:
        """Aktualisiert alle Gewichte mit einfachem Gradient Descent."""

        for index, gradient_matrix in enumerate(gradients["weights"]):
            self.weights[index] -= learning_rate * gradient_matrix
        for index, gradient_vector in enumerate(gradients["biases"]):
            self.biases[index] -= learning_rate * gradient_vector

    def evaluate(self, X: Array, y: Array) -> tuple[float, float]:
        """Berechnet Loss und Accuracy fuer einen Datensatzsplit."""

        probabilities = self.predict_proba(X)
        predictions = np.argmax(probabilities, axis=1)
        loss = cross_entropy_loss(probabilities, y)
        accuracy = float(np.mean(predictions == y))
        return loss, accuracy

    def inspect_hidden_neuron(
        self,
        X: Array,
        sample_index: int,
        layer_index: int,
        neuron_index: int,
        split_name: str = "train",
        input_labels: Sequence[str] | None = None,
    ) -> NeuronInspection:
        """Berechnet eine detailreiche Sicht auf ein einzelnes Hidden-Neuron."""

        if layer_index < 0 or layer_index >= self.num_hidden_layers:
            raise ValueError("inspect_hidden_neuron unterstuetzt nur vorhandene Hidden-Layer.")
        if sample_index < 0 or sample_index >= len(X):
            raise ValueError("sample_index liegt ausserhalb des gueltigen Bereichs.")
        if neuron_index < 0 or neuron_index >= self.hidden_sizes[layer_index]:
            raise ValueError("neuron_index liegt ausserhalb des gueltigen Bereichs.")

        sample = X[sample_index : sample_index + 1]
        cache = self.forward(sample)
        activation_name = self.layout.layers[layer_index][neuron_index]

        if layer_index == 0:
            source_values = sample[0]
            weights = self.weights[0][:, neuron_index]
            bias = float(self.biases[0][neuron_index])
            pre_activation = float(cache.pre_activations[0][0, neuron_index])
            output_value = float(cache.activations[0][0, neuron_index])
            if input_labels is None:
                input_labels = tuple(f"x{i}" for i in range(self.input_size))
        else:
            source_values = cache.activations[layer_index - 1][0]
            weights = self.weights[layer_index][:, neuron_index]
            bias = float(self.biases[layer_index][neuron_index])
            pre_activation = float(cache.pre_activations[layer_index][0, neuron_index])
            output_value = float(cache.activations[layer_index][0, neuron_index])
            input_labels = tuple(f"L{layer_index}:n{i}" for i in range(self.hidden_sizes[layer_index - 1]))

        if layer_index + 1 < self.num_hidden_layers:
            outgoing_weights = tuple(
                (
                    f"L{layer_index + 2}:n{target_index}",
                    float(self.weights[layer_index + 1][neuron_index, target_index]),
                )
                for target_index in range(self.weights[layer_index + 1].shape[1])
            )
        else:
            outgoing_weights = tuple(
                (f"y:{target_index}", float(self.weights[-1][neuron_index, target_index]))
                for target_index in range(self.weights[-1].shape[1])
            )

        derivative = float(
            ACTIVATIONS[activation_name].derivative(np.asarray([pre_activation], dtype=np.float64))[0]
        )

        input_terms = tuple(
            WeightedTerm(
                source_label=str(label),
                source_value=float(source_value),
                weight=float(weight),
                contribution=float(source_value * weight),
            )
            for label, source_value, weight in zip(input_labels, source_values, weights, strict=True)
        )
        top_terms = tuple(sorted(input_terms, key=lambda term: abs(term.contribution), reverse=True)[:8])

        return NeuronInspection(
            layer_index=layer_index,
            neuron_index=neuron_index,
            activation_name=activation_name,
            activation_formula=ACTIVATION_FORMULAS[activation_name],
            bias=bias,
            pre_activation=pre_activation,
            output_value=output_value,
            derivative=derivative,
            input_terms=input_terms,
            top_terms=top_terms,
            outgoing_weights=outgoing_weights,
            split_name=split_name,
            sample_index=sample_index,
        )

    def trace_sample(self, X: Array, target_index: int | None = None) -> SampleTrace:
        """Erzeugt eine didaktische Forward/Backward-Rechenspur fuer genau ein Sample."""

        if X.ndim != 2 or len(X) != 1:
            raise ValueError("trace_sample erwartet genau ein Sample mit Form (1, n_features).")

        cache = self.forward(X)
        forward_layers: list[ForwardLayerTrace] = []
        current_input = X[0]
        for layer_index, layer_layout in enumerate(self.layout.layers):
            forward_layers.append(
                ForwardLayerTrace(
                    layer_index=layer_index,
                    input_values=tuple(float(value) for value in current_input),
                    pre_activations=tuple(float(value) for value in cache.pre_activations[layer_index][0]),
                    activations=tuple(float(value) for value in cache.activations[layer_index][0]),
                    activation_names=tuple(layer_layout),
                )
            )
            current_input = cache.activations[layer_index][0]

        probabilities = cache.probabilities[0]
        prediction_index = int(np.argmax(probabilities))
        loss: float | None = None
        output_delta: tuple[float, ...] | None = None
        backward_layers: list[BackwardLayerTrace] = []

        if target_index is not None:
            dlogits = probabilities.copy()
            dlogits[target_index] -= 1.0
            loss = float(-np.log(max(probabilities[target_index], 1e-12)))
            output_delta = tuple(float(value) for value in dlogits)
            backward_layers.append(
                BackwardLayerTrace(
                    layer_label="Output",
                    deltas=output_delta,
                    weight_gradient_norm=float(np.linalg.norm(cache.activations[-1].T @ dlogits.reshape(1, -1))),
                    bias_gradient_norm=float(np.linalg.norm(dlogits)),
                )
            )

            upstream = dlogits.reshape(1, -1) @ self.weights[-1].T
            for hidden_index in reversed(range(self.num_hidden_layers)):
                dz = upstream * apply_activation_derivatives(
                    cache.pre_activations[hidden_index], self.layout.layers[hidden_index]
                )
                previous_activation = X if hidden_index == 0 else cache.activations[hidden_index - 1]
                weight_gradient = previous_activation.T @ dz
                bias_gradient = dz.sum(axis=0)
                backward_layers.append(
                    BackwardLayerTrace(
                        layer_label=f"L{hidden_index + 1}",
                        deltas=tuple(float(value) for value in dz[0]),
                        weight_gradient_norm=float(np.linalg.norm(weight_gradient)),
                        bias_gradient_norm=float(np.linalg.norm(bias_gradient)),
                    )
                )
                if hidden_index > 0:
                    upstream = dz @ self.weights[hidden_index].T

        return SampleTrace(
            forward_layers=tuple(forward_layers),
            logits=tuple(float(value) for value in cache.logits[0]),
            probabilities=tuple(float(value) for value in probabilities),
            prediction_index=prediction_index,
            target_index=target_index,
            loss=loss,
            output_delta=output_delta,
            backward_layers=tuple(backward_layers),
        )


def softmax(logits: Array) -> Array:
    """Numerisch stabile Softmax-Funktion fuer einen ganzen Batch."""

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def cross_entropy_loss(probabilities: Array, y_true: Array) -> float:
    """Mittlere Cross-Entropy ueber einen Batch."""

    clipped = np.clip(probabilities, 1e-12, 1.0)
    correct_class_probabilities = clipped[np.arange(len(y_true)), y_true]
    return float(-np.mean(np.log(correct_class_probabilities)))
