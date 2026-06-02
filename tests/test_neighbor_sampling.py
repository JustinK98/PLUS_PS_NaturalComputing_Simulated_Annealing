from __future__ import annotations

import unittest

import numpy as np

from activations import (
    SUPPORTED_ACTIVATIONS,
    parse_layout_spec,
    random_layout_spec,
    sample_neighbor,
    sample_set_neuron_neighbor,
)


class NeighborSamplingTests(unittest.TestCase):
    def test_random_layout_spec_is_deterministic_and_uses_supported_activations(self) -> None:
        first = random_layout_spec((3, 2), 99)
        second = random_layout_spec((3, 2), 99)

        self.assertEqual(first, second)
        layout = parse_layout_spec(first, (3, 2))
        self.assertEqual(layout.hidden_sizes, (3, 2))
        self.assertEqual(len(layout.flatten()), 5)
        self.assertTrue(all(name in SUPPORTED_ACTIVATIONS for name in layout.flatten()))

    def test_swap_ablation_sampler_can_select_swap_neighbor(self) -> None:
        layout = parse_layout_spec("relu,tanh,relu,tanh", (4,))
        rng = np.random.default_rng(123)

        neighbor = sample_neighbor(
            layout,
            ("swap_neurons",),
            rng,
        )

        self.assertTrue(neighbor.label.startswith("swap:"))
        self.assertEqual(len(neighbor.changes), 2)

    def test_set_neuron_sampler_changes_exactly_one_hidden_neuron(self) -> None:
        layout = parse_layout_spec("relu,tanh|gelu,swish", (2, 2))
        rng = np.random.default_rng(7)

        neighbor = sample_set_neuron_neighbor(layout, rng)

        changed = [
            (layer_index, neuron_index)
            for layer_index, (old_layer, new_layer) in enumerate(
                zip(layout.layers, neighbor.layout.layers)
            )
            for neuron_index, (old_activation, new_activation) in enumerate(
                zip(old_layer, new_layer)
            )
            if old_activation != new_activation
        ]
        self.assertEqual(len(changed), 1)
        changed_layer, changed_neuron = changed[0]
        new_activation = neighbor.layout.layers[changed_layer][changed_neuron]
        self.assertIn(new_activation, SUPPORTED_ACTIVATIONS)
        self.assertNotEqual(new_activation, layout.layers[changed_layer][changed_neuron])

    def test_set_neuron_sampler_hits_hidden_neurons_approximately_uniformly(self) -> None:
        layout = parse_layout_spec("relu,relu,relu,relu", (4,))
        rng = np.random.default_rng(1234)
        counts = [0, 0, 0, 0]

        for _ in range(4000):
            neighbor = sample_set_neuron_neighbor(layout, rng)
            _, _layer_text, neuron_text, _activation = neighbor.label.split(":")
            counts[int(neuron_text)] += 1

        expected = 1000
        for count in counts:
            self.assertLess(abs(count - expected), expected * 0.12)


if __name__ == "__main__":
    unittest.main()
