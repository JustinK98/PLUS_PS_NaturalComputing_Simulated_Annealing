from __future__ import annotations

import unittest

from activations import parse_layout_spec
from benchmarks import load_benchmark
from configs import DatasetConfig
from model import ModularMLP
from services.analysis_sample_service import build_analysis_sample
from services.neuron_analysis_service import build_neuron_analysis_payload


class NeuronAnalysisServiceTests(unittest.TestCase):
    def test_build_neuron_analysis_payload_contains_curve_and_terms(self) -> None:
        dataset = load_benchmark(DatasetConfig(name="test_activation", random_state=5))
        sample = build_analysis_sample(dataset, "train", 0, language="en")
        model = ModularMLP(
            input_size=dataset.input_size,
            hidden_sizes=(4, 3),
            output_size=dataset.output_size,
            layout=parse_layout_spec("relu|tanh", (4, 3)),
            random_state=5,
        )

        payload = build_neuron_analysis_payload(model, dataset, sample, 0, 0, language="en")

        self.assertEqual(payload.inspection.layer_index, 0)
        self.assertEqual(payload.title, "L1:n0")
        self.assertTrue(payload.equation_text)
        self.assertGreater(len(payload.curve_x), 50)
        self.assertEqual(len(payload.curve_x), len(payload.curve_y))
        self.assertGreater(len(payload.top_terms), 0)
        self.assertGreater(len(payload.outgoing_weights), 0)


if __name__ == "__main__":
    unittest.main()
