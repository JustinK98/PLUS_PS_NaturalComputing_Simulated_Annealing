from __future__ import annotations

import unittest

from activations import parse_layout_spec
from benchmarks import load_benchmark
from configs import DatasetConfig
from model import ModularMLP
from services.analysis_sample_service import build_analysis_sample
from services.stepper_service import build_step_entries


class StepperServiceTests(unittest.TestCase):
    def test_build_step_entries_contains_forward_and_backward_sections(self) -> None:
        dataset = load_benchmark(DatasetConfig(name="test_activation", random_state=7))
        sample = build_analysis_sample(dataset, "train", 0, language="en")
        model = ModularMLP(
            input_size=dataset.input_size,
            hidden_sizes=(4, 3),
            output_size=dataset.output_size,
            layout=parse_layout_spec("relu|tanh", (4, 3)),
            random_state=7,
        )
        trace = model.trace_sample(
            sample.scaled_sample.reshape(1, -1),
            target_index=sample.effective_target_index,
        )

        entries = build_step_entries(trace, dataset, sample, language="en")
        titles = [entry.title for entry in entries]

        self.assertIn("Input", titles)
        self.assertIn("Output", titles)
        self.assertIn("Loss", titles)
        self.assertTrue(any(title.startswith("Forward L1") for title in titles))
        self.assertTrue(any(title.startswith("Backward ") for title in titles))


if __name__ == "__main__":
    unittest.main()
