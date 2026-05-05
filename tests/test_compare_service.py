from __future__ import annotations

import unittest

from activations import parse_layout_spec
from benchmarks import load_benchmark
from configs import DatasetConfig
from model import ModularMLP
from services.analysis_sample_service import build_analysis_sample
from services.compare_service import build_compare_payload, check_baseline_compatibility


class CompareServiceTests(unittest.TestCase):
    def test_build_compare_payload_contains_sample_predictions_and_diff(self) -> None:
        dataset = load_benchmark(DatasetConfig(name="iris", random_state=3))
        sample = build_analysis_sample(dataset, "val", 0, language="en")
        baseline_model = ModularMLP(
            input_size=dataset.input_size,
            hidden_sizes=(8,),
            output_size=dataset.model_output_size,
            num_classes=dataset.output_size,
            layout=parse_layout_spec("relu", (8,)),
            random_state=3,
        )
        current_model = ModularMLP(
            input_size=dataset.input_size,
            hidden_sizes=(8,),
            output_size=dataset.model_output_size,
            num_classes=dataset.output_size,
            layout=parse_layout_spec("tanh", (8,)),
            random_state=4,
        )

        payload = build_compare_payload(
            current_model,
            baseline_model,
            dataset,
            "en",
            analysis_sample=sample,
        )

        self.assertIsNone(payload.compatibility_issue)
        self.assertEqual(payload.current_benchmark, "iris")
        self.assertTrue(payload.current_prediction)
        self.assertEqual(len(payload.current_probabilities), dataset.output_size)
        self.assertGreaterEqual(payload.natural_neighbor_count, 1)

    def test_check_baseline_compatibility_reports_dimension_mismatch(self) -> None:
        dataset = load_benchmark(DatasetConfig(name="iris", random_state=3))
        baseline_model = ModularMLP(
            input_size=30,
            hidden_sizes=(8,),
            output_size=dataset.model_output_size,
            num_classes=dataset.output_size,
            layout=parse_layout_spec("relu", (8,)),
            random_state=3,
        )
        current_model = ModularMLP(
            input_size=dataset.input_size,
            hidden_sizes=(8,),
            output_size=dataset.model_output_size,
            num_classes=dataset.output_size,
            layout=parse_layout_spec("relu", (8,)),
            random_state=3,
        )

        issue = check_baseline_compatibility(current_model, baseline_model, dataset, language="en")
        self.assertIsNotNone(issue)


if __name__ == "__main__":
    unittest.main()
