from __future__ import annotations

import unittest

from activations import parse_layout_spec
from benchmarks import load_benchmark
from configs import DatasetConfig
from model import ModularMLP
from services.analysis_sample_service import (
    build_analysis_sample,
    build_feature_rows,
    build_test_activation_rows,
    infer_test_activation_target,
)
from services.network_projection_service import build_input_projection


class AnalysisServiceTests(unittest.TestCase):
    def test_build_analysis_sample_from_dataset_split(self) -> None:
        dataset = load_benchmark(DatasetConfig(name="two_moons", random_state=7))
        sample = build_analysis_sample(dataset, "val", 1, language="en")

        self.assertEqual(sample.split_name, "val")
        self.assertFalse(sample.is_custom)
        self.assertEqual(len(sample.raw_sample), dataset.input_size)
        self.assertEqual(len(sample.scaled_sample), dataset.input_size)
        self.assertIn(sample.actual_target_name, dataset.target_names)
        self.assertEqual(sample.actual_target_index, sample.effective_target_index)

    def test_build_analysis_sample_for_custom_test_activation(self) -> None:
        dataset = load_benchmark(DatasetConfig(name="test_activation", random_state=5))
        raw_sample = [1.0, 0.0, 1.0]
        sample = build_analysis_sample(
            dataset,
            "val",
            0,
            language="de",
            use_custom_sample=True,
            custom_raw_sample=raw_sample,
        )

        self.assertTrue(sample.is_custom)
        self.assertEqual(sample.actual_target_index, infer_test_activation_target(sample.raw_sample))
        self.assertEqual(sample.actual_target_index, 1)

    def test_feature_rows_are_sorted_by_scaled_magnitude(self) -> None:
        dataset = load_benchmark(DatasetConfig(name="two_moons", random_state=7))
        sample = build_analysis_sample(dataset, "val", 0, language="en")
        rows = build_feature_rows(dataset, sample, limit=5)

        self.assertEqual(len(rows), min(5, dataset.input_size))
        magnitudes = [abs(row[3]) for row in rows]
        self.assertEqual(magnitudes, sorted(magnitudes, reverse=True))

    def test_test_activation_rows_keep_feature_order(self) -> None:
        dataset = load_benchmark(DatasetConfig(name="test_activation", random_state=5))
        sample = build_analysis_sample(dataset, "train", 0, language="en")
        rows = build_test_activation_rows(dataset, sample)

        self.assertEqual([row[0] for row in rows], list(dataset.feature_names))

    def test_input_projection_prefers_top_terms_for_selected_first_layer(self) -> None:
        dataset = load_benchmark(DatasetConfig(name="crossing_spirals", random_state=3))
        layout = parse_layout_spec("relu|tanh", (16, 16))
        model = ModularMLP(
            input_size=dataset.input_size,
            hidden_sizes=(16, 16),
            output_size=dataset.model_output_size,
            num_classes=dataset.output_size,
            layout=layout,
            random_state=3,
        )
        sample = build_analysis_sample(dataset, "val", 0, language="en")

        projection = build_input_projection(dataset, model, sample, (0, 0), max_inputs=4)

        self.assertEqual(len(projection.displayed_indices), 4)
        self.assertEqual(len(set(projection.displayed_indices)), 4)
        self.assertEqual(projection.total_input_size, dataset.input_size)


if __name__ == "__main__":
    unittest.main()
