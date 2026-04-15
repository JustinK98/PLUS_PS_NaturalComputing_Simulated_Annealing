from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from experiment_builder import ExperimentDefinition
from search_spaces import SearchSpaceDefinition
from services.experiment_service import (
    default_experiment_definition,
    load_experiment_definition,
    run_experiment_definition,
    save_experiment_definition,
    save_experiment_template,
)


class ExperimentServiceTests(unittest.TestCase):
    def test_template_save_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "template.json"
            save_experiment_template(output_path)
            definition = load_experiment_definition(output_path)
            self.assertIsInstance(definition, ExperimentDefinition)
            self.assertEqual(definition.experiment_id, "example_wine_manual")

    def test_definition_json_roundtrip(self) -> None:
        definition = default_experiment_definition()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "experiment.json"
            save_experiment_definition(definition, output_path)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            reloaded = ExperimentDefinition.from_dict(payload)
            self.assertEqual(reloaded.to_dict(), definition.to_dict())

    def test_run_experiment_definition_manual_smoke(self) -> None:
        definition = replace(
            default_experiment_definition(),
            experiment_id="test_activation_manual_smoke",
            benchmark="test_activation",
            hidden_sizes=(4, 3),
            layout_spec="relu|tanh",
            seeds=(11,),
            save_json=False,
            epochs=2,
            search_space=SearchSpaceDefinition(search_type="none"),
        )
        payload = run_experiment_definition(definition)
        self.assertEqual(payload["summary"].number_of_runs, 1)
        self.assertEqual(len(payload["run_results"]), 1)
        self.assertIn("val_accuracy", payload["run_results"][0].metrics)


if __name__ == "__main__":
    unittest.main()
