from __future__ import annotations

import unittest
from pathlib import Path

from services.experiment_service import load_saved_experiment
from services.preview_service import format_experiment_summary, format_run_summary


ROOT = Path(__file__).resolve().parents[1]


class ResultsStoreTests(unittest.TestCase):
    def test_existing_backend_regression_fixture_loads(self) -> None:
        payload = load_saved_experiment(ROOT / "outputs/backend_regression/backend_regression_grid")
        self.assertIn("manifest", payload)
        self.assertIn("summary", payload)
        self.assertGreater(len(payload["runs"]), 0)

    def test_preview_formatters_render_expected_fields(self) -> None:
        payload = load_saved_experiment(ROOT / "outputs/backend_regression/backend_regression_grid")
        summary_text = format_experiment_summary(payload, max_ranking_entries=2)
        self.assertIn("experiment_id:", summary_text)
        self.assertIn("Top ranking entries", summary_text)

        run_text = format_run_summary(payload["runs"][0])
        self.assertIn("run_id:", run_text)
        self.assertIn("metrics", run_text)


if __name__ == "__main__":
    unittest.main()
