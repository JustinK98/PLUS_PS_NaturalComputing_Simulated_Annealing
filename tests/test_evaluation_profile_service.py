from __future__ import annotations

import unittest

from services.evaluation_profile_service import (
    default_evaluation_profile_name,
    load_evaluation_benchmark_profile,
)


class EvaluationProfileServiceTests(unittest.TestCase):
    def test_pre_correction_profile_is_rejected_for_new_evaluations(self) -> None:
        with self.assertRaisesRegex(ValueError, "legacy_methodology_v1"):
            load_evaluation_benchmark_profile("tuned_20260530", "crossing_spirals")

    def test_promoted_default_profile_loads(self) -> None:
        self.assertEqual(default_evaluation_profile_name(), "mayer_corrected_20260604")

        profile = load_evaluation_benchmark_profile("default", "two_moons")

        self.assertEqual(profile["name"], "mayer_corrected_20260604")
        self.assertEqual(profile["training"]["epochs"], 400)
        self.assertEqual(profile["online_delta"]["max_steps"], 12500)
        self.assertEqual(profile["confirmation"]["runs"], 30)

    def test_unknown_profile_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unbekanntes Evaluationsprofil"):
            load_evaluation_benchmark_profile("missing", "two_moons")


if __name__ == "__main__":
    unittest.main()
