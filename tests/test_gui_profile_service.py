from __future__ import annotations

import unittest

from services.gui_profile_service import load_gui_benchmark_profile


class GuiProfileServiceTests(unittest.TestCase):
    def test_demo_profile_uses_fast_basics_defaults(self) -> None:
        profile = load_gui_benchmark_profile("demo", "two_moons")

        self.assertEqual(profile["training"]["epochs"], 100)
        self.assertEqual(profile["online_delta"]["max_steps"], 120)
        self.assertEqual(profile["online_delta"]["online_batch_size"], 32)

    def test_tuned_profile_reuses_promoted_evaluation_parameters(self) -> None:
        profile = load_gui_benchmark_profile("tuned_20260530", "concentric_circles")

        self.assertEqual(profile["training"]["batch_size"], 16)
        self.assertEqual(profile["online_delta"]["max_steps"], 480)
        self.assertEqual(profile["online_delta"]["online_batch_size"], 128)

    def test_unknown_gui_profile_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unbekanntes GUI-Profil"):
            load_gui_benchmark_profile("missing", "two_moons")


if __name__ == "__main__":
    unittest.main()
