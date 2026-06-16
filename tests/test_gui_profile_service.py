from __future__ import annotations

import unittest

from services.gui_profile_service import METHODICAL_SELECTED_PROFILE, load_gui_benchmark_profile


class GuiProfileServiceTests(unittest.TestCase):
    def test_demo_profile_uses_fast_basics_defaults(self) -> None:
        profile = load_gui_benchmark_profile("demo", "two_moons")

        self.assertEqual(profile["training"]["epochs"], 100)
        self.assertEqual(profile["online_delta"]["max_steps"], 120)
        self.assertEqual(profile["online_delta"]["online_batch_size"], 32)

    def test_legacy_tuned_profile_is_not_exposed_in_gui(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unbekanntes GUI-Profil"):
            load_gui_benchmark_profile("tuned_20260530", "concentric_circles")

    def test_promoted_profile_is_exposed_in_gui(self) -> None:
        profile = load_gui_benchmark_profile("mayer_corrected_20260604", "concentric_circles")

        self.assertEqual(profile["training"]["epochs"], 400)
        self.assertEqual(profile["training"]["batch_size"], 16)
        self.assertEqual(profile["online_delta"]["max_steps"], 50000)
        self.assertEqual(profile["online_delta"]["online_batch_size"], 32)

    def test_methodical_selected_profile_uses_locked_tuning_selection(self) -> None:
        profile = load_gui_benchmark_profile(METHODICAL_SELECTED_PROFILE, "concentric_circles")

        self.assertEqual(profile["training"]["learning_rate"], 0.1)
        self.assertEqual(profile["training"]["epochs"], 150)
        self.assertEqual(profile["online_delta"]["cooling_schedule"], "logarithmic")
        self.assertEqual(profile["online_delta"]["max_steps"], 25000)

    def test_unknown_gui_profile_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unbekanntes GUI-Profil"):
            load_gui_benchmark_profile("missing", "two_moons")


if __name__ == "__main__":
    unittest.main()
