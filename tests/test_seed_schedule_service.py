from __future__ import annotations

import unittest

from services.seed_schedule_service import build_run_coordinates, build_seed_schedule


class SeedScheduleServiceTests(unittest.TestCase):
    def test_schedule_is_stable(self) -> None:
        first = build_seed_schedule("two_moons", 2, 1)
        second = build_seed_schedule("two_moons", 2, 1)

        self.assertEqual(first, second)

    def test_replicates_share_layout_and_split_but_not_search_or_training_seeds(self) -> None:
        first = build_seed_schedule("two_moons", 2, 0)
        second = build_seed_schedule("two_moons", 2, 1)

        self.assertEqual(first.layout_seed, second.layout_seed)
        self.assertEqual(first.data_split_seed, second.data_split_seed)
        self.assertNotEqual(first.online_weight_seed, second.online_weight_seed)
        self.assertNotEqual(first.online_batch_seed, second.online_batch_seed)
        self.assertNotEqual(first.sa_proposal_seed, second.sa_proposal_seed)
        self.assertNotEqual(first.sa_acceptance_seed, second.sa_acceptance_seed)
        self.assertNotEqual(first.retraining_weight_seed, second.retraining_weight_seed)
        self.assertNotEqual(first.retraining_batch_seed, second.retraining_batch_seed)

    def test_layout_indices_change_layout_and_data_split_seeds(self) -> None:
        first = build_seed_schedule("concentric_circles", 0, 0)
        second = build_seed_schedule("concentric_circles", 1, 0)

        self.assertNotEqual(first.layout_seed, second.layout_seed)
        self.assertNotEqual(first.data_split_seed, second.data_split_seed)

    def test_proposal_and_acceptance_seeds_are_independent(self) -> None:
        schedule = build_seed_schedule("crossing_spirals", 0, 0)

        self.assertNotEqual(schedule.sa_proposal_seed, schedule.sa_acceptance_seed)

    def test_coordinates_form_layout_by_replicate_grid(self) -> None:
        coordinates = build_run_coordinates(2, 3)

        self.assertEqual(len(coordinates), 6)
        self.assertEqual(coordinates[0].run_id, "layout_0000_replicate_00")
        self.assertEqual(coordinates[-1].run_id, "layout_0001_replicate_02")


if __name__ == "__main__":
    unittest.main()
