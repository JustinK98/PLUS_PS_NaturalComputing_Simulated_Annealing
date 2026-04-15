from __future__ import annotations

import unittest

from search_spaces import SearchSpaceDefinition, SearchValueDefinition, expand_search_space


class SearchSpaceTests(unittest.TestCase):
    def test_none_search_returns_first_candidate_only(self) -> None:
        search_space = SearchSpaceDefinition(
            search_type="none",
            value_definitions=(
                SearchValueDefinition(
                    parameter_name="learning_rate",
                    kind="list",
                    value_type="float",
                    values=(0.001, 0.01, 0.1),
                ),
            ),
        )
        self.assertEqual(expand_search_space(search_space), [{"learning_rate": 0.001}])

    def test_grid_search_expands_all_combinations(self) -> None:
        search_space = SearchSpaceDefinition(
            search_type="grid_search",
            value_definitions=(
                SearchValueDefinition(
                    parameter_name="learning_rate",
                    kind="list",
                    value_type="float",
                    values=(0.001, 0.01),
                ),
                SearchValueDefinition(
                    parameter_name="epochs",
                    kind="range",
                    value_type="int",
                    range_start=5,
                    range_stop=10,
                    range_step=5,
                ),
            ),
        )
        configs = expand_search_space(search_space)
        self.assertEqual(
            configs,
            [
                {"learning_rate": 0.001, "epochs": 5},
                {"learning_rate": 0.001, "epochs": 10},
                {"learning_rate": 0.01, "epochs": 5},
                {"learning_rate": 0.01, "epochs": 10},
            ],
        )

    def test_random_search_is_reproducible(self) -> None:
        search_space = SearchSpaceDefinition(
            search_type="random_search",
            random_samples=3,
            random_state=7,
            value_definitions=(
                SearchValueDefinition(
                    parameter_name="batch_size",
                    kind="list",
                    value_type="int",
                    values=(8, 16, 32, 64),
                ),
            ),
        )
        first = expand_search_space(search_space)
        second = expand_search_space(search_space)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
