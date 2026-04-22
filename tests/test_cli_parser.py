from __future__ import annotations

import unittest

from cli.parser import build_parser, resolve_command
from main import _normalize_compat_argv


class CliParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = build_parser()

    def test_gui_subcommand_uses_new_mode_flag(self) -> None:
        args = self.parser.parse_args(
            ["gui", "--mode", "experiment_builder", "--detail-level", "expert", "--language", "en"]
        )
        self.assertEqual(resolve_command(args), "gui")
        self.assertEqual(args.gui_app_mode, "experiment_builder")
        self.assertEqual(args.gui_mode, "expert")
        self.assertEqual(args.gui_language, "en")

    def test_presentation_mode_is_valid_gui_mode(self) -> None:
        args = self.parser.parse_args(["gui", "--mode", "presentation"])
        self.assertEqual(resolve_command(args), "gui")
        self.assertEqual(args.gui_app_mode, "presentation")

        normalized = _normalize_compat_argv(["--mode", "presentation"])
        args = self.parser.parse_args(normalized)
        self.assertEqual(resolve_command(args), "gui")
        self.assertEqual(args.gui_app_mode, "presentation")

    def test_legacy_gui_flag_maps_to_gui_command(self) -> None:
        normalized = _normalize_compat_argv(["--gui", "--gui-app-mode", "playground"])
        args = self.parser.parse_args(normalized)
        self.assertEqual(resolve_command(args), "gui")
        self.assertEqual(args.gui_app_mode, "playground")

    def test_top_level_mode_alias_maps_to_gui_command(self) -> None:
        normalized = _normalize_compat_argv(["--mode", "playground"])
        args = self.parser.parse_args(normalized)
        self.assertEqual(resolve_command(args), "gui")
        self.assertEqual(args.gui_app_mode, "playground")

    def test_run_subcommand_maps_to_single_run(self) -> None:
        args = self.parser.parse_args(["run", "--benchmark", "wine", "--hidden-sizes", "16", "8"])
        self.assertEqual(resolve_command(args), "run")
        self.assertEqual(args.hidden_sizes, [16, 8])

    def test_experiment_subcommands_map_correctly(self) -> None:
        args = self.parser.parse_args(["experiment", "template", "--output", "tmp/example.json"])
        self.assertEqual(resolve_command(args), "experiment_template")

        args = self.parser.parse_args(["experiment", "analyze", "--path", "outputs/example"])
        self.assertEqual(resolve_command(args), "experiment_analyze")

        args = self.parser.parse_args(["experiment", "run", "--config", "tmp/example.json"])
        self.assertEqual(resolve_command(args), "experiment_run")


if __name__ == "__main__":
    unittest.main()
