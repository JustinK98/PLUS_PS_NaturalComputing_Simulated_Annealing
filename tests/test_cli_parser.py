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

    def test_activation_workflow_is_default_gui_mode(self) -> None:
        args = self.parser.parse_args(["gui"])
        self.assertEqual(resolve_command(args), "gui")
        self.assertEqual(args.gui_app_mode, "activation_workflow")

        normalized = _normalize_compat_argv(["--mode", "activation_workflow"])
        args = self.parser.parse_args(normalized)
        self.assertEqual(resolve_command(args), "gui")
        self.assertEqual(args.gui_app_mode, "activation_workflow")

    def test_legacy_gui_flag_maps_to_gui_command(self) -> None:
        normalized = _normalize_compat_argv(["--gui", "--gui-app-mode", "activation_workflow"])
        args = self.parser.parse_args(normalized)
        self.assertEqual(resolve_command(args), "gui")
        self.assertEqual(args.gui_app_mode, "activation_workflow")

    def test_top_level_mode_alias_maps_to_gui_command(self) -> None:
        normalized = _normalize_compat_argv(["--mode", "activation_workflow"])
        args = self.parser.parse_args(normalized)
        self.assertEqual(resolve_command(args), "gui")
        self.assertEqual(args.gui_app_mode, "activation_workflow")

    def test_removed_gui_modes_are_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["gui", "--mode", "demo"])
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["gui", "--mode", "playground"])
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["gui", "--mode", "presentation"])

    def test_run_subcommand_maps_to_single_run(self) -> None:
        args = self.parser.parse_args(["run", "--benchmark", "two_moons", "--hidden-sizes", "8"])
        self.assertEqual(resolve_command(args), "run")
        self.assertEqual(args.hidden_sizes, [8])

    def test_iris_benchmark_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["run", "--benchmark", "iris"])

    def test_experiment_subcommands_map_correctly(self) -> None:
        args = self.parser.parse_args(["experiment", "template", "--output", "tmp/example.json"])
        self.assertEqual(resolve_command(args), "experiment_template")

        args = self.parser.parse_args(["experiment", "analyze", "--path", "outputs/example"])
        self.assertEqual(resolve_command(args), "experiment_analyze")

        args = self.parser.parse_args(["experiment", "run", "--config", "tmp/example.json"])
        self.assertEqual(resolve_command(args), "experiment_run")

        args = self.parser.parse_args(
            [
                "experiment",
                "suite",
                "--exp",
                "online-delta",
                "--benchmark",
                "two_moons",
                "--learning-rate",
                "2",
                "--runs",
                "3",
                "--start-temperature",
                "0.03",
                "--cooling-parameter",
                "0.95",
                "--iterations-per-temperature",
                "5",
                "--min-temperature",
                "0.001",
            ]
        )
        self.assertEqual(resolve_command(args), "experiment_suite")
        self.assertEqual(args.learning_rate, 2)
        self.assertEqual(args.runs, 3)
        self.assertEqual(args.start_temperature, 0.03)
        self.assertEqual(args.cooling_parameter, 0.95)
        self.assertEqual(args.iterations_per_temperature, 5)
        self.assertEqual(args.min_temperature, 0.001)

        for exp in ("random-baseline", "all-baseline", "swap-ablation"):
            args = self.parser.parse_args(
                ["experiment", "suite", "--exp", exp, "--benchmark", "two_moons"]
            )
            self.assertEqual(resolve_command(args), "experiment_suite")
            self.assertEqual(args.exp, exp)

        with self.assertRaises(SystemExit):
            self.parser.parse_args(["experiment", "suite", "--exp", "propose"])

        args = self.parser.parse_args(
            ["experiment", "layout-grid", "--benchmark", "concentric_circles", "--max-candidates", "3"]
        )
        self.assertEqual(resolve_command(args), "experiment_layout_grid")
        self.assertEqual(args.max_candidates, 3)

        args = self.parser.parse_args(["experiment", "report", "--output-dir", "tmp/report_assets"])
        self.assertEqual(resolve_command(args), "experiment_report")
        self.assertEqual(args.output_dir, "tmp/report_assets")

        args = self.parser.parse_args(["experiment", "report-online-delta", "--path", "outputs/example"])
        self.assertEqual(resolve_command(args), "experiment_report_online_delta")
        self.assertEqual(args.path, "outputs/example")


if __name__ == "__main__":
    unittest.main()
