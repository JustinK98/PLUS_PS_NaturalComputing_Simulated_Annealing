from __future__ import annotations

import unittest

from cli.parser import build_parser, resolve_command


class CliParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = build_parser()

    def test_gui_subcommand_exposes_only_demo_configuration(self) -> None:
        args = self.parser.parse_args(
            [
                "gui",
                "--profile",
                "tuned_20260530",
            ]
        )
        self.assertEqual(resolve_command(args), "gui")
        self.assertEqual(args.gui_profile, "tuned_20260530")

    def test_gui_rejects_removed_language_and_detail_flags(self) -> None:
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["gui", "--detail-level", "expert"])
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["gui", "--language", "en"])

    def test_gui_subcommand_resolves_without_workspace_selector(self) -> None:
        args = self.parser.parse_args(["gui"])
        self.assertEqual(resolve_command(args), "gui")
        self.assertFalse(hasattr(args, "gui_app_mode"))

    def test_legacy_top_level_gui_aliases_are_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--gui"])
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--mode", "activation_workflow"])
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["gui", "--gui-app-mode", "activation_workflow"])

    def test_removed_gui_workspace_selector_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["gui", "--mode", "activation_workflow"])
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["gui", "--mode", "demo"])
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["gui", "--mode", "playground"])
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["gui", "--mode", "presentation"])
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["gui", "--mode", "experiment_builder"])

    def test_gui_rejects_free_topology_and_initial_layout_overrides(self) -> None:
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["gui", "--hidden-sizes", "16", "16"])
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["gui", "--layout", "tanh"])

    def test_run_subcommand_maps_to_single_run(self) -> None:
        args = self.parser.parse_args(["run", "--benchmark", "two_moons", "--hidden-sizes", "8"])
        self.assertEqual(resolve_command(args), "run")
        self.assertEqual(args.hidden_sizes, [8])

    def test_iris_benchmark_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["run", "--benchmark", "iris"])

    def test_experiment_subcommands_map_correctly(self) -> None:
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
                "--evaluation-profile",
                "tuned_20260530",
                "--export-layout-frames",
            ]
        )
        self.assertEqual(resolve_command(args), "experiment_suite")
        self.assertEqual(args.learning_rate, 2)
        self.assertEqual(args.runs, 3)
        self.assertEqual(args.start_temperature, 0.03)
        self.assertEqual(args.cooling_parameter, 0.95)
        self.assertEqual(args.iterations_per_temperature, 5)
        self.assertEqual(args.min_temperature, 0.001)
        self.assertEqual(args.evaluation_profile, "tuned_20260530")
        self.assertTrue(args.export_layout_frames)

        for exp in ("random-baseline", "all-baseline", "swap-ablation"):
            args = self.parser.parse_args(
                ["experiment", "suite", "--exp", exp, "--benchmark", "two_moons"]
            )
            self.assertEqual(resolve_command(args), "experiment_suite")
            self.assertEqual(args.exp, exp)

        with self.assertRaises(SystemExit):
            self.parser.parse_args(["experiment", "suite", "--exp", "propose"])

        args = self.parser.parse_args(["experiment", "report-online-delta", "--path", "outputs/example"])
        self.assertEqual(resolve_command(args), "experiment_report_online_delta")
        self.assertEqual(args.path, "outputs/example")

        args = self.parser.parse_args(
            [
                "experiment",
                "tune",
                "--benchmark",
                "all",
                "--phase",
                "sa-screen",
                "--workers",
                "2",
                "--resume",
                "outputs/hyperparameter_tuning/example",
                "--export-layout-frames",
            ]
        )
        self.assertEqual(resolve_command(args), "experiment_tune")
        self.assertEqual(args.benchmark, "all")
        self.assertEqual(args.phase, "sa-screen")
        self.assertEqual(args.workers, 2)
        self.assertEqual(args.resume, "outputs/hyperparameter_tuning/example")
        self.assertTrue(args.export_layout_frames)

        for removed_command in ("template", "analyze", "run", "layout-grid", "report"):
            with self.assertRaises(SystemExit):
                self.parser.parse_args(["experiment", removed_command])


if __name__ == "__main__":
    unittest.main()
