from __future__ import annotations

import unittest

from services.help_service import program_handbook_html, topic_html, workspace_help_html


class HelpServiceTests(unittest.TestCase):
    def test_program_handbook_contains_global_structure(self) -> None:
        html = program_handbook_html("en")
        self.assertIn("Activation Playground Guide", html)
        self.assertIn("Three workspaces", html)

    def test_topic_html_contains_plot_explanation(self) -> None:
        html = topic_html("training_plot", "de")
        self.assertIn("Trainingsplot", html)
        self.assertIn("Loss", html)

    def test_workspace_help_changes_with_workspace_and_benchmark(self) -> None:
        html = workspace_help_html("playground", "digits", "en")
        self.assertIn("Playground", html)
        self.assertIn("8x8 image input", html)


if __name__ == "__main__":
    unittest.main()
