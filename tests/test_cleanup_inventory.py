from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.cleanup_inventory import (
    InventoryEntry,
    RetentionRule,
    build_inventory,
    classify_file,
    load_config,
    summarize,
)
from scripts.apply_cleanup_inventory import (
    DEFAULT_CONFIRMATION_TOKEN,
    apply_cleanup_plan,
    build_cleanup_plan,
)
import json


class CleanupInventoryTests(unittest.TestCase):
    def test_first_matching_rule_wins(self) -> None:
        category, reason, pattern = classify_file(
            Path("final/layout_frames/frame_001.png"),
            rules=(
                RetentionRule("final/layout_frames/**", "delete", "regenerable frames"),
                RetentionRule("final/**/*.png", "keep", "generic PNG keep"),
            ),
            default_category="manual_review",
        )

        self.assertEqual(category, "delete")
        self.assertEqual(reason, "regenerable frames")
        self.assertEqual(pattern, "final/layout_frames/**")

    def test_unknown_file_requires_manual_review(self) -> None:
        category, reason, pattern = classify_file(
            Path("unknown/artifact.bin"),
            rules=(),
            default_category="manual_review",
        )

        self.assertEqual(category, "manual_review")
        self.assertIn("Manual review", reason)
        self.assertEqual(pattern, "<default>")

    def test_inventory_marks_only_delete_candidates_as_recoverable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            (output_root / "keep").mkdir()
            (output_root / "delete").mkdir()
            (output_root / "keep" / "summary.csv").write_bytes(b"1234")
            (output_root / "delete" / "frame.png").write_bytes(b"123456")

            entries = build_inventory(
                output_root,
                {
                    "default_category": "manual_review",
                    "rules": (
                        RetentionRule("keep/**", "keep", "reviewed snapshot"),
                        RetentionRule("delete/**", "delete", "regenerable"),
                    ),
                },
            )
            summary = summarize(entries)

        self.assertEqual(len(entries), 2)
        self.assertEqual(summary["total"]["size_bytes"], 10)
        self.assertEqual(summary["total"]["recoverable_bytes"], 6)
        self.assertEqual(summary["categories"]["keep"]["recoverable_bytes"], 0)
        self.assertEqual(summary["categories"]["delete"]["recoverable_bytes"], 6)

    def test_summarize_preserves_all_categories(self) -> None:
        summary = summarize(
            [
                InventoryEntry(
                    path="example.bin",
                    category="archive",
                    reason="historical",
                    matched_pattern="**",
                    size_bytes=3,
                    recoverable_bytes=0,
                )
            ]
        )

        self.assertEqual(set(summary["categories"]), {"keep", "archive", "delete", "manual_review"})
        self.assertEqual(summary["categories"]["archive"]["file_count"], 1)

    def test_reviewed_legacy_output_groups_are_archived(self) -> None:
        config = load_config(Path("configs/artifact_retention.json"))

        suite_category, _, _ = classify_file(
            Path("experiment_suites/20260529_example/aggregate/summary.csv"),
            rules=config["rules"],
            default_category=config["default_category"],
        )
        builder_category, _, _ = classify_file(
            Path("experiments/qt_experiment/summary.json"),
            rules=config["rules"],
            default_category=config["default_category"],
        )

        self.assertEqual(suite_category, "archive")
        self.assertEqual(builder_category, "archive")

    def test_cleanup_apply_defaults_to_non_destructive_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            output_root = workspace / "outputs"
            output_root.mkdir()
            delete_file = output_root / "preview.png"
            archive_file = output_root / "demo" / "history.json"
            archive_file.parent.mkdir()
            delete_file.write_bytes(b"delete")
            archive_file.write_bytes(b"archive")
            inventory_path = workspace / "inventory.json"
            self._write_inventory(
                inventory_path,
                files=(
                    ("preview.png", "delete", delete_file.stat().st_size),
                    ("demo/history.json", "archive", archive_file.stat().st_size),
                ),
            )

            plan = build_cleanup_plan(
                inventory_path,
                Path("outputs_archive/legacy_test"),
                workspace_root=workspace,
            )
            result = apply_cleanup_plan(plan, apply=False, confirmation=None)

            self.assertEqual(result["mode"], "dry_run_only")
            self.assertTrue(delete_file.exists())
            self.assertTrue(archive_file.exists())

    def test_cleanup_apply_requires_confirmation_and_preserves_unclassified_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            output_root = workspace / "outputs"
            output_root.mkdir()
            delete_file = output_root / "preview.png"
            archive_file = output_root / "demo" / "history.json"
            keep_file = output_root / "selected" / "summary.csv"
            archive_file.parent.mkdir()
            keep_file.parent.mkdir()
            delete_file.write_bytes(b"delete")
            archive_file.write_bytes(b"archive")
            keep_file.write_bytes(b"keep")
            inventory_path = workspace / "inventory.json"
            self._write_inventory(
                inventory_path,
                files=(
                    ("preview.png", "delete", delete_file.stat().st_size),
                    ("demo/history.json", "archive", archive_file.stat().st_size),
                    ("selected/summary.csv", "keep", keep_file.stat().st_size),
                ),
            )
            plan = build_cleanup_plan(
                inventory_path,
                Path("outputs_archive/legacy_test"),
                workspace_root=workspace,
            )

            with self.assertRaises(ValueError):
                apply_cleanup_plan(plan, apply=True, confirmation=None)

            result = apply_cleanup_plan(
                plan,
                apply=True,
                confirmation=DEFAULT_CONFIRMATION_TOKEN,
            )

            self.assertEqual(result["mode"], "applied")
            self.assertFalse(delete_file.exists())
            self.assertFalse(archive_file.exists())
            self.assertTrue((workspace / "outputs_archive" / "legacy_test" / "demo" / "history.json").exists())
            self.assertTrue(keep_file.exists())

    def test_cleanup_preflight_rejects_stale_inventory_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            output_root = workspace / "outputs"
            output_root.mkdir()
            first = output_root / "first.bin"
            stale = output_root / "stale.bin"
            first.write_bytes(b"first")
            stale.write_bytes(b"changed")
            inventory_path = workspace / "inventory.json"
            self._write_inventory(
                inventory_path,
                files=(
                    ("first.bin", "delete", first.stat().st_size),
                    ("stale.bin", "delete", 1),
                ),
            )

            with self.assertRaisesRegex(ValueError, "file size changed"):
                build_cleanup_plan(
                    inventory_path,
                    Path("outputs_archive/legacy_test"),
                    workspace_root=workspace,
                )

            self.assertTrue(first.exists())
            self.assertTrue(stale.exists())

    def test_cleanup_preflight_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            (workspace / "outputs").mkdir()
            inventory_path = workspace / "inventory.json"
            self._write_inventory(
                inventory_path,
                files=(("../outside.bin", "delete", 1),),
            )

            with self.assertRaisesRegex(ValueError, "relative path"):
                build_cleanup_plan(
                    inventory_path,
                    Path("outputs_archive/legacy_test"),
                    workspace_root=workspace,
                )

    @staticmethod
    def _write_inventory(
        path: Path,
        *,
        files: tuple[tuple[str, str, int], ...],
    ) -> None:
        path.write_text(
            json.dumps(
                {
                    "mode": "dry_run_only",
                    "output_root": "outputs",
                    "files": [
                        {
                            "path": relative_path,
                            "category": category,
                            "size_bytes": size_bytes,
                        }
                        for relative_path, category, size_bytes in files
                    ],
                }
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
