#!/usr/bin/env python3
"""Build a non-destructive inventory of generated project artifacts.

This script intentionally has no apply mode. It classifies local files and
writes a reviewable report before any cleanup operation is implemented.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any


VALID_CATEGORIES = ("keep", "archive", "delete", "manual_review")
DEFAULT_CONFIG_PATH = Path("configs") / "artifact_retention.json"
DEFAULT_REPORT_DIR = Path("/tmp") / "playground-cleanup-inventory"


@dataclass(frozen=True)
class RetentionRule:
    pattern: str
    category: str
    reason: str


@dataclass(frozen=True)
class InventoryEntry:
    path: str
    category: str
    reason: str
    matched_pattern: str
    size_bytes: int
    recoverable_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify generated outputs without changing any project files."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Retention JSON file. Default: configs/artifact_retention.json",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory for JSON, CSV, and Markdown reports. Default: /tmp/playground-cleanup-inventory",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = payload.get("output_root")
    default_category = payload.get("default_category")
    raw_rules = payload.get("rules")
    if not isinstance(output_root, str) or not output_root:
        raise ValueError("Retention config requires a non-empty string 'output_root'.")
    if default_category not in VALID_CATEGORIES:
        raise ValueError(f"Invalid default category: {default_category!r}")
    if not isinstance(raw_rules, list):
        raise ValueError("Retention config requires a list of 'rules'.")

    rules = []
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict):
            raise ValueError("Each retention rule must be an object.")
        rule = RetentionRule(
            pattern=str(raw_rule.get("pattern", "")),
            category=str(raw_rule.get("category", "")),
            reason=str(raw_rule.get("reason", "")),
        )
        if not rule.pattern or not rule.reason:
            raise ValueError("Each retention rule requires 'pattern' and 'reason'.")
        if rule.category not in VALID_CATEGORIES:
            raise ValueError(f"Invalid category in rule {rule.pattern!r}: {rule.category!r}")
        rules.append(rule)

    return {
        **payload,
        "rules": tuple(rules),
    }


def classify_file(
    relative_path: Path,
    *,
    rules: tuple[RetentionRule, ...],
    default_category: str,
) -> tuple[str, str, str]:
    path_text = relative_path.as_posix()
    for rule in rules:
        if fnmatch(path_text, rule.pattern):
            return rule.category, rule.reason, rule.pattern
    return (
        default_category,
        "No explicit retention rule matched. Manual review is required.",
        "<default>",
    )


def build_inventory(output_root: Path, config: dict[str, Any]) -> list[InventoryEntry]:
    if not output_root.exists():
        return []

    entries = []
    for file_path in sorted(path for path in output_root.rglob("*") if path.is_file()):
        relative_path = file_path.relative_to(output_root)
        category, reason, matched_pattern = classify_file(
            relative_path,
            rules=config["rules"],
            default_category=str(config["default_category"]),
        )
        size_bytes = file_path.stat().st_size
        entries.append(
            InventoryEntry(
                path=relative_path.as_posix(),
                category=category,
                reason=reason,
                matched_pattern=matched_pattern,
                size_bytes=size_bytes,
                recoverable_bytes=size_bytes if category == "delete" else 0,
            )
        )
    return entries


def summarize(entries: list[InventoryEntry]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total": {
            "file_count": len(entries),
            "size_bytes": sum(entry.size_bytes for entry in entries),
            "recoverable_bytes": sum(entry.recoverable_bytes for entry in entries),
        },
        "categories": {},
        "top_level_paths": {},
    }

    for category in VALID_CATEGORIES:
        category_entries = [entry for entry in entries if entry.category == category]
        summary["categories"][category] = {
            "file_count": len(category_entries),
            "size_bytes": sum(entry.size_bytes for entry in category_entries),
            "recoverable_bytes": sum(entry.recoverable_bytes for entry in category_entries),
        }

    top_level_groups: dict[tuple[str, str], list[InventoryEntry]] = defaultdict(list)
    for entry in entries:
        top_level = entry.path.split("/", maxsplit=1)[0]
        top_level_groups[(top_level, entry.category)].append(entry)
    for (top_level, category), grouped_entries in sorted(top_level_groups.items()):
        summary["top_level_paths"].setdefault(top_level, {})[category] = {
            "file_count": len(grouped_entries),
            "size_bytes": sum(entry.size_bytes for entry in grouped_entries),
            "recoverable_bytes": sum(entry.recoverable_bytes for entry in grouped_entries),
        }
    return summary


def write_reports(
    report_dir: Path,
    *,
    config_path: Path,
    output_root: Path,
    config: dict[str, Any],
    entries: list[InventoryEntry],
    summary: dict[str, Any],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    json_payload = {
        "generated_at": generated_at,
        "mode": "dry_run_only",
        "config_path": str(config_path),
        "output_root": str(output_root),
        "config_version": config.get("version"),
        "summary": summary,
        "files": [asdict(entry) for entry in entries],
    }
    (report_dir / "cleanup_inventory.json").write_text(
        json.dumps(json_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with (report_dir / "cleanup_inventory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(InventoryEntry.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(entry) for entry in entries)

    (report_dir / "cleanup_inventory.md").write_text(
        render_markdown_report(
            generated_at=generated_at,
            config_path=config_path,
            output_root=output_root,
            summary=summary,
            entries=entries,
        ),
        encoding="utf-8",
    )


def render_markdown_report(
    *,
    generated_at: str,
    config_path: Path,
    output_root: Path,
    summary: dict[str, Any],
    entries: list[InventoryEntry],
) -> str:
    total = summary["total"]
    lines = [
        "# Cleanup Inventory Dry Run",
        "",
        f"Generated: `{generated_at}`",
        "",
        f"Config: `{config_path}`",
        "",
        f"Output root: `{output_root}`",
        "",
        "This report is non-destructive. No files were moved or deleted.",
        "",
        "## Summary",
        "",
        "| Category | Files | Size | Recoverable after deletion |",
        "| --- | ---: | ---: | ---: |",
    ]
    for category in VALID_CATEGORIES:
        values = summary["categories"][category]
        lines.append(
            f"| `{category}` | {values['file_count']} | "
            f"{format_bytes(values['size_bytes'])} | "
            f"{format_bytes(values['recoverable_bytes'])} |"
        )
    lines.extend(
        [
            f"| **total** | **{total['file_count']}** | "
            f"**{format_bytes(total['size_bytes'])}** | "
            f"**{format_bytes(total['recoverable_bytes'])}** |",
            "",
            "## Top-Level Paths",
            "",
            "| Path | Category | Files | Size | Recoverable after deletion |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for top_level, categories in summary["top_level_paths"].items():
        for category, values in categories.items():
            lines.append(
                f"| `{top_level}` | `{category}` | {values['file_count']} | "
                f"{format_bytes(values['size_bytes'])} | "
                f"{format_bytes(values['recoverable_bytes'])} |"
            )

    lines.extend(
        [
            "",
            "## Largest Delete Candidates",
            "",
            "| Path | Size | Reason |",
            "| --- | ---: | --- |",
        ]
    )
    delete_entries = sorted(
        (entry for entry in entries if entry.category == "delete"),
        key=lambda entry: entry.size_bytes,
        reverse=True,
    )
    for entry in delete_entries[:30]:
        lines.append(f"| `{entry.path}` | {format_bytes(entry.size_bytes)} | {entry.reason} |")

    lines.extend(
        [
            "",
            "## Manual Review Groups",
            "",
            "| Matched rule | Files | Size | Reason |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    grouped_manual: dict[tuple[str, str], list[InventoryEntry]] = defaultdict(list)
    for entry in entries:
        if entry.category == "manual_review":
            grouped_manual[(entry.matched_pattern, entry.reason)].append(entry)
    for (pattern, reason), grouped_entries in sorted(grouped_manual.items()):
        lines.append(
            f"| `{pattern}` | {len(grouped_entries)} | "
            f"{format_bytes(sum(entry.size_bytes for entry in grouped_entries))} | {reason} |"
        )

    lines.extend(
        [
            "",
            "## Rule Match Counts",
            "",
            "| Rule | Category | Files |",
            "| --- | --- | ---: |",
        ]
    )
    counts = Counter((entry.matched_pattern, entry.category) for entry in entries)
    for (pattern, category), count in sorted(counts.items()):
        lines.append(f"| `{pattern}` | `{category}` | {count} |")
    lines.append("")
    return "\n".join(lines)


def format_bytes(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("Unreachable")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_root = Path(str(config["output_root"]))
    entries = build_inventory(output_root, config)
    summary = summarize(entries)
    write_reports(
        args.report_dir,
        config_path=args.config,
        output_root=output_root,
        config=config,
        entries=entries,
        summary=summary,
    )

    print("cleanup inventory: dry run only")
    print(f"output root:       {output_root}")
    print(f"report directory:  {args.report_dir}")
    print(f"files classified:  {summary['total']['file_count']}")
    print(f"total size:        {format_bytes(summary['total']['size_bytes'])}")
    print(f"delete candidates: {summary['categories']['delete']['file_count']}")
    print(f"recoverable size:  {format_bytes(summary['total']['recoverable_bytes'])}")
    for category in VALID_CATEGORIES:
        values = summary["categories"][category]
        print(
            f"{category:13} {values['file_count']:6d} files  "
            f"{format_bytes(values['size_bytes']):>10}"
        )


if __name__ == "__main__":
    main()
