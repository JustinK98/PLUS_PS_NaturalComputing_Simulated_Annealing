#!/usr/bin/env python3
"""Apply a reviewed output inventory with an explicit destructive opt-in.

The default mode is a dry run. Mutations require both ``--apply`` and the
literal confirmation token. A complete preflight runs before the first file is
deleted or moved, so stale inventories fail without partial cleanup.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any


DEFAULT_INVENTORY_PATH = Path("/tmp") / "playground-cleanup-inventory" / "cleanup_inventory.json"
DEFAULT_CONFIRMATION_TOKEN = "DELETE-AND-ARCHIVE-CLASSIFIED-OUTPUTS"
ACTIONABLE_CATEGORIES = ("delete", "archive")


@dataclass(frozen=True)
class CleanupAction:
    """One reviewed filesystem operation."""

    category: str
    relative_path: str
    source_path: str
    destination_path: str | None
    size_bytes: int


@dataclass(frozen=True)
class CleanupPlan:
    """Validated operations derived from an inventory snapshot."""

    inventory_path: str
    output_root: str
    archive_root: str
    actions: tuple[CleanupAction, ...]
    skipped_missing: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or apply reviewed output cleanup. Default is a non-destructive dry run."
        )
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_INVENTORY_PATH,
        help="Inventory JSON from scripts/cleanup_inventory.py.",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("outputs_archive") / f"legacy_{date.today():%Y%m%d}",
        help="Destination for entries classified as archive.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Execute the preflighted file mutations.",
    )
    parser.add_argument(
        "--confirm",
        default=None,
        help=f"Required with --apply: {DEFAULT_CONFIRMATION_TOKEN}",
    )
    parser.add_argument(
        "--plan-log",
        type=Path,
        default=None,
        help="Optional JSON path for the generated plan or applied-action log.",
    )
    return parser.parse_args()


def load_inventory(inventory_path: Path) -> dict[str, Any]:
    """Load and minimally validate one generated cleanup inventory."""

    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    if payload.get("mode") != "dry_run_only":
        raise ValueError("Cleanup inventory must come from the dry-run inventory script.")
    if not isinstance(payload.get("output_root"), str) or not payload["output_root"]:
        raise ValueError("Cleanup inventory requires a non-empty output_root.")
    if not isinstance(payload.get("files"), list):
        raise ValueError("Cleanup inventory requires a files list.")
    return payload


def build_cleanup_plan(
    inventory_path: Path,
    archive_root: Path,
    *,
    workspace_root: Path | None = None,
) -> CleanupPlan:
    """Build a mutation plan and reject stale or unsafe inventory entries."""

    payload = load_inventory(inventory_path)
    workspace = (workspace_root or Path.cwd()).resolve()
    output_root = _resolve_workspace_path(workspace, Path(str(payload["output_root"])))
    resolved_archive_root = _resolve_workspace_path(workspace, archive_root)
    _validate_archive_root(workspace, output_root, resolved_archive_root)

    actions: list[CleanupAction] = []
    skipped_missing: list[str] = []
    errors: list[str] = []
    for raw_entry in payload["files"]:
        if not isinstance(raw_entry, dict):
            errors.append("Inventory contains a non-object file entry.")
            continue
        category = str(raw_entry.get("category", ""))
        if category not in ACTIONABLE_CATEGORIES:
            continue
        relative_path_text = str(raw_entry.get("path", ""))
        try:
            relative_path = _validate_relative_path(relative_path_text)
            source_path = _resolve_inside(output_root, relative_path)
        except ValueError as exc:
            errors.append(f"{relative_path_text!r}: {exc}")
            continue
        if not source_path.exists():
            skipped_missing.append(relative_path.as_posix())
            continue
        if not source_path.is_file():
            errors.append(f"{relative_path.as_posix()!r}: source is not a regular file.")
            continue
        expected_size = int(raw_entry.get("size_bytes", -1))
        actual_size = source_path.stat().st_size
        if actual_size != expected_size:
            errors.append(
                f"{relative_path.as_posix()!r}: file size changed "
                f"from {expected_size} to {actual_size} bytes."
            )
            continue
        destination_path: Path | None = None
        if category == "archive":
            destination_path = _resolve_inside(resolved_archive_root, relative_path)
            if destination_path.exists():
                errors.append(
                    f"{relative_path.as_posix()!r}: archive destination already exists."
                )
                continue
        actions.append(
            CleanupAction(
                category=category,
                relative_path=relative_path.as_posix(),
                source_path=str(source_path),
                destination_path=str(destination_path) if destination_path is not None else None,
                size_bytes=actual_size,
            )
        )
    if errors:
        raise ValueError("Cleanup preflight failed:\n- " + "\n- ".join(errors))
    return CleanupPlan(
        inventory_path=str(inventory_path.resolve()),
        output_root=str(output_root),
        archive_root=str(resolved_archive_root),
        actions=tuple(actions),
        skipped_missing=tuple(skipped_missing),
    )


def apply_cleanup_plan(
    plan: CleanupPlan,
    *,
    apply: bool,
    confirmation: str | None,
) -> dict[str, Any]:
    """Return a dry-run summary or execute one fully preflighted cleanup plan."""

    if apply and confirmation != DEFAULT_CONFIRMATION_TOKEN:
        raise ValueError(
            "Destructive cleanup requires --confirm "
            f"{DEFAULT_CONFIRMATION_TOKEN}"
        )

    deleted_files = 0
    archived_files = 0
    deleted_bytes = 0
    archived_bytes = 0
    if apply:
        for action in plan.actions:
            source = Path(action.source_path)
            if action.category == "delete":
                source.unlink()
                deleted_files += 1
                deleted_bytes += action.size_bytes
            else:
                destination = Path(str(action.destination_path))
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
                archived_files += 1
                archived_bytes += action.size_bytes

    planned_delete = [action for action in plan.actions if action.category == "delete"]
    planned_archive = [action for action in plan.actions if action.category == "archive"]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "applied" if apply else "dry_run_only",
        "inventory_path": plan.inventory_path,
        "output_root": plan.output_root,
        "archive_root": plan.archive_root,
        "planned": {
            "delete_files": len(planned_delete),
            "delete_bytes": sum(action.size_bytes for action in planned_delete),
            "archive_files": len(planned_archive),
            "archive_bytes": sum(action.size_bytes for action in planned_archive),
            "skipped_missing_files": len(plan.skipped_missing),
        },
        "applied": {
            "deleted_files": deleted_files,
            "deleted_bytes": deleted_bytes,
            "archived_files": archived_files,
            "archived_bytes": archived_bytes,
        },
        "skipped_missing": list(plan.skipped_missing),
        "actions": [asdict(action) for action in plan.actions],
    }


def write_plan_log(path: Path, payload: dict[str, Any]) -> None:
    """Write one reviewable JSON plan or applied-action log."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def format_bytes(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def _resolve_workspace_path(workspace_root: Path, path: Path) -> Path:
    return (path if path.is_absolute() else workspace_root / path).resolve()


def _validate_archive_root(workspace_root: Path, output_root: Path, archive_root: Path) -> None:
    if archive_root == output_root:
        raise ValueError("Archive root must not equal output_root.")
    if not archive_root.is_relative_to(workspace_root):
        raise ValueError("Archive root must stay inside the workspace.")
    if archive_root.is_relative_to(output_root):
        raise ValueError("Archive root must stay outside output_root.")


def _validate_relative_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path_text or path.is_absolute() or ".." in path.parts:
        raise ValueError("inventory path must be a relative path without '..'.")
    return path


def _resolve_inside(root: Path, relative_path: Path) -> Path:
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("resolved path escapes its allowed root.")
    return candidate


def main() -> None:
    args = parse_args()
    plan = build_cleanup_plan(args.inventory, args.archive_root)
    payload = apply_cleanup_plan(
        plan,
        apply=bool(args.apply),
        confirmation=args.confirm,
    )
    if args.plan_log is not None:
        write_plan_log(args.plan_log, payload)
    planned = payload["planned"]
    applied = payload["applied"]
    print(f"cleanup mode:      {payload['mode']}")
    print(f"inventory:         {payload['inventory_path']}")
    print(f"archive root:      {payload['archive_root']}")
    print(
        "planned deletion:  "
        f"{planned['delete_files']} files  {format_bytes(planned['delete_bytes'])}"
    )
    print(
        "planned archive:   "
        f"{planned['archive_files']} files  {format_bytes(planned['archive_bytes'])}"
    )
    print(f"skipped missing:   {planned['skipped_missing_files']} files")
    if args.apply:
        print(
            "deleted:           "
            f"{applied['deleted_files']} files  {format_bytes(applied['deleted_bytes'])}"
        )
        print(
            "archived:          "
            f"{applied['archived_files']} files  {format_bytes(applied['archived_bytes'])}"
        )
    else:
        print("No files were changed. Add --apply and the explicit --confirm token after review.")


if __name__ == "__main__":
    main()
