"""CLI command for building report-ready result assets."""

from __future__ import annotations

from pathlib import Path

from configs import OUTPUT_DIR
from services.reporting_service import build_report_assets


def build_report_command(args) -> None:
    """Build consolidated CSV tables and figures from existing outputs."""

    result = build_report_assets(
        output_dir=Path(args.output_dir),
        layout_grid_dir=Path(args.layout_grid_dir),
        demo_sa_dir=Path(args.demo_sa_dir),
        neighborhood_summary_path=Path(args.neighborhood_summary),
    )
    print(f"report_assets: {result.output_dir}")
    print("files:")
    for path in result.files:
        print(f"  - {path}")
    if result.warnings:
        print("warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")


def default_report_output_dir() -> Path:
    return OUTPUT_DIR / "report_assets"
