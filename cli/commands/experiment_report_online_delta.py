"""CLI command for aggregated Online-Delta-SA reports."""

from __future__ import annotations

from services.online_delta_reporting_service import build_online_delta_report


def build_online_delta_report_command(args) -> None:
    """Builds Online-Delta aggregate artifacts from suite outputs."""

    result = build_online_delta_report(args.path, args.output_dir)

    print(f"input_path: {result.input_path}")
    print(f"output_dir: {result.output_dir}")
    print(f"runs:       {result.run_count}")
    print("artifacts:")
    for artifact in result.artifacts:
        print(f"  - {artifact}")
    if result.warnings:
        print("warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
