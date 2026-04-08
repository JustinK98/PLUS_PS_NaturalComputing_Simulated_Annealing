"""JSON-Speicherung und Laden fuer Experiment-Builder-Ergebnisse."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiment_builder import ExperimentDefinition, ExperimentSummary, RunResult


def save_experiment_results(
    definition: ExperimentDefinition,
    summary: ExperimentSummary,
    run_results: list[RunResult],
) -> Path:
    """Speichert Manifest, Summary und alle Runs unterhalb des Experimentordners."""

    experiment_dir = definition.output_path
    runs_dir = experiment_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment_id": definition.experiment_id,
        "benchmark": definition.benchmark,
        "run_mode": definition.run_mode,
        "search_type": definition.search_space.search_type,
        "primary_metric": definition.primary_metric,
        "created_at": definition.created_at,
        "definition": definition.to_dict(),
        "summary_file": "summary.json",
        "run_files": [f"runs/{run_result.run_definition.run_id}.json" for run_result in run_results],
    }
    _write_json(experiment_dir / "manifest.json", manifest)
    _write_json(experiment_dir / "summary.json", summary.to_dict())
    for run_result in run_results:
        _write_json(runs_dir / f"{run_result.run_definition.run_id}.json", run_result.to_dict())
    return experiment_dir


def load_experiment_results(manifest_or_directory: str | Path) -> dict[str, Any]:
    """Laedt ein gespeichertes Experiment inklusive Summary und Run-Dateien."""

    path = Path(manifest_or_directory)
    manifest_path = path / "manifest.json" if path.is_dir() else path
    manifest = _read_json(manifest_path)
    validate_experiment_payload(manifest)

    experiment_dir = manifest_path.parent
    summary = _read_json(experiment_dir / manifest["summary_file"])
    run_payloads = [
        _read_json(experiment_dir / relative_path)
        for relative_path in manifest["run_files"]
    ]
    for run_payload in run_payloads:
        validate_run_payload(run_payload)
    return {
        "manifest": manifest,
        "summary": summary,
        "runs": run_payloads,
        "experiment_dir": str(experiment_dir),
    }


def validate_experiment_payload(payload: dict[str, Any]) -> None:
    """Prueft die Kernstruktur eines gespeicherten Manifestes."""

    required_keys = {
        "experiment_id",
        "benchmark",
        "run_mode",
        "search_type",
        "primary_metric",
        "created_at",
        "definition",
        "summary_file",
        "run_files",
    }
    missing_keys = required_keys - set(payload)
    if missing_keys:
        raise ValueError(
            "Manifest fehlt mindestens ein Pflichtfeld: " + ", ".join(sorted(missing_keys))
        )


def validate_run_payload(payload: dict[str, Any]) -> None:
    """Prueft die Kernstruktur eines gespeicherten Run-JSONs."""

    required_keys = {
        "run_definition",
        "metrics",
        "history",
        "layout_spec",
        "created_at",
        "extra",
    }
    missing_keys = required_keys - set(payload)
    if missing_keys:
        raise ValueError(
            "Run-JSON fehlt mindestens ein Pflichtfeld: " + ", ".join(sorted(missing_keys))
        )

    run_definition_keys = {
        "run_id",
        "experiment_id",
        "config_id",
        "config_index",
        "seed",
        "benchmark",
        "hidden_sizes",
        "layout_spec",
        "run_mode",
        "primary_metric",
        "config_values",
    }
    missing_run_definition_keys = run_definition_keys - set(payload["run_definition"])
    if missing_run_definition_keys:
        raise ValueError(
            "run_definition fehlt mindestens ein Pflichtfeld: "
            + ", ".join(sorted(missing_run_definition_keys))
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Schreibt JSON lesbar auf Platte."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    """Liest JSON von Platte."""

    return json.loads(path.read_text(encoding="utf-8"))
