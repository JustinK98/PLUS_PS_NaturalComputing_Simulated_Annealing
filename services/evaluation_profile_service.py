"""Versioned evaluation profiles promoted from reviewed tuning artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from configs import default_hidden_sizes


CONFIG_PATH = Path("configs") / "evaluation_profiles.json"
DEFAULT_PROFILE_ALIAS = "default"


def default_evaluation_profile_name() -> str:
    """Return the promoted default profile name from the profile registry."""

    payload = _load_payload()
    profile_name = payload.get("default_profile")
    if not isinstance(profile_name, str) or not profile_name:
        raise ValueError("Kein default_profile in configs/evaluation_profiles.json gesetzt.")
    return profile_name


def load_evaluation_benchmark_profile(
    profile_name: str,
    benchmark: str,
) -> dict[str, Any]:
    """Load and validate one benchmark entry from a promoted profile."""

    payload = _load_payload()
    if profile_name == DEFAULT_PROFILE_ALIAS:
        profile_name = default_evaluation_profile_name()
    profiles = payload.get("profiles", {})
    if profile_name not in profiles:
        allowed = ", ".join(sorted(profiles))
        raise ValueError(f"Unbekanntes Evaluationsprofil '{profile_name}'. Erlaubt: {allowed}")
    profile = profiles[profile_name]
    status = str(profile.get("status", ""))
    if status.startswith("legacy_"):
        raise ValueError(
            f"Evaluationsprofil '{profile_name}' ist als '{status}' archiviert und darf "
            "nicht fuer neue Auswertungen verwendet werden."
        )
    benchmarks = profile.get("benchmarks", {})
    if benchmark not in benchmarks:
        raise ValueError(
            f"Evaluationsprofil '{profile_name}' enthaelt keinen Benchmark '{benchmark}'."
        )
    benchmark_profile = dict(benchmarks[benchmark])
    training = _require_dict(benchmark_profile, "training", profile_name, benchmark)
    online_delta = _require_dict(benchmark_profile, "online_delta", profile_name, benchmark)
    _require_keys(
        training,
        ("learning_rate", "epochs", "batch_size", "weight_scale"),
        profile_name,
        benchmark,
        "training",
    )
    _require_keys(
        online_delta,
        (
            "start_temperature",
            "cooling_parameter",
            "iterations_per_temperature",
            "max_steps",
            "min_temperature",
            "online_learning_rate",
            "online_batch_size",
        ),
        profile_name,
        benchmark,
        "online_delta",
    )
    return {
        "name": profile_name,
        "description": str(profile.get("description", "")),
        "source_artifact": str(profile.get("source_artifact", "")),
        "status": status,
        "benchmark": benchmark,
        "hidden_sizes": list(default_hidden_sizes(benchmark)),
        "training": training,
        "online_delta": online_delta,
        "confirmation": dict(benchmark_profile.get("confirmation", {})),
    }


def _load_payload() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _require_dict(
    payload: dict[str, Any],
    key: str,
    profile_name: str,
    benchmark: str,
) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(
            f"Evaluationsprofil '{profile_name}' fuer '{benchmark}' benoetigt '{key}'."
        )
    return dict(value)


def _require_keys(
    payload: dict[str, Any],
    keys: tuple[str, ...],
    profile_name: str,
    benchmark: str,
    section: str,
) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(
            f"Evaluationsprofil '{profile_name}' fuer '{benchmark}' fehlt in "
            f"'{section}': {', '.join(missing)}"
        )
