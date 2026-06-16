"""Small, explicit parameter profiles for the focused GUI demonstration."""

from __future__ import annotations

from typing import Any

from configs import (
    DEFAULT_ANNEALING_COOLING_PARAMETER,
    DEFAULT_ANNEALING_COOLING_SCHEDULE,
    DEFAULT_ANNEALING_ITERATIONS_PER_TEMPERATURE,
    DEFAULT_ANNEALING_MAX_STEPS,
    DEFAULT_ANNEALING_MIN_TEMPERATURE,
    DEFAULT_ANNEALING_START_TEMPERATURE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_GUI_PROFILE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WEIGHT_SCALE,
    default_epochs,
)
from services.evaluation_profile_service import load_evaluation_benchmark_profile


PROMOTED_EVALUATION_PROFILE = "mayer_corrected_20260604"
METHODICAL_SELECTED_PROFILE = DEFAULT_GUI_PROFILE
SUPPORTED_GUI_PROFILES = (DEFAULT_GUI_PROFILE, "demo", PROMOTED_EVALUATION_PROFILE)

_METHODICAL_SELECTED_BY_BENCHMARK: dict[str, dict[str, Any]] = {
    "two_moons": {
        "training": {
            "learning_rate": 0.2,
            "epochs": 300,
            "batch_size": 32,
            "weight_scale": 2.0,
        },
        "online_delta": {
            "start_temperature": 0.012837135917035067,
            "cooling_schedule": "linear",
            "cooling_parameter": 9.764034524566303e-06,
            "iterations_per_temperature": 10,
            "max_steps": 12500,
            "min_temperature": 0.00012837135917035067,
            "online_learning_rate": 0.2,
            "online_batch_size": 128,
        },
    },
    "concentric_circles": {
        "training": {
            "learning_rate": 0.1,
            "epochs": 150,
            "batch_size": 8,
            "weight_scale": 1.0,
        },
        "online_delta": {
            "start_temperature": 0.015640441213760042,
            "cooling_schedule": "logarithmic",
            "cooling_parameter": 0.5790593092043358,
            "iterations_per_temperature": 25,
            "max_steps": 25000,
            "min_temperature": 0.00015640441213760043,
            "online_learning_rate": 0.025,
            "online_batch_size": 64,
        },
    },
    "crossing_spirals": {
        "training": {
            "learning_rate": 0.15,
            "epochs": 750,
            "batch_size": 8,
            "weight_scale": 1.0,
        },
        "online_delta": {
            "start_temperature": 0.004458188656420654,
            "cooling_schedule": "linear",
            "cooling_parameter": 3.602576692057094e-05,
            "iterations_per_temperature": 25,
            "max_steps": 2500,
            "min_temperature": 4.458188656420654e-05,
            "online_learning_rate": 0.15,
            "online_batch_size": 64,
        },
    },
}


def load_gui_benchmark_profile(profile_name: str, benchmark: str) -> dict[str, Any]:
    """Return one GUI profile without changing the versioned evaluation profile."""

    if profile_name == "demo":
        return {
            "name": "demo",
            "description": "Schnelles didaktisches Profil fuer kurze GUI-Laeufe.",
            "training": {
                "learning_rate": DEFAULT_LEARNING_RATE,
                "epochs": default_epochs(benchmark),
                "batch_size": DEFAULT_BATCH_SIZE,
                "weight_scale": DEFAULT_WEIGHT_SCALE,
            },
            "online_delta": {
                "start_temperature": DEFAULT_ANNEALING_START_TEMPERATURE,
                "cooling_schedule": DEFAULT_ANNEALING_COOLING_SCHEDULE,
                "cooling_parameter": DEFAULT_ANNEALING_COOLING_PARAMETER,
                "iterations_per_temperature": DEFAULT_ANNEALING_ITERATIONS_PER_TEMPERATURE,
                "max_steps": DEFAULT_ANNEALING_MAX_STEPS,
                "min_temperature": DEFAULT_ANNEALING_MIN_TEMPERATURE,
                "online_learning_rate": DEFAULT_LEARNING_RATE,
                "online_batch_size": DEFAULT_BATCH_SIZE,
            },
        }
    if profile_name == METHODICAL_SELECTED_PROFILE:
        try:
            selected = _METHODICAL_SELECTED_BY_BENCHMARK[benchmark]
        except KeyError as exc:
            allowed = ", ".join(sorted(_METHODICAL_SELECTED_BY_BENCHMARK))
            raise ValueError(f"Unbekannter Benchmark '{benchmark}'. Erlaubt: {allowed}") from exc
        return {
            "name": profile_name,
            "description": (
                "Im methodischen Tuning selektierte Demonstrationsparameter; "
                "kein robuster allgemeiner SA-Vorteil bestaetigt."
            ),
            "training": dict(selected["training"]),
            "online_delta": dict(selected["online_delta"]),
        }
    if profile_name == PROMOTED_EVALUATION_PROFILE:
        profile = load_evaluation_benchmark_profile(profile_name, benchmark)
        return {
            "name": profile_name,
            "description": profile["description"],
            "training": profile["training"],
            "online_delta": profile["online_delta"],
        }
    allowed = ", ".join(SUPPORTED_GUI_PROFILES)
    raise ValueError(f"Unbekanntes GUI-Profil '{profile_name}'. Erlaubt: {allowed}")
