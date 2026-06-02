"""Small, explicit parameter profiles for the focused GUI demonstration."""

from __future__ import annotations

from typing import Any

from configs import (
    DEFAULT_ANNEALING_COOLING_PARAMETER,
    DEFAULT_ANNEALING_ITERATIONS_PER_TEMPERATURE,
    DEFAULT_ANNEALING_MAX_STEPS,
    DEFAULT_ANNEALING_MIN_TEMPERATURE,
    DEFAULT_ANNEALING_START_TEMPERATURE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WEIGHT_SCALE,
    default_epochs,
)
from services.evaluation_profile_service import load_evaluation_benchmark_profile


SUPPORTED_GUI_PROFILES = ("demo", "tuned_20260530")


def load_gui_benchmark_profile(profile_name: str, benchmark: str) -> dict[str, Any]:
    """Return one GUI profile without changing the versioned evaluation profile."""

    if profile_name == "demo":
        return {
            "name": "demo",
            "training": {
                "learning_rate": DEFAULT_LEARNING_RATE,
                "epochs": default_epochs(benchmark),
                "batch_size": DEFAULT_BATCH_SIZE,
                "weight_scale": DEFAULT_WEIGHT_SCALE,
            },
            "online_delta": {
                "start_temperature": DEFAULT_ANNEALING_START_TEMPERATURE,
                "cooling_parameter": DEFAULT_ANNEALING_COOLING_PARAMETER,
                "iterations_per_temperature": DEFAULT_ANNEALING_ITERATIONS_PER_TEMPERATURE,
                "max_steps": DEFAULT_ANNEALING_MAX_STEPS,
                "min_temperature": DEFAULT_ANNEALING_MIN_TEMPERATURE,
                "online_learning_rate": DEFAULT_LEARNING_RATE,
                "online_batch_size": DEFAULT_BATCH_SIZE,
            },
        }
    if profile_name == "tuned_20260530":
        return load_evaluation_benchmark_profile(profile_name, benchmark)
    allowed = ", ".join(SUPPORTED_GUI_PROFILES)
    raise ValueError(f"Unbekanntes GUI-Profil '{profile_name}'. Erlaubt: {allowed}")

