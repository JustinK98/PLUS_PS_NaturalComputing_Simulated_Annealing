"""Kleine Hilfsfunktionen fuer die Qt-GUI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from benchmarks import DatasetBundle


def parse_hidden_sizes(text: str, fallback: tuple[int, ...]) -> tuple[int, ...]:
    """Parst eine kommaseparierte Hidden-Size-Zeile."""

    cleaned = text.replace("/", ",").replace(";", ",")
    values = [part.strip() for part in cleaned.split(",") if part.strip()]
    if not values:
        return fallback
    hidden_sizes = tuple(int(value) for value in values)
    if any(size <= 0 for size in hidden_sizes):
        raise ValueError("Alle Hidden-Sizes muessen positiv sein.")
    return hidden_sizes


def sample_arrays(bundle: DatasetBundle, split_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Liefert skaliertes X, rohes X und Targets fuer einen Split."""

    if split_name == "train":
        return bundle.X_train, bundle.X_train_raw, bundle.y_train
    if split_name == "val":
        return bundle.X_val, bundle.X_val_raw, bundle.y_val
    if split_name == "test":
        return bundle.X_test, bundle.X_test_raw, bundle.y_test
    raise ValueError(f"Unbekannter Split '{split_name}'.")


def html_escape(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def normalize_output_path(text: str, fallback: str) -> str:
    cleaned = text.strip()
    return cleaned if cleaned else fallback


def manifest_path_from_input(path_text: str) -> Path:
    path = Path(path_text.strip())
    return path
