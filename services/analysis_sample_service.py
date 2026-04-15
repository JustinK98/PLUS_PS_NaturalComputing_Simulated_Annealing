"""UI-neutrale Analyse-Samples und benchmark-spezifische Sample-Aufbereitung."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from benchmarks import DatasetBundle


@dataclass(frozen=True)
class AnalysisSample:
    """Beschreibt genau das aktuell in der UI analysierte Beispiel."""

    raw_sample: np.ndarray
    scaled_sample: np.ndarray
    actual_target_index: int | None
    effective_target_index: int | None
    actual_target_name: str
    effective_target_name: str
    split_name: str
    sample_index: int
    source_label: str
    is_custom: bool


def split_arrays(bundle: DatasetBundle, split_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Liefert skaliertes X, rohes X und Targets fuer einen Split."""

    if split_name == "train":
        return bundle.X_train, bundle.X_train_raw, bundle.y_train
    if split_name == "val":
        return bundle.X_val, bundle.X_val_raw, bundle.y_val
    if split_name == "test":
        return bundle.X_test, bundle.X_test_raw, bundle.y_test
    raise ValueError(f"Unbekannter Split '{split_name}'.")


def infer_test_activation_target(raw_sample: np.ndarray) -> int:
    """Leitet fuer `test_activation` das didaktische Ziel aus dem Rohsample ab."""

    return 1 if float(np.sum(np.asarray(raw_sample, dtype=np.float64) >= 0.5)) >= 2 else 0


def build_analysis_sample(
    bundle: DatasetBundle,
    split_name: str,
    sample_index: int,
    *,
    language: str = "en",
    use_custom_sample: bool = False,
    custom_raw_sample: np.ndarray | None = None,
    analysis_target_index: int | None = None,
) -> AnalysisSample:
    """Erzeugt ein UI-neutrales Analyse-Sample fuer Demo, Playground und Preview."""

    actual_target_index: int | None
    source_label: str

    if use_custom_sample and custom_raw_sample is not None and bundle.name == "digits":
        raw_sample = np.asarray(custom_raw_sample, dtype=np.float64).copy()
        actual_target_index = None
        source_label = "Eigenes digits-Sample" if language == "de" else "Custom digits sample"
        bounded_index = int(sample_index)
    elif use_custom_sample and custom_raw_sample is not None and bundle.name == "test_activation":
        raw_sample = np.asarray(custom_raw_sample, dtype=np.float64).copy()
        actual_target_index = infer_test_activation_target(raw_sample)
        source_label = (
            "Eigenes test_activation-Sample"
            if language == "de"
            else "Custom test_activation sample"
        )
        bounded_index = int(sample_index)
    else:
        X_scaled, X_raw, y_split = split_arrays(bundle, split_name)
        if len(X_scaled) == 0:
            raise ValueError("Der gewaehlte Split enthaelt keine Samples.")
        bounded_index = min(max(int(sample_index), 0), len(X_scaled) - 1)
        raw_sample = np.asarray(X_raw[bounded_index], dtype=np.float64).copy()
        actual_target_index = int(y_split[bounded_index])
        source_label = (
            f"Datensatz-Sample aus {split_name}"
            if language == "de"
            else f"Dataset sample from {split_name}"
        )

    if bundle.scaler is not None:
        scaled_sample = bundle.scaler.transform(raw_sample.reshape(1, -1))[0]
    else:
        scaled_sample = raw_sample.copy()

    effective_target_index = actual_target_index if analysis_target_index is None else analysis_target_index
    actual_target_name = (
        bundle.target_names[actual_target_index]
        if actual_target_index is not None
        else ("unbekannt" if language == "de" else "unknown")
    )
    effective_target_name = (
        bundle.target_names[effective_target_index]
        if effective_target_index is not None
        else ("kein Ziel gesetzt" if language == "de" else "no target set")
    )

    return AnalysisSample(
        raw_sample=raw_sample,
        scaled_sample=scaled_sample,
        actual_target_index=actual_target_index,
        effective_target_index=effective_target_index,
        actual_target_name=actual_target_name,
        effective_target_name=effective_target_name,
        split_name=split_name,
        sample_index=bounded_index,
        source_label=source_label,
        is_custom=bool(use_custom_sample and custom_raw_sample is not None),
    )


def build_feature_rows(
    bundle: DatasetBundle,
    analysis_sample: AnalysisSample,
    *,
    limit: int = 16,
) -> list[tuple[int, str, float, float]]:
    """Erzeugt die sortierte Top-Feature-Sicht fuer tabellarische Benchmarks."""

    rows: list[tuple[float, str, float, float]] = []
    for feature_index, feature_name in enumerate(bundle.feature_names):
        raw_value = float(analysis_sample.raw_sample[feature_index])
        scaled_value = float(analysis_sample.scaled_sample[feature_index])
        rows.append((abs(scaled_value), str(feature_name), raw_value, scaled_value))
    rows.sort(key=lambda row: row[0], reverse=True)
    return [
        (rank, feature_name, raw_value, scaled_value)
        for rank, (_magnitude, feature_name, raw_value, scaled_value) in enumerate(rows[:limit], start=1)
    ]


def build_test_activation_rows(
    bundle: DatasetBundle,
    analysis_sample: AnalysisSample,
) -> list[tuple[str, float, float]]:
    """Erzeugt die Roh-/Skalierungs-Tabelle fuer `test_activation`."""

    return [
        (str(feature_name), float(raw_value), float(scaled_value))
        for feature_name, raw_value, scaled_value in zip(
            bundle.feature_names,
            analysis_sample.raw_sample,
            analysis_sample.scaled_sample,
            strict=True,
        )
    ]


def build_digit_matrix(raw_sample: np.ndarray) -> tuple[tuple[float, ...], ...]:
    """Formt ein flaches digits-Sample in eine 8x8-Matrix um."""

    array = np.asarray(raw_sample, dtype=np.float64).reshape(8, 8)
    return tuple(tuple(float(value) for value in row) for row in array)


def format_probability_lines(
    bundle: DatasetBundle,
    probabilities: np.ndarray,
) -> str:
    """Formatiert Klassenwahrscheinlichkeiten kompakt fuer UI-Panels."""

    return ", ".join(
        f"{bundle.target_names[index]}={float(probability):.3f}"
        for index, probability in enumerate(probabilities)
    )

