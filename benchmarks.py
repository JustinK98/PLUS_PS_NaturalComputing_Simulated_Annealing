"""Datensatz-Logik fuer die CSV-Benchmark-Suite des Playground-Projekts.

Didaktische Idee dieser Datei:

- Der Rest des Projekts soll nicht wissen muessen, wie die CSV-Dateien
  des Basics-Teams aufgebaut sind.
- Stattdessen bekommt der Rest des Codes immer ein einheitliches `DatasetBundle`.
- Alle Schritte rund um Laden, Standardisierung und Split liegen an einer Stelle.

Wenn spaeter neue Benchmarks dazukommen sollen, ist dies die richtige Datei.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Any

try:
    import numpy as np
except ImportError as exc:
    raise ImportError(
        "Dieses Projekt benoetigt NumPy. Installation z. B. mit 'pip install numpy'."
    ) from exc

from benchmark_registry import OFFICIAL_BENCHMARKS, benchmark_spec
from configs import ALL_BENCHMARKS, DatasetConfig

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise ImportError(
        "Dieses Projekt benoetigt scikit-learn. Installation z. B. mit "
        "'pip install scikit-learn'."
    ) from exc


@dataclass
class DatasetBundle:
    """Gebuendelte Datensatzinformationen fuer Training, Validierung und Test.

    Das Bundle ist die zentrale Datensatz-Struktur des Projekts.
    Alle Arrays sind bereits so vorbereitet, dass das Modell sofort trainiert
    werden kann.
    """

    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_train_raw: np.ndarray
    X_val_raw: np.ndarray
    X_test_raw: np.ndarray
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    input_size: int
    output_size: int
    model_output_size: int
    scaler: Any | None = None

    @property
    def train_size(self) -> int:
        """Anzahl der Trainingsbeispiele."""

        return len(self.y_train)

    @property
    def validation_size(self) -> int:
        """Anzahl der Validierungsbeispiele."""

        return len(self.y_val)

    @property
    def test_size(self) -> int:
        """Anzahl der Testbeispiele."""

        return len(self.y_test)


def load_benchmark(config: DatasetConfig) -> DatasetBundle:
    """Laedt einen Benchmark, skaliert ihn und erzeugt Train/Val/Test-Splits.

    Warum Standardisierung?
    Viele Modelle lernen deutlich stabiler, wenn Eingaben grob auf vergleichbaren
    Skalen liegen. Deshalb wird hier `StandardScaler` eingesetzt:

    - Trainingsdaten: fit + transform
    - Val/Test: nur transform mit derselben Skalierung

    Das ist wichtig, damit keine Information aus Val/Test "zurueck" in das
    Training auslaeuft.
    """

    if config.name in OFFICIAL_BENCHMARKS:
        return _load_csv_benchmark_bundle(config)

    if config.name == "test_activation":
        return _load_test_activation_bundle(config)

    supported = ", ".join(ALL_BENCHMARKS)
    raise ValueError(f"Unbekannter Benchmark '{config.name}'. Erlaubt sind: {supported}")


def describe_dataset(bundle: DatasetBundle) -> str:
    """Erzeugt eine kurze deutschsprachige Ein-Zeilen-Beschreibung.

    Diese Funktion ist vor allem fuer spaetere Erweiterungen praktisch, zum
    Beispiel fuer Log-Dateien oder Tabellenansichten.
    """

    return (
        f"{bundle.name}: {bundle.input_size} Eingaben, {bundle.output_size} Klassen, "
        f"{bundle.model_output_size} Output-Neuronen, "
        f"Train/Val/Test = {bundle.train_size}/{bundle.validation_size}/{bundle.test_size}"
    )


def _load_csv_benchmark_bundle(config: DatasetConfig) -> DatasetBundle:
    """Laedt einen offiziellen Benchmark aus den versionierten CSV-Dateien."""

    if not 0.0 < config.validation_size < 1.0:
        raise ValueError("validation_size muss zwischen 0 und 1 liegen.")

    spec = benchmark_spec(config.name)
    X_train_full, y_train_full = _read_csv_dataset(
        spec.train_csv,
        feature_columns=spec.feature_columns,
        label_column=spec.label_column,
    )
    X_test_raw, y_test = _read_csv_dataset(
        spec.test_csv,
        feature_columns=spec.feature_columns,
        label_column=spec.label_column,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=config.validation_size,
        stratify=y_train_full,
        random_state=config.random_state,
    )

    scaler = StandardScaler()
    X_train_raw = X_train.copy()
    X_val_raw = X_val.copy()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test_raw)

    return DatasetBundle(
        name=config.name,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        X_train_raw=X_train_raw,
        X_val_raw=X_val_raw,
        X_test_raw=X_test_raw,
        feature_names=spec.feature_columns,
        target_names=spec.target_names,
        input_size=len(spec.feature_columns),
        output_size=spec.class_count,
        model_output_size=spec.model_output_size,
        scaler=scaler,
    )


def _read_csv_dataset(
    path,
    *,
    feature_columns: tuple[str, ...],
    label_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Liest Feature- und Label-Arrays aus einem Benchmark-CSV."""

    rows: list[list[float]] = []
    labels: list[int] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_columns = [
            column
            for column in (*feature_columns, label_column)
            if column not in (reader.fieldnames or [])
        ]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"CSV '{path}' fehlt Spalten: {missing}")
        for row in reader:
            rows.append([float(row[column]) for column in feature_columns])
            labels.append(int(row[label_column]))
    if not rows:
        raise ValueError(f"CSV '{path}' enthaelt keine Datenzeilen.")
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def _load_test_activation_bundle(config: DatasetConfig) -> DatasetBundle:
    """Erzeugt einen kleinen kuenstlichen Datensatz fuer rohe Aktivierungs-Experimente.

    Dieser Modus ist nicht als Benchmark gedacht, sondern als einfaches
    Lernlabor mit wenigen Inputs und gut lesbaren Werten.
    """

    X_raw = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    # Klasse 1, wenn mindestens zwei Eingaben aktiv sind.
    y = np.asarray([0, 0, 0, 1, 0, 1, 1, 1], dtype=np.int64)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_raw,
        y,
        test_size=config.test_size,
        stratify=y,
        random_state=config.random_state,
    )
    validation_share_inside_train_val = config.validation_size / (1.0 - config.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_share_inside_train_val,
        stratify=y_train_val,
        random_state=config.random_state,
    )

    scaler = StandardScaler()
    X_train_raw = X_train.copy()
    X_val_raw = X_val.copy()
    X_test_raw = X_test.copy()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return DatasetBundle(
        name=config.name,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        X_train_raw=X_train_raw,
        X_val_raw=X_val_raw,
        X_test_raw=X_test_raw,
        feature_names=("input_a", "input_b", "input_c"),
        target_names=("Klasse 0", "Klasse 1"),
        input_size=3,
        output_size=2,
        model_output_size=2,
        scaler=scaler,
    )
