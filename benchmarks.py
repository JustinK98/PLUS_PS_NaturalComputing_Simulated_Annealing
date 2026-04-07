"""Datensatz-Logik fuer die drei kleinen Klassifikations-Benchmarks.

Didaktische Idee dieser Datei:

- Der Rest des Projekts soll nicht wissen muessen, wie scikit-learn Datensaetze
  intern liefert.
- Stattdessen bekommt der Rest des Codes immer ein einheitliches `DatasetBundle`.
- Alle Schritte rund um Laden, Standardisierung und Split liegen an einer Stelle.

Wenn spaeter neue Benchmarks dazukommen sollen, ist dies die richtige Datei.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import numpy as np
except ImportError as exc:
    raise ImportError(
        "Dieses Projekt benoetigt NumPy. Installation z. B. mit 'pip install numpy'."
    ) from exc

from configs import DatasetConfig, SUPPORTED_BENCHMARKS

try:
    from sklearn.datasets import load_breast_cancer, load_digits, load_wine
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


# Mapping von benutzerfreundlichen Namen auf scikit-learn-Loader.
# Wer spaeter weitere Datensaetze hinzufuegt, erweitert genau diese Tabelle.
RAW_DATASET_LOADERS = {
    "breast_cancer": load_breast_cancer,
    "wine": load_wine,
    "digits": load_digits,
}


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

    if config.name == "test_activation":
        return _load_test_activation_bundle(config)

    if config.name not in RAW_DATASET_LOADERS:
        supported = ", ".join(SUPPORTED_BENCHMARKS)
        raise ValueError(
            f"Unbekannter Benchmark '{config.name}'. Erlaubt sind: {supported}"
        )

    if not 0.0 < config.validation_size < 1.0:
        raise ValueError("validation_size muss zwischen 0 und 1 liegen.")
    if not 0.0 < config.test_size < 1.0:
        raise ValueError("test_size muss zwischen 0 und 1 liegen.")
    if config.validation_size + config.test_size >= 1.0:
        raise ValueError("validation_size + test_size muss kleiner als 1 sein.")

    # scikit-learn liefert typischerweise ein Bunch-Objekt mit `data` und `target`.
    raw_dataset = RAW_DATASET_LOADERS[config.name]()
    X = np.asarray(raw_dataset.data, dtype=np.float64)
    y = np.asarray(raw_dataset.target, dtype=np.int64)

    # Zuerst trennen wir einen finalen Test-Split ab.
    # Dieser Test-Split wird waehrend des Trainings nicht benutzt.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        stratify=y,
        random_state=config.random_state,
    )

    # Danach teilen wir den verbleibenden Rest weiter in Train und Validation.
    # Die Umrechnung ist noetig, weil validation_size als Anteil vom Gesamtdatensatz
    # gedacht ist, `train_test_split` hier aber auf den bereits verkleinerten Rest arbeitet.
    validation_share_inside_train_val = config.validation_size / (1.0 - config.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_share_inside_train_val,
        stratify=y_train_val,
        random_state=config.random_state,
    )

    # Standardisierung nur auf Basis der Trainingsdaten lernen.
    scaler = StandardScaler()
    X_train_raw = X_train.copy()
    X_val_raw = X_val.copy()
    X_test_raw = X_test.copy()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Manche Datensaetze haben echte Feature-Namen, andere nicht.
    # Dann vergeben wir einfache Namen wie x0, x1, x2, ...
    feature_names = getattr(raw_dataset, "feature_names", None)
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X.shape[1])]

    # Zielklassen-Namen, falls vom Datensatz bereitgestellt.
    target_names = getattr(raw_dataset, "target_names", None)
    if target_names is None:
        target_names = [str(label) for label in np.unique(y)]

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
        feature_names=tuple(str(name) for name in feature_names),
        target_names=tuple(str(name) for name in target_names),
        input_size=X.shape[1],
        output_size=len(np.unique(y)),
        scaler=scaler,
    )


def describe_dataset(bundle: DatasetBundle) -> str:
    """Erzeugt eine kurze deutschsprachige Ein-Zeilen-Beschreibung.

    Diese Funktion ist vor allem fuer spaetere Erweiterungen praktisch, zum
    Beispiel fuer Log-Dateien oder Tabellenansichten.
    """

    return (
        f"{bundle.name}: {bundle.input_size} Eingaben, {bundle.output_size} Klassen, "
        f"Train/Val/Test = {bundle.train_size}/{bundle.validation_size}/{bundle.test_size}"
    )


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
        scaler=scaler,
    )
