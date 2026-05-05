"""Registry fuer die offiziellen CSV-Benchmarks des Basics-Teams."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BENCHMARK_DATA_DIR = Path("data") / "benchmarks" / "basics_group"


@dataclass(frozen=True)
class BenchmarkSpec:
    """Beschreibt einen reproduzierbaren CSV-Benchmark."""

    name: str
    train_csv: Path
    test_csv: Path
    feature_columns: tuple[str, ...]
    label_column: str
    target_names: tuple[str, ...]
    hidden_sizes: tuple[int, ...]
    epochs: int
    model_output_size: int
    task_type: str

    @property
    def class_count(self) -> int:
        return len(self.target_names)


OFFICIAL_BENCHMARK_SPECS: dict[str, BenchmarkSpec] = {
    "concentric_circles": BenchmarkSpec(
        name="concentric_circles",
        train_csv=BENCHMARK_DATA_DIR / "concentric_circles_train.csv",
        test_csv=BENCHMARK_DATA_DIR / "concentric_circles_test.csv",
        feature_columns=("x", "y"),
        label_column="label",
        target_names=("outer_circle", "inner_circle"),
        hidden_sizes=(8,),
        epochs=100,
        model_output_size=1,
        task_type="binary",
    ),
    "iris": BenchmarkSpec(
        name="iris",
        train_csv=BENCHMARK_DATA_DIR / "iris_train.csv",
        test_csv=BENCHMARK_DATA_DIR / "iris_test.csv",
        feature_columns=(
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ),
        label_column="label",
        target_names=("setosa", "versicolor", "virginica"),
        hidden_sizes=(8,),
        epochs=150,
        model_output_size=3,
        task_type="multiclass",
    ),
    "crossing_spirals": BenchmarkSpec(
        name="crossing_spirals",
        train_csv=BENCHMARK_DATA_DIR / "crossing_spirals_train.csv",
        test_csv=BENCHMARK_DATA_DIR / "crossing_spirals_test.csv",
        feature_columns=("x", "y", "x_sin", "y_sin", "x_cos", "y_cos"),
        label_column="label",
        target_names=("red_spiral", "blue_spiral"),
        hidden_sizes=(16, 16),
        epochs=250,
        model_output_size=1,
        task_type="binary",
    ),
}


OFFICIAL_BENCHMARKS = tuple(OFFICIAL_BENCHMARK_SPECS)


def benchmark_spec(name: str) -> BenchmarkSpec:
    """Liefert die Registry-Definition fuer einen offiziellen Benchmark."""

    try:
        return OFFICIAL_BENCHMARK_SPECS[name]
    except KeyError as exc:
        supported = ", ".join(OFFICIAL_BENCHMARKS)
        raise ValueError(f"Unbekannter Benchmark '{name}'. Erlaubt sind: {supported}") from exc
