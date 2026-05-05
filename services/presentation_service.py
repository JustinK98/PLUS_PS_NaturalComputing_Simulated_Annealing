"""UI-neutrale Default-Zustaende fuer den gefuehrten Presentation Mode."""

from __future__ import annotations

from dataclasses import dataclass

from activations import parse_layout_spec
from annealing import AnnealingConfig
from annealing_objectives import ObjectiveConfig
from benchmarks import DatasetBundle, load_benchmark
from configs import DatasetConfig, TrainingConfig, default_hidden_sizes
from model import ModularMLP
from services.analysis_sample_service import AnalysisSample, build_analysis_sample
from services.annealing_service import AnnealingRunRequest
from services.annealing_session_service import AnnealingSession, create_session
from services.training_session_service import TrainingSessionSnapshot, create_training_session


PRESENTATION_BENCHMARK = "concentric_circles"
PRESENTATION_HIDDEN_SIZES = default_hidden_sizes(PRESENTATION_BENCHMARK)
PRESENTATION_LAYOUT = "relu"
PRESENTATION_SAMPLE_SPLIT = "val"
PRESENTATION_SAMPLE_INDEX = 0
PRESENTATION_RANDOM_STATE = 42
PRESENTATION_WEIGHT_SCALE = 0.05


@dataclass
class PresentationRuntimeState:
    """Gemeinsamer Zustand fuer beide Praesentationstracks."""

    dataset: DatasetBundle
    analysis_sample: AnalysisSample
    model: ModularMLP
    training_session: TrainingSessionSnapshot
    annealing_session: AnnealingSession | None = None
    selected_hidden: tuple[int, int] = (0, 0)


def create_presentation_state(language: str = "de") -> PresentationRuntimeState:
    """Erzeugt einen reproduzierbaren offiziellen Benchmark-Zustand fuer Live-Demos."""

    dataset = load_benchmark(
        DatasetConfig(name=PRESENTATION_BENCHMARK, random_state=PRESENTATION_RANDOM_STATE)
    )
    layout = parse_layout_spec(PRESENTATION_LAYOUT, PRESENTATION_HIDDEN_SIZES)
    model = ModularMLP(
        input_size=dataset.input_size,
        hidden_sizes=PRESENTATION_HIDDEN_SIZES,
        output_size=dataset.model_output_size,
        layout=layout,
        num_classes=dataset.output_size,
        weight_scale=PRESENTATION_WEIGHT_SCALE,
        random_state=PRESENTATION_RANDOM_STATE,
    )
    analysis_sample = build_analysis_sample(
        dataset,
        PRESENTATION_SAMPLE_SPLIT,
        PRESENTATION_SAMPLE_INDEX,
        language=language,
    )
    return PresentationRuntimeState(
        dataset=dataset,
        analysis_sample=analysis_sample,
        model=model,
        training_session=create_training_session(model),
    )


def presentation_training_config(epochs: int) -> TrainingConfig:
    """Kleine Trainingskonfiguration fuer sichtbare Vortragsaktionen."""

    return TrainingConfig(
        epochs=epochs,
        learning_rate=0.03,
        batch_size=32,
        random_state=PRESENTATION_RANDOM_STATE,
        shuffle=True,
    )


def presentation_annealing_request() -> AnnealingRunRequest:
    """Kompakter SA-Request fuer einen reproduzierbaren Preview-Lauf."""

    return AnnealingRunRequest(
        dataset_config=DatasetConfig(
            name=PRESENTATION_BENCHMARK,
            random_state=PRESENTATION_RANDOM_STATE,
        ),
        hidden_sizes=PRESENTATION_HIDDEN_SIZES,
        layout_spec=PRESENTATION_LAYOUT,
        objective_config=ObjectiveConfig(
            objective_name="validation_loss",
            candidate_epochs=4,
            learning_rate=0.03,
            batch_size=32,
            weight_scale=PRESENTATION_WEIGHT_SCALE,
            random_state=PRESENTATION_RANDOM_STATE,
            shuffle=True,
        ),
        annealing_config=AnnealingConfig(
            start_temperature=1.5,
            cooling_schedule="geometric",
            cooling_parameter=0.92,
            iterations_per_temperature=2,
            max_steps=8,
            min_temperature=0.02,
            neighborhood_operations=("set_neuron", "fill_layer", "swap_neurons"),
        ),
        random_state=PRESENTATION_RANDOM_STATE,
    )


def create_presentation_annealing_session() -> AnnealingSession:
    """Erzeugt eine frische SA-Session fuer den Presentation Mode."""

    return create_session(presentation_annealing_request())
