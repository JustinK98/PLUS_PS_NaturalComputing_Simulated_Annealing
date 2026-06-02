"""Online-Delta Simulated Annealing mit interleaved Mini-Batch-Training."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from activations import (
    ActivationLayout,
    generate_neighbors,
    parse_layout_spec,
    sample_neighbor,
    sample_set_neuron_neighbor,
)
from annealing import AnnealingConfig, acceptance_probability
from annealing_schedules import temperature_for_step
from benchmarks import DatasetBundle, load_benchmark
from configs import (
    DatasetConfig,
    TrainingConfig,
)
from model import ModularMLP
from trainer import evaluate_batch, train_one_batch


@dataclass(frozen=True)
class OnlineAnnealingRequest:
    """Konfiguration fuer den professor-nahen Online-Delta-SA-Modus."""

    dataset_config: DatasetConfig
    hidden_sizes: tuple[int, ...]
    layout_spec: str
    training_config: TrainingConfig
    annealing_config: AnnealingConfig
    weight_scale: float
    random_state: int
    include_test_metrics: bool = True

    def __post_init__(self) -> None:
        if self.weight_scale <= 0.0:
            raise ValueError("weight_scale muss positiv sein.")


@dataclass
class OnlineLayoutEvaluation:
    """Layout-Bewertung fuer den Online-Modus.

    `objective_value` und `comparable_score` sind bewusst der aktuelle Batch-Loss.
    Die Validierungswerte dienen separat zur Auswahl des besten beobachteten Layouts.
    """

    layout: ActivationLayout
    comparable_score: float
    objective_value: float
    train_loss: float
    val_loss: float
    test_loss: float
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    trained_model: ModularMLP
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class OnlineAnnealingStep:
    """Ein Online-Delta-SA-Schritt mit Batch- und Trainingsdiagnostik."""

    step_index: int
    temperature: float
    previous_layout: ActivationLayout
    candidate_layout: ActivationLayout
    previous_evaluation: OnlineLayoutEvaluation
    candidate_evaluation: OnlineLayoutEvaluation
    delta: float
    acceptance_probability: float
    random_draw: float
    accepted: bool
    reason_code: str
    neighbor_label: str
    best_score_after_step: float
    accepted_steps_after_step: int
    rejected_steps_after_step: int
    batch_loss_before: float
    candidate_loss_after: float
    trained_after_accept: bool
    epoch_index: int
    batch_index: int
    batch_start: int
    validation_loss_after_update: float


@dataclass
class OnlineAnnealingState:
    """Laufender Zustand eines Online-Delta-SA-Prozesses."""

    start_evaluation: OnlineLayoutEvaluation
    current_evaluation: OnlineLayoutEvaluation
    best_evaluation: OnlineLayoutEvaluation
    current_temperature: float
    step_index: int = 0
    accepted_steps: int = 0
    rejected_steps: int = 0
    accepted_worse_steps: int = 0
    history: list[OnlineAnnealingStep] = field(default_factory=list)
    latest_candidate: OnlineLayoutEvaluation | None = None
    latest_neighbor_label: str = ""

    @property
    def acceptance_rate(self) -> float:
        if self.step_index == 0:
            return 0.0
        return self.accepted_steps / self.step_index


@dataclass
class MiniBatchCursor:
    """Deterministische Mini-Batch-Folge fuer Online-SA."""

    train_size: int
    batch_size: int
    shuffle: bool
    rng: np.random.Generator
    epoch_index: int = 0
    batch_index: int = 0
    batch_start: int = 0
    order: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.train_size <= 0:
            raise ValueError("train_size muss positiv sein.")
        if self.batch_size <= 0:
            raise ValueError("batch_size muss positiv sein.")
        self.order = np.arange(self.train_size)
        if self.shuffle:
            self.order = self.rng.permutation(self.order)

    def current_indices(self) -> np.ndarray:
        if self.batch_start >= self.train_size:
            self._start_next_epoch()
        end = min(self.batch_start + self.batch_size, self.train_size)
        return self.order[self.batch_start:end]

    def advance(self) -> None:
        self.batch_start += self.batch_size
        self.batch_index += 1
        if self.batch_start >= self.train_size:
            self._start_next_epoch()

    def _start_next_epoch(self) -> None:
        self.epoch_index += 1
        self.batch_index = 0
        self.batch_start = 0
        self.order = np.arange(self.train_size)
        if self.shuffle:
            self.order = self.rng.permutation(self.order)


@dataclass
class OnlineAnnealingSession:
    """Laufende Online-Delta-SA-Session."""

    request: OnlineAnnealingRequest
    dataset: DatasetBundle
    model: ModularMLP
    start_layout: ActivationLayout
    rng: np.random.Generator
    batch_cursor: MiniBatchCursor
    state: OnlineAnnealingState | None = None
    end_evaluation: OnlineLayoutEvaluation | None = None


@dataclass(frozen=True)
class OnlineAnnealingSnapshot:
    """DTO fuer Qt-Panels, Tests und Experiment-Serialisierung."""

    is_initialized: bool
    is_complete: bool
    state_label: str
    decision_label: str
    reason_label: str
    start_evaluation: OnlineLayoutEvaluation | None
    candidate_evaluation: OnlineLayoutEvaluation | None
    current_evaluation: OnlineLayoutEvaluation | None
    best_evaluation: OnlineLayoutEvaluation | None
    end_evaluation: OnlineLayoutEvaluation | None
    last_step: OnlineAnnealingStep | None
    history: tuple[OnlineAnnealingStep, ...]
    accepted_steps: int
    acceptance_rate: float
    current_temperature: float
    stop_reasons: tuple[str, ...]
    epoch_index: int
    batch_index: int
    batch_start: int
    sa_evaluation_mode: str = "online_delta"


def create_online_session(request: OnlineAnnealingRequest) -> OnlineAnnealingSession:
    """Erzeugt eine neue Online-Delta-SA-Session."""

    dataset = load_benchmark(request.dataset_config)
    start_layout = parse_layout_spec(request.layout_spec, request.hidden_sizes)
    model = ModularMLP(
        input_size=dataset.input_size,
        hidden_sizes=request.hidden_sizes,
        output_size=dataset.model_output_size,
        layout=start_layout,
        num_classes=dataset.output_size,
        weight_scale=request.weight_scale,
        random_state=request.random_state,
    )
    rng = np.random.default_rng(request.random_state)
    batch_cursor = MiniBatchCursor(
        train_size=dataset.train_size,
        batch_size=request.training_config.batch_size,
        shuffle=request.training_config.shuffle,
        rng=np.random.default_rng(request.training_config.random_state),
    )
    return OnlineAnnealingSession(
        request=request,
        dataset=dataset,
        model=model,
        start_layout=start_layout,
        rng=rng,
        batch_cursor=batch_cursor,
    )


def evaluate_online_start(
    session: OnlineAnnealingSession,
    language: str = "de",
) -> OnlineAnnealingSnapshot:
    """Initialisiert die Online-Session ohne Training."""

    if session.state is None:
        X_batch, y_batch = _current_batch(session)
        batch_loss, batch_acc = evaluate_batch(session.model, X_batch, y_batch)
        start_evaluation = _build_evaluation(
            session,
            session.model.layout,
            batch_loss=batch_loss,
            batch_accuracy=batch_acc,
            phase="start",
        )
        session.state = OnlineAnnealingState(
            start_evaluation=start_evaluation,
            current_evaluation=start_evaluation,
            best_evaluation=start_evaluation,
            current_temperature=session.request.annealing_config.start_temperature,
        )
        session.end_evaluation = None
    return online_current_snapshot(session, language)


def online_step_once(
    session: OnlineAnnealingSession,
    language: str = "de",
) -> OnlineAnnealingSnapshot:
    """Fuehrt genau einen Online-Delta-SA-Schritt aus."""

    if session.state is None:
        evaluate_online_start(session, language)
    if not online_should_stop(session):
        _online_step(session)
    if online_should_stop(session) and session.state is not None:
        session.end_evaluation = session.state.current_evaluation
    return online_current_snapshot(session, language)


def online_run_steps(
    session: OnlineAnnealingSession,
    count: int,
    language: str = "de",
) -> OnlineAnnealingSnapshot:
    """Fuehrt bis zu `count` Online-SA-Schritte aus."""

    if count <= 0:
        return online_current_snapshot(session, language)
    if session.state is None:
        evaluate_online_start(session, language)
    for _ in range(count):
        if online_should_stop(session):
            break
        _online_step(session)
    if online_should_stop(session) and session.state is not None:
        session.end_evaluation = session.state.current_evaluation
    return online_current_snapshot(session, language)


def online_run_to_completion(
    session: OnlineAnnealingSession,
    language: str = "de",
) -> OnlineAnnealingSnapshot:
    """Fuehrt die Online-Session bis zum Stopkriterium aus."""

    if session.state is None:
        evaluate_online_start(session, language)
    while not online_should_stop(session):
        _online_step(session)
    if session.state is not None:
        session.end_evaluation = session.state.current_evaluation
    return online_current_snapshot(session, language)


def online_reset(
    session: OnlineAnnealingSession,
    language: str = "de",
) -> OnlineAnnealingSnapshot:
    """Setzt die Online-Session inklusive Gewichten auf den Start zurueck."""

    replacement = create_online_session(session.request)
    session.model = replacement.model
    session.rng = replacement.rng
    session.batch_cursor = replacement.batch_cursor
    session.state = None
    session.end_evaluation = None
    return online_current_snapshot(session, language)


def online_current_snapshot(
    session: OnlineAnnealingSession,
    language: str = "de",
) -> OnlineAnnealingSnapshot:
    """Liefert den aktuellen Online-SA-Zustand."""

    state = session.state
    if state is None:
        return OnlineAnnealingSnapshot(
            is_initialized=False,
            is_complete=False,
            state_label="preview" if language == "en" else "Vorschau",
            decision_label="no step yet" if language == "en" else "noch kein Schritt",
            reason_label="not started" if language == "en" else "nicht gestartet",
            start_evaluation=None,
            candidate_evaluation=None,
            current_evaluation=None,
            best_evaluation=None,
            end_evaluation=None,
            last_step=None,
            history=(),
            accepted_steps=0,
            acceptance_rate=0.0,
            current_temperature=session.request.annealing_config.start_temperature,
            stop_reasons=(),
            epoch_index=session.batch_cursor.epoch_index,
            batch_index=session.batch_cursor.batch_index,
            batch_start=session.batch_cursor.batch_start,
        )
    stop_reasons = tuple(_online_stop_reasons(session))
    last_step = state.history[-1] if state.history else None
    return OnlineAnnealingSnapshot(
        is_initialized=True,
        is_complete=bool(stop_reasons),
        state_label=_state_label(language, bool(stop_reasons), state.step_index),
        decision_label=_decision_label(last_step.reason_code if last_step is not None else None, language),
        reason_label=_reason_label(stop_reasons, state.latest_neighbor_label, language),
        start_evaluation=state.start_evaluation,
        candidate_evaluation=state.latest_candidate,
        current_evaluation=state.current_evaluation,
        best_evaluation=state.best_evaluation,
        end_evaluation=session.end_evaluation,
        last_step=last_step,
        history=tuple(state.history),
        accepted_steps=state.accepted_steps,
        acceptance_rate=state.acceptance_rate,
        current_temperature=state.current_temperature,
        stop_reasons=stop_reasons,
        epoch_index=session.batch_cursor.epoch_index,
        batch_index=session.batch_cursor.batch_index,
        batch_start=session.batch_cursor.batch_start,
    )


def online_should_stop(session: OnlineAnnealingSession) -> bool:
    state = session.state
    if state is None:
        return True
    return bool(_online_stop_reasons(session))


def _online_step(session: OnlineAnnealingSession) -> OnlineAnnealingStep:
    state = session.state
    if state is None:
        raise ValueError("OnlineAnnealingSession muss zuerst initialisiert werden.")

    previous_layout = session.model.layout
    X_batch, y_batch = _current_batch(session)
    batch_loss_before, batch_acc_before = evaluate_batch(session.model, X_batch, y_batch)
    previous_evaluation = _build_evaluation(
        session,
        previous_layout,
        batch_loss=batch_loss_before,
        batch_accuracy=batch_acc_before,
        phase="before_candidate",
    )

    neighbors = generate_neighbors(previous_layout, session.request.annealing_config.neighborhood_operations)
    if not neighbors:
        raise ValueError("Fuer das aktuelle Layout wurden keine Nachbarn erzeugt.")
    if tuple(session.request.annealing_config.neighborhood_operations) == ("set_neuron",):
        chosen_neighbor = sample_set_neuron_neighbor(previous_layout, session.rng)
    else:
        chosen_neighbor = sample_neighbor(
            previous_layout,
            session.request.annealing_config.neighborhood_operations,
            session.rng,
        )

    session.model.set_layout(chosen_neighbor.layout)
    candidate_loss_after, candidate_acc_after = evaluate_batch(session.model, X_batch, y_batch)
    candidate_evaluation = _build_evaluation(
        session,
        chosen_neighbor.layout,
        batch_loss=candidate_loss_after,
        batch_accuracy=candidate_acc_after,
        phase="candidate_no_training",
    )

    delta = float(candidate_loss_after - batch_loss_before)
    current_temperature = _temperature_for_iteration(session, state.step_index)
    probability = acceptance_probability(delta, current_temperature)
    random_draw = float(session.rng.random())
    accepted = delta <= 0.0 or random_draw <= probability
    reason_code = "improved_or_equal"
    if delta > 0.0:
        reason_code = "accepted_worse" if accepted else "rejected_worse"

    trained_after_accept = False
    if accepted:
        state.accepted_steps += 1
        if delta > 0.0:
            state.accepted_worse_steps += 1
        trained_after_accept = True
        train_one_batch(
            session.model,
            X_batch,
            y_batch,
            session.request.training_config.learning_rate,
        )
        post_loss, post_acc = evaluate_batch(session.model, X_batch, y_batch)
        current_evaluation = _build_evaluation(
            session,
            session.model.layout,
            batch_loss=post_loss,
            batch_accuracy=post_acc,
            phase="accepted_after_training",
        )
        state.current_evaluation = current_evaluation
        if current_evaluation.val_loss < state.best_evaluation.val_loss:
            state.best_evaluation = current_evaluation
    else:
        session.model.set_layout(previous_layout)
        state.rejected_steps += 1
        state.current_evaluation = previous_evaluation

    state.step_index += 1
    state.current_temperature = _temperature_for_iteration(session, state.step_index)
    state.latest_candidate = candidate_evaluation
    state.latest_neighbor_label = chosen_neighbor.label
    session.batch_cursor.advance()

    step_record = OnlineAnnealingStep(
        step_index=state.step_index,
        temperature=current_temperature,
        previous_layout=previous_layout,
        candidate_layout=chosen_neighbor.layout,
        previous_evaluation=previous_evaluation,
        candidate_evaluation=candidate_evaluation,
        delta=delta,
        acceptance_probability=float(probability),
        random_draw=random_draw,
        accepted=accepted,
        reason_code=reason_code,
        neighbor_label=chosen_neighbor.label,
        best_score_after_step=float(state.best_evaluation.val_loss),
        accepted_steps_after_step=state.accepted_steps,
        rejected_steps_after_step=state.rejected_steps,
        batch_loss_before=float(batch_loss_before),
        candidate_loss_after=float(candidate_loss_after),
        trained_after_accept=trained_after_accept,
        epoch_index=session.batch_cursor.epoch_index,
        batch_index=session.batch_cursor.batch_index,
        batch_start=session.batch_cursor.batch_start,
        validation_loss_after_update=float(state.current_evaluation.val_loss),
    )
    state.history.append(step_record)
    return step_record


def _build_evaluation(
    session: OnlineAnnealingSession,
    layout: ActivationLayout,
    *,
    batch_loss: float,
    batch_accuracy: float,
    phase: str,
) -> OnlineLayoutEvaluation:
    train_loss, train_acc = session.model.evaluate(session.dataset.X_train, session.dataset.y_train)
    val_loss, val_acc = session.model.evaluate(session.dataset.X_val, session.dataset.y_val)
    if session.request.include_test_metrics:
        test_loss, test_acc = session.model.evaluate(session.dataset.X_test, session.dataset.y_test)
    else:
        test_loss, test_acc = float("nan"), float("nan")
    return OnlineLayoutEvaluation(
        layout=layout,
        comparable_score=float(batch_loss),
        objective_value=float(batch_loss),
        train_loss=float(train_loss),
        val_loss=float(val_loss),
        test_loss=float(test_loss),
        train_accuracy=float(train_acc),
        val_accuracy=float(val_acc),
        test_accuracy=float(test_acc),
        trained_model=session.model.clone(),
        metadata={
            "sa_evaluation_mode": "online_delta",
            "phase": phase,
            "batch_loss": f"{float(batch_loss):.12g}",
            "batch_accuracy": f"{float(batch_accuracy):.12g}",
            "best_layout_selection_metric": "validation_loss_after_update",
        },
    )


def _current_batch(session: OnlineAnnealingSession) -> tuple[np.ndarray, np.ndarray]:
    indices = session.batch_cursor.current_indices()
    return session.dataset.X_train[indices], session.dataset.y_train[indices]


def _temperature_for_iteration(session: OnlineAnnealingSession, iteration_index: int) -> float:
    cooling_step_index = iteration_index // session.request.annealing_config.iterations_per_temperature
    return temperature_for_step(
        schedule_name=session.request.annealing_config.cooling_schedule,
        start_temperature=session.request.annealing_config.start_temperature,
        cooling_parameter=session.request.annealing_config.cooling_parameter,
        cooling_step_index=cooling_step_index,
    )


def _online_stop_reasons(session: OnlineAnnealingSession) -> list[str]:
    state = session.state
    if state is None:
        return []
    reasons: list[str] = []
    if state.step_index >= session.request.annealing_config.max_steps:
        reasons.append("max_steps_reached")
    if state.current_temperature <= session.request.annealing_config.min_temperature:
        reasons.append("temperature_below_threshold")
    if not generate_neighbors(state.current_evaluation.layout, session.request.annealing_config.neighborhood_operations):
        reasons.append("no_neighbors")
    return reasons


def _state_label(language: str, is_complete: bool, step_index: int) -> str:
    if is_complete:
        return "complete" if language == "en" else "abgeschlossen"
    if step_index == 0:
        return "start" if language == "en" else "Start"
    return "running" if language == "en" else "laeuft"


def _decision_label(reason_code: str | None, language: str) -> str:
    mapping_en = {
        "improved_or_equal": "improved or equal",
        "accepted_worse": "accepted worse",
        "rejected_worse": "rejected worse",
    }
    mapping_de = {
        "improved_or_equal": "verbessert oder gleich gut",
        "accepted_worse": "schlechter, aber akzeptiert",
        "rejected_worse": "schlechter und abgelehnt",
    }
    if reason_code is None:
        return "no step yet" if language == "en" else "noch kein Schritt"
    return (mapping_en if language == "en" else mapping_de).get(reason_code, reason_code)


def _reason_label(stop_reasons: tuple[str, ...], neighbor_label: str, language: str) -> str:
    if stop_reasons:
        mapping_en = {
            "max_steps_reached": "maximum number of steps reached",
            "temperature_below_threshold": "temperature below threshold",
            "no_neighbors": "no valid neighbors left",
        }
        mapping_de = {
            "max_steps_reached": "maximale Schrittzahl erreicht",
            "temperature_below_threshold": "Temperatur unter Mindestwert",
            "no_neighbors": "keine gueltigen Nachbarn mehr",
        }
        return ", ".join((mapping_en if language == "en" else mapping_de).get(reason, reason) for reason in stop_reasons)
    if neighbor_label:
        return neighbor_label
    return "not started" if language == "en" else "nicht gestartet"
