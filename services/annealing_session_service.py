"""Interaktive Session-Logik fuer didaktisches Simulated Annealing."""

from __future__ import annotations

from dataclasses import dataclass

from activations import parse_layout_spec
from annealing import AnnealingStep, annealing_stop_reasons
from annealing_objectives import ObjectiveEvaluation
from annealing_runner import AnnealingRunner
from services.annealing_service import AnnealingRunRequest


@dataclass
class AnnealingSession:
    """Laufende SA-Session inklusive Runner und Evaluator-Cache."""

    request: AnnealingRunRequest
    dataset: object
    start_layout: object
    runner: AnnealingRunner
    end_evaluation: ObjectiveEvaluation | None = None


@dataclass(frozen=True)
class AnnealingSessionSnapshot:
    """Serialisierbarer Zustand fuer Qt-Panels und Tests."""

    is_initialized: bool
    is_complete: bool
    state_label: str
    decision_label: str
    reason_label: str
    start_evaluation: ObjectiveEvaluation | None
    candidate_evaluation: ObjectiveEvaluation | None
    current_evaluation: ObjectiveEvaluation | None
    best_evaluation: ObjectiveEvaluation | None
    end_evaluation: ObjectiveEvaluation | None
    last_step: AnnealingStep | None
    history: tuple[AnnealingStep, ...]
    accepted_steps: int
    acceptance_rate: float
    current_temperature: float
    stop_reasons: tuple[str, ...]


def _state_label(snapshot_language: str, *, is_initialized: bool, is_complete: bool, step_index: int) -> str:
    if not is_initialized:
        return "preview" if snapshot_language == "en" else "Vorschau"
    if is_complete:
        return "complete" if snapshot_language == "en" else "abgeschlossen"
    if step_index == 0:
        return "start" if snapshot_language == "en" else "Start"
    return "running" if snapshot_language == "en" else "laeuft"


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
        translated = [
            (mapping_en if language == "en" else mapping_de).get(reason, reason)
            for reason in stop_reasons
        ]
        return ", ".join(translated)
    if neighbor_label:
        return neighbor_label
    return "not started" if language == "en" else "nicht gestartet"


def create_session(request: AnnealingRunRequest) -> AnnealingSession:
    """Erzeugt eine neue interaktive SA-Session aus einer bestehenden Request-Struktur."""

    from annealing_objectives import LayoutObjectiveEvaluator
    from benchmarks import load_benchmark

    dataset = load_benchmark(request.dataset_config)
    start_layout = parse_layout_spec(request.layout_spec, request.hidden_sizes)
    evaluator = LayoutObjectiveEvaluator(dataset, request.objective_config)
    runner = AnnealingRunner(
        evaluator=evaluator,
        config=request.annealing_config,
        random_state=request.random_state,
    )
    return AnnealingSession(
        request=request,
        dataset=dataset,
        start_layout=start_layout,
        runner=runner,
    )


def current_snapshot(session: AnnealingSession, language: str = "de") -> AnnealingSessionSnapshot:
    """Liefert den aktuellen Session-Zustand als kompaktes DTO."""

    state = session.runner.state
    if state is None:
        return AnnealingSessionSnapshot(
            is_initialized=False,
            is_complete=False,
            state_label=_state_label(language, is_initialized=False, is_complete=False, step_index=0),
            decision_label=_decision_label(None, language),
            reason_label=_reason_label((), "", language),
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
        )

    neighbors = session.runner.evaluator.cache_size()
    _ = neighbors  # keep cache query explicit for debugging; actual stop reasons use runner logic below.
    stop_reasons = tuple(_current_stop_reasons(session))
    last_step = state.history[-1] if state.history else None
    is_complete = bool(stop_reasons)
    return AnnealingSessionSnapshot(
        is_initialized=True,
        is_complete=is_complete,
        state_label=_state_label(
            language,
            is_initialized=True,
            is_complete=is_complete,
            step_index=state.step_index,
        ),
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
    )


def evaluate_start(session: AnnealingSession, language: str = "de") -> AnnealingSessionSnapshot:
    """Initialisiert die Session und liefert den Startzustand."""

    if session.runner.state is None:
        session.runner.initialize(session.start_layout)
        session.end_evaluation = None
    return current_snapshot(session, language)


def step_once(session: AnnealingSession, language: str = "de") -> AnnealingSessionSnapshot:
    """Fuehrt genau einen SA-Schritt aus."""

    if session.runner.state is None:
        session.runner.initialize(session.start_layout)
    if not session.runner.should_stop():
        session.runner.step()
    if session.runner.should_stop() and session.runner.state is not None:
        session.end_evaluation = session.runner.state.current_evaluation
    return current_snapshot(session, language)


def run_to_completion(session: AnnealingSession, language: str = "de") -> AnnealingSessionSnapshot:
    """Laeuft die Session bis zum Stoppkriterium fertig."""

    if session.runner.state is None:
        session.runner.initialize(session.start_layout)
    if not session.runner.should_stop():
        session.runner.run_until_complete()
    if session.runner.state is not None:
        session.end_evaluation = session.runner.state.current_evaluation
    return current_snapshot(session, language)


def reset(session: AnnealingSession, language: str = "de") -> AnnealingSessionSnapshot:
    """Setzt die Session auf den uninitialisierten Startzustand zurueck."""

    session.runner = AnnealingRunner(
        evaluator=session.runner.evaluator,
        config=session.runner.config,
        random_state=session.request.random_state,
    )
    session.end_evaluation = None
    return current_snapshot(session, language)


def _current_stop_reasons(session: AnnealingSession) -> list[str]:
    state = session.runner.state
    if state is None:
        return []
    if state.current_evaluation is None:
        return []
    from activations import generate_neighbors

    neighbors = generate_neighbors(
        state.current_evaluation.layout,
        session.request.annealing_config.neighborhood_operations,
    )
    return annealing_stop_reasons(
        state,
        session.request.annealing_config,
        len(neighbors),
    )
