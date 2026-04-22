"""Kleiner Task-Runner fuer spaetere GUI-Hintergrundlaeufe."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Queue
from threading import Thread
from typing import Any, Callable


@dataclass(frozen=True)
class TaskOutcome:
    """Ergebnis oder Fehler eines Hintergrund-Tasks."""

    status: str
    result: Any = None
    error: BaseException | None = None


class BackgroundTaskRunner:
    """Führt genau einen Hintergrund-Task und liefert das Ergebnis per Polling."""

    def __init__(self) -> None:
        self._queue: Queue[TaskOutcome] = Queue(maxsize=1)
        self._thread: Thread | None = None

    @property
    def is_running(self) -> bool:
        """Ob aktuell ein Hintergrund-Task laeuft."""

        return self._thread is not None and self._thread.is_alive()

    def start(self, task: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Startet einen neuen Task, sofern aktuell keiner laeuft."""

        if self.is_running:
            raise RuntimeError("Es laeuft bereits ein Hintergrund-Task.")

        def _worker() -> None:
            try:
                result = task(*args, **kwargs)
                self._queue.put(TaskOutcome(status="completed", result=result))
            except BaseException as exc:  # pragma: no cover - defensive thread wrapper
                self._queue.put(TaskOutcome(status="failed", error=exc))

        self._thread = Thread(target=_worker)
        self._thread.start()

    def poll(self) -> TaskOutcome | None:
        """Liefert ein fertiges Ergebnis, falls eines vorliegt."""

        try:
            outcome = self._queue.get_nowait()
        except Empty:
            return None
        self._join_finished_thread()
        return outcome

    def wait(self, timeout: float | None = None) -> TaskOutcome | None:
        """Wartet optional auf den aktuellen Task und liefert dessen Ergebnis."""

        thread = self._thread
        if thread is not None:
            thread.join(timeout)
        return self.poll()

    def _join_finished_thread(self) -> None:
        thread = self._thread
        if thread is not None and not thread.is_alive():
            thread.join(timeout=0)
            self._thread = None
