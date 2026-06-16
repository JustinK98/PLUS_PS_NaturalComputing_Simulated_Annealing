"""Qt-Hintergrundtasks fuer nicht-blockierende GUI-Läufe."""

from __future__ import annotations

from collections.abc import Callable
import traceback
from typing import Any

from PySide6 import QtCore


class TaskSignals(QtCore.QObject):
    started = QtCore.Signal()
    finished = QtCore.Signal()
    succeeded = QtCore.Signal(object)
    failed = QtCore.Signal(str)


class TaskRunnable(QtCore.QRunnable):
    """Wrapper um einen beliebigen Callable fuer den Qt-Threadpool."""

    def __init__(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.signals = TaskSignals()

    def run(self) -> None:
        self.signals.started.emit()
        try:
            result = self.func(*self.args, **self.kwargs)
        except Exception:  # pragma: no cover - Qt thread wrapper
            self.signals.failed.emit(traceback.format_exc())
        else:
            self.signals.succeeded.emit(result)
        finally:
            self.signals.finished.emit()


class SequentialTaskSignals(QtCore.QObject):
    started = QtCore.Signal()
    itemStarted = QtCore.Signal(int)
    itemSucceeded = QtCore.Signal(int, object)
    itemFailed = QtCore.Signal(int, str)
    finished = QtCore.Signal()


class SequentialTaskRunnable(QtCore.QRunnable):
    """Fuehrt eine Liste von Items nacheinander aus und isoliert Run-Fehler."""

    def __init__(self, func: Callable[[Any], Any], items: list[Any]) -> None:
        super().__init__()
        self.func = func
        self.items = items
        self.signals = SequentialTaskSignals()
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        self.signals.started.emit()
        try:
            for index, item in enumerate(self.items):
                if self._cancel_requested:
                    break
                self.signals.itemStarted.emit(index)
                try:
                    result = self.func(item)
                except Exception:  # pragma: no cover - Qt thread wrapper
                    self.signals.itemFailed.emit(index, traceback.format_exc())
                else:
                    self.signals.itemSucceeded.emit(index, result)
        finally:
            self.signals.finished.emit()


class BackgroundTaskController(QtCore.QObject):
    """Zentraler Task-Controller pro Workspace."""

    busyChanged = QtCore.Signal(bool)
    messageEmitted = QtCore.Signal(str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._pool = QtCore.QThreadPool.globalInstance()
        self._running = 0
        self._sequential_task: SequentialTaskRunnable | None = None

    @property
    def is_busy(self) -> bool:
        return self._running > 0

    def submit(
        self,
        func: Callable[..., Any],
        *args: Any,
        on_success: Callable[[Any], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        status_message: str | None = None,
        **kwargs: Any,
    ) -> None:
        runnable = TaskRunnable(func, *args, **kwargs)
        runnable.signals.finished.connect(self._on_finished)
        if on_success is not None:
            runnable.signals.succeeded.connect(on_success)
        if on_error is not None:
            runnable.signals.failed.connect(on_error)
        self._on_started()
        if status_message:
            self.messageEmitted.emit(status_message)
        self._pool.start(runnable)

    def submit_sequential(
        self,
        func: Callable[[Any], Any],
        items: list[Any],
        *,
        on_item_started: Callable[[int], None] | None = None,
        on_item_success: Callable[[int, Any], None] | None = None,
        on_item_error: Callable[[int, str], None] | None = None,
        on_finished: Callable[[], None] | None = None,
        status_message: str | None = None,
    ) -> None:
        if self._sequential_task is not None:
            raise RuntimeError("Es laeuft bereits ein sequenzieller Sammellauf.")
        runnable = SequentialTaskRunnable(func, items)
        self._sequential_task = runnable
        runnable.signals.finished.connect(self._on_finished)
        runnable.signals.finished.connect(self._clear_sequential_task)
        if on_item_started is not None:
            runnable.signals.itemStarted.connect(on_item_started)
        if on_item_success is not None:
            runnable.signals.itemSucceeded.connect(on_item_success)
        if on_item_error is not None:
            runnable.signals.itemFailed.connect(on_item_error)
        if on_finished is not None:
            runnable.signals.finished.connect(on_finished)
        self._on_started()
        if status_message:
            self.messageEmitted.emit(status_message)
        self._pool.start(runnable)

    def cancel_sequential(self) -> None:
        if self._sequential_task is not None:
            self._sequential_task.cancel()

    def _on_started(self) -> None:
        self._running += 1
        if self._running == 1:
            self.busyChanged.emit(True)

    def _on_finished(self) -> None:
        self._running = max(0, self._running - 1)
        if self._running == 0:
            self.busyChanged.emit(False)

    def _clear_sequential_task(self) -> None:
        self._sequential_task = None
