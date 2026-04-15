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


class BackgroundTaskController(QtCore.QObject):
    """Zentraler Task-Controller pro Workspace."""

    busyChanged = QtCore.Signal(bool)
    messageEmitted = QtCore.Signal(str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._pool = QtCore.QThreadPool.globalInstance()
        self._running = 0

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
        runnable.signals.started.connect(self._on_started)
        runnable.signals.finished.connect(self._on_finished)
        if on_success is not None:
            runnable.signals.succeeded.connect(on_success)
        if on_error is not None:
            runnable.signals.failed.connect(on_error)
        if status_message:
            runnable.signals.started.connect(lambda: self.messageEmitted.emit(status_message))
        self._pool.start(runnable)

    def _on_started(self) -> None:
        self._running += 1
        if self._running == 1:
            self.busyChanged.emit(True)

    def _on_finished(self) -> None:
        self._running = max(0, self._running - 1)
        if self._running == 0:
            self.busyChanged.emit(False)
