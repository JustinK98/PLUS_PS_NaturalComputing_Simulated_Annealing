from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from ui_qt.tasking import BackgroundTaskController, SequentialTaskRunnable


def _app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class SequentialTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _app()

    def test_sequential_runner_continues_after_item_failure(self) -> None:
        successes: list[tuple[int, int]] = []
        failures: list[int] = []

        def worker(value: int) -> int:
            if value == 2:
                raise ValueError("expected")
            return value * 10

        runnable = SequentialTaskRunnable(worker, [1, 2, 3])
        runnable.signals.itemSucceeded.connect(lambda index, result: successes.append((index, result)))
        runnable.signals.itemFailed.connect(lambda index, _trace: failures.append(index))

        runnable.run()

        self.assertEqual(successes, [(0, 10), (2, 30)])
        self.assertEqual(failures, [1])

    def test_sequential_runner_stops_between_items_after_cancel(self) -> None:
        processed: list[int] = []
        runnable = SequentialTaskRunnable(lambda value: processed.append(value), [1, 2, 3])
        runnable.cancel()

        runnable.run()

        self.assertEqual(processed, [])

    def test_controller_marks_task_busy_before_thread_starts(self) -> None:
        controller = BackgroundTaskController()

        controller.submit(lambda: 1)

        self.assertTrue(controller.is_busy)
        controller._pool.waitForDone(5_000)
        self.app.processEvents()
        self.assertFalse(controller.is_busy)


if __name__ == "__main__":
    unittest.main()
