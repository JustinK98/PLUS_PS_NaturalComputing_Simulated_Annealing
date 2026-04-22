from __future__ import annotations

import time
import unittest

from services.task_runner import BackgroundTaskRunner


class TaskRunnerTests(unittest.TestCase):
    def test_background_task_completes_and_returns_result(self) -> None:
        runner = BackgroundTaskRunner()
        runner.start(lambda: 21 * 2)
        deadline = time.time() + 2.0
        outcome = None
        while time.time() < deadline and outcome is None:
            outcome = runner.poll()
            if outcome is None:
                time.sleep(0.01)
        self.assertIsNotNone(outcome)
        assert outcome is not None
        self.assertEqual(outcome.status, "completed")
        self.assertEqual(outcome.result, 42)

    def test_start_rejects_parallel_task(self) -> None:
        runner = BackgroundTaskRunner()
        runner.start(lambda: time.sleep(0.2))
        with self.assertRaises(RuntimeError):
            runner.start(lambda: 1)
        outcome = runner.wait(timeout=1.0)
        self.assertIsNotNone(outcome)
        assert outcome is not None
        self.assertEqual(outcome.status, "completed")
        self.assertFalse(runner.is_running)


if __name__ == "__main__":
    unittest.main()
