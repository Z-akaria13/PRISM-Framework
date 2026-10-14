"""gui/worker.py: background Tasks deliver everything on the Tk thread, and Runners keep one at a time."""
import threading
import time

import pytest

from _support import wait_for
from prism.core.cancel import Cancelled
from prism.gui.worker import Runner, Task

pytestmark = pytest.mark.gui


def recorder():
    events = []

    def record(name):
        return lambda *args: events.append((name, args, threading.current_thread() is threading.main_thread()))

    callbacks = {f"on_{n}": record(n) for n in ("done", "error", "cancel", "progress", "log", "status", "finish")}
    return events, callbacks


def names(events):
    return [name for name, _, _ in events]


def test_a_task_runs_off_the_tk_thread_and_reports_on_it(app):
    events, callbacks = recorder()
    worker_threads = []

    def work(task):
        worker_threads.append(threading.current_thread() is threading.main_thread())
        task.status("halfway")
        task.progress(0.5)
        task.log("hello")
        return 42

    task = wait_for(app, Task(app, work, **callbacks).start())

    assert worker_threads == [False]
    assert names(events) == ["status", "progress", "log", "done", "finish"]
    assert all(on_tk_thread for _, _, on_tk_thread in events)
    assert events[3][1] == (42,) and task.state == "done" and task.result == 42


def test_an_error_is_delivered_with_its_traceback(app):
    events, callbacks = recorder()

    def work(task):
        raise ValueError("bad data")

    task = wait_for(app, Task(app, work, **callbacks).start())

    assert names(events) == ["error", "finish"]
    error, trace = events[0][1]
    assert isinstance(error, ValueError) and "bad data" in trace
    assert task.state == "failed"


def test_cancel_stops_the_work_at_its_next_check(app):
    events, callbacks = recorder()
    started, reached = threading.Event(), []

    def work(task):
        started.set()
        for i in range(1000):
            task.sleep(0.01)
            reached.append(i)
        return "finished anyway"

    task = Task(app, work, **callbacks).start()
    started.wait(5)
    task.cancel()
    wait_for(app, task)

    assert names(events) == ["cancel", "finish"]
    assert task.state == "cancelled" and len(reached) < 1000


def test_progress_raises_once_cancelled():
    task = Task(None, lambda t: None)
    task.progress(0.1)
    task.cancel()

    with pytest.raises(Cancelled):
        task.progress(0.2)


def test_a_new_request_supersedes_the_running_one_and_waits_for_it(app):
    runner = Runner(app)
    log, running = [], []
    release = threading.Event()

    def slow(task):
        running.append("slow")
        release.wait(5)
        task.check()
        return "slow result"

    def fast(task):
        running.append("fast")  # never at the same time as `slow`: it starts once `slow` has ended
        return "fast result"

    first = runner.submit(slow, on_done=log.append, on_cancel=lambda: log.append("slow cancelled"))
    while not running:
        time.sleep(0.001)
    second = runner.submit(fast, on_done=log.append)
    assert running == ["slow"] and runner.busy
    release.set()
    wait_for(app, first)
    wait_for(app, second)

    assert log == ["fast result"]  # the superseded task reports nothing
    assert running == ["slow", "fast"] and not runner.busy


def test_only_the_latest_of_several_waiting_requests_runs(app):
    runner = Runner(app)
    release, ran = threading.Event(), []

    first = runner.submit(lambda task: release.wait(5))
    dropped = runner.submit(lambda task: ran.append("dropped"))
    latest = runner.submit(lambda task: ran.append("latest"))
    release.set()
    wait_for(app, latest)

    assert ran == ["latest"] and dropped.state == "dropped" and first.finished


def test_runner_cancel_reports_the_cancellation(app):
    runner = Runner(app)
    log = []
    task = runner.submit(lambda t: [t.sleep(0.01) for _ in range(1000)], on_cancel=lambda: log.append("cancelled"))

    runner.cancel()
    wait_for(app, task)

    assert log == ["cancelled"]
