"""
BACKGROUND TASKS
Slow work (ingestion, extraction, cluster monitoring, probes, bootstrap intervals, gallery
loads) runs on a worker thread so the window stays responsive. Tk is not thread-safe, so the
worker never touches a widget: it puts messages on a queue, and the Tk thread drains the
queue with after(). Every callback given to a Task therefore runs on the Tk thread.
"""
import queue
import threading
import tkinter as tk
import traceback

from prism.core.cancel import Cancelled

POLL_MS = 30  # how often the Tk thread collects messages from a running task


class Task:
    """
    Runs `work(task)` on a daemon thread. `work` may call task.progress(fraction or percent),
    task.log(text) and task.status(text) to report, task.check() or task.sleep(seconds) to stop
    promptly when cancelled (they raise Cancelled), and returns a result.

    Callbacks, all on the Tk thread: on_progress(value), on_log(text), on_status(text), then
    exactly one of on_done(result), on_error(exception, traceback text) or on_cancel(), then
    on_finish().
    """

    def __init__(self, widget, work, *, on_done=None, on_error=None, on_cancel=None, on_progress=None,
                 on_log=None, on_status=None, on_finish=None):
        self.widget, self.work = widget, work
        self.callbacks = dict(done=on_done, error=on_error, cancelled=on_cancel, progress=on_progress,
                              log=on_log, status=on_status, finish=on_finish)
        self.state = "new"  # running, then done / failed / cancelled; "dropped" if superseded before it ran
        self.result = None
        self._messages = queue.Queue()
        self._cancel = threading.Event()
        self._superseded = False
        self._then = None  # Runner's hook, run after on_finish

    # --- Tk thread -----------------------------------------------------
    def start(self):
        self.state = "running"
        threading.Thread(target=self._run, daemon=True, name="prism-task").start()
        self._schedule()
        return self

    def cancel(self):
        """Ask the work to stop at its next check; on_cancel follows once it has."""
        self._cancel.set()

    @property
    def running(self):
        return self.state == "running"

    @property
    def finished(self):
        return self.state in ("done", "failed", "cancelled", "dropped")

    # --- worker thread -------------------------------------------------
    @property
    def cancelled(self):
        return self._cancel.is_set()

    def check(self):
        if self._cancel.is_set():
            raise Cancelled()

    def sleep(self, seconds):
        if self._cancel.wait(seconds):
            raise Cancelled()

    def progress(self, value):
        self.check()
        self._messages.put(("progress", value))

    def log(self, text):
        self._messages.put(("log", text))

    def status(self, text):
        self._messages.put(("status", text))

    def _run(self):
        try:
            result = self.work(self)
        except Cancelled:
            self._messages.put(("cancelled", None))
        except Exception as e:  # noqa: BLE001 - reported to the user through on_error
            self._messages.put(("failed", (e, traceback.format_exc())))
        else:
            self._messages.put(("done", result))

    # --- message pump (Tk thread) -------------------------------------
    def _schedule(self):
        try:
            self.widget.after(POLL_MS, self._drain)
        except (tk.TclError, RuntimeError):  # the window has been closed
            pass

    def _drain(self):
        while True:
            try:
                kind, payload = self._messages.get_nowait()
            except queue.Empty:
                break
            if kind in ("progress", "log", "status"):
                self._call(kind, payload)
                continue
            self.state = kind
            try:
                if kind == "done":
                    self.result = payload
                    self._call("done", payload)
                elif kind == "failed":
                    self._call("error", *payload)
                else:
                    self._call("cancelled")
            finally:
                self._call("finish", always=True)
                if self._then is not None:
                    self._then()
            return
        self._schedule()

    def _call(self, name, *args, always=False):
        callback = self.callbacks[name]
        if callback is not None and (always or not self._superseded):
            callback(*args)


class Runner:
    """
    At most one task at a time for one tab. A new request supersedes the current task: it is
    asked to stop, its results are dropped, and the new task starts once it has ended, so a
    user's connector is never called from two threads at once.
    """

    def __init__(self, widget):
        self.widget = widget
        self.current = None
        self.pending = None

    @property
    def busy(self):
        return (self.current is not None and self.current.running) or self.pending is not None

    def submit(self, work, **callbacks):
        task = Task(self.widget, work, **callbacks)
        if self.current is not None and self.current.running:
            self.current._superseded = True
            self.current.cancel()
            if self.pending is not None:
                self.pending.state = "dropped"
            self.pending = task
        else:
            self._start(task)
        return task

    def cancel(self):
        """Stop the current task and forget any waiting one."""
        if self.pending is not None:
            self.pending.state = "dropped"
            self.pending = None
        if self.current is not None and self.current.running:
            self.current.cancel()

    def _start(self, task):
        self.current = task
        task._then = self._next
        task.start()

    def _next(self):
        task, self.pending = self.pending, None
        if task is not None:
            self._start(task)
