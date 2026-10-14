import os
import sys

# Pyplot must not open real windows during tests; FigureCanvasTkAgg still works under Agg.
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest

from _support import REPO_ROOT

# Import PRISM as the `prism` package from this checkout (whether or not it is also installed).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _isolate_sys_path(monkeypatch):
    """core.loader prepends user project roots to sys.path and never removes them."""
    monkeypatch.setattr(sys, "path", list(sys.path))


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    """PyTorchInspector moves models to CUDA when it can; keep every run on CPU."""
    try:
        import torch
    except ImportError:
        return
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


class DialogRecorder:
    def __init__(self):
        self.calls = []  # (kind, title, message)
        self.answer = False  # what yes/no questions (askyesno) are answered with

    @property
    def kinds(self):
        return [kind for kind, _, _ in self.calls]

    def messages(self, kind):
        return [msg for k, _, msg in self.calls if k == kind]


@pytest.fixture
def dialogs(monkeypatch):
    """Record message boxes instead of blocking; fail loudly if a test opens a file picker."""
    from tkinter import filedialog, messagebox, simpledialog

    recorder = DialogRecorder()
    for kind in ("showinfo", "showwarning", "showerror"):
        def show(title, message="", *args, _kind=kind, **kwargs):
            recorder.calls.append((_kind, title, message))
        monkeypatch.setattr(messagebox, kind, show)

    def ask(title, message="", *args, **kwargs):
        recorder.calls.append(("askyesno", title, message))
        return recorder.answer

    monkeypatch.setattr(messagebox, "askyesno", ask)

    def interactive(*args, **kwargs):
        raise AssertionError("test triggered an interactive dialog")

    monkeypatch.setattr(filedialog, "askdirectory", interactive)
    monkeypatch.setattr(filedialog, "askopenfilename", interactive)
    monkeypatch.setattr(simpledialog, "askstring", interactive)
    return recorder


@pytest.fixture
def app(dialogs):
    """A fresh, hidden PrismApp with all seven tabs built."""
    tk = pytest.importorskip("tkinter")
    from prism.gui.app import PrismApp

    window = None
    for attempt in range(3):  # Tk init occasionally fails to read its own .tcl files on Windows
        try:
            window = PrismApp()
            break
        except tk.TclError as exc:
            error = exc
    if window is None:
        pytest.skip(f"Tk display unavailable: {error}")
    window.withdraw()
    yield window
    window.destroy()
