"""Shared helpers for the PRISM test suite (imported by test modules, never collected).

Everything that drives the extraction engine goes through `run_engine`, so a change to
the engine's signature only needs to be absorbed here.
"""
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PRISM_DIR = REPO_ROOT / "prism"
EXAMPLES_DIR = REPO_ROOT / "examples"
HEADLESS = PRISM_DIR / "headless.py"
H5_NAME = "prism_embeddings.h5"
CSV_NAME = "prism_metadata.csv"
FAILED_NAME = "prism_failed_rows.csv"
IDS_NAME = "__ids__"


def find_bash():
    """A bash that can run the generated script: $PRISM_TEST_BASH, Git Bash, or bash on PATH."""
    candidates = [os.environ.get("PRISM_TEST_BASH")]
    if os.name == "nt":
        candidates += [r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"]
    on_path = shutil.which("bash")
    if on_path and "system32" not in on_path.lower():  # WSL's bash.exe cannot run Windows Python paths
        candidates.append(on_path)
    return next((c for c in candidates if c and os.path.exists(c)), None)


def known_bug(ref):
    """Pin a defect listed in debt.MD.

    Strict: once the bug is fixed the test XPASSes, which fails the run as a reminder
    to delete the marker and keep the test as a regression guard.
    """
    return pytest.mark.xfail(reason=f"known bug, debt.MD {ref}", strict=True)


# ---------------------------------------------------------------------------
# Engine driver
# ---------------------------------------------------------------------------

@dataclass
class EngineRun:
    ok: bool
    out_dir: Path
    logs: list = field(default_factory=list)
    progress: list = field(default_factory=list)

    @property
    def metadata(self):
        return pd.read_csv(self.out_dir / CSV_NAME)

    @property
    def failed_rows(self):
        """prism_failed_rows.csv, or None if the run wrote none."""
        path = self.out_dir / FAILED_NAME
        return pd.read_csv(path) if path.exists() else None

    def datasets(self):
        """Model output and layer datasets (everything except the row IDs)."""
        with h5py.File(self.out_dir / H5_NAME, "r") as f:
            return {name: f[name][:] for name in f.keys() if name != IDS_NAME}

    def ids(self):
        with h5py.File(self.out_dir / H5_NAME, "r") as f:
            return list(f[IDS_NAME].asstr()[:])


def run_engine(out_dir, metadata, connector, inspector, model=None, layers=(),
               batch_size=4, id_col="subject_id", target_col="label", group_col=None,
               device="auto", overwrite=False, progress_callback=None, max_failed_percent=None):
    """Run the extraction in-process. `layers` is {name: pooling} or a list of names (default pooling).
    `max_failed_percent` defaults to AuditConfig's."""
    from prism.core.config import DEFAULT_POOLING, AuditConfig
    from prism.core.execution import run_extraction
    from prism.core.subject import Subject

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not isinstance(layers, dict):
        layers = {name: DEFAULT_POOLING for name in layers}
    config = AuditConfig(
        project_root="", script_path="", framework="in-process", csv_path="",
        id_col=id_col, target_col=target_col, group_col=group_col, layers=layers,
        batch_size=batch_size, output_dir=str(out_dir), device=device, overwrite=overwrite,
        **({} if max_failed_percent is None else {"max_failed_percent": max_failed_percent}),
    )
    run = EngineRun(ok=False, out_dir=out_dir)
    progress = progress_callback or run.progress.append
    run.ok = run_extraction(config, Subject(connector, model, inspector), metadata, progress, run.logs.append)
    return run


def write_legacy_cache(out_dir, metadata, model_output, layers=None):
    """Write a cache in the pre-contract layout (no row IDs, no recorded settings).

    The analysis tabs must keep reading these, so tests that only care about the analysis
    use this helper; caches in the current format come from `run_engine`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(out_dir / CSV_NAME, index=False)
    with h5py.File(out_dir / H5_NAME, "w") as f:
        f.create_dataset("__model_output__", data=np.asarray(model_output))
        for name, values in (layers or {}).items():
            f.create_dataset(name, data=np.asarray(values))
    return out_dir


def value_frame(n):
    """Metadata where each row's `value` is unique, so outputs can be traced back to rows."""
    return pd.DataFrame({
        "subject_id": [f"S{i:03d}" for i in range(n)],
        "value": range(n),
        "label": [i % 2 for i in range(n)],
    })


# ---------------------------------------------------------------------------
# In-process connectors
# ---------------------------------------------------------------------------

class ValueConnector:
    """Each item is the row's `value` and inference echoes it back, so alignment is checkable."""

    project_root = ""

    def __init__(self, fail_load=(), fail_infer=()):
        self.fail_load = set(fail_load)
        self.fail_infer = set(fail_infer)
        self.calls = []  # (row_id, value) for every load_item call

    @property
    def seen_row_ids(self):
        return [row_id for row_id, _ in self.calls]

    def load_model(self):
        return None

    def load_item(self, row_id, meta):
        self.calls.append((row_id, meta["value"]))
        if meta["value"] in self.fail_load:
            raise IOError(f"cannot read item {meta['value']}")
        return meta["value"]

    def run_inference(self, model, batch):
        if self.fail_infer & set(batch):
            raise RuntimeError("inference blew up")
        return np.asarray(batch, dtype=float)

    def render_item_for_gallery(self, item):
        return {"type": "text", "payload": f"value={item}"}


class TensorConnector:
    """Items are deterministic random tensors seeded by the row's `value`."""

    project_root = ""

    def __init__(self, shape, per_item=False):
        self.shape = shape
        self.per_item = per_item

    def load_item(self, row_id, meta):
        import torch
        gen = torch.Generator().manual_seed(int(meta["value"]))
        return torch.randn(self.shape, generator=gen)

    def run_inference(self, model, batch):
        import torch
        with torch.no_grad():
            if self.per_item:  # the "just loop over batch_data" pattern the template suggests
                return torch.cat([model(item.unsqueeze(0)) for item in batch])
            return model(torch.stack(batch))

    def inputs_for(self, metadata):
        import torch
        return torch.stack([self.load_item(None, row) for row in metadata.to_dict("records")])


def capture_activation(model, layer_name, inputs):
    """Reference activation of one (non-reused) module, computed in a single forward pass."""
    import torch
    module = dict(model.named_modules())[layer_name]
    store = {}

    def hook(mod, args, out):  # must return None, or the hook replaces the module's output
        store.setdefault("out", out.detach().cpu().numpy())

    handle = module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(inputs)
    finally:
        handle.remove()
    return store["out"]


def tiny_cnn():
    import torch
    import torch.nn as nn

    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 3, 3, padding=1)
            self.act = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(3, 2)

        def forward(self, x):
            return self.fc(self.pool(self.act(self.conv(x))).flatten(1)).softmax(dim=1)

    torch.manual_seed(0)
    return TinyCNN().eval()


# ---------------------------------------------------------------------------
# Connector scripts (for loader / Setup tab / headless tests)
# ---------------------------------------------------------------------------

BLACK_BOX_CONNECTOR = '''
import numpy as np
from prism import PrismConnector

class EchoConnector(PrismConnector):
    def load_model(self):
        return "black-box-model"

    def load_item(self, row_id, meta):
        return float(meta["value"])

    def run_inference(self, model, batch):
        return np.asarray(batch) / 10.0

    def render_item_for_gallery(self, item):
        return {"type": "text", "payload": str(item)}
'''

TORCH_CONNECTOR = '''
import torch
import torch.nn as nn
from prism import PrismConnector

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 3, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc(self.pool(torch.relu(self.conv(x))).flatten(1))

class TorchConnector(PrismConnector):
    def load_model(self):
        torch.manual_seed(0)
        return Net().eval()

    def load_item(self, row_id, meta):
        return torch.full((1, 4, 4), float(meta["value"]))

    def run_inference(self, model, batch):
        with torch.no_grad():
            return model(torch.stack(batch))

    def render_item_for_gallery(self, item):
        return {"type": "text", "payload": "tensor"}
'''


def write_script(path, source):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source))
    return path


# ---------------------------------------------------------------------------
# Headless runner
# ---------------------------------------------------------------------------

def make_project(root, connector_source, n=6):
    """A connector script plus metadata CSV, laid out like a user's project."""
    root = Path(root)
    script = write_script(root / "connector.py", connector_source)
    csv = root / "meta.csv"
    value_frame(n).to_csv(csv, index=False)
    return script, csv


def write_job(path, *, script, csv, project_root, output_dir, framework, layers, batch_size=4):
    """Write a job file by hand (not via AuditConfig), so tests pin the on-disk format.

    `layers` may be a {name: pooling} dict (current format) or a list of
    "name [Pooling]" labels (job files written before AuditConfig).
    """
    config = {
        "project_root": str(project_root),
        "script_path": str(script),
        "framework": framework,
        "csv_path": str(csv),
        "id_col": "subject_id",
        "target_col": "label",
        "group_col": None,
        "layers": layers,
        "batch_size": batch_size,
        "output_dir": str(output_dir),
    }
    Path(path).write_text(json.dumps(config, indent=2))
    return Path(path)


def run_headless(config_path, cwd):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [sys.executable, str(HEADLESS), "--config", str(config_path)],
        cwd=str(cwd), capture_output=True, text=True, env=env, timeout=300,
    )


# ---------------------------------------------------------------------------
# GUI helpers
# ---------------------------------------------------------------------------

def set_entry(entry, text):
    entry.delete(0, "end")
    entry.insert(0, text)


def run_mainloop_until(app, condition, timeout=60):
    """Worker threads call `after()`, which Tkinter only allows while mainloop() is running."""
    deadline = time.monotonic() + timeout

    def check():
        if condition() or time.monotonic() > deadline:
            app.quit()
        else:
            app.after(20, check)

    app.after(20, check)
    app.mainloop()
    app.update()
    assert condition(), "GUI did not reach the expected state in time"


def wait_for(app, task, timeout=60):
    """Pump Tk until a background Task has finished and its callbacks have run; returns it."""
    deadline = time.monotonic() + timeout
    while not task.finished:
        if time.monotonic() > deadline:
            raise AssertionError(f"background task still {task.state} after {timeout} s")
        app.update()
        time.sleep(0.005)
    app.update()
    return task


def wait_idle(app, runner, timeout=60):
    """Pump Tk until a tab's Runner has no task running or waiting."""
    deadline = time.monotonic() + timeout
    while runner.busy:
        if time.monotonic() > deadline:
            raise AssertionError(f"background work still running after {timeout} s")
        app.update()
        time.sleep(0.005)
    app.update()


def ingest(app, root, script, framework):
    """Ingest through the Setup tab and wait for it, as a user clicking INGEST SUBJECT would."""
    return wait_for(app, app.frame_setup.ingest(str(root), str(script), framework))
