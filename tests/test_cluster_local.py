"""The cluster path end to end, with a local stand-in for SLURM.

A fake `sbatch` really runs the batch script PRISM generates (with bash, in the background),
so these tests exercise the job file, the generated script, `python -u headless.py`, the log
redirect, the exit-code trailer, and the Execution tab's monitor tailing a log while the job
is still writing it. They cannot cover SLURM itself: queueing, time limits, the compute
node's environment, shared-filesystem paths or the GPU.
"""
import sys
import time

import pandas as pd
import pytest

from _support import find_bash, ingest, run_mainloop_until, set_entry, write_script
from prism.core.cache import AuditCache

pytestmark = [pytest.mark.gui, pytest.mark.slow]


FAKE_SBATCH = '''
import os
import subprocess
import sys

# Like sbatch: start the job in the background and report a job id straight away.
subprocess.Popen([{bash!r}, sys.argv[1]], cwd=os.getcwd(),
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
print("Submitted batch job 4242")
'''

SLOW_CONNECTOR = '''
import os
import time

import numpy as np
from core.connectorTemplate import WhydunnitConnector


class SlowConnector(WhydunnitConnector):
    def load_model(self):
        if os.environ.get("PRISM_TEST_BREAK_ON_NODE"):
            raise RuntimeError("model file missing on the compute node")
        return "model"

    def load_item(self, row_id, meta):
        return float(meta["value"])

    def run_inference(self, model, batch):
        time.sleep(1.0)  # each batch takes a second on the "GPU"
        return np.asarray(batch) / 10.0

    def render_item_for_gallery(self, item):
        return {"type": "text", "payload": str(item)}
'''


def dispatch(app, tmp_path, node_setup=""):
    """Fill in Setup and Evidence, dispatch to the local scheduler, and wait for the monitor.
    Returns (Execution tab, output dir, [(time, progress)], time the monitor finished)."""
    bash = find_bash()
    if bash is None:
        pytest.skip("needs bash (Git Bash on Windows) to run the generated batch script")
    project = tmp_path / "project"
    script = write_script(project / "connector.py", SLOW_CONNECTOR)
    csv = project / "meta.csv"
    pd.DataFrame({"subject_id": [f"S{i}" for i in range(8)], "value": range(8),
                  "label": [i % 2 for i in range(8)], "patient": [f"P{i // 2}" for i in range(8)]}).to_csv(csv, index=False)
    sbatch = write_script(tmp_path / "fake_sbatch.py", FAKE_SBATCH.format(bash=bash))

    ingest(app, project, script, "Scikit-Learn")
    evidence = app.frame_evidence
    evidence.load_csv(str(csv))
    evidence.cb_id.set("subject_id")
    evidence.cb_target.set("label")
    evidence.cb_group.set("patient")
    evidence.confirm_mapping()
    app.update()

    execution = app.frame_execution
    out = tmp_path / "out"
    out.mkdir()
    set_entry(execution.ent_outdir, str(out))
    set_entry(execution.ent_batch, "2")  # 4 batches of about a second each
    execution.run_mode.set("cluster")
    execution.cluster_config.update(
        submit_cmd=f'"{sys.executable}" "{sbatch}"',
        # On a real cluster this is `module load ...` / a conda activate; here, point `python` at this interpreter.
        env_setup=f'python() {{ "{sys.executable}" "$@"; }}\n{node_setup}',
    )
    progress, finished = [], []
    show_progress, log = execution.update_progress, execution.log

    def record_progress(value):
        progress.append((time.monotonic(), value))
        show_progress(value)

    def record_log(message):
        if "Remote Extraction Complete" in message or "Remote extraction failed" in message:
            finished.append(time.monotonic())
        log(message)

    execution.update_progress, execution.log = record_progress, record_log
    execution.start_execution()
    run_mainloop_until(app, lambda: str(execution.btn_run.cget("state")) == "normal", timeout=120)
    return execution, out, progress, finished[0] if finished else None


def test_a_dispatched_job_runs_and_reports_progress_while_it_runs(app, tmp_path):
    execution, out, progress, finished_at = dispatch(app, tmp_path)

    console = execution.txt_console.get("1.0", "end")
    assert execution.lbl_status.cget("text") == "Complete!", console
    assert "[CLUSTER] Job Accepted: Submitted batch job 4242" in console
    assert "[SUCCESS] Extracted 8/8" in console  # the engine's own log, relayed from the node
    midway = [t for t, value in progress if 0 < value < 100]
    assert midway, progress
    assert finished_at - midway[0] > 1.5  # progress arrived while the job was still running, not in a final burst
    assert progress[-1][1] == pytest.approx(100.0)
    cache = AuditCache.open(str(out))
    assert len(cache.metadata) == 8 and cache.config.group_col == "patient"
    assert app.active_cache_dir == str(out)


def test_a_job_that_dies_on_the_node_is_reported_as_failed(app, tmp_path):
    execution, out, _, _ = dispatch(app, tmp_path, node_setup="export PRISM_TEST_BREAK_ON_NODE=1")

    console = execution.txt_console.get("1.0", "end")
    assert execution.lbl_status.cget("text") == "Extraction Failed. Check console."
    assert "model file missing on the compute node" in console  # the node's traceback is shown
    assert "[CRITICAL ERROR] Remote job died with system exit code 1." in console
    assert "Remote extraction failed" in console and "Remote Extraction Complete" not in console
    assert app.active_cache_dir is None
    assert not (out / "prism_metadata.csv").exists()
